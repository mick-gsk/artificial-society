from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .knowledge import EpisodicMemory

# ---------------------------------------------------------------------------
# Device-Setup (Perf Tier 2, gemessen): fuer diese kleinen Batch-1-Netze ist
# die CPU 7-11x SCHNELLER als die GPU (docs/performance-notes.md), und der
# CUDA-Pfad hat einen offenen FP16-Autocast-Crash (docs/remote-host.md).
# Default ist deshalb CPU; Opt-in per AS_BRAIN_DEVICE=cuda — erst sinnvoll,
# wenn die Brains gebatcht laufen (Tier 4 / GPU-residente Engine).
# ---------------------------------------------------------------------------
device = torch.device(os.environ.get("AS_BRAIN_DEVICE", "cpu"))
USE_FP16 = device.type == "cuda"  # FP16 autocast nur auf GPU aktivieren

# Feature layout (57 total):
#   0..3   body state (energy, health, hydration, age)
#   4..14  cell percepts (food, water, temp, danger, fault, soil, pollution,
#          carrying_capacity, moisture, ash, disturbance)
#   15..16 social (nearby count, friends count)
#   17..20 traits (curiosity, aggression, cooperation, sociality)
#   21..30 equipment + episodic extras (tool, trust, resources x3, last_reward,
#          herb_presence, warmth, mat_count, inv_size)
#   31..33 structure features (camp_level, well_level, farm_level)
#   34..36 causal memory features (3)
#   37..48 episodic memory retrieval (12)
#   49..56 modulation modulators: stress, arousal, rest, satisfaction,
#          reward, affiliation, irritation, upkeep
#
# IMPORTANT: The brain never receives raw world labels like 'light',
# 'is_night', 'sleep_pressure', or 'fault_level'.  All such information
# reaches the brain ONLY through its modulatory consequences.  The agent
# must learn the correlations on its own.
INPUT_SIZE = 57
HIDDEN_SIZE = 96

# Emergenz v3: ACTION_SIZE = 7
# Neue 7. Dimension: research_drive (0..1)
# Das Netz entscheidet selbst wann es forscht -- nicht mehr gewuerfelt.
# Biologisches Vorbild: Neugierde als intrinsisch motiviertes Verhalten
# das durch Belohnungserfahrung verstaerkt oder abgeschwaecht wird.
ACTION_SIZE = 7

GAMMA = 0.97
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
ACTOR_COEF = 1.0
CRITIC_COEF = 0.5
WORLD_COEF = 0.35
ENTROPY_COEF = 0.004
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0
REWARD_CLAMP = 6.0
ROLLOUT_HORIZON = 128
PPO_EPOCHS = 20

# --- Model-Based Planning ---
# Tuned values previously applied at import by emergence_runtime; now the source of
# truth. (PLAN_CANDIDATES stays 12: it is only a def-time default arg, so the old
# runtime override to 8 never actually took effect.)
PLAN_CANDIDATES = 12
PLAN_HORIZON = 2  # Survival mode (default)
PLAN_HORIZON_RESEARCH = 6  # Research / invention mode
NOVELTY_WEIGHT = 0.15
VALUE_WEIGHT = 0.50
REWARD_WEIGHT = 0.35

# --- Neuronale Praedisposition durch Vererbung ---
WEIGHT_DERIVE_STRENGTH = 0.55
WEIGHT_PERTURBATION_SCALE = 0.018

# --- Imitationslernen (Spiegelneuronen-Analogie) ---
IMITATION_STRENGTH = 0.10
IMITATION_PERTURBATION = 0.01

# --- Episodic novelty (NGU-style) ---
EPISODIC_MEMORY_CAPACITY = 500
EPISODIC_K = 15

# --- Emergenz v3: Research-Drive Schwellenwert ---
# Wenn brain_step['research_drive'] > RESEARCH_DRIVE_THRESHOLD,
# initiiert Agent aktiv Forschung statt per Zufallswurf.
RESEARCH_DRIVE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# Physik v2 (Plan 3b, Spec C1–C4): Objekt-Slots, neue Köpfe, PPO-Anpassungen.
# Die v2-Netz-Teile existieren NUR im v2-Modus (Brain(physics_v2=True)) —
# v1-Brains bleiben form-identisch zu heute (Golden + alte Checkpoints).
# ---------------------------------------------------------------------------
OBJ_SLOTS = 10  # K=8 Boden (Chebyshev r≤4) + 2 Hand (Spec C1)
SLOT_FEATS_V2 = 17  # 13 Props + mass/25 + dx/4 + dy/4 + held
SLOT_EMBED_DIM = 32
QUERY_DIM = 8
OBJ_CTX_DIM = 2 * SLOT_EMBED_DIM  # attn-Pool ⊕ masked-Max-Pool = 64

ACTION_SIZE_V2 = 29  # 7 v1-Köpfe + 5 Verben + 1 Effort + 8 Target-Query + 8 Tool-Query
V1_HEAD_DIMS = 7
RESEARCH_DRIVE_DIM = 6  # v2: aus der log-prob-Summe ausgenommen (tote Kopplung, Spec C2)
VERB_SLICE = slice(7, 12)
VERBS_V2 = ("grasp", "release", "strike", "cut", "eat")
EFFORT_DIM = 12  # tanh-Output ∈ [−1,1] → effort01 = (a+1)/2
TARGET_QUERY_SLICE = slice(13, 21)
TOOL_QUERY_SLICE = slice(21, 29)
VERB_THRESHOLD = 0.5  # Aktivierung > 0.5 = „will"; höchstes Verb gewinnt
VERB_INIT_BIAS = 0.2  # optimistischer Init-Bias (Manipulations-Babbling-Prior, Spec C2)

GAMMA_V2 = 0.99  # Kredit-Horizont ~100 Ticks (Knapping→Kadaver→Schneiden→Essen)
PPO_EPOCHS_V2 = 4
N_MINIBATCHES_V2 = 4
MINIBATCH_SIZE_V2 = 32  # 4×32 aus dem 128er-Buffer
KL_EARLY_STOP_V2 = 0.02
ENTROPY_COEF_NEW = 0.01  # neue Kopf-Gruppe inkl. Kategorial (alte Gruppe: ENTROPY_COEF 0.004)
LOGSTD_FLOOR_NEW = -0.8  # log_std-Floor der neuen Köpfe
DEATH_REWARD_V2 = -3.0  # Terminal-Malus (C3), gemünzt in finalize_terminal

# Neugier-Target (C4.1): selbstbezügliche/nichtstationäre Obs-Dims raus —
# last_reward (26), Causal-Memory (34–36), Episodic-Retrieval (37–48), Modulator (49–56).
OBS_TARGET_EXCLUDED = frozenset({26} | set(range(34, 57)))
OBS_TARGET_INCLUDED_IDX = tuple(i for i in range(INPUT_SIZE) if i not in OBS_TARGET_EXCLUDED)
CURIO_TARGET_DIM = len(OBS_TARGET_INCLUDED_IDX) + OBJ_SLOTS * SLOT_FEATS_V2  # 33 + 170 = 203
CURIO_STAT_MOMENTUM = 0.01  # laufendes per-Dim Mean+Std (EMA)


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Stellt sicher, dass t mindestens 2D ist (batch-dim vorne)."""
    return t.unsqueeze(0) if t.dim() == 1 else t


class RolloutBuffer:
    def __init__(self):
        self.storage = []

    def add(self, item):
        self.storage.append(item)

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)


class Brain(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        action_size=ACTION_SIZE,
        plasticity: float = 1.0,
        physics_v2: bool = False,
    ):
        """
        plasticity-Gen (0.3..1.8) skaliert die Lernrate.
        Biologisches Vorbild: Neuronale Plastizitaet variiert zwischen Individuen.

        action_size = 7 (Emergenz v3):
          0: move_x
          1: move_y
          2: gather
          3: cooperate
          4: attack
          5: build
          6: research_drive  <-- NEU: Netz lernt selbst wann es forscht

        physics_v2 (Plan 3b): baut ZUSÄTZLICH (und NACH den v1-Modulen — die
        Init-RNG-Reihenfolge des v1-Pfads bleibt byte-identisch) den
        Attention-Set-Encoder und die neuen Köpfe; action_size wird 29,
        GRU-Input 128+64=192. v1-Brains bleiben form-identisch zu heute.
        """
        super().__init__()
        self.physics_v2 = bool(physics_v2)
        if self.physics_v2:
            action_size = ACTION_SIZE_V2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 160),
            nn.Tanh(),
            nn.Linear(160, 128),
            nn.Tanh(),
        )
        gru_input = 128 + (OBJ_CTX_DIM if self.physics_v2 else 0)
        self.gru = nn.GRUCell(gru_input, hidden_size)
        self.policy_mean = nn.Linear(hidden_size, action_size)
        self.policy_logstd = nn.Parameter(torch.full((action_size,), -0.45, dtype=torch.float32))
        self.value_head = nn.Linear(hidden_size, 1)
        self.world_fc = nn.Sequential(
            nn.Linear(hidden_size + action_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.next_obs_head = nn.Linear(128, input_size)
        self.reward_head = nn.Linear(128, 1)
        if self.physics_v2:
            # Set-Encoder (C1): geteilte Slot-Embeddings + 1-Head-Attention,
            # Query aus dem GRU-Zustand des VORTICKS.
            self.slot_embed = nn.Sequential(
                nn.Linear(SLOT_FEATS_V2, SLOT_EMBED_DIM), nn.LayerNorm(SLOT_EMBED_DIM)
            )
            self.attn_query = nn.Linear(hidden_size, SLOT_EMBED_DIM)
            # Slot-Auswahl (C2): Logits = query · W_sel(embed_i) / sqrt(8)
            self.w_sel = nn.Linear(SLOT_EMBED_DIM, QUERY_DIM, bias=False)
            # World-Model-Head über die rohen Slot-Features des nächsten Ticks (C4.1)
            self.next_slots_head = nn.Linear(128, OBJ_SLOTS * SLOT_FEATS_V2)
            with torch.no_grad():
                # Optimistischer Prior: +0.2 auf die Verb-Means (Spec C2) —
                # struktureller Prior, keine Belohnung.
                self.policy_mean.bias[VERB_SLICE] = VERB_INIT_BIAS
            # Laufendes per-Dim Mean+Std des Neugier-Vorhersagefehlers (C4.1)
            self.register_buffer("curio_err_mean", torch.zeros(CURIO_TARGET_DIM))
            self.register_buffer("curio_err_var", torch.ones(CURIO_TARGET_DIM))
        effective_lr = LEARNING_RATE * max(0.5, min(2.5, plasticity))
        self.optimizer = optim.Adam(self.parameters(), lr=effective_lr)
        self.rollout = RolloutBuffer()

        # --- NGU-style episodic novelty memory ---
        self.episodic_memory = EpisodicMemory(
            capacity=EPISODIC_MEMORY_CAPACITY,
            k=EPISODIC_K,
        )

        # Netz auf GPU verschieben
        self.to(device)

    # ------------------------------------------------------------------
    # Gewichtsvererbung
    # ------------------------------------------------------------------
    def derive_weights_from(
        self,
        parent_brain: Brain,
        strength: float = WEIGHT_DERIVE_STRENGTH,
        perturbation_scale: float = WEIGHT_PERTURBATION_SCALE,
    ):
        with torch.no_grad():
            for (_, spawn_param), (_, parent_param) in zip(
                self.named_parameters(), parent_brain.named_parameters()
            ):
                if spawn_param.shape != parent_param.shape:
                    # Spec C5: still überspringen ist beim Architektur-Wechsel
                    # v1→v2 GEWOLLT — nur form-gleiche Teile (encoder, value,
                    # v1-Anteile) werden vererbt, neue Module starten frisch.
                    continue
                perturbation = torch.randn_like(spawn_param) * perturbation_scale
                spawn_param.copy_(
                    strength * parent_param + (1.0 - strength) * spawn_param + perturbation
                )

    # ------------------------------------------------------------------
    # Imitationslernen
    # ------------------------------------------------------------------
    def imitate_from(
        self,
        model_brain: Brain,
        strength: float = IMITATION_STRENGTH,
        perturbation_scale: float = IMITATION_PERTURBATION,
    ):
        with torch.no_grad():
            for (_, my_param), (_, model_param) in zip(
                self.named_parameters(), model_brain.named_parameters()
            ):
                if my_param.shape != model_param.shape:
                    continue
                noise = torch.randn_like(my_param) * perturbation_scale
                my_param.copy_((1.0 - strength) * my_param + strength * model_param + noise)

    def initial_hidden(self):
        return torch.zeros(self.hidden_size, dtype=torch.float32, device=device)

    def forward(self, obs_tensor, hidden_tensor):
        # FP16 autocast nutzt Blackwell 5th-Gen Tensor Cores (2-4x Throughput)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_FP16):
            z = self.encoder(obs_tensor)
            next_hidden = self.gru(z, hidden_tensor)
            mean = torch.tanh(self.policy_mean(next_hidden))
            log_std = self.policy_logstd.clamp(-2.0, 0.7).unsqueeze(0).expand_as(mean)
            std = torch.exp(log_std)
            value = self.value_head(next_hidden).squeeze(-1)
        return mean, std, value, next_hidden

    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): Attention-Set-Encoder + v2-Forward (Spec C1)
    # ------------------------------------------------------------------
    def _clamped_logstd(self):
        """v1: Clamp (−2.0, 0.7) wie heute. v2: neue Köpfe mit Floor −0.8 (C2)."""
        if not self.physics_v2:
            return self.policy_logstd.clamp(-2.0, 0.7)
        alt = self.policy_logstd[:V1_HEAD_DIMS].clamp(-2.0, 0.7)
        neu = self.policy_logstd[V1_HEAD_DIMS:].clamp(LOGSTD_FLOOR_NEW, 0.7)
        return torch.cat([alt, neu])

    def _encode_slots(self, slot_feats, slot_mask, hidden_tensor):
        """embed_i = LayerNorm(Linear(17→32)); q = Linear(96→32)(h_prev);
        attn = masked_softmax(q·K^T/√32); obj_ctx = attn-Pool ⊕ masked-Max-Pool (64).
        Alle Slots maskiert ⇒ obj_ctx = 0 und attn = 0 — kein NaN (Spec C1/D3)."""
        embeds = self.slot_embed(slot_feats)  # (B, 10, 32)
        any_slot = slot_mask.any(dim=-1, keepdim=True)  # (B, 1)
        q = self.attn_query(hidden_tensor)  # (B, 32)
        scores = torch.einsum("bd,bsd->bs", q, embeds) / math.sqrt(SLOT_EMBED_DIM)
        scores = scores.masked_fill(~slot_mask, -1e9)
        attn = torch.softmax(scores, dim=-1) * slot_mask.float()  # all-masked ⇒ exakt 0
        pooled = torch.einsum("bs,bsd->bd", attn, embeds)  # (B, 32)
        maxed = embeds.masked_fill(~slot_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        maxed = torch.where(any_slot, maxed, torch.zeros_like(maxed))
        pooled = torch.where(any_slot, pooled, torch.zeros_like(pooled))
        return torch.cat([pooled, maxed], dim=-1), embeds, attn

    def forward_v2(self, obs_tensor, hidden_tensor, slot_feats, slot_mask):
        """v2-Forward: GRU-Input = concat(encoder(obs_57), obj_ctx) (128+64=192).
        Die 57 v1-Obs-Dims bleiben unverändert (Spec C1)."""
        z = self.encoder(obs_tensor)
        obj_ctx, embeds, attn = self._encode_slots(slot_feats, slot_mask, hidden_tensor)
        next_hidden = self.gru(torch.cat([z, obj_ctx], dim=-1), hidden_tensor)
        mean = torch.tanh(self.policy_mean(next_hidden))
        log_std = self._clamped_logstd().unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        value = self.value_head(next_hidden).squeeze(-1)
        return mean, std, value, next_hidden, embeds, attn

    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): Aktions-Sampling mit kategorialer Slot-Auswahl (C2)
    # ------------------------------------------------------------------
    def _continuous_log_prob(self, mean, std, action_tensor):
        """Gemeinsamer log-prob-Pfad für Sampling UND Retraining (bitgleiche
        Reproduktion, D3): identischer atanh-Transform wie evaluate_actions.
        v2: research_drive (Dim 6) ist aus der Summe ausgenommen (Spec C2)."""
        dist = torch.distributions.Normal(mean, std)
        clipped = torch.clamp(action_tensor, -0.999, 0.999)
        raw = 0.5 * torch.log((1 + clipped) / (1 - clipped))
        per_dim = dist.log_prob(raw) - torch.log(1 - clipped.pow(2) + 1e-6)
        if self.physics_v2:
            keep = torch.ones(per_dim.shape[-1], dtype=torch.bool, device=per_dim.device)
            keep[RESEARCH_DRIVE_DIM] = False
            per_dim = per_dim[..., keep]
        return per_dim.sum(dim=-1), dist

    def _slot_logits(self, query, embeds, admissible):
        """Auswahl-Logits = query · W_sel(embed_i) / √8 über zulässige Slots (C2)."""
        keys = self.w_sel(embeds)  # (B, 10, 8)
        logits = torch.einsum("bq,bsq->bs", query, keys) / math.sqrt(QUERY_DIM)
        return logits.masked_fill(~admissible, -1e9)

    @staticmethod
    def _categorical_terms(logits, idx):
        """(log-prob des gezogenen Slots, Entropie der Kategorial-Verteilung).
        Unzulässige Slots tragen p=0 exakt (softmax von −1e9 unterläuft zu 0)."""
        logp = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * logp).sum(dim=-1)
        chosen = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return chosen, entropy

    @staticmethod
    def _verb_feasible(verb, grasp_t, held_t, target_t) -> bool:
        """Machbarkeits-Vorprüfung je Verb: entweder werden BEIDE Kategorial-
        Terme gezogen oder keiner (leere zulässige Menge ⇒ No-op ohne
        log-prob-Anteil, Spec C2)."""
        n_held = int(held_t.sum())
        n_ground_atpos = int((target_t & ~held_t).sum())
        if verb == "grasp":
            return bool(grasp_t.any())
        if verb == "release":
            return n_held >= 1
        if verb == "eat":
            return bool(target_t.any())
        if verb == "strike":
            # Schläger MUSS aus der Hand kommen; Ziel = das andere Gehaltene
            # oder Boden an eigener Position.
            return n_held >= 1 and (n_ground_atpos >= 1 or n_held >= 2)
        if verb == "cut":
            if n_held >= 1:
                return n_ground_atpos >= 1 or n_held >= 2
            return n_ground_atpos >= 1  # bloße Hand
        return False

    def act_v2(self, features, hidden_state, slot_feats, slot_mask, masks):
        """v2-Aktionswahl (C2): Sampling, nie Argmax; Planner deaktiviert (C5).
        masks: dict aus perception_v2.admissible_masks ('grasp'/'held'/'target',
        np-bool (10,)). Rückgabe-Transitionen tragen Maske + Slot-Index, damit
        evaluate_actions_v2 exakt dieselbe Konditionierung reproduziert."""
        with torch.no_grad():
            obs = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            hidden = hidden_state.unsqueeze(0)
            feats = torch.as_tensor(slot_feats, dtype=torch.float32, device=device).unsqueeze(0)
            smask = torch.as_tensor(
                np.asarray(slot_mask), dtype=torch.bool, device=device
            ).unsqueeze(0)
            mean, std, value, next_hidden, embeds, attn = self.forward_v2(obs, hidden, feats, smask)

            dist = torch.distributions.Normal(mean, std)
            action = torch.tanh(dist.rsample())
            log_prob, _ = self._continuous_log_prob(mean, std, action)
            entropy = dist.entropy().sum(dim=-1)
            a = action.squeeze(0)

            grasp_t = torch.as_tensor(np.asarray(masks["grasp"]), dtype=torch.bool, device=device)
            held_t = torch.as_tensor(np.asarray(masks["held"]), dtype=torch.bool, device=device)
            target_t = torch.as_tensor(np.asarray(masks["target"]), dtype=torch.bool, device=device)

            verbs = a[VERB_SLICE]
            verb = None
            if float(verbs.max()) > VERB_THRESHOLD:
                kandidat = VERBS_V2[int(torch.argmax(verbs))]
                if self._verb_feasible(kandidat, grasp_t, held_t, target_t):
                    verb = kandidat
            effort = float((a[EFFORT_DIM] + 1.0) * 0.5)

            tool_idx = -1
            target_idx = -1
            tool_mask_used = torch.zeros_like(smask)
            target_mask_used = torch.zeros_like(smask)
            if verb in ("strike", "cut") and bool(held_t.any()):
                tool_mask_used = held_t.unsqueeze(0)
                logits = self._slot_logits(a[TOOL_QUERY_SLICE].unsqueeze(0), embeds, tool_mask_used)
                tool_idx = int(torch.distributions.Categorical(logits=logits).sample())
                lp, _ent = self._categorical_terms(logits, torch.tensor([tool_idx], device=device))
                log_prob = log_prob + lp
            if verb is not None:
                base = {
                    "grasp": grasp_t,
                    "release": held_t,
                    "strike": target_t,
                    "cut": target_t,
                    "eat": target_t,
                }[verb].clone()
                if tool_idx >= 0:
                    base[tool_idx] = False  # Ziel ≠ Werkzeug
                target_mask_used = base.unsqueeze(0)
                logits = self._slot_logits(
                    a[TARGET_QUERY_SLICE].unsqueeze(0), embeds, target_mask_used
                )
                target_idx = int(torch.distributions.Categorical(logits=logits).sample())
                lp, _ent = self._categorical_terms(
                    logits, torch.tensor([target_idx], device=device)
                )
                log_prob = log_prob + lp

            action_list = a.detach().tolist()
            return {
                "obs_tensor": obs,
                "hidden_in": hidden.detach(),
                "value": value.detach(),
                "next_hidden": next_hidden.squeeze(0).detach(),
                "action_tensor": action.detach(),
                "action_list": action_list,
                "log_prob": log_prob.detach(),
                "entropy": entropy.detach(),
                "slot_feats": feats.detach(),
                "slot_mask": smask.detach(),
                "verb": verb,
                "effort": effort,
                "target_idx": target_idx,
                "tool_idx": tool_idx,
                "target_mask": target_mask_used.detach(),
                "tool_mask": tool_mask_used.detach(),
                "slot_embeds": embeds.squeeze(0).detach(),
                "attn": attn.squeeze(0).detach(),
                "research_drive": float(action_list[RESEARCH_DRIVE_DIM]),
            }

    def store_transition_v2(
        self, brain_step, reward, done, next_obs, next_slot_feats, next_slot_mask
    ):
        """v2-Transition (C2): speichert zusätzlich Slot-Features, Slot-Maske,
        zulässige Masken und gezogene Indizes — evaluate_actions_v2 reproduziert
        damit exakt dieselbe log-prob-Konditionierung (variable Struktur)."""
        self.rollout.add(
            {
                "obs": brain_step["obs_tensor"].detach().squeeze(0),
                "hidden": brain_step["hidden_in"].detach().squeeze(0),
                "action": brain_step["action_tensor"].detach().squeeze(0),
                "log_prob": brain_step["log_prob"].detach().squeeze(0),
                "value": brain_step["value"].detach().squeeze(0),
                "reward": max(-REWARD_CLAMP, min(REWARD_CLAMP, reward)),
                "done": done,
                "next_obs": torch.tensor(next_obs, dtype=torch.float32, device=device),
                "slot_feats": brain_step["slot_feats"].detach().squeeze(0),
                "slot_mask": brain_step["slot_mask"].detach().squeeze(0),
                "target_mask": brain_step["target_mask"].detach().squeeze(0),
                "tool_mask": brain_step["tool_mask"].detach().squeeze(0),
                "target_idx": int(brain_step["target_idx"]),
                "tool_idx": int(brain_step["tool_idx"]),
                "next_slot_feats": torch.as_tensor(
                    np.asarray(next_slot_feats), dtype=torch.float32, device=device
                ),
                "next_slot_mask": torch.as_tensor(
                    np.asarray(next_slot_mask), dtype=torch.bool, device=device
                ),
            }
        )

    def evaluate_actions_v2(
        self,
        obs,
        hid,
        slot_feats,
        slot_mask,
        actions,
        target_mask,
        target_idx,
        tool_mask,
        tool_idx,
    ):
        """PPO-Retraining-Forward (C2): reproduziert die Sampling-log-prob
        bitgleich über denselben Code-Pfad (_continuous_log_prob, _slot_logits,
        _categorical_terms). idx = −1 ⇒ die Zeile hat keinen Slot-Term.
        Rückgabe: (log_prob, ent_alt (Dims 0..6), ent_neu (Dims 7..28 +
        Kategorial), value, next_hidden)."""
        mean, std, value, next_hidden, embeds, _ = self.forward_v2(obs, hid, slot_feats, slot_mask)
        log_prob, dist = self._continuous_log_prob(mean, std, actions)
        ent_per_dim = dist.entropy()
        ent_old = ent_per_dim[:, :V1_HEAD_DIMS].sum(dim=-1)
        ent_new = ent_per_dim[:, V1_HEAD_DIMS:].sum(dim=-1)

        for idx, mask_b, query_slice in (
            (tool_idx, tool_mask, TOOL_QUERY_SLICE),
            (target_idx, target_mask, TARGET_QUERY_SLICE),
        ):
            rows = idx >= 0
            if not bool(rows.any()):
                continue
            logits = self._slot_logits(actions[rows][:, query_slice], embeds[rows], mask_b[rows])
            lp, ent = self._categorical_terms(logits, idx[rows])
            lp_full = torch.zeros_like(log_prob)
            lp_full[rows] = lp
            ent_full = torch.zeros_like(ent_new)
            ent_full[rows] = ent
            log_prob = log_prob + lp_full
            ent_new = ent_new + ent_full  # Kategorial-Entropie → 0.01-Gruppe (C2)
        return log_prob, ent_old, ent_new, value, next_hidden

    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): World-Model + nextslot_err (Neugier-Quelle 1, C4.1)
    # ------------------------------------------------------------------
    def predict_world_v2(self, hidden_tensor, action_tensor):
        """v2-World-Model: prädiziert next-obs (57) UND die rohen Slot-Features
        (10×17 = 170) des nächsten Ticks — NICHT obj_ctx (das mit den eigenen
        Gewichten wandert und nie konvergiert, Spec C4.1)."""
        hidden_tensor = _ensure_2d(hidden_tensor)
        action_tensor = _ensure_2d(action_tensor)
        z = self.world_fc(torch.cat([hidden_tensor, action_tensor], dim=-1))
        next_obs = torch.tanh(self.next_obs_head(z))
        next_slots = torch.tanh(self.next_slots_head(z))  # Features ∈ [−1,1] (dx/4, dy/4)
        reward = self.reward_head(z).squeeze(-1)
        return next_obs, next_slots, reward

    def _curio_target(self, next_obs_t, next_slot_feats_t, next_slot_mask_t):
        """(target (203,), valid (203,)): 33 eingeschlossene Obs-Dims ⊕ 170
        Slot-Dims; Slot-Dims sind nur gültig, wo der Slot belegt ist."""
        incl = torch.tensor(OBS_TARGET_INCLUDED_IDX, dtype=torch.long, device=device)
        target = torch.cat([next_obs_t[incl], next_slot_feats_t.reshape(-1)])
        valid = torch.cat(
            [
                torch.ones(len(OBS_TARGET_INCLUDED_IDX), dtype=torch.bool, device=device),
                next_slot_mask_t.unsqueeze(-1).expand(OBJ_SLOTS, SLOT_FEATS_V2).reshape(-1),
            ]
        )
        return target, valid

    def nextslot_error(self, brain_step, next_obs, next_slot_feats, next_slot_mask):
        """Neugier-Quelle 1 (C4.1): Vorhersagefehler auf den rohen, maskierten
        Slot-Features + den bereinigten Obs-Dims des nächsten Ticks.
        Normalisierung PER DIM mit laufendem Mean+Std (EMA, Momentum 0.01) —
        nicht global, nicht nur Std. Der bisherige rew_err-Term (v1,
        pred_reward.abs() — gar kein Fehlerterm) ist ersatzlos gestrichen."""
        with torch.no_grad():
            pred_obs, pred_slots, _ = self.predict_world_v2(
                brain_step["next_hidden"], brain_step["action_tensor"]
            )
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            feats_t = torch.as_tensor(
                np.asarray(next_slot_feats), dtype=torch.float32, device=device
            )
            mask_t = torch.as_tensor(np.asarray(next_slot_mask), dtype=torch.bool, device=device)
            target, valid = self._curio_target(next_obs_t, feats_t, mask_t)
            incl = torch.tensor(OBS_TARGET_INCLUDED_IDX, dtype=torch.long, device=device)
            pred = torch.cat([pred_obs.squeeze(0)[incl], pred_slots.squeeze(0)])
            err = pred - target
            m = CURIO_STAT_MOMENTUM
            new_mean = torch.where(
                valid, (1.0 - m) * self.curio_err_mean + m * err, self.curio_err_mean
            )
            new_var = torch.where(
                valid,
                (1.0 - m) * self.curio_err_var + m * (err - new_mean).pow(2),
                self.curio_err_var,
            )
            self.curio_err_mean.copy_(new_mean)
            self.curio_err_var.copy_(new_var)
            z = (err - new_mean) / (new_var.sqrt() + 1e-6)
            return float(z[valid].pow(2).mean())

    def _world_loss_v2(
        self, next_hidden, actions, next_obs, next_slot_feats, next_slot_mask, rewards
    ):
        """Trainings-Loss des v2-World-Models: MSE auf den 33 eingeschlossenen
        Obs-Dims + maskierte MSE auf den Slot-Dims + Reward-Head-MSE."""
        pred_obs, pred_slots, pred_rew = self.predict_world_v2(next_hidden, actions)
        incl = torch.tensor(OBS_TARGET_INCLUDED_IDX, dtype=torch.long, device=device)
        obs_loss = F.mse_loss(pred_obs[:, incl], next_obs[:, incl])
        b = next_slot_feats.shape[0]
        flach = next_slot_feats.reshape(b, -1)
        maske = (
            next_slot_mask.unsqueeze(-1).expand(b, OBJ_SLOTS, SLOT_FEATS_V2).reshape(b, -1).float()
        )
        slot_loss = ((pred_slots - flach).pow(2) * maske).sum() / maske.sum().clamp(min=1.0)
        rew_loss = F.mse_loss(pred_rew.view(-1), rewards)
        return obs_loss + slot_loss + rew_loss

    def predict_world(self, hidden_tensor, action_tensor):
        # Beide Tensoren auf 2D normalisieren, damit torch.cat immer funktioniert
        hidden_tensor = _ensure_2d(hidden_tensor)
        action_tensor = _ensure_2d(action_tensor)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_FP16):
            z = self.world_fc(torch.cat([hidden_tensor, action_tensor], dim=-1))
            next_obs = torch.tanh(self.next_obs_head(z))
            reward = self.reward_head(z).squeeze(-1)
        return next_obs, reward

    def evaluate_actions(self, obs_tensor, hidden_tensor, action_tensor):
        mean, std, value, next_hidden = self.forward(obs_tensor, hidden_tensor)
        dist = torch.distributions.Normal(mean, std)
        clipped = torch.clamp(action_tensor, -0.999, 0.999)
        raw_action = 0.5 * torch.log((1 + clipped) / (1 - clipped))
        log_prob = (dist.log_prob(raw_action) - torch.log(1 - clipped.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value, next_hidden

    def imagine_rollout(
        self,
        hidden_tensor,
        action_tensor,
        horizon: int = PLAN_HORIZON,
        goal_vector: torch.Tensor | None = None,
    ):
        """
        Vektorisierter Multi-step imagined rollout.
        hidden_tensor: (n_candidates, hidden_size)
        action_tensor: (n_candidates, action_size)

        goal_vector : optional 1-D tensor of shape (INPUT_SIZE,).
            When provided, each step adds a goal-proximity bonus:
                goal_bonus = -||pred_next_obs - goal_vector|| * GOAL_WEIGHT
        """
        GOAL_WEIGHT = 0.40

        h = _ensure_2d(hidden_tensor)
        a = _ensure_2d(action_tensor)
        total_score = torch.zeros(a.shape[0], device=device)
        discount = 1.0

        for _ in range(horizon):
            pred_next_obs, pred_reward = self.predict_world(h, a)

            # k-NN episodic novelty (vektorisiert ueber alle Kandidaten)
            if len(self.episodic_memory.buffer) >= self.episodic_memory.k:
                # Cached stack: the buffer is stable across this rollout, so the
                # (N, D) tensor is built/uploaded once per buffer change instead of
                # once per horizon step. Identical to torch.stack(list(buffer)).to(device).
                stack = self.episodic_memory.stacked(device)  # (N, D)
                obs_expanded = pred_next_obs.unsqueeze(1)  # (C, 1, D)
                stack_expanded = stack.unsqueeze(0)  # (1, N, D)
                dists = torch.norm(obs_expanded - stack_expanded, dim=-1)  # (C, N)
                k = min(self.episodic_memory.k, dists.shape[1])
                knn = dists.topk(k, largest=False, dim=1).values.mean(dim=1)  # (C,)
                novelty = knn / (knn + self.episodic_memory.epsilon)
            else:
                novelty = torch.ones(a.shape[0], device=device)

            enc = self.encoder(pred_next_obs)
            h = self.gru(enc, h)
            next_value = self.value_head(h).squeeze(-1)

            step_score = (
                REWARD_WEIGHT * pred_reward + VALUE_WEIGHT * next_value + NOVELTY_WEIGHT * novelty
            )

            if goal_vector is not None:
                goal_dist = torch.norm(
                    pred_next_obs - goal_vector.unsqueeze(0).expand_as(pred_next_obs),
                    dim=-1,
                )
                step_score = step_score - GOAL_WEIGHT * goal_dist

            total_score = total_score + discount * step_score
            discount *= GAMMA

            mean_next = torch.tanh(self.policy_mean(h))
            log_std = self.policy_logstd.clamp(-2.0, 0.7).unsqueeze(0).expand_as(mean_next)
            std_next = torch.exp(log_std)
            a = torch.tanh(torch.distributions.Normal(mean_next, std_next).rsample())

        return h, total_score

    def plan_action(
        self,
        obs_tensor,
        hidden_tensor,
        n_candidates: int = PLAN_CANDIDATES,
        goal_vector: torch.Tensor | None = None,
        research_mode: bool = False,
        policy: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Vektorisiertes Planning: alle Kandidaten parallel auf GPU statt
        sequentielle Python-Schleife. Speedup ~8-12x bei PLAN_CANDIDATES=12.

        research_mode : bool
            When True, planner uses PLAN_HORIZON_RESEARCH (30) instead of
            PLAN_HORIZON (3). Wird automatisch aktiviert wenn research_drive > Threshold.
        policy : optional (mean, std)
            Precomputed policy head from the caller's forward(obs, hidden). When
            given, the redundant internal forward is skipped. Numerically
            identical: same obs/hidden -> same deterministic forward. forward()
            draws no RNG, so the candidate sampling sequence is unchanged.
        """
        horizon = PLAN_HORIZON_RESEARCH if research_mode else PLAN_HORIZON

        with torch.no_grad():
            if policy is None:
                mean, std, _, _ = self.forward(obs_tensor, hidden_tensor)
            else:
                mean, std = policy
            dist = torch.distributions.Normal(mean, std)
            # (n_candidates, 1, action_size) -> squeeze -> (n_candidates, action_size)
            raw_samples = dist.rsample((n_candidates,)).squeeze(1)
            action_samples = torch.tanh(raw_samples)  # (n_candidates, action_size)

            # hidden_tensor: (1, hidden_size) -> expand zu (n_candidates, hidden_size)
            h_expanded = _ensure_2d(hidden_tensor).expand(n_candidates, -1)

            # Alle n_candidates Kandidaten in einem einzigen Batch-Call
            _, scores = self.imagine_rollout(
                h_expanded,
                action_samples,
                horizon=horizon,
                goal_vector=goal_vector,
            )

            # Index with the 0-d argmax tensor directly: identical row, no host
            # sync (.item() forced a GPU->CPU transfer every plan step).
            best_action = action_samples[scores.argmax()]
            clipped = torch.clamp(best_action, -0.999, 0.999)
            raw_best = 0.5 * torch.log((1 + clipped) / (1 - clipped + 1e-8))
            log_prob = (dist.log_prob(raw_best) - torch.log(1 - clipped.pow(2) + 1e-6)).sum(dim=-1)
            return best_action, log_prob, scores

    def act(
        self,
        features,
        hidden_state,
        use_planning: bool = True,
        goal_vector: torch.Tensor | None = None,
        research_mode: bool = False,
    ):
        # Action selection builds no autograd graph: every output is detached and
        # stored; gradients are recomputed later in maybe_train via evaluate_actions.
        # no_grad (not inference_mode) so the stored obs/hidden/action tensors can be
        # re-fed into the grad-enabled PPO forward without "inference tensor" errors.
        with torch.no_grad():
            obs = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            hidden = hidden_state.unsqueeze(0)
            mean, std, value, next_hidden = self.forward(obs, hidden)

            if use_planning:
                # Reuse this forward's policy head; next_hidden here is identical to
                # the old post-plan re-forward (same obs/hidden, deterministic forward).
                action, log_prob, candidate_scores = self.plan_action(
                    obs,
                    hidden,
                    goal_vector=goal_vector,
                    research_mode=research_mode,
                    policy=(mean, std),
                )
            else:
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.rsample()
                action = torch.tanh(raw_action)
                clipped = torch.clamp(action, -0.999, 0.999)
                raw_a = 0.5 * torch.log((1 + clipped) / (1 - clipped + 1e-8))
                log_prob = (dist.log_prob(raw_a) - torch.log(1 - clipped.pow(2) + 1e-6)).sum(dim=-1)

            entropy = torch.distributions.Normal(mean, std).entropy().sum(dim=-1)

            # Emergenz v3: research_drive aus Dimension 6 extrahieren
            action_list = action.squeeze(0).detach().tolist()
            research_drive = float(action_list[6]) if len(action_list) > 6 else 0.0

            return {
                "obs_tensor": obs.detach(),
                "hidden_in": hidden.detach(),
                "value": value.detach(),
                "next_hidden": next_hidden.squeeze(0).detach(),
                "action_tensor": action.detach(),
                "action_list": action_list,
                "log_prob": log_prob.detach(),
                "entropy": entropy.detach(),
                "research_drive": research_drive,  # NEU: direkt zugreifbar fuer agent.update()
            }

    def intrinsic_reward(self, hidden_in, action_tensor, next_obs):
        """
        Combines prediction-error curiosity (world model) with
        NGU-style episodic novelty (state-space distance).

        hidden_in und action_tensor koennen 1D oder 2D sein --
        _ensure_2d in predict_world normalisiert beide Faelle.
        """
        with torch.no_grad():
            pred_next_obs, pred_reward = self.predict_world(
                _ensure_2d(hidden_in.to(device)),
                _ensure_2d(action_tensor.to(device)),
            )
            target = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            obs_err = F.mse_loss(pred_next_obs, target)
            rew_err = pred_reward.abs().mean()
            prediction_curiosity = (obs_err + 0.2 * rew_err).clamp(0.0, 2.0)

            episodic_novelty = self.episodic_memory.novelty(target.squeeze(0))

            combined = float(prediction_curiosity) * (0.5 + 0.5 * episodic_novelty)
            return float(combined)

    def store_transition(
        self, obs_tensor, hidden_in, action_tensor, log_prob, value, reward, done, next_obs
    ):
        self.rollout.add(
            {
                "obs": obs_tensor.detach().squeeze(0),
                "hidden": hidden_in.detach().squeeze(0),
                "action": action_tensor.detach().squeeze(0),
                "log_prob": log_prob.detach().squeeze(0),
                "value": value.detach().squeeze(0),
                "reward": max(-REWARD_CLAMP, min(REWARD_CLAMP, reward)),
                "done": done,
                "next_obs": torch.tensor(next_obs, dtype=torch.float32, device=device),
            }
        )

    def maybe_train(self):
        if len(self.rollout) < ROLLOUT_HORIZON:
            return None
        if self.physics_v2:
            # v2 (C2): eigener Trainings-Pfad (γ 0.99, Minibatches, KL-Stop).
            loss = self._train_v2(self.rollout.storage)
            self.rollout.clear()
            return loss
        batch = self.rollout.storage
        obs = torch.stack([item["obs"] for item in batch]).to(device)
        hid = torch.stack([item["hidden"] for item in batch]).to(device)
        actions = torch.stack([item["action"] for item in batch]).to(device)
        old_log_probs = torch.stack([item["log_prob"] for item in batch]).view(-1).to(device)
        values = torch.stack([item["value"] for item in batch]).view(-1).to(device)
        rewards = torch.tensor(
            [item["reward"] for item in batch], dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [1.0 if item["done"] else 0.0 for item in batch], dtype=torch.float32, device=device
        )
        next_obs = torch.stack([item["next_obs"] for item in batch]).to(device)

        with torch.no_grad():
            _, _, next_values, _ = self.forward(next_obs, hid)
            next_values = next_values.view(-1)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(batch))):
            delta = rewards[t] + GAMMA * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_loss = None
        for _ in range(PPO_EPOCHS):
            new_log_probs, entropy, new_values, next_hidden = self.evaluate_actions(
                obs, hid, actions
            )
            ratios = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratios * advantages
            clipped_r = torch.clamp(ratios, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            actor_loss = -torch.min(unclipped, clipped_r).mean()
            critic_loss = F.mse_loss(new_values.view(-1), returns)
            pred_next_obs, pred_reward = self.predict_world(next_hidden, actions)
            world_loss = F.mse_loss(pred_next_obs, next_obs) + F.mse_loss(
                pred_reward.view(-1), rewards
            )
            entropy_bonus = entropy.mean()
            loss = (
                ACTOR_COEF * actor_loss
                + CRITIC_COEF * critic_loss
                + WORLD_COEF * world_loss
                - ENTROPY_COEF * entropy_bonus
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu())

        self.rollout.clear()
        return last_loss

    # ------------------------------------------------------------------
    # Physik v2 (Plan 3b): PPO-Anpassungen (Spec C2)
    # ------------------------------------------------------------------
    def _train_v2(self, batch):
        """PPO über den v2-Buffer: GAMMA 0.99, 4 Epochen, 4×32-Minibatches,
        KL-Early-Stop 0.02, Entropy pro Kopf-Gruppe (alt 0.004 / neu 0.01 inkl.
        Kategorial). 20 Epochen auf einem Batch würden σ kollabieren und die
        Verben töten, bevor Kadaver und Klinge je koinzidieren (RL-Review)."""
        n = len(batch)
        if n < 2:
            return None
        obs = torch.stack([t["obs"] for t in batch]).to(device)
        hid = torch.stack([t["hidden"] for t in batch]).to(device)
        sf = torch.stack([t["slot_feats"] for t in batch]).to(device)
        sm = torch.stack([t["slot_mask"] for t in batch]).to(device)
        actions = torch.stack([t["action"] for t in batch]).to(device)
        tm = torch.stack([t["target_mask"] for t in batch]).to(device)
        om = torch.stack([t["tool_mask"] for t in batch]).to(device)
        ti = torch.tensor([t["target_idx"] for t in batch], dtype=torch.long, device=device)
        oi = torch.tensor([t["tool_idx"] for t in batch], dtype=torch.long, device=device)
        old_log_probs = torch.stack([t["log_prob"] for t in batch]).view(-1).to(device)
        values = torch.stack([t["value"] for t in batch]).view(-1).to(device)
        rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32, device=device)
        dones = torch.tensor(
            [1.0 if t["done"] else 0.0 for t in batch], dtype=torch.float32, device=device
        )
        next_obs = torch.stack([t["next_obs"] for t in batch]).to(device)
        nsf = torch.stack([t["next_slot_feats"] for t in batch]).to(device)
        nsm = torch.stack([t["next_slot_mask"] for t in batch]).to(device)

        with torch.no_grad():
            _, _, next_values, _, _, _ = self.forward_v2(next_obs, hid, nsf, nsm)
            next_values = next_values.view(-1)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + GAMMA_V2 * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + GAMMA_V2 * GAE_LAMBDA * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # F2 (Review): max(2, …) — mit max(1, …) ergäbe n∈{2..7} mb_size 1 und
        # der 1er-Skip unten würde JEDEN Minibatch überspringen: finalize_terminal
        # machte bei 2–7 Transitionen null Gradientenschritte, die −3.0-Terminal-
        # Transition verfiele (Spec C2 verlangt „geflusht UND trainiert").
        mb_size = MINIBATCH_SIZE_V2 if n >= ROLLOUT_HORIZON else max(2, n // N_MINIBATCHES_V2)
        last_loss = None
        for _ in range(PPO_EPOCHS_V2):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                if len(idx) < 2:
                    continue  # 1er-Minibatch: MSE/Std entartet
                new_log_probs, ent_old, ent_new, new_values, next_hidden = self.evaluate_actions_v2(
                    obs[idx],
                    hid[idx],
                    sf[idx],
                    sm[idx],
                    actions[idx],
                    tm[idx],
                    ti[idx],
                    om[idx],
                    oi[idx],
                )
                approx_kl = (old_log_probs[idx] - new_log_probs).mean()
                if float(approx_kl.detach()) > KL_EARLY_STOP_V2:
                    return last_loss  # KL-Early-Stop: GESAMTES Training abbrechen
                ratios = torch.exp(new_log_probs - old_log_probs[idx])
                unclipped = ratios * advantages[idx]
                clipped_r = torch.clamp(ratios, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages[idx]
                actor_loss = -torch.min(unclipped, clipped_r).mean()
                critic_loss = F.mse_loss(new_values.view(-1), returns[idx])
                world_loss = self._world_loss_v2(
                    next_hidden, actions[idx], next_obs[idx], nsf[idx], nsm[idx], rewards[idx]
                )
                loss = (
                    ACTOR_COEF * actor_loss
                    + CRITIC_COEF * critic_loss
                    + WORLD_COEF * world_loss
                    - ENTROPY_COEF * ent_old.mean()
                    - ENTROPY_COEF_NEW * ent_new.mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())
        return last_loss

    def finalize_terminal(self, death_reward=DEATH_REWARD_V2):
        """Echte Terminal-Transition beim Tod (C2/C3): done=True erreicht den
        Buffer, r_death (−3.0) wird GENAU EINMAL auf die letzte Transition
        gemünzt (Aufrufort: der ursachen-agnostische Todes-Aggregationspunkt
        Simulation.remove_dead), der Restbuffer wird geflusht und trainiert.
        Heute stirbt ein Agent, bevor store_transition den Todes-Tick je sieht
        — der Buffer verfiel untrainiert."""
        if not self.rollout.storage:
            return None
        last = self.rollout.storage[-1]
        last["reward"] = max(-REWARD_CLAMP, min(REWARD_CLAMP, last["reward"] + death_reward))
        last["done"] = True
        loss = self._train_v2(self.rollout.storage) if len(self.rollout) >= 2 else None
        self.rollout.clear()
        return loss
