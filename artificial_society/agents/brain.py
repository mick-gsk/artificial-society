from __future__ import annotations

import math
import os

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
#   4..14  cell percepts (food, water, temp, danger, disease, soil, pollution,
#          carrying_capacity, moisture, ash, disturbance)
#   15..16 social (nearby count, friends count)
#   17..20 genes (curiosity, aggression, cooperation, sociality)
#   21..30 equipment + episodic extras (tool, trust, resources x3, last_reward,
#          herb_presence, warmth, mat_count, inv_size)
#   31..33 structure features (camp_level, well_level, farm_level)
#   34..36 causal memory features (3)
#   37..48 episodic memory retrieval (12)
#   49..56 endocrine hormones: cortisol, adrenaline, melatonin, serotonin,
#          dopamine, oxytocin, inflammation, metabolism
#
# IMPORTANT: The brain never receives raw world labels like 'light',
# 'is_night', 'sleep_pressure', or 'disease_level'.  All such information
# reaches the brain ONLY through its hormonal consequences.  The agent
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
WEIGHT_INHERIT_STRENGTH = 0.55
WEIGHT_MUTATION_SCALE = 0.018

# --- Imitationslernen (Spiegelneuronen-Analogie) ---
IMITATION_STRENGTH = 0.10
IMITATION_MUTATION = 0.01

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
# last_reward (26), Causal-Memory (34–36), Episodic-Retrieval (37–48), Hormone (49–56).
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
          2: forage
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
    def inherit_weights_from(
        self,
        parent_brain: Brain,
        strength: float = WEIGHT_INHERIT_STRENGTH,
        mutation_scale: float = WEIGHT_MUTATION_SCALE,
    ):
        with torch.no_grad():
            for (_, child_param), (_, parent_param) in zip(
                self.named_parameters(), parent_brain.named_parameters()
            ):
                if child_param.shape != parent_param.shape:
                    # Spec C5: still überspringen ist beim Architektur-Wechsel
                    # v1→v2 GEWOLLT — nur form-gleiche Teile (encoder, value,
                    # v1-Anteile) werden vererbt, neue Module starten frisch.
                    continue
                mutation = torch.randn_like(child_param) * mutation_scale
                child_param.copy_(
                    strength * parent_param + (1.0 - strength) * child_param + mutation
                )

    # ------------------------------------------------------------------
    # Imitationslernen
    # ------------------------------------------------------------------
    def imitate_from(
        self,
        model_brain: Brain,
        strength: float = IMITATION_STRENGTH,
        mutation_scale: float = IMITATION_MUTATION,
    ):
        with torch.no_grad():
            for (_, my_param), (_, model_param) in zip(
                self.named_parameters(), model_brain.named_parameters()
            ):
                if my_param.shape != model_param.shape:
                    continue
                noise = torch.randn_like(my_param) * mutation_scale
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
