"""SharedLearner: owns WM + replay + slab + prototype; drives acting and training (spec §4.5)."""

from __future__ import annotations

import copy
import zlib

import torch

from .actor_critic import ActorCriticSlab
from .config import make_generator
from .prototype import PrototypeActor
from .replay import SharedReplay
from .util import twohot_mean
from .world_model import RSSMWorldModel


class SharedLearner:
    def __init__(self, cfg, global_seed: int):
        self.cfg = cfg
        self.global_seed = int(global_seed if global_seed is not None else 0)
        want = cfg.train_device
        if want.startswith("cuda") and not torch.cuda.is_available():
            print("[rssm] train_device=cuda unavailable — falling back to CPU")
            want = "cpu"
        self.train_device = want
        # nn.Module default init (nn.Linear/GRUCell reset_parameters) draws from the
        # *global* torch RNG with no generator hook available — isolate it: seed
        # deterministically from global_seed, construct, then restore the caller's
        # ambient global RNG state so SharedLearner construction is side-effect-free
        # (spec §6) and reproducible across instances sharing a seed. fork_rng (not
        # get/set_rng_state) because torch.manual_seed also reseeds CUDA/MPS/XPU
        # default generators — plain get/set_rng_state only saves/restores the CPU
        # generator, leaking a global CUDA-RNG side effect on GPU hosts.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed((self.global_seed * 0x9E3779B1 + zlib.crc32(b"wm-init")) % (2**63))
            self.wm = RSSMWorldModel(cfg).to(self.train_device)
        self._act_wm = (
            self.wm if self.train_device == "cpu" else copy.deepcopy(self.wm).cpu().eval()
        )
        self.slab = ActorCriticSlab(
            cfg, make_generator(self.global_seed, "slab-init"), device=self.train_device
        )
        # act() must stay CPU (spec §6/§8): the slab itself lives on train_device,
        # so keep a CPU mirror of just the actor params for the act path when
        # train_device != cpu — avoids feeding a CPU-generator act() call CUDA
        # tensors from self.slab.params (see actor_critic.ActorCriticSlab.act_single).
        self._act_actor = self.slab.actor_snapshot_cpu() if self.train_device != "cpu" else None
        self.prototype = PrototypeActor(cfg)
        self.replay = SharedReplay(cfg)
        self._gen_learner = make_generator(self.global_seed, "learner")
        self._gen_train = make_generator(self.global_seed, "train", device=self.train_device)
        self._agent_gens: dict[int, torch.Generator] = {}

    # --- per-agent RNG ------------------------------------------------------
    def _gen(self, agent_id: int) -> torch.Generator:
        g = self._agent_gens.get(agent_id)
        if g is None:
            g = make_generator(self.global_seed, ("agent", agent_id))
            self._agent_gens[agent_id] = g
        return g

    # --- lifecycle ------------------------------------------------------------
    def on_spawn(self, agent, origin: str, tick: int) -> None:
        if self.cfg.policy_mode == "actor":
            agent.rssm_slot = self.slab.acquire_slot(self.prototype.template())
            if self._act_actor is not None:  # newborn slot row must be current in the mirror
                self.slab.refresh_actor_mirror_slot(self._act_actor, agent.rssm_slot)
        else:  # mpc (arm C): no slab at all
            agent.rssm_slot = None
        agent.brain_arch = "rssm"
        agent.spawn_origin = origin
        agent.hidden_state = None
        self.replay.start_episode(agent.id, origin)

    def on_death(self, agent) -> None:
        self.replay.end_episode(agent.id)
        if agent.rssm_slot is not None:
            self.slab.release_slot(agent.rssm_slot)
        self._agent_gens.pop(agent.id, None)

    # --- acting (batch-1, CPU, no_grad) ----------------------------------------
    def act(self, agent, features) -> dict:
        cfg = self.cfg
        gen = self._gen(agent.id)
        obs = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        if agent.hidden_state is None:
            h, z = self._act_wm.initial_state(1, torch.device("cpu"))
            a_prev = torch.zeros(1, cfg.action_dim)
        else:
            h, z = agent.hidden_state
            a_prev = getattr(agent, "_rssm_last_action", torch.zeros(1, cfg.action_dim))
        if cfg.policy_mode == "mpc":
            action, h, z = self._act_mpc(agent, obs, h, z, a_prev, gen)
        else:
            with torch.no_grad():
                h, z, _, _ = self._act_wm.obs_step(h, z, a_prev, obs, gen)
                genes = obs[:, cfg.gene_slice[0] : cfg.gene_slice[1]]
                s_g = self._act_wm.head_input(h, z, genes)
                action, _ = self.slab.act_single(
                    agent.rssm_slot, s_g, gen, params_override=self._act_actor
                )
        agent._rssm_last_action = action.unsqueeze(0)
        return {
            "action_list": [float(x) for x in action],
            "next_hidden": (h, z),
            "obs_tensor": obs.squeeze(0),
            "action_tensor": action,
        }

    def _act_mpc(self, agent, obs, h, z, a_prev, gen):
        cfg = self.cfg
        with torch.no_grad():
            h, z, _, _ = self._act_wm.obs_step(h, z, a_prev, obs, gen)
            K = cfg.mpc_candidates
            cand = torch.rand(K, cfg.action_dim, generator=gen) * 2 - 1
            hh = h.expand(K, -1).contiguous()
            zz = z.expand(K, -1, -1).contiguous()
            genes = obs[:, cfg.gene_slice[0] : cfg.gene_slice[1]].expand(K, -1)
            score = torch.zeros(K)
            discount = torch.ones(K)
            a = cand
            for _ in range(cfg.mpc_horizon):
                hh, zz, _ = self._act_wm.img_step(hh, zz, a, gen)
                s_g = self._act_wm.head_input(hh, zz, genes)
                score += discount * twohot_mean(self._act_wm.reward_logits(s_g), self._act_wm.bins)
                discount *= cfg.gamma * self._act_wm.cont_prob(s_g)
                a = torch.rand(K, cfg.action_dim, generator=gen) * 2 - 1
            best = cand[int(score.argmax())]
        return best, h, z

    def store_transition(self, agent, brain_step, reward: float, done: bool) -> None:
        self.replay.add(
            agent.id, brain_step["obs_tensor"], brain_step["action_tensor"], reward, done
        )

    # --- training cadence -------------------------------------------------------
    def maybe_train(self, tick: int, agents) -> dict | None:
        cfg = self.cfg
        if tick % cfg.train_every != 0 or len(self.replay) < cfg.prefill_transitions:
            return None
        batch = self.replay.sample_sequences(self._gen_learner)
        batch = {k: v.to(self.train_device) for k, v in batch.items()}
        metrics = self.wm.train_batch(batch, self._gen_train)
        if cfg.policy_mode == "actor" and self.wm.wm_updates > cfg.wm_warmup_updates:
            img = self._imagination_pass(agents, tick)
            if img:
                metrics.update(img)
            self.prototype.update(self.slab)
        if self._act_wm is not self.wm:
            self._act_wm = copy.deepcopy(self.wm).cpu().eval()
        if cfg.policy_mode == "actor" and self._act_actor is not None:
            # actor params may have moved this cycle — mirror is only consulted
            # in "actor" mode (mpc always reads self.slab directly), so skip the
            # refresh there.
            self._act_actor = self.slab.actor_snapshot_cpu()
        return metrics

    def _imagination_pass(self, agents, tick: int) -> dict | None:
        cfg = self.cfg
        rssm_agents = [a for a in agents if a.alive and getattr(a, "rssm_slot", None) is not None]
        if not rssm_agents:
            return None
        h0s, z0s, genes_l, slots = [], [], [], []
        for a in rssm_agents:
            age = tick - getattr(a, "birth_tick", 0)
            own = min(cfg.own_start_frac_max, age / max(1, cfg.young_age_ticks))
            reps = cfg.young_ratio_cap if age < cfg.young_age_ticks else 1
            starts = self.replay.sample_starts(a.id, reps, own, self._gen_learner)
            if starts is None:
                continue
            obs_w, act_w = (t.to(self.train_device) for t in starts)
            with torch.no_grad():  # burn-in RE-ENCODE with the current WM (spec §4.2/§4.5)
                h, z, _, _ = self.wm.observe(obs_w, act_w, self._gen_train)
            h0s.append(h[:, -1])
            z0s.append(z[:, -1])
            genes_l.append(obs_w[:, -1, cfg.gene_slice[0] : cfg.gene_slice[1]])
            # slots indexes both self.slab.params (train_device) and self.slab.step_count
            # (always CPU, see ActorCriticSlab._adam_step) — build it on train_device to
            # match the params it indexes; _adam_step handles the CPU bookkeeping side.
            slots.append(
                torch.full((obs_w.shape[0],), a.rssm_slot, dtype=torch.long, device=obs_w.device)
            )
        if not h0s:
            return None
        return self.slab.imagination_update(
            self.wm,
            torch.cat(h0s),
            torch.cat(z0s),
            torch.cat(genes_l),
            torch.cat(slots),
            self._gen_train,
            None,
        )

    # --- checkpoint ---------------------------------------------------------------
    @staticmethod
    def _to_cpu_tree(obj):
        """Recursively move all tensors in a nested dict/list structure to CPU."""
        if torch.is_tensor(obj):
            return obj.detach().to("cpu")
        if isinstance(obj, dict):
            return {k: SharedLearner._to_cpu_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SharedLearner._to_cpu_tree(v) for v in obj]
        return obj

    def checkpoint_payload(self) -> dict:
        return {
            "wm": {k: v.cpu() for k, v in self.wm.state_dict().items()},
            "wm_opt": self._to_cpu_tree(self.wm.opt.state_dict()),
            "wm_updates": self.wm.wm_updates,
            "prototype": self.prototype.state_dict(),
            "slab": self.slab.state_dict_all(),
        }

    def load_checkpoint_payload(self, sd, agents) -> None:
        self.wm.load_state_dict({k: v.to(self.train_device) for k, v in sd["wm"].items()})
        self.wm.opt.load_state_dict(sd["wm_opt"])
        self.wm.wm_updates = sd["wm_updates"]
        self.prototype.load_state_dict(sd["prototype"])
        self.slab.load_state(sd["slab"])
        if self._act_wm is not self.wm:
            self._act_wm = copy.deepcopy(self.wm).cpu().eval()
        if self._act_actor is not None:  # slab params replaced wholesale — full refresh
            self._act_actor = self.slab.actor_snapshot_cpu()
        for a in agents:  # re-open episodes for living agents
            if getattr(a, "rssm_slot", None) is not None:
                self.replay.start_episode(a.id, getattr(a, "spawn_origin", "unknown"))
