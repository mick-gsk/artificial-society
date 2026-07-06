"""Per-agent actor-critics in a pre-allocated slab (spec §4.2/§8).

Stacked tensors (max_slots, out, in); batched linears via einsum; hand-rolled
elementwise Adam on the slab with per-slot step counts and alive-masked updates.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .util import make_bins, symlog, twohot, twohot_mean

LOGSTD_MIN, LOGSTD_MAX = -5.0, 1.0


def lambda_returns(reward, cont, value, gamma, lam):
    """reward/cont: (N,H); value: (N,H+1). Returns (N,H) λ-returns with γ·cont discounts."""
    H = reward.shape[1]
    R = torch.zeros_like(reward)
    nxt = value[:, -1]
    for t in range(H - 1, -1, -1):
        disc = gamma * cont[:, t]
        blended = (1 - lam) * value[:, t + 1] + lam * nxt
        R[:, t] = reward[:, t] + disc * blended
        nxt = R[:, t]
    return R


def _blin(w, b, x):
    """Batched linear: w (N,out,in), b (N,out), x (N,in) -> (N,out)."""
    return torch.einsum("noi,ni->no", w, x) + b


class ActorCriticSlab:
    ACTOR_KEYS = ("a_w1", "a_b1", "a_w2", "a_b2", "a_mu_w", "a_mu_b", "a_std_w", "a_std_b")
    CRITIC_KEYS = ("c_w1", "c_b1", "c_w2", "c_b2", "c_out_w", "c_out_b")

    def __init__(self, cfg, gen):
        self.cfg = cfg
        self.gen = gen
        d_in = cfg.deter + cfg.stoch * cfg.classes + (cfg.gene_slice[1] - cfg.gene_slice[0])
        h, N = cfg.ac_hidden, cfg.max_slots
        shapes = {
            "a_w1": (N, h, d_in),
            "a_b1": (N, h),
            "a_w2": (N, h, h),
            "a_b2": (N, h),
            "a_mu_w": (N, cfg.action_dim, h),
            "a_mu_b": (N, cfg.action_dim),
            "a_std_w": (N, cfg.action_dim, h),
            "a_std_b": (N, cfg.action_dim),
            "c_w1": (N, h, d_in),
            "c_b1": (N, h),
            "c_w2": (N, h, h),
            "c_b2": (N, h),
            "c_out_w": (N, cfg.num_bins, h),
            "c_out_b": (N, cfg.num_bins),
        }
        self.params = {k: torch.zeros(*s) for k, s in shapes.items()}
        self.slow = {k: torch.zeros_like(self.params[k]) for k in self.CRITIC_KEYS}
        self.adam_m = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.adam_v = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.step_count = torch.zeros(N, dtype=torch.long)
        self.alive = torch.zeros(N, dtype=torch.bool)
        self.ret_scale = 1.0  # EMA(p95-p5) of λ-returns
        self.bins = make_bins(cfg)

    # --- slots --------------------------------------------------------------
    def _fresh_init(self, slot):
        for _k, p in self.params.items():
            if p.dim() == 3:  # weights: truncated-normal-ish fan-in init
                fan_in = p.shape[2]
                p[slot] = torch.randn(p.shape[1], p.shape[2], generator=self.gen) / math.sqrt(
                    fan_in
                )
            else:
                p[slot] = 0.0
        self.params["c_out_w"][slot] = 0.0  # zero-init critic head (spec §4.2)
        self.params["a_std_b"][slot] = -1.0  # start with modest stddev
        for k in self.CRITIC_KEYS:
            self.slow[k][slot] = self.params[k][slot]

    def acquire_slot(self, init_params):
        free = (~self.alive).nonzero(as_tuple=True)[0]
        if len(free) == 0:
            raise RuntimeError("actor slab full — raise RSSMConfig.max_slots")
        slot = int(free[0])
        for k in self.params:
            self.adam_m[k][slot] = 0.0
            self.adam_v[k][slot] = 0.0
        self.step_count[slot] = 0
        if init_params is None:
            self._fresh_init(slot)
        else:
            for k, v in init_params.items():
                (self.params if k in self.params else self.slow)[k][slot] = v
            for k in self.CRITIC_KEYS:
                self.slow[k][slot] = self.params[k][slot]
        self.alive[slot] = True
        return slot

    def release_slot(self, slot):
        self.alive[slot] = False

    def snapshot_slot(self, slot):
        return {k: self.params[k][slot].clone() for k in self.params}

    # --- forward ------------------------------------------------------------
    def _actor_dist(self, P, idx, s_g):
        x = F.silu(_blin(P["a_w1"][idx], P["a_b1"][idx], s_g))
        x = F.silu(_blin(P["a_w2"][idx], P["a_b2"][idx], x))
        mu = _blin(P["a_mu_w"][idx], P["a_mu_b"][idx], x)
        logstd = _blin(P["a_std_w"][idx], P["a_std_b"][idx], x).clamp(LOGSTD_MIN, LOGSTD_MAX)
        return mu, logstd

    def _critic_logits(self, P, idx, s_g):
        x = F.silu(_blin(P["c_w1"][idx], P["c_b1"][idx], s_g))
        x = F.silu(_blin(P["c_w2"][idx], P["c_b2"][idx], x))
        return _blin(P["c_out_w"][idx], P["c_out_b"][idx], x)

    @staticmethod
    def _tanh_normal_sample(mu, logstd, gen):
        eps = torch.randn(mu.shape, generator=gen, device=mu.device)
        pre = mu + eps * logstd.exp()
        return torch.tanh(pre), pre

    @staticmethod
    def _log_prob(mu, logstd, pre, act):
        base = -0.5 * (((pre - mu) / logstd.exp()) ** 2 + 2 * logstd + math.log(2 * math.pi))
        # tanh log-det-Jacobian (stable form)
        jac = 2.0 * (math.log(2.0) - pre - F.softplus(-2.0 * pre))
        return (base + jac).sum(-1)

    def act_single(self, slot, s_g, gen):
        with torch.no_grad():
            idx = torch.tensor([slot])
            mu, logstd = self._actor_dist(self.params, idx, s_g)
            a, _ = self._tanh_normal_sample(mu, logstd, gen)
        return a.squeeze(0), None

    # --- training -----------------------------------------------------------
    def imagination_update(self, wm, h0, z0, genes, slots, gen, replay_batch):
        cfg = self.cfg
        N, H = h0.shape[0], cfg.horizon
        reinforce = cfg.actor_grad == "reinforce"
        for p in self.params.values():
            p.requires_grad_(True)
        wm_req = [p.requires_grad for p in wm.parameters()]
        for p in wm.parameters():
            p.requires_grad_(False)  # WM frozen during AC update (both modes)

        h, z = h0, z0
        s_list, logp_list, ent_list, rew_list, cont_list = [], [], [], [], []
        for _ in range(H):
            s_g = wm.head_input(h, z, genes)
            mu, logstd = self._actor_dist(self.params, slots, s_g)
            a, pre = self._tanh_normal_sample(mu, logstd, gen)
            logp = self._log_prob(mu, logstd, pre, a)
            s_list.append(s_g)
            logp_list.append(logp)
            ent_list.append(-logp)  # sample-based entropy estimate
            a_step = a if not reinforce else a.detach()
            h, z, _ = wm.img_step(h, z, a_step, gen)
            s_next = wm.head_input(h, z, genes)
            rew_list.append(twohot_mean(wm.reward_logits(s_next), wm.bins))
            cont_list.append(wm.cont_prob(s_next))
        S = torch.stack(s_list, 1)  # (N,H,1284)
        logp = torch.stack(logp_list, 1)
        entropy = torch.stack(ent_list, 1)
        reward = torch.stack(rew_list, 1)
        cont = torch.stack(cont_list, 1)

        flat = S.reshape(N * H, -1)
        idx_rep = slots.repeat_interleave(H)
        v_logits = self._critic_logits(self.params, idx_rep, flat).reshape(N, H, -1)
        values = twohot_mean(v_logits, self.bins)  # (N,H)
        with torch.no_grad():
            s_gH = wm.head_input(h, z, genes)
            v_last = twohot_mean(self._critic_logits(self.params, slots, s_gH), self.bins)
        val_ext = torch.cat([values.detach() if reinforce else values, v_last.unsqueeze(1)], 1)

        R = lambda_returns(reward, cont, val_ext, cfg.gamma, cfg.lam)  # (N,H)
        w = torch.cumprod(torch.cat([torch.ones(N, 1), (cfg.gamma * cont)[:, :-1]], 1), 1).detach()

        # return normalization S = EMA(p95-p5), divide by max(1,S)
        with torch.no_grad():
            spread = (torch.quantile(R, 0.95) - torch.quantile(R, 0.05)).item()
            self.ret_scale = cfg.retnorm_decay * self.ret_scale + (1 - cfg.retnorm_decay) * spread
        scale = max(1.0, self.ret_scale)

        if reinforce:
            adv = ((R - values.detach()) / scale).detach()
            actor_loss = -(w * adv * logp).mean() - cfg.entropy_eta * (w * entropy).mean()
        else:  # dynamics backprop: stop-grad baseline, differentiable R
            adv = (R - values.detach()) / scale
            actor_loss = -(w * adv).mean() - cfg.entropy_eta * (w * entropy).mean()

        # critic loss: twohot CE to λ-returns + slow-critic regularizer (+ replay term).
        # Critic inputs are detached here (spec §4.2, dynamics-mode caveat): in
        # "dynamics" mode the imagined states S depend on the actor's actions, so
        # feeding the un-detached `flat` into the critic's OWN training loss would
        # leak an extra, unspecified gradient term from critic_loss back into the
        # actor params (on top of the intended path through `values`/R above,
        # which must stay differentiable for the dynamics-mode return). Detaching
        # here only affects the critic's regression loss, not its own weights'
        # gradient (which flows via c_w1/etc. regardless of the input's grad flag).
        tgt = twohot(symlog(R.detach()), self.bins)
        v_logits_c = self._critic_logits(self.params, idx_rep, flat.detach()).reshape(N, H, -1)
        logp_v = torch.log_softmax(v_logits_c, -1)
        critic_loss = -(w * (tgt * logp_v).sum(-1)).mean()
        with torch.no_grad():
            slow_logits = self._critic_logits(self.slow, idx_rep, flat.detach()).reshape(N, H, -1)
        critic_loss = critic_loss + cfg.slow_critic_scale * (
            -(w * (torch.softmax(slow_logits, -1) * logp_v).sum(-1)).mean()
        )
        if replay_batch is not None:
            ro, ra = replay_batch["s_g"], replay_batch["returns"]  # built by learner
            rl = self._critic_logits(self.params, replay_batch["slots"], ro)
            critic_loss = critic_loss + cfg.replay_critic_scale * (
                -(twohot(symlog(ra), self.bins) * torch.log_softmax(rl, -1)).sum(-1).mean()
            )

        loss = actor_loss + critic_loss
        grads = torch.autograd.grad(loss, list(self.params.values()), allow_unused=True)
        self._adam_step(dict(zip(self.params.keys(), grads)), slots)
        for p, r in zip(wm.parameters(), wm_req):
            p.requires_grad_(r)
        for p in self.params.values():
            p.requires_grad_(False)
        self._slow_critic_ema(slots)
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "ret_scale": scale,
            "entropy": entropy.mean().item(),
        }

    def _adam_step(self, grads, slots):
        cfg = self.cfg
        self.step_count[slots] += 1
        with torch.no_grad():
            # global-norm clip over the touched slots' grads
            total = torch.sqrt(sum((g[slots] ** 2).sum() for g in grads.values() if g is not None))
            clip = min(1.0, cfg.ac_clip / (float(total) + 1e-8))
            for k, g in grads.items():
                if g is None:
                    continue
                g = g * clip
                m, v = self.adam_m[k], self.adam_v[k]
                m[slots] = cfg.adam_beta1 * m[slots] + (1 - cfg.adam_beta1) * g[slots]
                v[slots] = cfg.adam_beta2 * v[slots] + (1 - cfg.adam_beta2) * g[slots] ** 2
                t = self.step_count[slots].float().clamp(min=1)
                shape = [-1] + [1] * (m.dim() - 1)
                mhat = m[slots] / (1 - cfg.adam_beta1**t).reshape(*shape)
                vhat = v[slots] / (1 - cfg.adam_beta2**t).reshape(*shape)
                self.params[k].data[slots] -= cfg.ac_lr * mhat / (vhat.sqrt() + cfg.adam_eps)

    def _slow_critic_ema(self, slots):
        d = self.cfg.slow_critic_decay
        with torch.no_grad():
            for k in self.CRITIC_KEYS:
                self.slow[k][slots] = d * self.slow[k][slots] + (1 - d) * self.params[k][slots]

    # --- persistence ----------------------------------------------------------
    def state_dict_all(self):
        return {
            "params": self.params,
            "slow": self.slow,
            "adam_m": self.adam_m,
            "adam_v": self.adam_v,
            "step_count": self.step_count,
            "alive": self.alive,
            "ret_scale": self.ret_scale,
        }

    def load_state(self, sd):
        for name in ("params", "slow", "adam_m", "adam_v"):
            for k, v in sd[name].items():
                getattr(self, name)[k].copy_(v)
        self.step_count.copy_(sd["step_count"])
        self.alive.copy_(sd["alive"])
        self.ret_scale = sd["ret_scale"]
