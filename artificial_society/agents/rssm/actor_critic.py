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

    def __init__(self, cfg, gen, device="cpu"):
        self.cfg = cfg
        self.gen = gen
        self.device = torch.device(device)
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
        # params/adam moments live on `device` (train_device, may be cuda) — the act
        # path never touches these directly when train_device != cpu, see
        # SharedLearner._act_actor. step_count/alive are pure bookkeeping (small,
        # scalar-ish, indexed from many call sites) and stay on CPU regardless.
        self.params = {k: torch.zeros(*s, device=self.device) for k, s in shapes.items()}
        self.slow = {k: torch.zeros_like(self.params[k]) for k in self.CRITIC_KEYS}
        self.adam_m = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.adam_v = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.step_count = torch.zeros(N, dtype=torch.long)
        self.alive = torch.zeros(N, dtype=torch.bool)
        self.ret_scale = 1.0  # EMA(p95-p5) of λ-returns
        self.bins = make_bins(cfg).to(self.device)

    # --- slots --------------------------------------------------------------
    def _fresh_init(self, slot):
        for _k, p in self.params.items():
            if p.dim() == 3:  # weights: truncated-normal-ish fan-in init
                fan_in = p.shape[2]
                # self.gen is always CPU (SharedLearner constructs it via
                # make_generator with no device kwarg); draw on CPU, then move onto
                # p's device (a no-op when p is itself CPU).
                draw = torch.randn(p.shape[1], p.shape[2], generator=self.gen) / math.sqrt(fan_in)
                p[slot] = draw.to(p.device)
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
            # init_params comes from PrototypeActor.template(), always CPU (spec
            # §GPU-pilot); explicit .to(device) makes the cross-device copy into a
            # cuda-resident slab an intentional, documented transfer rather than
            # relying on an implicit copy inside indexed tensor assignment.
            for k, v in init_params.items():
                target = self.params if k in self.params else self.slow
                target[k][slot] = v.to(target[k].device)
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
        # Draw on the GENERATOR's device, not mu's: a CPU generator cannot fill a
        # CUDA tensor directly (and vice versa). The act path deliberately pairs a
        # CPU generator with CPU mu (via the CPU actor mirror), so this `.to()` is a
        # no-op there; it only does real work for train/imagination calls where
        # mu and gen already agree on train_device anyway.
        eps = torch.randn(mu.shape, generator=gen, device=gen.device).to(mu.device)
        pre = mu + eps * logstd.exp()
        return torch.tanh(pre), pre

    @staticmethod
    def _log_prob(mu, logstd, pre, act):
        base = -0.5 * (((pre - mu) / logstd.exp()) ** 2 + 2 * logstd + math.log(2 * math.pi))
        # tanh log-det-Jacobian (stable form): jac == log(1 - tanh(pre)**2).
        # Change-of-variables density is base - jac (review fix, NEW-2); the
        # pre-fix `base + jac` inverted the sign.
        jac = 2.0 * (math.log(2.0) - pre - F.softplus(-2.0 * pre))
        return (base - jac).sum(-1)

    def act_single(self, slot, s_g, gen, params_override=None):
        """params_override: optional CPU actor-param mirror (see actor_snapshot_cpu)
        used by SharedLearner.act() so the act path stays fully CPU even when this
        slab's own params live on train_device."""
        P = params_override if params_override is not None else self.params
        with torch.no_grad():
            idx = torch.tensor([slot], device=P["a_w1"].device)
            mu, logstd = self._actor_dist(P, idx, s_g)
            a, _ = self._tanh_normal_sample(mu, logstd, gen)
        return a.squeeze(0), None

    def actor_snapshot_cpu(self) -> dict:
        """CPU clones of the actor (not critic) params, for the CPU act-path mirror."""
        return {k: self.params[k].detach().to("cpu").clone() for k in self.ACTOR_KEYS}

    def refresh_actor_mirror_slot(self, mirror: dict, slot: int) -> None:
        """Copy one slot's current actor params into an existing CPU mirror dict."""
        for k in self.ACTOR_KEYS:
            mirror[k][slot] = self.params[k][slot].detach().to("cpu")

    # --- training -----------------------------------------------------------
    def _losses(self, wm, h0, z0, genes, slots, gen, replay_batch):
        """Build the imagined rollout and return (actor_loss, critic_loss, entropy, scale).

        Pure loss construction, no grad/optimizer side effects — split out so tests
        can probe the gradient topology (which params each loss actually touches)
        without going through `_adam_step`.
        """
        cfg = self.cfg
        N, H = h0.shape[0], cfg.horizon
        reinforce = cfg.actor_grad == "reinforce"
        h, z = h0, z0
        s_list, logp_list, logp_score_list, ent_list, rew_list, cont_list = [], [], [], [], [], []
        for _ in range(H):
            s_g = wm.head_input(h, z, genes)
            mu, logstd = self._actor_dist(self.params, slots, s_g)
            a, pre = self._tanh_normal_sample(mu, logstd, gen)
            # Pathwise (reparameterized) log-prob — used ONLY for the entropy
            # bonus below, where `-logp`'s gradient through mu/logstd via `pre`
            # is a valid reparameterized entropy gradient.
            logp = self._log_prob(mu, logstd, pre, a)
            s_list.append(s_g)
            logp_list.append(logp)
            ent_list.append(-logp)  # sample-based entropy estimate
            if reinforce:
                # Score-function term (review fix, Critical 1): density evaluated
                # at a CONSTANT sample (pre/a detached) so d(logp_score)/d(mu,
                # logstd) is the textbook score (pre-mu)/sigma^2. Feeding the
                # non-detached, reparameterized `pre` here instead (the pre-fix
                # bug) is wrong: pre = mu + eps*sigma makes d(pre-mu)/dmu == 0
                # identically, so autograd's total derivative silently cancels
                # the intended score term, leaving only a small wrong-direction
                # leftover through the tanh log-det-Jacobian (probe: bandit true
                # grad -0.893 vs the buggy estimator's +0.218).
                logp_score_list.append(self._log_prob(mu, logstd, pre.detach(), a.detach()))
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
        ones = torch.ones(N, 1, device=reward.device)
        w = torch.cumprod(torch.cat([ones, (cfg.gamma * cont)[:, :-1]], 1), 1).detach()

        # return normalization S = EMA(p95-p5), divide by max(1,S)
        with torch.no_grad():
            spread = (torch.quantile(R, 0.95) - torch.quantile(R, 0.05)).item()
            self.ret_scale = cfg.retnorm_decay * self.ret_scale + (1 - cfg.retnorm_decay) * spread
        scale = max(1.0, self.ret_scale)

        if reinforce:
            logp_score = torch.stack(logp_score_list, 1)
            adv = ((R - values.detach()) / scale).detach()
            actor_loss = -(w * adv * logp_score).mean() - cfg.entropy_eta * (w * entropy).mean()
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

        return actor_loss, critic_loss, entropy, scale

    def imagination_update(self, wm, h0, z0, genes, slots, gen, replay_batch):
        for p in self.params.values():
            p.requires_grad_(True)
        wm_req = [p.requires_grad for p in wm.parameters()]
        for p in wm.parameters():
            p.requires_grad_(False)  # WM frozen during AC update (both modes)

        actor_loss, critic_loss, entropy, scale = self._losses(
            wm, h0, z0, genes, slots, gen, replay_batch
        )

        # Split grad computation (review fix, dynamics-mode cross-term leak): in
        # "dynamics" mode actor_loss is differentiable w.r.t. critic params too (it
        # depends on the critic's own `values` forward pass through the un-detached
        # λ-return bootstrap). A single torch.autograd.grad(actor_loss + critic_loss,
        # all_params) would therefore let the ACTOR objective push the CRITIC's
        # weights. Requesting grads against each param list separately keeps the
        # actor's pathwise gradient through critic params as intermediate *nodes*
        # (chain rule still traverses them to reach actor params) while excluding
        # d(actor_loss)/d(critic_params) and d(critic_loss)/d(actor_params) from the
        # respective updates. retain_graph=True on the first call because the second
        # call's backward may still need buffers from the shared forward graph.
        actor_param_list = [self.params[k] for k in self.ACTOR_KEYS]
        critic_param_list = [self.params[k] for k in self.CRITIC_KEYS]
        actor_grads = torch.autograd.grad(
            actor_loss, actor_param_list, retain_graph=True, allow_unused=True
        )
        critic_grads = torch.autograd.grad(critic_loss, critic_param_list, allow_unused=True)
        grads = dict(zip(self.ACTOR_KEYS, actor_grads))
        grads.update(zip(self.CRITIC_KEYS, critic_grads))
        self._adam_step(grads, slots)
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
        # `slots` indexes self.params/adam_m/adam_v (train_device) but step_count is
        # always CPU (bookkeeping) — fancy-indexing requires the index tensor's
        # device to match the indexed tensor's, so keep a CPU copy for step_count
        # and bring the derived bias-correction term `t` back to train_device.
        device = self.params["a_w1"].device
        slots_cpu = slots.cpu()
        self.step_count[slots_cpu] += 1
        with torch.no_grad():
            # global-norm clip over the touched slots' grads. `slots` legitimately
            # contains duplicates (a young agent can get multiple imagination starts
            # in one learner batch); dedup so a slot's grad isn't squared in k times.
            uniq = slots.unique()
            total = torch.sqrt(sum((g[uniq] ** 2).sum() for g in grads.values() if g is not None))
            clip = min(1.0, cfg.ac_clip / (float(total) + 1e-8))
            t = self.step_count[slots_cpu].float().clamp(min=1).to(device)
            for k, g in grads.items():
                if g is None:
                    continue
                g = g * clip
                m, v = self.adam_m[k], self.adam_v[k]
                m[slots] = cfg.adam_beta1 * m[slots] + (1 - cfg.adam_beta1) * g[slots]
                v[slots] = cfg.adam_beta2 * v[slots] + (1 - cfg.adam_beta2) * g[slots] ** 2
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
        # Checkpoint payload must be device-independent (spec §GPU-pilot): a
        # checkpoint saved on a cuda-trained run must load on a CPU-only box and
        # vice versa. `.detach().to("cpu")` is a no-op (same object, no clone) when
        # a tensor is already on CPU and not requiring grad — which is always true
        # here outside of imagination_update's transient window — so this preserves
        # the CPU-slab "live reference" behavior tests rely on while adding a real
        # CPU copy for cuda-device slabs.
        return {
            "params": {k: v.detach().to("cpu") for k, v in self.params.items()},
            "slow": {k: v.detach().to("cpu") for k, v in self.slow.items()},
            "adam_m": {k: v.detach().to("cpu") for k, v in self.adam_m.items()},
            "adam_v": {k: v.detach().to("cpu") for k, v in self.adam_v.items()},
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
