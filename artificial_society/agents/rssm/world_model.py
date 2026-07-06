"""Shared Dreamer-v3-style RSSM world model (spec §4.1). One instance, sim-owned."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .util import make_bins, mlp, symlog, twohot, zero_init_


def st_sample(logits: torch.Tensor, unimix: float, gen: torch.Generator) -> torch.Tensor:
    """Straight-through one-hot sample from (B, stoch, classes) logits with unimix."""
    probs = torch.softmax(logits, -1)
    probs = (1.0 - unimix) * probs + unimix / probs.shape[-1]
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, 1, generator=gen).reshape(probs.shape[:-1])
    onehot = torch.nn.functional.one_hot(idx, probs.shape[-1]).float()
    # order matters for exact one-hot forward values: (probs - probs.detach())
    # cancels to bit-exact 0.0 (same float subtracted from itself), whereas
    # onehot + probs - probs.detach() accumulates ~1 ULP of rounding error
    # per addition and fails exact-sum-to-1 checks.
    return (probs - probs.detach()) + onehot  # straight-through gradients


def _unimix_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    p = torch.softmax(logits, -1)
    return (1.0 - unimix) * p + unimix / p.shape[-1]


class RSSMWorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        z_dim = cfg.stoch * cfg.classes
        gene_dim = cfg.gene_slice[1] - cfg.gene_slice[0]
        head_in = cfg.deter + z_dim + gene_dim
        self.embed = mlp(cfg.obs_dim, cfg.mlp_hidden, cfg.mlp_hidden)
        self.post_net = nn.Linear(cfg.deter + cfg.mlp_hidden, z_dim)
        self.prior_net = mlp(cfg.deter, cfg.mlp_hidden, z_dim)
        self.pre_gru = mlp(z_dim + cfg.action_dim, cfg.mlp_hidden, cfg.mlp_hidden, layers=1)
        self.gru = nn.GRUCell(cfg.mlp_hidden, cfg.deter)
        self.decoder = mlp(head_in, cfg.mlp_hidden, cfg.obs_dim)
        self.reward_head = mlp(head_in, cfg.mlp_hidden, cfg.num_bins)
        zero_init_(self.reward_head[-1])
        self.cont_head = mlp(head_in, cfg.mlp_hidden, 1)
        self.register_buffer("bins", make_bins(cfg))
        w = torch.ones(cfg.obs_dim)
        for d in cfg.irreducible_dims:
            w[d] = cfg.irreducible_weight
        self.register_buffer("recon_weight", w)
        self.opt = torch.optim.Adam(
            self.parameters(),
            lr=cfg.wm_lr,
            eps=cfg.adam_eps,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
        self.wm_updates = 0

    # --- state machinery -------------------------------------------------
    def initial_state(self, batch: int, device=None):
        c = self.cfg
        dev = device or next(self.parameters()).device
        return (
            torch.zeros(batch, c.deter, device=dev),
            torch.zeros(batch, c.stoch, c.classes, device=dev),
        )

    def _advance(self, h, z, a_prev):
        x = self.pre_gru(torch.cat([z.flatten(1), a_prev], -1))
        return self.gru(x, h)

    def obs_step(self, h, z, a_prev, obs, gen):
        h = self._advance(h, z, a_prev)
        prior_logits = self.prior_net(h).reshape(-1, self.cfg.stoch, self.cfg.classes)
        e = self.embed(symlog(obs))
        post_logits = self.post_net(torch.cat([h, e], -1)).reshape(
            -1, self.cfg.stoch, self.cfg.classes
        )
        z_post = st_sample(post_logits, self.cfg.unimix, gen)
        return h, z_post, post_logits, prior_logits

    def img_step(self, h, z, a, gen):
        h = self._advance(h, z, a)
        prior_logits = self.prior_net(h).reshape(-1, self.cfg.stoch, self.cfg.classes)
        z_prior = st_sample(prior_logits, self.cfg.unimix, gen)
        return h, z_prior, prior_logits

    def observe(self, obs, act, gen):
        B, L, _ = obs.shape
        h, z = self.initial_state(B, obs.device)
        hs, zs, posts, priors = [], [], [], []
        a_prev = torch.zeros(B, self.cfg.action_dim, device=obs.device)
        for t in range(L):
            h, z, post_l, prior_l = self.obs_step(h, z, a_prev, obs[:, t], gen)
            a_prev = act[:, t]
            hs.append(h)
            zs.append(z)
            posts.append(post_l)
            priors.append(prior_l)
        stack = lambda xs: torch.stack(xs, 1)  # noqa: E731
        return stack(hs), stack(zs), stack(posts), stack(priors)

    # --- heads ------------------------------------------------------------
    def head_input(self, h, z, genes):
        return torch.cat([h, z.flatten(-2), genes], -1)

    def decode(self, s_g):
        return self.decoder(s_g)

    def reward_logits(self, s_g):
        return self.reward_head(s_g)

    def cont_prob(self, s_g):
        return torch.sigmoid(self.cont_head(s_g)).squeeze(-1)

    # --- losses -----------------------------------------------------------
    def kl_losses(self, post_logits, prior_logits, mask):
        c = self.cfg
        q, qs = _unimix_probs(post_logits, c.unimix), _unimix_probs(post_logits.detach(), c.unimix)
        p, ps = (
            _unimix_probs(prior_logits, c.unimix),
            _unimix_probs(prior_logits.detach(), c.unimix),
        )

        def kl(a, b):
            return (a * (a.clamp_min(1e-8).log() - b.clamp_min(1e-8).log())).sum(-1).sum(-1)

        # per-timestep clamp (free bits) BEFORE averaging; mask out padding
        dyn = (torch.clamp(kl(qs, p), min=1.0) * mask).sum() / mask.sum()
        rep = (torch.clamp(kl(q, ps), min=1.0) * mask).sum() / mask.sum()
        return dyn, rep

    def train_batch(self, batch, gen):
        c = self.cfg
        obs, act = batch["obs"], batch["act"]
        mask = batch["mask"]
        h, z, post_l, prior_l = self.observe(obs, act, gen)
        genes = obs[..., c.gene_slice[0] : c.gene_slice[1]]
        s_g = self.head_input(h, z, genes)
        recon = ((self.decode(s_g) - symlog(obs)) ** 2 * self.recon_weight).mean(-1)
        recon = (recon * mask).sum() / mask.sum()
        target = twohot(symlog(batch["reward"]), self.bins)
        logp = torch.log_softmax(self.reward_logits(s_g), -1)
        rew_loss = (-(target * logp).sum(-1) * mask).sum() / mask.sum()
        cont_logit = self.cont_head(s_g).squeeze(-1)
        cont_loss = nn.functional.binary_cross_entropy_with_logits(
            cont_logit, batch["cont"], reduction="none"
        )
        cont_loss = (cont_loss * mask).sum() / mask.sum()
        kl_dyn, kl_rep = self.kl_losses(post_l, prior_l, mask)
        loss = (
            c.beta_pred * (recon + rew_loss + cont_loss) + c.beta_dyn * kl_dyn + c.beta_rep * kl_rep
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.parameters(), c.wm_clip)
        self.opt.step()
        self.wm_updates += 1
        return {
            "loss": loss.item(),
            "recon": recon.item(),
            "reward": rew_loss.item(),
            "cont": cont_loss.item(),
            "kl_dyn": kl_dyn.item(),
            "kl_rep": kl_rep.item(),
        }
