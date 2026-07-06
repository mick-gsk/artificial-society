"""Shared numeric primitives: symlog, fixed twohot support, MLP factory (spec §4.1)."""

from __future__ import annotations

import torch
from torch import nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def make_bins(cfg) -> torch.Tensor:
    """Fixed twohot support in symlog space (Dreamer-v3: 255 bins over [-20, 20])."""
    return torch.linspace(cfg.bin_low, cfg.bin_high, cfg.num_bins)


def twohot(y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """y: symlog-space targets (...,). Returns (..., num_bins) twohot distribution."""
    y = y.clamp(bins[0], bins[-1])
    idx = torch.searchsorted(bins, y.detach(), right=True).clamp(1, len(bins) - 1)
    lo, hi = bins[idx - 1], bins[idx]
    w_hi = ((y - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)
    out = torch.zeros(*y.shape, len(bins), dtype=torch.float32, device=y.device)
    out.scatter_(-1, (idx - 1).unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
    out.scatter_add_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


def twohot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value under softmax(logits) over the symlog support, mapped back via symexp."""
    return symexp((torch.softmax(logits, -1) * bins).sum(-1))


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """Linear→LayerNorm→SiLU ×layers, then plain Linear head."""
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def zero_init_(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
