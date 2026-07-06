from __future__ import annotations

import torch

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.agents.rssm.util import (
    make_bins,
    mlp,
    symexp,
    symlog,
    twohot,
    twohot_mean,
    zero_init_,
)

CFG = RSSMConfig()


def test_symlog_roundtrip():
    x = torch.tensor([-1e4, -3.7, 0.0, 0.5, 2e5])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4, atol=1e-5)


def test_twohot_roundtrip():
    bins = make_bins(CFG)
    assert bins.shape == (255,)
    vals = torch.tensor([-70.0, -3.0, 0.0, 0.25, 1.5, 400.0])
    probs = twohot(symlog(vals), bins)
    assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-6)
    # decode via log-probs path used at train time: twohot_mean expects logits
    logits = (probs + 1e-8).log()
    assert torch.allclose(twohot_mean(logits, bins), vals, rtol=1e-2, atol=1e-2)


def test_twohot_clamps_outside_support():
    bins = make_bins(CFG)
    probs = twohot(torch.tensor([999.0]), bins)  # symlog-space input beyond bin_high
    assert probs[0, -1] == 1.0


def test_mlp_shapes_and_zero_init():
    net = mlp(10, 32, 5)
    y = net(torch.randn(4, 10))
    assert y.shape == (4, 5)
    zero_init_(net[-1])
    assert torch.all(net[-1].weight == 0) and torch.all(net[-1].bias == 0)
