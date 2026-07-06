from __future__ import annotations

import torch

from artificial_society.agents.rssm.config import RSSMConfig, make_generator


def test_defaults_match_spec():
    cfg = RSSMConfig()
    assert (cfg.obs_dim, cfg.action_dim) == (57, 7)
    assert (cfg.deter, cfg.stoch, cfg.classes) == (256, 32, 32)
    assert (cfg.beta_pred, cfg.beta_dyn, cfg.beta_rep) == (1.0, 0.5, 0.1)
    assert cfg.unimix == 0.01
    assert (cfg.num_bins, cfg.bin_low, cfg.bin_high) == (255, -20.0, 20.0)
    assert (cfg.wm_lr, cfg.ac_lr, cfg.adam_eps) == (1e-4, 3e-5, 1e-8)
    assert (cfg.wm_clip, cfg.ac_clip) == (1000.0, 100.0)
    assert (cfg.batch_size, cfg.seq_len) == (16, 64)
    assert (cfg.horizon, cfg.gamma, cfg.lam) == (15, 0.997, 0.95)
    assert cfg.entropy_eta == 3e-4
    assert cfg.actor_grad == "reinforce"
    assert (cfg.retnorm_decay, cfg.slow_critic_decay) == (0.99, 0.98)
    assert cfg.replay_critic_scale == 0.3
    assert (cfg.train_every, cfg.young_ratio_cap) == (8, 4)
    assert cfg.prototype_decay == 0.995
    assert cfg.gene_slice == (17, 21)
    assert cfg.max_slots >= 128


def test_generator_isolated_from_global():
    torch.manual_seed(0)
    before = torch.get_rng_state()
    g = make_generator(42, "learner")
    _ = torch.rand(1000, generator=g)
    assert torch.equal(before, torch.get_rng_state())  # global untouched


def test_generator_deterministic_and_keyed():
    a1 = torch.rand(8, generator=make_generator(42, ("agent", 7)))
    a2 = torch.rand(8, generator=make_generator(42, ("agent", 7)))
    b = torch.rand(8, generator=make_generator(42, ("agent", 8)))
    c = torch.rand(8, generator=make_generator(43, ("agent", 7)))
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b)
    assert not torch.equal(a1, c)


def test_generator_device_param_default_cpu():
    assert make_generator(1, "x").device.type == "cpu"
