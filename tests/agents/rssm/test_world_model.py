from __future__ import annotations

import torch

from artificial_society.agents.rssm.config import RSSMConfig, make_generator
from artificial_society.agents.rssm.world_model import RSSMWorldModel

CFG = RSSMConfig()


def test_gene_slice_static_over_life():
    from artificial_society.simulation import Simulation

    sim = Simulation(
        headless=True,
        load_checkpoint=False,
        seed=42,
        grid_w=20,
        grid_h=15,
        initial_population=8,
    )
    ag = sim.agents[0]
    lo, hi = RSSMConfig().gene_slice
    f0 = ag.local_features(sim.world, sim.agents)[lo:hi]
    for _ in range(5):
        sim.step()
    if ag.alive:
        f1 = ag.local_features(sim.world, sim.agents)[lo:hi]
        assert f0 == f1  # genes never change within a life


def _batch(B=3, L=5, gen=None):
    g = gen or make_generator(0, "data")
    return {
        "obs": torch.rand(B, L, CFG.obs_dim, generator=g),
        "act": torch.rand(B, L, CFG.action_dim, generator=g) * 2 - 1,
        "reward": torch.rand(B, L, generator=g),
        "cont": torch.ones(B, L),
        "mask": torch.ones(B, L),
    }


def test_shapes_and_determinism():
    wm = RSSMWorldModel(CFG)
    b = _batch()
    h, z, post, prior = wm.observe(b["obs"], b["act"], make_generator(7, "s"))
    assert h.shape == (3, 5, 256) and z.shape == (3, 5, 32, 32)
    assert post.shape == prior.shape == (3, 5, 32, 32)
    assert torch.all(z.sum(-1) == 1.0)  # one-hot per categorical
    h2, z2, _, _ = wm.observe(b["obs"], b["act"], make_generator(7, "s"))
    assert torch.equal(z, z2)  # same generator seed → identical samples


def test_kl_free_bits_floor():
    wm = RSSMWorldModel(CFG)
    logits = torch.zeros(4, 6, 32, 32)
    kl_dyn, kl_rep = wm.kl_losses(logits, logits.clone(), torch.ones(4, 6))
    assert abs(kl_dyn.item() - 1.0) < 1e-5  # identical dists → clamped to 1 nat
    assert abs(kl_rep.item() - 1.0) < 1e-5


def test_train_batch_steps_and_counts():
    wm = RSSMWorldModel(CFG)
    before = [p.clone() for p in wm.parameters()]
    m = wm.train_batch(_batch(), make_generator(1, "t"))
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values())
    assert wm.wm_updates == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, wm.parameters()))


def test_decoder_weights_downweight_irreducible():
    wm = RSSMWorldModel(CFG)
    w = wm.recon_weight
    for d in CFG.irreducible_dims:
        assert w[d] == CFG.irreducible_weight
    keep = [i for i in range(CFG.obs_dim) if i not in CFG.irreducible_dims]
    assert torch.all(w[keep] == 1.0)


def test_reward_head_zero_init():
    wm = RSSMWorldModel(CFG)
    s_g = torch.randn(2, 256 + 32 * 32 + 4)
    assert torch.all(wm.reward_logits(s_g) == 0)  # zero-init final layer
