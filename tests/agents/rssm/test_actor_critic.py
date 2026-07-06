from __future__ import annotations

import dataclasses

import torch

from artificial_society.agents.rssm.actor_critic import ActorCriticSlab, lambda_returns
from artificial_society.agents.rssm.config import RSSMConfig, make_generator
from artificial_society.agents.rssm.world_model import RSSMWorldModel

CFG = dataclasses.replace(RSSMConfig(), max_slots=8, horizon=5)


def _slab(cfg=CFG):
    return ActorCriticSlab(cfg, make_generator(0, "slab"))


def _starts(n=4):
    g = make_generator(1, "st")
    return (torch.randn(n, 256, generator=g), torch.zeros(n, 32, 32), torch.rand(n, 4, generator=g))


def test_lambda_returns_recursion():
    # H=3, gamma=1, lam=1 → pure discounted sum with continues folded in
    rew = torch.tensor([[1.0, 1.0, 1.0]])
    cont = torch.tensor([[1.0, 0.0, 1.0]])
    val = torch.zeros(1, 4)
    R = lambda_returns(rew, cont, val, gamma=1.0, lam=1.0)
    assert torch.allclose(R[0], torch.tensor([2.0, 1.0, 1.0]))  # death at t=1 cuts the tail


def test_act_single_deterministic_and_slot_isolated():
    slab = _slab()
    s0, _s1 = slab.acquire_slot(None), slab.acquire_slot(None)
    s_g = torch.randn(1, 256 + 1024 + 4, generator=make_generator(2, "x"))
    a1, _ = slab.act_single(s0, s_g, make_generator(3, ("a", 1)))
    a2, _ = slab.act_single(s0, s_g, make_generator(3, ("a", 1)))
    assert torch.equal(a1, a2) and a1.shape == (7,)
    assert torch.all(a1 <= 1.0) and torch.all(a1 >= -1.0)


def test_imagination_update_moves_only_given_slots():
    slab = _slab()
    slots = torch.tensor([slab.acquire_slot(None), slab.acquire_slot(None)])
    frozen = slab.acquire_slot(None)
    before_frozen = slab.params["a_w1"][frozen].clone()
    before_active = slab.params["a_w1"][slots[0]].clone()
    wm = RSSMWorldModel(CFG)
    h0, z0, genes = _starts(2)
    m = slab.imagination_update(wm, h0, z0, genes, slots, make_generator(4, "img"), None)
    assert all(map(lambda v: torch.isfinite(torch.tensor(float(v))), m.values()))
    assert torch.equal(slab.params["a_w1"][frozen], before_frozen)  # masked update
    assert not torch.equal(slab.params["a_w1"][slots[0]], before_active)


def test_reinforce_gradient_flow_contract():
    """Reinforce mode: WM must receive ZERO grad from the actor loss; actor params must receive grad."""
    slab = _slab()
    slots = torch.tensor([slab.acquire_slot(None)])
    wm = RSSMWorldModel(CFG)
    h0, z0, genes = _starts(1)
    wm_before = [p.clone() for p in wm.parameters()]
    slab.imagination_update(wm, h0, z0, genes, slots, make_generator(5, "img"), None)
    assert all(torch.equal(a, b) for a, b in zip(wm_before, wm.parameters()))  # WM untouched
    assert all(p.grad is None for p in wm.parameters())  # and no grads left


def test_dynamics_mode_end_to_end_gradient():
    cfg = dataclasses.replace(CFG, actor_grad="dynamics")
    slab = ActorCriticSlab(cfg, make_generator(0, "slab"))
    slots = torch.tensor([slab.acquire_slot(None)])
    wm = RSSMWorldModel(cfg)
    h0, z0, genes = _starts(1)
    before = slab.params["a_w1"][slots[0]].clone()
    slab.imagination_update(wm, h0, z0, genes, slots, make_generator(6, "img"), None)
    assert not torch.equal(slab.params["a_w1"][slots[0]], before)  # end-to-end grads arrived
    assert all(p.grad is None for p in wm.parameters())  # WM frozen during actor update


def test_slot_hygiene_on_release_and_reacquire():
    slab = _slab()
    s = slab.acquire_slot(None)
    wm = RSSMWorldModel(CFG)
    h0, z0, genes = _starts(1)
    slab.imagination_update(wm, h0, z0, genes, torch.tensor([s]), make_generator(7, "i"), None)
    assert slab.step_count[s] > 0
    slab.release_slot(s)
    assert not slab.alive[s]
    s2 = slab.acquire_slot(None)
    assert s2 == s  # freed slot reused
    assert slab.step_count[s2] == 0  # step count reset
    assert torch.all(slab.adam_m["a_w1"][s2] == 0)  # both moments zeroed
    assert torch.all(slab.adam_v["a_w1"][s2] == 0)


def test_prototype_snapshot_roundtrip():
    slab = _slab()
    s = slab.acquire_slot(None)
    snap = slab.snapshot_slot(s)
    s2 = slab.acquire_slot(snap)
    assert torch.equal(slab.params["a_w1"][s], slab.params["a_w1"][s2])
