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


def _snapshot(slab):
    """Deep-clone the slab's full optimizer/param state so a run can be replayed
    from an identical starting point (`state_dict_all()` returns live references,
    not clones, so it can't be used as-is for restore-after-mutation)."""
    return {
        "params": {k: v.clone() for k, v in slab.params.items()},
        "adam_m": {k: v.clone() for k, v in slab.adam_m.items()},
        "adam_v": {k: v.clone() for k, v in slab.adam_v.items()},
        "step_count": slab.step_count.clone(),
        "ret_scale": slab.ret_scale,
    }


def _restore(slab, snap):
    for k, v in snap["params"].items():
        slab.params[k].data.copy_(v)
    for k, v in snap["adam_m"].items():
        slab.adam_m[k].copy_(v)
    for k, v in snap["adam_v"].items():
        slab.adam_v[k].copy_(v)
    slab.step_count.copy_(snap["step_count"])
    slab.ret_scale = snap["ret_scale"]


def _naive_combined_update(slab, wm, h0, z0, genes, slots, seed):
    """Replicates the PRE-FIX behavior: a single
    `torch.autograd.grad(actor_loss + critic_loss, all_params)` call feeding
    `_adam_step`, instead of the split actor/critic grad calls. Used as a reference
    to diff against the real (fixed) `imagination_update` from an identical
    snapshotted starting state.
    """
    for p in slab.params.values():
        p.requires_grad_(True)
    wm_req = [p.requires_grad for p in wm.parameters()]
    for p in wm.parameters():
        p.requires_grad_(False)
    actor_loss, critic_loss, _entropy, _scale = slab._losses(
        wm, h0, z0, genes, slots, make_generator(seed, "img"), None
    )
    actor_params = [slab.params[k] for k in slab.ACTOR_KEYS]
    critic_to_actor = torch.autograd.grad(
        critic_loss, actor_params, retain_graph=True, allow_unused=True
    )
    loss = actor_loss + critic_loss
    grads = torch.autograd.grad(loss, list(slab.params.values()), allow_unused=True)
    slab._adam_step(dict(zip(slab.params.keys(), grads)), slots)
    for p, r in zip(wm.parameters(), wm_req):
        p.requires_grad_(r)
    for p in slab.params.values():
        p.requires_grad_(False)
    return critic_to_actor


def _run_mode_comparison(cfg, seed):
    """Run the REAL (fixed) imagination_update, then reset to the same starting
    state and run a manual replication of the pre-fix combined-grad update; return
    both resulting critic params (keyed slot-only) plus critic_loss's raw gradient
    onto actor params (must be None/zero in both modes regardless of the fix,
    since critic_loss's inputs are always detached from the actor's rollout)."""
    slab = ActorCriticSlab(cfg, make_generator(0, "slab"))
    slots = torch.tensor([slab.acquire_slot(None)])
    wm = RSSMWorldModel(cfg)
    h0, z0, genes = _starts(1)
    snap = _snapshot(slab)

    slab.imagination_update(wm, h0, z0, genes, slots, make_generator(seed, "img"), None)
    real_critic = {k: slab.params[k][slots].clone() for k in slab.CRITIC_KEYS}

    _restore(slab, snap)
    critic_to_actor = _naive_combined_update(slab, wm, h0, z0, genes, slots, seed)
    naive_critic = {k: slab.params[k][slots].clone() for k in slab.CRITIC_KEYS}

    return real_critic, naive_critic, critic_to_actor


def test_reinforce_mode_critic_update_matches_naive_combined_grad():
    """Sanity companion to the dynamics regression below: reinforce mode has no
    actor->critic dependency (the advantage is fully `.detach()`-ed), so splitting
    the grad call per the review fix must be a functional no-op here — the real
    (split) update and a manually replicated pre-fix combined-loss update, run from
    an identical starting state, must land on identical critic params. Also checks
    the always-true invariant that critic_loss never reaches actor params.
    """
    real_critic, naive_critic, critic_to_actor = _run_mode_comparison(CFG, seed=8)
    assert all(g is None or torch.all(g == 0) for g in critic_to_actor)
    for k in ActorCriticSlab.CRITIC_KEYS:
        assert torch.allclose(real_critic[k], naive_critic[k], atol=1e-6)


def test_dynamics_mode_critic_update_excludes_actor_objective():
    """Regression for the review finding: in dynamics mode, R^lambda (and hence
    actor_loss) is differentiable w.r.t. critic params through the un-detached value
    bootstrap, so the pre-fix single `torch.autograd.grad(actor_loss + critic_loss,
    all_params)` call let the actor objective's nonzero d(actor_loss)/d(critic_params)
    contaminate the critic's own weight update (~3% spurious norm per review).
    This runs the REAL `imagination_update` (which must split the grad calls) and an
    identically-seeded manual replication of the pre-fix combined-loss update from
    the same snapshotted starting state, and asserts the resulting critic params
    diverge — proving the split materially changes (fixes) what the critic learns,
    not just an inert refactor. Also checks the always-true invariant that
    critic_loss never reaches actor params.
    """
    cfg = dataclasses.replace(CFG, actor_grad="dynamics")
    real_critic, naive_critic, critic_to_actor = _run_mode_comparison(cfg, seed=9)
    assert all(g is None or torch.all(g == 0) for g in critic_to_actor)
    assert any(
        not torch.allclose(real_critic[k], naive_critic[k], atol=1e-6)
        for k in ActorCriticSlab.CRITIC_KEYS
    )


def test_prototype_snapshot_roundtrip():
    slab = _slab()
    s = slab.acquire_slot(None)
    snap = slab.snapshot_slot(s)
    s2 = slab.acquire_slot(snap)
    assert torch.equal(slab.params["a_w1"][s], slab.params["a_w1"][s2])


def test_act_single_params_override_matches_self_params():
    """GPU-pilot blocker regression: SharedLearner.act() must be able to draw an
    identical action whether it reads self.params directly (train_device == cpu)
    or an equivalent CPU actor-mirror dict (train_device != cpu)."""
    slab = _slab()
    s0 = slab.acquire_slot(None)
    s_g = torch.randn(1, 256 + 1024 + 4, generator=make_generator(2, "x"))
    a_direct, _ = slab.act_single(s0, s_g, make_generator(3, ("a", 1)))
    override = slab.actor_snapshot_cpu()
    a_override, _ = slab.act_single(s0, s_g, make_generator(3, ("a", 1)), params_override=override)
    assert torch.equal(a_direct, a_override)


def test_reinforce_score_term_matches_analytic_score_and_pins_cancellation_bug():
    """CRITICAL 1 regression (final review): the reinforce-mode advantage-weighted
    term must evaluate `_log_prob` at a DETACHED sample (`pre.detach()`,
    `a.detach()`) so its mu-gradient equals the textbook Gaussian score
    (pre-mu)/sigma^2 — the score-function estimator's defining property (a
    positive advantage must increase the sampled action's probability).

    Before the fix, the non-detached, reparameterized `pre = mu + eps*sigma` was
    fed into `_log_prob` instead. Since d(pre-mu)/dmu == 0 identically for that
    `pre`, autograd's total derivative silently cancelled the intended score
    term, leaving only a small, wrong-direction leftover through the tanh
    log-det-Jacobian (probe from the review: bandit true grad -0.893 vs. the
    buggy estimator's +0.218 — opposite sign).

    This test bypasses the world model entirely and probes `_log_prob`'s
    gradient math directly, per the reviewer's simplest-robust-design note.
    """
    gen = make_generator(0, "score-term-probe")
    mu = torch.zeros(64, 1, requires_grad=True)
    logstd = torch.zeros(64, 1)  # sigma == 1 keeps the analytic formula simple
    eps = torch.randn(64, 1, generator=gen)
    pre = mu + eps * logstd.exp()
    a = torch.tanh(pre)
    adv = torch.ones(64, 1)  # fixed positive advantage

    # --- fixed formulation: density at a CONSTANT sample (the review fix) ---
    logp_score = ActorCriticSlab._log_prob(mu, logstd, pre.detach(), a.detach())
    loss_fixed = -(adv.detach() * logp_score).mean()
    (grad_fixed,) = torch.autograd.grad(loss_fixed, mu)

    # Analytic Gaussian score d(logp)/d(mu) with pre held constant: (pre-mu)/sigma^2.
    analytic_score = (pre.detach() - mu.detach()) / logstd.exp() ** 2
    expected = -(adv * analytic_score) / mu.shape[0]
    assert torch.allclose(grad_fixed, expected, atol=1e-6)
    # Score-function property: positive advantage → the loss gradient DESCENT
    # step (-grad_fixed) moves mu toward the sampled `pre` (increases its prob).
    assert torch.all((-grad_fixed).sign() == (pre.detach() - mu.detach()).sign())

    # --- OLD (buggy) formulation: density at the reparameterized, non-detached pre ---
    logp_old = ActorCriticSlab._log_prob(mu, logstd, pre, a)
    loss_old = -(adv.detach() * logp_old).mean()
    (grad_old,) = torch.autograd.grad(loss_old, mu)

    # Pin the cancellation: the old formulation's mu-gradient is materially
    # different from (not just a rescaling of) the correct score term above.
    assert not torch.allclose(grad_old, grad_fixed, atol=1e-3)


def test_actor_mirror_slot_refresh_plumbing():
    """GPU-pilot blocker regression: actor_snapshot_cpu()/refresh_actor_mirror_slot
    are the mechanism SharedLearner uses to keep a CPU act-path mirror of a
    train_device slab in sync — verify the mirror starts stale for a newly
    acquired slot and becomes exact after a per-slot refresh, without disturbing
    other slots' mirrored rows."""
    slab = _slab()
    s0 = slab.acquire_slot(None)
    mirror = slab.actor_snapshot_cpu()
    before_s0 = mirror["a_w1"][s0].clone()
    s1 = slab.acquire_slot(None)  # acquired after the snapshot — mirror row stale
    assert not torch.equal(mirror["a_w1"][s1], slab.params["a_w1"][s1].cpu())
    slab.refresh_actor_mirror_slot(mirror, s1)
    for k in ActorCriticSlab.ACTOR_KEYS:
        assert torch.equal(mirror[k][s1], slab.params[k][s1].cpu())
    assert torch.equal(mirror["a_w1"][s0], before_s0)  # untouched slot unaffected
