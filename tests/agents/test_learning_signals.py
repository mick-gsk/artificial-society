"""Audit group 4: learning-signal correctness.

- Novelty buffer got every observation twice per tick (halved capacity,
  kNN distances biased toward 0).
- Novelty score knn/(knn+1e-3) saturated at ~1 — no signal.
- intrinsic_reward contained 0.2*|pred_reward|, a magnitude prior, not an
  error term.
- PPO bootstrap paired next_obs with the PRE-step hidden.
"""

from __future__ import annotations

import torch

from artificial_society.agents.brain import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    ROLLOUT_HORIZON,
    Brain,
)
from artificial_society.agents.knowledge import NoveltyMemory
from artificial_society.simulation import Simulation


def _fill_rollout(b: Brain, with_next_hidden: bool = True) -> None:
    h = b.initial_hidden()
    for _ in range(ROLLOUT_HORIZON):
        step = b.act([0.05] * INPUT_SIZE, h, use_planning=False)
        b.store_transition(
            step["obs_tensor"],
            step["hidden_in"],
            step["action_tensor"],
            step["log_prob"],
            step["value"],
            reward=0.1,
            done=False,
            next_obs=[0.06] * INPUT_SIZE,
            next_hidden=step["next_hidden"] if with_next_hidden else None,
        )
        h = step["next_hidden"]


# --- fix 6: exactly one novelty registration per observation -----------------
def test_intrinsic_reward_inserts_exactly_once():
    torch.manual_seed(0)
    b = Brain()
    before = len(b.episodic_memory.buffer)
    b.intrinsic_reward(torch.zeros(1, HIDDEN_SIZE), torch.zeros(1, 7), [0.1] * INPUT_SIZE)
    assert len(b.episodic_memory.buffer) == before + 1


def test_agent_update_registers_one_observation_per_tick():
    sim = Simulation(
        headless=True,
        seed=29,
        grid_w=15,
        grid_h=12,
        initial_population=4,
        load_checkpoint=False,
    )
    a = sim.agents[0]
    before = len(a.brain.episodic_memory.buffer)
    a.update(sim.world, sim.agents, tick=1)
    assert len(a.brain.episodic_memory.buffer) == before + 1, (
        "one tick must register exactly one observation (the old double "
        "novelty() call inserted duplicates)"
    )


# --- fix 10: novelty discriminates -------------------------------------------
def test_novelty_discriminates_repetition_from_outliers():
    mem = NoveltyMemory(capacity=100, k=5)
    for _ in range(mem.k):
        mem.novelty(torch.zeros(8))  # cold start
    repeated = [mem.novelty(torch.zeros(8)) for _ in range(50)]
    outlier = mem.novelty(torch.full((8,), 10.0))
    # The old formula returned ~0.99 for both cases.
    assert repeated[-1] < 0.4, f"repetition must score low, got {repeated[-1]}"
    assert outlier > 0.5, f"an outlier must score high, got {outlier}"


def test_setstate_backfills_ema_attrs():
    mem = NoveltyMemory(capacity=16, k=2)
    for i in range(4):
        mem.novelty(torch.full((4,), float(i)))
    state = mem.__getstate__()
    state.pop("_dist_ema", None)  # emulate a pre-fix checkpoint
    state.pop("_ema_decay", None)
    old = NoveltyMemory.__new__(NoveltyMemory)
    old.__setstate__(state)
    assert old._dist_ema is None
    assert old._ema_decay == 0.995
    old.novelty(torch.ones(4))  # must not raise


# --- fix 11: intrinsic reward is independent of the reward head --------------
def test_intrinsic_reward_ignores_reward_head_magnitude():
    torch.manual_seed(0)
    b1 = Brain()
    torch.manual_seed(0)
    b2 = Brain()
    with torch.no_grad():
        b2.reward_head.weight.mul_(100.0)
        b2.reward_head.bias.add_(50.0)
    h = torch.zeros(1, HIDDEN_SIZE)
    act_t = torch.zeros(1, 7)
    obs = [0.2] * INPUT_SIZE
    r1 = b1.intrinsic_reward(h, act_t, obs)
    r2 = b2.intrinsic_reward(h, act_t, obs)
    assert abs(r1 - r2) < 1e-6, (
        "curiosity must measure observation prediction error, not the "
        "magnitude of the predicted reward"
    )


# --- fix 12: bootstrap uses the post-step hidden ------------------------------
def test_maybe_train_bootstraps_with_post_step_hidden(monkeypatch):
    torch.manual_seed(0)
    b = Brain()
    _fill_rollout(b)
    stored_next_hidden = torch.stack([item["next_hidden"] for item in b.rollout.storage])

    captured = {}
    orig_forward = Brain.forward

    def spy(self, obs_t, hid_t):
        # The first forward inside maybe_train is the bootstrap call.
        if "hidden" not in captured:
            captured["hidden"] = hid_t.detach().clone()
        return orig_forward(self, obs_t, hid_t)

    monkeypatch.setattr(Brain, "forward", spy)
    assert b.maybe_train() is not None
    assert torch.allclose(captured["hidden"], stored_next_hidden), (
        "bootstrap must pair next_obs with the stored post-step hidden"
    )


def test_maybe_train_falls_back_for_legacy_entries():
    torch.manual_seed(0)
    b = Brain()
    _fill_rollout(b, with_next_hidden=False)  # legacy callers stored None
    b.rollout.storage[0].pop("next_hidden")  # even older pickles lack the key
    loss = b.maybe_train()
    assert loss is not None and abs(loss) < 1e6
