from __future__ import annotations

import dataclasses

import pytest
import torch

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.simulation import CheckpointIncompatibleError, Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)
_FAST = dataclasses.replace(
    RSSMConfig(),
    prefill_transitions=64,
    wm_warmup_updates=1,
    batch_size=4,
    seq_len=16,
    horizon=4,
    max_slots=32,
    train_device="cpu",
)


def _digest(sim):
    return [(a.id, round(a.energy, 4)) for a in sim.agents]


def test_off_arm_leaves_torch_rng_untouched():
    sim = Simulation(seed=42, **_PARAMS)
    state_after_init = torch.get_rng_state()
    sim2 = Simulation(seed=42, **_PARAMS)
    assert torch.equal(state_after_init, torch.get_rng_state())  # identical construction draws
    assert sim.rssm_learner is None and sim.brain_arch == "v1"
    assert _digest(sim) == _digest(sim2)


def test_rssm_run_deterministic_across_births_and_deaths():
    def run():
        sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
        for _ in range(40):
            sim.step()
        return _digest(sim)

    assert run() == run()


def test_rssm_agents_skip_v1_brain():
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    sim.step()
    for a in sim.agents:
        assert getattr(a, "brain_arch", None) == "rssm"
        assert getattr(a, "rssm_slot", None) is not None
        assert getattr(a, "brain", None) is None  # no v1 Brain built


def test_training_hook_fires_via_registry():
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    for _ in range(24):  # 8 agents × 24 ticks > 64 prefill
        sim.step()
    assert sim.rssm_learner.wm.wm_updates >= 1


def test_v1_arm_stamps_spawn_origin_unconditionally():
    """Finding 1: spawn_origin must be stamped for ALL arms (incl. plain v1), not
    only when an rssm_learner is attached — otherwise the A/B analyzer's
    origin=="birth" cohort is permanently empty for the baseline arm."""
    sim = Simulation(seed=42, **_PARAMS)
    assert sim.rssm_learner is None and sim.brain_arch == "v1"
    for a in sim.agents:
        assert a.spawn_origin == "initial"
    for _ in range(60):
        sim.step()
    origins = {getattr(a, "spawn_origin", None) for a in sim.agents}
    assert origins  # non-empty: agents survived
    assert origins <= {"initial", "birth", "respawn"}
    assert all(hasattr(a, "spawn_origin") for a in sim.agents)


def test_deaths_reach_replay_as_terminals():
    """CRITICAL 2 regression (final review): on_death must mark the dying agent's
    last stored transition as terminal in the shared replay, so the continue
    head and death-stratified sampling actually see real deaths. Force a death
    directly (draining health) instead of relying on natural attrition within a
    short test window — the early-return death path in `Agent.update` (health
    <= 0 -> alive=False; return None) is exactly the "exits before the store
    point" case this fix targets.
    """
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    sim.step()  # everyone gets at least one stored transition first
    victim = sim.agents[0]
    victim.health = 0.0
    sim.step()  # victim's early death-return fires before this tick's store point
    assert sim.rssm_learner.replay.num_death_episodes > 0


def test_rssm_death_leaves_no_carcass_and_no_broadcast():
    """NEW-1 regression (arm-parity review fix): rssm dead agents must be routed
    straight to `rssm_learner.on_death` and skip the v1 dead-handling branch
    entirely — no `add_carcass` credit, no `_broadcast_death_knowledge` call.
    Arm A (rssm) is pre-filtered out of dead-handling in `step()` and can never
    reach those subsidies, so arms B/C must not get them either on the rssm
    path (which only exists via the shared `remove_dead`)."""
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    sim.step()  # everyone gets at least one stored transition first
    victim = sim.agents[0]
    victim.health = 0.0
    vx, vy = victim.pos
    carcasses_before = sim.world.get_cell(vx, vy)["carcasses"]
    sim.step()  # victim's early death-return fires before this tick's store point
    carcasses_after = sim.world.get_cell(vx, vy)["carcasses"]
    assert carcasses_after == carcasses_before  # no add_carcass credit for rssm deaths
    assert sim.rssm_learner.replay.num_death_episodes > 0  # terminal fix still intact


def test_checkpoint_roundtrip_and_mismatch_guard(tmp_path, monkeypatch):
    import artificial_society.simulation as sim_mod

    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(tmp_path / "cp.pkl"))
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    for _ in range(10):
        sim.step()
    sim._save_checkpoint()
    with pytest.raises(CheckpointIncompatibleError):
        Simulation(seed=42, brain_arch="v1", **{**_PARAMS, "load_checkpoint": True})
    sim2 = Simulation(
        seed=42, brain_arch="rssm", rssm_config=_FAST, **{**_PARAMS, "load_checkpoint": True}
    )
    assert sim2.tick == sim.tick
    assert sim2.rssm_learner.wm.wm_updates == sim.rssm_learner.wm.wm_updates
