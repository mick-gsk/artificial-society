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
