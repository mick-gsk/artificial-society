from __future__ import annotations

import argparse
import dataclasses
import importlib.util as _ilu
import json
import pathlib as _pathlib

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.simulation import Simulation

_REPO_ROOT = _pathlib.Path(__file__).resolve().parents[3]
_spec = _ilu.spec_from_file_location("rssm_ab", _REPO_ROOT / "scripts" / "rssm_ab.py")
rssm_ab = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rssm_ab)

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
    policy_mode="mpc",
)


def test_mpc_mode_runs_without_slab():
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    for _ in range(20):
        sim.step()
    assert all(getattr(a, "rssm_slot", None) is None for a in sim.agents)
    assert len(sim.agents) > 0


def test_rmst_uncensored_and_censored():
    # all die before tau → plain mean of min(life, tau)
    assert abs(rssm_ab.rmst([10, 20, 30], [False] * 3, tau=25) - (10 + 20 + 25) / 3) < 1e-9
    # censoring: KM keeps survival at 1.0 past a censored obs with no later deaths
    v = rssm_ab.rmst([10, 15], [False, True], tau=20)
    assert v > rssm_ab.rmst([10, 15], [False, False], tau=20)  # censored ≥ death-observed


def test_wilcoxon_exact_signs():
    p = rssm_ab.wilcoxon_exact([1.0, 2.0, 3.0, 4.0, 5.0])  # all positive deltas
    assert p == min(1.0, 2 * (1 / 2**5) * 1)  # most extreme rank sum


def test_cfg_override_reflected_in_constructed_sim():
    sim = rssm_ab._arm_sim(
        "B", seed=1, cpu=True, cfg_overrides={"train_every": 4, **_fast_overrides()}
    )
    assert sim.rssm_learner.cfg.train_every == 4
    assert sim.rssm_learner.cfg.train_device == "cpu"
    assert sim.rssm_learner.cfg.policy_mode == "actor"


def _fast_overrides():
    # Keep the smoke test cheap: same knobs as module-level _FAST, minus the
    # fields _arm_sim itself pins (train_device, policy_mode) and minus
    # max_slots — _arm_sim hardcodes initial_population=36, so a small slab
    # (the module-level _FAST test uses initial_population=8) would raise on
    # acquire_slot exhaustion; leave max_slots at its roomy default (256).
    return dict(
        prefill_transitions=64,
        wm_warmup_updates=1,
        batch_size=4,
        seq_len=16,
        horizon=4,
    )


def test_cfg_override_unknown_field_raises():
    import pytest

    with pytest.raises(ValueError):
        rssm_ab.parse_cfg_overrides(["not_a_real_field=1"])


def test_cfg_override_wrong_arm_raises():
    import pytest

    with pytest.raises(ValueError):
        rssm_ab._arm_sim("A", seed=1, cpu=True, cfg_overrides={"train_every": 4})


def test_v1_const_setattr_takes_effect():
    import artificial_society.agents.brain as brain_mod

    orig = brain_mod.NOVELTY_WEIGHT
    try:
        v1_consts = rssm_ab.parse_v1_consts(["NOVELTY_WEIGHT=0.05", "PPO_EPOCHS=40"])
        assert v1_consts == {"NOVELTY_WEIGHT": 0.05, "PPO_EPOCHS": 40}
        sim = rssm_ab._arm_sim("A", seed=1, cpu=True, v1_consts=v1_consts)
        assert brain_mod.NOVELTY_WEIGHT == 0.05
        assert brain_mod.PPO_EPOCHS == 40
        assert len(sim.agents) > 0
    finally:
        brain_mod.NOVELTY_WEIGHT = orig
        brain_mod.PPO_EPOCHS = 20


def test_v1_const_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError):
        rssm_ab.parse_v1_consts(["NOT_A_REAL_CONSTANT=1"])


def test_v1_const_wrong_arm_raises():
    import pytest

    with pytest.raises(ValueError):
        rssm_ab._arm_sim("B", seed=1, cpu=True, v1_consts={"NOVELTY_WEIGHT": 0.05})


def test_tag_round_trip_in_analyze(tmp_path, capsys):
    args = argparse.Namespace(
        arm="B",
        seed=1,
        ticks=30,
        out=str(tmp_path),
        cpu=True,
        tag="k4test",
        cfg=["train_every=4", *[f"{k}={v}" for k, v in _fast_overrides().items()]],
        v1_const=[],
    )
    rssm_ab.cmd_run(args)
    events_path = tmp_path / "B_k4test_s1.events.jsonl"
    summary_path = tmp_path / "B_k4test_s1.summary.json"
    assert events_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["tag"] == "k4test"
    assert summary["cfg_overrides"]["train_every"] == 4

    # A 30-tick run is too short for real births (RMST's cohort requires
    # origin=="birth", spec §10.2) — append one synthetic birth+death so the
    # analyze pass below has something to compute an RMST over. This exercises
    # cmd_analyze's arm-key parsing (the actual thing under test here), not
    # the demography model.
    with events_path.open("a") as ev:
        ev.write(json.dumps({"e": "birth", "id": -999, "t": 0, "origin": "birth"}) + "\n")
        ev.write(json.dumps({"e": "death", "id": -999, "t": 5}) + "\n")

    analyze_args = argparse.Namespace(dir=str(tmp_path), tau=10, transient=0)
    rssm_ab.cmd_analyze(analyze_args)
    out = capsys.readouterr().out
    assert "B_k4test" in out
