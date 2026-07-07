"""Emergence-metrics observer (systems/emergence_metrics.py, registry order=68).

Design doc: docs/superpowers/specs/2026-07-07-emergence-metrics-design.md. This
system is a passive, read-only observer auto-imported into every sim via
``systems.registry.discover()`` — the golden-safety gate is the FULL suite
(``tests/test_regression_golden.py``), not this file; this file locks the
module's own counting/sampling/caps contracts plus a direct RNG-safety check.
"""

from __future__ import annotations

import json
import random
import types

import numpy as np
import pytest
import torch

from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.props import IDX2
from artificial_society.simulation import Simulation
from artificial_society.systems import emergence_metrics, registry
from artificial_society.systems.emergence_metrics import _tick
from artificial_society.systems.language import TOKEN_WORLD

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _fresh(**kw):
    params = dict(_PARAMS, seed=42)
    params.update(kw)
    return Simulation(**params)


# --- 1. Registration -------------------------------------------------------


def test_registers_at_order_68_before_stats():
    sim = _fresh()
    assert hasattr(sim, "emergence_metrics")
    assert "emergence_metrics" in sim.systems

    orders = {s.name: s.order for s in registry.specs()}
    assert orders["emergence_metrics"] == 68
    assert orders["emergence_metrics"] < orders["stats"]
    assert orders["technology"] < orders["emergence_metrics"]


def test_sampled_tick_draws_no_rng():
    sim = _fresh()
    torch_state_before = torch.get_rng_state().clone()
    random_state_before = random.getstate()

    _tick(sim, emergence_metrics.SAMPLE_EVERY)  # a sample point (0 % SAMPLE_EVERY == 0 too)

    assert torch.equal(torch.get_rng_state(), torch_state_before)
    assert random.getstate() == random_state_before


def test_non_sample_tick_is_a_total_noop():
    sim = _fresh()
    before = dict(sim.emergence_metrics.last)
    _tick(sim, emergence_metrics.SAMPLE_EVERY + 1)  # not a multiple of SAMPLE_EVERY
    assert sim.emergence_metrics.last == before


# --- 2. Synthetic counting ---------------------------------------------------


def _fake_agent(alive=True, causal_memory=None, token_memory=None, agent_id=0):
    return types.SimpleNamespace(
        id=agent_id,
        alive=alive,
        hands=None,
        causal_memory=causal_memory,
        token_memory=token_memory,
    )


def test_culture_spread_and_shared_seqs_exact():
    seq_a = ("strike", "granite", "")
    seq_b = ("cut", "wood", "fiber")

    agents = [
        _fake_agent(
            agent_id=i,
            causal_memory=types.SimpleNamespace(sequences=seqs),
        )
        for i, seqs in enumerate(
            [
                {seq_a: {}, seq_b: {}},  # agent 0 knows both
                {seq_a: {}},  # agent 1 knows only A
                {seq_a: {}},  # agent 2 knows only A
            ]
        )
    ]
    sim = types.SimpleNamespace(agents=agents, world=None, technology=None)

    em = emergence_metrics.EmergenceMetrics()
    em.sample(sim, 0)

    assert em.last["culture_distinct_seqs"] == 2.0
    assert em.last["culture_max_spread"] == 3.0  # seq_a known by all 3
    assert em.last["culture_shared_seqs"] == 1.0  # only seq_a has spread >= 3
    assert em.last["novel_causal_keys"] == 2.0  # first sample: both keys are new


def test_language_active_share_and_convergence_exact():
    ctx = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def make_tm(assoc, contexts):
        return types.SimpleNamespace(
            associations=assoc,
            mean_context=lambda token_id, contexts=contexts: contexts.get(token_id),
        )

    # T1: used by agents 0 and 1 with identical context -> converged (share 0.5, cos 1.0).
    # T2: used by agent 2 only -> not "active" (needs >= 2 qualifying users).
    tm0 = make_tm({"T1": types.SimpleNamespace(use_count=3)}, {"T1": ctx})
    tm1 = make_tm({"T1": types.SimpleNamespace(use_count=5)}, {"T1": ctx})
    tm2 = make_tm({"T2": types.SimpleNamespace(use_count=10)}, {"T2": ctx})
    tm3 = make_tm({}, {})

    agents = [
        _fake_agent(agent_id=0, token_memory=tm0),
        _fake_agent(agent_id=1, token_memory=tm1),
        _fake_agent(agent_id=2, token_memory=tm2),
        _fake_agent(agent_id=3, token_memory=tm3),
    ]
    sim = types.SimpleNamespace(agents=agents, world=None, technology=None)

    world_log_len_before = len(TOKEN_WORLD.world_log)

    em = emergence_metrics.EmergenceMetrics()
    em.sample(sim, 0)

    assert em.last["lang_tokens_active"] == 1.0  # only T1 has >= 2 qualifying users
    assert em.last["lang_max_share"] == 0.5  # T1: 2/4 agents
    assert em.last["lang_converged"] == 1.0  # T1 shares >= 0.5 and cos sim 1.0 > 0.7

    # No-emit guarantee: TokenWorld.check_convergence is never called.
    assert len(TOKEN_WORLD.world_log) == world_log_len_before


# --- 3. v2 path ---------------------------------------------------------


def test_tool_holders_counts_agent_with_sharp_held_object():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    obj = make_object("granite", 1.0)
    obj.props[IDX2["hardness"]] = 1.0
    obj.props[IDX2["sharpness"]] = 1.0  # effective_sharpness == 1.0 > 0.3
    agent.hands.held.append(obj)

    sim.emergence_metrics.sample(sim, 0)

    assert sim.emergence_metrics.last["tool_holders"] == 1.0


def test_verb_counter_deltas_and_success_rate_hand_built_metrics():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    em = sim.emergence_metrics
    metrics = sim.world.objects.metrics

    metrics["verbs_fired"] = {"strike": 5, "cut": 2}
    metrics["verbs_failed"] = {"strike": 1}
    em.sample(sim, 0)

    assert em.last["verb_fired_strike"] == 5.0
    assert em.last["verb_fired_cut"] == 2.0
    assert em.last["verb_success_rate"] == (7 - 1) / 7

    metrics["verbs_fired"]["strike"] = 8  # +3
    metrics["verbs_failed"]["strike"] = 3  # +2, cut unchanged (+0)
    em.sample(sim, 25)

    assert em.last["verb_fired_strike"] == 3.0
    assert "verb_fired_cut" not in em.series or em.series["verb_fired_cut"][-1][0] != 25
    assert em.last["verb_success_rate"] == (3 - 2) / 3


def test_behavior_entropy_from_verb_histogram():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    em = sim.emergence_metrics
    metrics = sim.world.objects.metrics

    # Uniform 2-way split -> entropy == log(2).
    metrics["verbs_fired"] = {"strike": 10, "cut": 10}
    em.sample(sim, 0)

    assert em.last["behavior_entropy"] == pytest.approx(np.log(2))


def test_behavior_entropy_absent_when_no_verbs_fired():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    sim.emergence_metrics.sample(sim, 0)
    assert "behavior_entropy" not in sim.emergence_metrics.last


def test_tool_keys_absent_on_v1_arm():
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    sim.emergence_metrics.sample(sim, 0)
    last = sim.emergence_metrics.last
    assert "tool_holders" not in last
    assert "inventions" not in last


# --- 4. Sampling interval + caps ---------------------------------------------


def test_series_grows_only_at_sample_multiples(monkeypatch):
    monkeypatch.setattr(emergence_metrics, "SAMPLE_EVERY", 5)
    sim = _fresh()
    for t in range(20):  # ticks 0, 5, 10, 15 are multiples of 5 -> 4 samples
        _tick(sim, t)
    assert len(sim.emergence_metrics.series["tech_capabilities"]) == 4


def test_series_and_samples_are_capped(monkeypatch):
    monkeypatch.setattr(emergence_metrics, "SERIES_CAP", 5)
    sim = _fresh()
    em = sim.emergence_metrics
    for t in range(20):
        em.sample(sim, t)
    assert len(em.series["tech_capabilities"]) == 5
    assert len(em.samples) == 5
    # Cap keeps the *most recent* samples.
    assert em.series["tech_capabilities"][-1][0] == 19


# --- 5. dump_jsonl round-trip -------------------------------------------


def test_dump_jsonl_round_trip(tmp_path):
    sim = _fresh()
    em = sim.emergence_metrics
    for t in (0, 25, 50):
        em.sample(sim, t)

    out = tmp_path / "emergence.jsonl"
    em.dump_jsonl(str(out))

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert [row["tick"] for row in rows] == [0, 25, 50]
    for row in rows:
        assert "tech_capabilities" in row
