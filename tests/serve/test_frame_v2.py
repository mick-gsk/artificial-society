"""Tests for Frame-Schema v2 — compact behaviour fields + on-demand detail.

CPU, deterministic. Complements ``test_frame.py`` (v1 shape). Everything here
must stay read-only: ``build_frame`` never mutates the sim, so a double call
without an intervening ``sim.step()`` returns identical dicts.
"""

from __future__ import annotations

import json

from artificial_society.serve.frame import (
    NEED_COLD,
    NEED_HUNGER,
    NEED_NONE,
    NEED_THIRST,
    NEED_TIRED,
    _agent_flags,
    _dominant_need,
    _goal_code,
    behavior_legend,
    build_frame,
)
from artificial_society.simulation import Simulation
from artificial_society.systems.goal_stack import GoalStack, SubGoal


def _sim(ticks: int = 5, pop: int = 6) -> Simulation:
    sim = Simulation(
        headless=True,
        seed=1,
        grid_w=12,
        grid_h=8,
        initial_population=pop,
        load_checkpoint=False,
    )
    for _ in range(ticks):
        sim.step()
    sim.stats.update(sim.tick, sim.agents, sim.world, sim.tribes, sim.technology)
    return sim


# ---------------------------------------------------------------------------
# _dominant_need
# ---------------------------------------------------------------------------
def test_dominant_need_hunger():
    sim = _sim()
    a = sim.agents[0]
    world = sim.world
    from artificial_society.agents.agent import MAX_ENERGY

    # Drain energy hard, everything else comfortable.
    a.energy = 0.05 * MAX_ENERGY
    a.hydration = 100.0
    a.disease_id = None
    a.tool = "sharp_stone"  # has a tool -> no tool drive
    a.endocrine.h[2] = 0.0  # no melatonin -> not tired
    a.genes["curiosity"] = 0.5
    # Put the agent somewhere warm so cold drive stays below hunger.
    world.F["temperature"][a.y, a.x] = 25.0
    assert _dominant_need(a, world) == NEED_HUNGER


def test_dominant_need_none_when_all_low():
    sim = _sim()
    a = sim.agents[0]
    world = sim.world
    from artificial_society.agents.agent import MAX_ENERGY

    a.energy = MAX_ENERGY  # full
    a.hydration = 100.0
    a.disease_id = None
    a.tool = "sharp_stone"
    a.endocrine.h[2] = 0.0
    a.genes["curiosity"] = 0.5
    world.F["temperature"][a.y, a.x] = 25.0  # warm -> no cold
    assert _dominant_need(a, world) == NEED_NONE


def test_dominant_need_thirst():
    sim = _sim()
    a = sim.agents[0]
    world = sim.world
    from artificial_society.agents.agent import MAX_ENERGY

    a.energy = MAX_ENERGY
    a.hydration = 5.0  # very thirsty
    a.disease_id = None
    a.tool = "sharp_stone"
    a.endocrine.h[2] = 0.0
    world.F["temperature"][a.y, a.x] = 25.0
    assert _dominant_need(a, world) == NEED_THIRST


def test_dominant_need_tired():
    sim = _sim()
    a = sim.agents[0]
    world = sim.world
    from artificial_society.agents.agent import MAX_ENERGY

    a.energy = MAX_ENERGY
    a.hydration = 100.0
    a.disease_id = None
    a.tool = "sharp_stone"
    a.endocrine.h[2] = 0.9  # high melatonin -> tired dominates
    world.F["temperature"][a.y, a.x] = 25.0
    assert _dominant_need(a, world) == NEED_TIRED


def test_dominant_need_cold():
    sim = _sim()
    a = sim.agents[0]
    world = sim.world
    from artificial_society.agents.agent import MAX_ENERGY

    a.energy = MAX_ENERGY
    a.hydration = 100.0
    a.disease_id = None
    a.tool = "sharp_stone"
    a.endocrine.h[2] = 0.0
    world.F["temperature"][a.y, a.x] = -30.0  # freezing -> cold drive = 1.0
    assert _dominant_need(a, world) == NEED_COLD


# ---------------------------------------------------------------------------
# goal code parsing
# ---------------------------------------------------------------------------
def test_goal_code_parsing():
    # PROP_DIMS order: flammable(1) hardness(2) edibility(3) ... sharpness(9) ...
    assert _goal_code("need_flammable") == 1
    assert _goal_code("need_edibility") == 3
    assert _goal_code("need_sharpness") == 9
    assert _goal_code("need_scent") == 12
    assert _goal_code("need_bogus") == 0
    assert _goal_code("") == 0
    assert _goal_code("something_else") == 0


def test_goal_fields_in_frame_when_pushed():
    sim = _sim()
    a = sim.agents[0]
    a.goal_stack = GoalStack()
    a.goal_stack.push(
        SubGoal(
            action="strike",
            target_x=7,
            target_y=3,
            max_ticks=20,
            ticks_spent=4,
            label="need_sharpness",
        )
    )
    f = build_frame(sim)
    ad = next(x for x in f["agents"] if x["id"] == a.id)
    assert ad["gl"] == 9  # sharpness index+1
    assert ad["gp"] == 4
    assert ad["gm"] == 20
    assert ad["gx"] == 7
    assert ad["gy"] == 3


def test_goal_fields_absent_without_goal():
    sim = _sim()
    a = sim.agents[0]
    a.goal_stack = GoalStack()  # empty
    f = build_frame(sim)
    ad = next(x for x in f["agents"] if x["id"] == a.id)
    for k in ("gl", "gp", "gm", "gx", "gy"):
        assert k not in ad


# ---------------------------------------------------------------------------
# flag bitfield
# ---------------------------------------------------------------------------
def test_flag_sick_bit():
    sim = _sim()
    a = sim.agents[0]
    a.disease_id = None
    a.pregnant = False
    a.message_vector = [0.0, 0.0, 0.0, 0.0]
    a.goal_stack = GoalStack()
    assert _agent_flags(a) == 0
    a.disease_id = "flu"
    assert _agent_flags(a) & 1


def test_flag_pregnant_bit():
    sim = _sim()
    a = sim.agents[0]
    a.disease_id = None
    a.pregnant = True
    a.message_vector = [0.0, 0.0, 0.0, 0.0]
    a.goal_stack = GoalStack()
    assert _agent_flags(a) == 2


def test_flag_communicating_bit():
    sim = _sim()
    a = sim.agents[0]
    a.disease_id = None
    a.pregnant = False
    a.goal_stack = GoalStack()
    a.message_vector = [0.0, 0.2, 0.0, 0.0]  # > 0.15 threshold
    assert _agent_flags(a) & 4
    a.message_vector = [0.1, 0.1, 0.1, 0.1]  # all below threshold
    assert not (_agent_flags(a) & 4)


def test_flag_goal_depth_bit():
    sim = _sim()
    a = sim.agents[0]
    a.disease_id = None
    a.pregnant = False
    a.message_vector = [0.0, 0.0, 0.0, 0.0]
    a.goal_stack = GoalStack()
    a.goal_stack.push(SubGoal(action="carry", label="need_scent"))
    assert not (_agent_flags(a) & 8)  # depth 1
    a.goal_stack.push(SubGoal(action="strike", label="need_sharpness"))
    assert _agent_flags(a) & 8  # depth 2


# ---------------------------------------------------------------------------
# detail blob + on-demand keying
# ---------------------------------------------------------------------------
def test_no_detail_key_without_inspect_ids():
    f = build_frame(_sim())
    assert "detail" not in f


def test_detail_present_and_json_serializable():
    sim = _sim()
    aid = sim.agents[0].id
    f = build_frame(sim, {aid})
    assert "detail" in f
    assert str(aid) in f["detail"]
    d = f["detail"][str(aid)]
    # required always-present sections
    for k in ("id", "age", "gen", "kids", "hyd", "needs", "horm", "rew", "learn"):
        assert k in d, k
    assert d["id"] == aid
    assert len(d["horm"]) == 8
    assert set(d["needs"]) == {
        "hunger",
        "durst",
        "kaelte",
        "krank",
        "werkzeug",
        "forschung",
        "muede",
    }
    # the WHOLE frame must round-trip through json (no numpy scalars leaking)
    json.dumps(f)


def test_detail_dead_agent_excluded():
    sim = _sim()
    # id that does not exist
    f = build_frame(sim, {999999})
    assert "detail" not in f  # no living inspected agent -> key omitted


def test_detail_read_only_double_call_identical():
    sim = _sim()
    aid = sim.agents[0].id
    f1 = build_frame(sim, {aid})
    f2 = build_frame(sim, {aid})
    assert f1 == f2  # no mutation between the two calls


def test_detail_deterministic_top_sorts():
    """Trust / ToM / inventory top-N must sort deterministically (value, then id)."""
    sim = _sim()
    a = sim.agents[0]
    # Two peers with identical trust magnitude -> id must break the tie stably.
    a.trust = {50: 0.5, 20: 0.5, 30: -0.9}
    a.material_inventory = {"wood": 1.0, "stone": 1.0, "flint": 5.0}
    f = build_frame(sim, {a.id})
    d = f["detail"][str(a.id)]
    # strongest |trust| first (0.9), then the 0.5 ties ordered by id asc (20,50)
    assert d["trust"][0] == [30, -0.9]
    assert [t[0] for t in d["trust"][1:]] == [20, 50]
    # inventory: highest qty first, ties by id (name) asc
    assert list(d["inv"].keys())[0] == "flint"
    json.dumps(f)


# ---------------------------------------------------------------------------
# determinism: build_frame (incl. detail) never perturbs the sim
# ---------------------------------------------------------------------------
def _run_history(with_frames: bool, ticks: int = 300) -> list:
    """Run a seeded sim, optionally calling build_frame(sim, {id}) each tick.

    Returns the full stats history tuple so two runs can be compared exactly.
    """
    sim = Simulation(
        headless=True,
        seed=4321,
        grid_w=16,
        grid_h=12,
        initial_population=10,
        load_checkpoint=False,
    )
    inspect = {sim.agents[0].id} if sim.agents else set()
    for _ in range(ticks):
        sim.step()
        sim.stats.update(sim.tick, sim.agents, sim.world, sim.tribes, sim.technology)
        if with_frames:
            # exercise both the compact path and the full detail blob
            build_frame(sim, inspect)
    return (
        list(sim.stats.population_history),
        list(sim.stats.energy_history),
        list(sim.stats.food_history),
        list(sim.stats.cooperation_history),
        list(sim.stats.knowledge_history),
    )


def test_build_frame_does_not_perturb_determinism():
    baseline = _run_history(with_frames=False)
    with_frames = _run_history(with_frames=True)
    assert baseline == with_frames, (
        "build_frame (with detail) altered the simulation trajectory — "
        "it must be a pure read-only snapshot"
    )


# ---------------------------------------------------------------------------
# hello legend
# ---------------------------------------------------------------------------
def test_behavior_legend_shape():
    leg = behavior_legend()
    assert leg["needs"][0] == "none"
    assert len(leg["needs"]) == 8
    assert len(leg["goal_props"]) == 12
    # goal code gl maps 1-based into goal_props
    assert _goal_code("need_" + leg["goal_props"][0]) == 1
    assert _goal_code("need_" + leg["goal_props"][11]) == 12
    assert leg["acts"][:5] == ["idle", "forage", "cooperate", "attack", "build"]
    assert leg["acts"][5] == "sleep"
    # JSON-native
    json.dumps(leg)
