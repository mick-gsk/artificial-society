"""Unit tests for the functional-complexity metric and the recombiner.

These pin the core scientific claim of Stage 0a: functional/irreducible depth
collapses redundant deep chains that raw DAG depth would over-count, and the
knock-out test confirms that recipe inputs functionally matter.
"""

from __future__ import annotations

import random

import pytest

from artificial_society.environment.materials import MATERIALS, N_PROPS, combine_vectors
from artificial_society.research import metrics
from artificial_society.research.recombiner import run_recombiner


def _vec(**dims) -> list[float]:
    v = [0.5] * N_PROPS
    for i, val in dims.items():
        v[i] = val
    return v


def _synthetic_entries():
    """A 3-deep redundant chain (functionally equivalent) plus one distinct branch."""
    # cluster A: structurally depth 1/2/3 but vectors within func_tau of each other
    a1 = [0.5] * N_PROPS
    a2 = [0.5] * N_PROPS
    a2[0] = 0.55  # dist 0.05 from a1
    a3 = [0.5] * N_PROPS
    a3[0] = 0.60  # dist 0.10 from a1  (<0.15 func_tau)
    # distinct branch, far away (> func_tau from cluster A)
    b = [0.0] * N_PROPS
    b[5] = 0.95
    return [
        {"id": "mat_0000", "recipe": ["strike", "stone", "flint"], "vector": a1},  # sd 1
        {"id": "mat_0001", "recipe": ["bundle", "mat_0000", "dry_grass"], "vector": a2},  # sd 2
        {"id": "mat_0002", "recipe": ["bundle", "mat_0001", "dry_grass"], "vector": a3},  # sd 3
        {"id": "mat_0003", "recipe": ["bundle", "mat_0000", "stone"], "vector": b},  # sd 2
    ]


def test_structural_depths():
    sd = metrics.structural_depths(_synthetic_entries())
    assert sd == {"mat_0000": 1, "mat_0001": 2, "mat_0002": 3, "mat_0003": 2}


def test_functional_depth_collapses_redundant_chains():
    entries = _synthetic_entries()
    fd = metrics.functional_depths(entries, func_tau=0.15)
    # the redundant 2/3-deep artifacts inherit the shallow depth of their equivalent
    assert fd["mat_0000"] == 1
    assert fd["mat_0001"] == 1
    assert fd["mat_0002"] == 1
    # the genuinely distinct branch keeps its own depth
    assert fd["mat_0003"] == 2


def test_analyze_registry_summary():
    out = metrics.analyze_registry(_synthetic_entries(), func_tau=0.15)
    assert out["n_entries"] == 4
    assert out["max_structural_depth"] == 3
    assert out["max_functional_depth"] == 2  # the distinct branch, not the depth-3 chain
    assert out["n_functional_clusters"] == 2


def test_analyze_registry_empty():
    out = metrics.analyze_registry([])
    assert out["n_entries"] == 0
    assert out["max_functional_depth"] == 0


def test_knockout_marks_required_input():
    env = {"moisture": 0.5}
    # 'strike' of two hard materials deterministically yields a sharpened product
    v = combine_vectors(MATERIALS["flint"], MATERIALS["stone"], "strike", env)
    assert v is not None
    entry = {
        "id": "mat_0000",
        "recipe": ["strike", "flint", "stone"],
        "vector": [float(x) for x in v],
    }
    res = metrics.knockout_validate([entry], sample=10, seed=0)
    assert res["tested"] == 1
    assert res["required"] == 1  # removing the second input destroys the effect
    assert res["required_frac"] == 1.0


def test_recombiner_is_reproducible():
    e1, _ = run_recombiner(seed=7, n_attempts=300)
    e2, _ = run_recombiner(seed=7, n_attempts=300)
    assert len(e1) == len(e2)
    assert [x["recipe"] for x in e1] == [x["recipe"] for x in e2]


def test_recombiner_does_not_perturb_global_rng():
    random.seed(123)
    before = [random.random() for _ in range(3)]
    random.seed(123)
    run_recombiner(seed=7, n_attempts=50)
    after = [random.random() for _ in range(3)]
    assert before == after


def test_recombiner_produces_discoveries():
    entries, series = run_recombiner(seed=1, n_attempts=2000)
    assert len(entries) > 0
    assert series[-1]["attempt"] == 2000
    assert series[-1]["n_discoveries"] == len(entries)


# --------------------------------------------------------------------------
# Value-based DVs (Path-A retrofit, Schritt A) — reward/transmission-aware,
# replacing artifact-geometry max depth as the sole primary DV.
# --------------------------------------------------------------------------

# single-task basis on dim 0; base ceiling forced to 0 so any positive vec advances
_T0 = {"t0": (lambda V: V[:, 0])}
_ZERO_BASE = [[0.0] * N_PROPS]


def _byidx(d: dict[int, float]) -> list[float]:
    """Build a 12-dim vector from {index: value} pairs, zeros elsewhere."""
    v = [0.0] * N_PROPS
    for i, val in d.items():
        v[i] = val
    return v


def test_accumulated_useful_depth_counts_only_genuine_advances():
    # m0 (d1, t0=0.3) advances; m1 (d2, t0=0.2) does NOT (below frontier 0.3);
    # m2 (d3, t0=0.5) advances. Churn (m1) earns no depth credit.
    entries = [
        {"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 0.3})},
        {"id": "m1", "recipe": ["bundle", "m0", "stone"], "vector": _byidx({0: 0.2})},
        {"id": "m2", "recipe": ["bundle", "m1", "stone"], "vector": _byidx({0: 0.5})},
    ]
    struct = {"m0": 1, "m1": 2, "m2": 3}
    out = metrics.accumulated_useful_depth(
        entries, struct=struct, task_basis=_T0, base_vectors=_ZERO_BASE
    )
    assert out["n_useful_advances"] == 2  # m0 and m2 only
    assert out["useful_depth_max"] == 3  # deepest genuine advance
    assert out["accumulated_useful_depth"] == 3  # max advance depth for the single task
    assert out["per_task"]["t0"] == 3


def test_accumulated_useful_depth_base_ceiling_blocks_saturated_task():
    # base already reaches t0=0.6, so an artifact at 0.5 cannot advance the frontier
    entries = [
        {"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 0.5})},
    ]
    out = metrics.accumulated_useful_depth(
        entries, struct={"m0": 1}, task_basis=_T0, base_vectors=[[0.6] + [0.0] * (N_PROPS - 1)]
    )
    assert out["n_useful_advances"] == 0
    assert out["accumulated_useful_depth"] == 0


def test_population_functional_value_weights_by_adoption():
    # two functionally distinct artifacts; value scales with `uses`
    entries = [
        {"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 1.0}), "uses": 10},
        {"id": "m1", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({5: 1.0}), "uses": 0},
    ]
    out = metrics.population_functional_value(entries, task_basis=_T0)
    # only m0 has positive t0 utility AND positive uses
    assert out["population_functional_value"] == 10.0
    assert out["weight_source"] == "uses"
    assert out["provisional"] is True


def test_transmitted_frontier_advances_blocked_without_attribution():
    # current export has discovered_by = -1 everywhere -> DV3 is not computable
    entries = [
        {
            "id": "m0",
            "recipe": ["bundle", "stone", "flint"],
            "vector": _byidx({0: 0.9}),
            "uses": 5,
            "discovered_by": -1,
        },
    ]
    out = metrics.transmitted_frontier_advances(
        entries, struct={"m0": 1}, task_basis=_T0, base_vectors=_ZERO_BASE
    )
    assert out["computable"] is False
    assert out["transmitted_frontier_advances"] == 0


def test_transmitted_frontier_advances_counts_when_attributed():
    entries = [
        {
            "id": "m0",
            "recipe": ["bundle", "stone", "flint"],
            "vector": _byidx({0: 0.9}),
            "uses": 5,
            "discovered_by": 3,
        },
    ]
    out = metrics.transmitted_frontier_advances(
        entries, struct={"m0": 1}, task_basis=_T0, base_vectors=_ZERO_BASE, k=2
    )
    assert out["computable"] is True
    assert out["transmitted_frontier_advances"] == 1


# --------------------------------------------------------------------------
# M5(a) — graded_useful_advance: a NON-FLOORED, arm-symmetric, churn-immune
# functional metric. Replaces the per-layer-max DV2 (which empirically pins at
# DV2=2 for the embodied arm in both arms) with the margin-normalised MAGNITUDE
# of frontier advances summed over structural layers, so the score has headroom
# above the floor without becoming hackable by blind volume/churn.
# --------------------------------------------------------------------------


def test_graded_useful_advance_lifts_off_floor():
    # single-task chain advancing 0.3 -> 0.6 -> 0.9 across structural depths 1/2/3.
    # margin-normalised advance per layer = 0.3 / 0.02 = 15  ->  total 45, far above
    # the empirical DV2 floor (=2). This is the M5(a) acceptance gate.
    entries = [
        {"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 0.3})},
        {"id": "m1", "recipe": ["bundle", "m0", "stone"], "vector": _byidx({0: 0.6})},
        {"id": "m2", "recipe": ["bundle", "m1", "stone"], "vector": _byidx({0: 0.9})},
    ]
    struct = {"m0": 1, "m1": 2, "m2": 3}
    out = metrics.graded_useful_advance(
        entries, struct=struct, task_basis=_T0, base_vectors=_ZERO_BASE, margin=0.02
    )
    assert out["graded_useful_advance"] == pytest.approx(45.0)
    assert out["graded_useful_advance"] > 2  # lifts off the empirical DV2 floor
    assert out["per_task"]["t0"] == pytest.approx(45.0)


def test_graded_useful_advance_depth_weight_rewards_deep_advances():
    # Two single-task sets with EQUAL total advance magnitude (0.6) but different
    # structural depth. Unweighted graded is equal (it is depth-blind by construction);
    # depth-weighted graded rewards the deep, cumulative chain ("standing on shoulders").
    shallow = [{"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 0.6})}]
    deep = [
        {"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 0.2})},
        {"id": "m1", "recipe": ["bundle", "m0", "stone"], "vector": _byidx({0: 0.4})},
        {"id": "m2", "recipe": ["bundle", "m1", "stone"], "vector": _byidx({0: 0.6})},
    ]
    kw = dict(task_basis=_T0, base_vectors=_ZERO_BASE, margin=0.02)
    s_unw = metrics.graded_useful_advance(shallow, struct={"m0": 1}, **kw)["graded_useful_advance"]
    d_unw = metrics.graded_useful_advance(
        deep, struct={"m0": 1, "m1": 2, "m2": 3}, **kw
    )["graded_useful_advance"]
    assert s_unw == pytest.approx(d_unw)  # equal total magnitude -> unweighted is blind to depth
    s_w = metrics.graded_useful_advance(
        shallow, struct={"m0": 1}, depth_weight=True, **kw
    )["graded_useful_advance"]
    d_w = metrics.graded_useful_advance(
        deep, struct={"m0": 1, "m1": 2, "m2": 3}, depth_weight=True, **kw
    )["graded_useful_advance"]
    assert d_w > s_w  # depth-weighting discriminates the cumulative (deep) chain
    assert s_w == pytest.approx(30.0)
    assert d_w == pytest.approx(60.0)  # 10*1 + 10*2 + 10*3


def test_graded_useful_advance_is_churn_immune():
    # m0 (d1) advances to 0.5; m1 (d2, 0.3) and m2 (d3, 0.4) sit BELOW the running
    # frontier and earn ZERO graded credit despite being structurally deeper.
    entries = [
        {"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 0.5})},
        {"id": "m1", "recipe": ["bundle", "m0", "stone"], "vector": _byidx({0: 0.3})},
        {"id": "m2", "recipe": ["bundle", "m1", "stone"], "vector": _byidx({0: 0.4})},
    ]
    struct = {"m0": 1, "m1": 2, "m2": 3}
    out = metrics.graded_useful_advance(
        entries, struct=struct, task_basis=_T0, base_vectors=_ZERO_BASE, margin=0.02
    )
    assert out["graded_useful_advance"] == pytest.approx(25.0)  # only m0: 0.5 / 0.02


def test_graded_useful_advance_is_arm_symmetric():
    # identical geometry; differ ONLY in uses/discovered_by (the instrumentation fields
    # the disembodied recombiner structurally lacks). A hack-resistant functional metric
    # must ignore them -> the recombiner cannot be flattered or floored by instrumentation.
    base = [
        {"id": "m0", "recipe": ["bundle", "stone", "flint"], "vector": _byidx({0: 0.4})},
        {"id": "m1", "recipe": ["bundle", "m0", "stone"], "vector": _byidx({0: 0.8})},
    ]
    learned = [{**e, "uses": 10, "discovered_by": 3} for e in base]
    recombiner = [{**e, "uses": 0, "discovered_by": -1} for e in base]
    struct = {"m0": 1, "m1": 2}
    kw = dict(struct=struct, task_basis=_T0, base_vectors=_ZERO_BASE, margin=0.02)
    a = metrics.graded_useful_advance(learned, **kw)["graded_useful_advance"]
    b = metrics.graded_useful_advance(recombiner, **kw)["graded_useful_advance"]
    assert a == b


# --------------------------------------------------------------------------
# E0 / M2 — embodiment-matched recombiner null. Same blind uniform combine
# machinery as run_recombiner, but the budget is split over N agents each with
# their OWN local pool (fragmentation) sharing discoveries only at share_fidelity.
# Isolates the disembodied recombiner's structural advantage (one perfect-memory
# monopolist pool) from learning, to test whether the -9.4 gap is an embodiment
# confound rather than a learning failure.
# --------------------------------------------------------------------------


def _is_discovered(mat) -> bool:
    return isinstance(mat, str) and mat.startswith("mat_")


def test_embodied_recombiner_produces_attributed_discoveries():
    from artificial_society.research.recombiner import run_embodied_recombiner

    entries, series = run_embodied_recombiner(seed=1, n_attempts=2000, n_agents=4)
    assert len(entries) > 0
    assert series[-1]["attempt"] == 2000
    assert series[-1]["n_discoveries"] == len(entries)
    # every discovery is attributed to a real agent in [0, n_agents)
    assert all(0 <= e["discovered_by"] < 4 for e in entries)


def test_embodied_recombiner_is_reproducible():
    from artificial_society.research.recombiner import run_embodied_recombiner

    e1, _ = run_embodied_recombiner(seed=7, n_attempts=600, n_agents=4, share_fidelity=0.5)
    e2, _ = run_embodied_recombiner(seed=7, n_attempts=600, n_agents=4, share_fidelity=0.5)
    assert [x["recipe"] for x in e1] == [x["recipe"] for x in e2]
    assert [x["discovered_by"] for x in e1] == [x["discovered_by"] for x in e2]


def test_embodied_recombiner_isolates_pools_without_sharing():
    # With share_fidelity=0 an agent can only build on base materials and its OWN
    # discoveries -> no recipe input may be a material discovered by another agent.
    from artificial_society.research.recombiner import run_embodied_recombiner

    entries, _ = run_embodied_recombiner(
        seed=2, n_attempts=1500, n_agents=3, share_fidelity=0.0
    )
    owner = {e["id"]: e["discovered_by"] for e in entries}
    for e in entries:
        a = e["discovered_by"]
        for inp in (e["recipe"][1], e["recipe"][2]):
            if _is_discovered(inp):
                assert owner[inp] == a  # never builds on another agent's discovery


def test_embodied_recombiner_does_not_perturb_global_rng():
    from artificial_society.research.recombiner import run_embodied_recombiner

    random.seed(123)
    before = [random.random() for _ in range(3)]
    random.seed(123)
    run_embodied_recombiner(seed=7, n_attempts=50, n_agents=4)
    after = [random.random() for _ in range(3)]
    assert before == after
