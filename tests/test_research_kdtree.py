"""A2: the KDTree functional-depth path must be BIT-IDENTICAL to the O(n^2) one.

The whole point of A2 is a speedup that changes nothing scientifically: the gate
DVs computed via the ``cKDTree`` radius query must equal, value-for-value, the
ones from the original brute-force neighbourhood scan. We assert exact equality
(not "close") on randomised registries, on a boundary-stress case, and at the
``analyze_registry`` DV level.
"""

from __future__ import annotations

import numpy as np
import pytest

from artificial_society.research import analyze_gate, metrics
from artificial_society.research.export import dump_run, load_run
from artificial_society.research.metrics import analyze_registry


def _random_registry(n: int, rng: np.random.Generator, dims: int = 12) -> list[dict]:
    """A synthetic entry list: random 12-dim vectors + a plausible struct depth."""
    entries = []
    for i in range(n):
        entries.append(
            {
                "id": f"mat_{i:04d}",
                "recipe": ["bundle", f"mat_{max(0, i - 1):04d}", "stone"],
                "vector": [round(float(x), 6) for x in rng.random(dims)],
            }
        )
    return entries


def _arrays(entries, func_tau):
    struct = metrics.structural_depths(entries)
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    sd = np.array([struct[e["id"]] for e in entries], dtype=np.int64)
    ids = [e["id"] for e in entries]
    return V, sd, ids


@pytest.mark.parametrize("n", [1, 7, 60, 250])
@pytest.mark.parametrize("func_tau", [0.05, 0.10, 0.15, 0.20, 0.40])
def test_kdtree_matches_bruteforce_exactly(n, func_tau):
    rng = np.random.default_rng(1234 + n)
    entries = _random_registry(n, rng)
    V, sd, ids = _arrays(entries, func_tau)

    brute = metrics._functional_depths_bruteforce(V, sd, ids, func_tau)
    kd = metrics._functional_depths_kdtree(V, sd, ids, func_tau)

    assert kd == brute


def test_kdtree_boundary_stress():
    """Points placed at exactly +/- epsilon of func_tau must classify identically."""
    func_tau = 0.15
    base = [0.5] * 12
    entries = [{"id": "mat_0000", "recipe": None, "vector": base}]
    # neighbours at distances straddling func_tau along axis 0
    for k, delta in enumerate(
        [func_tau - 1e-6, func_tau, func_tau + 1e-6, func_tau - 1e-3, func_tau + 1e-3]
    ):
        v = list(base)
        v[0] = 0.5 + delta
        entries.append({"id": f"mat_{k + 1:04d}", "recipe": None, "vector": v})
    V, sd, ids = _arrays(entries, func_tau)
    assert metrics._functional_depths_kdtree(
        V, sd, ids, func_tau
    ) == metrics._functional_depths_bruteforce(V, sd, ids, func_tau)


def test_analyze_registry_dvs_identical_with_and_without_scipy(monkeypatch):
    """The gate DVs must not depend on whether the KDTree fast path is taken."""
    rng = np.random.default_rng(99)
    entries = _random_registry(120, rng)

    fast = metrics.analyze_registry(entries, func_tau=0.15)  # KDTree path (scipy present)

    monkeypatch.setattr(metrics, "_HAS_SCIPY", False)  # force brute-force
    slow = metrics.analyze_registry(entries, func_tau=0.15)

    assert fast == slow


def test_gate_struct_cache_is_bit_identical(tmp_path):
    """A2 caching: reusing one cached struct across taus must not change any DV."""
    seeds = [1001, 1002, 1003]
    for s in seeds:
        for arm, k in (("learned", 90), ("recombiner", 140)):
            entries = _random_registry(k, np.random.default_rng(s * 31 + len(arm)))
            dump_run(str(tmp_path / f"{arm}_seed{s}.json"), {"seed": s, "arm": arm}, [], entries)

    paired = analyze_gate._collect(str(tmp_path))
    cache = analyze_gate._load_cache(paired)

    for tau in analyze_gate.FUNC_TAU_SWEEP:
        lv, rv, got_seeds = analyze_gate._dv_for_tau(cache, tau, "max_functional_depth")
        assert got_seeds == seeds
        for i, s in enumerate(seeds):
            le = load_run(paired[s]["learned"])["entries"]
            re = load_run(paired[s]["recombiner"])["entries"]
            # fresh recompute (struct recomputed per tau) must equal the cached path
            assert lv[i] == analyze_registry(le, tau)["max_functional_depth"]
            assert rv[i] == analyze_registry(re, tau)["max_functional_depth"]


def test_compare_dvs_reports_all_candidates(tmp_path):
    """B2 pilot helper: summarise every candidate DV and rank them by decisiveness."""
    for s in (1001, 1002, 1003, 1004):
        for arm, k in (("learned", 120), ("recombiner", 160)):
            entries = _random_registry(k, np.random.default_rng(s * 17 + len(arm)))
            dump_run(str(tmp_path / f"{arm}_seed{s}.json"), {"seed": s, "arm": arm}, [], entries)

    report = analyze_gate.compare_dvs(str(tmp_path))
    assert report["n_seeds"] == 4
    assert set(report["summaries"]) == set(analyze_gate.CANDIDATE_DVS)
    assert set(report["ranking"]) == set(analyze_gate.CANDIDATE_DVS)
    for dv in analyze_gate.CANDIDATE_DVS:
        assert "sd_diff" in report["summaries"][dv]
        assert "dz" in report["summaries"][dv]
