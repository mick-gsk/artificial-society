"""End-to-end smoke test for the Stage-0a pilot pipeline.

Runs both arms tiny, in-process, then applies the gate analysis. Asserts the
pipeline produces a well-formed verdict; it does NOT assert which path (that is
the empirical gate question, answered by the full pilot).
"""

from __future__ import annotations

from artificial_society.research import analyze_gate
from artificial_society.research.export import dump_run
from artificial_society.research.run_single import run_learned, run_recombiner_arm


def test_pilot_pipeline_smoke(tmp_path):
    seeds = [1001, 1002]
    for s in seeds:
        meta, series, entries = run_learned(s, ticks=12, grid_w=12, grid_h=9, pop=6)
        assert meta["n_attempts"] >= 0
        assert len(series) == 12
        dump_run(str(tmp_path / f"learned_seed{s}.json"), meta, series, entries)

        rmeta, rseries, rentries = run_recombiner_arm(s, max(1, meta["n_attempts"]), 0.5)
        dump_run(str(tmp_path / f"recombiner_seed{s}.json"), rmeta, rseries, rentries)

    report = analyze_gate.analyze(str(tmp_path), plot=False)
    assert report["n_seeds"] == 2
    assert report["primary"]["verdict"] in {
        "PATH_A",
        "BORDERLINE_LEAN_A",
        "PATH_B_OR_RETROFIT",
    }
    # sensitivity sweep must report a verdict per func_tau
    assert set(report["sensitivity"].keys()) == set(analyze_gate.FUNC_TAU_SWEEP)
