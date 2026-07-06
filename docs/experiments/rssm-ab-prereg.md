# Pre-registration: RSSM sample-efficiency A/B (spec §10.2)

Frozen BEFORE confirmatory seeds. Pilot (3 seeds) may amend §Numbers once, before confirmatory runs.

## Design (frozen)
- Arms: A = v1 baseline; B = RSSM imagination actor; C = RSSM-WM-as-MPC ablation.
- Scale: grid 60×40, initial_population 36 (dashboard defaults). Run length: past founder
  transient to baseline plateau (pilot determines ticks; default 20000).
- Primary endpoint: KM-RMST(τ) of natural-born agents (spawn_origin=="birth", birth ≥ transient
  cutoff), right-censored at run end. Implemented in scripts/rssm_ab.py::rmst (unit-tested).
- Primary regime: transitions-matched. Compute axis reported as secondary.
- Transitions axis = cumulative stored transitions (B/C, `SharedLearner.transitions_stored`) /
  cumulative agent-ticks (A, `sum(len(sim.agents))` per tick), as logged per 500 ticks.
- Terminal marking = death flagged on the agent's last stored transition (≤1 tick offset
  approximation — a v1-substrate agent that dies exits `Agent.update` before that tick's store).
- Confirmatory tests: {B-vs-A, B-vs-C} on RMST, Wilcoxon signed-rank on within-seed deltas,
  Holm-corrected (2 comparisons). All other metrics exploratory.
- Collapse := population < 8 (MIN_POPULATION) for ≥ 200 consecutive ticks. Collapsed runs
  retained; worst-rank sensitivity analysis reported.
- Minimum effect of interest: +25% RMST (B vs A). No interim peeking at RMST.
- Tuning parity: equal tuning-run budget per arm, all tuning runs logged in this directory.

## Numbers (filled from the 3-seed pilot, then frozen)
- transient cutoff: ___ ticks (founder-overshoot end per pilot population curves)
- τ: ___ ticks; run length: ___ ticks
- SD of within-seed deltas: ___ ; seed-pair correlation r: ___ (if r < 0.3 → unpaired analysis)
- collapse fraction per arm: ___
- N (paired seeds) for 80% power at +25% RMST: ___ ; if N > 12 → downgrade to estimation-with-CI.
- n≥6 paired seeds required for any confirmatory Wilcoxon claim at α=0.05 (exact-test floor:
  p_min=0.25 at n=3).
