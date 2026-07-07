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

## Numbers (filled from the 3-seed pilot 2026-07-06/07, hereby frozen)
- transient cutoff: **1500** ticks (founder overshoot peaks ≈ tick 1000 in 7/9 runs, decayed by 1500)
- τ: **2000** ticks; run length: **20000** ticks
- Pilot RMST: A = 739.5 / 722.9 / 768.3 (median 739.5); B = 785.7 / 797.0 / 783.0 (median 785.7);
  C = 704.8 / 745.0 / 712.1 (median 712.1). B > A in 3/3 seeds (deltas +46.1/+74.2/+14.7, mean
  +45.0 ≈ +6%); B > C in 3/3 (deltas +80.9/+52.0/+70.9, mean +67.9 ≈ +9.5%).
- SD of within-seed B−A deltas: **29.8**; seed-pair correlation r ≈ **−0.89** (n=3, unstable, but
  < 0.3) → **unpaired analysis** per the frozen rule (Mann-Whitney U replaces Wilcoxon signed-rank).
- collapse fraction per arm: **0 / 0 / 0** (no run's logged population ever dropped below 8).
- Power: observed between-run SDs (A ≈ 22.8, B ≈ 7.4) put the +25%-RMST MDE (≈ +185 ticks) at
  d ≫ 2 → nominally n ≤ 4/arm; but SD estimates at n=3 are unreliable and the exact-test floor
  binds → **N = 8 seeds per arm** for confirmatory runs. NOTE: the observed pilot effect (+6%)
  is **below the pre-declared +25% minimum effect of interest** — per the frozen rule, unless the
  confirmatory effect reaches the MDE, results are reported as **estimation-with-CI**, not as a
  confirmed superiority claim.
- Transitions axis: cumulative transitions at T=20000 — A 648k–777k, B 674k–873k, C 729k–1014k
  (population feedback: better arms generate more data; the transitions-matched view evaluates at
  the min common count, reported alongside the tick-based view).
- n≥6 paired seeds required for any confirmatory Wilcoxon claim at α=0.05 (exact-test floor:
  p_min=0.25 at n=3) — superseded by the unpaired switch above.
