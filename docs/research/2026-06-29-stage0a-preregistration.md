# Stage 0a Pilot/Gate — Pre-Registration (B1 sample size · B2 primary DV · B3 inference)

- **Date:** 2026-06-29
- **Status:** Pre-registered **before** the confirmatory run. The decision *rules*
  below are fixed now; the pilot-derived *numbers* (§B1/§B2 "Pilot results") are
  computed from the exploratory n=12 pilot and then **locked** before any
  confirmatory data is collected.
- **Scope:** the open-ended-innovation Stage-0a gate (research lane only; no hot
  files). Companion to the design spec
  [`docs/superpowers/specs/2026-06-28-open-ended-innovation-research-design.md`](../superpowers/specs/2026-06-28-open-ended-innovation-research-design.md)
  (§9 Statistik & Pre-Registration, §11 Kill-Kriterien, §14 Stage 0).
- **Code under test:** `artificial_society/research/{metrics,analyze_gate,stats,recombiner,run_pilot,run_single}.py`
  at the commit recorded in §"Frozen parameters". Implements GPU-PC handoff
  `docs/handoff/perf-and-precision-recommendations.md` items A1/A2/B1–B3.

> **Why pre-register.** The adversarial review (design §3) showed naive emergence
> metrics are confounded by the recombination operator. The gate is only credible
> if the DV, sample size, and test are fixed *before* the confirmatory comparison,
> so the verdict cannot be reverse-engineered. The n=12 pilot is **exploratory**:
> its sole licensed uses are (a) estimating the paired-difference SD to size the
> confirmatory run (B1) and (b) choosing the lowest-variance valid DV (B2). It does
> **not** decide the gate.

---

## Design (fixed)

- **Paired, matched-seed** comparison: for each seed `s`, the learned/social arm
  and the **compute-matched random-recombiner null** are both run; the unit of
  analysis is the per-seed difference `d_s = DV(learned_s) − DV(recombiner_s)`.
  Compute matching = the recombiner runs exactly the learned arm's measured
  `combine_vectors` call count (`run_pilot` already enforces this per seed).
- **Determinism:** `PYTHONHASHSEED=0`, subprocess-per-run isolation, fixed per-worker
  BLAS/torch thread count. Parallelising seeds (`--workers`) is byte-identical to
  serial (test: `tests/test_research_parallel.py`).
- **Functional-depth DV** (not raw DAG depth) at observer parameter `func_tau`,
  reported across the sweep `{0.10, 0.15, 0.20}` with **primary `func_tau = 0.15`**.

---

## B2 — Primary dependent variable (selection rule fixed; choice locked from pilot)

**Candidates** (all returned by `metrics.analyze_registry`, same units = functional
depth): `max_functional_depth`, `p95_functional_depth`, `mean_functional_depth`.

**Rule (fixed):** `max_functional_depth` is a single extreme order statistic and is
the noisiest summary. On the n=12 pilot, compute for each candidate the between-seed
SD of the paired differences and the standardised paired effect
`dz = mean(d_s) / sd(d_s)` at `func_tau = 0.15`
(`analyze_gate.compare_dvs` / `python -m artificial_society.research.analyze_gate --compare-dvs`).
**Pre-register the candidate with the largest `dz` (most decisive / lowest relative
between-seed variance) as the confirmatory primary DV.** The other two are reported
as **sensitivity** DVs only. (Prediction from the handoff: `p95_functional_depth`.)

**Pilot results (locked from the n=12 pilot — filled before the confirmatory run):**

| DV | mean_diff | sd_diff | dz | CI half-width (n=12) |
|---|---|---|---|---|
| max_functional_depth | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| p95_functional_depth | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| mean_functional_depth | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

→ **Pre-registered primary DV: _TBD_ (largest dz).** Sensitivity DVs: the other two.

---

## B1 — Confirmatory sample size (rule fixed; n locked from pilot)

**Rule (fixed):** from the pilot's estimated paired-difference SD of the **chosen
primary DV** (`sd_diff` at `func_tau = 0.15`), the confirmatory n is the larger of
two pre-registered criteria, then floored:

1. **Target-CI-width:** smallest n so a 95% CI on the mean paired difference has
   half-width ≤ **`H = 0.5` functional-depth units**
   (`stats.required_n_for_halfwidth(sd_diff, 0.5)` = `ceil((z₀.₉₇₅·sd_diff/H)²)`).
2. **Power:** smallest n giving a two-sided paired test power **≥ 0.90** at the
   pilot's observed `dz` (`stats.required_n_for_power(dz, power=0.90)`, noncentral-t).
3. **Design floor:** `n ≥ 30` (design §9 "n≥30, ggf. deutlich mehr").

**Confirmatory `n = max(criterion 1, criterion 2, 30)`.** Seeds: a contiguous block
`1001 … 1000+n` (extends the pilot's `1001–1012`).

**Pilot results (locked — filled before the confirmatory run):**

- `sd_diff` (primary DV, func_tau 0.15) = _TBD_ ; `dz` = _TBD_
- criterion 1 (CI half-width ≤ 0.5) → n = _TBD_
- criterion 2 (power ≥ 0.90 at dz) → n = _TBD_
- **Pre-registered confirmatory n = _TBD_** ; seeds = `1001 … _TBD_`

---

## B3 — Confirmatory inference (fixed)

Reported by `analyze_gate` for the primary DV at `func_tau = 0.15`:

- **Per-arm bootstrap CIs** (existing) and **paired-difference BCa CI**
  (`stats.bca_mean_ci`, bias-corrected + accelerated — better than the percentile
  bootstrap at small n).
- **Exact paired sign-flip permutation test** (`stats.exact_sign_flip_test`): all
  `2ⁿ` sign assignments enumerated for an exact two-sided p-value (Monte-Carlo
  sign-flips only if `n > 20`). **Significance threshold α = 0.05.**
- **Standardised effect size** Cohen's `dz` with bootstrap CI (`stats.cohens_dz`).
- **Wilcoxon signed-rank p** (`stats.wilcoxon_p`) as a rank-based corroborating
  secondary (the design §9 robust/rank-based requirement).

**Pre-registered decision rule (confirmatory):**

- **PATH_A** ⟺ (per-arm bootstrap CIs **separated**) **AND** (paired-difference CI
  **> 0**) **AND** (exact permutation **p < 0.05**). Reported as
  `confirmatory_path_a` in the gate report. Effect reported as `dz [CI]`.
- **BORDERLINE_LEAN_A** ⟺ paired-difference CI > 0 (and/or permutation p < 0.05) but
  per-arm CIs overlap.
- **PATH_B_OR_RETROFIT** ⟺ otherwise (no CI-separated, permutation-significant
  advantage over the null) → retrofit P-A (policy-coupling) and repeat, or pivot to
  the methodology/null-result paper (design §11).

**Sensitivity / robustness (reported, not decisive):** the `func_tau` sweep
`{0.10, 0.15, 0.20}`, the two non-primary DVs, the matched-moisture (B4) and
matched-ingredient (B5) sensitivity nulls (when run), and the Wilcoxon p. Confirmatory
vs sensitivity analyses are kept explicitly separate (design §9 garden-of-forking-paths).

---

## Frozen parameters

- World/sim: `grid_w=30, grid_h=20, pop=24`, learned-arm `ticks = 8000` (confirmed
  Stage-0a scope); recombiner compute-matched per seed; recombiner `moisture = 0.5`
  (B4 samples the empirical learned-arm distribution as a sensitivity arm).
- Dedup radius 0.08; `func_tau ∈ {0.10, 0.15, 0.20}`, primary 0.15.
- `PYTHONHASHSEED=0`; `CUDA_VISIBLE_DEVICES=-1` (CPU); per-worker single-thread BLAS
  for the confirmatory run.
- Bootstrap resamples `N_BOOTSTRAP = 10000`, seeds fixed in code (deterministic CIs).
- Code commit: _TBD (record the feat/infra-research-precision merge commit)_.

---

## A1/A2 (speed — science-neutral, no pre-registration impact)

- **A1** parallel seeds (`run_pilot --workers`): identical per-seed bytes vs serial.
- **A2** cKDTree functional-depth scan + per-(seed,arm) structural-depth caching:
  **bit-identical** DVs to the O(n²) scan (tests: `tests/test_research_kdtree.py`).
  These change wall-clock only, never a reported value.
