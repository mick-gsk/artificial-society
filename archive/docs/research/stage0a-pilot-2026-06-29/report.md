# Stage 0a Pilot — Results & Gate Decision

- **Date:** 2026-06-29
- **Branch / code:** `feat/infra-research-stage0a` @ `eba8982`
- **Spec:** [`../../superpowers/specs/2026-06-28-open-ended-innovation-research-design.md`](../../superpowers/specs/2026-06-28-open-ended-innovation-research-design.md)
- **Verdict:** **`PATH_B_OR_RETROFIT`** (pre-registered, robust to the exploration-volume confound)

## Setup

- **Arms (paired by seed):** *learned/social* = the unmodified `Simulation`; *random-recombiner* =
  the disembodied standalone operator (`artificial_society/research/recombiner.py`) driving
  `combine_vectors` with uniform inputs/actions, **compute-matched** to the learned arm's measured
  number of `combine_vectors` calls.
- **Scale:** n = 12 paired seeds, 1500 ticks, grid 30×20, pop 24, **CPU**
  (`CUDA_VISIBLE_DEVICES=-1` — the GPU FP16 path crashes and CPU is 7–11× faster for these tiny
  nets anyway). 8000 ticks was measured at ~9 h CPU; 1500 is past the recombiner's functional
  plateau and the learned arm is not policy-coupled, so more ticks are not expected to change the
  sign. Ran in ~15 min (8 parallel seed-workers on a Ryzen 9800X3D).
- **Primary DV (pre-registered):** maximum **functional / irreducible** depth (collapses redundant
  deep chains via a functional-equivalence radius `func_tau`), *not* raw DAG depth.
- **Stats:** percentile bootstrap CIs (10k); gate rule = learned beats null with separated per-arm
  CIs **and** paired-difference CI > 0. `func_tau` sweep 0.10 / 0.15 / 0.20.

## Result 1 — Gate (compute/attempt-matched)

| arm | max functional depth (mean) | 95% CI |
|---|---|---|
| random-recombiner | **32.33** | [31.42, 33.33] |
| learned/social | **22.92** | [21.00, 24.83] |
| **paired difference** | **−9.42** | **[−11.33, −7.17]** |

Stable across `func_tau` ∈ {0.10, 0.15, 0.20}. The learned/social arm is **significantly *below*** the
null. See `gate_figure.png` — all 12 paired lines slope down.

## Result 2 — Robustness to the exploration-volume confound

`max_functional_depth` is an extreme-value statistic that grows with sample size. At equal #attempts
the disembodied recombiner makes ~5–6× more discoveries (learned mean ≈ 5,402; recombiner mean ≈
28,528), so part of its lead is pure sampling. Controlling for it (registry is append-only, so
`entries[:k]` = the state after k discoveries):

| comparison | learned | recombiner | paired diff | verdict |
|---|---|---|---|---|
| attempt-matched (gate) | 22.92 | 32.33 | −9.42 [−11.33, −7.17] | PATH_B_OR_RETROFIT |
| **discovery-matched** | 22.92 | 25.75 | **−2.83 [−4.83, −0.58]** | PATH_B_OR_RETROFIT |

≈ **70 % of the recombiner's raw lead was a sample-size artifact.** The residual −2.83 still excludes
0 (paired): even discovery-for-discovery the random recombiner reaches *slightly deeper* functional
structure than the learned/social agents. `rarefaction_figure.png` shows both curves rising with
#discoveries (the confound) with the recombiner consistently a few units above the learned arm over
the common range.

## Result 3 — Inventive efficiency & demography (no learned advantage)

Same exported data; see `learning_figure.png` (`analyze_learning.py`).

- **Inventive efficiency** (new discoveries per combination attempt, common range): learned **0.135**
  vs recombiner **0.702** — the embodied learned/social agents are **5.2× less efficient inventors**
  than uniform random recombination, throughout the run. Their invention is dominated by
  re-discovery of locally-available materials; there is no learned novelty-seeking that would
  approach, let alone beat, the operator.
- **Demography:** population boom-busts (24 → ~310 → ~110 over 1500 ticks); **energy/agent** peaks
  early (~tick 100) then declines and plateaus — **no upward adaptation trend**. The innovation
  unfolds amid strong ecological dynamics, not a learning ramp.

*Caveat:* a declining per-attempt efficiency is expected in any saturating discovery process; the
informative quantity is the **learned-vs-operator gap** (5.2×) and its persistence, not the decline.

## Interpretation

1. **No open-ended-learning advantage in the current system.** The learned/social machinery provides
   no functional-innovation gain over uniform random recombination — even slightly negative, both
   compute-matched and discovery-matched. Consistent with the design's fatal finding: invention is
   **not policy-coupled** (the PPO policy does not choose *what* to combine).
2. **Naive complexity metrics are confounded.** Raw `max functional depth` overstated the gap ~3×
   purely via exploration volume — a concrete, quantified instance of the confound the methodology
   is meant to expose. The discovery-matched / rarefaction control is the fix.

## Decision

Pre-registered branch reached: **`PATH_B_OR_RETROFIT`**.

- **Path B — methodology / null-result paper (recommended).** Contribution = a confound-controlled
  protocol (compute-match **and** discovery-match + rarefaction) showing that, in a neural
  multi-agent testbed, learned/social transmission does **not** beat a random recombiner on
  functional innovation, and that naive metrics inflate "open-endedness" via sampling. Robust and
  data-supported today.
- **Retrofit P-A (policy-coupling) + re-pilot (alternative).** Tests "learning loses only because
  invention isn't policy-coupled." The attempt-matched deficit is large (−9.4) but the
  discovery-matched deficit is modest (−2.8), so the payoff is uncertain. P-A is the expensive
  hot-file rebuild (`agents/agent.py`, `agents/brain.py`) → core-lead + golden regeneration.

## Caveats / limits

- 1500 ticks (not 8000); a full 8000-tick confirmation run can be done overnight if needed.
- The recombiner has full access to all 24 base materials and is disembodied — a deliberately
  **strong/conservative** null. A future fairer variant could restrict it to the materials agents
  actually encounter, or match the per-tick environment distribution.
- DV is max functional depth; the discovery-matched + rarefaction views are the confound controls.

## Reproduce

```bash
# on the GPU PC (or any CPU host), branch feat/infra-research-stage0a:
CUDA_VISIBLE_DEVICES=-1 python -m artificial_society.research.run_pilot --ticks 1500 --workers 8
python -m artificial_society.research.analyze_gate       --outdir artificial_society/research/out
python -m artificial_society.research.analyze_robustness --outdir artificial_society/research/out
```
