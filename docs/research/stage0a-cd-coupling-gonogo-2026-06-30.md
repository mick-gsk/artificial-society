# Stage-0a — C+D generation-coupling go/no-go (2026-06-30)

- **Date:** 2026-06-30
- **Provenance:** cheap pre-commitment probe before building the full C→B→D retrofit
  ([`../superpowers/plans/2026-06-30-retrofit-path-a-cbd.md`](../superpowers/plans/2026-06-30-retrofit-path-a-cbd.md)),
  on top of the Schritt-A measurement reform
  ([`stage0a-schrittA-dv-redesign-2026-06-30.md`](stage0a-schrittA-dv-redesign-2026-06-30.md)).
- **Status:** internal result feeding the Path-B paper ([`path-b-paper-outline.md`](path-b-paper-outline.md)).
  Not a standalone paper claim — see *Limitations*.

## TL;DR

A first attempt to **couple invention generation to value** — Phase C (value-gated marginal
reward over a transmitted tribe frontier) **plus** Phase D (imagined means-ends selection of *what*
to combine via a pure, uncounted `_combine_pure` twin) — **does not move the functional-depth gate**.
Across 4 paired seeds × 1500 ticks, **DV2 `accumulated_useful_depth` stayed pinned at 2 in every arm
and every seed**, identical to the OFF baseline. Per-attempt discovery efficiency *fell* (0.127 →
0.029), driven by a greedy diversity-collapse in the means-ends search. The retrofit's central
prediction — that coupling generation flips the gate — is not supported by this probe. Combined with
Schritt A, this closes the Path-A decision in favour of **Path B**.

## Method

- **C (value reward):** flat `NEW_DISCOVERY_BONUS` replaced by
  `material_reward + RATCHET_GAIN · marginal`, where `marginal` is the gain over a monotone,
  tribe-keyed functional frontier (`systems/_lineage_frontier.py`). Reward-blind metrics cannot see
  this by construction (Schritt A), but it removes the novelty pump that would otherwise reward churn.
- **D (means-ends generation):** before each real combine, score `top_n` need-relevant inputs ×
  `PRIMITIVE_ACTIONS` with a **deterministic, uncounted** `_combine_pure(...)` twin (distinct function
  object → never counted by `research.instrument`), pick by
  `value + voi_weight · novelty + RATCHET_GAIN · marginal`, execute only the choice via the real
  (counted) `combine_vectors`. Compute-match integrity verified by a unit test (`cc.n == 0` for the
  imagined search).
- **A/B:** single env flag `AS_PATHA_CD` gates C+D so OFF is **byte-identical** to the live system
  (golden trajectory + headless digest stay green; 23/23 tests). Paired by seed; 4 seeds (1001–1004),
  1500 ticks, grid 30×20, pop 24, CPU, `PYTHONHASHSEED=0`, `CUDA_VISIBLE_DEVICES=-1` — the locked
  pre-registration regime.
- **DVs:** per-attempt efficiency = `n_discoveries / executed combine_vectors calls`; DV2
  `accumulated_useful_depth` (`research/metrics.py`, Schritt A).

## Results

| seed | eff OFF | eff ON (C+D) | DV2 OFF | DV2 ON | discoveries OFF→ON |
|-----:|--------:|-------------:|--------:|-------:|-------------------:|
| 1001 | 0.114   | 0.038        | 2       | 2      | 4605 → 1446        |
| 1002 | 0.140   | 0.028        | 2       | 2      | 3897 → 1281        |
| 1003 | 0.141   | 0.027        | 2       | 2      | 5412 → 1229        |
| 1004 | 0.115   | 0.023        | 2       | 2      | 5364 → 1192        |

- **Efficiency:** mean 0.127 → 0.029 (≈0.23×); improved in **0/4** seeds.
- **DV2 (the gate):** **flat at 2** in all 8 runs. The OFF baseline (0.127, DV2=2) reproduces the
  Stage-0a pilot scale (efficiency ≈0.135; learned DV2 ≈2 vs recombiner ≈32).
- A 250-tick smoke had shown C+D *ahead* (+35% efficiency); this is a pre-collapse transient that
  reverses by ~1500 ticks.

## Interpretation

1. **The gate does not respond to generation coupling (load-bearing).** DV2 is reward-blind, so C
   cannot move it by construction; and D — which *does* change what gets discovered — did not lift DV2
   off its floor. Even where C+D was active, the realized functional-depth geometry was unchanged.
2. **Efficiency collapse = greedy diversity loss.** The means-ends search herds onto a few
   high-value combinations and re-discovers them, while the random baseline keeps exploring breadth.
   This is the design doc's predicted failure mode (D forced-fix #2: a greedy need-maximiser lowers
   diversity).

## Limitations (why this is a probe, not a paper claim on its own)

- **Greedy D, not the plan's spec.** This D used **argmax** + a 512-entry novelty cap, *not* the
  plan's *seeded softmax + strong value-of-information* term. A plan-faithful D would likely recover
  efficiency — but, given DV2 sits at the floor for the OFF baseline too, would not be expected to
  move DV2. The efficiency number should **not** be read as the lever's ceiling.
- **C and D conflated** in the ON arm (no C-only cell). C alone is reward-blind to DV2, so this does
  not change the gate conclusion, but it does limit attribution of the efficiency change.
- **Not a full learned end-to-end coupling.** Cook et al. (2024) couple generation/selection through
  the *learner's policy*; this probe couples through reward shaping + an imagined-search heuristic.
  Refuting *all* policy-coupling is beyond this experiment — see *Implication*.

## Implication for the paper

- Strengthens contribution #2 (robust null): the deficit survives **measurement reform** (Schritt A)
  **and** a **first generation-coupling attempt** (this note).
- Sharpens the boundary of the claim: we show naive value+means-ends coupling does not rescue the
  gate; full learned end-to-end coupling (à la Cook et al.) is the explicit edge of our claim and
  named as future work, not something we assert to have refuted.
- Resolves the outline's open **Decision note** toward Path B.

## Reproduction

Flag-gated A/B harness (worker + driver) in the session scratchpad; results in
`gonogo_results.jsonl`. The C-phase implementation is committed (`be22f20`→`2bdae80`); D was
prototyped behind `AS_PATHA_CD` and is not committed. To re-run a definitive (plan-faithful) D,
implement seeded-softmax + VoI selection on top of the committed C and re-pilot at the locked regime.
