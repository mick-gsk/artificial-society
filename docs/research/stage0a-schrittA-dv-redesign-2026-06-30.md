# Schritt A — DV redesign + fairness proof on the existing pilot

- **Date:** 2026-06-30
- **Branch / code:** `feat/infra-research-stage0a` (research lane, **non-hot**: `research/metrics.py`,
  `research/analyze_gate.py`, `tests/test_research_metrics.py` only — golden untouched).
- **Input data:** the existing n=12 × 1500-tick pilot exports
  ([`stage0a-pilot-2026-06-29/report.md`](stage0a-pilot-2026-06-29/report.md)).
- **Design step executed:** [`retrofit-path-a-design.md`](retrofit-path-a-design.md) §A + §3.1–3.2
  ("DV redesign FIRST; prove fairness on existing JSONs *before* any hot-file work or re-pilot").

## What Schritt A was supposed to settle

The pilot's gate DV (`max_functional_depth`) reads only the **geometry** of the global discovery
registry — it is blind to reward, adoption (`uses`), agent attribution (`discovered_by`) and
transmission. The adversarial review concluded that four of the five retrofit mechanisms therefore
**cannot move that DV by construction**, and that the *first* lever is a measurement reform: DVs that
score the **realised, function-advancing, adopted, transmitted** frontier — the cumulative-culture
criterion (Tennie/Call/Tomasello 2009; Henrich 2015), not artifact count.

Two questions, both answerable **without touching hot files or re-running the sim**:

1. **Instrumentation check** — does the learned-arm export even carry the signals the new DVs need?
2. **Fairness proof** — on the new DVs, does the random recombiner collapse (validating the DV), and
   does the learned arm now look better?

## Result 1 — Instrumentation check (Step 1): the export is partly blind

| field | learned arm | recombiner arm | usable for new DVs? |
|---|---|---|---|
| `uses` | 88.5 % > 0, mean ≈ 686, max ≈ 26 k | **all 0** (disembodied) | **polluted + asymmetric** — incremented on every `get_vector` lookup, not on real homeostatic use; 0 for the recombiner *by construction* |
| `discovered_by` | **all −1** (no attribution) | all −1 | **DV3 not computable** |
| `tick` (per entry) | **all 0** | populated (= attempt index) | no per-entry timing for the learned arm |

So **DV1 (`population_functional_value`)** rests on a polluted, arm-asymmetric weight, and
**DV3 (`transmitted_frontier_advances`)** is *not computable at all* from the current export. This is
exactly the "instrumentation gap" the design flagged as the central single-point-of-failure — now
confirmed empirically rather than assumed.

## Result 2 — Fairness comparison (`analyze_gate --compare-dvs`, n=12, func_tau=0.15)

| DV | learned | recombiner | paired diff | sd_diff | dz | verdict |
|---|---|---|---|---|---|---|
| `population_functional_value` | 3,431,130 | 0.000 | +3,431,130 | 757,987 | +4.53 | **PATH_A** *(provisional — artefact)* |
| `mean_functional_depth` | 7.19 | 15.79 | −8.60 | 1.27 | −6.78 | PATH_B_OR_RETROFIT |
| `useful_depth_max` | 1.17 | 18.08 | −16.92 | 4.40 | −3.85 | PATH_B_OR_RETROFIT |
| `accumulated_useful_depth` (DV2) | 2.17 | 32.25 | −30.08 | 11.58 | −2.60 | PATH_B_OR_RETROFIT |
| `max_functional_depth` (old gate) | 22.92 | 32.33 | −9.42 | 3.85 | −2.45 | PATH_B_OR_RETROFIT |
| `transmitted_frontier_advances` (DV3) | — | — | — | — | — | instrumentation-blocked |

**The DV redesign does not rescue the learned arm — it deepens the deficit.** On *every legitimate*
DV the learned/social arm is further below the random recombiner than the old gate showed. The only
DV where learned "wins" is `population_functional_value`, and that win is the **instrumentation
artefact** we predicted: the recombiner scores 0 purely because its `uses` is structurally 0, not
because of any behaviour. It is correctly flagged `*provisional` and **must not** be used until `uses`
is replaced by a clean `record_use` signal. (This is itself a clean teaching example of
confound-laundering for the methodology paper.)

## Result 3 — DV2 deep-dive (the clean, arm-symmetric, churn-immune DV)

`accumulated_useful_depth` is a pure function of vectors + recipes + the fixed task basis — no
instrumentation — so it is the DV that can be trusted on the existing data. Per-task max advance depth
(mean over seeds), with base-material ceilings:

| task | base ceiling | learned | recombiner |
|---|---|---|---|
| `warm_meal` (edible·hot·safe) | 0.000 (headroom) | **0.00** | **18.08** |
| `edible_safe` | 0.95 | 0.00 | 2.83 |
| `cutting_tool` (sharp·hard) | 0.90 | 1.17 | 9.00 |
| `fragrant` | 0.90 | 1.00 | 2.33 |
| `portable_light` | 1.00 (base-saturated) | 0.00 | 0.00 |
| `heat` | 1.00 (base-saturated) | 0.00 | 0.00 |

The headline: on **`warm_meal`** — the canonical "cooking" innovation, the one task base materials do
*not* provide — the embodied learned agents (who get hungry and live beside fire) **never advance the
frontier at all**, while the blind recombiner reaches depth 18. The learned arm barely advances any
task frontier; its raw functional depth (max-depth 22.9) is largely **functionally aimless churn**.

**Discovery-matched DV2** (recombiner truncated to the learned arm's discovery count, isolating the
volume confound): paired diff **−21.75** (sd 7.37) — vs −2.83 for discovery-matched max-depth. The
functional lens makes the learned arm look *much* worse, and the deficit survives discovery-matching.

> Caveat: DV2 credits the *structural depth* of advancing artifacts, so it inherits max-depth's
> volume sensitivity (more attempts → deeper advances). The discovery-matched −21.75 is the honest
> control. The lowest-variance single statistic remains `mean_functional_depth` (sd_diff 1.27,
> dz −6.78). Two of the six tasks (`heat`, `portable_light`) are base-saturated and contribute 0 to
> both arms — the final pre-registered basis should drop them in favour of headroom conjunctions
> (`warm_meal` is the model).

## Conclusion

1. **Measurement reform delivered & validated.** `accumulated_useful_depth` (DV2) is a fair,
   arm-symmetric, churn-immune DV grounded in the cumulative-culture criterion, computable on existing
   data. The recombiner does **not** collapse on it — it is a genuine, strong null.
2. **The reform refutes the retrofit's cheap-win hope.** Redesigning the DV does not surface a hidden
   learned advantage; on the human-grounded functional DV the learned/social arm is ~6 % of the
   recombiner and on the cooking task never beats base. The bar for a retrofit is now **higher**
   (flip −21.75 discovery-matched DV2 toward > 0), and that can only come from the generation-coupling
   mechanisms (C→B→D) actually making agents advance task frontiers — not from measurement.
3. **DV1/DV3 are instrumentation-blocked.** Their apparent signal is an artefact (DV1) or absent
   (DV3). Using them requires new export fields the pilot lacks.

This strengthens **Path B** (the null is now richer: naive depth metrics *flatter* the learned arm;
under a value-based DV the deficit is larger and survives discovery-matching, plus a documented
confound-laundering example). It makes the **retrofit** a clearer, larger gamble than the −9.4
framing suggested.

## Step-1 instrumentation backlog (required before DV1/DV3 are trustworthy)

Only needed on the retrofit path; all touch the sim, so they are **not** Schritt A:

- **Clean `record_use`** at homeostatic-relief sites (eat/warm/cut/…), replacing the `get_vector`
  increment that pollutes `uses`. Touches `environment/materials.py` (**hot**, registry) + the relief
  call-sites → core-lead.
- **Populate `discovered_by`** on registration (the inventing agent's id) and **persist a per-entry
  `tick`**. `register(...)` is in `materials.py` (**hot**); the call-sites are in `systems/invention.py`
  / `need_driven_invention.py` (non-hot) → core-lead for the signature, lane dev for the call-sites.
- **Export the per-agent adopter set** (who re-used each recipe) so DV3's "adopted by ≥k agents /
  transmitted ≥1 hop" is real, not a `uses≥k` proxy. Read-only export add in `research/run_single.py`
  once the sim records it.

## Reproduce

```bash
# research lane, branch feat/infra-research-stage0a:
python -m pytest tests/test_research_metrics.py -q
python -m artificial_society.research.analyze_gate --outdir <pilot_out> --compare-dvs
```
