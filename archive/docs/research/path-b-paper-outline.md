# Path B — Paper outline (methodology / null-result)

Working plan for the paper the Stage-0a gate points to. Reversible: if we later pursue the
P-A retrofit (policy-coupling) and it beats the null, this converts into a positive mechanism
paper; the control protocol below carries over unchanged.

## Framing

- **Working title:** *"Open-endedness or just the operator? A confound-controlled test of learned
  social innovation in a neural multi-agent world."*
- **Venue:** ALOE @ ICLR / CoLLAs (workshop MVP) → *Artificial Life* (journal extension).
- **Thesis (one sentence):** In a neural multi-agent testbed, learned/social transmission produces
  **no** functional-innovation advantage over a compute-matched random recombiner, and the apparent
  "open-endedness" of naive complexity metrics is largely an artifact of exploration volume — so
  open-ended-learning claims need the control protocol we provide.

## Contributions

1. **A confound-controlled measurement protocol** for open-ended-innovation claims: functional /
   irreducible complexity (knock-out validated) instead of DAG depth; **compute-matched** *and*
   **discovery-matched** comparison to a random-recombiner null; rarefaction to expose sample-size
   effects; bootstrap CIs; pre-registration.
2. **An empirical null result** in *Artificial Society*: learned/social ≈/< random recombiner on
   functional innovation (attempt-matched −9.4; discovery-matched −2.8, both paired-significant),
   with ~70 % of the raw gap shown to be a sampling artifact. The deficit is **robust to measurement
   reform** — under value-based DVs the gap *widens* (DV2 paired −30.1, discovery-matched −21.75;
   `stage0a-schrittA-dv-redesign-2026-06-30.md`) — **and to a first generation-coupling attempt**
   (value reward + imagined means-ends generation: functional-depth DV unmoved, flat at floor;
   `stage0a-cd-coupling-gonogo-2026-06-30.md`).
3. **Open tooling + reproducible pipeline** (`artificial_society/research/`, config + seeds → figures).

## Positioning vs related work (must-distinguish)

- **Cook/Cohen/Foerster et al. 2024 (AGI: cultural accumulation in RL, NeurIPS).** They obtain
  *positive* accumulation with mechanisms that tie generation/selection to the learner. **Contrast:**
  we show that *without* policy-coupling you recover nothing beyond the recombination operator, and
  that naive metrics hide this — a measurement/critique complement, not a competitor.
- **Leibo et al. 2019 (autocurricula manifesto), Hughes et al. 2024 (open-endedness for ASI).**
  We supply the *control* their claims need.
- **Dolson 2019 (MODES), Bedau-Packard.** We position functional complexity + neutral-shadow within
  this metric family and add the operator/sampling confound controls.
- **SAPIENS, Derex & Boyd.** Network/demography effects are explicitly **out of scope** for Path B
  (they belong to Path A / the journal extension).

## Methods (have)

Standalone compute-matched random recombiner; functional/irreducible depth (`func_tau` sweep) +
knock-out; attempt- and discovery-matching (append-only prefixes); rarefaction; 12 paired seeds;
percentile bootstrap; CPU determinism (`PYTHONHASHSEED=0`). See `stage0a-pilot-2026-06-29/report.md`.

## What the paper still needs (gap list, cheap → expensive)

1. **Learning-signal analysis — partially done** (`analyze_learning.py`, Result 3). Inventive
   efficiency (learned **5.2× below** the random operator, persistent) and no energy-adaptation are
   shown from the exported series. A direct **reward/return curve** (does PPO return rise with
   training?) would sharpen the "no learning" claim but needs a small export add + one short re-run —
   optional for the workshop MVP.
2. **Fairer-null sensitivity (cheap).** Re-run the recombiner restricted to the materials agents
   actually encounter (ingredient-matched), and with env-sampled `moisture`; confirm the sign holds.
3. **The other two nulls (medium, Stage 0b, hot files).** frozen-brain + random-action strengthen the
   "behaviour/learning doesn't matter" claim; optional for the workshop MVP.
4. **Full-scale confirmation (expensive, optional).** 8000-tick × n≥30 run (overnight on the GPU PC)
   for the camera-ready.
5. **MODES/Bedau adaptive-vs-neutral pass (medium).** Frame the result in the standard ALIFE toolbox.

## Threats addressed (paper subset of spec §10)

F1 random recombiner reproduces the signal → it's the primary baseline. F3 DAG depth artifact →
functional depth + knock-out. Sample-size confound → discovery-match + rarefaction. M3 200-tick cap →
uncapped export. M4 singleton contamination → subprocess isolation.

## Reproducibility

Config + seed list + `PYTHONHASHSEED=0`; `run_pilot` → `analyze_gate` / `analyze_robustness`; release
code + data + notebooks, Zenodo DOI; OSF pre-registration of DV, nulls, matching, rejection rule.

## Decision note (resolved 2026-06-30 → Path B)

Two follow-ups closed the Path-A question this outline originally left open:

- **Schritt A (measurement reform).** Redesigning the DV to value-based, churn-immune metrics did
  **not** rescue the learned arm; the deficit *deepened* (DV2 paired −30.1). The null is not a metric
  artifact. (`stage0a-schrittA-dv-redesign-2026-06-30.md`)
- **C+D generation-coupling go/no-go.** A first coupling attempt (value-gated marginal reward +
  imagined means-ends generation) left the functional-depth gate **flat at the floor** (DV2 = 2 in
  every arm/seed) while collapsing discovery efficiency. (`stage0a-cd-coupling-gonogo-2026-06-30.md`)

Honest reading: a positive flip is not reachable without **mechanism change well beyond** reward
shaping + heuristic generation. **Full learned end-to-end policy-coupling (Cook et al. 2024)** is the
named boundary of our claim and the journal-extension's question — not something this paper asserts to
have refuted. **Path B is the committed write-up.**
