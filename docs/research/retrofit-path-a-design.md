# Retrofit (Path A) — human-plausible mechanisms to beat the random recombiner

- **Date:** 2026-06-29
- **Provenance:** ultracode multi-agent workflow (16 agents: 5 mechanism-research × 3 codebase-seam ×
  synthesis × 6 adversarial verdicts × integration), grounded in the Stage-0a pilot
  ([`stage0a-pilot-2026-06-29/report.md`](stage0a-pilot-2026-06-29/report.md)) and the spec
  ([`../superpowers/specs/2026-06-28-open-ended-innovation-research-design.md`](../superpowers/specs/2026-06-28-open-ended-innovation-research-design.md)).
- **Status:** design for review — **not yet implemented**. Touches hot files → core-lead + golden regen.

## 1. Core insight (sharpened by adversarial review)

The pilot was lost **on the one axis the recombiner wins by construction**: `max_functional_depth`
is a volume/extreme-value statistic; the disembodied recombiner draws frictionlessly from a global
pool, so more attempts mechanically buy more depth. That stays true.

**The decisive, adversarially-forced correction:** four of the five in-sim mechanisms attack
diagnoses the **offline gate DV cannot see**:

- `research/metrics.py` (`analyze_registry`) computes every DV (`max/p95/mean_functional_depth`,
  `n_functional_clusters`) **only from the exported vectors of the single global
  `DISCOVERY_REGISTRY`** via `combine_vectors`. It never reads reward, `uses`, `discovered_by`,
  CausalMemory, prestige, fidelity, transmission, or inheritance.
- Therefore **reward surgery, transmission, retention, and demes cannot move the current gate DV by
  construction.** The registry is already a global singleton (the population union), so the
  "max-pooling over the population" a transmission mechanism would buy already happens — there is no
  per-agent frontier left to pool.

**Honest consequence — two levers, both required, else nothing is testable:**

1. **Redesign the DV** so it measures adopted, transmitted, function-advancing value (not artifact
   geometry of the global registry). Only then *can* retention/transmission/demes act at all.
2. **Couple generation to the policy** (WHEN + WHAT to combine) and switch reward to functional
   value. This is the only lever that actually improves the measured **per-attempt efficiency**
   (0.135 vs 0.702 — 5.2× worse).

Everything else is an *amplifier* presupposing a ratchet that must first exist. A substrate that sits
*below* random has no ratchet to amplify.

## 2. Surviving, prioritized mechanism set

> Adversarial verdicts: **A** precondition (measurement reform); **C→B→D** the real lever
> (all `revise`, must be bundled & ordered); **E** + retention strictly subordinate; **F dropped**.
> Every mechanism's solo `would_flip_gate = no` — the flip, if any, is the *bundle* under the new DV.

### A — DV redesign FIRST (precondition; non-hot, golden untouched)
Retire `max_functional_depth` as the *sole* primary DV. New offline DVs in `research/metrics.py`
read the **realized, rebuilt frontier**:
- **DV1 `population_functional_value`** = Σ_cluster `adoption_c · v_c`, with per-task ceilings.
- **DV2 `accumulated_useful_depth`** (anti-churn core) = deep artifacts count only if they advanced a
  task-frontier no shallower artifact reached. Lower variance than the order-statistic max.
- **DV3 `transmitted_frontier_advances`** = clusters that *both* advanced a frontier *and* were
  adopted by ≥k agents / transmitted ≥1 hop.

Fixed, **pre-registered** task basis B = {edibility-given-hunger, heat, light, sharpness,
low-toxicity, scent}, identical weights/state for both arms; `func_tau` sweep {0.10,0.15,0.20} + k
sensitivity. **Anti-confound (mandatory):** pre-register the DV *before* the unblinded re-pilot
(swapping it after = confound laundering); **prove fairness on the existing n=12 pilot JSONs**
(`analyze_gate --compare-dvs`) that the recombiner already collapses on the new DVs (`uses=0`,
`discovered_by=-1`, no transmission). *Human basis:* cumulative-culture criterion (Tennie/Call/
Tomasello 2009; Henrich 2015) — adopted, transmitted, function-advancing improvement, not artifact count.

### C — Kill the novelty pump; reward realized + marginal functional value (hot; necessary hygiene, not gate-flipping)
Replace flat `NEW_DISCOVERY_BONUS=4.5` (`invention.py:85,164`) + rediscovery floor with a
value-gated bonus = `material_reward(new_vec, state) + RATCHET_GAIN · marginal`; neutralize inline
+0.5/+1.0 (`agent.py:1174,1184`) and `endocrine.apply_discovery` re-pumps — audit **all** reward
sites. *Forced fixes:* honest de-scope (the testable prediction is **discoveries-per-attempt
efficiency**, not registry depth vs recombiner); there is **no** per-agent `capability`/frontier
(reinventing per-agent state breaks determinism) → make the retained frontier a **tribe/lineage-level**
object that is explicitly transmitted (makes the ratchet a *social* object); keep `marginal`
**continuous & unbounded in count** (concave gains over many channels & recombination depth) to avoid
the [0,1]-clip saturation that silences the ratchet; freeze `RATCHET_GAIN` pre-pilot; rewrite
`tests/systems/test_invention_rewards.py`. *Human basis:* Boyd & Richerson (1985) guided variation;
Oudeyer & Kaplan (2007), Schmidhuber (1991) — intrinsic motivation as competence progress, provably
not novelty; Tennie et al. (2009).

### B — Policy-couple generation via `research_drive` (decide WHEN to invent) (hot; only after C)
Replace the fixed schedule (`agent.py:1170-1184`: `tick%3`, cooldown, `random()<inv_prob`) with a
learned gate: invent iff `brain_step['research_drive']` (action dim 6, currently computed at
`agent.py:1083` and **discarded**) > threshold. *Forced fixes (else "PPO learns when to invent" is
simply false):* **credit-assignment** — the whole-tick scalar reward can't isolate the gradient on
dim 6 → give `research_drive` its own Bernoulli head / auxiliary advantage term; **range bug** —
`research_drive` is tanh∈[-1,1] but THRESHOLD=0.4 assumes [0,1] → rescale `0.5·(tanh+1)`; **novelty
dependence** — only meaningful after C; deterministic bootstrap floor + stochastic ε (RNG via
`rng.seed_all`). *Human basis:* curiosity/research-drive as an intrinsic homeostatic need (Oudeyer &
Kaplan 2007; SDT); innovation as a deliberate, mechanism-limited act (Köhler 1925; Beck et al. 2011).

### D — Imagined means-ends selection of WHAT to combine (hot; the dominant efficiency lever)
Before each real combine: build a candidate set (top-N need-scored inputs + inventory `mat_XXXX`),
**predict** each (a,b,action) with a combine, score predicted need-fulfilment + `material_reward`,
execute only the argmax (seeded softmax). Replaces `random.choice` at `invention.py:126-129` and
`need_driven_invention.py:216-217`; removes the hardcoded affinity table (`_select_action_by_need`).
*Forced fixes (SEVERE, else it worsens the gate):* **compute-match self-destruct** — the imagined
evals must NOT go through the instrumented `combine_vectors` (the gate counts calls by function
identity and feeds `cc.n` to the recombiner budget; ~20-40 imagined evals/attempt would give the
recombiner 20-40× more attempts → bigger lead) → route prediction through a **separate pure
`_combine_pure()`**, compute-match on *executed* combines only, report planning overhead separately;
**planner-goal ≠ gate-goal** — add a novelty/under-explored (value-of-information) term so search
goes into new *and* need-relevant territory (else a greedy need-maximizer lowers depth/diversity);
**stochastic guard** — exclude/expectation-average the `rub` ignition branch. *Human basis:* Newell &
Simon (1972) means-ends; Battaglia/Hamrick/Tenenbaum (2013), Hegarty (2004) mental simulation;
Wolpert (1995) forward models; Gibson (1979) affordances. (Skeptic: `is_human_like: yes`.)

### E — Payoff-biased, high-fidelity transmission of valuable recipes (hot; only after A+D)
Store `output_value` on `CausalMemory.record`; payoff/prestige-weight `sample_for_transmission`;
gate observe/teach on a prestige score (windowed `last_reward` + lifetime high-value count); replace
the FIDELITY_BASE=0.72 global resample (which shatters deep recipes) with **local perturbation**
(keep structure, swap one leaf to its 12-dim nearest neighbour → fidelity above the Lewis-Laland
threshold but <1.0); payoff-rank vertical inheritance. *Forced fixes:* only testable after the DV
redesign; define `output_value` on full multidimensional utility + novelty-of-function (the
neutral-state scalar collapses to ~1-2D → mode-collapse); the generation clause (prefer recombining
deeper known `mat_XXXX`, with an exploration floor) is really part of D. *Human basis:* Henrich &
Gil-White (2001) prestige bias; Rendell et al. (2010); Lewis & Laland (2012) fidelity threshold;
Horner & Whiten (2005) overimitation; Csibra & Gergely (2009). (`is_human_like: yes`.)

### F — Partial-connectivity demes + knower-floor — **DROPPED**
Can't move the gate (global append-only registry); the "top-k most-used material" migrant re-injects
known materials into *more* random recombination (the very depth=volume artifact); demes against a
global registry *lower* each agent's input diversity. Reconsider **only** if a re-pilot shows learned
≥ random per-attempt, then evaluated per-deme with forgetting enabled. (Derex & Boyd 2016; Henrich
2004 — valid but irrelevant until a ratchet exists.)

## 3. The minimal re-pilot (binding order)

1. **Instrument first (research lane, read-only):** verify the learned arm persists non-zero `uses`
   + per-agent adopter sets; if not, DV1/DV3 floor the *learned* arm too → add read-only counters
   before trusting any verdict. *(Open risk: instrumentation gap. `uses` is currently polluted —
   `DiscoveryRegistry.get_vector:297` increments it; clean to a real `record_use` at homeostatic-
   relief sites.)*
2. **Land the new DVs** + run `analyze_gate --compare-dvs` on the existing n=12 JSONs → (a) fairness
   proof (recombiner collapses already), (b) pick the lowest-variance DV, re-pre-register with `func_tau`/k.
3. **Land C → B → D** (order binding: B needs C; D needs the compute-match fix). Hot files via
   core-lead; all new RNG via `rng.seed_all`; golden goes **intentionally RED**, core-lead regenerates once.
4. **Land E** (retain-by-use with *clean* `uses`; payoff-biased transmission with non-degenerate
   `output_value`). Note: CausalMemory eviction is *already* value-ranked and inventory compaction
   *already* quantity-ranked — don't redo those.
5. **Re-run** at identical pre-registered seeds/ticks/grid/pop. **Do NOT retrofit the null** (its
   disadvantage IS retention+transmission+max-pooling). Compute-match on *executed* combines.
6. **Apply the gate rule on DV1-3.** Verify discoveries-per-attempt rose, `knockout required_frac`
   rose, paired-diff CI moves from −9.4 toward >0 (target: BORDERLINE_LEAN_A → PATH_A).
7. **Ablations:** full / high-fidelity+value-blind / value-biased+low-fidelity (Lewis-Laland predicts
   collapse) / teaching-off. Demes (F) stay out until (i) learned ≥ random per-attempt.

## 4. What stays at risk (honest uncertainty)

- Reward-blind gate is the central single-point-of-failure: if the DV redesign or `uses`/adopter
  instrumentation doesn't land cleanly, the learned arm floors too and the re-pilot is uninformative
  (Step 1 not optional).
- Pre-registration integrity: swapping the DV post-unblinding = confound laundering → the
  existing-data fairness proof must convince *before* the re-pilot, else Path B (methodology paper)
  stays the honest option.
- Compute-match self-destructs without the `_combine_pure` split; PPO won't learn "when to invent"
  without the dim-6 credit path; a pure need-maximizer can lower diversity (novelty/VoI term is a new
  Goodhart surface); the marginal ratchet can saturate under [0,1] clipping; fidelity must clear the
  Lewis-Laland threshold yet stay <1.0; Rogers-paradox/copy-trap can plateau under the recombiner
  ceiling; embodiment overhead may still mean fewer total attempts — the win must come from
  per-attempt efficiency + max-pooling.

**Overall:** viable, but the center of gravity moved. The lever is **not** primarily
reward/transmission/demes (four are provably inert vs the old gate) — it is **DV redesign (A) +
generation coupling with a fixed compute-match and credit path (C→B→D)**. F is cut; E and retention
are strictly subordinate and only act bundled with A+D. Even with perfect execution it is open
whether the learned arm drives per-attempt efficiency far enough to lift −9.4 above the null — the
first target is BORDERLINE_LEAN_A, not a guaranteed PATH_A.
