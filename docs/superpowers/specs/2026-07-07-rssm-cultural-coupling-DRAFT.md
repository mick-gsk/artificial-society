# RSSM Cultural Coupling — Design Spec

> **Status: reviewed by 2 tailored review agents 2026-07-07; findings
> incorporated.** This document is the follow-on to the RSSM Dreamer brain
> ([2026-07-04-rssm-dreamer-brain-design.md](2026-07-04-rssm-dreamer-brain-design.md)).
> The review round corrected two false premises (the knowledge write-side is
> **live**, and emergence-metrics **is landed**), re-staged the rollout around a
> Stage-0 measurement of the already-live channel, and promoted a spread/depth
> metric to primary. Remaining open items are in §11 — do not treat them as
> decided.

## 1. Context & Goal

The RSSM lane (arms A/B/C) is proven: imagination-trained actors beat the v1
baseline (+6% survival RMST, 3/3 seeds) and beat the WM-as-MPC ablation (3/3).
The learning mechanism works; the effect is small. The project goal is
**open-ended emergence** — learned tool use, building, proto-language, never
scripted — and the standing hypothesis (M1 diagnosis) is that the bottleneck is
the *learning machinery and its coupling*, not world mechanics. Population-level
intelligence amplifying individual learning (culture) is the most plausible
amplifier we have not yet measured on the rssm path.

**What is actually true today (corrected from the draft's premises):**

- The **knowledge channel is already live end-to-end for rssm agents.** rssm
  requires `physics_v2=False` ([simulation.py:160-161](../../../artificial_society/simulation.py#L160)),
  so rssm agents take the **v1 invention path**, whose functions write causal
  facts: `agent_try_invention` calls `causal_memory.record(...)`
  ([invention.py:176](../../../artificial_society/systems/invention.py#L176),
  also [:248](../../../artificial_society/systems/invention.py#L248)) and
  `agent_invent_from_need` does likewise
  ([need_driven_invention.py:424](../../../artificial_society/systems/need_driven_invention.py#L424)).
  Both are invoked inside the `if not self.physics_v2:` blocks at
  [agent.py:1487-1502](../../../artificial_society/agents/agent.py#L1487).
  Neither function touches `.brain`, so both are safe for `brain=None`.
  Empirically confirmed by R1: a headless rssm sim (8 agents, 80 ticks) leaves
  **every survivor with a non-empty `causal_memory.sequences`** (4–10 entries,
  63 total). **Dims 34–36 grow from tick 0 — they are not near-constant.**
- The **transmission and read paths are live too.** `social_learning_step` runs
  for rssm agents ([agent.py:1484-1485](../../../artificial_society/agents/agent.py#L1484),
  `tick % 3`); its `receive_transmitted` / `sample_for_transmission` paths carry
  causal knowledge between co-located agents, and `feature_vector()` splices the
  result into obs **dims 34–36** ([agent.py:524](../../../artificial_society/agents/agent.py#L524),
  [:561](../../../artificial_society/agents/agent.py#L561)) which `learner.act`
  encodes into the WM. Only the **neural weight-imitation branch** (`imitate_from`)
  no-ops for rssm, and it does so *safely* via a `getattr` guard on
  `agent.brain`/`other.brain` ([social_learning.py:95-98](../../../artificial_society/systems/social_learning.py#L95)).
- **Emergence-metrics is landed.** `systems/emergence_metrics.py` exists and is a
  registered tick hook (order 68, [emergence_metrics.py:310](../../../artificial_society/systems/emergence_metrics.py#L310)).
  It records `culture_max_spread` ([:283](../../../artificial_society/systems/emergence_metrics.py#L283)),
  `culture_distinct_seqs`, `culture_shared_seqs`, `novel_causal_keys`,
  `tech_capabilities` ([:292](../../../artificial_society/systems/emergence_metrics.py#L292)),
  and `lang_converged` ([:256](../../../artificial_society/systems/emergence_metrics.py#L256)).
  `culture_max_spread` is computed directly from the population's
  `causal_memory.sequences` — the count of agents sharing the most-widespread
  causal key ([:262-283](../../../artificial_society/systems/emergence_metrics.py#L262)).

**Consequence for this design:** the cheapest "make the channel live" work
(the draft's K1 write hook) is **not needed and would double-write** — a second
`record()` on the reward path duplicates what invention already writes. So the
first job is not to *build* the knowledge channel but to **measure** it, then to
add the one channel that is genuinely missing on the rssm path — a **local,
observation-gated policy channel** — under a control that proves it carries
*cultural content*, not just generic regularization.

**Goal:** *diegetic* (in-world, local, observable) cultural transmission on the
rssm path, flag-gated as a separate experiment, measured against controls that
isolate cultural *content*, so that a population's transmitted traits spread
**further and are sourced from successful demonstrators** — without adding
another non-diegetic shortcut.

### 1.1 The central tension: what is already non-diegetic, and why it matters

The RSSM lane's `PrototypeActor` warm-start **is already non-diegetic culture.**
A newborn's slab row is initialized from a single global EMA of the population's
actors ([learner.py:69-77](../../../artificial_society/agents/rssm/learner.py#L69),
[actor_critic.py:93](../../../artificial_society/agents/rssm/actor_critic.py#L93)),
and the shared world model is a global knowledge conduit fed by pooled experience.
The RSSM spec is explicit that this lane is **not** evidence of emergent social
learning — a shared WM "short-circuits exactly that story"
([RSSM spec §3](2026-07-04-rssm-dreamer-brain-design.md)). Bolting more global
parameter-averaging onto that would be *more* of the same shortcut, not culture.

So this design must state, up front, **what stays and what the emergence story
requires:**

| Mechanism | Diegetic? | Disposition in the coupling lane |
|---|---|---|
| Shared RSSM world model | No (global conduit) | **Stays.** It is the substrate; the A/B already prices it in. Not "culture" we claim. |
| `PrototypeActor` EMA warm-start | No (global average) | **Stays as the *control*.** It is the null we must beat: "born into population skill" via a global average. The coupling lane's job is to show a *local, observable* channel adds signal **on top of** (or replacing) it. |
| `CausalMemory` transmission (observe/teach at radius 2) | **Yes** (local, proximity-gated) | **Already live for rssm** (§1, §4.2). Stage 0 measures it; no new write code. |
| Imagination-level imitation from an **observed, observably-successful neighbor** | **Yes** (requires proximity + a real observation event + a *diegetic* success cue) | **New.** This is the policy channel whose emergence we can honestly claim (Stage 1). |
| Slab-row parameter copy from an observed neighbor | **Yes** (local) but weight-space-blunt | **Exploratory probe only** (§4.1). Homogenization + non-identifiability risk; not the claimed channel. |
| Proto-language tokens marked in the world | **Yes** (physical token objects) | **New, largest lift** — currently ungrounded in *every* policy (Stage 2, §4.3). |

**Design principle:** every channel we *claim* as culture must be **local**
(proximity- or artifact-mediated, not a global average), **observable** (mediated
by a real in-world event — a neighbor being nearby, a token existing in a cell),
and gated on a **diegetically observable success cue** (visible energy / health /
materials / age — never a privileged read of another agent's cumulative reward or
policy; §4.1, F4). The prototype warm-start meets none of that bar and is
therefore kept as a **baseline/control**, not counted as an emergence result.

## 2. Fixed Constraints (inherited — do not relitigate)

1. **No Lamarckian genetic inheritance.** No parent→child weight copy. (RSSM spec
   §2.5 — cultural warm-start is from the *prototype*, never a parent.) Any
   policy-transmission channel here is between **contemporaneous, co-located**
   agents (peer imitation), not birth inheritance.
2. **Substrate is the v1 obs/action space:** `INPUT_SIZE=57`, `ACTION_SIZE=7`
   ([brain.py:24-56](../../../artificial_society/agents/brain.py#L24)). Changing
   `INPUT_SIZE` breaks the v1 golden (§5) — a hard constraint on the language
   channel.
3. **rssm code never touches the global RNG streams.** All rssm randomness draws
   from rssm-owned `torch.Generator`s seeded from `(global_seed, …)` via the
   existing `make_generator` scheme (RSSM spec §6). Every new stochastic decision
   here obeys the same rule.
4. **Determinism / golden is sacred.** The v1 default arm must stay byte-identical;
   off-flag construction draws zero extra RNG (RSSM spec §6, CLAUDE.md invariants).
5. **Hot files are a frozen contract.** `agents/agent.py`, `agents/brain.py`,
   `simulation.py` route through core-lead. This lane's edits to them must be
   minimal, flag-gated, and flagged at merge (RSSM spec §11). With the K1 write
   hook dropped (§1), the only remaining hot-file touch is a read-only
   **transmission-disable ablation flag** (Stage 0, §4.2).

## 3. Non-Goals / Scope

- **Not** re-opening the A/B (arms A/B/C). This is a **new arm/flag** measured
  against arm B as its control.
- **Not** touching v1/v2 brains, their tests, or `imitate_from`'s v1 behavior.
- **Not** claiming the *prototype warm-start* as emergent culture — it is the
  control (§1.1).
- **Not** adding a knowledge write hook. The knowledge channel is already live via
  the invention paths (§1); a second `record()` would double-write.
- **Not** a rewrite of the obs layout for v1. Language grounding, if pursued,
  uses an rssm-only obs extension (§4.3), never a change to the shared 57-dim
  vector that v1's golden pins.
- **Not** full GPU-resident acting — the act path stays batch-1 CPU (RSSM spec §8).
- **Not** a claim of proto-language or open-ended innovation. See §6.1 ("What this
  can and cannot claim").

## 4. Mechanism options with trade-offs

Two channels are in scope (knowledge = measure-only; policy = the new build);
language is deferred to Stage 2. Each is separately flag-gateable so the staged
rollout (§5) can turn them on one at a time.

### 4.1 Policy transmission (the new channel)

There is **no** live policy channel for rssm agents: `imitate_from` no-ops on
`brain=None` ([social_learning.py:95-98](../../../artificial_society/systems/social_learning.py#L95)),
and the slab rows are only culturally connected through the global prototype EMA
(non-diegetic). This is the one genuinely-missing local channel, and it is the
scientifically meaningful target.

**A diegetic success gate, not a reward read (F4).** The v1 `imitate_from` gate
keys on `other_reward > agent_reward * IMITATION_SUCCESS_RATIO`
([social_learning.py:88-99](../../../artificial_society/systems/social_learning.py#L88)).
Cumulative reward is **not observable in-world** — importing it as the gate would
smuggle a privileged read and dissolve the very distinction between a local
channel and the global prototype average. Replace it with a **diegetically
observable success proxy**: visible energy / health / accumulated materials / age
of the neighbor. Only that keeps "local + observably-successful pairwise"
categorically different from "unconditional global average."

**Option B — imagination-level imitation (behavioral cloning in imagined states)
— THE CLAIMED CHANNEL.**
When agent *i* observes demonstrator *j* nearby and *j* is more successful **on
the diegetic proxy**, record *j*'s **previous-tick executed action** (surfaced
from world/neighbor state, never `j.policy` — F5/Q3) into a small per-observer
"demo" buffer, keyed by the WM-encoded state where it was observed. During *i*'s
imagination pass ([learner.py `_imagination_pass`](../../../artificial_society/agents/rssm/learner.py)),
add an auxiliary loss pulling *i*'s actor toward the demonstrated action in that
encoded state: `L_imit = α · E[ ‖ a_i(s) − a_j_demo ‖² ]` over demo states.
- *Pros:* Semantically correct — imitates *behavior conditioned on situation*, not
  raw weights, so it survives weight-space non-identifiability. Composes with the
  existing actor loss and the return-normalized objective. The diegetic-proxy gate
  controls *what enters the demo buffer* (local + observable). This is the version
  whose positive result is scientifically meaningful.
- *Cons:* More moving parts (a demo buffer keyed per observer, its own determinism
  surface, its own eviction). Extra compute in the hot training hook. Requires the
  previous-executed-action observable to be surfaced from neighbor state.

**Option A — slab-row parameter copy — EXPLORATORY PROBE ONLY (demoted, F3).**
On a triggered imitation event, blend the observer's actor params toward the
demonstrator's row:
`params[k][slot_self] ← (1−α)·params[k][slot_self] + α·params[k][slot_other]`
for actor keys, using the existing per-row machinery
([actor_critic.py:120 `snapshot_slot`](../../../artificial_society/agents/rssm/actor_critic.py#L120),
`ACTOR_KEYS`; direct tensor-index writes are the established idiom and leave Adam
moments untouched — R1). Cheap, but a raw param blend is blunt: two rows can
encode near-identical policies with different weights (permutation/scale), so
linear blending can *degrade* both, and it amplifies the homogenization guardrail
(RSSM spec §11). **Kept only as an optional exploratory probe, never the claimed
channel.** If run, refresh the CPU act-mirror for the touched slot
([`refresh_actor_mirror_slot`](../../../artificial_society/agents/rssm/actor_critic.py#L171)).

**Option C — status quo (prototype only).** The control (arm B); not a candidate.

**Lean:** **Option B is the channel we build and claim.** Option A is an
exploratory probe measured only if it is cheap to include; it is not on the
critical path.

### 4.2 Knowledge transmission (CausalMemory) — measure, do not build

`CausalMemory` is **agent-side and brain-agnostic** — `ensure_fields` creates it
for **every** arch including rssm ([agent.py:165-166](../../../artificial_society/agents/agent.py#L165)).
As established in §1, on the rssm path the **write** side is live (invention
functions call `record`), the **transmission** side is live
(`social_learning_step` + `receive_transmitted`/`sample_for_transmission`), and
the **read** side is live (dims 34–36 → `learner.act`). The channel is
**live end-to-end**. There is nothing to build here.

The Stage-0 work is therefore **measurement + a causal-contribution ablation**:

- **K0 — quantify the live channel (Stage 0).** On existing tuning/confirmatory
  rssm runs, read out `emergence_metrics`: `culture_max_spread`,
  `culture_distinct_seqs`, `culture_shared_seqs`, `novel_causal_keys`, plus a
  count of non-zero `sample_for_transmission` events (transmission liveness).
  Establishes the baseline the policy channel must beat and confirms dims 34–36
  are non-inert. No new mechanism.
- **K-ablation — transmission-disable flag (Stage 0).** Add a flag-gated switch
  that **suppresses the causal transmission step** (skip `receive_transmitted` /
  short-circuit `social_learning_step`'s causal-transfer branch for rssm) so the
  knowledge channel's *causal* contribution to RMST and to `culture_max_spread`
  can be measured by difference (transmission-on vs transmission-off). This is a
  **read-only ablation** — it disables an existing path, it does not add a write.
  It is one flag-gated branch; the guard is a hot-file line flagged to core-lead.
- **K2 — campfire pool for rssm.** `campfire_knowledge_sharing`
  ([social_learning.py:104](../../../artificial_society/systems/social_learning.py#L104))
  pools `KnowledgeGraph` facts at fire cells, but rssm agents have no
  `KnowledgeGraph` (`getattr(a, "knowledge", None)` → None → skipped). Deferred —
  group sharing is a stronger claim best made after the bilateral policy channel
  works.

### 4.3 Language (proto-tokens) — Stage 2

**Confirmed by review:** proto-language tokens are **not in the obs vector at
all.** `local_features` ([agent.py:527-565](../../../artificial_society/agents/agent.py#L527))
has no token channel; **obs dims 15–16 are `nearby_count` and `friends_count`**
([brain.py:28](../../../artificial_society/agents/brain.py#L28),
[agent.py:544-545](../../../artificial_society/agents/agent.py#L544)) — plain
social-density scalars, **not** token channels. The RSSM decoder's down-weighting
of 15–16 (RSSM spec §4.1) concerns social-nearby noise and has nothing to do with
language. `_maybe_mark_language` / `_observe_tokens` run for all agents
([agent.py:1537-1539](../../../artificial_society/agents/agent.py#L1537)) and
populate `TokenMemory`/`TOKEN_WORLD`, but nothing feeds a token percept back into
the 57-dim observation. **Language is currently ungrounded in *every* policy.**

**Options:**
- **L1 — defer (Stage 0–1).** Grounding requires a **new obs channel**; because
  `INPUT_SIZE=57` is pinned by the v1 golden (§2.2), it must be an **rssm-only obs
  extension**, never appended to the shared 57-dim vector. Real design (new dims,
  WM encoder width for the rssm arm, new decoder group, determinism for the token
  draw) — not on the Stage-0/1 critical path.
- **L2 — rssm-only token-percept tail (Stage 2).** Add `token_feats` (fixed small
  width) to what `learner.act` encodes, sourced from `TOKEN_WORLD.tokens_at` + the
  observer's `TokenMemory`. Fully diegetic; largest lift; strongest emergence
  claim. Pursued only if Stage 1 shows a population effect.
- **L3 — reward-shaped language only.** Rejected as a *culture* claim: shapes
  behavior without the agent perceiving symbols, so no communication channel
  emerges.

## 5. Recommendation & staged rollout

Measure the already-live channel first; then add the one genuinely-missing local
channel under a content control; then, only on signal, the expensive grounded
language.

- **Stage 0 — measure the live knowledge channel (no new mechanism).**
  Read out emergence-metrics (`culture_max_spread`, transmission liveness, etc.)
  on existing rssm runs (K0), and add the **transmission-disable ablation flag**
  (K-ablation) to measure the knowledge channel's causal contribution by
  difference. Deliverable: a baseline spread/liveness profile and a
  transmission-on-vs-off contrast. No policy code yet.
- **Stage 1 — policy channel: imagination-level BC (Option B).** Behavioral
  cloning in imagined states from an **observably-successful** neighbor (diegetic
  proxy gate, §4.1/F4), **with a scrambled-donor control** (§6/F1). Slab-row copy
  (Option A) is an optional exploratory probe only.
- **Stage 2 — grounded language (L2).** rssm-only obs-tail extension. Only if
  Stage 1 shows a population effect worth the obs-extension cost.
- **Throughout:** the `PrototypeActor` warm-start stays **on** as the control
  substrate; an optional sub-arm turns it *off* (P−proto) to test whether the
  local channel substitutes for the global average — but this ranks **below** the
  scrambled-donor control (§6/F1).

## 6. Experiment design sketch

**Question:** does a *local, observable, success-gated* cultural channel increase
the **spread and successful-sourcing of transmitted traits** beyond the
non-diegetic prototype warm-start — and beyond a content-free regularizer?

- **Arms:**
  - **B (control):** full RSSM, prototype warm-start on, `culture_coupling=off`.
    (Exactly the proven arm B.)
  - **B−trans (Stage-0 ablation):** B with causal transmission **disabled**
    (K-ablation, §4.2) — isolates the live knowledge channel's contribution.
  - **B+P (Stage 1):** B + policy transmission (Option B) from an observably-
    successful neighbor.
  - **B+P−scram (killer null, F1):** identical to B+P but the demo donor is a
    **randomly chosen** agent, not an observably-successful observed neighbor.
    This holds the *amount* of extra imitation signal fixed and removes only its
    *cultural content* (successful-source selection + observation-gating).
    **Culture is claimed only if B+P beats B+P−scram**, not merely if B+P beats B —
    otherwise the effect is a generic regularizer. Outranks P−proto.
  - *(optional)* **P−proto:** B+P with prototype warm-start **off**, testing
    whether the local channel substitutes for the global average.
- **Pairing / regime:** inherit the RSSM A/B harness — paired seeds, shared
  founders/terrain, **transitions-matched** primary axis (`transitions_stored`),
  dual sample-and-compute reporting, device assertion (RSSM spec §10.2). Cultural
  coupling changes behavior → environments diverge → same "pairing buys variance
  reduction, not identity" caveat applies.
- **Endpoints (primary — spread/depth, F2):** `culture_max_spread` and
  transmission-chain depth from `emergence_metrics`
  ([emergence_metrics.py:283](../../../artificial_society/systems/emergence_metrics.py#L283)).
  Does the local channel increase how far a transmitted trait spreads through the
  population, relative to the prototype and scrambled-donor controls?
  (Transmission-chain depth: if not directly emitted today, derive it from the
  transmission-event log / `novel_causal_keys` provenance; wire it as a first-class
  metric under Stage 0.)
- **Endpoint (co-primary — fitness, F2):** survival RMST (same as A/B) — the local
  channel must not lift spread at the cost of survival.
- **Endpoints (mechanism / emergence, secondary):**
  - *Successful-source selection:* fraction of accepted imitations whose donor was
    above the population median on the diegetic proxy (must exceed chance; the
    direct discriminator vs. scrambled-donor).
  - *Competence-vs-age*, split by `spawn_origin` — do agents born into a
    culturally-active population cross competence **sooner**? (RSSM spec §11.)
  - *Transmission liveness:* count of non-zero `sample_for_transmission` events and
    causal-fact population per capita (Stage-0 baseline; must stay >0).
  - *Diversity guardrail (F5/Q4):* pre-register that **action-distribution
    diversity must not fall below the prototype control while capability-spread
    rises**. A channel that just collapses to one policy is not culture.
- **Pre-registration:** commit thresholds and the stopping rule (as the RSSM lane
  did) before the multi-seed run; 3-seed pilot → power calc → N.
- **Falsifiable null:** if B+P does not beat **B+P−scram** on `culture_max_spread`
  / chain depth (and shows no successful-source selection above chance), the
  channel adds no cultural content — report and stop (mirrors the RSSM lane's
  honesty about small effects).

### 6.1 What this can and cannot claim

Stated honestly, in R2-F2's language:

**Can claim (if the controls pass):** *"A local, observation-gated channel
increases transmitted-trait spread beyond a global-average control, and selects
successful sources."* Concretely: relative to the non-diegetic prototype EMA
(B) and to a content-free regularizer (B+P−scram), a proximity- and
success-gated policy channel raises `culture_max_spread` / transmission-chain
depth without collapsing action diversity, and its accepted imitations are
disproportionately sourced from observably-successful demonstrators.

**Cannot claim:** this is **NOT proto-language** (language stays ungrounded until
Stage 2, §4.3), **NOT open-ended / cumulative innovation** (we measure spread and
sourcing of *existing* causal traits, not the invention of new ones), and **NOT**
evidence that the shared world model is unnecessary (the WM remains a global
conduit that may render the channel partly redundant — §11/Q7). A positive result
is a bounded, honest claim about *social transmission dynamics*, not about
emergent culture writ large.

## 7. Integration seams (all behind the flag)

- **`RSSMConfig`:** add `culture_coupling: str = "off"` (values `off` | `policy`)
  plus per-channel knobs (`imit_alpha`, `imit_trust_min`, `imit_success_proxy ∈
  {energy, health, materials, age}`, `imit_mode ∈ {imagination, slab_probe}`,
  `imit_donor ∈ {successful, scrambled}`) and, **orthogonally**, a separate
  `disable_causal_transmission: bool = False` for the Stage-0 ablation (kept
  independent so B−trans composes with `culture_coupling="off"`). Defaults
  preserve arm B exactly.
- **Policy channel (learner-side):** a demo-buffer + aux loss in
  `_losses`/`_imagination_pass` (Option B); an optional `blend_row` probe
  (Option A). Own generator for the stochastic imitation trigger and donor draw
  (`(global_seed, "imit", id)` / `(global_seed, "scram", id)`), never the global
  stream. The scrambled-donor variant draws its donor from the same generator so
  B+P and B+P−scram are seed-comparable.
- **Knowledge channel:** *no write code.* Stage 0 is measurement (read
  `emergence_metrics`) plus the **transmission-disable ablation** — one flag-gated
  branch guarding the causal-transfer step, flagged to core-lead.
- **`social_learning_step` reuse:** its causal observe/teach already runs for rssm
  (§4.2); the weight-imitation branch stays a v1-only no-op. The rssm policy
  channel is driven from the SharedLearner side, **not** by extending
  `social_learning_step` (keeps the systems-lane file v1-pure).
- **Language (Stage 2 only):** rssm-only obs tail in `learner.act`, never a change
  to `local_features`/`INPUT_SIZE`.

## 8. Determinism & RNG

- Every new stochastic decision (imitation trigger, donor selection incl.
  scrambled, demo sampling) draws from an **rssm-owned generator** seeded via the
  existing `make_generator` scheme from `(global_seed, tag, agent_id)` — never
  `random`/`numpy`/`torch` globals (Constraint §2.3; RSSM spec §6). Per-agent
  seeding keeps draws independent of iteration order under births/deaths.
- **Off-flag = zero extra draws.** With `culture_coupling="off"`, no generator is
  constructed and no branch is entered; arm B stays byte-identical. Add the
  guard-style unit assertion the RSSM lane uses: an off-flag `Simulation.__init__`
  leaves `torch.get_rng_state()` unchanged, and the golden digest does not move.
- The **transmission-ablation** flag must itself be deterministic and off-by-
  default: disabling a path removes work but draws no new RNG; the golden is
  unaffected while `culture_coupling="off"`.
- Determinism test **across births/deaths** for each channel (same seed → identical
  transmission/imitation events), matching the RSSM lane's birth/death test.

## 9. Fixed-Constraints checklist (self-audit)

- [x] No parent→child weight copy (peer imitation only; §2.1).
- [x] `INPUT_SIZE=57` unchanged; language tail is rssm-only (§2.2, §4.3-L1).
- [x] rssm-owned generators only; globals untouched (§8).
- [x] v1 golden unchanged; off-flag zero-draw (§8).
- [x] Hot-file edits minimal + flagged to core-lead — now only the transmission-
      ablation guard; the K1 write hook is dropped (§1, §4.2, §10).
- [x] Prototype warm-start kept as control, not claimed as culture (§1.1).
- [x] Cultural *content* isolated by a scrambled-donor control, not just channel
      presence (§6/F1).
- [x] Success gate is a diegetic proxy, not a reward/policy read (§4.1/F4).

## 10. Hot-file budget

| File | Edit | Size |
|---|---|---|
| `agents/agent.py` | **none** — no K1 write hook (channel already live, §1) | 0 |
| `simulation.py` / `agent.py` | one flag-gated branch to **disable** causal transmission (K-ablation, read-only) | ~1–3 lines |
| `agents/rssm/config.py` | `culture_coupling` + knobs | new fields (not hot) |
| `agents/rssm/actor_critic.py` | optional `blend_row` probe (rssm-owned) | new (not hot) |
| `agents/rssm/learner.py` | demo buffer + aux loss (Option B), donor draw | new (not hot) |
| `agents/brain.py` | **none** in stages 0–1 (language tail is learner-side) | 0 |

The hot-file surface is **smaller than the draft's**: the K1 `agent.py` write line
is gone. The only frozen-contract touch is the transmission-disable ablation
guard; everything else lives in the isolated `agents/rssm/` tree.

## 11. Open questions (for the review-agent round → carried forward)

Resolved by this round (kept for traceability):
- **~~Q1 (K liveness)~~ — RESOLVED.** The knowledge write side is **live** via
  the invention paths (§1); K1 is dropped, replaced by measurement + a
  transmission-disable ablation (§4.2).
- **~~Q2 (A vs B)~~ — RESOLVED.** Option B (imagination BC) is the claimed
  channel; Option A is an exploratory probe only (§4.1/F3).
- **~~Q3 (demo observable)~~ — RESOLVED.** Previous-tick executed action, surfaced
  from world/neighbor state, never `j.policy` (§4.1/F5).

Still open:
- **Q4 (diversity floor).** Exact pre-registered threshold below which
  action-distribution diversity signals "collapsed to one policy" rather than
  "learned from neighbors" (§6 guardrail).
- **Q5 (proxy choice).** Which diegetic success cue (energy / health / materials /
  age, or a composite) best matches the intended "observably successful"
  semantics without leaking hidden state.
- **Q6 (chain-depth metric).** Is transmission-chain depth directly derivable from
  the current transmission-event log, or does it need a first-class emitter added
  to `emergence_metrics.py` under Stage 0?
- **Q7 (WM redundancy).** Because the shared WM already pools all experience, the
  policy/knowledge channels may be partly redundant with it. A WM-frozen isolation
  sub-condition is a **follow-up, not pilot-blocking** (F5); report transmission
  liveness meanwhile.
- **Q8 (language sub-spec).** L2 obs-tail width, encoder-width handling for the
  rssm-only extension, and decoder grouping for token percepts — full sub-spec
  deferred to Stage 2.
