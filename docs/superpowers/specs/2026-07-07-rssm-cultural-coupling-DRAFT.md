# RSSM Cultural Coupling — Design Spec (DRAFT)

> **Status: DRAFT.** This document is the follow-on to the RSSM Dreamer brain
> ([2026-07-04-rssm-dreamer-brain-design.md](2026-07-04-rssm-dreamer-brain-design.md)).
> It goes through a tailored review-agent round before any plan is written. Open
> questions are collected in §11 — do not treat unresolved items as decided.

## 1. Context & Goal

The RSSM lane (arms A/B/C) is proven: imagination-trained actors beat the v1
baseline (+6% survival RMST, 3/3 seeds) and beat the WM-as-MPC ablation (3/3).
The learning mechanism works; the effect is small. The project goal is
**open-ended emergence** — learned tool use, building, proto-language, never
scripted — and the standing hypothesis (M1 diagnosis) is that the bottleneck is
the *learning machinery and its coupling*, not world mechanics. Population-level
intelligence amplifying individual learning (culture) is the most plausible
amplifier we have not yet connected to the rssm path.

**The cultural layer is v1-only.** Three systems exist, all built for the v1
`Brain`:

- `systems/social_learning.py` — `social_learning_step` (observe/teach causal
  sequences + neural weight imitation).
- `systems/culture.py` — `CausalMemory` (per-agent causal-sequence store, with
  `receive_transmitted` for imperfect transmission; feeds obs dims 34–36).
- `systems/language.py` — proto-tokens (`TokenMemory`, `TOKEN_WORLD`, mark/observe).

For an rssm agent, the policy lives in an `ActorCriticSlab` **row** indexed by
`agent.rssm_slot` — there is **no `agent.brain`**. The one culturally-active
piece today, `imitate_from`, copies v1 `Brain` weights and **silently no-ops**
for rssm agents (guarded on `agent.brain`/`other.brain` both being non-`None`,
[social_learning.py:95-98](../../../artificial_society/systems/social_learning.py#L95)).

**Goal:** *diegetic* (in-world, local, observable) cultural transmission on the
rssm path, flag-gated as a separate experiment, so that a population learns
faster/further **together** than the sum of individuals — without adding another
non-diegetic shortcut.

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
| Slab-row / imagination imitation from an **observed neighbor** | **Yes** (requires proximity + a real observation event) | **New.** This is the thing whose emergence we can honestly claim. |
| `CausalMemory` transmission (observe/teach at radius 2) | **Yes** (local, proximity-gated) | **New for rssm** (already coded, but the write side is dead — §4.2). |
| Proto-language tokens marked in the world | **Yes** (physical token objects) | **New, largest lift** — currently ungrounded in *every* policy (§4.3). |

**Design principle (proposed):** every channel we *claim* as culture must be
**local** (proximity- or artifact-mediated, not a global average) and
**observable** (mediated by a real in-world event — an agent being nearby, a
token existing in a cell). The prototype warm-start does not meet that bar and is
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
   from rssm-owned `torch.Generator`s seeded from `(global_seed, …)`
   (RSSM spec §6). Every new stochastic decision here obeys the same rule.
4. **Determinism / golden is sacred.** The v1 default arm must stay byte-identical;
   off-flag construction draws zero extra RNG (RSSM spec §6, CLAUDE.md invariants).
5. **Hot files are a frozen contract.** `agents/agent.py`, `agents/brain.py`,
   `simulation.py` route through core-lead. This lane's edits to them must be
   minimal, flag-gated, and flagged at merge (RSSM spec §11).

## 3. Non-Goals / Scope

- **Not** re-opening the A/B (arms A/B/C). This is a **new arm/flag** measured
  against arm B as its control.
- **Not** touching v1/v2 brains, their tests, or `imitate_from`'s v1 behavior.
- **Not** claiming the *prototype warm-start* as emergent culture — it is the
  control (§1.1).
- **Not** a rewrite of the obs layout for v1. Language grounding, if pursued,
  uses an rssm-only obs extension (§4.3), never a change to the shared 57-dim
  vector that v1's golden pins.
- **Not** full GPU-resident acting — the act path stays batch-1 CPU (RSSM spec §8).

## 4. Mechanism options with trade-offs

Three channels, evaluated independently. Each is separately flag-gateable so the
staged rollout (§5) can turn them on one at a time.

### 4.1 Policy transmission

The v1 analog is `imitate_from` — neural weight nudging toward a more successful,
trusted neighbor, gated on `other_reward > agent_reward * IMITATION_SUCCESS_RATIO`,
`trust >= IMITATION_MIN_TRUST`, curiosity-scaled probability
([social_learning.py:88-99](../../../artificial_society/systems/social_learning.py#L88)).
We want the same *decision rule*, retargeted to slab rows.

**Option A — slab-row imitation (copy/blend a neighbor's actor row).**
On a triggered imitation event, blend the observer's actor params toward the
demonstrator's row:
`params[k][slot_self] ← (1−α)·params[k][slot_self] + α·params[k][slot_other]`
for actor keys, using the existing per-row machinery
([actor_critic.py:120 `snapshot_slot`](../../../artificial_society/agents/rssm/actor_critic.py#L120),
`ACTOR_KEYS`, `acquire_slot`'s per-row copy at
[:93-115](../../../artificial_society/agents/rssm/actor_critic.py#L93)).
- *Pros:* Direct analog of `imitate_from`; cheap (one blend, no gradient); reuses
  the trust/success gate verbatim; **local + observable** → diegetic. Adam-moment
  hygiene already understood (must **not** copy moments; treat like a partial
  re-init and let per-slot Adam adapt — mirror the slot-hygiene rule, RSSM spec §8).
- *Cons:* Slab rows are only comparable because they share the WM's state space —
  fine here (WM is shared), but a raw param blend is a blunt instrument: two rows
  can encode near-identical policies with different weights (permutation/scale),
  so linear blending can *degrade* both. Risk of homogenization (already a tracked
  guardrail, RSSM spec §11) is amplified. Needs the CPU act-mirror refreshed for
  the touched slot ([`refresh_actor_mirror_slot`](../../../artificial_society/agents/rssm/actor_critic.py#L171)).

**Option B — imagination-level imitation (behavioral cloning in imagined states).**
When agent *i* observes demonstrator *j* nearby and *j* is more successful, record
*j*'s recent (obs→action) pairs into a small per-observer "demo" buffer. During
*i*'s imagination pass ([learner.py `_imagination_pass`](../../../artificial_society/agents/rssm/learner.py)),
add an auxiliary loss pulling *i*'s actor toward the demonstrated action **in the
WM-encoded state where it was observed**:
`L_imit = α · E[ ‖ a_i(s) − a_j_demo ‖² ]` over demo states.
- *Pros:* Semantically correct — imitates *behavior conditioned on situation*, not
  raw weights, so it survives weight-space non-identifiability. Composes naturally
  with the existing actor loss and the return-normalized objective. Trust/success
  gate controls *what enters the demo buffer* (still local + observable).
- *Cons:* More moving parts (a demo buffer keyed per observer, its own
  determinism surface, its own eviction). Extra compute in the hot training hook.
  Requires storing demonstrator actions the observer can't actually see today
  (need a diegetic-plausibility check: is observing a neighbor's action in-world
  legitimate? The v1 story says agents "copy action sequences" —
  [social_learning.py docstring](../../../artificial_society/systems/social_learning.py) —
  so yes, but the observed action should be the *previous-tick executed* action,
  not a privileged read of *j*'s policy).

**Option C — status quo (prototype only).**
Keep only the global EMA warm-start.
- *Pros:* Zero new code/risk; already the arm-B behavior.
- *Cons:* Non-diegetic (§1.1); provides no *local* channel; cannot be claimed as
  emergent social learning. This is the **control**, not a candidate.

**Lean:** **B is the honest target; A is the cheap first step.** A gives us a
working diegetic channel and a null-beating test at low cost; B is the version
whose result would be scientifically meaningful. Stage A → B (§5).

### 4.2 Knowledge transmission (CausalMemory)

`CausalMemory` is **agent-side and brain-agnostic** — it lives on
`agent.causal_memory` and `ensure_fields` creates it for **every** arch including
rssm ([agent.py:165-166](../../../artificial_society/agents/agent.py#L165)). Its
`feature_vector()` already feeds obs **dims 34–36**
([agent.py:524](../../../artificial_society/agents/agent.py#L524), spliced at
[:561 `*causal_f`](../../../artificial_society/agents/agent.py#L561)), so an rssm
agent's WM already *reads* transmitted causal knowledge through the encoder.
`social_learning_step` also runs for rssm agents (called unconditionally at
[agent.py:1485](../../../artificial_society/agents/agent.py#L1485), `tick % 3`),
so the **observe/teach transmission** paths (`receive_transmitted`,
`sample_for_transmission`) are live for rssm — only the **weight-imitation branch**
inside it no-ops (§4.1). **So the read side works and the transmission side works.**

**The dead link is the WRITE side.** Grepping the agent/culture path,
`CausalMemory.record()` ([culture.py:27](../../../artificial_society/systems/culture.py#L27))
has **no caller in `agent.py` or `culture.py`** — only `feature_vector` (read),
`sample_for_transmission`, and `receive_transmitted` are called. Original causal
experiences appear to be written only from a v1 invention/discovery path that is
gated off for v2 (`if not self.physics_v2`,
[agent.py:1487-1511](../../../artificial_society/agents/agent.py#L1487)) and has
**no rssm equivalent**. Net effect for rssm: nobody mints original causal facts →
`sample_for_transmission` returns nothing → nothing to transmit → dims 34–36 stay
near-constant → the channel is inert end-to-end. **This must be verified (§11-Q1),
then fixed:** give rssm agents a diegetic write site — when an agent's executed
action produces a salient outcome (large positive reward delta / material state
change), `causal_memory.record(...)` the (action, materials → effect) tuple. That
is local and observable by construction.

Separately note: `_resolve_causal_pending`
([agent.py:938](../../../artificial_society/agents/agent.py#L938)) drives the
**different** `CausalModelV2` curiosity system and is called **only in the
`elif self.physics_v2` dispatch branch**
([agent.py:1350-1360](../../../artificial_society/agents/agent.py#L1350)). rssm
agents hit the `if learner is not None` branch *first* and **never** run it — so
`CausalModelV2` is dead for rssm. That is fine (it is a v2 curiosity mechanism,
out of scope), but it means dims 34–36 are the *only* causal signal the rssm WM
sees, which is exactly why fixing the CausalMemory write side matters.

**Options:**
- **K1 — minimal write hook (recommended).** Add one flag-gated `record` call on
  the rssm reward path (near [store_transition, agent.py:1561-1565](../../../artificial_society/agents/agent.py#L1561)),
  triggered by an outcome-salience threshold. Cheapest; makes the existing
  transmission + obs plumbing come alive. Pure agent-side, no WM change.
- **K2 — campfire pool for rssm.** Also wire `campfire_knowledge_sharing`
  ([social_learning.py:104](../../../artificial_society/systems/social_learning.py#L104)),
  which pools `KnowledgeGraph` facts/macros at fire cells. But rssm agents have no
  `KnowledgeGraph` (`getattr(a, "knowledge", None)` → None → skipped), so this is a
  larger build (give rssm agents a KG, or retarget campfire to CausalMemory).
  Defer — group sharing is a stronger claim best made after bilateral works.
- **K3 — none.** Leave dims 34–36 inert. Rejected: wastes obs capacity the WM must
  otherwise learn to ignore (the grouped-decoder down-weighting, RSSM spec §4.1,
  would just mask a dead input).

### 4.3 Language (proto-tokens)

**Finding that reframes the task's premise:** proto-language tokens are **not in
the obs vector at all.** The full `local_features` return
([agent.py:527-565](../../../artificial_society/agents/agent.py#L527)) has no
token/language channel; `brain.py` never references tokens
(only the layout comment). **Obs dims 15–16 are `nearby_count` and `friends_count`**
([brain.py:28](../../../artificial_society/agents/brain.py#L28),
[agent.py:544-545](../../../artificial_society/agents/agent.py#L544)) — plain
social-density scalars, **not** token channels. The RSSM decoder's down-weighting
of 15–16 as "irreducible-from-egocentric-view" (RSSM spec §4.1) is about
*social-nearby noise* and has **nothing to do with language**.

So the adjudication is: **language is currently ungrounded in *every* policy.**
`_maybe_mark_language` / `_observe_tokens` run for all agents
([agent.py:1537-1539](../../../artificial_society/agents/agent.py#L1537)) and
populate `TokenMemory`/`TOKEN_WORLD`, but nothing feeds a token percept back into
the 57-dim observation — the network can mark and "observe" tokens but never
*perceives* them as input. Down-weighting 15–16 does not hurt language grounding
because 15–16 were never language.

**Options:**
- **L1 — defer (recommended for stages 1–2).** Grounding language requires a **new
  obs channel** (e.g., a small "token-percept" vector: nearest-token pigment/context
  summary at the agent's cell). Because `INPUT_SIZE=57` is pinned by the v1 golden
  (§2.2), this must be an **rssm-only obs extension** — a separate feature tail the
  SharedLearner reads, never appended to the shared 57-dim vector. That is a real
  design (new dims, new WM encoder width for the rssm arm, new decoder group,
  determinism for the token draw) and should not ride stages 1–2.
- **L2 — rssm-only token-percept tail.** Add `token_feats` (fixed small width) to
  what `learner.act` encodes, sourced from `TOKEN_WORLD.tokens_at` + the observer's
  `TokenMemory` association. Fully diegetic (tokens are physical). Largest lift;
  strongest emergence claim (proto-language grounded in policy).
- **L3 — reward-shaped language only.** Keep language out of obs; let token
  mark/observe feed the existing small reward bonus. Rejected as a *culture* claim:
  it shapes behavior without the agent perceiving symbols, so no communication
  channel emerges.

## 5. Recommendation & staged rollout

Beat the non-diegetic prototype control with the **cheapest diegetic channel
first**, then add semantic correctness, then (only if signal appears) the
expensive grounded-language build.

- **Stage 1 — Knowledge write hook (K1) + slab-row imitation (Option A).**
  Cheapest diegetic wins. K1 makes dims 34–36 live (one gated `record` call);
  Option A retargets the proven `imitate_from` gate to slab rows. Both are
  agent-side/slab-side, no WM change, minimal hot-file surface. This alone yields
  a testable "rssm+culture vs rssm" contrast.
- **Stage 2 — Imagination-level imitation (Option B).** Replace/augment Stage-1
  Option A with behavioral cloning in imagined states — the scientifically honest
  policy channel. Gate independently so A-vs-B is itself measurable.
- **Stage 3 — Grounded language (L2).** Only if Stages 1–2 show a population
  effect worth the obs-extension cost. Highest lift, highest emergence value.
- **Throughout:** the `PrototypeActor` warm-start stays **on** as the control
  substrate; an optional sub-arm turns it *off* to isolate whether local channels
  can carry culture without the global average (§6 guardrail against just
  re-deriving the prototype).

## 6. Experiment design sketch

**Question:** does *local, observable* cultural coupling add competence beyond the
non-diegetic prototype warm-start?

- **Arms:**
  - **B (control):** full RSSM, prototype warm-start on, `culture_coupling=off`.
    (Exactly the proven arm B.)
  - **B+K:** B + knowledge write hook (K1).
  - **B+P:** B + policy transmission (Stage-1 A, then Stage-2 B as sub-variants).
  - **B+KP:** both channels.
  - *(optional)* **P−proto:** policy transmission with prototype warm-start
    **off**, to test whether the local channel substitutes for the global average
    (guards against "culture" that is really just the prototype re-derived).
- **Pairing / regime:** inherit the RSSM A/B harness — paired seeds, shared
  founders/terrain, **transitions-matched** primary axis (`transitions_stored`,
  [learner.py](../../../artificial_society/agents/rssm/learner.py)), dual
  sample-and-compute reporting, device assertion (RSSM spec §10.2). Cultural
  coupling changes behavior → environments diverge → same "pairing buys variance
  reduction, not identity" caveat applies.
- **Endpoints (primary):** survival RMST (same as A/B) — does culture lift the
  population beyond the prototype?
- **Endpoints (mechanism / emergence):**
  - *Competence-vs-age*, split by `spawn_origin` — do agents born into a
    culturally-active population cross competence **sooner**? (The make-or-break
    newborn metric, RSSM spec §11.)
  - *Transmission liveness:* count of non-zero `sample_for_transmission` events and
    causal-fact population per capita (must be >0 for K to be non-inert — direct
    check on the §4.2 dead-write fix).
  - *Diversity guardrail* (RSSM spec §11): does the local channel homogenize the
    population *more* than the prototype already does? A channel that just collapses
    to one policy is not culture.
  - *Emergence metrics (external, in-progress):* the parallel emergence-metrics
    system is **not yet landed** (`systems/` has no metrics module today — only
    `culture.py`/`technology.py` mention diversity). Wire its endpoints (e.g.
    innovation depth, cumulative-culture depth) as **secondary** when available;
    do not block Stage 1 on it. Fallback secondary: mean causal-fact depth and
    token-convergence rate from `_maybe_collect_language_convergence`
    ([agent.py:1539](../../../artificial_society/agents/agent.py#L1539)).
- **Pre-registration:** commit thresholds and the stopping rule (as the RSSM lane
  did) before the multi-seed run; 3-seed pilot → power calc → N.
- **Falsifiable null:** if B+K/B+P do not beat B on RMST **and** show no
  competence-vs-age advantage, the local channels add nothing over the prototype —
  report and stop (mirrors the RSSM lane's honesty about small effects).

## 7. Integration seams (all behind the flag)

- **`RSSMConfig`:** add `culture_coupling: str = "off"` (values `off` |
  `knowledge` | `policy` | `both`) plus per-channel knobs
  (`imit_alpha`, `imit_trust_min`, `imit_success_ratio`, `causal_record_thresh`,
  `imit_mode ∈ {slab, imagination}`). Defaults preserve arm B exactly.
- **Policy channel (slab):** new method on `ActorCriticSlab` (`blend_row` for
  Option A; a demo-buffer + aux loss in `_losses`/`_imagination_pass` for Option B).
  Own generator for the stochastic imitation trigger (`(global_seed, "imit", id)`),
  never the global stream.
- **Knowledge channel:** the K1 `record` call sits on the rssm reward path near
  [agent.py:1561](../../../artificial_society/agents/agent.py#L1561), guarded by
  `learner is not None and cfg.culture_coupling in {knowledge, both}` — one
  hot-file line, flagged to core-lead.
- **`social_learning_step` reuse:** its causal observe/teach already runs for rssm
  (§4.2); the weight-imitation branch stays a v1-only no-op. The rssm policy
  channel is driven from the SharedLearner side, **not** by extending
  `social_learning_step` (keeps the systems-lane file v1-pure).
- **Language (Stage 3 only):** rssm-only obs tail in `learner.act`, never a change
  to `local_features`/`INPUT_SIZE`.

## 8. Determinism & RNG

- Every new stochastic decision (imitation trigger, demo sampling, record
  gating if probabilistic) draws from an **rssm-owned generator** seeded from
  `(global_seed, tag, agent_id)` — never `random`/`numpy`/`torch` globals
  (Constraint §2.3; RSSM spec §6). Per-agent seeding keeps draws independent of
  iteration order under births/deaths.
- **Off-flag = zero extra draws.** With `culture_coupling="off"`, no generator is
  constructed and no branch is entered; arm B stays byte-identical. Add the same
  guard-style unit assertion the RSSM lane uses: an off-flag `Simulation.__init__`
  leaves `torch.get_rng_state()` unchanged, and the golden digest does not move.
- Determinism test **across births/deaths** for each channel (same seed → identical
  transmission events), matching the RSSM lane's birth/death determinism test.

## 9. Fixed-Constraints checklist (self-audit)

- [x] No parent→child weight copy (peer imitation only; §2.1).
- [x] `INPUT_SIZE=57` unchanged; language tail is rssm-only (§2.2, §4.3-L1).
- [x] rssm-owned generators only; globals untouched (§8).
- [x] v1 golden unchanged; off-flag zero-draw (§8).
- [x] Hot-file edits minimal + flagged to core-lead (§7).
- [x] Prototype warm-start kept as control, not claimed as culture (§1.1).

## 10. Hot-file budget

| File | Edit | Size |
|---|---|---|
| `agents/agent.py` | one gated `record` call on rssm reward path (K1) | ~1–3 lines |
| `agents/rssm/config.py` | `culture_coupling` + knobs | new fields (not hot) |
| `agents/rssm/actor_critic.py` | `blend_row` / demo-loss (rssm-owned) | new (not hot) |
| `agents/rssm/learner.py` | drive imitation from act/train hooks | new (not hot) |
| `agents/brain.py` | **none** in stages 1–2 (language tail is learner-side) | 0 |

Only the single `agent.py` line touches a frozen contract; everything else lives
in the isolated `agents/rssm/` tree.

## 11. Open questions (for the review-agent round)

- **Q1 (blocking for K).** *Verify* that `CausalMemory.record` truly has no live
  writer on the rssm path (grep found none in `agent.py`/`culture.py`; confirm no
  writer in `systems/technology.py` / invention modules fires for rssm). If a
  writer exists and is arch-agnostic, K1 shrinks to "confirm liveness"; if not,
  K1 is required for the channel to be non-inert.
- **Q2.** Policy channel: is a **raw slab-row blend (A)** acceptable given
  weight-space non-identifiability, or do we go straight to **imagination
  imitation (B)** and skip A? (Cost vs. correctness; affects Stage 1 scope.)
- **Q3.** Diegetic legitimacy of the demo signal in Option B: is reading a
  neighbor's *previous executed action* the right observable, and how is it
  surfaced without a privileged read of *j*'s policy?
- **Q4.** Homogenization vs. culture: what diversity floor distinguishes "learned
  from neighbors" from "collapsed to one policy"? Needs a pre-registered threshold.
- **Q5.** Should the **P−proto** sub-arm (prototype off) be in the pilot or
  deferred? It is the cleanest test that local channels carry culture, but adds a
  cell to the design.
- **Q6.** Language (L2): obs-tail width, encoder-width handling for the rssm-only
  extension, and decoder grouping (predictable vs. irreducible) for token percepts
  — full sub-spec deferred to Stage 3.
- **Q7.** Interaction with the **shared WM**: because the WM already pools all
  experience, a knowledge/policy channel may be partly redundant with it. Do we
  need a WM-frozen sub-condition to isolate the cultural channel's contribution?
```
