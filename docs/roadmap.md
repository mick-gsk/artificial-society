# Artificial Society — Roadmap & Status (for Claude Code agents)

This is the **single source of work** for agents (and humans) developing this repo in
parallel. Read [agent-quickstart.md](agent-quickstart.md) first (5-minute onboarding),
then claim a task from the backlog below. Lane rules live in [ownership.md](ownership.md);
conventions in [../CLAUDE.md](../CLAUDE.md).

The original architecture analysis that motivated all this is in the plan file
`~/.claude/plans/ziel-des-projekts-eager-rossum.md` (German).

> **Direction (2026-07):** the research effort (open-ended-innovation paper) is
> **abandoned and archived** under [`archive/`](../archive/README.md). The project's goal
> is now the simulation itself: a complex world in which agents **learn** capabilities —
> communication/language, tool-making, building, shaping their environment — the way
> humanity did. Nothing is pre-granted or scripted; capabilities must be discovered and
> learned. See §4b for the capability roadmap and
> [`docs/superpowers/specs/2026-07-02-foundation-consolidation-design.md`](superpowers/specs/2026-07-02-foundation-consolidation-design.md).

---

## 1. Where the project stands (current status)

The simulation is an agent-based **emergent world**: neural-brained agents in a 2-D biome
world with resources, weather, seasons, and society-level systems. The goal is behaviour
that **emerges from local rules + resource flows + selection**, never scripted events.

**What was wrong (audit):** most subsystems existed but were *unwired*. The live loop was
`bootstrap.run` + import-time monkeypatches (`emergence_runtime`), which silently dropped
world regrowth, births/evolution, disease, society systems, and stats (a `_safe_call`
wrapper swallowed the resulting signature errors). Net effect: agents foraged a
never-regrowing world, trained brains, and never reproduced (population kept alive only by
random respawn → no selection).

**What now works (done):**
- **Runnable + headless + deterministic.** `Simulation(headless=, seed=, load_checkpoint=)`;
  one explicit `Simulation.step()`/`run(max_ticks=)`; the dual bootstrap loop is gone.
  All randomness routes through `artificial_society.rng.seed_all`. Cross-process
  reproducibility needs `PYTHONHASHSEED=0`.
- **System registry.** Systems self-register (`systems/registry.py`, `@register`); adding
  one is a single new file, never an edit to `simulation.py`. Built-ins live in
  `systems/_builtins.py`.
- **Ecology is alive.** Resource regrowth + world events + field diffusion tick each step
  (`seasons`→`weather`→`world_regrowth`). Environment lane added a logistic carrying-capacity
  ceiling, so the world is genuinely **scarce** (mean food ≈ 0.15× capacity).
- **Reproduction is alive.** Completed pregnancies spawn inheriting children
  (genes + brain weights + causal/episodic memory). Selection can now operate.
- **Emergence rewards rebalanced.** Genuine vector discoveries now out-reward the 5 legacy
  scripted recipes, so open-ended invention isn't funnelled into the scripted outcomes.
- **Observability.** Headless ecology graph (population / food / energy over time).
- **A determinism contract guards all of it:** `tests/test_regression_golden.py` (60-tick
  golden trajectory) + `tests/test_headless.py` (initial-state digest). **104 tests pass.**
- **Phases 2–5 are integrated** (Phase-4 physics, Phase-5 de-scripting, Tier-3 brain perf
  all merged; golden regenerated on `core/foundation` for the combined behaviour change).

**Observed dynamics** (seed 3, 160 ticks): food/capacity falls 0.27→0.13, average energy
rises to the cap then turns down under scarcity, population grows via births. That is the
scarcity + reproduction + selection loop the project is about — appearing for the first time.

---

## 2. Architecture in one screen

```
SimulationCore (simulation.py — HOT)         systems/registry.py (HOT) — @register / discover / build_systems / tick_systems
  step():                                    systems/_builtins.py (core) — registers the legacy systems
    seasons->weather published (tick hooks)    seasons(10) weather(20) world_regrowth(25)
    update_territory_claims; tick_materials    tribes(30) economy(40) technology(50) evolution(60) stats(70)
    for agent: agent.update(...) -> births
    remove_dead; immunity; hamilton
    registry.tick_systems(self, tick)  <----- dormant built-ins tick here once given a tick= hook
    emergency_respawn; checkpoint
```

- **Agents** (`agents/agent.py`, `brain.py` — HOT): dataclass state + PPO brain with
  model-based planning. `Agent.update` is a normal method again (Phase 1b removed the
  `emergence_runtime` monkeypatch layer).
- **Determinism**: `rng.seed_all` seeds python/numpy/torch; `Simulation.__init__` also resets
  the global `DISCOVERY_REGISTRY` / `TOKEN_WORLD`. Set `PYTHONHASHSEED=0` for cross-process.

**Hot/core files (frozen contract — only the `core-lead`, serially):** `simulation.py`,
`world.py`, `rng.py`, `agents/agent.py`, `agents/brain.py`, `environment/materials.py`,
`systems/registry.py`, `systems/_builtins.py`, and the determinism tests.

---

## 3. The five rules (non-negotiable)

1. **Determinism is sacred.** All randomness via `artificial_society.rng.seed_all`. Never
   seed bare `random`/`numpy`. Cross-process reproducibility needs `PYTHONHASHSEED=0`.
2. **Never edit a determinism test to make it pass.** A red golden trajectory means your
   change altered behaviour. If that change is *intentional*, the `core-lead` regenerates the
   golden deliberately (procedure in §6); domain lanes leave it red and flag it.
3. **Stay in your lane.** One lane per worker. Hot files are a frozen contract — need a change
   there? Hand off to `core-lead` (serial).
4. **New system = one new self-registering file** under `systems/` (`@register(name, order,
   tick=...)`). Never edit `simulation.py` to add a system. Use `/new-system` to scaffold.
5. **Gate before done:** `bash scripts/check.sh` (= `/check`) — ruff on changed files + full
   pytest. CI re-runs it on every PR.

**How behaviour-changing domain work interacts with the golden:** make your change, add a
**lane test** under `tests/<lane>/` asserting the new intended behaviour, run the gate. The
golden test will go red — that is expected; **do not touch it**; note it in your PR. The
`core-lead` reviews the behaviour change and regenerates the golden at integration.

---

## 4. Roadmap (remaining work, by phase)

### Phase 1b — finish removing the monkeypatch layer  ·  lane: **core (serial)**  ·  golden: neutral  ·  ✅ DONE
**Done (branch `core/unpatch-phase1b`):** the monkeypatch layer is gone.
- `patched_update` → `Agent.update`, `_collect_resources_from_materials`/`_build_from_resources`
  → `Agent._collect_resources`/`_build` (the old class-body `update` was dead code; relocated
  verbatim). Helpers `_ensure_runtime_fields`, `_compact_material_inventory`, language helpers,
  `_RESOURCE_ALIASES`, and `x`/`y` properties moved into `agents/agent.py`; the per-tick neighbour
  cache is now `Agent._nearby_cached` (a method, so economy/social_learning can call it without an
  import cycle).
- `_maybe_trade_cached` → `EconomySystem.maybe_trade`; `_social_learning_step_cached` →
  `social_learning.social_learning_step`. **NB:** the live cached `social_learning_step` had only
  3 blocks — the source's KnowledgeGraph "block 4" was dead, so it was dropped (keeping it would
  shift the RNG sequence and break the golden).
- `DiscoveryRegistry.known_ids` memoised + `register` invalidation folded into
  `environment/materials.py`.
- `runtime_patches.py` and `systems/emergence_runtime.py` deleted; import removed from
  `simulation.py`; stale discovery-skip entry removed from `registry.py`.
- **Verified:** golden unchanged, full suite green (39 tests), 200-tick health sane,
  `grep apply_emergence_integration` empty.

### Phase 2 remainder — activate the remaining dormant systems  ·  ✅ DONE (`core/phase2-activation`)
Give `_builtins.py` tick hooks (core) or add new systems (systems lane):
- **stats** — `tick=lambda sim,t: sim.stats.update(t, sim.agents, sim.world, sim.tribes,
  sim.technology)`. **Golden-neutral** (read-only). Quick win; feeds the ecology graph.
  (Note: `statistics.py` has a latent bug — it does `getattr(a,'life_stage',…)` where
  `life_stage` is a *method*; life-stage counts are always 0. Fix while you're there.)
- **tribes / economy / technology** — tick hooks calling their `update(...)` (+ tribe
  formation needs per-agent `consider_join`). Behaviour-changing → golden regen.
- **disease** — currently no agent ever gets infected (`spread_diseases` was the only caller
  and it was dead). Needs an environmental infection source, then activate spread.
- Each: a liveness test under `tests/` asserting the system actually does something.

### Phase 3 — state as a single source of truth  ·  ✅ DONE (`core/phase3-ssot`, `core/phase3c`)
- Make `World` the authoritative state façade; mediate cell mutation through a small API.
- **Unify agent field init**: today fields are set across the dataclass + `_ensure_new_fields`
  + `_ensure_runtime_fields` + checkpoint migration. Collapse to one `ensure_fields`.
- **Full, reproducible checkpoints**: serialise the global registries
  (`DISCOVERY_REGISTRY`, `TOKEN_WORLD`) and reset *all* accumulating singletons on construction
  — currently only discovery+language reset; `SEQUENCE_LIBRARY` and `RECIPE_DISCOVERY`
  (`systems/goal_stack_ext.py`) still accumulate across in-process runs.

### Phase 4 — physics / biology consistency  ·  ✅ DONE (`core/phase4-physics`, merged `1b7330c`)
- **Energy conservation (core).** Foraging adds energy with no matching cell debit
  (`apply_consumption` is dead code); cooperation/Hamilton/territory **create** energy from
  nothing; `cell['food']` and `cell['plant_food']/['meat_food']` are competing representations
  of the same quantity. Route consumption through one path, make social bonuses redistributive,
  and add a Σ(agent_energy)+Σ(cell_food) invariant check. This is what keeps energy pinned near
  the cap; fixing it turns scarcity into real selection pressure.
- **`carcass`/`carcasses` bug (core).** Agents read `cell['carcass']`; cells store
  `cell['carcasses']` — carcass meat is invisible to foragers.
- **Aging thresholds (core).** `AGE_LIMIT`(5000) < `AGE_HEALTH_DECAY_HARD`(4500→ ok now, verify)
  — make the senescence ramp reachable and consistent.
- **Scarcity/biome tuning (environment).** Temperature/season effects on regrowth & danger;
  biome-specific scarcity; tune so populations reach a resource-limited equilibrium, not
  extinction or explosion.

### Phase 5 — de-script the remaining emergence blockers  ·  ✅ DONE (`feat/systems-phase5-descript` + `core/phase5-inventory-essentials`, merged `ed58a90`/`73c9599`)
Each was an independent, behaviour-changing task (golden regenerated at integration):
- Open **trade** to discovered materials (not just wood/stone/fiber).
- **Language**: token creation is gated on specific material properties — loosen to let
  signalling emerge from any usable marker.
- **Remedy/disease**: cures are a hardcoded lookup — let remedies be discovered.
- **Logic gates**: only AND/OR/NOT/LATCH are recognised — generalise topology detection.
- **Action clusters** (`need_driven_invention.py`): hardcoded `ACTIONS_FOR_HUNGER` etc. cap
  the behaviour space — let need→action mapping be learned.
- Inventory compaction protects a hardcoded "essentials" list — stop privileging scripted items.

---

## 4b. Capability roadmap (the new direction — each slice gets its own spec first)

**The one non-negotiable principle:** agents are never *given* a capability. The world
makes a capability *possible* and *advantageous*; agents must discover and learn it. If a
design hands agents a behaviour ("when X, do Y"), it is scripting and gets rejected. The
right question for every slice is: *what affordance does the world offer, and what pressure
makes learning it worthwhile?*

1. **Tools / crafting.** Open the combination space: material properties + physical
   actions (`invention.py` primitives) determine outcomes; no hardcoded recipe list
   decides what may exist. Composite tools (handle + blade) should be reachable, not
   enumerated. Spec + laufender Plan: docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md, docs/superpowers/plans/2026-07-02-physik-v2-kern.md.
2. **Building / environment shaping.** Persistent world modification driven by learned
   advantage (shelter → warmth → survival in winter), with structures decaying unless
   maintained — so building has to *earn* its cost.
3. **Language / communication.** Wake the dormant token substrate
   (`systems/language.py`, `agents/communication.py` 4-dim `message_vector`). Ground
   symbols in learned utility: emitting/attending to signals must pay off through the
   reward channel (found food, avoided danger, coordinated action), never through
   scripted meanings. Success looks like: stable signal→situation conventions that
   differ across tribes/lineages. Bewusst nach Tools: Sprache braucht erst etwas Kommunikationswürdiges (Spec 2026-07-02, §9).
4. **Learning machinery.** The abandoned research's one transferable finding: the learned
   policy generated *less* novelty than a random-recombination null — the bottleneck is
   credit assignment, memory, and generation–policy coupling, not the number of world
   mechanics. Improve the brain's ability to notice, remember, and exploit rare payoffs
   (better intrinsic motivation, longer credit horizons, richer episodic reuse) so
   capabilities 1–3 can actually be learned. (Background: `archive/README.md`.)

Sequencing note: slices 1–3 each depend on 4 to some degree; expect to interleave — a thin
slice of 4 whenever a capability plateaus.

---

## 5. Parallel task backlog (claim one — one lane per worker)

Tasks A–K are ✅ done (Phases 2, 4 and 5 above). Remaining + new backlog:

| # | Task | Lane | Golden | Depends on |
|---|------|------|--------|-----------|
| L | Tribe / lineage / discovery overlays | visualization | neutral | — |
| M | Per-agent inspector overlay (state, brain, memory) | visualization | neutral | — |
| N | Rebase + integrate live web viz (`feat/infra-live-viz`, predates Phase 1b+) | infra | neutral | — |
| O | **Capability slice 1: tools/crafting — Physik v2 Kern** (Spec + Plan 2026-07-02) | environment | neutral | — |
| P | **Capability slice 4 (thin): intrinsic motivation & credit horizon** (§4b) | agents | regen | — |
| Q | **GPU-resident big-world engine** (Tier 5; target 200-1000 agents / 200×200+ / 100k+ ticks; design note 2026-07-02) | core | own baseline | O (ports v2 physics) |

Branch naming: `feat/<lane>-<topic>` (or `core/<topic>`). One worktree per branch.

---

## 6. Procedures

**Run / test**
```bash
PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy ../venv/bin/python -m pytest -q   # full suite
bash scripts/check.sh                                                    # = /check (ruff + pytest)
python -m artificial_society.main --headless --seed 42 --ticks 2000      # batch run
```

**Regenerate the golden (core-lead only, for an intentional behaviour change)**
```bash
PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy ../venv/bin/python -c \
  "import json; from tests._util import compute_trajectory; \
   json.dump(compute_trajectory(), open('tests/golden_trajectory.json','w'), indent=0)"
```
Before committing a regen: run the integrated sim ~150 ticks and confirm it is *healthy*
(population neither collapses nor explodes; energy/food trends are sane), and that two
in-process runs are identical.

**Parallel round (orchestrator / core-lead)**
1. Commit a green baseline. 2. `git worktree add -b feat/<lane>-<topic> ../as-<lane> HEAD`
per lane. 3. Brief each lane agent (constraints from §3; one focused task). 4. On return:
review (lane discipline + determinism), merge disjoint branches, regenerate the golden once
for the combined behaviour change, verify health, run the gate, push. 5. `git worktree remove`.

---

## 7. Gotchas / lessons learned

- **Files change under you** during parallel work — always **Read before Edit** (the Edit
  tool will refuse a stale write; re-Read and retry).
- **`PYTHONHASHSEED`**: per-process hash randomisation changes set/dict iteration order, so
  seeded runs diverge across processes unless it's pinned to `0`. The golden test pins it in a
  subprocess; CI inherits it.
- **def-time vs call-time constants**: when baking a constant that `emergence_runtime` used to
  mutate, check whether it's read at call-time (safe to bake) or bound as a function-default at
  def-time (baking changes behaviour). `PLAN_CANDIDATES` was a def-time default → its runtime
  mutation was a dead no-op.
- ~~**`emergence_runtime` still patches `Agent.update`/economy/social-learning at import**~~ —
  resolved in Phase 1b; the patch layer is deleted and those are now plain source methods.
- **`materials.py` (HOT)**: 14+ importers depend on its property-vector layout and
  `DISCOVERY_REGISTRY`; treat as frozen. `brain.py` `INPUT_SIZE` is likewise a contract.
- Construction resets only `DISCOVERY_REGISTRY`/`TOKEN_WORLD`; `SEQUENCE_LIBRARY`/
  `RECIPE_DISCOVERY` still accumulate in-process (Phase 3).
