# De-Biologize Naming — Design Spec

**Date:** 2026-07-06
**Branch:** `chore/de-biologize-naming` (worktree `/Users/moritzbecker/projekt/as-debio`, off `main @ 773577b`)
**Status:** Approved (design), pending implementation plan

## Goal

Rename identifiers and symbols in the `artificial-society` codebase that suggest
**biology**, replacing them with neutral agent-/systems-vocabulary — **without
changing the behaviour of the code or the simulation.**

This is a pure lexical/structural refactor. No control flow, no numeric
constant, no algorithm changes. The simulation trajectory must remain
bit-for-bit identical.

## Scope decisions (locked)

| Dimension | Decision |
|---|---|
| **Target vocabulary** | Neutral / agent-systems terms |
| **Aggressiveness** | Only *clearly* biological terms. `energy, health, age, grow, spawn, die/death, alive, brain, body, food` are **kept** (physics-legit or game-standard). |
| **Depth** | Internal identifiers only (variables, functions, classes, attributes, module files) **plus** comments/docstrings. External contracts stay byte-stable: WebSocket/HTTP frame JSON keys, config/manifest keys, CLI flags. |
| **Branch / compat** | New branch off `main`. Old `.pkl` checkpoints may break (regenerated). No `__setstate__` shim needed. Other actor branches (`rssm-dreamer`, `m1-pilot`, `live-viz-v2`) rebase themselves later. |

### Why external contracts survive (verified during exploration)

- **No biological term appears as a WS/HTTP frame JSON key.** The frame schema
  is untouched by this rename. (kept terms energy/health/food may appear — they
  are kept anyway.)
- **No CLI flags** use biological terms.
- **Config:** only a *file path* reference in `artificial_society_manifest.json`
  (`agents\genetics.py`) — updated when the module file is renamed.
- **Golden regression** (`tests/golden_trajectory.json`) is a purely numeric
  array `[tick, count, Σenergy, Σhealth]` with **no field names**. Its generator
  `tests/_util.py` reads only *kept* attributes (`energy`, `health`, `alive`),
  so it needs no edit and the trajectory stays identical.

## Mapping table (frozen)

Neutral agent-/systems vocabulary. Kept terms: `energy, health, age, grow,
spawn, die, death, alive, brain, body, food, eat, plant, plant_food, selection`.

### Genetics / heredity — module `agents/genetics.py` → `agents/traits.py`
- `gene(s)` → `trait(s)`
- `genome` / `genotype` → `trait_set`
- `genetic` → `trait_based`
- `allele` → `variant`
- `inherit_genes` / `inherit` → `derive_traits` / `derive`
- `heredity` / `hereditary` → `derived`
- `mutation` / `mutate` → `perturbation` / `perturb`
- `mutation_rate` → `perturbation_rate`
- `dna` → `trait_code`
- `lineage` / `ancestor` → `origin` / `parent`

### Reproduction
- `reproduce` / `reproduction` → `replicate` / `replication`
- `REPRODUCTION_COST` / `REPRODUCTION_COOLDOWN` → `REPLICATION_COST` / `REPLICATION_COOLDOWN`
- `mate` (noun/verb) → `partner`; `mating` → `pairing`; `_last_mate_id` → `_last_partner_id`
- `breed` → `replicate`
- `fertility` / `fecundity` → `replication_rate`
- `pregnant` / `pregnancy` → `pending_spawn`
- `gestation` → `spawn_delay`
- `offspring` → `spawn`; `children` (count) → `spawn_count`
- `born` / `birth` → `spawn`
- `embryo` / `egg` → `pending_spawn`

### Metabolism
- `metabolism` / `metabolic` → `upkeep`; `metabolize` → `consume_upkeep`
- `digest` → `process`
- `starve` / `starvation` → `deplete` / `depletion`
- `hunger` → `energy_need`
- `nutrient` / `nutrition` → `resource_value`
- `calorie` → (fold into `energy`)

### Foraging / plants / hunting — module `environment/fermentation.py` → `environment/spoilage.py`
- `forage` / `forager` → `gather` / `gatherer`
- `vegetation` → `resource_cover`
- `photosynthesis` → `resource_regrowth`
- `predator` / `prey` → `attacker` / `target`
- `hunt` → `pursue`
- `pheromone` → `signal_marker`; `scent` → `trail`

### Health / disease / injury — module `systems/disease.py` → `systems/faults.py`
- `disease` → `fault`
- `sick` / `is_sick` → `impaired` / `is_impaired`
- `infect` / `infection` / `contagion` → `spread` / `propagation`
- `immune` / `immunity` → `resistant` / `resistance`
- `symptom` → `indicator`
- `heal` / `healing` → `recover`
- `wound` / `injury` → `damage`

### Endocrine / drives — module `agents/endocrine.py` → `agents/modulation.py`
- `endocrine` → `modulation`
- `hormone(s)` → `modulator(s)`
- `cortisol` → `stress`
- `adrenaline` → `arousal`
- `dopamine` → `reward`
- `serotonin` → `satisfaction`
- `melatonin` → `rest`
- `oxytocin` → `affiliation`
- `circadian` → `day_cycle`

### Lifecycle / organism
- `organism` / `creature` / `animal` → `agent` / `entity`
- `species` → `agent_type`
- `senescence` → `decay`
- `lifespan` → `max_age`
- `evolve` / `evolution` → `adapt` / `adaptation`
- `fitness` → `score`

### Borderline — resolved (kept unless noted)
- `eat` / `do_eat` → **kept**
- `plant` / `plant_food` → **kept** (world-field key `world.F["plant_food"]`, read across the system — higher risk, low bio-signal)
- `selection` → **kept** (generic)
- `fitness` → `score` (only borderline that is renamed)

### Test file rename
- `tests/agents/test_genetics_strength.py` → `tests/agents/test_traits_strength.py`

## Boundary cases (internal-symbol vs. wire-literal)

Because depth = "internal only", three cases sit exactly on the internal/wire
boundary. Each is handled so the wire stays stable:

1. **`world.F["plant_food"]`** — NumPy SoA field key read across the system.
   → **Kept** (plant is borderline + high blast radius).
2. **`'mate'` action label** (`agents/theory_of_mind.py`) — verify whether it is
   emitted to a frame or user-facing log.
   → If wire-visible: keep the string literal, rename only the internal symbol.
   → If internal-only: rename normally to `'partner'`.
3. **Hormone-name dict keys** (`CORTISOL`, `ADRENALINE`, ... in `endocrine.py`) —
   verify whether any hormone name is emitted to the dashboard/frame.
   → If wire-visible: keep the emitted string, rename the internal symbol.
   → If internal-only: rename per the mapping.

## Execution approach

**Token-accurate, per-concept, test-gated.** For each concept cluster:

1. Build a substitution map from *real identifier tokens* (AST/token scan, not
   free-text) so substring collisions (`eat` in `create`, `mate` in `estimate`)
   are impossible.
2. Apply with word boundaries; review the diff.
3. Rename comments/docstrings in the same pass (symbol tools skip these).
4. Rename module files with import fixups + manifest path fixup.
5. Run the full test suite + golden. Commit the cluster atomically.

Clusters (rough order, each an atomic commit): genetics/heredity → reproduction
→ metabolism → foraging/plants/hunting → health/disease → endocrine/drives →
lifecycle/organism → module-file renames + manifest → final leak-audit.

Rejected alternatives: symbol-only tools (rope/LibCST) miss comments and split
one concept across many symbols; blind global `sed` over-replaces and misses
derived forms.

## Verification — "behaviour unchanged" gates

1. **Golden trajectory** `[tick, count, Σenergy, Σhealth]` stays byte-identical
   (numeric — proves logic invariance). Its generator reads only kept
   attributes, so no golden regeneration is needed.
2. **Full test suite** green after **every** cluster commit.
3. **Wire-schema diff:** emit one frame before/after, diff JSON keys → identical.
4. **Checkpoint round-trip in-session** works (old `.pkl` discarded, as decided).
5. **Leak audit:** final grep over the whole branch diff → no rename-list
   biological term remains.

Under Ultracode, the verification phase (leak audit + wire-schema diff + golden)
can run as a multi-agent workflow; the rename itself stays a linear, test-gated
sequence on this single branch (parallel worktrees would conflict on the many
cross-module references).

## Out of scope

- User-facing strings (log messages, dashboard labels, chronicle text) — depth
  is internal-only.
- Wire/serialization contract keys (frame JSON, config, CLI).
- Any behavioural / numeric change.
- The `archive/` tree (frozen research artifacts).
