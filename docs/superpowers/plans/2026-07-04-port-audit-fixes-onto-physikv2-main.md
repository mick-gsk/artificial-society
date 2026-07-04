# Port audit-fixes onto physik-v2 main — Implementation Plan

> Sub-effort spun out of Task 3 of the „Reibung & Klarheit"-Konsolidierung after
> investigation showed `main` (physik-v2) is population-healthy on its own and
> `core/audit-fixes` is a near-disjoint development line. We do NOT merge the
> branch; we port only the fixes genuinely missing from main, each adapted to
> main's rewritten code, TDD + health-verified, preserving the Plan-4 decision
> (no learned inheritance in the v2 birth path).

**Goal:** Bring the 4 genuinely-missing physics/conservation corrections from `core/audit-fixes` onto physik-v2 `main` without regressing its healthy population equilibrium or violating Plan-4.

**Baseline (verified 2026-07-04):** `main` @ 356836f, 500-tick health run seed 42 = pop 36→64, no collapse (HEALTHY). 318 tests green. This equilibrium is the thing each port must not break.

## Global Constraints

- Repo-Root `/Users/moritzbecker/projekt/artificial-society`, on `main`. Python `/Users/moritzbecker/projekt/venv/bin/python`, pytest from repo root.
- **Plan-4 is binding:** the v2 birth path must NOT inherit learned weights/memory/materials (main deliberately removed this — commit title „Lamarck-Geburtsvererbung im v2-Pfad entfernt"). Any birth-path port keeps random init; only add the *energy transfer*, not the Lamarckian inheritance from audit-fixes.
- **Health gate:** after every behaviour-changing port, run the 500-tick seed-42 health check (`scratchpad/health_main.py`) and confirm still HEALTHY (final pop ≥ ~15, no collapse). A port that pushes main out of its healthy band is reworked, not accepted.
- **Golden:** regenerated ONCE at the end (Task P6), after the combined health run — never per-port.
- Determinism via `artificial_society.rng`; headless `SDL_VIDEODRIVER=dummy`; golden regen needs `PYTHONHASHSEED=0`.
- Source of truth for each fix = the audit commit diff; target = main's rewritten code (adapt, don't paste).
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Scope (from triage)

PORT (missing from main): 67af97e wind · 916288a birth-energy · c9a2525 density-fertility · d357b7e events.
SKIP (already in main / superseded): g1 0faed36 · g2 21d405f · g5 632b7d6 · g6 c3379f9.
TRIAGE then port-only-if-needed: g3 35daca2 (action/economy) · g4 a29ef9e (learning-signals, old brain.py) · fix22 fb6bb9d (picklable done_fn).

## Tasks

### Task P1 — Port proportional wind damage (67af97e)
- Source: `git show 67af97e -- artificial_society/environment/resources.py tests/environment/test_plant_regrowth_wind.py`
- Target: `resources.py` lines ~371 & ~571 (`- 0.012 * wind` flat term, both the dict and vectorized regrowth paths).
- Method: replace the flat `-0.012*wind` drain with the proportional form from the audit commit, in BOTH regrowth paths (they must stay equivalent — `test_vectorized_equivalence` guards this). Port `test_plant_regrowth_wind.py`.
- Gate: new test passes; full suite minus golden green; health run HEALTHY. Commit.

### Task P2 — Port birth energy transfer (916288a), Plan-4-aware
- Source: `git show 916288a -- artificial_society/simulation.py artificial_society/agents/agent.py tests/agents/test_action_effects.py`
- Target: `simulation.py::spawn_child_from_parent` (the v2 path) + any agent.py energy hook.
- Method: add ONLY the energy-transfer logic (`transfer = min(child.energy, max(0, parent.energy - BIRTH_ENERGY_FLOOR)); child.energy = transfer; parent.energy -= transfer`) plus `BIRTH_ENERGY_FLOOR` constant, into main's v2 birth path. Do NOT add the audit branch's `inherit_weights_from` / causal_memory / material inheritance (Plan-4). Port the energy-transfer assertions from test_action_effects; drop any inheritance assertions.
- Gate: birth-energy test passes; `test_energy_conservation` green; full suite minus golden green; health run HEALTHY (this one can suppress growth — confirm it doesn't tip to decline). Commit.

### Task P3 — Port density-dependent fertility (c9a2525)
- Source: `git show c9a2525 -- artificial_society/agents/agent.py tests/agents/test_density_dependent_fertility.py`
- Target: main's conception/reproduction path in agent.py.
- Method: port the local-food-coupled conception gate, adapted to main's agent.py reproduction code. Port test_density_dependent_fertility.
- Gate: test passes; suite minus golden green; health run HEALTHY. Commit.

### Task P4 — Port physics-driven events (d357b7e)
- Source: `git show d357b7e` (adds `environment/events.py` + wiring in simulation.py; tests `test_event_dynamics.py`, `test_event_field_effects.py` — the latter already committed on audit-fixes as a4bf084).
- Target: new file `environment/events.py`; wire into main's `simulation.step()` / environment update. Register via the systems registry if it fits (CLAUDE.md: prefer registry over editing simulation.py) — else minimal wiring.
- Method: bring events.py across; adapt wiring to main's tick loop / cell_store. Port test_event_dynamics + test_event_field_effects.
- Gate: event tests pass; suite minus golden green; health run HEALTHY (events add disturbance — confirm no collapse). Commit.

### Task P5 — Triage g3 / g4 / fix22, port only genuine gaps
- For each: `git show <c>`, then check main for an equivalent.
  - g3 35daca2 (action/economy correctness): port any correctness fix genuinely absent from main's rewritten action code; skip what physik-v2 already handles.
  - g4 a29ef9e (learning-signal fixes): built on old brain.py; main has brain v2. Port only if a fix maps cleanly to brain v2; otherwise document as not-applicable (record in ledger + this file).
  - fix22 fb6bb9d (picklable done_fn): if main's `goal_stack` `done_fn` is an unpicklable closure, apply the picklable form; else skip. Low priority (mid-run save is gated behind `--resume` after the checkpoint fix).
- Gate per ported item: its test green, suite minus golden green. Document skipped items with the reason.

### Task P6 — Combined health run + golden regen + full green + push
- Longer health run (≥500 ticks, seed 42; consider a second seed) confirming combined main stays HEALTHY.
- Regenerate golden (`PYTHONHASHSEED=0 SDL_VIDEODRIVER=dummy … compute_trajectory → tests/golden_trajectory.json`).
- Full suite green. Commit golden regen. Push `origin main`.

## Definition of Done
- main carries the 4 physics corrections (+ any genuine g3/g4/fix22 gaps), still HEALTHY, full suite + regenerated golden green, pushed.
- Plan-4 intact (no learned birth inheritance).
- `core/audit-fixes` now fully represented in main → eligible for archive in the umbrella plan's Task 5.
