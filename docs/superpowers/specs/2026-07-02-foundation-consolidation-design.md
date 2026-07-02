# Foundation Consolidation — Design

**Date:** 2026-07-02
**Branch:** `core/foundation` (from local `main` @ 8ea9a09)
**Status:** Approved by owner (conversation, 2026-07-02)

## Context & Goal

The project was originally built to run a research experiment on open-ended
innovation. The research is abandoned; the living simulation (world, agents,
brains, environment, society systems) is the product going forward. The new
direction: grow the simulation into a complex world in which agents *learn*
capabilities — communication/language, tool-making, building, environment
shaping — rather than being handed them. Nothing is pre-granted; capabilities
must emerge from learning, as they did for humans.

This spec covers the **first slice only**: consolidating a clean foundation.
New capabilities (language first) get their own specs afterwards.

## Decisions (owner-approved)

1. **Research handling:** move into a top-level `archive/` folder inside the
   repo — visible, not deleted.
2. **Base:** local `main` + Phase-4 physics + Phase-5 de-scripting.
   *Discovered during execution:* local `main` @ 8ea9a09 already contains
   Phase-4 (`1b7330c`), both Phase-5 branches (`ed58a90`, `73c9599`) and
   Tier-3 brain perf (`b08b88a`) — so the merge phases collapse to
   verification. Live-viz and other feat branches: not in this slice.
3. **Borderline modules:** keep `systems/need_driven_invention.py` (live,
   load-bearing invention mechanic); archive `systems/causal_model.py`
   (research prototype; on `core/foundation` it only exists as WIP on the
   research branch — nothing to remove from the foundation tree).
4. **Repo hygiene:** full cleanup; every branch is tagged
   (`archive/branch-<name>`) before deletion; stale `as-*` worktrees removed.

## What counts as research (to archive)

- `artificial_society/research/` — 11 modules (pilot runners, functional-depth
  metrics, random-recombiner null model, gate analysis, export, instrument)
- Research tests: `tests/test_research_smoke.py`, `tests/test_research_metrics.py`,
  `tests/test_pathA_m1.py`; `tests/test_characterization.py` only if it is
  research-bound (verify before moving)
- `docs/research/` entirely (experiment plan, umwelt-design docs, pilot data,
  reports, figures)
- `docs/superpowers/specs/2026-06-28-open-ended-innovation-research-design.md`
- Dead files: `checkpoint.pkl`, `artificial_society/systems/reproduction_metrics.py`
- Project-root clutter (outside repo, plain deletion/move):
  `oee-thesen-literatur-check.md`, `ultracode-prompt-umwelt-design.md`,
  `shared-chat.txt`

## What stays (simulation core)

- Engine: `simulation.py`, `world.py`, `main.py`, `renderer.py`, `rng.py`
- Agents: all of `agents/` (brain/PPO, memory, endocrine, genetics, ToM, …)
- Environment: all of `environment/`
- Systems: all of `systems/` incl. `invention.py`, `logic_gates.py`,
  `need_driven_invention.py`, `language.py` (dormant language substrate —
  future first capability slice)
- Visualization, serve, core tests, golden trajectory

## Phases

- **0 · Secure WIP** ✅ — research WIP committed on
  `feat/research-causal-model-proto` (`4d23ac8`), tagged
  `archive/wip-causal-proto`.
- **1 · Branch** ✅ — `core/foundation` created from `main` @ 8ea9a09.
- **2/3 · Integration** ✅ (collapsed) — Phase-4/5 + Tier-3 already merged in
  local `main`.
- **4 · Verify green** — run full test suite headless; golden trajectory must
  be green on this base (it was regenerated during the earlier integration);
  headless smoke run (`--headless --seed 42 --ticks 500`).
- **5 · Archive move** — create `archive/` (with README explaining what/why),
  `git mv` the research artifacts listed above, remove dead files, ensure
  pytest ignores `archive/`, suite green again.
- **6 · Docs reorientation** — README/CLAUDE.md/roadmap rewritten toward the
  emergent-learned-capabilities vision; no research framing; roadmap lists the
  capability slices (language → tools → building → environment shaping).
- **7 · Repo hygiene** — tag then delete stale research/feature branches;
  remove stale `as-*` worktrees; keep `main`, `core/foundation`, and branches
  with unmerged wanted work (`feat/infra-live-viz` explicitly kept, unmerged).
- **8 · Handover** — final verification; promotion `core/foundation` → `main`
  and pushing to origin are owner-gated.

## Risks / Notes

- Golden regeneration only if Phase-4/5 base turns out red — then regenerate
  once, document it.
- `frontend/` (untracked, node_modules) in the main worktree: leftover from
  live-viz runs; handled in hygiene (delete locally, branch keeps the source).
- Heavy simulation runs belong on the GPU-PC; this consolidation is
  lightweight and runs locally.

## After this slice (roadmap, each its own spec)

1. **Language/communication** — wake the dormant `systems/language.py`
   substrate; ground tokens in learned utility, never scripted meanings.
2. **Tools/crafting** — physical combination space without hardcoded recipes.
3. **Building/environment shaping** — persistent, advantage-driven world
   modification.
4. **Learning machinery** — address the pilot's core finding (learned policy
   below random-recombination null): credit assignment, memory, open-endedness.
