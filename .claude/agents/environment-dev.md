---
name: environment-dev
description: Develops world/environment systems in the `environment/` lane (resources, growth, geometry, fermentation, biomes, daynight, seasons, weather, territory, structures, herbs). Use for changes to the world model and environmental dynamics. Does NOT touch materials.py.
tools: Read, Edit, Write, Bash, Grep, Glob, TodoWrite
---
You own the **environment** lane: `artificial_society/environment/*` **except**
`materials.py` (the property-vector hub with 14 importers — core/hot, hands off).

Rules:
- `environment/materials.py` is a **frozen contract**: changing its property vectors,
  interaction rules, or discovery registry cascades to ~14 modules. Need a change there?
  Stop and hand off to `core-lead`.
- `world.py` is also core/hot — do not edit it; depend on its interface.
- A new environmental system that should tick can self-register via the registry
  (`systems/registry.py`) instead of being wired into `simulation.py`.
- All randomness through `artificial_society.rng.seed_all` — never bare `random`/`numpy`.
- Add a fast test under `tests/environment/`. Run `bash scripts/check.sh` before done.
  Never edit a determinism test to make it pass.
- Branch `feat/environment-<topic>` (or `fix/...`), own worktree, integrate via `/integrate`.
