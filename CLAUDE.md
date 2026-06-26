# Artificial Society

Agent-based emergence simulation in pure Python. Agents with neural "brains" live in a
2D grid world with biomes, weather, seasons, resources; higher-level systems (tribes,
economy, technology, evolution, culture, language, trade) produce emergent behaviour.

## Layout

- `artificial_society/` — package root
  - `simulation.py` — the `Simulation` god-object: owns world, agents, all systems, the tick loop
  - `world.py`, `renderer.py`, `main.py` — world model, pygame rendering, entry point
  - `rng.py` — central RNG (`seed_all`); **all randomness must route through here** (see Determinism)
  - `bootstrap.py`, `runtime_patches.py` — monkeypatch the `Simulation` class at import time
  - `agents/` — per-agent systems: brain, genetics, memory, endocrine, theory_of_mind, culture, …
  - `environment/` — biomes, weather, seasons, daynight, resources, territory, growth, …
  - `systems/` — society-level systems: tribes, economy, technology, evolution, language, trade, …
  - `visualization/` — pygame overlays, matplotlib graphs, statistics tracking
- `tests/` — pytest; `conftest.py` forces headless SDL and adds repo root to `sys.path`

## Running

```bash
# Visual run (opens a pygame window)
venv/bin/python -m artificial_society.main

# Headless, programmatic
venv/bin/python -c "from artificial_society.simulation import Simulation; Simulation(headless=True, grid_w=20, grid_h=15, initial_population=8).run()"
```

`Simulation(headless=True, ...)` constructs without opening a window (`sim.screen` and
`sim.renderer` are `None`). Use it for tests, batch runs, and any non-visual work.

## Tests

```bash
venv/bin/python -m pytest -q
```

`conftest.py` sets `SDL_VIDEODRIVER=dummy` / `SDL_AUDIODRIVER=dummy` so pygame never
opens a real window or audio device. When running pytest outside conftest, export those
yourself first.

## Invariants — do not break

- **Determinism / reproducibility.** A given seed must produce an identical initial
  world + population. `tests/test_headless.py` locks this via a state digest. Any new
  randomness must use `artificial_society.rng` (`seed_all`), never bare `random`/`numpy`
  global state seeded elsewhere, or the digest tests will fail.
- **Headless construction must open no display.** Keep pygame window/audio creation behind
  the `headless` flag.
- **Import-time patching.** `sitecustomize.py` and `runtime_patches.py` patch `Simulation`
  before/at import. If construction behaves differently than the class source suggests,
  check `bootstrap.patch_simulation_class` and `runtime_patches`.

## Environment

- Python 3.9 — `venv/` lives one level up at `../venv` (`venv/bin/python`).
- Deps: numpy, torch, pygame, matplotlib, networkx, sympy (+ pytest). See `requirements.txt`.
- Lint/format: `ruff` (config in `pyproject.toml`). A PostToolUse hook auto-runs
  `ruff check --fix` + `ruff format` on edited Python files.

## Conventions

- `from __future__ import annotations` + `dataclasses` + `typing` are used throughout — keep new code typed.
- Systems are loosely coupled and ticked from `Simulation`; add a new system as its own
  module under `systems/` (or `environment/` / `agents/`) and wire it into the tick loop.
