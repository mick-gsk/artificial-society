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

### Dashboard / remote GPU hosting

```bash
# Web dashboard: drives a headless sim in a background thread, serves a LAN UI on :8000
pip install -e ".[serve]"            # fastapi + uvicorn
PYTHONHASHSEED=0 python -m artificial_society.serve   # or scripts/run-dashboard.{bat,sh}
```

`artificial_society/serve/` (infra lane) hosts the sim on the GPU PC and is controlled
remotely from the MacBook. `runner.SimulationRunner` steps the sim in a thread and calls
`sim.stats.update(...)` after each tick (`step` does not collect stats itself); `app` exposes
`/api/{run,stop,status,history,graph.png,health}`. Full setup: `docs/serve-setup.md`.

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
- **Adding a system — use the registry, do not edit `simulation.py`.** Create one new
  module under `systems/` that self-registers via `@register(...)` (see
  `systems/registry.py`). `Simulation` builds it (`sim.<name>` + `sim.systems` bus) and,
  if it has a `tick` hook, ticks it in ascending `order`. The legacy built-in systems are
  registered in `systems/_builtins.py` (dormant). The `/new-system` command scaffolds this.

## Parallel multi-agent development

This repo is structured so several Claude Code agents work in parallel with minimal
conflicts. **New agent? Start with [`docs/agent-quickstart.md`](docs/agent-quickstart.md),
then claim a task from [`docs/roadmap.md`](docs/roadmap.md).** Full lane detail in
[`docs/ownership.md`](docs/ownership.md); the essentials:

- **Lanes.** Each file belongs to one lane: `core` (serial only), `agents`,
  `environment`, `systems`, `visualization`, `infra`. Take one lane; touch only its files.
- **Hot files are a frozen contract** (`simulation.py`, `world.py`, `agents/agent.py`,
  `agents/brain.py`, `environment/materials.py`, `systems/registry.py`). Don't edit them
  in a domain lane — route changes through the `core-lead` (serial). `CODEOWNERS` marks them.
- **Subagents & commands.** `.claude/agents/` defines one dev per lane plus `core-lead`
  and `reviewer`. `.claude/commands/`: `/new-system`, `/check`, `/integrate`.
- **Workflow.** Work in a git worktree on a lane-prefixed branch
  (`feat/systems-<topic>`, `core/<topic>`, …): develop → `/check` → `/integrate` (rebase,
  re-check, push, PR) → CI gate + review → merge → remove worktree.
- **Gate.** CI (`.github/workflows/ci.yml`) runs the full pytest suite — including the
  determinism contract (golden trajectory + headless digest) — on every PR, and lints the
  Python files the PR changes. `scripts/check.sh` (= `/check`) mirrors it locally.
- **Determinism is sacred.** All randomness via `artificial_society.rng.seed_all`; never
  edit a determinism test to make it pass — a red golden trajectory means your change
  altered behaviour.
