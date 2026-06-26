# Artificial Society

An agent-based **emergence simulation**. Agents with neural "brains" live in a 2D grid
world of biomes, weather, seasons and resources. From their local decisions, higher-level
systems — tribes, economy, technology, evolution, culture, language, trade — produce
emergent behaviour over time.

Each agent runs a small neural network (a policy net, a world model and an episodic
memory, see [`agents/brain.py`](artificial_society/agents/brain.py)), so the simulation is
genuinely compute-heavy and benefits from a CUDA GPU — `brain.py` selects the device
automatically (`cuda` if available, otherwise `cpu`).

## Quickstart

Requires **Python 3.9+**. From the repository root:

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e .                    # core dependencies (numpy, torch, pygame, …)
```

> **GPU note:** for an NVIDIA RTX 50-series (Blackwell / sm_120) card, install torch from
> the CUDA 12.8 index *first* — the generic pin does not reliably cover sm_120:
> `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
> See [`docs/serve-setup.md`](docs/serve-setup.md).

## Running

### Visual (pygame window)

```bash
python -m artificial_society.main
```

### Headless (batch / reproducible runs)

```bash
PYTHONHASHSEED=0 python -m artificial_society.main --headless --seed 42 --ticks 2000
```

Useful flags: `--seed`, `--ticks`, `--grid-w`, `--grid-h`, `--pop`. A fixed
`PYTHONHASHSEED` is required for byte-reproducible seeded runs (per-process hash
randomisation otherwise changes set/dict iteration order).

### Web dashboard (remote GPU hosting)

Host the simulation headless on a GPU PC and control/observe it from another machine on the
same network (LAN). Starts a small server that drives the sim in a background thread and
serves a live dashboard on port 8000.

```bash
pip install -e ".[serve]"                              # adds fastapi + uvicorn
PYTHONHASHSEED=0 python -m artificial_society.serve     # or scripts/run-dashboard.{bat,sh}
```

Then open `http://localhost:8000` (or `http://<host-ip>:8000` from another machine). The
dashboard shows the active compute device, lets you start/stop runs and renders live stats,
charts and the ecology graph. Full Windows/GPU setup: [`docs/serve-setup.md`](docs/serve-setup.md).

## Tests

```bash
python -m pytest -q
```

`conftest.py` forces a dummy SDL driver so pygame never opens a real window during tests.

## Project layout

```
artificial_society/
  simulation.py          the Simulation god-object: world, agents, systems, tick loop
  world.py, renderer.py, main.py   world model, pygame rendering, CLI entry point
  rng.py                 central RNG — all randomness must route through seed_all
  agents/                per-agent systems: brain, genetics, memory, culture, …
  environment/           biomes, weather, seasons, resources, territory, …
  systems/               society-level systems: tribes, economy, technology, evolution, …
  visualization/         pygame overlays, matplotlib graphs, statistics tracking
  serve/                 headless web dashboard (runner + FastAPI app + static UI)
tests/                   pytest suite (incl. determinism / golden-trajectory contracts)
docs/                    setup, roadmap and contributor guides
scripts/                 run-dashboard.{bat,sh}, check.sh, …
```

## Determinism (important)

A given seed must produce an identical initial world and population — this is locked by the
golden-trajectory and headless-digest tests. All randomness must go through
`artificial_society.rng.seed_all`; never use bare `random`/`numpy` global state, and never
edit a determinism test just to make it pass (a red golden trajectory means behaviour
changed).

## Contributing

This repo is structured for several contributors (and parallel Claude Code agents) working
in isolated *lanes*. Start with [`docs/agent-quickstart.md`](docs/agent-quickstart.md),
claim a task from [`docs/roadmap.md`](docs/roadmap.md), and see
[`docs/ownership.md`](docs/ownership.md) for lane and hot-file rules.
[`CLAUDE.md`](CLAUDE.md) holds the conventions and invariants in full.
