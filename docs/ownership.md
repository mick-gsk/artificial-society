# Ownership & Lanes — Multi-Agent Development

This repo is set up so several Claude Code agents (or humans) can develop **in
parallel** with minimal merge conflicts. The mechanism is simple: every file
belongs to exactly **one lane**, each parallel worker takes **one lane**, and a
small set of **hot files** is a shared resource changed only serially.

## Lanes

| Lane | Owns | Parallel safety |
|------|------|-----------------|
| **core** *(serial only)* | `simulation.py`, `world.py`, `rng.py`, `systems/registry.py`, `agents/agent.py`, `agents/brain.py`, `environment/materials.py`, `bootstrap.py`, `runtime_patches.py`, `systems/emergence_runtime.py`, and the determinism tests (`tests/test_headless.py`, `tests/test_regression_golden.py`, `tests/_util.py`, `tests/golden_trajectory.json`) | none — one change at a time, lead-reviewed |
| **agents** | `agents/*` except `agent.py` / `brain.py`: `endocrine`, `emotional_memory`, `theory_of_mind`, `knowledge`, `memory`, `genetics`, `life_stage`, `episodic_strategy`, `communication`, `culture` | medium |
| **environment** | `environment/*` except `materials.py`: `resources`, `growth`, `geometry`, `fermentation`, `biomes`, `daynight`, `seasons`, `weather`, `territory`, `structures`, `herbs` | medium |
| **systems** | `systems/*` except `registry.py` / `emergence_runtime.py`: `tribes`, `economy`, `technology`, `evolution`, `invention`, `need_driven_invention`, `remedy`, `trade`, `language`, `goal_stack*`, `logic_gates`, `conductivity`, `social_learning`, `strategy`, `culture`, `reproduction_metrics`, `world_objects` | high (via registry) |
| **visualization** | `visualization/*` + `renderer.py` | very high (isolated) |
| **infra** | `.github/`, `.claude/`, `scripts/`, `docs/`, `CLAUDE.md`, `CODEOWNERS` | high |

## Hot files (🔒 — shared, serialized)

`simulation.py`, `agents/agent.py`, `agents/brain.py` (INPUT_SIZE contract),
`environment/materials.py` (14 importers), `systems/registry.py`.

**Rule:** a domain agent must **not** edit a hot file in parallel. Need a change
there? It becomes a coordinated **core** change (own serial branch, lead review).
Treat hot files as a frozen contract: depend on them, don't reshape them mid-feature.

## Conflict-avoidance rules

1. **One lane per worker, one worker per lane.** core is always serial.
2. **New system = one new file.** Register it via `systems/registry.py`
   (`@register(name=..., order=...)`); do **not** edit `simulation.py`. Two agents
   adding systems then touch disjoint files → no text conflict. See CLAUDE.md.
3. **Determinism:** all randomness routes through `artificial_society.rng`
   (`seed_all`). Never seed bare `random` / `numpy` global state elsewhere.
4. **Never edit a determinism test to make it pass.** A red golden trajectory means
   your change altered behaviour — fix the change, not the test.
5. **The gate is CI, not your laptop.** Even a clean text-merge is re-validated by
   the full pytest suite (golden trajectory + headless digest) on the merge result.

## Per-agent workflow

```
git worktree add ../as-<lane>-<topic> feat/<lane>-<topic>   # isolated checkout
# ... develop in your lane only ...
/check        # ruff + pytest (incl. golden) locally
/integrate    # rebase on main, re-check, push, open PR
# CI gate (green) + 1 review  →  merge  →  git worktree remove ../as-<lane>-<topic>
```

Branch prefix = lane (`feat/systems-trade`, `fix/environment-resource-regrowth`,
`core/...`). CI and CODEOWNERS key off paths.

## Manual setup the harness can't do for you

These require the repo owner (the harness blocks an agent from self-granting them):

- **Branch protection on `main`:** require CI green + 1 review (+ "Require review
  from Code Owners" to enforce `CODEOWNERS`).
- **Allow feature-branch pushes in `.claude/settings.json`** (currently all
  `git push` is denied). Suggested permission block — apply via `/update-config`:
  - allow: `Bash(git push)`, `Bash(git push -u origin *)`, `Bash(git push origin HEAD*)`,
    `Bash(git commit *)`, `Bash(git checkout *)`, `Bash(git switch *)`,
    `Bash(git branch *)`, `Bash(git fetch *)`, `Bash(git rebase *)`,
    `Bash(git worktree *)`, `Bash(gh pr *)`, `Bash(gh run *)`
  - deny: `Bash(git push --force*)`, `Bash(git push -f*)`, `Bash(git push * main)`,
    `Bash(git push * main *)`, `Bash(git push * main:*)`, `Bash(git push origin main*)`,
    `Bash(git push * HEAD:main*)`
