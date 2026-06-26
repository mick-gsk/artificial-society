# Agent Quickstart — read this first (5 min)

You are one of several Claude Code agents developing `artificial-society` **in parallel**.
This page gets you productive without breaking anyone else.

## 1. Orient (2 min)
- **What the project is + current status + your task list:** [roadmap.md](roadmap.md).
- **Which files you may touch (lanes) + the workflow:** [ownership.md](ownership.md).
- **Conventions + the registry pattern:** [../CLAUDE.md](../CLAUDE.md).
- **Your role:** `.claude/agents/<lane>-dev.md` (or `core-lead.md` / `reviewer.md`).

## 2. Pick exactly one lane
| Lane | You own | Parallel-safe |
|------|---------|---------------|
| `agents` | `agents/*` except `agent.py`/`brain.py` | yes |
| `environment` | `environment/*` except `materials.py` | yes |
| `systems` | `systems/*` except `registry.py`/`_builtins.py`/`emergence_runtime.py` | yes |
| `visualization` | `visualization/*` + `renderer.py` | yes (isolated) |
| `core` | the hot files + determinism tests | **serial only** |

**Hot/core files are a frozen contract** (`simulation.py`, `world.py`, `agents/agent.py`,
`agents/brain.py`, `environment/materials.py`, `systems/registry.py`, `systems/_builtins.py`).
Need a change there? You're not allowed in a domain lane — hand off to `core-lead`.

## 3. Set up an isolated workspace
```bash
# from the repo root, branch-per-lane in its own worktree:
git worktree add -b feat/<lane>-<topic> ../as-<lane> HEAD
cd ../as-<lane>
```
The Python interpreter is `../venv/bin/python` (Python 3.9). Tests need
`SDL_VIDEODRIVER=dummy`; reproducible runs need `PYTHONHASHSEED=0`.

## 4. The five rules (full text in roadmap.md §3)
1. All randomness via `artificial_society.rng.seed_all` — never bare `random`/`numpy`.
2. **Never edit a determinism test to make it pass.** Golden red = your change altered
   behaviour. If intentional, leave it red, add a lane test, flag it for `core-lead`.
3. Stay in your lane; don't touch hot files.
4. New system = one self-registering file (`@register(...)`), never edit `simulation.py`.
   Scaffold with `/new-system`.
5. Gate before done: `bash scripts/check.sh`.

## 5. Build → verify → hand off
```bash
# ... make your focused, single-lane change ...
# add a fast test under tests/<lane>/ asserting the NEW behaviour
bash scripts/check.sh            # ruff (changed files) + full pytest; mirrors CI
git add -A && git commit -m "<lane>: <what changed>"
# (core-lead / owner pushes + opens the PR via /integrate; CI + review gate the merge)
```
If the golden trajectory goes red and your change *intentionally* alters the simulation,
that's expected — don't touch it; say so in your summary. If it goes red and you *didn't*
mean to change the sim (e.g. a visualization change), you accidentally mutated sim state —
revert that.

## 6. Top gotchas
- **Read before Edit** — files change under you during parallel work.
- `emergence_runtime` still monkeypatches `Agent.update`/`economy.maybe_trade`/
  `social_learning` at import until **Phase 1b** lands; edits to those targets are dead until then.
- `PYTHONHASHSEED=0` is required for cross-process reproducibility.
- Don't add new systems to `_builtins.py` — that's the legacy registry; use a new file.

Now open [roadmap.md](roadmap.md) §5 and claim a task.
