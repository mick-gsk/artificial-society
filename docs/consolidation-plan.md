# Monkeypatch consolidation — remaining work (core-lead)

The simulation was historically assembled by patching `Agent`/`Simulation` at import
time. That patching is the worst parallel-dev hazard: behaviour lives in wholesale
*replacement* functions in a separate module, so two agents changing agent behaviour
collide in a 200-line patch that merges badly. Consolidation = move that behaviour into
the real classes so each change happens in one expected, `core`-owned place.

## Already done (by the team + this restructuring)

- Per-agent / brain tuning constants are baked into `agents/agent.py` and
  `agents/brain.py` (see the comments there) — no longer mutated at import.
- `main.py` drives the explicit `Simulation.run` directly; the bootstrap `run`
  override is gone.
- `simulation.py` has the explicit, registry-driven `step()` / `run()` (this work).
- **Dead bootstrap layer removed** (`bootstrap.py`, `sitecustomize.py`) — monkeypatch
  layers went from 3 → 1. `test_main_cli.py` asserts the override stays gone.

## Remaining active patch (one layer)

`simulation.py` imports `runtime_patches` → imports `systems/emergence_runtime.py` →
`apply_emergence_integration()` runs at import and patches:

| Patched | Source function | Fold target |
|---------|-----------------|-------------|
| `Agent.update` | `patched_update` (emergence_runtime ~298–521) | real method in `agents/agent.py` |
| `Agent._collect_resources` | `_collect_resources_from_materials` | real method in `agents/agent.py` |
| `Agent._build` | `_build_from_resources` | real method in `agents/agent.py` |
| `Agent.spawn_random/spawn_child` wrappers | `_wrap_classmethods` (call `_ensure_runtime_fields`) | fold into the real classmethods |
| `Agent.x` / `Agent.y` properties | `_ensure_class_properties` | declare on the class |
| `DiscoveryRegistry` methods | `_patch_discovery_registry` | real methods where `DiscoveryRegistry` is defined |
| `EconomySystem.maybe_trade` | `_maybe_trade_cached` | real method in `systems/economy.py` |
| `social_learning_step` | `_social_learning_step_cached` | real function in `systems/social_learning.py` |

Helper functions used by `patched_update` (`_ensure_runtime_fields`,
`_compact_material_inventory`, `_cached_nearby_agents`, `_maybe_mark_language`,
`_observe_tokens`, `_maybe_collect_language_convergence`) move next to wherever their
new home methods live (in `agents/agent.py` or a small imported helper module).

## Why this is a dedicated, serial core-lead task — not parallel work

It is a large diff in `agents/agent.py` (a hot file) and overlaps the in-flight
`architecture/emergent-hierarchy` and `fix/invention-reward-and-action-space` branches.
Doing it in parallel would *create* the merge conflicts the restructuring removes. Run
it on a single `core/consolidate-agent-update` branch, serially, ideally synced with
those branches first.

## Sequence (each step behaviour-preserving — `scripts/check.sh` green after every one)

1. Fold `patched_update` + its helpers into `Agent.update` (+ `_collect_resources`,
   `_build`) in `agents/agent.py`; delete those three assignments from
   `apply_emergence_integration`.
2. Fold the `spawn_random`/`spawn_child` wrappers into the real classmethods; drop
   `_wrap_classmethods`.
3. Declare `Agent.x` / `Agent.y` on the class; drop `_ensure_class_properties`.
4. Move the `DiscoveryRegistry` methods to its definition; drop `_patch_discovery_registry`.
5. Replace `EconomySystem.maybe_trade` and `social_learning_step` with their real
   implementations in their own modules; drop those patches.
6. When `apply_emergence_integration()` is empty, delete it, `runtime_patches.py`, and
   the `simulation.py` import of `runtime_patches`. Keep any genuinely shared helper
   logic as a normal imported module.

**Invariant:** the golden trajectory (`tests/test_regression_golden.py`) and headless
digest (`tests/test_headless.py`) stay byte-identical at every step. If a step can't be
done behaviour-preservingly, stop and reassess — do not regenerate the golden to "fix" it.
