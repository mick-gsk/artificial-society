---
description: Scaffold a new self-registering simulation system (no edit to simulation.py).
argument-hint: <name> [order]
---
Create a new society-level system named `$ARGUMENTS` as **one new file** under
`artificial_society/systems/`. Do **not** edit `simulation.py` or `systems/_builtins.py`
— that is the whole point of the registry seam (`systems/registry.py`).

Steps:

1. Create `artificial_society/systems/<name>.py` containing a class for the system
   and a registered factory:

   ```python
   from artificial_society.systems.registry import register

   class <Name>System:
       def __init__(self):
           ...

   @register(name="<name>", order=<order or 100>)
   def _build(sim):
       return <Name>System()
   ```

   - If the system must run every tick, add a module-level
     `def _tick(sim, tick): ...` and pass `tick=_tick` to `@register`. It runs in
     ascending `order`; it can read `sim.world`, `sim.agents`, and other systems via
     `sim.systems[...]`. Omit `tick` for a construct-only (dormant) system.
   - All randomness MUST route through `artificial_society.rng` (`seed_all`) — never
     bare `random`/`numpy` global state, or the determinism tests will fail.

2. Add a fast unit test at `tests/systems/test_<name>.py` (see
   `tests/systems/test_registry.py` for the pattern). Keep it deterministic and headless.

3. Run `/check`. The new system is auto-discovered (`registry.discover()`) — confirm
   it appears in `sim.systems` and that the golden trajectory still passes (a dormant
   system must not change it; a ticking system intentionally may — coordinate with
   core-lead before regenerating the golden, per `docs/ownership.md`).
