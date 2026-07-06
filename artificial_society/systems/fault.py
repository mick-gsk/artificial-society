"""Fault system (Phase 2).

Activates propagation, which was wired but dead: no agent ever got spread because
the environmental trigger (:func:`try_environmental_propagation`) was never called, and
person-to-person propagation (:meth:`Simulation.spread_faults`) had no call site.

This module supplies only those two missing call sites:

* an **environmental source** — each alive agent may contract a fault from its current
  cell (scurvy in deserts on low plant intake; wound fever at low health);
* **propagation** — spread carriers then spread to nearby susceptible agents.

Per-tick indicator drain (``Agent._fault_tick``) and resistance/recovery
(``Simulation.tick_resistance_and_recovery``) are already wired in the step loop, so this
file does not touch agents or ``simulation.py``. Behaviour-changing → golden regen.

Determinism: the propagation draws use ``remedy``'s global ``random``, which
``rng.seed_all`` seeds. ``order=35`` puts it after tribes (30) / world_regrowth (25) —
so it sees the post-regrowth cell state — and before economy (40) / stats (70), so the
same tick's prices and statistics reflect the new propagation.
"""

from __future__ import annotations

from artificial_society.systems.registry import register
from artificial_society.systems.remedy import try_environmental_propagation


class FaultSystem:
    """Marker system; all behaviour lives in the registered tick hook."""


def _tick(sim, tick: int) -> None:
    for agent in sim.agents:
        if not agent.alive:
            continue
        # Environmental source: the cell can seed a fresh propagation (no-op if already impaired).
        try_environmental_propagation(agent, sim.world.get_cell(*agent.pos))
    # Propagation: carriers spread nearby susceptible agents.
    sim.spread_faults()


@register(name="fault", order=35, tick=_tick)
def _build(sim):
    return FaultSystem()
