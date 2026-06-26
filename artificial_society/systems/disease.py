"""Disease system (Phase 2).

Activates infection, which was wired but dead: no agent ever got infected because
the environmental trigger (:func:`try_environmental_infection`) was never called, and
person-to-person contagion (:meth:`Simulation.spread_diseases`) had no call site.

This module supplies only those two missing call sites:

* an **environmental source** — each alive agent may contract a disease from its current
  cell (scurvy in deserts on low plant intake; wound fever at low health);
* **contagion** — infected carriers then spread to nearby susceptible agents.

Per-tick symptom drain (``Agent._disease_tick``) and immunity/recovery
(``Simulation.tick_immunity_and_recovery``) are already wired in the step loop, so this
file does not touch agents or ``simulation.py``. Behaviour-changing → golden regen.

Determinism: the infection draws use ``remedy``'s global ``random``, which
``rng.seed_all`` seeds. ``order=35`` puts it after tribes (30) / world_regrowth (25) —
so it sees the post-regrowth cell state — and before economy (40) / stats (70), so the
same tick's prices and statistics reflect the new infections.
"""

from __future__ import annotations

from artificial_society.systems.registry import register
from artificial_society.systems.remedy import try_environmental_infection


class DiseaseSystem:
    """Marker system; all behaviour lives in the registered tick hook."""


def _tick(sim, tick: int) -> None:
    for agent in sim.agents:
        if not agent.alive:
            continue
        # Environmental source: the cell can seed a fresh infection (no-op if already sick).
        try_environmental_infection(agent, sim.world.get_cell(*agent.pos))
    # Contagion: carriers infect nearby susceptible agents.
    sim.spread_diseases()


@register(name="disease", order=35, tick=_tick)
def _build(sim):
    return DiseaseSystem()
