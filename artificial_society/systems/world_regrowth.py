"""World regrowth system (Phase 2).

Re-activates ``world.update_environment`` — resource regrowth, world-disturbance
events, and field diffusion — which existed but was never called by the live
loop, so the world only ever depleted. It ticks at ``order=25``: after the
``seasons`` (10) and ``weather`` (20) systems have published their state for the
tick, and (from ``simulation.step``) after agents have foraged, so each tick the
world depletes then regrows.

The season/weather state is read defensively: if those systems are dormant the
regrowth still runs with neutral defaults (``regrow_cell`` uses ``dict.get``).
"""

from __future__ import annotations

from artificial_society.systems.registry import register


class WorldRegrowth:
    """Marker system; all behaviour lives in the registered tick hook."""


def _tick(sim, tick: int) -> None:
    sim.world.update_environment(
        getattr(sim, "_season_state", {}) or {},
        getattr(sim, "_weather_state", {}) or {},
        tick,
    )


@register(name="world_regrowth", order=25, tick=_tick)
def _build(sim):
    return WorldRegrowth()
