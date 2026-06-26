"""Registration of the built-in (legacy) systems with the registry.

These predate the registry and are registered here in one place so ``simulation.py``
no longer hard-codes them in ``__init__``. **Do not add new systems here** — instead
create a module under ``systems/`` that self-registers via ``@register(...)``;
:func:`artificial_society.systems.registry.discover` picks it up automatically.

Systems start dormant (``tick=None``: constructed but not ticked). Phase 2 activates
them one at a time by giving them a ``tick`` hook here — an intentional, golden-
regenerating behaviour change (see docs and tests/test_phase2_systems.py).
"""

from __future__ import annotations

from artificial_society.environment.seasons import SeasonCycle
from artificial_society.environment.weather import WeatherSystem
from artificial_society.systems.economy import EconomySystem
from artificial_society.systems.evolution import EvolutionSystem
from artificial_society.systems.registry import register_system
from artificial_society.systems.technology import TechnologySystem
from artificial_society.systems.tribes import TribeSystem
from artificial_society.visualization.statistics import StatisticsTracker


# --- Phase 2 tick hooks: activate dormant built-ins one at a time ----------------
def _tick_seasons(sim, tick: int) -> None:
    """Advance the season cycle and publish its state on the sim for this tick."""
    sim._season_state = sim.seasons.update(tick)


def _tick_weather(sim, tick: int) -> None:
    """Advance weather (uses the current season state) and publish it for this tick."""
    sim._weather_state = sim.weather.update(sim.world, getattr(sim, "_season_state", {}), tick)


def _tick_tribes(sim, tick: int) -> None:
    """Form and maintain tribes: per-agent join decision, then membership sync + cleanup.

    ``TribeSystem`` has no single ``update()`` — formation is a per-agent decision over a
    deterministic spatial neighbourhood (radius 2, matching trade/social conventions).
    Behaviour-changing: ``tribe_id`` feeds the kin-selection (Hamilton) energy rewards.
    """
    alive = [a for a in sim.agents if a.alive]
    for agent in alive:
        ax, ay = agent.pos
        nearby = [
            b
            for b in alive
            if b is not agent and abs(b.pos[0] - ax) <= 2 and abs(b.pos[1] - ay) <= 2
        ]
        sim.tribes.consider_join(agent, nearby)
    sim.tribes.update_membership(alive)
    sim.tribes.cleanup(alive)


def _tick_economy(sim, tick: int) -> None:
    """Recompute dynamic resource prices from the population's current holdings."""
    sim.economy.update(sim.agents, sim.tribes)


def _tick_technology(sim, tick: int) -> None:
    """Refresh which causal sequences (technologies) each tribe currently knows."""
    sim.technology.update(sim.agents, sim.tribes)


def _tick_stats(sim, tick: int) -> None:
    """Collect read-only population/world statistics. Ticks last to see post-update state."""
    sim.stats.update(tick, sim.agents, sim.world, sim.tribes, sim.technology)


# `order` preserves the original __init__ construction sequence and the
# seasons -> weather -> world_regrowth dependency chain (world_regrowth is order 25).
# disease (order 35) is a new self-registering module (systems/disease.py), sitting between
# tribes and economy so the same tick's stats/prices reflect new infections.
register_system("seasons", lambda sim: SeasonCycle(), order=10, tick=_tick_seasons)
register_system("weather", lambda sim: WeatherSystem(), order=20, tick=_tick_weather)
register_system("tribes", lambda sim: TribeSystem(), order=30, tick=_tick_tribes)
register_system("economy", lambda sim: EconomySystem(), order=40, tick=_tick_economy)
register_system("technology", lambda sim: TechnologySystem(), order=50, tick=_tick_technology)
register_system("evolution", lambda sim: EvolutionSystem(), order=60)
register_system("stats", lambda sim: StatisticsTracker(), order=70, tick=_tick_stats)
