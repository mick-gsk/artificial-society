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


# `order` preserves the original __init__ construction sequence and the
# seasons -> weather -> world_regrowth dependency chain (world_regrowth is order 25).
register_system("seasons", lambda sim: SeasonCycle(), order=10, tick=_tick_seasons)
register_system("weather", lambda sim: WeatherSystem(), order=20, tick=_tick_weather)
register_system("tribes", lambda sim: TribeSystem(), order=30)
register_system("economy", lambda sim: EconomySystem(), order=40)
register_system("technology", lambda sim: TechnologySystem(), order=50)
register_system("evolution", lambda sim: EvolutionSystem(), order=60)
register_system("stats", lambda sim: StatisticsTracker(), order=70)
