"""Registration of the built-in (legacy) systems with the registry.

These predate the registry and are registered here in one place so ``simulation.py``
no longer hard-codes them in ``__init__``. **Do not add new systems here** — instead
create a module under ``systems/`` that self-registers via ``@register(...)``;
:func:`artificial_society.systems.registry.discover` picks it up automatically.

Every system is registered with ``tick=None`` (dormant): they are *constructed* but
not ticked from ``simulation.step()`` today (the ``TODO(phase2)`` re-wiring). Keeping
them dormant makes the registry switch exactly behaviour-preserving.
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

# `order` preserves the original __init__ construction sequence. It is
# determinism-neutral (none of these constructors draw from the seeded RNG) but
# kept tidy and explicit.
register_system("seasons", lambda sim: SeasonCycle(), order=10)
register_system("weather", lambda sim: WeatherSystem(), order=20)
register_system("tribes", lambda sim: TribeSystem(), order=30)
register_system("economy", lambda sim: EconomySystem(), order=40)
register_system("technology", lambda sim: TechnologySystem(), order=50)
register_system("evolution", lambda sim: EvolutionSystem(), order=60)
register_system("stats", lambda sim: StatisticsTracker(), order=70)
