"""Wind must not sterilise the world (population-stability root-cause fix).

The plant update used to subtract a *flat* ``0.012 * wind`` every tick. Wind
swings 0..8 (mean ~5), so that term stripped ~0.06 plant/cell/tick — 6-9x the
~0.01/tick regrowth — from *every* cell regardless of how much was standing.
Vegetation therefore collapsed to zero within a few hundred ticks and never
recovered, even with nobody eating it (verified by a no-consumption run), which
capped the whole world at a starvation carrying capacity.

Wind damage is now proportional to the standing biomass it acts on
(``WIND_PLANT_LOSS * wind * plant_food``): it still trims lush cells but cannot
strip bare ground, so growth always rebuilds the base to an equilibrium. These
tests pin both halves of that contract and stay agnostic to the exact constant.
"""

from __future__ import annotations

import copy

from artificial_society.environment.resources import regrow_cell
from artificial_society.rng import seed_all
from artificial_society.world import World

SEASON = {"food_factor": 1.0, "temperature_shift": 0.0}
CALM = {"rain_map": 0.3, "temperature_shift": 0.0, "wind": 0.0}
STORM = {"rain_map": 0.3, "temperature_shift": 0.0, "wind": 8.0}
TICK = 5


def _fertile_cell(world):
    """Find a cell whose biome can actually grow plants and prime it to grow."""
    for y in range(world.height):
        for x in range(world.width):
            if world.biomes[y][x] not in ("water", "desert"):
                for field, val in (
                    ("soil_fertility", 70.0),
                    ("moisture", 70.0),
                    ("carrying_capacity", 50.0),
                    ("pollution", 0.0),
                    ("usage_pressure", 0.0),
                    ("disturbance", 0.0),
                    ("ash", 0.0),
                ):
                    world.set_cell(x, y, field, val)
                return x, y
    raise AssertionError("no plant-capable biome in the test world")


def _regrow(world, x, y, weather):
    regrow_cell(world, x, y, world.biomes[y][x], SEASON, weather, TICK, world.event_field(x, y))


def test_bare_cell_regrows_even_in_a_storm():
    seed_all(7)
    world = World(30, 20)
    x, y = _fertile_cell(world)

    # Sanity: the cell is genuinely growth-capable when calm.
    world.set_cell(x, y, "plant_food", 0.0)
    _regrow(world, x, y, CALM)
    assert world.get_cell(x, y)["plant_food"] > 0.0, "test setup: cell must be able to grow"

    # The real contract: a fully grazed cell must still recover under max wind —
    # wind may not strip biomass that isn't there.
    world.set_cell(x, y, "plant_food", 0.0)
    _regrow(world, x, y, STORM)
    assert world.get_cell(x, y)["plant_food"] > 0.0, (
        "a storm must not stop bare ground from regrowing (flat wind decay bug)"
    )


def test_wind_still_damages_standing_vegetation():
    seed_all(7)
    world = World(30, 20)
    x, y = _fertile_cell(world)
    world.set_cell(x, y, "plant_food", 10.0)

    calm_world = copy.deepcopy(world)
    storm_world = copy.deepcopy(world)
    _regrow(calm_world, x, y, CALM)
    _regrow(storm_world, x, y, STORM)

    assert storm_world.get_cell(x, y)["plant_food"] < calm_world.get_cell(x, y)["plant_food"], (
        "wind must still cost a lush cell some biomass (term not neutralised)"
    )
