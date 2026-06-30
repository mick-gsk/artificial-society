"""Biome-specific scarcity + temperature/season coupling (Phase 4, environment).

Per the lane convention these assert on CELL state only (never agent energy):

- harsh biomes (desert, mountain) plateau at a lower standing food stock than
  fertile ones, so location is a real selection pressure;
- temperature couples into regrowth (cold/heat suppress plant growth) so seasons
  bite — winter is a genuine food-scarcity event, not cosmetic;
- cold couples into danger so agents perceive (and migrate away from) winter
  zones.

Deterministic and fast: pure `regrow_cell` math on a one-cell World stand-in
(Phase 3 made regrow_cell take (world, x, y, ...)), no sim runs.
"""

from __future__ import annotations

import pytest

from artificial_society.environment.resources import (
    BIOME_SCARCITY_CEILING,
    PLANT_TEMP_FLOOR,
    PLANT_TEMP_OPTIMUM,
    biome_scarcity_ceiling,
    initial_cell_state,
    plant_temperature_factor,
    regrow_cell,
)


class _OneCellWorld:
    """Minimal World stand-in exposing the get_cell / set_cell façade regrow_cell
    routes through, holding a single controlled cell."""

    def __init__(self, cell):
        self.cells = [[cell]]
        self.width = 1
        self.height = 1

    def get_cell(self, x, y):
        return self.cells[0][0]

    def set_cell(self, x, y, key, value):
        self.get_cell(x, y)[key] = value

    def adjust_cell(self, x, y, **deltas):
        cell = self.get_cell(x, y)
        for key, delta in deltas.items():
            cell[key] = cell.get(key, 0.0) + delta


TEMPERATE = ({"food_factor": 1.0}, {"rain_map": 0.4})


def _equilibrium_food_ratio(biome, ticks=1000):
    world = _OneCellWorld(initial_cell_state(biome))
    cell = world.get_cell(0, 0)
    season, weather = TEMPERATE
    for tick in range(ticks):
        regrow_cell(world, 0, 0, biome, season, weather, tick, {})
    return cell["food"] / cell["carrying_capacity"]


# ---------------------------------------------------------------------------
# Biome-specific scarcity
# ---------------------------------------------------------------------------


def test_biome_scarcity_ceiling_is_ordered_and_scarce():
    c = BIOME_SCARCITY_CEILING
    # Fertile > harsh, and everything stays genuinely scarce (< half capacity)
    # so the global scarcity sanity (test_scarcity.py) is preserved.
    assert c["forest"] >= c["grassland"] >= c["swamp"] > c["mountain"] > c["desert"]
    assert c["water"] == 0.0
    assert all(0.0 <= v < 0.5 for v in c.values())


def test_unlisted_biome_falls_back_to_global_default():
    from artificial_society.environment.resources import SCARCITY_CEILING_FACTOR

    assert biome_scarcity_ceiling("forest") == BIOME_SCARCITY_CEILING["forest"]
    assert biome_scarcity_ceiling("no_such_biome") == SCARCITY_CEILING_FACTOR


def test_harsh_biomes_plateau_below_fertile_ones():
    forest = _equilibrium_food_ratio("forest")
    grassland = _equilibrium_food_ratio("grassland")
    desert = _equilibrium_food_ratio("desert")
    mountain = _equilibrium_food_ratio("mountain")

    assert desert < forest, "desert standing food should be far below forest"
    assert desert < grassland
    assert mountain < forest, "mountain standing food should be below forest"


# ---------------------------------------------------------------------------
# Temperature -> regrowth
# ---------------------------------------------------------------------------


def test_plant_temperature_factor_peaks_at_optimum():
    assert plant_temperature_factor(PLANT_TEMP_OPTIMUM) == pytest.approx(1.0)
    cold = plant_temperature_factor(PLANT_TEMP_OPTIMUM - 30)
    hot = plant_temperature_factor(PLANT_TEMP_OPTIMUM + 30)
    assert cold < 1.0 and hot < 1.0
    assert all(PLANT_TEMP_FLOOR <= plant_temperature_factor(t) <= 1.0 for t in range(-20, 55))


def test_cold_suppresses_regrowth():
    """A cell forced cold regrows less plant food than a temperate twin."""
    warm = _OneCellWorld(initial_cell_state("forest"))
    cold = _OneCellWorld(initial_cell_state("forest"))
    # Deplete both, then regrow under temperate vs hard-winter conditions.
    for w in (warm, cold):
        w.set_cell(0, 0, "plant_food", 0.0)
    warm_season, warm_weather = {"food_factor": 1.0}, {"rain_map": 0.4}
    # Winter: low food_factor + a strong negative temperature shift -> cold cell.
    cold_season = {"food_factor": 1.0, "temperature_shift": -10}
    cold_weather = {"rain_map": 0.4, "temperature_shift": -10}
    for tick in range(120):
        regrow_cell(warm, 0, 0, "forest", warm_season, warm_weather, tick, {})
        regrow_cell(cold, 0, 0, "forest", cold_season, cold_weather, tick, {})

    assert cold.get_cell(0, 0)["plant_food"] < warm.get_cell(0, 0)["plant_food"], (
        "temperature did not couple into regrowth"
    )


def test_winter_season_reduces_standing_food_vs_spring():
    spring = _OneCellWorld(initial_cell_state("grassland"))
    winter = _OneCellWorld(initial_cell_state("grassland"))
    spring_season = {"food_factor": 1.35, "temperature_shift": 2}
    winter_season = {"food_factor": 0.55, "temperature_shift": -8}
    weather = {"rain_map": 0.3}
    for tick in range(600):
        regrow_cell(spring, 0, 0, "grassland", spring_season, weather, tick, {})
        regrow_cell(winter, 0, 0, "grassland", winter_season, weather, tick, {})

    assert winter.get_cell(0, 0)["food"] < spring.get_cell(0, 0)["food"], (
        "winter is not a real food-scarcity event"
    )


# ---------------------------------------------------------------------------
# Cold -> danger (perceptual: drives migration)
# ---------------------------------------------------------------------------


def test_cold_cell_is_more_dangerous_than_warm_cell():
    warm = _OneCellWorld(initial_cell_state("grassland"))
    cold = _OneCellWorld(initial_cell_state("grassland"))
    warm_weather = {"rain_map": 0.2, "temperature_shift": 6}
    cold_weather = {"rain_map": 0.2, "temperature_shift": -16}
    regrow_cell(warm, 0, 0, "grassland", {"food_factor": 1.0}, warm_weather, 0, {})
    regrow_cell(
        cold, 0, 0, "grassland", {"food_factor": 1.0, "temperature_shift": -8}, cold_weather, 0, {}
    )

    assert cold.get_cell(0, 0)["danger"] > warm.get_cell(0, 0)["danger"], (
        "cold did not raise perceived danger"
    )
