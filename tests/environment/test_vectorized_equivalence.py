"""The vectorized world update must be BIT-IDENTICAL to the scalar reference.

``regrow_grid`` + ``event_fields_grid`` + the vectorized ``diffuse_fields`` are
transliterations of ``regrow_cell`` + ``event_field`` + the old per-cell
diffusion. Neither path has cross-cell reads or RNG, so exact float equality is
the contract — any divergence means a transliteration bug, not tolerance noise.
"""

from __future__ import annotations

import copy

import numpy as np

from artificial_society.cell_store import FLOAT_FIELDS
from artificial_society.environment.resources import diffuse_step, regrow_cell, regrow_grid
from artificial_society.rng import seed_all
from artificial_society.world import World

SEASON = {"food_factor": 0.85, "temperature_shift": -3.0}
WEATHER = {"rain_map": 0.4, "temperature_shift": 1.5, "wind": 3.0, "storm_risk": 0.25}
TICK = 7


def _scalar_reference_update(world) -> None:
    """The pre-vectorization update loop, verbatim (minus herbs, which kept
    their own per-cell loop in the live path and are not part of this
    comparison)."""
    for y in range(world.height):
        for x in range(world.width):
            regrow_cell(
                world,
                x,
                y,
                world.biomes[y][x],
                SEASON,
                WEATHER,
                TICK,
                world.event_field(x, y),
            )
    avgs = [[None for _ in range(world.width)] for _ in range(world.height)]
    for y in range(world.height):
        for x in range(world.width):
            cells = [c for _, _, c, _ in world.neighbors(x, y, 1)]
            n = len(cells)
            avgs[y][x] = {
                "pollution": sum(c["pollution"] for c in cells) / n,
                "disease": sum(c["disease"] for c in cells) / n,
                "moisture": sum(c["moisture"] for c in cells) / n,
                "ash": sum(c["ash"] for c in cells) / n,
                "disturbance": sum(c["disturbance"] for c in cells) / n,
            }
    for y in range(world.height):
        for x in range(world.width):
            diffuse_step(world, x, y, avgs[y][x])


def test_vectorized_world_update_bit_identical():
    seed_all(1234)
    world = World(40, 30)

    # Diversify state so every code path has signal: field noise, structures,
    # and at least one active disturbance event.
    rng = np.random.default_rng(99)
    for k in ("plant_food", "meat_food", "pollution", "spoilage", "carcasses", "usage_pressure"):
        world.F[k] = np.clip(world.F[k] + rng.uniform(0.0, 30.0, world.F[k].shape), 0.0, 100.0)
    world.get_cell(5, 5)["structures"]["farm"] = 1.0
    world.get_cell(9, 9)["structures"]["well"] = 1.0
    world.get_cell(12, 3)["structures"]["camp"] = 1.0
    world.update_events(0, SEASON, WEATHER)  # tick % 55 == 0 -> spawns an event
    assert world.active_events, "test setup should have at least one active event"

    scalar_world = copy.deepcopy(world)
    vector_world = copy.deepcopy(world)

    _scalar_reference_update(scalar_world)
    regrow_grid(vector_world, SEASON, WEATHER, TICK, vector_world.event_fields_grid())
    vector_world.diffuse_fields()

    for field in FLOAT_FIELDS:
        assert (scalar_world.F[field] == vector_world.F[field]).all(), (
            f"vectorized field {field!r} diverged from the scalar reference"
        )
    assert (scalar_world.tick_grid == vector_world.tick_grid).all()
