"""Scarcity properties of the resource-regrowth system.

These tests guard the *environment-lane* invariant that foraged food is
genuinely scarce: regrowth must NOT refill cells to abundance each tick.

The assertions are made on CELL state (food / carrying_capacity), never on
agent energy -- agent energy is influenced by core systems (cooperation,
Hamilton, territory) that are out of scope for this lane.

Deterministic and fast: small grid, few agents, short headless run.
"""

from __future__ import annotations

import statistics

from artificial_society.environment.resources import (
    SCARCITY_CEILING_FACTOR,
    initial_cell_state,
    regrow_cell,
)
from artificial_society.simulation import Simulation


class _OneCellWorld:
    """Minimal World stand-in holding a single cell, exposing the same
    get_cell / set_cell mutation façade regrow_cell now routes through.

    Phase 3 made World the authoritative cell-state owner, so regrow_cell takes
    (world, x, y, ...) instead of a bare cell. These unit tests exercise the
    regrowth math on one controlled cell, so a one-cell store is sufficient and
    keeps them deterministic (no biome-grid randomness)."""

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


def _mean_food_ratio(world) -> float:
    """Mean of per-cell food / carrying_capacity over all land cells."""
    ratios = []
    for row in world.cells:
        for cell in row:
            cap = cell["carrying_capacity"]
            if cap <= 0:  # water cells have no capacity
                continue
            ratios.append(cell["food"] / cap)
    return statistics.mean(ratios)


def test_standing_food_stays_scarce_in_headless_run():
    """After a short seeded run the mean per-cell food is well below
    half the carrying capacity -- i.e. the world is not saturated."""
    sim = Simulation(
        headless=True,
        grid_w=18,
        grid_h=12,
        initial_population=6,
    )
    for _ in range(60):
        sim.step()

    ratio = _mean_food_ratio(sim.world)
    # Genuine scarcity: standing food < 50% of carrying capacity.
    assert ratio < 0.5, f"world not scarce: mean food/capacity = {ratio:.3f}"


def test_untouched_cell_equilibrium_below_capacity():
    """An UNTOUCHED, never-foraged cell must NOT grow to abundance.

    Regression guard: previously regrowth saturated standing food to
    ~1.0-1.2x carrying_capacity. With the logistic ceiling it must
    plateau below the carrying capacity for every productive biome.
    """
    for biome in ("forest", "grassland", "swamp"):
        world = _OneCellWorld(initial_cell_state(biome))
        cell = world.get_cell(0, 0)
        cap = cell["carrying_capacity"]
        # Let it regrow undisturbed for a long time (no consumption).
        for tick in range(1000):
            regrow_cell(
                world,
                0,
                0,
                biome,
                {"food_factor": 1.0},
                {"rain_map": 0.4},
                tick,
                {},
            )
        ratio = cell["food"] / cap
        assert ratio < 0.6, f"{biome}: untouched cell saturated to food/capacity = {ratio:.3f}"


def test_depleted_cell_recovers_slowly():
    """Recovery must be slow relative to consumption: a cell depleted to
    zero stays nearly empty for many ticks (over-use depletes, recovery
    is slow)."""
    world = _OneCellWorld(initial_cell_state("forest"))
    cell = world.get_cell(0, 0)
    cap = cell["carrying_capacity"]
    # Bring it to its natural equilibrium first.
    for tick in range(800):
        regrow_cell(world, 0, 0, "forest", {"food_factor": 1.0}, {"rain_map": 0.4}, tick, {})
    # Forage everything.
    world.set_cell(0, 0, "plant_food", 0.0)
    world.set_cell(0, 0, "meat_food", 0.0)
    world.set_cell(0, 0, "food", 0.0)
    # Regrow for a modest window.
    for tick in range(30):
        regrow_cell(world, 0, 0, "forest", {"food_factor": 1.0}, {"rain_map": 0.4}, 900 + tick, {})
    ratio = cell["food"] / cap
    # 30 ticks of regrowth must not refill a depleted cell to abundance.
    assert ratio < 0.15, f"depleted cell refilled too fast: ratio={ratio:.3f}"


def test_scarcity_ceiling_factor_is_scarce():
    """Sanity guard on the tunable: the configured ceiling keeps the
    standing-stock target below half of capacity."""
    assert 0.0 < SCARCITY_CEILING_FACTOR < 0.5
