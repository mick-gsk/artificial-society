"""Birth must TRANSFER energy from the mother, not mint it (audit fix 23).

Newborns spawn with CHILD_START_ENERGY while parents pay only ~REPRODUCTION_COST
at conception. The old net mint per birth subsidised population overshoot far
past the world's food-regrowth carrying capacity, ending in mass starvation.
Birth is now a transfer capped by BIRTH_ENERGY_FLOOR: a starving mother bears a
weak child. (Ported onto the physik-v2 path — energy only, no learned
inheritance; Plan-4 stays intact.)
"""

from __future__ import annotations

import pytest

from artificial_society.agents.agent import BIRTH_ENERGY_FLOOR, CHILD_START_ENERGY
from artificial_society.simulation import Simulation


def _sim(seed):
    return Simulation(
        headless=True, seed=seed, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8
    )


def test_birth_transfers_energy_from_the_mother():
    sim = _sim(seed=23)
    mother = sim.agents[0]
    mother.sex = "f"
    mother.energy = 200.0
    mother._last_mate_id = None

    total_before = mother.energy
    child = sim.spawn_child_from_parent(mother, dict(mother.genes))

    assert child.energy == pytest.approx(CHILD_START_ENERGY)  # rich mother: full start
    assert child.energy + mother.energy == pytest.approx(total_before), (
        "birth must conserve energy (transfer, not mint)"
    )


def test_starving_mother_bears_weak_child():
    sim = _sim(seed=23)
    mother = sim.agents[0]
    mother.sex = "f"
    mother.energy = 40.0
    mother._last_mate_id = None

    child = sim.spawn_child_from_parent(mother, dict(mother.genes))

    assert child.energy == pytest.approx(40.0 - BIRTH_ENERGY_FLOOR), (
        "a poor mother can only afford a weak child"
    )
