"""Spawn must TRANSFER energy from the mother, not mint it (audit fix 23).

Newborns spawn with SPAWN_START_ENERGY while parents pay only ~REPLICATION_COST
at conception. The old net mint per spawn subsidised population overshoot far
past the world's food-regrowth carrying capacity, ending in mass depletion.
Spawn is now a transfer capped by SPAWN_ENERGY_FLOOR: a depleting mother bears a
weak spawn. (Ported onto the physik-v2 path — energy only, no learned
derivation; Plan-4 stays intact.)
"""

from __future__ import annotations

import pytest

from artificial_society.agents.agent import SPAWN_ENERGY_FLOOR, SPAWN_START_ENERGY
from artificial_society.simulation import Simulation


def _sim(seed):
    return Simulation(
        headless=True, seed=seed, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8
    )


def test_spawn_transfers_energy_from_the_mother():
    sim = _sim(seed=23)
    mother = sim.agents[0]
    mother.sex = "f"
    mother.energy = 200.0
    mother._last_partner_id = None

    total_before = mother.energy
    spawn = sim.spawn_agent_from_parent(mother, dict(mother.traits))

    assert spawn.energy == pytest.approx(SPAWN_START_ENERGY)  # rich mother: full start
    assert spawn.energy + mother.energy == pytest.approx(total_before), (
        "birth must conserve energy (transfer, not mint)"
    )


def test_depleting_mother_bears_weak_spawn():
    sim = _sim(seed=23)
    mother = sim.agents[0]
    mother.sex = "f"
    mother.energy = 40.0
    mother._last_partner_id = None

    spawn = sim.spawn_agent_from_parent(mother, dict(mother.traits))

    assert spawn.energy == pytest.approx(40.0 - SPAWN_ENERGY_FLOOR), (
        "a poor mother can only afford a weak child"
    )
