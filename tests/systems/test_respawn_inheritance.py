"""B2 regression: emergency respawn must use principled inheritance.

The old crutch injected fresh age-0 agents at random SCATTERED cells; they landed
alone and died before reaching MIN_REPRODUCTION_AGE, so respawns fired forever.
The fix spawns a replacement ADJACENT to a living agent, with inherited traits and
a non-zero juvenile age. The old behavior stays selectable via RESPAWN_INHERITANCE
for a crutch-off A/B experiment.
"""

import artificial_society.simulation as sim_mod
from artificial_society.simulation import Simulation


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _fresh_sim():
    return Simulation(headless=True, grid_w=20, grid_h=15, initial_population=6)


def test_inheritance_respawn_is_adjacent_and_juvenile():
    sim = _fresh_sim()
    living_before = [a for a in sim.agents if a.alive]
    living_positions = [a.pos for a in living_before]
    n_before = len(sim.agents)

    saved = sim_mod.RESPAWN_INHERITANCE
    try:
        sim_mod.RESPAWN_INHERITANCE = True
        sim.emergency_respawn()
    finally:
        sim_mod.RESPAWN_INHERITANCE = saved

    new_agents = sim.agents[n_before:]
    assert len(new_agents) == sim_mod.RESPAWN_COUNT
    for child in new_agents:
        # non-zero juvenile age head-start
        assert child.age == sim_mod.RESPAWN_JUVENILE_AGE > 0
        # spawned adjacent to (or on top of) some agent that was alive before
        assert any(_chebyshev(child.pos, p) <= 1 for p in living_positions), (
            f"respawn at {child.pos} is not adjacent to any living agent"
        )


def test_ab_flag_off_restores_scattered_age_zero_crutch():
    sim = _fresh_sim()
    n_before = len(sim.agents)

    saved = sim_mod.RESPAWN_INHERITANCE
    try:
        sim_mod.RESPAWN_INHERITANCE = False
        sim.emergency_respawn()
    finally:
        sim_mod.RESPAWN_INHERITANCE = saved

    new_agents = sim.agents[n_before:]
    assert len(new_agents) == sim_mod.RESPAWN_COUNT
    # Old crutch: fresh age-0 strangers.
    assert all(child.age == 0 for child in new_agents)


def test_falls_back_to_scatter_when_no_living_agent():
    sim = _fresh_sim()
    for a in sim.agents:
        a.alive = False
    n_before = len(sim.agents)

    saved = sim_mod.RESPAWN_INHERITANCE
    try:
        sim_mod.RESPAWN_INHERITANCE = True  # still falls back: nobody to inherit from
        sim.emergency_respawn()
    finally:
        sim_mod.RESPAWN_INHERITANCE = saved

    new_agents = sim.agents[n_before:]
    assert len(new_agents) == sim_mod.RESPAWN_COUNT
    assert all(child.age == 0 for child in new_agents)
