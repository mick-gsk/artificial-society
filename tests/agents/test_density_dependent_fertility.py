"""Density-dependent fertility (population-stability fix).

Reproduction used to key off the mother's personal energy alone
(``energy >= REPRODUCTION_ENERGY``). Because agents hoard energy far above that
threshold (MAX_ENERGY 240 vs. threshold 60), they bred off fat reserves
accumulated when food was plentiful — blind to current crowding — so the
population overshot the world's carrying capacity and then mass-starved.

A mother now conceives only when the *local* environment can feed another
mouth: local food per nearby agent must clear a floor. That is the logistic
negative feedback real ecosystems have — as density rises, per-capita food
falls and fertility drops, so the population approaches carrying capacity
instead of overshooting it.
"""

from __future__ import annotations

from artificial_society.agents.agent import Agent
from artificial_society.simulation import Simulation


def _sim(seed: int = 31, pop: int = 2):
    return Simulation(
        headless=True,
        seed=seed,
        grid_w=15,
        grid_h=12,
        initial_population=pop,
        load_checkpoint=False,
    )


def _make_pair(sim, x=7, y=6):
    """A fertile female + a fertile male standing together, everyone else gone."""
    mother, father = sim.agents[0], sim.agents[1]
    for a, sex in ((mother, "f"), (father, "m")):
        a.sex = sex
        a.age = 200
        a.energy = 120.0
        a.reproduction_cooldown = 0
        a.pregnant = False
        a._cached_nearby_agents = None
    mother.pos = (x, y)
    father.pos = (x, y)
    return mother, father


def _flood_food(world, x, y, radius, value):
    for cx in range(x - radius - 1, x + radius + 2):
        for cy in range(y - radius - 1, y + radius + 2):
            if world.in_bounds(cx, cy):
                world.set_cell(cx, cy, "food", value)


def test_conceives_when_local_food_is_ample():
    sim = _sim()
    mother, father = _make_pair(sim)
    sim.agents = [mother, father]  # sparse: only the couple
    _flood_food(sim.world, 7, 6, radius=2, value=20.0)

    mother._try_reproduce(sim.world, sim.agents)

    assert mother.pregnant, "ample local food + no crowding must allow conception"


def test_no_conception_when_crowded_and_food_scarce():
    sim = _sim()
    mother, father = _make_pair(sim)
    # Pack the neighbourhood: many mouths competing for little food.
    crowd = [Agent.spawn_random(7, 6) for _ in range(20)]
    for a in crowd:
        a._cached_nearby_agents = None
    sim.agents = [mother, father, *crowd]
    _flood_food(sim.world, 7, 6, radius=2, value=0.5)

    result = mother._try_reproduce(sim.world, sim.agents)

    assert result is None
    assert not mother.pregnant, "a starving, crowded neighbourhood must suppress fertility"


def test_local_food_per_capita_counts_neighbours_and_food():
    sim = _sim()
    mother, _ = _make_pair(sim)
    sim.agents = [mother]  # nobody nearby -> divisor is 1
    _flood_food(sim.world, 7, 6, radius=2, value=4.0)

    pc_alone = mother._local_food_per_capita(sim.world, sim.agents)

    # 5x5 = 25 cells * 4.0 food, divided by (0 neighbours + 1).
    assert pc_alone == 100.0

    mother._cached_nearby_agents = None
    crowd = [Agent.spawn_random(7, 6) for _ in range(3)]
    sim.agents = [mother, *crowd]
    pc_crowded = mother._local_food_per_capita(sim.world, sim.agents)

    # same food, now divided by (3 neighbours + 1).
    assert pc_crowded == 25.0
