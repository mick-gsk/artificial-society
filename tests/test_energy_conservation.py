"""Energy conservation (Phase 4) — the core physics invariant.

The simulation is meant to run on genuine scarcity + selection. That only works
if energy is *conserved*: foraging must be a transfer from the finite world, and
social bonuses must move energy around rather than mint it.

These tests lock the invariant at the unit level — the only legitimate energy
*sources* (regrowth, sleep) and *sinks* (metabolism, death) are deliberately
excluded so a single operation can be checked in isolation:

- foraging transfers exactly the food it removes (plant / carcass / meat paths),
- the carcass meat is actually reachable (regression for the `carcass`/`carcasses`
  field-name bug — carnivores used to read a key the world never wrote),
- a corpse is worth exactly CORPSE_ENERGY of harvestable food (not double-counted
  across the carcasses and meat_food pools),
- cooperation and Hamilton kin-rewards conserve total energy (redistributive,
  never generative),
- consumption actually *persists* against regrowth (the original bug debited the
  derived `food` aggregate, which `regrow_cell` overwrote each tick).

Phase 3 made World the authoritative cell-state owner, so the resources helpers
take (world, x, y, ...); cell-level tests use a one-cell World stand-in.
"""

from __future__ import annotations

import pytest

from artificial_society.agents.agent import CORPSE_ENERGY, MAX_ENERGY
from artificial_society.environment.resources import (
    add_carcass,
    apply_consumption,
    initial_cell_state,
    regrow_cell,
)
from artificial_society.simulation import HAMILTON_TICK_INTERVAL, Simulation

EPS = 1e-6


class _OneCellWorld:
    """Minimal World stand-in holding a single cell, exposing the get_cell /
    set_cell façade the resources helpers route through."""

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


def _sim():
    """Small, fast, deterministic headless sim — a source of valid agents/world."""
    return Simulation(
        headless=True,
        load_checkpoint=False,
        seed=42,
        grid_w=10,
        grid_h=8,
        initial_population=6,
    )


def _put_agent_on_land(sim, agent):
    """Park an agent on a known land cell and return that cell."""
    x, y = sim.world.land_positions[0]
    agent.pos = (x, y)
    return sim.world.get_cell(x, y)


# ---------------------------------------------------------------------------
# Foraging is a conservative transfer
# ---------------------------------------------------------------------------


def test_herbivore_forage_transfers_exactly_the_plant_food_it_removes():
    sim = _sim()
    agent = sim.agents[0]
    agent.genes["diet_preference"] = -0.6  # herbivore
    agent.energy = 50.0
    agent.hydration = 100.0  # isolate: skip the water branch
    cell = _put_agent_on_land(sim, agent)
    cell["plant_food"] = 60.0
    cell["meat_food"] = 0.0
    cell["food"] = cell["plant_food"]

    e0 = agent.energy
    p0 = cell["plant_food"]
    agent._forage(sim.world, {})
    gained = agent.energy - e0
    removed = p0 - cell["plant_food"]

    assert gained > 0.0, "herbivore on a plant-rich cell should gain energy"
    assert gained == pytest.approx(removed, abs=EPS), "energy minted: gain != food removed"


def test_carnivore_eats_carcasses_and_transfers_exactly_what_it_removes():
    """Regression for the carcass/carcasses bug: carnivores read `cell['carcass']`
    but the world only ever stored `cell['carcasses']`, so carcass meat was
    invisible. The carcass is also NOT part of the derived `food` aggregate, so
    the old outer `if food_available > 0` guard skipped it entirely."""
    sim = _sim()
    agent = sim.agents[0]
    agent.genes["diet_preference"] = 0.8  # carnivore
    agent.energy = 50.0
    agent.hydration = 100.0
    cell = _put_agent_on_land(sim, agent)
    cell["plant_food"] = 0.0
    cell["meat_food"] = 0.0
    cell["carcasses"] = 40.0
    cell["food"] = 0.0  # nothing in the derived aggregate — only carcass meat

    e0 = agent.energy
    c0 = cell["carcasses"]
    agent._forage(sim.world, {})
    gained = agent.energy - e0
    removed = c0 - cell["carcasses"]

    assert gained > 0.0, "carcass meat is invisible to the carnivore (bug not fixed)"
    assert removed > 0.0, "carcass pool was not debited"
    assert gained == pytest.approx(removed, abs=EPS)


def test_carnivore_meat_pool_forage_is_conservative():
    sim = _sim()
    agent = sim.agents[0]
    agent.genes["diet_preference"] = 0.8
    agent.energy = 50.0
    agent.hydration = 100.0
    cell = _put_agent_on_land(sim, agent)
    cell["plant_food"] = 0.0
    cell["meat_food"] = 50.0
    cell["carcasses"] = 0.0
    cell["food"] = cell["meat_food"] * 0.7

    e0 = agent.energy
    m0 = cell["meat_food"]
    agent._forage(sim.world, {})
    gained = agent.energy - e0
    removed = m0 - cell["meat_food"]

    assert gained > 0.0
    assert gained == pytest.approx(removed, abs=EPS)


# ---------------------------------------------------------------------------
# A corpse is worth exactly CORPSE_ENERGY (no double-counting)
# ---------------------------------------------------------------------------


def test_corpse_total_harvestable_energy_equals_corpse_value():
    """A death deposits exactly CORPSE_ENERGY of harvestable food, split across
    the two consumable pools a forager can reach (carcasses + meat_food). They
    must SUM to the corpse value -- not each receive it -- or every death mints
    ~1.45x energy now that the carcasses pool is actually edible."""
    world = _OneCellWorld(initial_cell_state("grassland"))
    cell = world.get_cell(0, 0)
    carc0, meat0 = cell["carcasses"], cell["meat_food"]
    add_carcass(world, 0, 0, CORPSE_ENERGY)
    harvestable = (cell["carcasses"] - carc0) + (cell["meat_food"] - meat0)
    assert harvestable == pytest.approx(CORPSE_ENERGY, abs=1e-6), (
        "a corpse is double-counted across carcasses and meat_food"
    )


# ---------------------------------------------------------------------------
# Social bonuses redistribute, never mint
# ---------------------------------------------------------------------------


def test_cooperation_forage_bonus_is_zero_sum():
    sim = _sim()
    a, b, c = sim.agents[0], sim.agents[1], sim.agents[2]
    for agent in (a, b, c):
        agent.alive = True
        agent.pos = (2, 2)
        agent.tribe_id = None
    # self is the poorest; b and c are richer donors (and below the 160 donor
    # threshold so the separate altruistic-share block stays out of it).
    a.energy, b.energy, c.energy = 80.0, 130.0, 120.0
    total0 = a.energy + b.energy + c.energy

    a._cooperate([a, b, c], {}, tick=0)
    total1 = a.energy + b.energy + c.energy

    assert total1 == pytest.approx(total0, abs=1e-4), "cooperation minted/destroyed energy"
    assert a.energy > 80.0, "cooperator should have received the pooled bonus"
    assert (b.energy < 130.0) or (c.energy < 120.0), "donors should have contributed"


def test_cooperation_never_mints_when_no_richer_neighbour():
    """If no nearby member is better off than self, there is nobody to pool from,
    so self must not gain free energy."""
    sim = _sim()
    a, b = sim.agents[0], sim.agents[1]
    for agent in (a, b):
        agent.alive = True
        agent.pos = (3, 3)
        agent.tribe_id = None
    a.energy, b.energy = 120.0, 90.0  # self richer than the only neighbour
    total0 = a.energy + b.energy

    a._cooperate([a, b], {}, tick=0)

    assert a.energy + b.energy == pytest.approx(total0, abs=1e-4)
    assert a.energy <= 120.0 + EPS, "cooperator minted energy with no donor present"


def test_hamilton_rewards_conserve_total_energy():
    sim = _sim()
    sim.tick = HAMILTON_TICK_INTERVAL  # make the interval gate fire
    members = sim.agents[:3]
    for i, m in enumerate(members):
        m.alive = True
        m.tribe_id = 7
        m.energy = 100.0 + 10.0 * i
        m.children = i  # differentiate kin weight
    # Everyone else: solo, untouched by redistribution.
    for other in sim.agents[3:]:
        other.tribe_id = None

    alive = [m for m in sim.agents if m.alive]
    total0 = sum(m.energy for m in alive)
    sim._apply_hamilton_rewards()
    total1 = sum(m.energy for m in alive)

    assert total1 <= total0 + EPS, "Hamilton minted energy"
    assert total1 == pytest.approx(total0, abs=1e-4), "Hamilton was not zero-sum"
    # Redistribution actually happened (kin weights differ -> shares differ).
    assert any(m.energy != (100.0 + 10.0 * i) for i, m in enumerate(members))


def test_hamilton_redistributes_toward_kin_with_children():
    sim = _sim()
    sim.tick = HAMILTON_TICK_INTERVAL
    childless, parent = sim.agents[0], sim.agents[1]
    for m in (childless, parent):
        m.alive = True
        m.tribe_id = 9
        m.energy = 120.0
    childless.children = 0
    parent.children = 4
    for other in sim.agents[2:]:
        other.tribe_id = None

    sim._apply_hamilton_rewards()

    assert parent.energy > childless.energy, "kin-investment weighting not applied"


def test_hamilton_does_not_mint_from_a_negative_energy_member():
    """A member that has gone negative (reachable via the unguarded pregnancy
    metabolism tick) must contribute 0 to the pool and must NOT be silently lifted
    toward 0 -- the old min()-based tax did exactly that, and when the resulting
    pool went non-positive the lift was no longer offset, minting energy. Stressed
    with a strongly-negative member so the pool flips sign and the bug would show."""
    sim = _sim()
    sim.tick = HAMILTON_TICK_INTERVAL
    a, b, c = sim.agents[:3]
    for m in (a, b, c):
        m.alive = True
        m.tribe_id = 5
        m.children = 0
    a.energy = -50.0
    b.energy = 60.0
    c.energy = 70.0
    for other in sim.agents[3:]:
        other.tribe_id = None

    total0 = a.energy + b.energy + c.energy
    sim._apply_hamilton_rewards()
    total1 = a.energy + b.energy + c.energy

    assert total1 == pytest.approx(total0, abs=1e-4), "Hamilton minted/destroyed energy"
    assert a.energy < 0.0, "negative member was silently lifted toward 0 (energy minted)"


# ---------------------------------------------------------------------------
# Consumption persists against regrowth (the original minting mechanism)
# ---------------------------------------------------------------------------


def test_consumption_persists_against_regrowth():
    """The original bug debited the derived `food` field, which `regrow_cell`
    recomputes from `plant_food`/`meat_food` every tick — so the debit vanished
    and energy was effectively free. Routing through `apply_consumption` debits
    the source pool, so a consumed cell stays measurably more depleted than an
    untouched twin under an identical regrow schedule."""
    consumed = _OneCellWorld(initial_cell_state("forest"))
    untouched = _OneCellWorld(initial_cell_state("forest"))
    season, weather = {"food_factor": 1.0}, {"rain_map": 0.4}

    # Grow both to their natural equilibrium.
    for tick in range(400):
        regrow_cell(consumed, 0, 0, "forest", season, weather, tick, {})
        regrow_cell(untouched, 0, 0, "forest", season, weather, tick, {})

    # Now forage one of them every tick for a window.
    for tick in range(400, 460):
        apply_consumption(consumed, 0, 0, plant=6.0)
        regrow_cell(consumed, 0, 0, "forest", season, weather, tick, {})
        regrow_cell(untouched, 0, 0, "forest", season, weather, tick, {})

    assert consumed.get_cell(0, 0)["plant_food"] < untouched.get_cell(0, 0)["plant_food"] - 1.0, (
        "consumption did not persist — debit was erased by regrowth"
    )


def test_forage_respects_energy_cap_without_minting_beyond_it():
    sim = _sim()
    agent = sim.agents[0]
    agent.genes["diet_preference"] = -0.6
    agent.energy = MAX_ENERGY - 1.0
    agent.hydration = 100.0
    cell = _put_agent_on_land(sim, agent)
    cell["plant_food"] = 60.0
    cell["food"] = cell["plant_food"]

    agent._forage(sim.world, {})
    assert agent.energy <= MAX_ENERGY + EPS
