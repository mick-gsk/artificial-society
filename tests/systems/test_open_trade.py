"""Lane test: trade is open to any held material, not just wood/stone/fiber.

Phase 5 de-scripting (Task 1). The legacy ``EconomySystem.maybe_trade`` only ever
bartered the hardcoded ``("wood", "stone", "fiber")`` triad over ``agent.resources``.
After de-scripting, two trusted, adjacent agents barter complementary surpluses of *any*
material they hold in ``material_inventory`` (named or discovered ``mat_*``), so emergent
specialisation is no longer capped at three goods.
"""

from artificial_society.simulation import Simulation


def _two_adjacent_trusted_agents():
    sim = Simulation(
        headless=True,
        seed=7,
        grid_w=20,
        grid_h=15,
        initial_population=8,
        load_checkpoint=False,
    )
    a, b = sim.agents[0], sim.agents[1]
    # Same neighbourhood (within trade radius 2) and mutual trust above the gate.
    a.pos = (5, 5)
    b.pos = (5, 6)
    a._cached_nearby_agents = None
    b._cached_nearby_agents = None
    a.trust[b.id] = 1.0
    b.trust[a.id] = 1.0
    return sim, a, b


def test_trade_swaps_non_wood_stone_fiber_materials():
    sim, a, b = _two_adjacent_trusted_agents()
    # Complementary surpluses of materials OUTSIDE the hardcoded triad.
    a.material_inventory = {"flint": 3.0}
    b.material_inventory = {"bone": 3.0}

    sim.economy.maybe_trade(a, sim.agents)

    # A material beyond wood/stone/fiber changed hands in both directions.
    assert a.material_inventory.get("bone", 0.0) > 0.0, (
        "agent A never received a discovered material"
    )
    assert b.material_inventory.get("flint", 0.0) > 0.0, (
        "agent B never received a discovered material"
    )
    assert sim.economy.trade_count >= 1


def test_legacy_resource_triad_still_tradeable():
    """Opening trade must not remove the original wood/stone/fiber barter."""
    sim, a, b = _two_adjacent_trusted_agents()
    a.material_inventory = {}
    b.material_inventory = {}
    a.resources = {"wood": 3, "stone": 0, "fiber": 0}
    b.resources = {"wood": 0, "stone": 3, "fiber": 0}

    sim.economy.maybe_trade(a, sim.agents)

    assert a.resources.get("stone", 0) > 0, "agent A never received stone"
    assert b.resources.get("wood", 0) > 0, "agent B never received wood"
    assert sim.economy.trade_count >= 1
