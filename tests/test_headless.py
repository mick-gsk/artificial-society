"""Phase 0: the simulation must be constructible headless and reproducible.

These lock the two foundations the refactor depends on:
  * construct/operate without opening a pygame window, and
  * identical initial world+population for a given seed.
"""

from artificial_society.simulation import Simulation


def _initial_state_digest(sim):
    """A cheap, order-stable fingerprint of the freshly constructed world+agents."""
    biome_sample = tuple(
        sim.world.biomes[y][x]
        for y in range(0, sim.world.height, 3)
        for x in range(0, sim.world.width, 3)
    )
    agent_sample = tuple(
        (a.pos, a.sex, round(a.genes["speed"], 6), round(a.genes["curiosity"], 6))
        for a in sim.agents
    )
    return biome_sample, agent_sample


def test_headless_construction_opens_no_display():
    sim = Simulation(
        headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8
    )
    assert sim.tick == 0
    assert len(sim.agents) == 8
    assert sim.world.width == 20 and sim.world.height == 15
    assert sim.screen is None
    assert sim.renderer is None


def test_same_seed_yields_identical_initial_state():
    a = Simulation(
        headless=True, load_checkpoint=False, seed=42, grid_w=24, grid_h=16, initial_population=12
    )
    b = Simulation(
        headless=True, load_checkpoint=False, seed=42, grid_w=24, grid_h=16, initial_population=12
    )
    assert _initial_state_digest(a) == _initial_state_digest(b)


def test_different_seed_yields_different_initial_state():
    a = Simulation(
        headless=True, load_checkpoint=False, seed=1, grid_w=24, grid_h=16, initial_population=12
    )
    b = Simulation(
        headless=True, load_checkpoint=False, seed=2, grid_w=24, grid_h=16, initial_population=12
    )
    assert _initial_state_digest(a) != _initial_state_digest(b)
