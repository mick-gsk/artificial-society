"""Audit group 2: the per-tick neighbor snapshot must actually refresh.

Before the fix, ensure_fields initialised ``_cached_nearby_agents`` to ``[]``
and nothing ever invalidated it, so ``_nearby_cached`` returned that
permanently-valid empty cache for the agent's whole life — silently disabling
social learning, trade and ToM observation (their only data source).
"""

from __future__ import annotations

from artificial_society.agents.agent import Agent
from artificial_society.simulation import Simulation


def _sim(seed: int = 11, pop: int = 6):
    return Simulation(
        headless=True,
        seed=seed,
        grid_w=20,
        grid_h=15,
        initial_population=pop,
        load_checkpoint=False,
    )


def test_ensure_fields_initialises_no_snapshot():
    agent = Agent.spawn_random(1, 1)
    assert agent._cached_nearby_agents is None


def test_snapshot_refreshes_across_ticks():
    sim = _sim()
    a, b = sim.agents[0], sim.agents[1]
    a.pos, b.pos = (5, 5), (5, 6)

    a.update(sim.world, sim.agents, tick=1)
    assert b in (a._cached_nearby_agents or []), (
        "adjacent neighbor missing from the tick's snapshot"
    )

    # b leaves; the next tick's snapshot must drop it (the old bug kept the
    # first — empty — snapshot forever).
    b.pos = (18, 13)
    a.update(sim.world, sim.agents, tick=2)
    assert b not in (a._cached_nearby_agents or [])


def test_snapshot_is_shared_within_a_tick():
    sim = _sim()
    a, b = sim.agents[0], sim.agents[1]
    a.pos, b.pos = (5, 5), (5, 6)
    a._cached_nearby_agents = None

    snap1 = a._nearby_cached(sim.agents, 2)
    assert b in snap1
    b.pos = (18, 13)  # moves mid-tick: snapshot intentionally stays
    assert a._nearby_cached(sim.agents, 2) is snap1


def test_tom_observes_neighbors_smoke():
    """120-tick smoke: ToM models fill up — structurally impossible pre-fix,
    because the ToM loop iterated a permanently empty neighbor list."""
    sim = _sim(seed=42, pop=10)
    for _ in range(120):
        sim.step()
    assert any(a.tom.models for a in sim.agents), (
        "no agent ever observed a neighbor — snapshot pipeline dead"
    )
