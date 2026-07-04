"""K1 (M1-Batterie): the per-tick neighbor snapshot must actually be live.

Before the fix, ``ensure_fields`` initialised ``_cached_nearby_agents`` to ``[]``
and nothing ever invalidated it, so ``_nearby_cached`` returned that
permanently-valid empty cache for the agent's whole life — silently disabling
social_learning_step, economy.maybe_trade and the ToM observation loop (all fed
exclusively by this cache). The M1 diagnostic battery proved this: the nosocial
arm was byte-identical to its reference.

These tests lock the fix on both simulation paths (v1 and physics_v2):
  (a) the snapshot returns real neighbors when agents stand adjacent;
  (b) the snapshot is invalidated per tick (tick n+1 sees the changed
      neighborhood);
  (c) in the v2 path social_learning_step reaches neighbors within a few ticks.
"""

import artificial_society.agents.agent as agent_mod
from artificial_society.agents.agent import Agent, ensure_fields
from artificial_society.simulation import Simulation


def _sim(seed=11, pop=6, physics_v2=False):
    return Simulation(
        headless=True,
        seed=seed,
        grid_w=20,
        grid_h=15,
        initial_population=pop,
        load_checkpoint=False,
        physics_v2=physics_v2,
    )


def test_ensure_fields_initialises_no_snapshot():
    agent = Agent.spawn_random(1, 1)
    ensure_fields(agent)
    assert agent._cached_nearby_agents is None


def test_snapshot_has_real_neighbors_and_refreshes():
    """(a) adjacent neighbor is in the tick's snapshot; (b) it refreshes."""
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
    assert b not in (a._cached_nearby_agents or []), (
        "stale neighbor survived into the next tick — snapshot not invalidated"
    )


def test_v2_social_learning_reaches_neighbors():
    """(c) in the physics_v2 path, social_learning_step must reach neighbors.

    agent.py binds social_learning_step via ``from ... import`` to its own module
    namespace, so we spy on ``agent_mod.social_learning_step`` (the exact symbol
    the update loop calls). The spy records ticks where the agent's per-tick
    snapshot is non-empty, then delegates to the real function unchanged.
    """
    sim = _sim(seed=42, pop=10, physics_v2=True)
    hits = {"with_neighbors": 0}
    real = agent_mod.social_learning_step

    def spy(agent, agents, tick):
        if agent._nearby_cached(agents, 2):
            hits["with_neighbors"] += 1
        return real(agent, agents, tick)

    agent_mod.social_learning_step = spy
    try:
        for _ in range(60):
            sim.step()
    finally:
        agent_mod.social_learning_step = real

    assert hits["with_neighbors"] > 0, (
        "social_learning_step never saw a neighbor in the v2 path — "
        "snapshot pipeline dead"
    )


def test_v2_tom_observes_neighbors():
    """Cross-check via ToM: at least one agent must build a ToM model of a
    neighbor within a short v2 run (the ToM loop iterates the same snapshot)."""
    sim = _sim(seed=42, pop=10, physics_v2=True)
    for _ in range(120):
        sim.step()
    assert any(a.tom.models for a in sim.agents), (
        "no agent ever observed a neighbor — snapshot pipeline dead"
    )
