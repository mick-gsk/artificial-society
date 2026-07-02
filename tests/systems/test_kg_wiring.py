"""Audit group 5 (fix 21): first-hand experience must reach the KnowledgeGraph.

Before the fix, kg.record was only called by the campfire knowledge pooling —
which pools from other agents' (empty) graphs. Inventions wrote exclusively
into agent.causal_memory, so inheritance/imitation/pooling always copied
empty fact stores.
"""

from __future__ import annotations

import artificial_society.systems.invention as inv_mod
from artificial_society.simulation import Simulation
from artificial_society.systems.invention import agent_try_invention


def _sim_agent():
    sim = Simulation(
        headless=True,
        seed=31,
        grid_w=15,
        grid_h=12,
        initial_population=4,
        load_checkpoint=False,
    )
    a = sim.agents[0]
    a.knowledge.facts.clear()
    # Guarantee inputs for the invention regardless of the cell's biome.
    cell = sim.world.get_cell(*a.pos)
    mats = cell.setdefault("materials", {})
    mats["dry_wood"] = 2.0
    mats["flint"] = 2.0
    return sim, a


def test_invention_records_first_hand_facts():
    sim, a = _sim_agent()

    agent_try_invention(a, sim.world, *a.pos)

    assert a.knowledge.facts, "invention must record a first-hand causal fact"
    key, fact = next(iter(a.knowledge.facts.items()))
    assert len(key) == 3, "fact key must be (action, mat_a, mat_b)"
    assert fact.tries == 1


def test_failed_invention_records_negative_evidence(monkeypatch):
    sim, a = _sim_agent()
    # Force a full failure: no legacy outcomes, no emergent material.
    monkeypatch.setattr(inv_mod, "apply_interaction", lambda *args, **kw: [])
    monkeypatch.setattr(inv_mod, "combine_vectors", lambda *args, **kw: None)

    agent_try_invention(a, sim.world, *a.pos)

    assert a.knowledge.facts, "failures must be recorded too (negative evidence)"
    fact = next(iter(a.knowledge.facts.values()))
    assert fact.successes == 0
    assert fact.confidence < 0, "a failed try must push confidence negative"


def test_facts_now_flow_through_inheritance():
    sim, a = _sim_agent()
    agent_try_invention(a, sim.world, *a.pos)
    # Boost confidence above the inheritance threshold.
    fact = next(iter(a.knowledge.facts.values()))
    for _ in range(4):
        fact.update(fact.outcome_ids, success=True)

    child_kg = type(a.knowledge)()
    child_kg.inherit_from(a.knowledge)

    assert child_kg.facts, "inheritance must transmit first-hand facts"
