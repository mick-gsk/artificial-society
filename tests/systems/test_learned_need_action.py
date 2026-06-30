"""Lane test: the need->action mapping is learned, not a hardcoded cluster.

Phase 5 de-scripting (Task 4). ``_choose_action_by_need`` previously mapped each need to a
fixed action cluster (e.g. hunger -> ["eat","place_on_heat","bundle"]). After de-scripting,
the agent learns from its own reward history which actions pay off under each need, so an
action *outside* the old cluster can come to dominate once it proves rewarding.
"""

from artificial_society.rng import seed_all
from artificial_society.systems.invention import (
    _choose_action_by_need,
    _dominant_need,
    _record_need_action,
)


class _HungryAgent:
    def __init__(self):
        self.energy = 10.0  # energy_ratio ~0.04 -> dominant need is hunger
        self.health = 100.0
        self.disease_id = None


def test_learned_action_outside_hunger_cluster_dominates():
    seed_all(123)
    agent = _HungryAgent()
    # 'strike' is NOT in the old ACTIONS_FOR_HUNGER cluster, but the agent has learned
    # it pays off under hunger.
    agent._need_action_stats = {"hunger": {"strike": {"n": 5, "reward": 10.0}}}
    cell = {"temperature": 20}

    counts = {}
    for _ in range(200):
        a = _choose_action_by_need(agent, None, "stone", "wood", cell)
        counts[a] = counts.get(a, 0) + 1

    # With the cluster hardcoded, 'strike' could only appear via the ~2.5% exploration
    # floor; learned, it dominates.
    assert counts.get("strike", 0) > 100, counts


def test_record_need_action_learns_from_reward():
    agent = _HungryAgent()
    cell = {"temperature": 20}
    need = _dominant_need(agent, cell)
    assert need == "hunger"

    _record_need_action(agent, need, "strike", 4.0)
    _record_need_action(agent, need, "strike", 2.0)
    _record_need_action(agent, need, "eat", 0.1)

    stats = agent._need_action_stats["hunger"]
    assert stats["strike"]["n"] == 2
    assert stats["strike"]["reward"] == 6.0
    # The higher-mean-reward action ranks first.
    seed_all(1)
    picks = {_choose_action_by_need(agent, None, "stone", "wood", cell) for _ in range(50)}
    assert "strike" in picks
