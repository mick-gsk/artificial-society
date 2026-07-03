"""Audit group 2: sleeping agents must not act.

Before the fix, the ``is_sleeping`` guard covered only the primary action
block (forage/cooperate/attack/build); sleeping agents kept reproducing,
social-learning, inventing, cooking and trading. Gestation and perception
intentionally continue during sleep.
"""

from __future__ import annotations

import artificial_society.agents.agent as agent_mod
from artificial_society.simulation import Simulation

# tick must hit every cadence gate: % 3 (social learning, invention), % 4 (cook)
TICK = 12


class _CountingEconomy:
    def __init__(self):
        self.calls = 0

    def maybe_trade(self, agent, agents):
        self.calls += 1


def _sim_agent():
    sim = Simulation(
        headless=True,
        seed=13,
        grid_w=15,
        grid_h=12,
        initial_population=6,
        load_checkpoint=False,
    )
    a = sim.agents[0]
    a.age = 200  # adult: stage gates (can_reproduce/can_attack/can_build) open
    a._need_inv_cooldown = 0
    return sim, a


def _patch_counters(monkeypatch, calls):
    def count(name, ret=0.0):
        def _f(*args, **kwargs):
            calls[name] += 1
            return ret

        return _f

    monkeypatch.setattr(agent_mod, "social_learning_step", count("social"))
    monkeypatch.setattr(agent_mod, "agent_invent_from_need", count("need_inv"))
    monkeypatch.setattr(agent_mod, "agent_try_invention", count("invent"))
    monkeypatch.setattr(agent_mod, "agent_try_cook", count("cook"))
    monkeypatch.setattr(
        agent_mod.Agent, "_try_reproduce", lambda self, world, agents: calls.__setitem__(
            "reproduce", calls["reproduce"] + 1
        )
    )


def test_sleeping_agent_takes_no_actions(monkeypatch):
    sim, a = _sim_agent()
    calls = dict.fromkeys(("social", "need_inv", "invent", "cook", "reproduce"), 0)
    _patch_counters(monkeypatch, calls)
    economy = _CountingEconomy()

    a.is_sleeping = True
    # _sleep_tick would wake the agent (sleep_drive of a fresh agent is ~0);
    # pin it asleep — we test the gates, not the sleep controller.
    monkeypatch.setattr(a, "_sleep_tick", lambda mods: None, raising=False)

    a.update(sim.world, sim.agents, tick=TICK, economy=economy)

    assert calls == dict.fromkeys(calls, 0), f"sleeping agent acted: {calls}"
    assert economy.calls == 0


def test_awake_agent_acts(monkeypatch):
    sim, a = _sim_agent()
    calls = dict.fromkeys(("social", "need_inv", "invent", "cook", "reproduce"), 0)
    _patch_counters(monkeypatch, calls)
    # All dice pass, so the probabilistic gates (invention, cook) fire too.
    monkeypatch.setattr(agent_mod.random, "random", lambda: 0.0)
    monkeypatch.setattr(a, "_sleep_tick", lambda mods: None, raising=False)
    economy = _CountingEconomy()

    a.is_sleeping = False
    a.update(sim.world, sim.agents, tick=TICK, economy=economy)

    assert all(n >= 1 for n in calls.values()), f"awake agent skipped actions: {calls}"
    assert economy.calls == 1


def test_gestation_continues_during_sleep(monkeypatch):
    sim, a = _sim_agent()
    a.is_sleeping = True
    monkeypatch.setattr(a, "_sleep_tick", lambda mods: None, raising=False)
    a.pregnant = True
    a.gestation = 5.0
    a.stored_child_genes = dict(a.genes)

    a.update(sim.world, sim.agents, tick=TICK)

    assert a.gestation < 5.0, "gestation must keep progressing during sleep"
