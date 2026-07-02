"""Audit group 5: the net's research_drive (action dim 6) must be causal.

Before the fix, brain_step["research_drive"] was never read anywhere:
research_mode came from a timer, and invention attempts were pure dice —
"the net decides when to research" existed only as a comment.
"""

from __future__ import annotations

import torch

import artificial_society.agents.agent as agent_mod
from artificial_society.agents.agent import Agent, ensure_fields
from artificial_society.agents.brain import HIDDEN_SIZE, INPUT_SIZE
from artificial_society.simulation import Simulation


def _sim_agent(seed: int = 37):
    sim = Simulation(
        headless=True,
        seed=seed,
        grid_w=15,
        grid_h=12,
        initial_population=4,
        load_checkpoint=False,
    )
    a = sim.agents[0]
    a.age = 200
    a._need_inv_cooldown = 0
    return sim, a


def test_drive_gates_invention_attempts(monkeypatch):
    sim, a = _sim_agent()
    calls = {"need_inv": 0, "invent": 0}
    monkeypatch.setattr(
        agent_mod,
        "agent_invent_from_need",
        lambda *x, **k: calls.__setitem__("need_inv", calls["need_inv"] + 1) or 0.0,
    )
    monkeypatch.setattr(
        agent_mod,
        "agent_try_invention",
        lambda *x, **k: calls.__setitem__("invent", calls["invent"] + 1) or 0.0,
    )
    monkeypatch.setattr(agent_mod, "agent_tick_with_goals", lambda *x: (None, 0.0))
    monkeypatch.setattr(agent_mod.random, "random", lambda: 0.0)  # dice always pass
    monkeypatch.setattr(a, "_sleep_tick", lambda mods: None, raising=False)

    a._research_drive = 0.0  # the net said: no research
    a.update(sim.world, sim.agents, tick=12)
    assert calls == {"need_inv": 0, "invent": 0}, (
        "drive below threshold must block invention attempts"
    )

    a._research_drive = 1.0  # the net said: research
    a._need_inv_cooldown = 0
    a.update(sim.world, sim.agents, tick=15)
    assert calls["need_inv"] == 1 and calls["invent"] == 1, (
        "drive above threshold must enable invention attempts"
    )


def test_drive_is_refreshed_from_the_net(monkeypatch):
    sim, a = _sim_agent()
    monkeypatch.setattr(agent_mod, "agent_tick_with_goals", lambda *x: (None, 0.0))

    def fake_act(self, features, hidden_state, use_planning=True, goal_vector=None,
                 research_mode=False):
        action = torch.zeros(1, 7)
        action[0, 6] = -0.5  # raw tanh research_drive
        return {
            "obs_tensor": torch.zeros(1, INPUT_SIZE),
            "hidden_in": torch.zeros(1, HIDDEN_SIZE),
            "value": torch.zeros(1),
            "next_hidden": torch.zeros(HIDDEN_SIZE),
            "action_tensor": action,
            "action_list": action.squeeze(0).tolist(),
            "log_prob": torch.zeros(1),
            "entropy": torch.zeros(1),
            "research_drive": -0.5,
        }

    monkeypatch.setattr(type(a.brain), "act", fake_act)
    a.update(sim.world, sim.agents, tick=1)
    assert abs(a._research_drive - 0.25) < 1e-9, "tanh [-1,1] must map to [0,1]"


def test_ensure_fields_backfills_research_drive():
    a = Agent.spawn_random(1, 1)
    delattr(a, "_research_drive")
    ensure_fields(a)
    assert a._research_drive == 0.5  # tanh(0) mapped: research on by default


# --- fix 3: macro recording is wired ------------------------------------------
def test_macro_recording_and_bonus():
    a = Agent.spawn_random(1, 1)
    assert a._record_macro_if_successful("forage", 1.0) == 0.0  # window too short
    bonus = 0.0
    for _ in range(6):
        bonus = a._record_macro_if_successful("forage", 1.0)
    assert a.knowledge.macro_actions, "successful sequences must be recorded"
    assert bonus > 0.0, "reproducing a confirmed macro must yield a reward bonus"


def test_macros_recorded_through_update(monkeypatch):
    """Integration: two high-reward forage ticks leave a macro in the KG."""
    sim, a = _sim_agent(seed=41)
    a.genes["diet_preference"] = -1.0  # herbivore: forage takes the plant path
    a._need_inv_cooldown = 5
    monkeypatch.setattr(agent_mod, "agent_tick_with_goals", lambda *x: (None, 0.0))
    monkeypatch.setattr(a, "_sleep_tick", lambda mods: None, raising=False)

    def fake_act(self, features, hidden_state, use_planning=True, goal_vector=None,
                 research_mode=False):
        action = torch.zeros(1, 7)
        action[0, 2] = 1.0  # forage hard, no movement
        return {
            "obs_tensor": torch.zeros(1, INPUT_SIZE),
            "hidden_in": torch.zeros(1, HIDDEN_SIZE),
            "value": torch.zeros(1),
            "next_hidden": torch.zeros(HIDDEN_SIZE),
            "action_tensor": action,
            "action_list": action.squeeze(0).tolist(),
            "log_prob": torch.zeros(1),
            "entropy": torch.zeros(1),
            "research_drive": 0.0,
        }

    monkeypatch.setattr(type(a.brain), "act", fake_act)
    for tick in (1, 2):
        sim.world.set_cell(*a.pos, "plant_food", 150.0)  # rich cell -> reward > 0.5
        a.energy = 50.0  # room to eat (forage gain is capped by MAX_ENERGY)
        a.update(sim.world, sim.agents, tick=tick)
        assert a.last_action_mode == "forage"

    assert a.knowledge.macro_actions, (
        "high-reward action sequences must be recorded as macros during update"
    )
