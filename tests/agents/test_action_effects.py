"""Audit group 3: action effects must match their bookkeeping.

- _maybe_craft_tool debited the wrong pool (negative inventory, free tool).
- Kill loot was minted instead of transferred from the victim.
- last_action_mode claimed actions that had no effect (phantom stimuli for
  emotional_memory).
- children counters incremented at conception instead of birth.
- progress_pregnancy could push energy below 0.
"""

from __future__ import annotations

import torch

import artificial_society.agents.agent as agent_mod
from artificial_society.agents.agent import MAX_ENERGY, Agent
from artificial_society.agents.brain import HIDDEN_SIZE, INPUT_SIZE
from artificial_society.simulation import Simulation


def _sim(seed: int = 17, pop: int = 4):
    return Simulation(
        headless=True,
        seed=seed,
        grid_w=15,
        grid_h=12,
        initial_population=pop,
        load_checkpoint=False,
    )


# --- fix 5: craft-tool pool deduction ----------------------------------------
def test_craft_tool_debits_the_qualifying_pool():
    a = Agent.spawn_random(2, 2)
    a.tool = None
    a.material_inventory = {"sharp_stone": 0.5}  # fraction: does NOT qualify
    a.resources = {"wood": 0, "stone": 2, "fiber": 0}

    a._maybe_craft_tool()

    assert a.tool == "sharp_stone"
    assert a.material_inventory["sharp_stone"] == 0.5, "fraction must stay untouched"
    assert a.resources["stone"] == 1, "the qualifying stone must be paid"


def test_craft_tool_prefers_inventory_when_it_qualifies():
    a = Agent.spawn_random(2, 2)
    a.tool = None
    a.material_inventory = {"sharp_stone": 2.0}
    a.resources = {"wood": 0, "stone": 1, "fiber": 0}

    a._maybe_craft_tool()

    assert a.tool == "sharp_stone"
    assert a.material_inventory["sharp_stone"] == 1.0
    assert a.resources["stone"] == 1


def test_craft_tool_needs_a_full_unit():
    a = Agent.spawn_random(2, 2)
    a.tool = None
    a.material_inventory = {"sharp_stone": 0.5}
    a.resources = {"wood": 0, "stone": 0, "fiber": 0}

    a._maybe_craft_tool()

    assert a.tool is None
    assert a.material_inventory["sharp_stone"] == 0.5


# --- fixes 7/8: attack loot is a transfer, no attacker self-stress -----------
def test_kill_loot_comes_out_of_the_victim(monkeypatch):
    attacker = Agent.spawn_random(5, 5)
    victim = Agent.spawn_random(5, 6)
    attacker.trust[victim.id] = -1.0
    victim.health = 0.5  # any hit kills
    victim.energy = 50.0
    attacker.energy = 100.0
    monkeypatch.setattr(agent_mod.random, "random", lambda: 0.0)  # always attack
    monkeypatch.setattr(agent_mod.random, "choice", lambda seq: seq[0])

    loot = attacker._attack([attacker, victim], {})

    assert not victim.alive
    assert loot > 0
    assert victim.energy == 50.0 - loot, "loot must be debited from the victim"
    assert attacker.energy == min(MAX_ENERGY, 100.0 + loot)


def test_survived_attack_gives_attacker_no_victim_stress(monkeypatch):
    attacker = Agent.spawn_random(5, 5)
    victim = Agent.spawn_random(5, 6)
    attacker.trust[victim.id] = -1.0
    victim.health = 100.0  # survives
    monkeypatch.setattr(agent_mod.random, "random", lambda: 0.0)
    monkeypatch.setattr(agent_mod.random, "choice", lambda seq: seq[0])
    received = []
    monkeypatch.setattr(
        agent_mod.EndocrineSystem,
        "apply_attack_received",
        lambda self: received.append(self),
    )

    attacker._attack([attacker, victim], {})

    assert victim.alive
    assert victim.endocrine in received, "the victim must receive the stress response"
    assert attacker.endocrine not in received, (
        "attacker must not receive the victim's stress response"
    )


# --- fix 9: mode reflects effect, not attempt --------------------------------
def test_mode_stays_idle_when_cooperate_has_no_effect(monkeypatch):
    sim = _sim()
    a = sim.agents[0]
    for other in sim.agents[1:]:  # nobody in cooperation range
        other.pos = (14, 11)
    a.pos = (2, 2)
    a._need_inv_cooldown = 5  # keep invention noise out of this tick
    monkeypatch.setattr(agent_mod, "agent_tick_with_goals", lambda *args: (None, 0.0))

    def fake_act(self, features, hidden_state, use_planning=True, goal_vector=None,
                 research_mode=False):
        action = torch.zeros(1, 7)
        action[0, 3] = 0.9  # cooperate hard, everything else off
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
    a.update(sim.world, sim.agents, tick=1)

    assert a.last_action_mode == "idle", (
        "cooperate with nobody around had no effect; mode must not claim it"
    )


# --- fixes 20a/20b: children at birth, pregnancy energy clamp ----------------
def test_children_counted_at_birth_not_conception():
    sim = _sim(seed=23)
    mother, father = sim.agents[0], sim.agents[1]
    mother.sex, father.sex = "f", "m"
    for a in (mother, father):
        a.age = 200
        a.energy = 150.0
        a.reproduction_cooldown = 0
        a.pregnant = False
    mother.pos, father.pos = (5, 5), (5, 6)

    mother._try_reproduce(sim.agents)

    assert mother.pregnant, "test setup: conception must have happened"
    assert mother.children == 0 and father.children == 0, (
        "conception must not increment children"
    )

    sim.spawn_child_from_parent(mother, dict(mother.genes))
    assert mother.children == 1
    assert father.children == 1, "father found via _last_mate_id at birth"


def test_pregnancy_energy_never_negative():
    a = Agent.spawn_random(2, 2)
    a.pregnant = True
    a.gestation = 5.0
    a.stored_child_genes = dict(a.genes)
    a.energy = 0.01

    a.progress_pregnancy()

    assert a.energy == 0.0
