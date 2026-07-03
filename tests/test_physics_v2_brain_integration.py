"""3b-Integration: v2-Agent nimmt Slots wahr, act_v2 läuft im Sim-Tick, Verben mappen auf do_*."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.agent import Agent, attach_body
from artificial_society.agents.perception_v2 import build_slots
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _sim_v2(seed=42):
    return Simulation(seed=seed, physics_v2=True, **_PARAMS)


def _erzwinge_verb(agent, verb_idx, target_slot, tool_slot=-1):
    """Monkeypatch-freier Zwang: wir rufen _execute_embodied direkt mit einem
    handgebauten brain_step — die Policy selbst bleibt Sampling (Prinzip 1);
    getestet wird hier NUR das Mapping Verb→do_* (Spec C2)."""
    view = build_slots(agent, agent._test_world.objects)
    action = torch.zeros(1, 29)
    step = {
        "verb": ("grasp", "release", "strike", "cut", "eat")[verb_idx],
        "effort": 0.9,
        "target_idx": target_slot,
        "tool_idx": tool_slot,
        "action_tensor": action,
        "slot_embeds": torch.zeros(10, 32),
    }
    return agent._execute_embodied(agent._test_world, step, view), view


def test_attach_body_ruestet_v2_brain_und_causal_model():
    torch.manual_seed(0)
    agent = Agent.spawn_random(3, 3)
    attach_body(agent)
    assert agent.brain.physics_v2 is True
    assert agent.brain.action_size == 29
    assert agent.causal_model is not None
    assert agent._novelty_buckets is not None
    assert agent._causal_pending is None


def test_v2_kind_erbt_gewichte_vom_v2_eltern_brain():
    sim = _sim_v2()
    eltern = sim.agents[0]
    kind = sim.spawn_child_from_parent(eltern, dict(eltern.genes))
    assert kind.brain.physics_v2 is True
    assert kind.brain.gru.input_size == 192
    assert "strength" in kind.genes


def test_verb_mapping_grasp_und_release():
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    stein = make_object("granite", 1.0)
    sim.world.objects.add(stein, agent.pos, source="spawned")
    view = build_slots(agent, sim.world.objects)
    slot = view.objs.index(stein)

    result, _ = _erzwinge_verb(agent, 0, slot)  # grasp
    assert result.ok and stein in agent.hands.held
    assert sim.world.objects.metrics["verbs_fired"]["grasp"] == 1

    view2 = build_slots(agent, sim.world.objects)
    slot2 = view2.objs.index(stein)  # jetzt Hand-Slot 8
    assert slot2 == 8
    step = {
        "verb": "release",
        "effort": 0.5,
        "target_idx": slot2,
        "tool_idx": -1,
        "action_tensor": torch.zeros(1, 29),
        "slot_embeds": torch.zeros(10, 32),
    }
    result2 = agent._execute_embodied(sim.world, step, view2)
    assert result2.ok and agent.hands.held == []
    assert sim.world.objects.position_of(stein) == agent.pos


def test_verb_mapping_eat_wendet_deltas_an():
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    fleisch = make_object("raw_meat", 0.5)
    fleisch.props[10] = 0.5  # toxicity hoch → Health-Delta sichtbar
    sim.world.objects.add(fleisch, agent.pos, source="spawned")
    agent.energy, agent.health = 50.0, 90.0
    view = build_slots(agent, sim.world.objects)
    slot = view.objs.index(fleisch)
    result, _ = _erzwinge_verb(agent, 4, slot)  # eat
    assert result.ok
    assert agent.energy > 50.0  # energy_delta_sim angewendet
    assert agent.health < 90.0  # health_delta (Toxin) angewendet


def test_verb_mapping_strike_setzt_causal_target_auf_fragment():
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    agent.body.strength = 1.0
    hammer = make_object("granite", 1.5)
    flint = make_object("flint", 0.8)
    agent.hands.held.append(hammer)
    sim.world.objects.ledger["spawned"] += 1.5
    sim.world.objects.add(flint, agent.pos, source="spawned")
    view = build_slots(agent, sim.world.objects)
    ziel, werkzeug = view.objs.index(flint), view.objs.index(hammer)
    result, _ = _erzwinge_verb(agent, 2, ziel, tool_slot=werkzeug)  # strike
    assert result.ok and result.fragments
    schwerstes = max(result.fragments, key=lambda f: f.mass)
    assert agent._causal_next_target is schwerstes  # C4: massereichstes Fragment


def test_v2_sim_laueft_und_planner_bleibt_stumm():
    """8 Agenten × 6 Ticks über sim.step(): act_v2 im Einsatz, kein plan_action."""
    aufrufe = []
    from artificial_society.agents.brain import Brain

    original = Brain.plan_action

    def spion(self, *a, **kw):
        aufrufe.append(1)
        return original(self, *a, **kw)

    Brain.plan_action = spion
    try:
        sim = _sim_v2()
        for _ in range(6):
            sim.step()
    finally:
        Brain.plan_action = original
    assert aufrufe == [], "C5: Planner ist im v2-Modus deaktiviert"
    assert all(a.brain.physics_v2 for a in sim.agents)
    assert sum(len(a.brain.rollout) for a in sim.agents) > 0, "Transitionen müssen fließen"
    assert not any(np.isnan(a.last_reward) for a in sim.agents)


def test_d4_metriken_verbs_und_discovery_je_agent():
    """D4 (erst nach 3b): Verb-Raten + DiscoveryV2-Events je Agent im Snapshot."""
    sim = _sim_v2()
    agent = sim.agents[0]
    agent._test_world = sim.world
    agent.body.strength = 1.0
    hammer = make_object("granite", 1.5)
    flint = make_object("flint", 0.8)
    agent.hands.held.append(hammer)
    sim.world.objects.ledger["spawned"] += 1.5
    sim.world.objects.add(flint, agent.pos, source="spawned")
    view = build_slots(agent, sim.world.objects)
    _erzwinge_verb(agent, 2, view.objs.index(flint), tool_slot=view.objs.index(hammer))

    snap = sim.world.objects.metrics_snapshot()
    assert snap["verbs_fired"]["strike"] == 1  # Versuche (F4)
    assert snap.get("verbs_failed", {}).get("strike", 0) == 0  # erfolgreicher Schlag
    assert snap["discovery_events_by_agent"].get(agent.id, 0) >= 1, (
        "frische Fragmente sind neue Eigenschafts-Punkte → DiscoveryV2-Event je Agent"
    )


def test_d4_neugier_zerlegung_wird_geloggt():
    """D4: die drei Neugier-Quellen werden separat geloggt (Pilot-Diagnostik)."""
    sim = _sim_v2()
    for _ in range(3):
        sim.step()
    snap = sim.world.objects.metrics_snapshot()
    assert set(snap["curiosity_sums"]) == {"nextslot", "causal", "novelty"}
    assert all(np.isfinite(v) for v in snap["curiosity_sums"].values())
    assert snap["curiosity_sums"]["nextslot"] > 0.0  # Vorhersagefehler früh > 0
    for agent in sim.agents:
        assert set(agent.curiosity_last) == {"nextslot", "causal", "novelty"}


def test_v2_smoke_12_ticks_keine_nans():
    """Kurzer Suite-Smoke: 12 Ticks v2 — keine Exceptions, keine NaNs, Buffer wachsen."""
    sim = _sim_v2(seed=7)
    for _ in range(12):
        sim.step()
    for agent in sim.agents:
        assert np.isfinite(agent.last_reward)
        assert np.isfinite(agent.energy) and np.isfinite(agent.health)
        assert not torch.isnan(agent.hidden_state).any()
    assert sum(len(a.brain.rollout) for a in sim.agents) > 0
    haende = sum(
        a.hands.carried_mass_kg() for a in sim.agents if getattr(a, "hands", None) is not None
    )
    lhs, rhs = sim.world.objects.conservation_terms(held_mass_kg=haende)
    assert abs(lhs - rhs) < 1e-6, "Massen-Ledger hält auch unter Policy-Aktionen"
