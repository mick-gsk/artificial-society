"""Plan 3a: physics_v2-Flag, Body/Hands-Embodiment, Überlast-Drop im Tick (Spec A, B4)."""

from __future__ import annotations

from artificial_society.agents.agent import Agent, ensure_fields
from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG, Body, Hands
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def test_v2_sim_embodied_alle_agenten():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    assert sim.physics_v2 is True
    for a in sim.agents:
        assert a.physics_v2 is True
        assert isinstance(a.body, Body)
        assert a.body.body_mass == BODY_MASS_DEFAULT_KG and a.body.strength == 0.5
        assert isinstance(a.hands, Hands) and a.hands.held == []


def test_v1_sim_bleibt_unveraendert():
    sim = Simulation(seed=42, **_PARAMS)
    assert sim.physics_v2 is False
    for a in sim.agents:
        assert a.physics_v2 is False and a.body is None and a.hands is None


def test_ensure_fields_reinstauriert_embodiment():
    """Checkpoint-Agent (Flag gesetzt, Body fehlt) wird vollständig reinstauriert."""
    agent = Agent.spawn_random(1, 1)
    agent.physics_v2 = True
    agent.body = None
    agent.hands = None
    ensure_fields(agent)
    assert isinstance(agent.body, Body) and isinstance(agent.hands, Hands)


def test_ueberlast_drop_laeuft_im_sim_tick():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    schwer = make_object("granite", 6.0)
    leicht = make_object("granite", 5.0)
    agent.hands.held.extend([schwer, leicht])
    agent.body.fatigue = 1.0  # Kapazität 8.4 kg < 11 kg gehalten
    sim.step()
    assert schwer not in agent.hands.held, "schwerstes Objekt muss zwangsabgelegt werden"
    assert sim.world.objects.position_of(schwer) is not None


def test_respawns_werden_embodied():
    """Deckt den emergency_respawn-Pfad; spawn_child_from_parent nutzt dasselbe
    attach_body-Muster (Code-identisch, im Sozial-RNG schwer deterministisch erzwingbar)."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    sim.agents = sim.agents[:1]  # unter MIN_POPULATION → Respawn im nächsten Tick
    sim.step()
    assert len(sim.agents) > 1, "Respawn muss stattgefunden haben"
    for a in sim.agents:
        assert a.physics_v2 is True and isinstance(a.body, Body)
