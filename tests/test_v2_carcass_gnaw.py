"""v2-Kalorien-Fix (Option A): innater roher Handbiss aus einem ko-lokierten
Kadaver-Objekt am automatischen `forage`-Skalar.

Stellt die v1-Fleisch-Zugänglichkeit (Tod→Nahrung-Kreislauf) im physics_v2-Pfad
wieder her, ohne den Brain-Aktionsraum zu berühren, massenerhaltend über den
bestehenden `do_eat`-Pfad. Guards G1 (physics_v2-Gate), G2 (nur bei Hunger),
G3 (Energie ausschließlich über do_eat) hier festgenagelt.
"""

from __future__ import annotations

import pytest

from artificial_society.agents.agent import MAX_ENERGY
from artificial_society.environment.physics.actions import BITE_MASS_KG
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _held_mass(sim) -> float:
    total = 0.0
    for a in sim.agents:
        hands = getattr(a, "hands", None)
        if hands is not None:
            total += hands.carried_mass_kg()
    return total


def test_auto_kadaver_biss_ist_massenerhaltend():
    """(1) Nach einem Auto-Kadaver-Biss ist die Objektmasse um genau die Biss-Masse
    gesunken UND ledger['eaten'] entsprechend erhöht; conservation-Invariante hält."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    assert agent.body is not None and agent.hands is not None
    x, y = agent.pos
    sim.world.set_cell(x, y, "plant_food", 0.0)  # isoliert die Fleisch-Linie
    carcass = make_object("carcass", 70.0)
    sim.world.objects.add(carcass, (x, y), source="from_carcass")

    agent.energy = 100.0  # hungrig (< MAX) → Auto-Biss feuert (G2)
    masse_vorher = carcass.mass
    eaten_vorher = sim.world.objects.ledger["eaten"]

    agent._gather(sim.world, {})

    assert masse_vorher - carcass.mass == pytest.approx(BITE_MASS_KG)
    assert sim.world.objects.ledger["eaten"] - eaten_vorher == pytest.approx(BITE_MASS_KG)
    lhs, rhs = sim.world.objects.conservation_terms(_held_mass(sim))
    assert lhs == pytest.approx(rhs)


def test_hungriger_v2_agent_gewinnt_energie_aus_fleisch():
    """(2) Ein hungriger v2-Agent auf einer Zelle mit Kadaver-Objekt gewinnt Energie
    aus dem Fleisch — über die (hier abgeschaltete) Pflanzen-Einnahme hinaus."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    x, y = agent.pos
    sim.world.set_cell(x, y, "plant_food", 0.0)  # kein Pflanzenbiss → Gewinn nur aus Fleisch
    sim.world.objects.add(make_object("carcass", 70.0), (x, y), source="from_carcass")

    agent.energy = 100.0
    meat_vorher = agent.meat_eaten
    gain = agent._gather(sim.world, {})

    assert agent.energy > 100.0
    assert agent.meat_eaten == meat_vorher + 1
    assert gain > 0.0


def test_g1_v1_agent_macht_keinen_kadaver_objekt_biss():
    """(3/G1) Ein v1-Agent (physics_v2=False) mit diet<0 betritt denselben
    forage-Zweig, darf aber KEINEN Kadaver-Objekt-Biss machen (v1-Pfad unberührt,
    Golden-Schutz)."""
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    agent = sim.agents[0]
    agent.traits["diet_preference"] = -1.0  # Herbivore → betritt `if diet < 0 or ...`
    assert not agent.physics_v2
    x, y = agent.pos
    carcass = make_object("carcass", 70.0)
    sim.world.objects.add(carcass, (x, y), source="from_carcass")
    eaten_vorher = sim.world.objects.ledger["eaten"]

    agent._gather(sim.world, {})

    assert carcass.mass == pytest.approx(70.0)  # unangetastet
    assert sim.world.objects.ledger["eaten"] == pytest.approx(eaten_vorher)


def test_g2_satter_agent_macht_keinen_biss():
    """(4/G2) Ein satter Agent (energy≈MAX) mahlt das knappe, nicht-nachwachsende
    Kadaver-Objekt NICHT sinnlos in den eaten-Ledger."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    x, y = agent.pos
    carcass = make_object("carcass", 70.0)
    sim.world.objects.add(carcass, (x, y), source="from_carcass")
    eaten_vorher = sim.world.objects.ledger["eaten"]

    agent.energy = MAX_ENERGY
    agent._gather(sim.world, {})

    assert carcass.mass == pytest.approx(70.0)  # kein Biss
    assert sim.world.objects.ledger["eaten"] == pytest.approx(eaten_vorher)


def test_determinismus_gleicher_seed_gleiches_ergebnis():
    """(5) Gleicher Seed → identische Energie-Trajektorie über den Auto-Biss-Pfad."""

    def _run():
        sim = Simulation(seed=123, physics_v2=True, **_PARAMS)
        agent = sim.agents[0]
        x, y = agent.pos
        sim.world.objects.add(make_object("carcass", 70.0), (x, y), source="from_carcass")
        agent.energy = 100.0
        energies = []
        for _ in range(6):
            agent._gather(sim.world, {})
            energies.append(agent.energy)
        return energies

    assert _run() == _run()
