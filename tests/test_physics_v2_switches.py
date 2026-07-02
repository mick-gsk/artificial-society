"""B6-Mechanik-Spalte im v2-Modus: Fleisch-Zellpools, v1-Erfindung, Kochen, Goal-Stack,
Hamilton — AUS. Reward-Spalte bleibt unangetastet (Plan 3b)."""

from __future__ import annotations

import pytest

import artificial_society.agents.agent as agent_mod
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def test_v2_forage_laesst_fleisch_zellpools_unangetastet():
    """B6: Zell-Fleisch/Aas-Pools AUS — ein einziger Pfad für Fleischkalorien (Kadaver-Objekte)."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    agent = sim.agents[0]
    agent.genes["diet_preference"] = 1.0  # Karnivor — würde in v1 zuerst Aas/Fleisch nehmen
    x, y = agent.pos
    sim.world.set_cell(x, y, "carcasses", 50.0)
    sim.world.set_cell(x, y, "meat_food", 50.0)
    sim.world.set_cell(x, y, "plant_food", 20.0)

    agent._forage(sim.world, {})

    cell = sim.world.get_cell(x, y)
    assert cell["carcasses"] == pytest.approx(50.0)
    assert cell["meat_food"] == pytest.approx(50.0)
    assert cell["plant_food"] < 20.0  # Pflanzen-Zell-Foraging bleibt AN


def test_v1_forage_unveraendert():
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    agent = sim.agents[0]
    agent.genes["diet_preference"] = 1.0
    x, y = agent.pos
    sim.world.set_cell(x, y, "carcasses", 50.0)
    agent._forage(sim.world, {})
    assert sim.world.get_cell(x, y)["carcasses"] < 50.0


def _spy(monkeypatch, name, rueckgabe=None):
    aufrufe = []

    def _zaehler(*args, **kwargs):
        aufrufe.append(args)
        return rueckgabe

    monkeypatch.setattr(agent_mod, name, _zaehler)
    return aufrufe


def test_v2_erfindung_kochen_goalstack_tot(monkeypatch):
    """B6: beide Invention-Trigger-Pfade, Kochen und der Goal-Stack-Planner feuern nie."""
    inv_need = _spy(monkeypatch, "agent_invent_from_need")
    inv_rand = _spy(monkeypatch, "agent_try_invention")
    kochen = _spy(monkeypatch, "agent_try_cook")
    # (None, 0.0) statt None: im Failing-Lauf (Schalter existiert noch nicht) wird
    # der Spy aufgerufen und sein Rückgabewert in agent.py:1127–1128 entpackt
    # (`goal_action, goal_shaping = ...`) — so schlägt unten sauber die
    # Spy-Assertion fehl statt eines Unpack-TypeError.
    goals = _spy(monkeypatch, "agent_tick_with_goals", rueckgabe=(None, 0.0))

    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    for _ in range(13):  # deckt tick%3- und tick%4-Fenster mehrfach ab
        sim.step()

    assert inv_need == [] and inv_rand == [] and kochen == [] and goals == []


def test_v2_hamilton_umverteilung_aus():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    a, b = sim.agents[0], sim.agents[1]
    a.tribe_id = b.tribe_id = 1
    a.energy, b.energy = 200.0, 10.0
    sim.tick = 20  # HAMILTON_TICK_INTERVAL
    sim._apply_hamilton_rewards()
    assert a.energy == 200.0 and b.energy == 10.0
