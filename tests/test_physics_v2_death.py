"""D1: Kadaver-Einmal-Münzung über sim.step(), Verzweigungstest Flag aus, Kill ohne Doppel-Energie."""

from __future__ import annotations

import math

import pytest

from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _carcasses_at(sim, pos):
    return [o for o in sim.world.objects.objects_at(pos) if o.kind == "carcass"]


def test_v2_tod_im_step_muenzt_genau_einen_kadaver():
    """Der Test läuft über sim.step() (Agent stirbt im eigenen Update), NICHT über
    direkten remove_dead()-Aufruf — sonst testet er den per Pre-Filtering
    umgangenen Pfad grün (Spec B3/D1)."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    opfer = sim.agents[0]
    gehalten = [make_object("granite", 2.0), make_object("flint", 1.0)]
    opfer.hands.held.extend(gehalten)
    sim.world.objects.ledger["spawned"] += 3.0  # Handbestückung bilanzieren (Testaufbau)
    opfer.health = 0.01
    opfer.energy = 0.0  # verhungert im nächsten Update
    pos = opfer.pos
    pool_vorher = sim.world.get_cell(*pos)["carcasses"]

    sim.step()

    assert opfer not in sim.agents
    kadaver = _carcasses_at(sim, pos)
    assert len(kadaver) == 1, "genau EIN Kadaver-Objekt"
    # rel=1e-3: das registrierte System verwest den Kadaver im SELBEN Tick um
    # genau einen DECAY_RATE-Schritt (≈ 2.9e-4 relativ) — das ist korrekt.
    assert kadaver[0].mass == pytest.approx(BODY_MASS_DEFAULT_KG, rel=1e-3)
    assert sim.world.get_cell(*pos)["carcasses"] <= pool_vorher + 1e-9, (
        "kein v1-Zell-Credit im v2-Modus (Pool darf höchstens von selbst zerfallen)"
    )
    for obj in gehalten:  # gehaltene Objekte liegen am Boden
        assert sim.world.objects.position_of(obj) is not None
    haende = sum(a.hands.carried_mass_kg() for a in sim.agents if a.physics_v2)
    lhs, rhs = sim.world.objects.conservation_terms(held_mass_kg=haende)
    assert math.isclose(lhs, rhs, rel_tol=1e-9)


def test_verzweigung_flag_aus_v1_pfad_unveraendert():
    """D1: direkter Assert des v1-Zweigs (Golden beweist nur den Default-Pfad,
    nicht die Verzweigung) — daher direkter remove_dead()-Aufruf."""
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    toter = sim.agents[0]
    toter.alive = False
    pos = toter.pos
    pool_vorher = sim.world.get_cell(*pos)["carcasses"]

    sim.remove_dead()

    assert toter not in sim.agents
    assert sim.world.get_cell(*pos)["carcasses"] > pool_vorher  # CORPSE_ENERGY-Zell-Credit
    assert sim.world.objects.total_mass() == 0.0  # KEIN PhysObject


def _erzwungener_kill(sim):
    angreifer, ziel = sim.agents[0], sim.agents[1]
    ziel.pos = angreifer.pos  # adjazent (Radius 1 schließt dieselbe Zelle ein)
    angreifer.genes["aggression"] = (
        5.0  # threshold 4.7 → random.random() > 4.7 nie → Angriff sicher
    )
    angreifer.trust[ziel.id] = -1.0
    ziel.health = 1.0
    rueckgabe = angreifer._attack(sim.agents, {})
    assert not ziel.alive, "Testaufbau: der Angriff muss töten"
    return angreifer, rueckgabe


def test_v2_kill_ohne_doppel_energie():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    energie_vorher = sim.agents[0].energy
    angreifer, rueckgabe = _erzwungener_kill(sim)
    assert angreifer.energy == pytest.approx(energie_vorher), "kein Loot im v2-Modus"
    assert rueckgabe == 0.0
    sim.remove_dead()  # Energie aus dem Kill gibt es ausschließlich über den Kadaver
    assert any(o.kind == "carcass" for o, _ in sim.world.objects.all_objects())


def test_v1_kill_vergibt_weiterhin_loot():
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    ziel_energie = sim.agents[1].energy
    energie_vorher = sim.agents[0].energy
    angreifer, rueckgabe = _erzwungener_kill(sim)
    erwartet = min(240.0, energie_vorher + ziel_energie * 0.3)  # MAX_ENERGY-Klemme
    assert angreifer.energy == pytest.approx(erwartet)
    assert rueckgabe == pytest.approx(ziel_energie * 0.3)
