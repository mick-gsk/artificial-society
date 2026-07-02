"""F4/F5 (3a-Final-Review): ε-Cull + Verwesung gehaltener Objekte — ledger-bilanziert."""

from __future__ import annotations

import math
import random

import pytest

from artificial_society.environment.phys_objects import (
    EPSILON_CULL_MASS_KG,
    ObjectLayer,
    tick_decay,
)
from artificial_society.environment.physics.actions import DECAY_RATE
from artificial_society.environment.physics.body import Hands
from artificial_society.environment.physics.objects import make_object


def _layer():
    return ObjectLayer(8, 8, rng=random.Random(1))


def test_epsilon_cull_entfernt_husk_bilanziert():
    layer = _layer()
    layer.add(make_object("carcass", 5e-7), (2, 2), source="from_carcass")
    tick_decay(layer)
    assert layer.objects_at((2, 2)) == [], "Husk muss entfernt sein"
    assert layer.ledger["decayed"] == pytest.approx(5e-7), "Rest fließt nach ledger['decayed']"
    lhs, rhs = layer.conservation_terms()
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)


def test_epsilon_cull_wirkt_auch_auf_nicht_verwesende_winzlinge():
    """Winz-Fragmente aus strike/cut (Stein) verwesen nie — dürfen die Slots trotzdem
    nicht fluten: der Cull greift unabhängig vom Verwesungs-Gate."""
    layer = _layer()
    layer.add(make_object("granite", 5e-7), (3, 3), source="spawned")
    tick_decay(layer)
    assert layer.objects_at((3, 3)) == []
    assert layer.ledger["decayed"] == pytest.approx(5e-7)


def test_normale_objekte_bleiben_unberuehrt_vom_cull():
    layer = _layer()
    stein = make_object("granite", 2.0)
    layer.add(stein, (1, 1), source="spawned")
    tick_decay(layer)
    assert layer.objects_at((1, 1)) == [stein]
    assert stein.mass == 2.0  # Granit verwest nicht, Cull greift nicht
    assert EPSILON_CULL_MASS_KG == 1e-6


def test_gehaltenes_fleisch_verwest_mit():
    """F5: Frischhalte-Loophole zu — Tragen konserviert nicht."""
    layer = _layer()
    hands = Hands()
    fleisch = make_object("raw_meat", 1.0)
    hands.held.append(fleisch)
    layer.ledger["spawned"] += 1.0  # Handbestückung bilanzieren (Testaufbau)
    tick_decay(layer, hands_list=(hands,))
    assert fleisch.mass == pytest.approx(1.0 * (1.0 - DECAY_RATE))
    assert layer.ledger["decayed"] == pytest.approx(1.0 * DECAY_RATE)
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)


def test_gehaltener_stein_verwest_nicht():
    layer = _layer()
    hands = Hands()
    stein = make_object("granite", 1.0)
    hands.held.append(stein)
    layer.ledger["spawned"] += 1.0
    tick_decay(layer, hands_list=(hands,))
    assert stein.mass == 1.0


def test_husk_in_der_hand_wird_gecullt():
    layer = _layer()
    hands = Hands()
    kruemel = make_object("carcass", 5e-7)
    hands.held.append(kruemel)
    layer.ledger["spawned"] += 5e-7
    tick_decay(layer, hands_list=(hands,))
    assert hands.held == []
    assert layer.ledger["decayed"] == pytest.approx(5e-7)
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9)


def test_verwesung_der_haende_laeuft_ueber_den_sim_tick():
    """Integrations-Nachweis über das registrierte System (kein direkter Mechanik-Aufruf)."""
    from artificial_society.simulation import Simulation

    sim = Simulation(
        seed=42,
        physics_v2=True,
        headless=True,
        load_checkpoint=False,
        grid_w=20,
        grid_h=15,
        initial_population=8,
    )
    traeger = sim.agents[0]
    fleisch = make_object("raw_meat", 1.0)
    traeger.hands.held.append(fleisch)
    sim.world.objects.ledger["spawned"] += 1.0
    masse_vorher = fleisch.mass
    sim.step()
    assert fleisch.mass < masse_vorher, "Hand-Verwesung muss im Sim-Tick laufen"
