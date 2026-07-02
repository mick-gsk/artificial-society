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
