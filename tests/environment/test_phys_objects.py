"""ObjectLayer (Plan 3a, Spec B1): sparse Registry, Ledger, per-Welt-Discovery."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from artificial_society.environment.phys_objects import ObjectLayer
from artificial_society.environment.physics.objects import PhysObject, make_object
from artificial_society.environment.physics.props import pv


def _layer() -> ObjectLayer:
    return ObjectLayer(10, 8)


def test_add_und_objects_at():
    layer = _layer()
    a = make_object("granite", 2.0)
    layer.add(a, (3, 4))
    assert layer.objects_at((3, 4)) == [a]
    assert layer.objects_at((0, 0)) == []
    assert layer.position_of(a) == (3, 4)
    assert layer.total_mass() == 2.0


def test_add_validiert_props_masse_und_bounds():
    layer = _layer()
    with pytest.raises(ValueError):  # Props außerhalb [0,1]
        layer.add(PhysObject(props=pv(hardness=1.5), mass=1.0), (0, 0))
    with pytest.raises(ValueError):  # außerhalb des Grids
        layer.add(make_object("granite", 1.0), (10, 0))
    obj = make_object("granite", 1.0)
    layer.add(obj, (1, 1))
    with pytest.raises(ValueError):  # Doppel-Add desselben Objekts
        layer.add(obj, (2, 2))
    with pytest.raises(ValueError):  # unbekannte Ledger-Quelle
        layer.add(make_object("flint", 1.0), (0, 0), source="geschenkt")


def test_remove_ist_identitaetsbasiert():
    layer = _layer()
    a = make_object("granite", 1.0)
    b = make_object("granite", 1.0)  # wertgleich, aber anderes Objekt
    layer.add(a, (2, 2))
    layer.add(b, (2, 2))
    layer.remove(a)
    assert layer.objects_at((2, 2)) == [b]
    with pytest.raises(KeyError):
        layer.remove(a)


def test_objects_near_chebyshev():
    layer = _layer()
    nah = make_object("flint", 0.5)
    diagonal = make_object("flint", 0.5)
    fern = make_object("flint", 0.5)
    layer.add(nah, (5, 5))
    layer.add(diagonal, (6, 6))
    layer.add(fern, (8, 5))
    gefunden = layer.objects_near((5, 5), radius=1)
    assert {id(o) for o, _ in gefunden} == {id(nah), id(diagonal)}
    assert (diagonal, (6, 6)) in gefunden


def test_ledger_und_erhaltungsterme():
    layer = _layer()
    layer.add(make_object("granite", 3.0), (1, 1), source="spawned")
    layer.add(make_object("carcass", 70.0), (2, 2), source="from_carcass")
    layer.add(make_object("flint", 1.0), (3, 3))  # ledger-neutral (z. B. abgelegt)
    assert layer.ledger["spawned"] == 3.0
    assert layer.ledger["from_carcass"] == 70.0
    assert layer.ledger["eaten"] == 0.0 and layer.ledger["decayed"] == 0.0
    lhs, rhs = layer.conservation_terms(held_mass_kg=0.0)
    # Boden 74 == spawned 3 + from_carcass 70 + neutral 1 → neutral zählt links,
    # also balanciert die Invariante nur, wenn neutrale Adds Bewegungen sind
    # (Hand→Boden). Hier: bewusst unbalanciert um genau 1.0.
    assert math.isclose(lhs - rhs, 1.0, rel_tol=1e-9)


def test_discovery_ist_instanz_pro_layer():
    a, b = _layer(), _layer()
    assert a.discovery is not b.discovery
    a.discovery.register(np.zeros(13, dtype=np.float32))
    assert a.discovery.known_ids() and not b.discovery.known_ids()


def test_pickle_roundtrip_baut_rueckwaertsmap_neu():
    """Default-rng ist das random-MODUL (nicht picklebar) — der Roundtrip MUSS es
    trotzdem überleben: exakt der Checkpoint-Pfad (_save_checkpoint pickelt world
    inkl. world.objects, simulation.py:343)."""
    import random as random_mod

    layer = _layer()  # Default-rng = random-Modul
    layer.add(make_object("granite", 2.5), (4, 3), source="spawned")
    layer.add(make_object("carcass", 70.0), (4, 3), source="from_carcass")
    layer2 = pickle.loads(pickle.dumps(layer))
    objekte = layer2.objects_at((4, 3))
    assert len(objekte) == 2
    assert all(layer2.position_of(o) == (4, 3) for o in objekte)
    assert layer2.ledger == layer.ledger
    assert math.isclose(layer2.total_mass(), layer.total_mass(), rel_tol=1e-12)
    assert layer2.rng is random_mod  # Modul-Default wieder angebunden
    layer2.rng.random()  # und benutzbar


def test_pickle_erhaelt_explizite_rng_instanz():
    import random

    layer = ObjectLayer(10, 8, rng=random.Random(5))
    erwartet = random.Random(5).random()
    layer2 = pickle.loads(pickle.dumps(layer))
    assert isinstance(layer2.rng, random.Random)
    assert layer2.rng.random() == erwartet  # RNG-Zustand überlebt das Pickling
