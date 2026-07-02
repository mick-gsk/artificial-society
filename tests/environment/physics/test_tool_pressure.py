"""Das Kalibrierungsziel der Spec (§4): Körper-Grenzen machen Werkzeuge messbar wertvoll.

Volle Kette ohne jede Sim-Integration: Body → Schlagenergie → Feuerstein schlagen →
Klinge → Kadaver schneiden vs. bloße Hand; plus der Transport-Druck (ganzer Kadaver
untragbar → Zerteilen lohnt) und der Ermüdungs-Effekt auf die Schlagfähigkeit.
"""

from __future__ import annotations

import random

from artificial_society.environment.physics import Body, Hands, cut, make_object, strike
from artificial_society.environment.physics.props import IDX2


def _blade(rng_seed: int = 42):
    result = strike(
        make_object("flint", 0.8),
        make_object("granite", 1.0),
        Body(body_mass=70.0, strength=0.7).strike_energy_j(effort=1.0),  # 42.5 J
        random.Random(rng_seed),
    )
    assert result.fractured
    return max(result.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))


def test_tool_delta_full_chain_body_to_blade():
    # Ein realer Körper kann mit selbst geschlagener Klinge ≫ mehr Fleisch gewinnen.
    blade = _blade()
    hand_cut = cut(make_object("carcass", 25.0), None)
    blade_cut = cut(make_object("carcass", 25.0), blade)
    assert hand_cut.extracted is not None and blade_cut.extracted is not None
    assert blade_cut.extracted.mass > 6 * hand_cut.extracted.mass


def test_transport_pressure_forces_cutting():
    # Ganzer Kadaver: untragbar. Geschnittenes Stück: tragbar. Zerteilen LOHNT.
    body = Body(body_mass=70.0, strength=0.7)
    hands = Hands()
    assert not hands.can_grasp(make_object("carcass", 25.0), body)
    piece = cut(make_object("carcass", 25.0), _blade()).extracted
    assert piece is not None
    assert hands.grasp(piece, body)


def test_fatigue_erodes_knapping_capability():
    # Nach 200 kräftigen Schlägen: Feuerstein geht noch, Granit nicht mehr.
    body = Body(body_mass=70.0, strength=1.0)
    for _ in range(200):
        body.exert_strike(45.0)
    weak_energy = body.strike_energy_j(effort=1.0)  # 50 * (1 - 0.6*0.9) = 23.0 J
    hammer = make_object("granite", 1.2)
    assert not strike(make_object("granite", 0.8), hammer, weak_energy, random.Random(1)).fractured
    assert strike(make_object("flint", 0.8), hammer, weak_energy, random.Random(1)).fractured
