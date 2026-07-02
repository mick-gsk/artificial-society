"""Verkörperte Aktions-Mechanik (Plan 3a, Spec B4/B5) — Konstanten, Kopplung, do_*-Funktionen."""

from __future__ import annotations

import math

from artificial_society.agents.agent import MEAT_ENERGY
from artificial_society.environment.phys_objects import ObjectLayer
from artificial_society.environment.physics.actions import (
    DECAY_RATE,
    KCAL_PER_KG_PER_NUTRITION,
    SIM_ENERGY_PER_KCAL,
    TOX_SPOILAGE_CAP,
    TOX_SPOILAGE_PER_TICK,
    WORK_SIM_ENERGY_PER_JOULE,
    do_grasp,
    do_release,
    enforce_carry_budget,
)
from artificial_society.environment.physics.body import Body, Hands
from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.props import IDX2


def test_energie_kopplung_produkt_anker():
    """B5: nutrition-Konvention × SIM_ENERGY_PER_KCAL ≙ MEAT_ENERGY ± 1 (Produkt-Test —
    zwei Konstanten, eine Bilanz, kein stilles Driften)."""
    nutrition_raw_meat = float(MATERIALS_V2["raw_meat"][IDX2["nutrition"]])
    sim_energy_pro_kg = nutrition_raw_meat * KCAL_PER_KG_PER_NUTRITION * SIM_ENERGY_PER_KCAL
    assert abs(sim_energy_pro_kg - MEAT_ENERGY) <= 1.0  # 0.35*4000*0.032 = 44.8 ≈ 45


def test_arbeits_metabolik_anker_200_schlaege():
    """B5-Anker: 200 kräftige Schläge (45 J) kosten ≈ 0.3 Sim-Energie (bewusst klein —
    der reale Begrenzer ist die Ermüdung)."""
    kosten = 200 * 45.0 * WORK_SIM_ENERGY_PER_JOULE
    assert 0.2 < kosten < 0.4


def test_verwesungs_konstanten_anker():
    """B3.4: Halbwertszeit 10 Tage à 240 Ticks; Toxin-Kappe 0.6 nach ~5 Tagen erreicht."""
    assert math.isclose(DECAY_RATE, math.log(2.0) / (10 * 240), rel_tol=1e-12)
    assert TOX_SPOILAGE_PER_TICK == 5e-4
    assert TOX_SPOILAGE_CAP == 0.6
    # frisches Fleisch (0.02) erreicht die Kappe nach ≈ 1160 Ticks ≈ 5 Tagen
    ticks_bis_kappe = (TOX_SPOILAGE_CAP - 0.02) / TOX_SPOILAGE_PER_TICK
    assert 4 * 240 <= ticks_bis_kappe <= 6 * 240


def _setup(strength: float = 0.5):
    return Body(body_mass=70.0, strength=strength), Hands(), ObjectLayer(10, 10)


def test_grasp_nimmt_boden_objekt_im_radius_1():
    body, hands, layer = _setup()
    stein = make_object("granite", 2.0)
    layer.add(stein, (5, 6), source="spawned")  # Chebyshev-1 zu (5, 5)
    res = do_grasp(body, hands, layer, (5, 5), stein)
    assert res.ok and res.verb == "grasp"
    assert hands.held == [stein]
    assert layer.position_of(stein) is None  # nicht mehr am Boden


def test_grasp_noop_ausser_reichweite_volle_haende_budget():
    body, hands, layer = _setup()
    fern = make_object("granite", 1.0)
    layer.add(fern, (9, 9), source="spawned")
    assert not do_grasp(body, hands, layer, (5, 5), fern).ok  # außer Reichweite

    schwer = make_object("granite", 25.0)  # > carry_capacity 16.8 kg (70 kg, strength 0.5)
    layer.add(schwer, (5, 5), source="spawned")
    res = do_grasp(body, hands, layer, (5, 5), schwer)
    assert not res.ok and res.reason == "hands_full_or_too_heavy"
    assert layer.position_of(schwer) == (5, 5)  # No-op: bleibt liegen

    for _ in range(2):  # Hände füllen (MAX_HELD = 2)
        o = make_object("flint", 0.5)
        layer.add(o, (5, 5), source="spawned")
        assert do_grasp(body, hands, layer, (5, 5), o).ok
    dritter = make_object("flint", 0.5)
    layer.add(dritter, (5, 5), source="spawned")
    assert not do_grasp(body, hands, layer, (5, 5), dritter).ok


def test_release_legt_an_eigener_position_ab():
    body, hands, layer = _setup()
    stein = make_object("granite", 2.0)
    layer.add(stein, (5, 5), source="spawned")
    do_grasp(body, hands, layer, (5, 5), stein)
    res = do_release(body, hands, layer, (3, 2), stein)
    assert res.ok
    assert hands.held == []
    assert layer.position_of(stein) == (3, 2)
    nicht_gehalten = make_object("flint", 0.5)
    assert not do_release(body, hands, layer, (3, 2), nicht_gehalten).ok


def test_ueberlast_drop_wirft_schwerstes_bis_budget_passt():
    """B4: sinkt die Kapazität (Ermüdung), fällt zu Tick-Beginn das schwerste Objekt."""
    body, hands, layer = _setup()
    schwer = make_object("granite", 6.0)
    leicht = make_object("granite", 5.0)
    layer.add(schwer, (5, 5), source="spawned")
    layer.add(leicht, (5, 5), source="spawned")
    assert do_grasp(body, hands, layer, (5, 5), schwer).ok  # Kapazität ausgeruht: 16.8 kg
    assert do_grasp(body, hands, layer, (5, 5), leicht).ok
    body.fatigue = 1.0  # Kapazität fällt auf 8.4 kg — 11 kg sind Überlast
    dropped = enforce_carry_budget(body, hands, layer, (5, 5))
    assert dropped == [schwer]
    assert hands.held == [leicht]
    assert layer.position_of(schwer) == (5, 5)
    lhs, rhs = layer.conservation_terms(held_mass_kg=hands.carried_mass_kg())
    assert math.isclose(lhs, rhs, rel_tol=1e-9)
