"""Verkörperte Aktions-Mechanik (Plan 3a, Spec B4/B5) — Konstanten, Kopplung, do_*-Funktionen."""

from __future__ import annotations

import math

from artificial_society.agents.agent import MEAT_ENERGY
from artificial_society.environment.physics.actions import (
    DECAY_RATE,
    KCAL_PER_KG_PER_NUTRITION,
    SIM_ENERGY_PER_KCAL,
    TOX_SPOILAGE_CAP,
    TOX_SPOILAGE_PER_TICK,
    WORK_SIM_ENERGY_PER_JOULE,
)
from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
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
