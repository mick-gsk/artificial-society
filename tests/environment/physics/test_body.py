"""Body v1: Tragkapazität und Schlagenergie aus Masse, Kraft, Ermüdung."""

from __future__ import annotations

import pytest

from artificial_society.environment.physics.body import (
    CARRY_FRACTION_SUSTAINED,
    STRIKE_ENERGY_MAX_J,
    Body,
)


def test_carry_capacity_baseline():
    # 70-kg-Körper, volle Kraft, ausgeruht: 30 % des Körpergewichts.
    assert Body(body_mass=70.0, strength=1.0).carry_capacity_kg() == pytest.approx(21.0)
    assert CARRY_FRACTION_SUSTAINED == 0.30


def test_carry_capacity_scales_with_strength_and_fatigue():
    weak = Body(body_mass=70.0, strength=0.5)
    assert weak.carry_capacity_kg() == pytest.approx(16.8)  # 21 * (0.6+0.4*0.5)
    tired = Body(body_mass=70.0, strength=1.0, fatigue=1.0)
    assert tired.carry_capacity_kg() == pytest.approx(10.5)  # 21 * 0.5


def test_strike_energy_baseline_and_anchor_range():
    # Kräftiger Schlag eines starken, ausgeruhten Körpers = oberes Anker-Ende (50 J).
    strong = Body(body_mass=70.0, strength=1.0)
    assert strong.strike_energy_j(effort=1.0) == pytest.approx(STRIKE_ENERGY_MAX_J)
    # Minimaler Effort bleibt über 0 (5 J * Kraftfaktor).
    assert strong.strike_energy_j(effort=0.0) == pytest.approx(5.0)


def test_strike_energy_scales_and_clamps_effort():
    body = Body(body_mass=70.0, strength=0.7)
    assert body.strike_energy_j(effort=1.0) == pytest.approx(42.5)  # 50 * 0.85
    assert body.strike_energy_j(effort=2.0) == body.strike_energy_j(effort=1.0)
    assert body.strike_energy_j(effort=-1.0) == body.strike_energy_j(effort=0.0)


def test_fatigue_dampens_strike_energy():
    exhausted = Body(body_mass=70.0, strength=1.0, fatigue=1.0)
    assert exhausted.strike_energy_j(effort=1.0) == pytest.approx(20.0)  # 50 * 0.4


def test_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        Body(body_mass=0.0, strength=0.5)
    with pytest.raises(ValueError):
        Body(body_mass=70.0, strength=1.5)
    with pytest.raises(ValueError):
        Body(body_mass=70.0, strength=0.5, fatigue=-0.1)
