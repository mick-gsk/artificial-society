"""Schneidephysik: scharfe Klinge schlägt bloße Hand um ein Vielfaches; Masse erhalten."""

from __future__ import annotations

import math
import random

import pytest

from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.processes import cut, strike
from artificial_society.environment.physics.props import IDX2


def _sharpest_flint_fragment():
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), 20.0, random.Random(42))
    return max(result.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))


def test_blade_beats_bare_hand_by_wide_margin():
    hand = cut(make_object("carcass", 20.0), None)
    blade = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    assert hand.extracted is not None and blade.extracted is not None
    assert blade.extracted.mass > 6 * hand.extracted.mass


def test_cut_conserves_mass():
    result = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    assert math.isclose(result.extracted.mass + result.remainder.mass, 20.0, rel_tol=1e-6)


def test_cut_does_not_change_properties():
    # Schneiden macht nichts nahrhafter oder giftiger — nur kleiner.
    result = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    assert (result.extracted.props == MATERIALS_V2["carcass"]).all()
    assert (result.remainder.props == MATERIALS_V2["carcass"]).all()


def test_blade_beats_dull_granite_fragment():
    dull_result = strike(
        make_object("granite", 0.8), make_object("granite", 1.2), 45.0, random.Random(42)
    )
    dull = max(dull_result.fragments, key=lambda f: float(f.props[IDX2["sharpness"]]))
    blade_cut = cut(make_object("carcass", 20.0), _sharpest_flint_fragment())
    dull_cut = cut(make_object("carcass", 20.0), dull)
    assert blade_cut.extracted.mass > 4 * dull_cut.extracted.mass


def test_blade_cannot_cut_rock():
    # Gestein ist nicht schneidbar — auch nicht mit der schärfsten Klinge.
    result = cut(make_object("granite", 10.0), _sharpest_flint_fragment())
    assert result.extracted is None
    assert result.remainder.mass == pytest.approx(10.0)


def test_no_blade_duplication_by_cutting_a_blade():
    # Eine Klinge lässt sich nicht in zwei gleich scharfe Klingen zerschneiden.
    blade_a = _sharpest_flint_fragment()
    blade_b = _sharpest_flint_fragment()
    assert cut(blade_a, blade_b).extracted is None


def test_bare_hand_cannot_cut_stone():
    assert cut(make_object("granite", 10.0), None).extracted is None
