"""Bruchphysik: nur spröde+feinkörnige Ziele liefern scharfe Fragmente; Masse erhalten."""

from __future__ import annotations

import math
import random

from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.processes import strike
from artificial_society.environment.physics.props import IDX2

HAND_STRIKE_J = 20.0  # kräftiger Handschlag mit Schlagstein (Anker: 10–50 J)
STRONG_STRIKE_J = 45.0


def test_flint_strike_yields_sharp_fragments():
    result = strike(
        make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(42)
    )
    assert result.fractured
    assert len(result.fragments) >= 2
    assert max(f.props[IDX2["sharpness"]] for f in result.fragments) > 0.5


def test_granite_fragments_are_dull():
    result = strike(
        make_object("granite", 0.8), make_object("granite", 1.2), STRONG_STRIKE_J, random.Random(42)
    )
    assert result.fractured
    assert max(f.props[IDX2["sharpness"]] for f in result.fragments) < 0.15


def test_tough_wood_does_not_shatter_under_hand_strike():
    result = strike(
        make_object("dry_wood", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(42)
    )
    assert not result.fractured


def test_insufficient_energy_no_fracture():
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), 2.0, random.Random(42))
    assert not result.fractured


def test_soft_striker_cannot_knap():
    result = strike(
        make_object("flint", 0.8), make_object("dry_wood", 1.0), STRONG_STRIKE_J, random.Random(42)
    )
    assert not result.fractured


def test_mass_is_conserved():
    result = strike(
        make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(7)
    )
    assert result.fractured
    assert math.isclose(sum(f.mass for f in result.fragments), 0.8, rel_tol=1e-6)


def test_deterministic_given_same_rng_seed():
    a = strike(
        make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(3)
    )
    b = strike(
        make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(3)
    )
    assert [f.mass for f in a.fragments] == [f.mass for f in b.fragments]


def test_striking_creates_no_nutrition():
    # Kein Zauber: ein Schlag erzeugt keinen Nährwert.
    result = strike(
        make_object("flint", 0.8), make_object("granite", 1.0), HAND_STRIKE_J, random.Random(42)
    )
    for f in result.fragments:
        assert f.props[IDX2["nutrition"]] == 0.0


def test_tough_and_liquid_targets_never_shatter_even_at_high_energy():
    # Der Kalibrierungs-Anker sagt: zähe Stoffe zersplittern nicht — auch nicht
    # bei hoher Energie. Flüssigkeiten (Wasser) zerbrechen erst recht nicht.
    hammer = make_object("granite", 1.2)
    for kind in ("dry_wood", "raw_meat", "carcass", "water"):
        result = strike(make_object(kind, 0.5), hammer, 60.0, random.Random(42))
        assert not result.fractured, kind
