"""Startmaterial-Tests: Normierung, keine Start-Schärfe, Kalibrierungspflicht."""

from __future__ import annotations

from artificial_society.environment.physics.calibration import entry_for
from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.props import IDX2, N_PROPS_V2

EXPECTED = {
    "granite",
    "flint",
    "dry_wood",
    "plant_fiber",
    "clay_moist",
    "water",
    "berries",
    "raw_meat",
    "carcass",
}


def test_expected_seed_material_set():
    assert set(MATERIALS_V2) == EXPECTED


def test_vectors_have_v2_shape_and_are_normalized():
    for name, vec in MATERIALS_V2.items():
        assert vec.shape == (N_PROPS_V2,), name
        assert (vec >= 0.0).all() and (vec <= 1.0).all(), name


def test_no_seed_material_starts_sharp():
    # Schärfe ist NUR Prozessergebnis — auch Feuerstein ist roh nicht scharf.
    for name, vec in MATERIALS_V2.items():
        assert vec[IDX2["sharpness"]] == 0.0, name


def test_flint_is_the_knappable_one():
    flint = MATERIALS_V2["flint"]
    granite = MATERIALS_V2["granite"]
    assert flint[IDX2["brittleness"]] > 0.8
    assert flint[IDX2["grain_fineness"]] > 0.9
    assert granite[IDX2["grain_fineness"]] < 0.3


def test_every_seed_material_is_calibrated():
    for name in MATERIALS_V2:
        entry = entry_for("material", name)
        assert entry is not None, f"Material '{name}' ohne Kalibrierungs-Eintrag"
        assert entry.anchor.strip() and entry.source.strip()
