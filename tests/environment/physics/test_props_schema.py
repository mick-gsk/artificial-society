"""Schema-Tests Physik v2: Dimensionsliste, Index-Map, Vektor-Builder."""

from __future__ import annotations

import numpy as np
import pytest

from artificial_society.environment.physics.calibration import cal, entry_for
from artificial_society.environment.physics.props import IDX2, N_PROPS_V2, PROP_DIMS_V2, pv


def test_schema_has_13_unique_dims():
    assert N_PROPS_V2 == 13
    assert len(set(PROP_DIMS_V2)) == 13
    assert {name: i for i, name in enumerate(PROP_DIMS_V2)} == IDX2


def test_pv_builds_float32_vector_with_zero_defaults():
    v = pv(hardness=0.7, moisture=0.1)
    assert v.dtype == np.float32
    assert v.shape == (N_PROPS_V2,)
    assert v[IDX2["hardness"]] == pytest.approx(0.7)
    assert v[IDX2["moisture"]] == pytest.approx(0.1)
    assert float(v.sum()) == pytest.approx(0.8)


def test_pv_rejects_unknown_dimension():
    with pytest.raises(KeyError):
        pv(mana=1.0)


def test_every_dim_has_calibration_entry():
    for dim in PROP_DIMS_V2:
        entry = entry_for("dim", dim)
        assert entry is not None, f"Dimension '{dim}' ohne Kalibrierungs-Eintrag"
        assert entry.anchor.strip() and entry.source.strip()


def test_cal_rejects_empty_anchor_and_bad_kind():
    with pytest.raises(ValueError):
        cal("dim", "x", "", "quelle")
    with pytest.raises(ValueError):
        cal("spell", "x", "anker", "quelle")
