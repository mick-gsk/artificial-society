"""PhysObject: intensiver Eigenschaftsvektor + extensive Masse (kg)."""

from __future__ import annotations

import numpy as np
import pytest

from artificial_society.environment.physics.materials_v2 import MATERIALS_V2
from artificial_society.environment.physics.objects import PhysObject, make_object
from artificial_society.environment.physics.props import N_PROPS_V2


def test_make_object_copies_material_vector():
    obj = make_object("flint", 0.8)
    assert obj.kind == "flint"
    assert obj.mass == pytest.approx(0.8)
    obj.props[0] = 0.123
    assert MATERIALS_V2["flint"][0] != np.float32(0.123)  # Original unverändert


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        PhysObject(props=np.zeros(3, dtype=np.float32), mass=1.0)


def test_rejects_non_positive_mass():
    with pytest.raises(ValueError):
        PhysObject(props=np.zeros(N_PROPS_V2, dtype=np.float32), mass=0.0)
    with pytest.raises(ValueError):
        make_object("granite", -1.0)
