"""Physik v2 — real kalibrierte Material- und Prozessphysik.

Spec: docs/superpowers/specs/2026-07-02-realphysik-emergenz-schnitt1-design.md.
Realitäts-Gate: jede Dimension / jedes Material / jeder Prozess braucht einen
Kalibrierungs-Eintrag (calibration.cal) — tests/environment/physics/
test_reality_gate.py erzwingt das als Merge-Bedingung.
"""

from .body import (
    CARRY_FRACTION_SUSTAINED,
    MAX_HELD,
    STRIKE_ENERGY_MAX_J,
    STRIKE_ENERGY_MIN_J,
    Body,
    Hands,
)
from .calibration import CALIBRATION, CalEntry, cal, entry_for, render_markdown
from .discovery import DISCOVERY_V2, DiscoveryV2
from .materials_v2 import MATERIALS_V2
from .objects import PhysObject, make_object
from .processes import (
    PROCESSES,
    CutResult,
    StrikeResult,
    cut,
    effective_sharpness,
    fracture_threshold_j,
    strike,
)
from .props import IDX2, N_PROPS_V2, PROP_DIMS_V2, pv

__all__ = [
    "CARRY_FRACTION_SUSTAINED",
    "MAX_HELD",
    "STRIKE_ENERGY_MAX_J",
    "STRIKE_ENERGY_MIN_J",
    "Body",
    "Hands",
    "CALIBRATION",
    "CalEntry",
    "cal",
    "entry_for",
    "render_markdown",
    "DISCOVERY_V2",
    "DiscoveryV2",
    "MATERIALS_V2",
    "PhysObject",
    "make_object",
    "PROCESSES",
    "CutResult",
    "StrikeResult",
    "cut",
    "effective_sharpness",
    "fracture_threshold_j",
    "strike",
    "IDX2",
    "N_PROPS_V2",
    "PROP_DIMS_V2",
    "pv",
]
