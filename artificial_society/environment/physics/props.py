"""Physik v2 — reale Eigenschaftsdimensionen.

Jede Dimension ist auf [0, 1] normiert über einen dokumentierten realen Anker
(z.B. hardness = Mohs / 10). Die Normierungsformeln stehen in calibration.py
(SSOT) und im generierten docs/physics/kalibrierung.md.

`sharpness` ist eine abgeleitete Größe: KEIN Startmaterial darf sie > 0 setzen —
Schärfe entsteht ausschließlich als Prozessergebnis (Bruch). Ein Gate-Test
erzwingt das.
"""

from __future__ import annotations

import numpy as np

PROP_DIMS_V2 = [
    "hardness",
    "density",
    "brittleness",
    "tensile_strength",
    "sharpness",
    "flammability",
    "ignition_temp",
    "melting_point",
    "thermal_conductivity",
    "nutrition",
    "toxicity",
    "moisture",
    "grain_fineness",
]
IDX2 = {name: i for i, name in enumerate(PROP_DIMS_V2)}
N_PROPS_V2 = len(PROP_DIMS_V2)  # 13


def pv(**kwargs: float) -> np.ndarray:
    """Eigenschaftsvektor aus Keyword-Args; fehlende Dimensionen = 0."""
    v = np.zeros(N_PROPS_V2, dtype=np.float32)
    for k, val in kwargs.items():
        v[IDX2[k]] = float(val)
    return v
