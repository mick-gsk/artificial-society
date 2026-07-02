"""Prozesse der Physik v2 — Wirkung folgt IMMER aus Eigenschaften, nie aus Labels.

Erhaltung: die Masse aller Outputs eines Prozesses ist exakt die Masse der
Inputs (Tests sichern das). Zufall läuft ausschließlich über einen explizit
übergebenen random.Random (Determinismus-Kontrakt des Repos).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .calibration import cal
from .objects import PhysObject
from .props import IDX2

# Schlag-Physik (realer Anker: siehe cal()-Eintrag unten)
FRACTURE_BASE_J_PER_KG = 60.0
MIN_STRIKER_HARDNESS = 0.5
MIN_HARDNESS_MARGIN = -0.15  # Schlagstein darf höchstens 0.15 weicher sein als das Ziel


@dataclass
class StrikeResult:
    fractured: bool
    fragments: list = field(default_factory=list)


def fracture_threshold_j(target: PhysObject) -> float:
    """Nötige Aufprallenergie: zähere Ziele und mehr Masse brauchen mehr Energie."""
    brittleness = float(target.props[IDX2["brittleness"]])
    return FRACTURE_BASE_J_PER_KG * (1.05 - brittleness) * max(target.mass, 0.1)


def strike(
    target: PhysObject,
    striker: PhysObject,
    impact_energy_j: float,
    rng: random.Random,
) -> StrikeResult:
    """Hart-Hammer-Perkussion: Ziel zerbricht, wenn Schläger hart genug und
    Energie über der Bruchschwelle liegt. Fragment-Schärfe ergibt sich aus
    Sprödigkeit × Gefügefeinheit (muscheliger Bruch), nie aus einem Label."""
    striker_hardness = float(striker.props[IDX2["hardness"]])
    target_hardness = float(target.props[IDX2["hardness"]])
    if striker_hardness < MIN_STRIKER_HARDNESS:
        return StrikeResult(fractured=False)
    if striker_hardness - target_hardness < MIN_HARDNESS_MARGIN:
        return StrikeResult(fractured=False)
    if impact_energy_j < fracture_threshold_j(target):
        return StrikeResult(fractured=False)

    n_fragments = rng.randint(2, 3)
    weights = [0.5 + rng.random() for _ in range(n_fragments)]
    total = sum(weights)
    brittleness = float(target.props[IDX2["brittleness"]])
    grain = float(target.props[IDX2["grain_fineness"]])
    fragments = []
    for w in weights:
        props = target.props.copy()
        props[IDX2["sharpness"]] = brittleness * grain * (0.6 + 0.4 * rng.random())
        fragments.append(
            PhysObject(props=props, mass=(w / total) * target.mass, kind=f"{target.kind}_fragment")
        )
    return StrikeResult(fractured=True, fragments=fragments)


PROCESSES: dict = {"strike": strike}

cal(
    "process",
    "strike",
    "Hart-Hammer-Perkussion: nur spröde, feinkörnige Gesteine (muscheliger Bruch) liefern "
    "scharfe Abschläge; nötige Schlagenergie im Bereich eines kräftigen Handschlags (10–50 J); "
    "zähe Stoffe (Holz, Fleisch) zersplittern so nicht",
    "Experimentelle Archäologie: Feuersteinschlagen/Lithik; Bruchmechanik spröder Stoffe",
)
