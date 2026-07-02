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
MIN_FRACTURE_BRITTLENESS = 0.35  # zähe Stoffe (Holz, Fleisch) verformen/dämpfen statt zu zersplittern; Flüssigkeiten (brittleness 0) zerbrechen nicht


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
    if float(target.props[IDX2["brittleness"]]) < MIN_FRACTURE_BRITTLENESS:
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


# Schneide-Physik (realer Anker: siehe cal()-Eintrag unten)
BARE_HAND_SHARPNESS = 0.05  # Hände/Zähne/Reißen
MAX_CUT_FRACTION = 0.9
BARE_HAND_HARDNESS = (
    0.30  # Hände/Zähne: Zahnschmelz Mohs ~5 (0.5), wirksam begrenzt durch Biss-/Greifmechanik
)
CUTTABLE_HARDNESS_CEILING = (
    0.35  # nur weiche Stoffe sind schneidbar; Gestein wird geschlagen/geschliffen, nie geschnitten
)


@dataclass
class CutResult:
    extracted: PhysObject | None
    remainder: PhysObject


def effective_sharpness(tool: PhysObject | None) -> float:
    """Ohne Werkzeug: Hände/Zähne. Mit Werkzeug: Schärfe, getragen von Härte —
    eine scharfe, aber weiche Kante verformt sich beim Schnitt."""
    if tool is None:
        return BARE_HAND_SHARPNESS
    return float(tool.props[IDX2["sharpness"]]) * (0.5 + 0.5 * float(tool.props[IDX2["hardness"]]))


def cut(target: PhysObject, tool: PhysObject | None) -> CutResult:
    """Ein Schneidevorgang trennt einen Massenanteil ab. Ertrag steigt mit
    effektiver Schärfe des Werkzeugs, sinkt mit Zähigkeit des Ziels — und nur
    Ziele, die deutlich weicher als das Werkzeug und insgesamt weich sind,
    lassen sich überhaupt schneiden (Gestein wird geschlagen, nicht geschnitten).
    Eigenschaften bleiben unverändert (nur kleiner) — Masse exakt erhalten."""
    tool_hardness = BARE_HAND_HARDNESS if tool is None else float(tool.props[IDX2["hardness"]])
    target_hardness = float(target.props[IDX2["hardness"]])
    if tool_hardness <= target_hardness:
        return CutResult(extracted=None, remainder=target)
    cuttability = max(0.0, 1.0 - target_hardness / CUTTABLE_HARDNESS_CEILING)
    toughness = 1.0 - float(target.props[IDX2["brittleness"]])
    yield_fraction = min(
        effective_sharpness(tool) * max(0.0, 1.15 - toughness) * cuttability,
        MAX_CUT_FRACTION,
    )
    extracted_mass = yield_fraction * target.mass
    if extracted_mass <= 1e-9:
        return CutResult(extracted=None, remainder=target)
    extracted = PhysObject(
        props=target.props.copy(), mass=extracted_mass, kind=f"{target.kind}_piece"
    )
    remainder = PhysObject(
        props=target.props.copy(), mass=target.mass - extracted_mass, kind=target.kind
    )
    return CutResult(extracted=extracted, remainder=remainder)


PROCESSES: dict = {"strike": strike, "cut": cut}

cal(
    "process",
    "strike",
    "Hart-Hammer-Perkussion: nur spröde, feinkörnige Gesteine (muscheliger Bruch) liefern "
    "scharfe Abschläge; nötige Schlagenergie im Bereich eines kräftigen Handschlags (10–50 J); "
    "zähe Stoffe (Holz, Fleisch) zersplittern so nicht",
    "Experimentelle Archäologie: Feuersteinschlagen/Lithik; Bruchmechanik spröder Stoffe",
)

cal(
    "process",
    "cut",
    "Schneiden: Ertrag steigt mit Kantenschärfe × Härte des Werkzeugs und sinkt mit der "
    "Zähigkeit des Ziels; schneidbar sind nur Stoffe, die weicher als das Werkzeug und "
    "insgesamt weich sind (Fleisch, Pflanzen, bedingt Holz) — Gestein ist nicht schneidbar, "
    "sondern nur schlagbearbeitbar; einen Kadaver mit bloßer Hand zu zerwirken ist nahezu "
    "unmöglich, mit Steinklinge effizient",
    "Experimentelle Archäologie: Zerwirken mit Steinklingen",
)
