"""Physische Objekte der Physik v2.

Ein Objekt = intensiver Eigenschaftsvektor (props, [0,1]-normiert laut
Kalibrierung) + extensive Masse in kg. Masse ist die Erhaltungsgröße aller
Prozesse. `kind` ist reines Herkunfts-/Diagnoselabel und darf NIE Physik-Input
sein — Wirkung folgt ausschließlich aus props (Spec §3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials_v2 import MATERIALS_V2
from .props import N_PROPS_V2


@dataclass(eq=False)
class PhysObject:
    """Physische Objekte haben Identität, keine Wert-Gleichheit: zwei gleiche
    Granitsteine sind zwei verschiedene Objekte (wichtig u.a. für Hands.release).
    """

    props: np.ndarray
    mass: float
    kind: str = "unknown"

    def __post_init__(self) -> None:
        self.props = np.asarray(self.props, dtype=np.float32)
        if self.props.shape != (N_PROPS_V2,):
            raise ValueError(f"props braucht shape ({N_PROPS_V2},), hat {self.props.shape}")
        if self.mass <= 0:
            raise ValueError("mass muss > 0 sein")


def make_object(kind: str, mass: float) -> PhysObject:
    """Objekt eines Startmaterials erzeugen (kopiert den Materialvektor)."""
    return PhysObject(props=MATERIALS_V2[kind].copy(), mass=mass, kind=kind)
