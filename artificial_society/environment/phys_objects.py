"""ObjectLayer — sparse Registry physischer Objekte (Plan 3a, Spec B1–B3).

Die Objekt-Schicht lebt als Schwester-Attribut ``world.objects`` neben den
Zell-Arrays ``world.F/S`` — sie schreibt NIE in Zell-Arrays und taucht NIE in
``world.obj[y][x]`` auf (GPU-Garantie, Spec Prinzip 6). Masse ist die
Erhaltungsgröße: jeder Zu-/Abfluss läuft über den Ledger
(spawned / from_carcass / eaten / decayed); Bewegungen Boden↔Hand sind
ledger-neutral. DiscoveryV2 wird PRO Welt instanziert (kein Modul-Singleton).
"""

from __future__ import annotations

import random as _random
import types
from collections.abc import Iterator

import numpy as np

from artificial_society.environment.physics.discovery import DiscoveryV2
from artificial_society.environment.physics.objects import PhysObject
from artificial_society.environment.physics.props import N_PROPS_V2

_LEDGER_SOURCES = ("spawned", "from_carcass")


class ObjectLayer:
    def __init__(self, width: int, height: int, rng=None):
        self.width = int(width)
        self.height = int(height)
        # Duck-typed RNG (braucht .random/.uniform/.randrange). Default: das
        # global via artificial_society.rng.seed_all geseedete random-Modul.
        self.rng = rng if rng is not None else _random
        self._by_pos: dict = {}  # (x, y) -> list[PhysObject]
        self._pos_by_id: dict = {}  # id(obj) -> (x, y); prozess-lokal, s. __setstate__
        self.discovery = DiscoveryV2()
        self.ledger: dict = {"spawned": 0.0, "from_carcass": 0.0, "eaten": 0.0, "decayed": 0.0}

    # -- Kern-API ------------------------------------------------------------
    def add(self, obj: PhysObject, pos, source: str | None = None) -> None:
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"pos {pos!r} liegt außerhalb des {self.width}x{self.height}-Grids")
        props = np.asarray(obj.props)
        if props.shape != (N_PROPS_V2,) or bool(np.any(props < 0.0)) or bool(np.any(props > 1.0)):
            raise ValueError("props müssen 13-dimensional in [0,1] sein")
        if obj.mass <= 0.0:
            raise ValueError("mass muss > 0 sein")
        if id(obj) in self._pos_by_id:
            raise ValueError("Objekt liegt bereits im Layer (Doppel-Add)")
        if source is not None:
            if source not in _LEDGER_SOURCES:
                raise ValueError(f"unbekannte Ledger-Quelle: {source!r}")
            self.ledger[source] += obj.mass
        self._by_pos.setdefault((x, y), []).append(obj)
        self._pos_by_id[id(obj)] = (x, y)

    def remove(self, obj: PhysObject) -> None:
        """Entfernt per Identität (zwei wertgleiche Steine sind zwei Objekte)."""
        pos = self._pos_by_id.pop(id(obj), None)
        if pos is None:
            raise KeyError("Objekt liegt nicht im Layer")
        bucket = self._by_pos[pos]
        for i, kandidat in enumerate(bucket):
            if kandidat is obj:
                del bucket[i]
                break
        if not bucket:
            del self._by_pos[pos]

    def position_of(self, obj: PhysObject):
        return self._pos_by_id.get(id(obj))

    def objects_at(self, pos) -> list:
        return list(self._by_pos.get((int(pos[0]), int(pos[1])), ()))

    def objects_near(self, pos, radius: int) -> list:
        """Alle Boden-Objekte im Chebyshev-Radius, als (obj, pos)-Paare.

        Iteriert die sparse Positions-Map (Insertion-Order ⇒ deterministisch).
        """
        x, y = int(pos[0]), int(pos[1])
        out = []
        for (px, py), bucket in self._by_pos.items():
            if abs(px - x) <= radius and abs(py - y) <= radius:
                out.extend((obj, (px, py)) for obj in bucket)
        return out

    def all_objects(self) -> Iterator:
        for pos, bucket in self._by_pos.items():
            for obj in list(bucket):
                yield obj, pos

    def total_mass(self) -> float:
        """Gesamtmasse am Boden (Hände zählt die Erhaltungs-Invariante separat)."""
        return sum(obj.mass for bucket in self._by_pos.values() for obj in bucket)

    def conservation_terms(self, held_mass_kg: float = 0.0):
        """(lhs, rhs) der Massen-Invariante (Spec D1):
        boden + hände + eaten + decayed == spawned + from_carcass."""
        lhs = self.total_mass() + held_mass_kg + self.ledger["eaten"] + self.ledger["decayed"]
        rhs = self.ledger["spawned"] + self.ledger["from_carcass"]
        return lhs, rhs

    # -- Pickling: id()-Rückwärts-Map ist prozess-lokal; Modul-rng nicht picklebar --
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_pos_by_id")
        if isinstance(state.get("rng"), types.ModuleType):
            state["rng"] = None  # Modul-Default ist prozess-global, nicht picklebar
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.__dict__.get("rng") is None:
            self.rng = _random  # Modul-Default wieder anbinden (global via seed_all geseedet)
        self._pos_by_id = {id(obj): pos for pos, bucket in self._by_pos.items() for obj in bucket}
