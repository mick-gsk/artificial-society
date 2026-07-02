"""ObjectLayer — sparse Registry physischer Objekte (Plan 3a, Spec B1–B3).

Die Objekt-Schicht lebt als Schwester-Attribut ``world.objects`` neben den
Zell-Arrays ``world.F/S`` — sie schreibt NIE in Zell-Arrays und taucht NIE in
``world.obj[y][x]`` auf (GPU-Garantie, Spec Prinzip 6). Masse ist die
Erhaltungsgröße: jeder Zu-/Abfluss läuft über den Ledger
(spawned / from_carcass / eaten / decayed); Bewegungen Boden↔Hand sind
ledger-neutral. DiscoveryV2 wird PRO Welt instanziert (kein Modul-Singleton).
"""

from __future__ import annotations

import math
import random as _random
import types
from collections.abc import Iterator

import numpy as np

from artificial_society.environment.physics.calibration import cal
from artificial_society.environment.physics.discovery import DiscoveryV2
from artificial_society.environment.physics.objects import PhysObject, make_object
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


# ---------------------------------------------------------------------------
# Spawning (Spec B2): Biome-gebunden, kalibrierte Massen, langsame Regeneration
# ---------------------------------------------------------------------------
SPAWN_INITIAL_DENSITY = 0.03  # 3 % der passenden Biom-Zellen tragen initial ein Objekt
SPAWN_RATE_PER_CELL_TICK = 1e-5  # ≈ 0.0024 Objekte je Biom-Zelle und Tag (240 Ticks)

# (material, biome_tag, rate_mult, mass_min_kg, mass_max_kg).
# "shore" = Nicht-Wasser-Zelle mit mind. einem Wasser-Nachbarn (Chebyshev 1).
SPAWN_TABLE = (
    ("granite", "mountain", 1.0, 0.5, 8.0),
    ("flint", "mountain", 0.5, 0.3, 4.0),  # Knollen
    ("flint", "grassland", 0.05, 0.3, 4.0),  # selten: Kiesel
    ("dry_wood", "forest", 1.0, 0.5, 6.0),
    ("plant_fiber", "grassland", 1.0, 0.05, 0.4),
    ("plant_fiber", "swamp", 1.0, 0.05, 0.4),
    ("clay_moist", "swamp", 1.0, 0.5, 5.0),
    ("clay_moist", "shore", 1.0, 0.5, 5.0),
)

# Vom Realitäts-Gate geprüfte Spawn-Parameter (kind "spawn").
CALIBRATED_SPAWN_PARAMS = (
    "initial_density",
    "regen_rate",
    "granite",
    "flint",
    "dry_wood",
    "plant_fiber",
    "clay_moist",
)


def _eligible_cells(biomes, biome_tag: str) -> list:
    """Zellen (x, y), auf denen ein biome_tag spawnen darf; Scan-Reihenfolge row-major."""
    h, w = len(biomes), len(biomes[0])
    if biome_tag != "shore":
        return [(x, y) for y in range(h) for x in range(w) if biomes[y][x] == biome_tag]
    out = []
    for y in range(h):
        for x in range(w):
            if biomes[y][x] == "water":
                continue
            nachbar_wasser = any(
                0 <= y + dy < h and 0 <= x + dx < w and biomes[y + dy][x + dx] == "water"
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if dx or dy
            )
            if nachbar_wasser:
                out.append((x, y))
    return out


def _cells_for(layer: ObjectLayer, biomes, biome_tag: str) -> list:
    """Pro Layer gecachte Eignungs-Listen (Biome sind nach Weltgenerierung statisch)."""
    cache = getattr(layer, "_spawn_cells", None)
    if cache is None:
        cache = {}
        layer._spawn_cells = cache
    if biome_tag not in cache:
        cache[biome_tag] = _eligible_cells(biomes, biome_tag)
    return cache[biome_tag]


def _poisson(lam: float, rng) -> int:
    """Knuth-Poisson — für die winzigen Pro-Tick-Raten (λ ≪ 1) fast immer 1 RNG-Draw."""
    if lam <= 0.0:
        return 0
    schwelle = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= schwelle:
            return k
        k += 1


def seed_initial(layer: ObjectLayer, biomes) -> None:
    """Start-Seeding beim Welt-Aufbau (nur physics_v2): 3 % der passenden Zellen."""
    for material, biome_tag, rate_mult, m_lo, m_hi in SPAWN_TABLE:
        for pos in _cells_for(layer, biomes, biome_tag):
            if layer.rng.random() < SPAWN_INITIAL_DENSITY * rate_mult:
                layer.add(
                    make_object(material, layer.rng.uniform(m_lo, m_hi)), pos, source="spawned"
                )


def tick_spawn(layer: ObjectLayer, biomes) -> None:
    """Langsame Regeneration: Poisson über alle geeigneten Zellen einer Quelle."""
    for material, biome_tag, rate_mult, m_lo, m_hi in SPAWN_TABLE:
        cells = _cells_for(layer, biomes, biome_tag)
        if not cells:
            continue
        lam = SPAWN_RATE_PER_CELL_TICK * rate_mult * len(cells)
        for _ in range(_poisson(lam, layer.rng)):
            pos = cells[layer.rng.randrange(len(cells))]
            layer.add(make_object(material, layer.rng.uniform(m_lo, m_hi)), pos, source="spawned")


cal(
    "spawn",
    "initial_density",
    "Start-Seeding: 3 % der geeigneten Biom-Zellen tragen initial ein Objekt "
    "(SPAWN_INITIAL_DENSITY = 0.03, je Quelle skaliert mit rate_mult)",
    "Größenordnung Oberflächen-Vorkommen von Lesesteinen/Totholz; Pilot-feinjustierbar (Spec B2)",
)
cal(
    "spawn",
    "regen_rate",
    "Regeneration 1e-5 Objekte je Zelle und Tick ≈ 0.0024/Zelle/Tag (240 Ticks/Tag); auf "
    "200×200 mit ~15 % Gebirge ≈ 14 neue Steine/Tag — versiegt nicht, flutet nicht",
    "Auslegungsrechnung Spec B2 (Pilot-feinjustierbar, nie zur Laufzeit pro Agent)",
)
cal(
    "spawn",
    "granite",
    "Granit-Gerölle 0.5–8 kg im Gebirge (Lesesteine/Hangschutt)",
    "Geologie: Hangschutt/Lesesteine im Mittelgebirge",
)
cal(
    "spawn",
    "flint",
    "Feuerstein 0.3–4 kg: Knollen im Gebirge (rate_mult 0.5), selten als Kiesel im "
    "Grasland (rate_mult 0.05)",
    "Geologie: Feuerstein-Knollen in Kreide/Schotterfluren",
)
cal(
    "spawn",
    "dry_wood",
    "Totholz-Äste 0.5–6 kg im Wald",
    "Forstökologie: Totholzaufkommen in Wäldern",
)
cal(
    "spawn",
    "plant_fiber",
    "Gras-/Bastbündel 0.05–0.4 kg in Grasland und Sumpf",
    "Ethnobotanik: Sammelmengen Faserpflanzen",
)
cal(
    "spawn",
    "clay_moist",
    "Ufer-Lehm 0.5–5 kg in Sumpf und an Ufern (Nicht-Wasser-Zelle mit Wasser-Nachbar)",
    "Sedimentologie: Ton-/Lehmablagerungen an Gewässerrändern",
)
