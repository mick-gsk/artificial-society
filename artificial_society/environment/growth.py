"""
Biological Growth System
--------------------------
Materialien mit growth_rate > 0 replizieren sich unter geeigneten
Umweltbedingungen (Licht, Feuchtigkeit, Wärme, Boden).

Kein Hardcode von 'Mais' oder 'Weizen'. Stattdessen:
  - Seed-Vektoren haben eine growth_rate-Eigenschaft (13. Dimension)
  - Unter günstigen Bedingungen entsteht neues Material in Nachbarzellen
  - Jahreszeiten (seasons.py) modulieren die Wachstumsrate
  - Agenten können Samen sammeln und gezielt pflanzen → Landwirtschaft

Aus dem Wachstum-System emergiert:
  - Nahrungssicherung ohne Jagd
  - Territoriales Verhalten um Felder
  - Handel mit Saatgut
  - Selektion: Agenten pflanzen die "besten" Körner (emergente Domestizierung)
"""

import random
import numpy as np
from typing import Optional

from artificial_society.environment.materials import (
    IDX, N_PROPS, PROP_DIMS, MATERIALS, DISCOVERY_REGISTRY, _v
)

# ---------------------------------------------------------------------------
# 13. Dimension: growth_rate
# Wir erweitern das Materialsystem um eine Wachstums-Dimension.
# Bestehende Materialien haben growth_rate=0 (keine Änderung).
# ---------------------------------------------------------------------------
GROWTH_IDX = N_PROPS  # Index 12 — angehängt
N_PROPS_EXT = N_PROPS + 1  # 13 Dimensionen

GROWTH_SEEDS: dict[str, np.ndarray] = {}

def _gv(base_mat: str, growth_rate: float) -> np.ndarray:
    """Erstellt einen Growable-Vektor: base material + growth_rate Dimension."""
    base = MATERIALS.get(base_mat, np.zeros(N_PROPS, dtype=np.float32))
    ext  = np.zeros(N_PROPS_EXT, dtype=np.float32)
    ext[:N_PROPS] = base
    ext[GROWTH_IDX] = float(growth_rate)
    return ext


# Seed-Materialien (wachstumsfähig)
GROWTH_SEEDS = {
    'seed_grain': _gv('raw_root',   growth_rate=0.6),   # Getreidekorn
    'seed_herb':  _gv('crushed_herb', growth_rate=0.5), # Kräutersamen
    'seed_fiber': _gv('fiber',      growth_rate=0.4),   # Faserpflanzensamen
    'spore':      _gv('dry_grass',  growth_rate=0.3),   # Pilzspore / Grasspore
    'root_cut':   _gv('raw_root',   growth_rate=0.45),  # Abgetrenntes Rhizom
}

# Ins MATERIALS-Dict eintragen (als normal-dimensionale Vektoren, ohne growth)
for seed_name, ext_vec in GROWTH_SEEDS.items():
    MATERIALS[seed_name] = ext_vec[:N_PROPS].copy()


# ---------------------------------------------------------------------------
# Wachstumsbedingungen
# ---------------------------------------------------------------------------
def _growth_factor(vec: np.ndarray, env: dict, season: str = 'summer') -> float:
    """
    Berechnet wie stark ein Samen in dieser Umgebung wächst.
    Gibt einen Faktor [0..1] zurück.
    """
    moisture    = env.get('moisture', 0.5)       # 0..1
    light       = env.get('light', 0.5)          # 0..1
    temperature = env.get('temperature', 20)     # Celsius
    soil_type   = env.get('soil', 'loam')        # loam, sand, rock, water

    # Temperatur-Optimum: 10-30°C
    temp_factor = max(0.0, 1.0 - abs(temperature - 20) / 20.0)

    # Feuchtigkeit: linear 0.2..0.8 optimal
    moist_factor = min(1.0, max(0.0, (moisture - 0.15) / 0.65))

    # Licht: photosynthese
    light_factor = min(1.0, light * 1.4)

    # Boden
    soil_factor = {'loam': 1.0, 'sand': 0.5, 'rock': 0.1, 'clay': 0.7,
                   'water': 0.0, 'ash': 0.8}.get(soil_type, 0.6)

    # Jahreszeit
    season_factor = {'spring': 1.2, 'summer': 1.0, 'autumn': 0.6,
                     'winter': 0.05}.get(season, 1.0)

    return float(
        temp_factor * 0.25 +
        moist_factor * 0.30 +
        light_factor * 0.25 +
        soil_factor  * 0.10 +
        (season_factor - 1.0) * 0.10  # saisonal moduliert
    )


# ---------------------------------------------------------------------------
# Wachstums-Tick
# ---------------------------------------------------------------------------
def tick_growth(world, season: str = 'summer') -> list[dict]:
    """
    Verarbeitet biologisches Wachstum für alle Zellen.
    Gibt Liste von Growth-Events zurück.
    """
    events = []
    height, width = world.height, world.width

    for y in range(height):
        for x in range(width):
            cell   = world.cells[y][x]
            slot   = cell.get('materials', {})
            biome  = world.biomes[y][x] if hasattr(world, 'biomes') else 'grassland'

            if biome == 'water':
                continue

            env = {
                'moisture':    cell.get('moisture',    50) / 100.0,
                'light':       cell.get('light',       0.5),
                'temperature': cell.get('temperature', 20),
                'soil':        _biome_to_soil(biome),
            }

            for mat_id, qty in list(slot.items()):
                if qty < 0.05:
                    continue

                # Hole den aktuellen Vektor (ggf. aus Discovery Registry)
                ext_vec = GROWTH_SEEDS.get(mat_id)
                if ext_vec is None:
                    # Prüfe ob mat_id ein growable discovered material ist
                    base_vec = _get_growth_vector(mat_id)
                    if base_vec is None:
                        continue
                    growth_rate = float(base_vec[GROWTH_IDX]) if len(base_vec) > N_PROPS else 0.0
                else:
                    growth_rate = float(ext_vec[GROWTH_IDX])

                if growth_rate < 0.05:
                    continue

                factor = _growth_factor(ext_vec[:N_PROPS] if ext_vec is not None
                                        else np.zeros(N_PROPS), env, season)
                effective_growth = growth_rate * factor

                if effective_growth < 0.01:
                    continue

                # In-place growth: mehr vom gleichen Material
                if random.random() < effective_growth * 0.3:
                    slot[mat_id] = min(5.0, qty + effective_growth * 0.5)
                    events.append({'type': 'growth', 'x': x, 'y': y,
                                   'mat': mat_id, 'delta': effective_growth * 0.5})

                # Seed dispersal: Samen streuen in Nachbarzellen
                if qty > 0.3 and random.random() < effective_growth * 0.15:
                    nx, ny = _random_neighbour(x, y, width, height)
                    ncell  = world.cells[ny][nx]
                    nslot  = ncell.get('materials', {})
                    if nslot.get(mat_id, 0.0) < 0.1:  # noch nicht vorhanden
                        nslot[mat_id] = 0.1
                        ncell['materials'] = nslot
                        events.append({'type': 'seed_dispersal', 'from': (x, y),
                                       'to': (nx, ny), 'mat': mat_id})

    return events


# ---------------------------------------------------------------------------
# Agent pflanzt Samen (emergente Landwirtschaft)
# ---------------------------------------------------------------------------
def agent_plant_seed(
    agent,
    seed_mat: str,
    cell: dict,
    env: dict,
) -> float:
    """
    Agent platziert einen Samen bewusst in eine Zelle.
    Reward proportional zur erwarteten Wachstumsrate.
    Dieser Mechanismus ist der Kern für emergente Landwirtschaft:
      Agenten lernen, dass bestimmte Samen an bestimmten Orten wachsen.
    """
    inv = getattr(agent, 'material_inventory', {})
    if inv.get(seed_mat, 0.0) < 0.1:
        return 0.0  # Kein Samen im Inventar

    ext_vec     = GROWTH_SEEDS.get(seed_mat)
    if ext_vec is None:
        return 0.0
    growth_rate = float(ext_vec[GROWTH_IDX])

    season = getattr(agent, 'world', None)
    season = season.season if season else 'summer'

    factor = _growth_factor(ext_vec[:N_PROPS], env, season)

    # Samen aus Inventar entnehmen
    inv[seed_mat] = max(0.0, inv[seed_mat] - 0.1)

    # In Zelle pflanzen
    slot = cell.get('materials', {})
    slot[seed_mat] = slot.get(seed_mat, 0.0) + 0.1
    cell['materials'] = slot

    # Belohnung: proportional zur Wachstumserwartung
    reward = growth_rate * factor * 1.5
    return float(reward)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _biome_to_soil(biome: str) -> str:
    return {
        'forest':    'loam',
        'grassland': 'loam',
        'mountain':  'rock',
        'desert':    'sand',
        'swamp':     'clay',
        'tundra':    'sand',
    }.get(biome, 'loam')


def _random_neighbour(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    dx, dy = random.choice([(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)])
    return max(0, min(w-1, x+dx)), max(0, min(h-1, y+dy))


def _get_growth_vector(mat_id: str) -> Optional[np.ndarray]:
    """Gibt extended vector (N_PROPS+1) zurück wenn mat growable ist."""
    if mat_id in GROWTH_SEEDS:
        return GROWTH_SEEDS[mat_id]
    # Discovered materials: check if close to a seed
    vec = DISCOVERY_REGISTRY.get_vector(mat_id)
    if vec is not None and float(vec.sum()) > 0:
        # Kleine Partikel (masse < 0.2, edibility > 0) könnten Samen sein
        if float(vec[IDX['mass']]) < 0.2 and float(vec[IDX['edibility']]) > 0.1:
            ext = np.zeros(N_PROPS_EXT, dtype=np.float32)
            ext[:N_PROPS] = vec
            ext[GROWTH_IDX] = 0.2  # langsames Wachstum für unbekannte Samen
            return ext
    return None
