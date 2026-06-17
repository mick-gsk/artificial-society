"""
Conductivity & Current Flow System
-------------------------------------
Wenn Materialien mit hoher Leitfaehigkeit (conductivity > 0.5) eine
zusammenhaengende Kette zwischen zwei Punkten bilden, fliesst Energie.

Das ist die physikalische Grundlage fuer:
  - Waerme-Uebertragung (Topf auf Feuer)
  - Licht-Uebertragung (gluehende Holzkohle-Kette)
  - (Phase C) Logik-Gatter wenn Schalter-Materialien hinzukommen

Keine Elektronik explizit hardcodiert.
Nur: conductivity-Ketten werden analysiert, Events werden ausgeloest.

Event-Typen:
  _current_flows     -- Leitfaehige Kette geschlossen
  _heat_transfer     -- Waerme fliesst entlang Kette
  _light_propagates  -- Licht wird weitergeleitet
  _chain_broken      -- Kette unterbrochen
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from artificial_society.environment.materials import (
    IDX, get_vector, MATERIALS
)


CONDUCTIVITY_THRESHOLD = 0.45   # Minimum fuer Leiter
HEAT_TRANSFER_RATE     = 0.15   # Pro Tick und Link
LIGHT_TRANSFER_RATE    = 0.10


# ---------------------------------------------------------------------------
# Conductive node: eine Zelle mit leitfaehigem Material
# ---------------------------------------------------------------------------
@dataclass
class ConductiveNode:
    x: int
    y: int
    conductivity: float
    heat:         float
    light:        float
    mat_ids:      list   # Materialien die zur Leitfaehigkeit beitragen


def _cell_conductivity(cell: dict) -> float:
    """Berechnet effektive Leitfaehigkeit einer Zelle."""
    slot = cell.get('materials', {})
    total = 0.0
    for mat_id, qty in slot.items():
        vec = get_vector(mat_id)
        total += float(vec[IDX['conductivity']]) * min(1.0, qty)
    # Objekte koennen auch leitfaehig sein
    for obj in cell.get('objects', []):
        vecs = getattr(obj, 'vectors', [])
        for v in vecs:
            total += float(v[IDX['conductivity']]) * 0.3
    return min(1.0, total)


def _cell_heat(cell: dict) -> float:
    from artificial_society.environment.materials import material_heat
    return material_heat(cell.get('materials', {}))


def _cell_light(cell: dict) -> float:
    from artificial_society.environment.materials import material_light
    return material_light(cell.get('materials', {}))


# ---------------------------------------------------------------------------
# BFS: Finde leitfaehige Ketten im Grid
# ---------------------------------------------------------------------------
def find_conductive_chains(world, min_length: int = 2) -> list[list[tuple]]:
    """
    Findet alle zusammenhaengenden leitfaehigen Zellketten.
    Gibt Liste von Ketten zurueck, jede Kette ist eine Liste von (x,y).
    """
    visited  = set()
    chains   = []
    height, width = world.height, world.width

    # Alle leitfaehigen Zellen bestimmen
    conductive = set()
    for y in range(height):
        for x in range(width):
            cell = world.cells[y][x]
            if _cell_conductivity(cell) >= CONDUCTIVITY_THRESHOLD:
                conductive.add((x, y))

    # BFS fuer zusammenhaengende Komponenten
    for start in conductive:
        if start in visited:
            continue
        chain   = []
        queue   = [start]
        visited.add(start)
        while queue:
            cx, cy = queue.pop(0)
            chain.append((cx, cy))
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = cx+dx, cy+dy
                if (nx, ny) in conductive and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        if len(chain) >= min_length:
            chains.append(chain)

    return chains


# ---------------------------------------------------------------------------
# Energie-Transfer entlang einer Kette
# ---------------------------------------------------------------------------
def propagate_along_chain(
    world,
    chain: list[tuple],
) -> list[dict]:
    """
    Propagiert Waerme und Licht entlang einer leitfaehigen Kette.
    Quell-Zelle (hoechste Waerme/Licht) speist die Kette.
    Gibt Events zurueck.
    """
    events = []
    if not chain:
        return events

    # Finde Quell-Zelle (max heat oder light)
    source_heat  = 0.0
    source_light = 0.0
    for (x, y) in chain:
        cell = world.cells[y][x]
        h    = _cell_heat(cell)
        l    = _cell_light(cell)
        if h > source_heat:  source_heat  = h
        if l > source_light: source_light = l

    if source_heat < 0.05 and source_light < 0.05:
        return events

    # Propagation: exponentiell abnehmend entlang Kette
    for i, (x, y) in enumerate(chain):
        cell = world.cells[y][x]
        cond = _cell_conductivity(cell)
        decay = cond ** (i * 0.3 + 1)

        transferred_heat  = source_heat  * decay * HEAT_TRANSFER_RATE
        transferred_light = source_light * decay * LIGHT_TRANSFER_RATE

        if transferred_heat > 0.01:
            cell['conducted_heat']  = cell.get('conducted_heat', 0.0) + transferred_heat
            events.append({
                'type': '_heat_transfer',
                'x': x, 'y': y,
                'value': round(transferred_heat, 3),
                'chain_pos': i,
            })
        if transferred_light > 0.01:
            cell['conducted_light'] = cell.get('conducted_light', 0.0) + transferred_light
            events.append({
                'type': '_light_propagates',
                'x': x, 'y': y,
                'value': round(transferred_light, 3),
                'chain_pos': i,
            })

    # Kette-Closed-Event wenn Anfang und Ende unterschiedlich sind
    if len(chain) >= 3:
        events.append({
            'type':        '_current_flows',
            'chain_len':   len(chain),
            'source_heat': round(source_heat, 3),
            'chain':       chain[:3],  # nur erste 3 Punkte loggen
        })

    return events


# ---------------------------------------------------------------------------
# Switch-Material: Phase-C Vorbereitung
# ---------------------------------------------------------------------------
def is_switch_material(vec: np.ndarray) -> bool:
    """
    Ein Material ist ein 'Schalter' wenn:
    - Mittlere Leitfaehigkeit (0.3..0.6)
    - Hohe Haerte (stabil)
    - Durch Druck/Waerme seine Leitfaehigkeit aendert
    Basis fuer spaetere Logikgatter-Emergenz.
    """
    cond = float(vec[IDX['conductivity']])
    hard = float(vec[IDX['hardness']])
    return 0.25 < cond < 0.65 and hard > 0.5


def compute_switch_state(
    vec: np.ndarray, heat: float, pressure: float
) -> bool:
    """
    Bestimmt ob ein Switch-Material aktuell leitet.
    heat > 0.5 oder pressure > 0.6 schaltet es durch.
    Emergenter Transistor-Analog.
    """
    cond = float(vec[IDX['conductivity']])
    if heat > 0.5:
        return cond + 0.3 > CONDUCTIVITY_THRESHOLD
    if pressure > 0.6:
        return cond + 0.2 > CONDUCTIVITY_THRESHOLD
    return cond > CONDUCTIVITY_THRESHOLD


# ---------------------------------------------------------------------------
# Tick-Level Integration
# ---------------------------------------------------------------------------
def tick_conductivity(world) -> list[dict]:
    """
    Tick: Finde alle leitfaehigen Ketten, propagiere Energie.
    Gibt alle Events dieser Tick-Periode zurueck.
    """
    all_events = []
    chains     = find_conductive_chains(world, min_length=2)
    for chain in chains:
        events = propagate_along_chain(world, chain)
        all_events.extend(events)
    return all_events
