"""
Fermentation & Biological Decay System
----------------------------------------
Organische Materialien (edibility > 0, dryness < 0.6) driften
über Zeit durch biologische Prozesse:

  Stage 0: frisch
  Stage 1: fermentiert  → Alkohol-analog (toxicity↑, scent↑, edibility±)
  Stage 2: gereift      → Essig/Käse-analog (toxicity stabilisiert, scent↑↑)
  Stage 3: verfault     → Totalverlust (toxicity↑↑, edibility→0)

Die Geschwindigkeit hängt ab von:
  - Feuchtigkeit der Umgebung (moisture)
  - Temperatur (heat_emission des Containers / env.temperature)
  - Versiegelung (in einem sealed CompositeObject → langsamer)
  - Dryness des Materials selbst

Keine Rezepte. Kein 'make_beer()'. Die Agenten entdecken, dass
bestimmte Bedingungen bestimmte Effekte erzeugen — durch Reward.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from artificial_society.environment.materials import (
    IDX, N_PROPS, DISCOVERY_REGISTRY, get_vector
)


# ---------------------------------------------------------------------------
# Fermentation-Zustand eines Material-Lots
# ---------------------------------------------------------------------------
@dataclass
class FermentationState:
    mat_id:       str
    base_vector:  np.ndarray      # ursprünglicher Vektor
    current_vec:  np.ndarray      # aktueller (gedrifteter) Vektor
    ticks_stored: int  = 0
    stage:        int  = 0        # 0=fresh 1=fermenting 2=aged 3=rotten
    moisture_acc: float = 0.0     # akkumulierte Feuchtigkeit
    heat_acc:     float = 0.0     # akkumulierte Wärme
    sealed:       bool  = False   # in versiegeltem Behälter?


# ---------------------------------------------------------------------------
# Fermentations-Raten (Ticks bis Stage-Wechsel bei optimalen Bedingungen)
# ---------------------------------------------------------------------------
STAGE_TICKS = {
    0: 80,    # frisch → fermentierend
    1: 160,   # fermentierend → gereift
     2: 300,   # gereift → verfault (wenn unversiegelt)
}


def _fermentation_rate(state: FermentationState, env: dict) -> float:
    """
    Rate [0..1] mit der Fermentation voranschreitet.
    Hohe Feuchtigkeit + moderate Wärme = schnell.
    Versiegelt = langsam (kontrollierte Fermentation → Qualität).
    """
    moisture = env.get('moisture', 0.5)
    temp     = env.get('temperature', 20)
    dryness  = float(state.current_vec[IDX['dryness']])

    # Optimale Fermentationstemperatur 15-35°C
    temp_factor = 1.0 - abs(temp - 25) / 30.0
    temp_factor = max(0.0, min(1.0, temp_factor))

    rate = moisture * 0.5 + (1.0 - dryness) * 0.3 + temp_factor * 0.2

    # Versiegelt: viel langsamer aber nicht null (anaerobe Fermentation)
    if state.sealed:
        rate *= 0.25

    return float(np.clip(rate, 0.0, 1.0))


def _apply_stage_drift(state: FermentationState, stage: int) -> np.ndarray:
    """
    Verändert den Materialvektor entsprechend dem Fermentationsstadium.
    Rein physikalisch — keine String-Lookups.
    """
    vec = state.base_vector.copy()
    edib = float(vec[IDX['edibility']])
    tox  = float(vec[IDX['toxicity']])
    scent = float(vec[IDX['scent']])

    if stage == 1:  # Fermentierend: Alkohol entsteht
        vec[IDX['toxicity']]  = min(1.0, tox + 0.2)
        vec[IDX['scent']]     = min(1.0, scent + 0.35)
        vec[IDX['edibility']] = max(0.0, edib - 0.1)   # leicht reduziert
        vec[IDX['dryness']]   = max(0.0, float(vec[IDX['dryness']]) - 0.15)
        # CO2-Emission analog: leichte Wärmeabgabe
        vec[IDX['heat_emission']] = min(0.15, float(vec[IDX['heat_emission']]) + 0.05)

    elif stage == 2:  # Gereift: Essig/Käse-analog
        vec[IDX['toxicity']]  = min(0.4, tox + 0.1)   # stabilisiert (Säure schützt)
        vec[IDX['scent']]     = min(1.0, scent + 0.55) # intensiver Geruch
        vec[IDX['edibility']] = min(1.0, edib + 0.05)  # manchmal besser (Käse)
        vec[IDX['solubility']] = min(1.0, float(vec[IDX['solubility']]) + 0.2)  # säurehaltig
        vec[IDX['dryness']]   = max(0.0, float(vec[IDX['dryness']]) - 0.25)

    elif stage == 3:  # Verfault
        vec[IDX['toxicity']]  = min(1.0, tox + 0.6)
        vec[IDX['edibility']] = 0.0
        vec[IDX['scent']]     = min(1.0, scent + 0.3)  # Verwesungsgeruch
        vec[IDX['mass']]     *= 0.6                     # Masse schwindet
        vec[IDX['dryness']]   = max(0.0, float(vec[IDX['dryness']]) - 0.4)

    return np.clip(vec, 0.0, 1.0)


def tick_fermentation(state: FermentationState, env: dict) -> FermentationState:
    """
    Einen Tick Fermentation berechnen.
    Gibt aktualisierten FermentationState zurück.
    """
    if not _is_fermentable(state.current_vec):
        return state

    rate = _fermentation_rate(state, env)
    state.ticks_stored += 1
    state.moisture_acc  = state.moisture_acc * 0.95 + env.get('moisture', 0.5) * 0.05
    state.heat_acc      = state.heat_acc     * 0.95 + env.get('temperature', 20) / 40.0 * 0.05

    threshold = STAGE_TICKS.get(state.stage, 9999)
    # Beschleunigt durch akkumulierte Bedingungen
    effective_ticks = state.ticks_stored * rate

    new_stage = state.stage
    if effective_ticks > threshold and state.stage < 3:
        new_stage = state.stage + 1
        # Registriere neues Material wenn Stage-Wechsel
        new_vec = _apply_stage_drift(state, new_stage)
        state.current_vec = new_vec
        state.stage       = new_stage
        # Gib dem neuen Fermentationsprodukt eine ID in der Discovery Registry
        DISCOVERY_REGISTRY.register(
            new_vec,
            discoverer_id = -2,  # -2 = natürlicher Prozess
            tick          = state.ticks_stored,
            recipe        = (f'ferment_stage{new_stage}', state.mat_id, None),
        )

    return state


def _is_fermentable(vec: np.ndarray) -> bool:
    """Nur organische, nicht-völlig-trockene Materialien fermentieren."""
    return (
        float(vec[IDX['edibility']]) > 0.05
        and float(vec[IDX['dryness']]) < 0.85
    )


# ---------------------------------------------------------------------------
# World-level Fermentation Manager
# ---------------------------------------------------------------------------
class FermentationManager:
    """
    Verwaltet alle aktiven Fermentationsprozesse in der Welt.
    Wird von invention.py pro Tick aufgerufen.
    """
    def __init__(self):
        # key: (cell_x, cell_y, mat_id) → FermentationState
        self.states: dict[tuple, FermentationState] = {}

    def register_material(self, x: int, y: int, mat_id: str, vec: np.ndarray,
                          sealed: bool = False):
        key = (x, y, mat_id)
        if key not in self.states:
            self.states[key] = FermentationState(
                mat_id      = mat_id,
                base_vector = vec.copy(),
                current_vec = vec.copy(),
                sealed      = sealed,
            )

    def tick(self, world) -> list[dict]:
        """Tick alle aktiven Fermentationsprozesse. Gibt Events zurück."""
        events = []
        to_remove = []
        for (x, y, mat_id), state in self.states.items():
            try:
                cell = world.cells[y][x]
                env  = {
                    'moisture':    cell.get('moisture', 50) / 100.0,
                    'temperature': cell.get('temperature', 20),
                }
                old_stage = state.stage
                state     = tick_fermentation(state, env)
                self.states[(x, y, mat_id)] = state

                if state.stage != old_stage:
                    events.append({
                        'type':     'fermentation_stage_change',
                        'x': x, 'y': y,
                        'mat_id':   mat_id,
                        'stage':    state.stage,
                        'new_vec':  state.current_vec,
                    })

                # Entfernen wenn völlig verfault und keine Masse mehr
                if state.stage == 3 and float(state.current_vec[IDX['mass']]) < 0.05:
                    to_remove.append((x, y, mat_id))
            except (IndexError, KeyError):
                to_remove.append((x, y, mat_id))

        for key in to_remove:
            del self.states[key]

        return events

    def get_current_vector(self, x: int, y: int, mat_id: str) -> Optional[np.ndarray]:
        state = self.states.get((x, y, mat_id))
        return state.current_vec.copy() if state else None


FERMENTATION_MANAGER = FermentationManager()
