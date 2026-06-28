"""
Emergente Logikgatter
-----------------------
Logikgatter entstehen NICHT durch Hardcode.
Sie entstehen wenn Switch-Materialien (conductivity.is_switch_material)
in einer bestimmten Topologie angeordnet werden:

  AND-Gate:  Zwei Eingangsketten muessen BEIDE leitfaehig sein
             um die Ausgangskette zu aktivieren.
             Physikalisch: zwei serielle Switches vor dem Ausgang.

  OR-Gate:   Eine von zwei Eingangsketten reicht.
             Physikalisch: zwei parallele Pfade zum Ausgang.

  NOT-Gate:  Ausgang ist aktiv WENN Eingang NICHT leitet.
             Physikalisch: Switch-Material das bei Waerme
             seine Leitfaehigkeit VERLIERT (inverse Charakteristik).

  LATCH:     AND(set_input, NOT(reset_input)) -- Speicher.
             Einmal gesetzt bleibt der Zustand erhalten.
             Emergenter Informationsspeicher.

Die Agenten wissen nicht dass sie Logikgatter bauen.
Sie legen Kohle und Steine. Die Gate-Engine analysiert die Topologie
und loest Events aus. Der Brain lernt: bestimmte Anordnungen
kontrollieren wo Waerme/Licht hinfliesst.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from artificial_society.environment.materials import IDX, get_vector
from artificial_society.systems.conductivity import (
    _cell_conductivity, CONDUCTIVITY_THRESHOLD,
    is_switch_material, compute_switch_state,
)


# ---------------------------------------------------------------------------
# Gate-Typen
# ---------------------------------------------------------------------------
GATE_TYPES = ('AND', 'OR', 'NOT', 'LATCH', 'WIRE')  # WIRE = einfache Verbindung


@dataclass
class Gate:
    gate_id:    str
    gate_type:  str
    input_cells:  list[tuple]   # (x,y) Eingangs-Zellen
    output_cells: list[tuple]   # (x,y) Ausgangs-Zellen
    switch_cells: list[tuple]   # (x,y) Switch-Knoten
    state:        bool  = False  # Aktueller Ausgabe-Zustand
    ticks_active: int   = 0
    ticks_total:  int   = 0

    def duty_cycle(self) -> float:
        """Wie oft war das Gate aktiv? Basis fuer Nuetzlichkeits-Bewertung."""
        return self.ticks_active / max(1, self.ticks_total)


# ---------------------------------------------------------------------------
# Topologie-Analyse
# ---------------------------------------------------------------------------
def _get_cell_heat(world, x: int, y: int) -> float:
    try:
        cell = world.cells[y][x]
        from artificial_society.environment.materials import material_heat
        return material_heat(cell.get('materials', {}))
    except (IndexError, KeyError):
        return 0.0


def _is_switch_cell(world, x: int, y: int) -> bool:
    """Prueft ob eine Zelle Switch-Materialien enthaelt."""
    try:
        cell = world.cells[y][x]
        for mat_id, qty in cell.get('materials', {}).items():
            if qty < 0.1:
                continue
            vec = get_vector(mat_id)
            if is_switch_material(vec):
                return True
    except (IndexError, KeyError):
        pass
    return False


def _switch_is_on(world, x: int, y: int) -> bool:
    """Gibt zurueck ob Switch-Zelle aktuell leitet."""
    try:
        cell  = world.cells[y][x]
        heat  = _get_cell_heat(world, x, y)
        pres  = cell.get('pressure', 0.0)
        for mat_id, qty in cell.get('materials', {}).items():
            if qty < 0.1:
                continue
            vec = get_vector(mat_id)
            if is_switch_material(vec):
                return compute_switch_state(vec, heat, pres)
    except (IndexError, KeyError):
        pass
    return False


def _derive_gate_type(input_cells: list, switch_cells: list) -> str:
    """Abgeleitetes, rein beschreibendes Label (kein Verhaltens-Schalter).

    Phase 5: Der Typ folgt der erkannten Topologie statt eine feste Klassen-
    zugehoerigkeit zu erzwingen. Mehrere serielle Switches -> AND-artig; mehrere
    Eingaenge auf einen Ausgang -> OR-artig; ein einzelner gesteuerter Leiter -> WIRE.
    """
    if len(switch_cells) >= 2:
        return 'AND'
    if len(input_cells) >= 2:
        return 'OR'
    return 'WIRE'


def detect_gates(world) -> list[Gate]:
    """
    Generische Topologie-Erkennung (Phase 5 de-scripting).

    Frueher emittierte detect_gates nur AND/OR ueber ein Nachbarn-Zaehl-if/elif;
    NOT/LATCH wurden nie erzeugt und ein simpler Switch mit genau zwei leitfaehigen
    Anschluessen war unsichtbar. Jetzt ist JEDER Switch-Knoten mit >=2 leitfaehigen
    Nachbarn ein Gate beliebiger Eingangs-Arity; das Verhalten wird in evaluate_gate
    aus der physikalischen Switch-Leitung abgeleitet, gate_type ist nur noch ein Label.
    """
    gates = []
    height = world.height
    width = world.width
    gate_ctr = getattr(world, '_gate_counter', 0)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Nur Switch-Knoten als Ausgangspunkt
            if not _is_switch_cell(world, x, y):
                continue

            neighbours = [(x + dx, y + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
            in_bounds = [
                (nx, ny) for nx, ny in neighbours
                if 0 <= nx < width and 0 <= ny < height
            ]
            conductive_nb = [
                (nx, ny) for nx, ny in in_bounds
                if _cell_conductivity(world.cells[ny][nx]) >= CONDUCTIVITY_THRESHOLD
            ]
            if len(conductive_nb) < 2:
                continue  # ohne >=2 leitfaehige Anschluesse kein Gate

            switch_nb = [
                (nx, ny) for nx, ny in in_bounds
                if _is_switch_cell(world, nx, ny)
            ]

            # Letzter leitfaehiger Anschluss = Ausgang, der Rest = Eingaenge
            # (deterministische Reihenfolge -> reproduzierbar). Beliebige Arity.
            output_cells = conductive_nb[-1:]
            input_cells = conductive_nb[:-1]
            switch_cells = [(x, y)] + switch_nb

            gates.append(Gate(
                gate_id=f'gate_{gate_ctr:04d}',
                gate_type=_derive_gate_type(input_cells, switch_cells),
                input_cells=input_cells,
                output_cells=output_cells,
                switch_cells=switch_cells,
            ))
            gate_ctr += 1

    world._gate_counter = gate_ctr
    return gates


# ---------------------------------------------------------------------------
# Gate-Zustand berechnen
# ---------------------------------------------------------------------------
def evaluate_gate(gate: Gate, world) -> bool:
    """
    Generische, physikalisch abgeleitete Auswertung (Phase 5 de-scripting).

    Der Ausgang ist aktiv, wenn der Switch-Pfad leitet (ALLE Switch-Knoten des Gates
    leiten gerade) UND mindestens ein Eingang aktiv ist. Das gilt fuer beliebige
    Eingangs-Arity und subsumiert die frueheren AND/OR/NOT-Faelle physikalisch (NOT
    entsteht aus invertierenden Switches via compute_switch_state). Ein als LATCH
    gelabeltes Gate behaelt zusaetzlich seinen Zustand (Speicher).
    """
    def cell_active(cx, cy) -> bool:
        try:
            return _cell_conductivity(world.cells[cy][cx]) >= CONDUCTIVITY_THRESHOLD
        except (IndexError, KeyError):
            return False

    path_conducts = (
        all(_switch_is_on(world, sx, sy) for sx, sy in gate.switch_cells)
        if gate.switch_cells
        else True
    )
    driven = any(cell_active(ix, iy) for ix, iy in gate.input_cells)
    active = path_conducts and driven

    if gate.gate_type == 'LATCH':
        # Speicher: Set dominiert; expliziter Reset (2. Eingang inaktiv) loescht.
        if active:
            gate.state = True
        elif len(gate.input_cells) >= 2 and not cell_active(*gate.input_cells[1]):
            gate.state = False
        return gate.state

    return active


# ---------------------------------------------------------------------------
# Gate Network
# ---------------------------------------------------------------------------
class GateNetwork:
    """
    Verwaltet alle erkannten Gates der Welt.
    Wird pro Tick aktualisiert.
    """
    def __init__(self):
        self.gates:  dict[str, Gate] = {}
        self.events: list[dict]      = []

    def tick(self, world) -> list[dict]:
        tick_events = []

        # Neue Gates erkennen (alle N Ticks -- teuer)
        tick = getattr(world, 'tick', 0)
        if tick % 10 == 0:
            new_gates = detect_gates(world)
            for g in new_gates:
                if g.gate_id not in self.gates:
                    self.gates[g.gate_id] = g
                    print(f'[GATE] New {g.gate_type}-Gate detected: {g.gate_id} '
                          f'at switch={g.switch_cells[:1]}')

        # Bestehende Gates auswerten
        for gate_id, gate in list(self.gates.items()):
            gate.ticks_total += 1
            try:
                new_state = evaluate_gate(gate, world)
            except (IndexError, KeyError):
                del self.gates[gate_id]
                continue

            if new_state != gate.state:
                event = {
                    'type':      'GATE_STATE_CHANGE',
                    'gate_id':   gate_id,
                    'gate_type': gate.gate_type,
                    'new_state': new_state,
                    'outputs':   gate.output_cells,
                }
                tick_events.append(event)
                self.events.append(event)

                # Ausgabe-Zellen aktivieren/deaktivieren
                if new_state:
                    for ox, oy in gate.output_cells:
                        try:
                            world.cells[oy][ox]['gate_signal'] = 1.0
                        except (IndexError, KeyError):
                            pass
                else:
                    for ox, oy in gate.output_cells:
                        try:
                            world.cells[oy][ox]['gate_signal'] = 0.0
                        except (IndexError, KeyError):
                            pass

            gate.state = new_state
            if new_state:
                gate.ticks_active += 1

        return tick_events

    def summary(self) -> dict:
        return {
            'total_gates': len(self.gates),
            'by_type': {
                t: sum(1 for g in self.gates.values() if g.gate_type == t)
                for t in GATE_TYPES
            },
            'active': sum(1 for g in self.gates.values() if g.state),
        }


GATE_NETWORK = GateNetwork()
