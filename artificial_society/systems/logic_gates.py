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


def detect_gates(world) -> list[Gate]:
    """
    Analysiert das Grid auf Gate-Topologien.
    Einfache heuristische Erkennung:

    AND:  Switch S1 und S2 beide auf demselben Pfad zum Ausgang
    OR:   Zwei separate Pfade die zu demselben Ausgang fuehren
    NOT:  Switch mit inverser Charakteristik (low conductivity bei Waerme)
    LATCH: AND(in, NOT(reset)) Kombination
    """
    gates   = []
    height  = world.height
    width   = world.width
    gate_ctr = getattr(world, '_gate_counter', 0)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Nur Switch-Knoten als Ausgangspunkt
            if not _is_switch_cell(world, x, y):
                continue

            neighbours = [(x+dx, y+dy) for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]]
            conductive_nb = [
                (nx, ny) for nx, ny in neighbours
                if 0 <= nx < width and 0 <= ny < height
                and _cell_conductivity(world.cells[ny][nx]) >= CONDUCTIVITY_THRESHOLD
            ]
            switch_nb = [
                (nx, ny) for nx, ny in neighbours
                if 0 <= nx < width and 0 <= ny < height
                and _is_switch_cell(world, nx, ny)
            ]

            if len(conductive_nb) < 2:
                continue  # Kein Gate moeglich

            # --- AND: zwei Switches auf einem Pfad ---
            if len(switch_nb) >= 1 and len(conductive_nb) >= 2:
                input_cells  = conductive_nb[:1]
                output_cells = conductive_nb[1:2]
                gate = Gate(
                    gate_id      = f'gate_{gate_ctr:04d}',
                    gate_type    = 'AND',
                    input_cells  = input_cells,
                    output_cells = output_cells,
                    switch_cells = [(x, y)] + switch_nb[:1],
                )
                gates.append(gate)
                gate_ctr += 1

            # --- OR: zwei Eingaenge, ein Ausgang ---
            elif len(conductive_nb) >= 3:
                gate = Gate(
                    gate_id      = f'gate_{gate_ctr:04d}',
                    gate_type    = 'OR',
                    input_cells  = conductive_nb[:2],
                    output_cells = conductive_nb[2:3],
                    switch_cells = [(x, y)],
                )
                gates.append(gate)
                gate_ctr += 1

    world._gate_counter = gate_ctr
    return gates


# ---------------------------------------------------------------------------
# Gate-Zustand berechnen
# ---------------------------------------------------------------------------
def evaluate_gate(gate: Gate, world) -> bool:
    """Berechnet den Ausgabe-Zustand eines Gates."""
    def cell_active(cx, cy) -> bool:
        return _cell_conductivity(world.cells[cy][cx]) >= CONDUCTIVITY_THRESHOLD

    if gate.gate_type == 'AND':
        return all(_switch_is_on(world, sx, sy) for sx, sy in gate.switch_cells)

    elif gate.gate_type == 'OR':
        return any(cell_active(ix, iy) for ix, iy in gate.input_cells)

    elif gate.gate_type == 'NOT':
        if not gate.input_cells:
            return False
        ix, iy = gate.input_cells[0]
        return not _switch_is_on(world, ix, iy)

    elif gate.gate_type == 'LATCH':
        # Einfaches SR-Latch: Set dominiert
        if len(gate.input_cells) >= 2:
            set_in   = cell_active(*gate.input_cells[0])
            reset_in = cell_active(*gate.input_cells[1])
            if set_in:
                gate.state = True
            elif reset_in:
                gate.state = False
        return gate.state

    elif gate.gate_type == 'WIRE':
        if not gate.input_cells:
            return False
        return cell_active(*gate.input_cells[0])

    return False


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
