"""Lane test: logic-gate detection is generalised beyond the four hardcoded types.

Phase 5 de-scripting (Task 3). The old ``detect_gates`` only ever emitted AND/OR via a
neighbour-count ``if/elif``, so a switch node with exactly two conductive links and no
switch-neighbour (a simple controlled connection) was invisible, and NOT/LATCH were never
produced. After de-scripting, detection is topology-derived: any switch node with >= 2
conductive links is a gate of arbitrary input arity, and its output is computed from the
physical switch state rather than a per-named-type truth table.
"""

from artificial_society.systems.logic_gates import detect_gates, evaluate_gate


class _FakeWorld:
    """Minimal grid: cells are plain dicts with a 'materials' slot."""

    def __init__(self, w=5, h=5):
        self.width = w
        self.height = h
        self.cells = [[{"materials": {}} for _ in range(w)] for _ in range(h)]

    def put(self, x, y, materials, **extra):
        cell = {"materials": dict(materials)}
        cell.update(extra)
        self.cells[y][x] = cell


def _switch_with_two_links(pressure=0.0):
    w = _FakeWorld()
    # Switch node (stone) flanked by two conductive cells (charcoal). The old detector
    # recognises neither AND (needs a switch-neighbour) nor OR (needs 3 conductive
    # neighbours), so it sees nothing here.
    w.put(2, 2, {"stone": 1.0}, pressure=pressure)
    w.put(1, 2, {"charcoal": 1.0})
    w.put(3, 2, {"charcoal": 1.0})
    return w


def test_two_link_switch_node_is_detected_as_a_gate():
    gates = detect_gates(_switch_with_two_links())
    assert len(gates) >= 1, "generic detector failed to recognise a 2-link switch gate"
    g = gates[0]
    assert (2, 2) in g.switch_cells
    # Both conductive neighbours are accounted for as input/output — none dropped.
    linked = set(g.input_cells) | set(g.output_cells)
    assert {(1, 2), (3, 2)} <= linked


def test_gate_output_follows_physical_switch_state():
    # Pressure > 0.6 switches the stone node on -> signal passes -> output active.
    w_on = _switch_with_two_links(pressure=0.9)
    assert evaluate_gate(detect_gates(w_on)[0], w_on) is True

    # No pressure / heat -> stone does not conduct -> output inactive.
    w_off = _switch_with_two_links(pressure=0.0)
    assert evaluate_gate(detect_gates(w_off)[0], w_off) is False
