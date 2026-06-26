"""Phase 3: World exposes a small authoritative cell-mutation API.

External systems should route cell writes through World instead of reaching into
`world.cells[y][x]` directly. The façade must be behaviour-identical to the in-place
writes it replaces: `set_cell` is a plain assignment, `adjust_cell` adds deltas with a
zero default (matching the `cell.get(key, 0.0) + delta` idiom).
"""

from __future__ import annotations

from artificial_society.world import World


def test_set_cell_assigns_value():
    w = World(8, 6)
    w.set_cell(2, 3, "warmth", 0.7)
    assert w.get_cell(2, 3)["warmth"] == 0.7


def test_adjust_cell_adds_delta_accumulating():
    w = World(8, 6)
    w.set_cell(1, 1, "conducted_heat", 0.0)
    w.adjust_cell(1, 1, conducted_heat=0.5)
    assert w.get_cell(1, 1)["conducted_heat"] == 0.5
    w.adjust_cell(1, 1, conducted_heat=0.25)
    assert w.get_cell(1, 1)["conducted_heat"] == 0.75


def test_adjust_cell_missing_key_defaults_to_zero():
    w = World(8, 6)
    cell = w.get_cell(3, 2)
    cell.pop("conducted_light", None)
    w.adjust_cell(3, 2, conducted_light=0.4)
    assert w.get_cell(3, 2)["conducted_light"] == 0.4


def test_adjust_cell_multiple_keys_at_once():
    w = World(8, 6)
    w.set_cell(4, 2, "conducted_heat", 0.0)
    w.set_cell(4, 2, "conducted_light", 0.0)
    w.adjust_cell(4, 2, conducted_heat=0.3, conducted_light=0.6)
    cell = w.get_cell(4, 2)
    assert cell["conducted_heat"] == 0.3
    assert cell["conducted_light"] == 0.6
