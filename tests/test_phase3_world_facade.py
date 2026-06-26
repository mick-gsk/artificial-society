"""Phase 3: World exposes a small authoritative cell-mutation API.

External systems should route cell writes through World instead of reaching into
`world.cells[y][x]` directly. The façade must be behaviour-identical to the in-place
writes it replaces: `set_cell` is a plain assignment, `adjust_cell` adds deltas with a
zero default (matching the `cell.get(key, 0.0) + delta` idiom).
"""

from __future__ import annotations

import pathlib
import re

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


# Single-key direct write to a `cell` / `ncell` local, e.g. `cell["food"] = ...`
# or `ncell['materials'] += ...`. Nested sub-dict writes (`cell["structures"]["camp"]`)
# and variable-key writes (`cell[key] = ...`) are out of scope — the façade mediates
# named single-field cell state.
_DIRECT_CELL_WRITE = re.compile(r"""\b(?:cell|ncell)\[['"][a-z_]+['"]\]\s*(?:[-+*/]=|=(?!=))""")


def test_facade_completeness_no_direct_cell_writes_outside_world():
    """SSOT regression guard: every module except world.py routes cell mutation through
    the World façade. A new direct `cell["x"] = ...` anywhere else re-fragments the state
    ownership Phase 3 consolidated — this test fails until it goes through set_cell/adjust_cell."""
    pkg = pathlib.Path(__file__).resolve().parent.parent / "artificial_society"
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        if path.name == "world.py":  # World is the authoritative owner
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if _DIRECT_CELL_WRITE.search(code):
                offenders.append(f"{path.relative_to(pkg.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "Direct cell writes bypass the World façade:\n" + "\n".join(offenders)
