"""Struct-of-arrays cell storage with dict-compatible views (perf Tier 1).

The world's numeric per-cell state lives in one ``numpy`` ``float64`` array per
field (``World.F[field][y, x]``) plus one array per structure kind
(``World.S[kind]``); everything object-valued or dynamic (``biome``, ``herbs``,
``materials``, ``objects``, …) stays in a per-cell dict (``World.obj[y][x]``).

``CellView`` re-exposes one cell as the dict it used to be, so every existing
consumer (``cell["food"]``, ``cell.get("materials", {})``,
``cell["structures"]["farm"]``, ``world.cells[y][x]``) keeps working unchanged,
while grid-wide math (regrowth, diffusion, means) runs as vectorized array ops.

Values read through a view are plain Python ``float``/``int`` — exactly the
types the dict storage held — so downstream numerics and serialization are
unchanged.
"""

from __future__ import annotations

import numpy as np

# Numeric per-cell fields stored as (H, W) float64 arrays. Superset of
# environment/resources.initial_cell_state's float keys.
FLOAT_FIELDS: tuple[str, ...] = (
    "food",
    "plant_food",
    "meat_food",
    "water",
    "temperature",
    "danger",
    "soil_fertility",
    "pollution",
    "usage_pressure",
    "carrying_capacity",
    "spoilage",
    "carcasses",
    "disease",
    "moisture",
    "ash",
    "disturbance",
)
FLOAT_FIELD_SET = frozenset(FLOAT_FIELDS)

STRUCT_KEYS: tuple[str, ...] = ("camp", "farm", "well")


class StructView:
    """Dict-like view of one cell's structure levels (camp/farm/well)."""

    __slots__ = ("_world", "_x", "_y")

    def __init__(self, world, x: int, y: int):
        self._world = world
        self._x = x
        self._y = y

    def __getitem__(self, key: str) -> float:
        return float(self._world.S[key][self._y, self._x])

    def __setitem__(self, key: str, value) -> None:
        self._world.S[key][self._y, self._x] = value

    def get(self, key: str, default=0.0):
        arr = self._world.S.get(key)
        return float(arr[self._y, self._x]) if arr is not None else default

    def __contains__(self, key: str) -> bool:
        return key in self._world.S

    def keys(self):
        return self._world.S.keys()

    def values(self):
        return [float(self._world.S[k][self._y, self._x]) for k in self._world.S]

    def items(self):
        return [(k, float(self._world.S[k][self._y, self._x])) for k in self._world.S]

    def __repr__(self) -> str:  # debugging aid
        return f"StructView({dict(self.items())})"


class CellView:
    """Dict-compatible view of one cell backed by the world's field arrays."""

    __slots__ = ("_world", "x", "y")

    def __init__(self, world, x: int, y: int):
        self._world = world
        self.x = x
        self.y = y

    # -- mapping protocol ----------------------------------------------------
    def __getitem__(self, key: str):
        if key in FLOAT_FIELD_SET:
            return float(self._world.F[key][self.y, self.x])
        if key == "structures":
            return StructView(self._world, self.x, self.y)
        if key == "tick":
            return int(self._world.tick_grid[self.y, self.x])
        return self._world.obj[self.y][self.x][key]

    def __setitem__(self, key: str, value) -> None:
        if key in FLOAT_FIELD_SET:
            self._world.F[key][self.y, self.x] = value
        elif key == "tick":
            self._world.tick_grid[self.y, self.x] = value
        elif key == "structures":
            # Replacing the whole structures mapping (rare): copy values in.
            for k in STRUCT_KEYS:
                self._world.S[k][self.y, self.x] = value[k]
        else:
            self._world.obj[self.y][self.x][key] = value

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default):
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def pop(self, key: str, *default):
        """Remove a dynamic (object-dict) key. Array-backed fields are part of
        the world's permanent schema and cannot be removed — that would be a
        silent data-model change, so it raises instead."""
        if key in FLOAT_FIELD_SET or key in ("structures", "tick"):
            raise TypeError(f"cannot pop array-backed cell field {key!r}")
        return self._world.obj[self.y][self.x].pop(key, *default)

    def __delitem__(self, key: str) -> None:
        self.pop(key)

    def __contains__(self, key: str) -> bool:
        return (
            key in FLOAT_FIELD_SET
            or key in ("structures", "tick")
            or key in self._world.obj[self.y][self.x]
        )

    def keys(self):
        return (
            list(FLOAT_FIELDS) + ["structures", "tick"] + list(self._world.obj[self.y][self.x])
        )

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __repr__(self) -> str:  # debugging aid
        return f"CellView(x={self.x}, y={self.y})"


class CellRow:
    """One row of the grid; ``world.cells[y][x]`` compatibility."""

    __slots__ = ("_world", "_y")

    def __init__(self, world, y: int):
        self._world = world
        self._y = y

    def __getitem__(self, x: int) -> CellView:
        return CellView(self._world, x, self._y)

    def __len__(self) -> int:
        return self._world.width

    def __iter__(self):
        for x in range(self._world.width):
            yield CellView(self._world, x, self._y)


class CellGrid:
    """``world.cells`` compatibility wrapper over the field arrays."""

    __slots__ = ("_world",)

    def __init__(self, world):
        self._world = world

    def __getitem__(self, y: int) -> CellRow:
        return CellRow(self._world, y)

    def __len__(self) -> int:
        return self._world.height

    def __iter__(self):
        for y in range(self._world.height):
            yield CellRow(self._world, y)


def build_arrays(world, cell_dicts) -> None:
    """Populate ``world.F/S/obj/tick_grid`` from a grid of plain cell dicts
    (fresh ``initial_cell_state`` cells or a pre-array-storage checkpoint)."""
    h = world.height
    w = world.width
    world.F = {k: np.empty((h, w), dtype=np.float64) for k in FLOAT_FIELDS}
    world.S = {k: np.empty((h, w), dtype=np.float64) for k in STRUCT_KEYS}
    world.tick_grid = np.zeros((h, w), dtype=np.int64)
    world.obj = [[{} for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            cell = cell_dicts[y][x]
            for k in FLOAT_FIELDS:
                world.F[k][y, x] = cell.get(k, 0.0)
            structures = cell.get("structures", {})
            for k in STRUCT_KEYS:
                world.S[k][y, x] = structures.get(k, 0.0)
            world.tick_grid[y, x] = cell.get("tick", 0)
            extra = {
                k: v
                for k, v in cell.items()
                if k not in FLOAT_FIELD_SET and k not in ("structures", "tick")
            }
            world.obj[y][x] = extra
