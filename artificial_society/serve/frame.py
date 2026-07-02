"""Serialize the spatial simulation state into a compact, JSON-friendly frame.

The dashboard's WebSocket layer (see :mod:`artificial_society.serve.app`) pushes one
of these per render tick so the browser can draw the live world — agent positions, a
biome/resource grid and active disturbances — rather than only aggregate numbers.

``build_frame`` is **read-only**: it never draws RNG, mutates state, or touches the
hot-file class sources beyond reading public attributes. That keeps the determinism
contract intact and lets the runner build a frame off its own thread safely (the same
thread that owns ``sim.step``).

Scalars are quantized to small ints so a 60x40 grid (~2400 cells) stays in the
single-digit-KB range per frame at ~20 Hz.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from typing import Any

import numpy as np

from artificial_society.environment.biomes import BIOME_BASE_COLOR
from artificial_society.environment.materials import IDX, get_vector

# Biomes are fixed after world generation — cache the flattened index list per
# world instead of rebuilding a w*h Python loop for every ~20 Hz frame.
_BIOME_IDX_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Stable index order so the client can cache the legend sent in the hello frame.
_BIOME_NAMES: list[str] = sorted(BIOME_BASE_COLOR)
_BIOME_INDEX: dict[str, int] = {name: i for i, name in enumerate(_BIOME_NAMES)}

# Agent life-stage -> small int (mirrors Agent.life_stage()).
_STAGE_IDX: dict[str, int] = {"child": 0, "adult": 1, "elder": 2}

# Agent action mode -> small int (mirrors Agent.last_action_mode; 5 = sleeping,
# which overrides the mode because a sleeping agent executes no actions).
_ACT_IDX: dict[str, int] = {"idle": 0, "forage": 1, "cooperate": 2, "attack": 3, "build": 4}
ACT_SLEEP = 5

# --- ground materials (Physik v2 / Embodiment) --------------------------------
#
# Cells carry a sparse ``materials`` dict (id -> mass). For the picture world we
# reduce each id to a small visual class; the client owns one sprite per class.
# Classes double as a priority order (highest first) when a cell holds several
# materials or the frame item cap forces a cut — fire, fresh flakes and
# discovered materials must always make it onto the wire.
ITEM_FIRE = 11
ITEM_SHARD = 10  # sharpness above _SHARP_MIN — knapped flakes / blades
ITEM_WONDER = 9  # discovered (registry) material without an edge
ITEM_FLINT = 8
ITEM_BONE = 7
ITEM_MEAT = 6
ITEM_CLAY = 5
ITEM_STONE = 4
ITEM_WOOD = 3
ITEM_FIBER = 2
ITEM_BERRY = 1

_SHARP_MIN = 0.45
_ITEM_QTY_MIN = 0.15  # ignore trace amounts (growth seeds, mined-out slots)
_ITEM_CAP = 800  # per frame, priority-ranked
_ITEM_REFRESH_TICKS = 10  # ground materials change slowly; rescan every N ticks

# Raw materials blanket most of the map; drawn 1:1 they bury the landscape and
# the special finds. Common classes (flint deposits included — they cover whole
# mountainsides) only ship from richer slots and only for a deterministic ~1/3
# of cells — enough to say "here lies wood", calm enough to keep the rare
# classes (flakes, discoveries, fire) visible at a glance.
_THINNED_CLASSES = frozenset(
    (ITEM_BERRY, ITEM_FIBER, ITEM_WOOD, ITEM_STONE, ITEM_CLAY, ITEM_FLINT)
)
_COMMON_QTY_MIN = 0.5
_COMMON_KEEP_MOD = 3  # keep cells whose spatial hash % mod == 0

_NAMED_ITEM_CLASS: dict[str, int] = {
    "fire": ITEM_FIRE,
    "ember": ITEM_FIRE,
    "sharp_stone": ITEM_SHARD,
    "flint": ITEM_FLINT,
    "carcass": ITEM_BONE,
    "bone": ITEM_BONE,
    "raw_meat": ITEM_MEAT,
    "cooked_meat": ITEM_MEAT,
    "clay_moist": ITEM_CLAY,
    "clay": ITEM_CLAY,
    "stone": ITEM_STONE,
    "granite": ITEM_STONE,
    "dry_wood": ITEM_WOOD,
    "wet_wood": ITEM_WOOD,
    "wood": ITEM_WOOD,
    "plant_fiber": ITEM_FIBER,
    "fiber": ITEM_FIBER,
    "dry_grass": ITEM_FIBER,
    "berries": ITEM_BERRY,
}

# id -> visual class, resolved once per material id (registry lookups allocate).
_MAT_CLASS_CACHE: dict[str, int] = {}

# world -> (tick_bucket, items) — reuse the scan between refreshes.
_ITEMS_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _item_class(mat_id: str) -> int:
    """Visual class for a material id; 0 = not worth drawing."""
    cls = _MAT_CLASS_CACHE.get(mat_id)
    if cls is None:
        cls = _NAMED_ITEM_CLASS.get(mat_id, 0)
        if cls == 0:
            try:
                vec = get_vector(mat_id)
                sharp = float(vec[IDX["sharpness"]]) if vec is not None else 0.0
            except Exception:
                sharp = 0.0
            if sharp >= _SHARP_MIN:
                cls = ITEM_SHARD
            elif mat_id.startswith(("mat_", "pmat_")):
                cls = ITEM_WONDER
        _MAT_CLASS_CACHE[mat_id] = cls
    return cls


def _ground_items(world, tick: int) -> list[int]:
    """Flat ``[k0, x0, y0, k1, x1, y1, …]`` of the most visible material per cell.

    Priority-ranked before the cap so fire/flakes/discoveries always ship; ties
    within a class break on a spatial hash so a cap cut thins the whole map
    uniformly instead of chopping off the bottom rows.
    """
    bucket = tick // _ITEM_REFRESH_TICKS
    cached = _ITEMS_CACHE.get(world)
    if cached is not None and cached[0] == bucket:
        return cached[1]

    found: list[tuple[int, int, int, int]] = []  # (-class, hash, x, y)
    for y, row in enumerate(world.obj):
        for x, cell in enumerate(row):
            slot = cell.get("materials")
            if not slot:
                continue
            cell_hash = (x * 73856093 ^ y * 19349663) & 0xFFFF
            best = 0
            for mat_id, qty in slot.items():
                if qty < _ITEM_QTY_MIN:
                    continue
                cls = _item_class(mat_id)
                if cls <= best:
                    continue
                if cls in _THINNED_CLASSES and (
                    qty < _COMMON_QTY_MIN or cell_hash % _COMMON_KEEP_MOD
                ):
                    continue
                best = cls
            if best:
                found.append((-best, cell_hash, x, y))

    found.sort()
    items: list[int] = []
    for negcls, _, x, y in found[:_ITEM_CAP]:
        items.extend((-negcls, x, y))
    _ITEMS_CACHE[world] = (bucket, items)
    return items


def _tool_class(agent) -> int:
    """0 = empty hands, 1 = blunt tool, 2 = sharp blade."""
    tool = getattr(agent, "tool", None)
    if not tool:
        return 0
    return 2 if _item_class(tool) == ITEM_SHARD else 1


def biome_legend() -> list[dict[str, Any]]:
    """Stable ``name -> {idx, rgb}`` legend; sent once in the WS hello frame."""
    return [
        {"name": name, "idx": i, "rgb": list(BIOME_BASE_COLOR[name])}
        for i, name in enumerate(_BIOME_NAMES)
    ]


def _hex(rgb: Iterable[float]) -> str:
    """``(r, g, b)`` 0..255 -> ``"#rrggbb"`` (clamped, rounded)."""
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_frame(sim) -> dict[str, Any]:
    """Read-only snapshot of the spatial state for one render tick.

    Layout (all JSON-native):
        ``tick``      : int
        ``grid``      : {w, h}
        ``daylight``  : float 0..1 (world.day_state light level)
        ``cells``     : {food[w*h], water[w*h], biome[w*h]} — row-major, quantized ints
        ``structures``: [{x, y, k}] sparse; k ∈ {camp, farm, well}
        ``items``     : flat [k, x, y, …] sparse ground materials; k = ITEM_* class
        ``agents``    : [{id, x, y, e, hp, st, act, tl, cg, tribe, col}]
        ``events``    : [{kind, x, y, r, i}]
        ``stats``     : the aggregate dict the cards already use (``sim.stats.last``)
    """
    world = sim.world
    w, h = world.width, world.height
    biomes = world.biomes

    # Straight off the struct-of-arrays field storage (perf Tier 1): two
    # vectorized ops instead of 2*w*h per-cell view reads at ~20 Hz.
    food = np.rint(world.F["food"]).astype(int).ravel().tolist()
    water = np.rint(world.F["water"]).astype(int).ravel().tolist()
    biome_idx = _BIOME_IDX_CACHE.get(world)
    if biome_idx is None:
        biome_idx = [
            _BIOME_INDEX.get(biomes[y][x], 0) for y in range(h) for x in range(w)
        ]
        _BIOME_IDX_CACHE[world] = biome_idx

    # Structures are rare — ship them sparse straight off the S arrays.
    structures = []
    for kind in ("camp", "farm", "well"):
        arr = world.S.get(kind)
        if arr is not None:
            for sy, sx in np.argwhere(arr != 0.0):
                structures.append({"x": int(sx), "y": int(sy), "k": kind})

    agents = [
        {
            "id": a.id,
            "x": a.x,
            "y": a.y,
            "e": round(a.energy),
            "hp": round(a.health),
            "st": _STAGE_IDX.get(a.life_stage(), 1),
            "act": ACT_SLEEP
            if getattr(a, "is_sleeping", False)
            else _ACT_IDX.get(getattr(a, "last_action_mode", "idle"), 0),
            "tl": _tool_class(a),
            "cg": min(9, int(sum(getattr(a, "material_inventory", {}).values()))),
            "tribe": a.tribe_id,
            "col": _hex(a.display_color()),
        }
        for a in sim.agents
    ]

    events = [
        {
            "kind": e.get("kind", "?"),
            "x": e.get("x", 0),
            "y": e.get("y", 0),
            "r": e.get("radius", 0),
            "i": round(float(e.get("intensity", 0.0)), 2),
        }
        for e in getattr(world, "active_events", [])
    ]

    return {
        "type": "frame",
        "tick": sim.tick,
        "grid": {"w": w, "h": h},
        "daylight": round(float(getattr(world, "day_state", {}).get("light", 1.0)), 3),
        "cells": {"food": food, "water": water, "biome": biome_idx},
        "structures": structures,
        "items": _ground_items(world, sim.tick),
        "agents": agents,
        "events": events,
        "stats": dict(getattr(sim.stats, "last", {})),
    }
