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
        ``agents``    : [{id, x, y, e, hp, st, act, tribe, col}]
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
        "agents": agents,
        "events": events,
        "stats": dict(getattr(sim.stats, "last", {})),
    }
