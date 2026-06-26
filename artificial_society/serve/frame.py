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

from collections.abc import Iterable
from typing import Any

from artificial_society.environment.biomes import BIOME_BASE_COLOR

# Stable index order so the client can cache the legend sent in the hello frame.
_BIOME_NAMES: list[str] = sorted(BIOME_BASE_COLOR)
_BIOME_INDEX: dict[str, int] = {name: i for i, name in enumerate(_BIOME_NAMES)}

# Agent life-stage -> small int (mirrors Agent.life_stage()).
_STAGE_IDX: dict[str, int] = {"child": 0, "adult": 1, "elder": 2}


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
        ``tick``  : int
        ``grid``  : {w, h}
        ``cells`` : {food[w*h], water[w*h], biome[w*h]} — row-major, quantized ints
        ``agents``: [{id, x, y, e, st, tribe, col}]
        ``events``: [{kind, x, y, r, i}]
        ``stats`` : the aggregate dict the cards already use (``sim.stats.last``)
    """
    world = sim.world
    w, h = world.width, world.height
    cells = world.cells
    biomes = world.biomes

    food: list[int] = []
    water: list[int] = []
    biome_idx: list[int] = []
    for y in range(h):
        row_c = cells[y]
        row_b = biomes[y]
        for x in range(w):
            c = row_c[x]
            food.append(round(c["food"]))
            water.append(round(c["water"]))
            biome_idx.append(_BIOME_INDEX.get(row_b[x], 0))

    agents = [
        {
            "id": a.id,
            "x": a.x,
            "y": a.y,
            "e": round(a.energy),
            "st": _STAGE_IDX.get(a.life_stage(), 1),
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
        "cells": {"food": food, "water": water, "biome": biome_idx},
        "agents": agents,
        "events": events,
        "stats": dict(getattr(sim.stats, "last", {})),
    }
