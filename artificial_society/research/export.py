"""Serialisation for a single run (one arm × one seed).

A run export is a JSON object with three parts:

* ``meta``    — run parameters (arm, seed, ticks/attempts, grid, population, …).
* ``series``  — uncapped per-tick (learned) / per-checkpoint (recombiner) snapshots
  ``{tick|attempt, n_attempts, n_discoveries, pop, energy}``. Deliberately *not*
  routed through ``StatisticsTracker`` (which caps at 200 ticks).
* ``entries`` — the full, append-only discovery lineage: every ``mat_XXXX`` with
  its ``recipe = (action, mat_a, mat_b)`` and 12-dim property vector. Because the
  registry is append-only, ``entries[:k]`` reconstructs the state after ``k``
  discoveries, so over-time curves need no extra storage.
"""

from __future__ import annotations

import json
from typing import Any


def entry_to_json(e: dict) -> dict:
    """Convert one ``DiscoveryRegistry`` entry to a JSON-safe dict."""
    recipe = e.get("recipe")
    if recipe is not None:
        recipe = [recipe[0], recipe[1], recipe[2] if len(recipe) > 2 else None]
    return {
        "id": e["id"],
        "recipe": recipe,
        "vector": [round(float(x), 6) for x in e["vector"]],
        "discovered_by": int(e.get("discovered_by", -1)),
        "tick": int(e.get("tick", 0)),
        "uses": int(e.get("uses", 0)),
    }


def dump_run(path: str, meta: dict, series: list[dict], entries: list[dict]) -> None:
    """Write a run export to ``path`` as JSON."""
    data: dict[str, Any] = {
        "meta": meta,
        "series": series,
        "entries": [entry_to_json(e) for e in entries],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_run(path: str) -> dict:
    """Load a run export written by :func:`dump_run`."""
    with open(path) as f:
        return json.load(f)
