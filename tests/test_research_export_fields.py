"""TDD test: research export entries must carry real discovered_by and tick.

After the fix in materials.apply_interaction, the `discovered_by` and `tick`
fields on discovery entries should reflect the actual inventing agent, not the
hardcoded -1/0 defaults that apply_interaction previously always used.

One seed × 200 ticks via run_learned is enough to produce discoveries.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def test_export_has_real_discovered_by_and_tick():
    """At least some entries in a learned run must have discovered_by >= 0 and tick > 0."""
    from artificial_society.research.run_single import run_learned

    _meta, _series, entries = run_learned(seed=42, ticks=200, grid_w=20, grid_h=15, pop=8)

    # There should be at least some discoveries
    assert len(entries) > 0, "Expected at least one discovery entry in 200 ticks"

    # Before fix: ALL entries had discovered_by == -1 and tick == 0.
    # After fix: at least some should have a real agent id (>= 0) and real tick (> 0).
    has_real_discoverer = any(int(e.get("discovered_by", -1)) >= 0 for e in entries)
    has_real_tick = any(int(e.get("tick", 0)) > 0 for e in entries)

    assert has_real_discoverer, (
        f"All {len(entries)} entries have discovered_by == -1; "
        "apply_interaction must pass real agent id to DISCOVERY_REGISTRY.register"
    )
    assert has_real_tick, (
        f"All {len(entries)} entries have tick == 0; "
        "apply_interaction must pass real tick to DISCOVERY_REGISTRY.register"
    )


def test_export_entry_has_adopters_field():
    """Every entry in a learned-run export must include an 'adopters' field (list)."""
    from artificial_society.research.export import entry_to_json

    fake_entry = {
        "id": "mat_0001",
        "recipe": ("combine", "wood", "stone"),
        "vector": [0.1] * 12,
        "discovered_by": 3,
        "tick": 17,
        "uses": 0,
    }
    result = entry_to_json(fake_entry)
    assert "adopters" in result, "entry_to_json must include 'adopters' key"
    assert result["adopters"] == [], "adopters should be an empty list for now"
    assert result["discovered_by"] == 3
    assert result["tick"] == 17
