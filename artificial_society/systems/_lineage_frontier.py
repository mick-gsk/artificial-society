"""Transmitted, tribe-level functional frontier — the social ratchet for Path-A.

Keyed by tribe id so every tribe member shares one monotone 'best realized
functional value' object; an invention's reward credit is the *marginal* gain it
adds over that shared frontier (guided variation, Boyd & Richerson 1985). Module
-level + run-resettable to match the DISCOVERY_REGISTRY singleton lifecycle.
"""
from __future__ import annotations

_FRONTIER: dict[int, float] = {}


def reset_frontiers() -> None:
    _FRONTIER.clear()


def frontier_value(tribe_id: int) -> float:
    return _FRONTIER.get(int(tribe_id), 0.0)


def update_frontier(tribe_id: int, value: float) -> float:
    tid = int(tribe_id)
    prev = _FRONTIER.get(tid, 0.0)
    marginal = max(0.0, float(value) - prev)
    if value > prev:
        _FRONTIER[tid] = float(value)
    return marginal
