"""Low-density Allee trap — two scattered fertile survivors must still pair.

System-level bug (5000-tick pilot): population booms to ~50 then crashes to 2
and stays pinned there; `births_cum` freezes because the two survivors never
co-locate to conceive. Root cause: in the physics_v2 + taxis path the move-head
is world-inert, so `innate_locomotion` is the ONLY locomotion — and it had no
search behaviour. A well-fed, reproductive agent that senses no mate within
``MATE_SEEK_RADIUS`` (8) simply *rests*. Two fertile survivors more than 8 cells
apart therefore freeze in place forever (an Allee trap): reproduction can never
restart even though both are alive and fertile.

These tests pin the fix: a reproductive agent with no mate in seek range now
drifts to a deterministic rendezvous point, so scattered survivors reliably
converge and produce a birth. Conception itself stays perception-/co-location-
bound (MATE_SEEK_RADIUS / the ±5 conception box in `_try_reproduce` are
unchanged) — no teleport-mating, no global conception radius, no RNG.
"""

from __future__ import annotations

import inspect
import random

import numpy as np

from artificial_society.agents.agent import Agent, PARTNER_SEEK_RADIUS


class FakeWorld:
    """Uniform, fully-passable, food-bearing map (only the taxis/repro surface)."""

    def __init__(self, width, height, food=10.0):
        self.width = width
        self.height = height
        self._food = food

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x, y):
        return {"food": self._food, "passable": True}


def _make_agent(x, y, sex, energy=130.0, age=200):
    a = Agent.spawn_random(x, y)
    a.sex = sex
    a.energy = energy
    a.age = age
    a.alive = True
    a.pending_spawn = False
    a.replication_cooldown = 0
    a.is_sleeping = False
    a.physics_v2 = True
    a.trust = {}
    a.traits["sense_radius"] = 3
    return a


def _cheby(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _run(sep, ticks=300, grid=(24, 18)):
    """Drive the v2 taxis/repro loop for two agents `sep` cells apart.

    Mirrors the update() ordering (move -> try_reproduce -> progress_pregnancy)
    without the full brain machinery, so the test is fast and deterministic.
    Returns (births, min_separation_reached).
    """
    Agent.taxis_enabled = True
    w, h = grid
    world = FakeWorld(w, h)
    f = _make_agent(1, h // 2, "f")
    m = _make_agent(1 + sep, h // 2, "m")
    agents = [f, m]
    births = 0
    min_sep = _cheby(f.pos, m.pos)
    for _ in range(ticks):
        for ag in agents:
            # per-tick neighbour snapshot invalidation (as update() does)
            ag._cached_nearby_agents = None
            ag._cached_nearby_radius = None
            ag.innate_locomotion(world, agents)
        min_sep = min(min_sep, _cheby(f.pos, m.pos))
        for ag in agents:
            ag._cached_nearby_agents = None
            ag._cached_nearby_radius = None
            ag._try_replicate(world, agents)
            if ag.progress_pending_spawn() is not None:
                births += 1
            if ag.replication_cooldown > 0:
                ag.replication_cooldown -= 1
    return births, min_sep


def test_close_pair_replicates_baseline():
    """Sanity: within seek range the existing homing already pairs (control)."""
    try:
        births, min_sep = _run(sep=4)
    finally:
        Agent.taxis_enabled = False
    # they converge into the conception box (homing pairs oscillate adjacent,
    # which is well inside the ±5 box — exact co-location is not required)
    assert min_sep <= 5, f"agents within seek range should meet, min_sep={min_sep}"
    assert births >= 1, "a close fertile pair must produce at least one birth"


def test_low_density_survivors_still_pair_and_replicate():
    """FAILS pre-fix: two fertile survivors 12 cells apart never converge.

    12 > MATE_SEEK_RADIUS (8): neither senses the other, so without a search
    drive both freeze and `births` stays 0 forever. The rendezvous fallback must
    bring them together and yield a birth.
    """
    assert 12 > PARTNER_SEEK_RADIUS  # precondition: beyond mutual sensing range
    try:
        births, min_sep = _run(sep=12)
    finally:
        Agent.taxis_enabled = False
    assert min_sep <= 5, f"scattered survivors must converge, min_sep={min_sep}"
    assert births >= 1, "low-density survivors must be able to restart reproduction"


def test_far_survivors_pair_across_full_grid():
    """Even near-maximal separation on the pilot grid must resolve."""
    try:
        births, min_sep = _run(sep=20)
    finally:
        Agent.taxis_enabled = False
    assert births >= 1, f"far survivors must still pair (min_sep={min_sep})"


def test_rendezvous_fallback_is_rng_free():
    """The new search fallback must not draw from any RNG stream (determinism)."""
    Agent.taxis_enabled = True
    world = FakeWorld(24, 18)
    a = _make_agent(1, 9, "f")
    b = _make_agent(20, 9, "m")  # far out of seek range -> fallback fires
    agents = [a, b]
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        for _ in range(5):
            for ag in agents:
                ag._cached_nearby_agents = None
                ag._cached_nearby_radius = None
                ag.innate_locomotion(world, agents)
    finally:
        Agent.taxis_enabled = False
    assert random.getstate() == py_state, "fallback must not draw python random"
    assert np.array_equal(np.random.get_state()[1], np_state[1]), "no numpy draw"


def test_locomotion_source_still_rng_free():
    """Whitebox: locomotion + its helpers reference no RNG / no sampled action."""
    src = "".join(
        inspect.getsource(m)
        for m in (
            Agent.innate_locomotion,
            Agent._nearest_food_cell,
            Agent._nearest_compatible_partner,
            Agent._partner_rendezvous_cell,
        )
    )
    for forbidden in ("random", "np.random", "brain_step", "action_list", "action_tensor"):
        assert forbidden not in src, f"locomotion must not reference `{forbidden}`"
