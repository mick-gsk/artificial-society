"""Audit group 8 (follow-up): disturbance events must have real, physically
coherent effects on the world *fields* — not just on agents.

``test_event_dynamics.py`` locks event genesis, lifecycle and the direct agent
consequences. This file locks the other half: the application of active events
to the environment fields in
``environment/resources.py::regrow_grid`` (the event-application block) and the
spatial field model in ``world.event_fields_grid``.

Coverage:
- A. direct field effects   (storm→+moisture/+water, fire→−plant/+ash, drought,
                              blight)
- B. spatial profile        (linear falloff, hard radius boundary, stacking)
- C. feedback couplings      (ash→fertility/regrowth, moisture→regrowth,
                              disturbance→danger/regrowth, diffusion)
- D. invariants             (clipping, no RNG consumption)
- E. known gaps as xfail    (fire does not yet damage structures/objects)

Effects are isolated by comparing an event world against an identical
event-free world (same seed → bit-identical construction), so the ordinary
regrowth background cancels and only the event contribution remains. Tests
drive ``regrow_grid`` directly with a hand-built ``active_events`` list, which
exercises exactly the application layer without genesis noise and consumes no
RNG (determinism-safe: no ``Simulation.step`` and no golden-trajectory impact).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from artificial_society.environment.resources import regrow_grid
from artificial_society.rng import seed_all
from artificial_society.world import World

# Neutral drivers: no ambient rain/wind/storm, so anything that moves a field
# is attributable to the injected event (or to a field we set ourselves).
SEASON = {"food_factor": 1.0, "temperature_shift": 0.0}
CALM = {"storm_risk": 0.0, "rain_map": 0.0, "temperature_shift": 0.0, "wind": 0.0}

_DEFAULTS = {
    "moisture": 50.0,
    "plant_food": 60.0,
    "meat_food": 40.0,
    "water": 50.0,
    "ash": 10.0,
    "pollution": 20.0,
    "disease": 10.0,
    "soil_fertility": 40.0,
    "disturbance": 0.0,
    "carrying_capacity": 90.0,
}


def _world(**overrides: float) -> World:
    """Deterministic 20×15 world with uniform, mid-range fields.

    Mid-range values keep both the event world and its baseline away from the
    clip boundaries, so a per-cell delta equals the raw event coefficient.
    """
    seed_all(1234)
    w = World(20, 15)
    # Uniform forest: every cell is land with positive regrowth, so a field
    # coupling can never be masked by a biome that happens to zero plant growth
    # (water) or leave no regrowth headroom. Re-initing the biome statics draws
    # no RNG, so determinism is unaffected.
    w.biomes = [["forest"] * w.width for _ in range(w.height)]
    w._init_biome_statics()
    fields = {**_DEFAULTS, **overrides}
    for key, value in fields.items():
        w.F[key][:] = value
    return w


def _interior_land(w: World) -> tuple[int, int]:
    """First interior land cell (deterministic). Interior so an r≤3 event stays
    in bounds; land so plant/meat regrowth is non-zero (water zeroes them)."""
    for y in range(3, w.height - 3):
        for x in range(3, w.width - 3):
            if w.get_biome(x, y) != "water":
                return x, y
    raise AssertionError("no interior land cell in the test world")


def _event(kind: str, x: int, y: int, *, radius: int = 3, intensity: float = 1.0, **extra):
    e = {"kind": kind, "x": x, "y": y, "radius": radius, "intensity": intensity, "ttl": 100}
    e.update(extra)
    return e


def _regrow(w: World, tick: int = 5) -> None:
    regrow_grid(w, SEASON, CALM, tick, w.event_fields_grid())


def _event_delta(kind: str, field: str, *, world_kwargs=None, **event_kwargs) -> float:
    """(field at event centre with the event) − (same cell without it)."""
    world_kwargs = world_kwargs or {}
    base = _world(**world_kwargs)
    target = _world(**world_kwargs)
    cx, cy = _interior_land(target)
    target.active_events.append(_event(kind, cx, cy, **event_kwargs))
    _regrow(base)
    _regrow(target)
    return float(target.F[field][cy, cx] - base.F[field][cy, cx])


# ---------------------------------------------------------------------------
# A. Direct field effects
# ---------------------------------------------------------------------------
def test_storm_raises_moisture_and_water():
    """Rain: a storm wets the ground and washes ash down."""
    assert _event_delta("storm", "moisture") == pytest.approx(8.0, abs=0.5)
    assert _event_delta("storm", "water") == pytest.approx(0.45, abs=0.1)
    assert _event_delta("storm", "ash") < 0.0  # rain suppresses ash


def test_fire_burns_plants_and_creates_ash():
    """Fire destroys vegetation/meat, dries the cell, and produces ash + smoke."""
    assert _event_delta("fire", "plant_food") == pytest.approx(-5.8, abs=0.6)
    assert _event_delta("fire", "meat_food") == pytest.approx(-1.8, abs=0.5)
    assert _event_delta("fire", "ash") == pytest.approx(7.0, abs=0.5)
    assert _event_delta("fire", "pollution") == pytest.approx(4.0, abs=0.5)
    assert _event_delta("fire", "moisture") < -10.0  # strong drying


def test_drought_dries_soil_and_withers_plants():
    assert _event_delta("drought", "moisture") == pytest.approx(-12.0, abs=0.6)
    assert _event_delta("drought", "water") < 0.0
    assert _event_delta("drought", "plant_food") == pytest.approx(-2.4, abs=0.5)


def test_blight_eats_plants_and_raises_fault():
    assert _event_delta("blight", "plant_food") == pytest.approx(-3.6, abs=0.5)
    assert _event_delta("blight", "disease") == pytest.approx(4.0, abs=0.5)
    assert _event_delta("blight", "pollution") > 0.0


def test_every_event_raises_the_disturbance_field():
    """The generic ``disturbance`` channel aggregates every active event, so any
    disturbance (here a fire) raises ``F['disturbance']``."""
    w = _world()
    cx, cy = _interior_land(w)
    w.active_events.append(_event("fire", cx, cy))
    fields = w.event_fields_grid()
    assert fields["fire"][cy, cx] == pytest.approx(1.0)
    assert fields["disturbance"][cy, cx] == pytest.approx(1.0)
    assert _event_delta("fire", "disturbance") == pytest.approx(9.0, abs=0.6)


# ---------------------------------------------------------------------------
# B. Spatial profile
# ---------------------------------------------------------------------------
def test_strength_is_linear_falloff_to_radius():
    w = _world()
    cx, cy = _interior_land(w)
    w.active_events.append(_event("fire", cx, cy, radius=4, intensity=1.0))
    fire = w.event_fields_grid()["fire"]
    assert fire[cy, cx] == pytest.approx(1.0)  # centre = intensity
    assert fire[cy, cx + 2] == pytest.approx(0.5, abs=1e-9)  # d=2, r=4 → 1−2/4
    assert fire[cy, cx + 4] == pytest.approx(0.0, abs=1e-9)  # d=r → 0


def test_no_effect_beyond_radius():
    w = _world()
    cx, cy = _interior_land(w)
    w.active_events.append(_event("fire", cx, cy, radius=3, intensity=1.0))
    fire = w.event_fields_grid()["fire"]
    assert fire[cy, cx + 5] == 0.0  # strictly zero outside the radius


def test_overlapping_events_stack_additively():
    w = _world()
    cx, cy = _interior_land(w)
    a = _event("fire", cx, cy, radius=4)
    b = _event("fire", cx + 2, cy, radius=4)

    w.active_events = [a]
    only_a = w.event_fields_grid()["fire"][cy, cx + 1]
    w.active_events = [b]
    only_b = w.event_fields_grid()["fire"][cy, cx + 1]
    w.active_events = [a, b]
    both = w.event_fields_grid()["fire"][cy, cx + 1]

    assert both == pytest.approx(only_a + only_b)


# ---------------------------------------------------------------------------
# C. Feedback couplings
# ---------------------------------------------------------------------------
def test_fire_leaves_persistent_ash_next_tick():
    """Ash from a fire outlives the fire itself: after the event is gone, the
    cell still carries elevated ash the following tick (fuel for the coupling
    tests below)."""
    base = _world()
    target = _world()
    cx, cy = _interior_land(target)
    target.active_events.append(_event("fire", cx, cy))

    _regrow(base, tick=5)
    _regrow(target, tick=5)
    target.active_events.clear()  # fire has burned out
    _regrow(base, tick=6)
    _regrow(target, tick=6)

    assert target.F["ash"][cy, cx] > base.F["ash"][cy, cx]


def test_ash_boosts_fertility_and_suppresses_regrowth():
    """The ash coupling read at the start of a tick: more ash → richer soil
    (+0.04·ash) but less immediate plant growth (−0.02·ash)."""
    hi = _world(ash=80.0, plant_food=15.0)
    lo = _world(ash=0.0, plant_food=15.0)
    cx, cy = _interior_land(hi)

    _regrow(hi)
    _regrow(lo)

    assert hi.F["soil_fertility"][cy, cx] > lo.F["soil_fertility"][cy, cx]
    assert hi.F["plant_food"][cy, cx] < lo.F["plant_food"][cy, cx]


def test_low_moisture_suppresses_regrowth():
    """The mechanism a drought triggers: dry soil → lower moisture_factor →
    less plant regrowth in the same tick."""
    dry = _world(moisture=10.0, plant_food=15.0)
    wet = _world(moisture=90.0, plant_food=15.0)
    cx, cy = _interior_land(dry)

    _regrow(dry)
    _regrow(wet)

    assert dry.F["plant_food"][cy, cx] < wet.F["plant_food"][cy, cx]


def test_drought_lowers_future_regrowth():
    """End-to-end: a drought tick leaves the cell with less plant_food the next
    tick than an undisturbed cell (dries + withers)."""
    base = _world(plant_food=15.0)
    target = _world(plant_food=15.0)
    cx, cy = _interior_land(target)
    target.active_events.append(_event("drought", cx, cy))

    _regrow(base, tick=5)
    _regrow(target, tick=5)
    target.active_events.clear()
    _regrow(base, tick=6)
    _regrow(target, tick=6)

    assert target.F["plant_food"][cy, cx] < base.F["plant_food"][cy, cx]


def test_disturbance_raises_danger_same_tick():
    """Events raise the disturbance field, which feeds the danger field the same
    tick (+0.18·disturbance)."""
    assert _event_delta("fire", "danger") > 0.0


def test_high_disturbance_suppresses_regrowth():
    """The disturbance→stress_factor coupling: a disturbed cell regrows less."""
    hi = _world(disturbance=80.0, plant_food=15.0)
    lo = _world(disturbance=0.0, plant_food=15.0)
    cx, cy = _interior_land(hi)

    _regrow(hi)
    _regrow(lo)

    assert hi.F["plant_food"][cy, cx] < lo.F["plant_food"][cy, cx]


def test_ash_and_disturbance_diffuse_to_neighbors():
    """After the event block, ``diffuse_fields`` spreads ash and disturbance to
    the 3×3 neighborhood (90% self / 10% neighbors)."""
    for field in ("ash", "disturbance"):
        w = _world(**{field: 0.0})
        cx, cy = _interior_land(w)
        w.F[field][cy, cx] = 80.0
        w.diffuse_fields()
        assert w.F[field][cy, cx] < 80.0  # centre bleeds out
        assert w.F[field][cy, cx + 1] > 0.0  # neighbor gains


# ---------------------------------------------------------------------------
# D. Invariants
# ---------------------------------------------------------------------------
def test_event_effects_are_clipped_to_field_bounds():
    """Events can never push a field past its physical bounds."""
    # Drought cannot drive moisture negative: it saturates at the 0 floor.
    w0 = _world(moisture=5.0)
    cx0, cy0 = _interior_land(w0)
    w0.active_events.append(_event("drought", cx0, cy0))
    _regrow(w0)
    assert w0.F["moisture"][cy0, cx0] == 0.0
    # Fire ash saturates at 100.
    w = _world(ash=98.0)
    cx, cy = _interior_land(w)
    w.active_events.append(_event("fire", cx, cy))
    _regrow(w)
    assert w.F["ash"][cy, cx] == pytest.approx(100.0)
    # plant_food never goes negative even under a triple hit.
    w2 = _world(plant_food=2.0)
    cx2, cy2 = _interior_land(w2)
    for kind in ("fire", "drought", "blight"):
        w2.active_events.append(_event(kind, cx2, cy2))
    _regrow(w2)
    assert w2.F["plant_food"][cy2, cx2] >= 0.0


def test_regrow_grid_consumes_no_rng():
    """The event-application path is pure array math — reproducibility relies on
    it drawing zero random numbers."""
    w = _world()
    cx, cy = _interior_land(w)
    w.active_events.append(_event("fire", cx, cy))

    py_state = random.getstate()
    np_state = np.random.get_state()
    _regrow(w)

    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])


def test_events_do_not_touch_structures_without_fire():
    """Baseline invariant: a plain regrow tick (no fire) leaves built structures
    untouched. The *fire*-vs-structure case is the xfail below."""
    w = _world()
    cx, cy = _interior_land(w)
    for s in ("camp", "farm", "well"):
        w.S[s][cy, cx] = 1.0
    _regrow(w)
    for s in ("camp", "farm", "well"):
        assert w.S[s][cy, cx] == 1.0


# ---------------------------------------------------------------------------
# E. Known gaps — desired behaviour, expected to fail until implemented
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="fire should damage structures — not implemented yet")
def test_fire_damages_structures():
    """A wildfire burning over a camp/farm/well should degrade it."""
    w = _world(moisture=15.0, plant_food=100.0)
    cx, cy = _interior_land(w)
    for s in ("camp", "farm", "well"):
        w.S[s][cy, cx] = 1.0
    w.active_events.append(_event("fire", cx, cy, intensity=1.0))
    _regrow(w)
    assert any(w.S[s][cy, cx] < 1.0 for s in ("camp", "farm", "well"))


@pytest.mark.xfail(
    strict=True, reason="fire should consume flammable objects — not implemented yet"
)
def test_fire_consumes_flammable_objects():
    """Flammable material lying in a burning cell should be consumed (and feed
    the ash it helped ignite)."""
    w = _world(moisture=15.0, plant_food=100.0)
    cx, cy = _interior_land(w)
    w.get_cell(cx, cy).setdefault("materials", {})["wood"] = 5.0
    w.active_events.append(_event("fire", cx, cy, intensity=1.0))
    _regrow(w)
    assert w.get_cell(cx, cy)["materials"].get("wood", 0.0) < 5.0
