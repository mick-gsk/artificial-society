"""Aging / senescence thresholds (Phase 4).

`_age_tick` ramps health decay in two stages and then hard-caps lifespan:

    START (3500) <  HARD (4500) <  LIMIT (5000)
    age > START : -0.04 health/tick   (soft senescence)
    age >= HARD : -0.12 health/tick   (steep senescence)
    age >= LIMIT: death

These tests lock the ordering and prove the steep ramp is actually *reachable
while the agent is still alive* (i.e. death-by-old-age does not pre-empt the
senescence curve), which is the consistency the roadmap asks Phase 4 to verify.
"""

from __future__ import annotations

import pytest

from artificial_society.agents.agent import (
    AGE_HEALTH_DECAY_HARD,
    AGE_HEALTH_DECAY_START,
    AGE_LIMIT,
)
from artificial_society.simulation import Simulation

SOFT_RATE = 0.04
HARD_RATE = 0.12


def _agent():
    sim = Simulation(
        headless=True,
        load_checkpoint=False,
        seed=42,
        grid_w=10,
        grid_h=8,
        initial_population=4,
    )
    return sim.agents[0]


def test_threshold_ordering_is_consistent():
    assert AGE_HEALTH_DECAY_START < AGE_HEALTH_DECAY_HARD < AGE_LIMIT


def test_no_senescence_decay_before_start():
    a = _agent()
    a.age = AGE_HEALTH_DECAY_START - 100
    a.health = 100.0
    a.alive = True
    for _ in range(50):  # stays below START
        a._age_tick()
    assert a.health == pytest.approx(100.0), "health decayed before the senescence start age"


def test_soft_then_steep_rate():
    a = _agent()
    # Soft band: just past START.
    a.age = AGE_HEALTH_DECAY_START + 10
    a.health = 100.0
    a.alive = True
    a._age_tick()
    assert a.health == pytest.approx(100.0 - SOFT_RATE, abs=1e-9)

    # Steep band: at/after HARD.
    a.age = AGE_HEALTH_DECAY_HARD + 10
    a.health = 100.0
    a._age_tick()
    assert a.health == pytest.approx(100.0 - HARD_RATE, abs=1e-9)


def test_steep_ramp_is_reachable_while_alive():
    a = _agent()
    a.age = AGE_HEALTH_DECAY_START
    a.health = 100.0
    a.alive = True

    # Age through the entire soft band.
    while a.age < AGE_HEALTH_DECAY_HARD:
        a._age_tick()

    assert a.alive, "agent died before reaching the steep senescence ramp"
    assert a.health < 100.0, "soft senescence band applied no decay"
    # ~1000 ticks of soft decay from a healthy start.
    assert a.health == pytest.approx(
        100.0 - SOFT_RATE * (AGE_HEALTH_DECAY_HARD - AGE_HEALTH_DECAY_START), abs=0.5
    )

    # One more tick is now in the steep band.
    h_before = a.health
    a._age_tick()
    assert (h_before - a.health) == pytest.approx(HARD_RATE, abs=1e-9)


def test_death_at_age_limit():
    a = _agent()
    a.age = AGE_LIMIT - 1
    a.health = 100.0  # healthy: death here is purely the lifespan cap
    a.alive = True
    a._age_tick()
    assert a.age == AGE_LIMIT
    assert not a.alive, "agent outlived the hard age limit"
