"""Audit group 8: disturbance events must follow world physics.

Genesis is state-driven (storm_risk, dryness+heat, fuel+ignition, moist dense
vegetation) instead of the old `tick % 55` timer + flat dice at a uniformly
random position; the warm-up from the pre-rewrite loop is enforced again; and
events have direct physical consequences for agents (fire burns, storms
exhaust, camps shelter).
"""

from __future__ import annotations

import random

from artificial_society.agents.agent import Agent
from artificial_society.environment import events as ev
from artificial_society.rng import seed_all
from artificial_society.world import World

RISKY_WEATHER = {"storm_risk": 1.0}
CALM_WEATHER = {"storm_risk": 0.0}


def _world(moisture=60.0, plant=10.0, temperature=25.0):
    seed_all(1234)
    w = World(20, 15)
    w.F["moisture"][:] = moisture
    w.F["plant_food"][:] = plant
    w.F["temperature"][:] = temperature
    return w


def test_warmup_suppresses_all_genesis(monkeypatch):
    """Restores the intent of the pre-rewrite loop: no disturbances early on."""
    w = _world(moisture=5.0, plant=100.0)  # dry, hot, fueled: everything wants to happen
    w.get_cell(4, 3).setdefault("materials", {})["ember"] = 1.0
    monkeypatch.setattr(random, "random", lambda: 0.0)  # every dice would pass

    for tick in range(0, 60):
        w.update_events(tick, {}, RISKY_WEATHER)

    assert w.active_events == [], "no event may form before EVENT_WARMUP_TICKS"


def test_storm_genesis_follows_storm_risk(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)

    w = _world(moisture=80.0, plant=10.0)  # wet + bare: no drought/fire/blight noise
    for tick in range(300):
        w.update_events(tick, {}, CALM_WEATHER)
    assert not any(e["kind"] == "storm" for e in w.active_events), (
        "zero storm risk must mean zero storms"
    )

    w2 = _world(moisture=80.0, plant=10.0)
    seen_storm = False
    for tick in range(600):
        w2.update_events(tick, {}, RISKY_WEATHER)
        seen_storm = seen_storm or any(e["kind"] == "storm" for e in w2.active_events)
    assert seen_storm, "high storm risk must eventually produce a storm"


def test_storms_move_along_their_wind(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    w = _world(moisture=80.0)
    storm = {"kind": "storm", "x": 10, "y": 7, "radius": 5, "intensity": 1.0, "ttl": 60,
             "vx": 1, "vy": 0}
    w.active_events.append(storm)

    w.update_events(2, {}, CALM_WEATHER)  # tick % STORM_MOVE_EVERY == 0

    assert (storm["x"], storm["y"]) == (11, 7), "storm must track along its wind vector"


def test_drought_forms_over_driest_warm_region_and_rain_ends_it(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    w = _world(moisture=60.0, temperature=25.0)
    w.F["moisture"][4:8, 4:8] = 5.0  # parched patch around (4..7, 4..7)

    w.update_events(ev.DROUGHT_CHECK_EVERY, {}, CALM_WEATHER)

    droughts = [e for e in w.active_events if e["kind"] == "drought"]
    assert droughts, "a dry warm region must seed a drought"
    d = droughts[0]
    assert 3 <= d["x"] <= 8 and 3 <= d["y"] <= 8, "drought must sit on the driest region"

    # Rain arrives: moisture recovers above the hysteresis exit -> collapse.
    w.F["moisture"][:] = 80.0
    for tick in range(21, 70):
        w.update_events(tick, {}, CALM_WEATHER)
    assert not any(e["kind"] == "drought" for e in w.active_events), (
        "rain must end a drought"
    )


def test_cold_region_gets_no_drought(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    w = _world(moisture=5.0, temperature=5.0)  # bone-dry but cold

    w.update_events(ev.DROUGHT_CHECK_EVERY, {}, CALM_WEATHER)

    assert not any(e["kind"] == "drought" for e in w.active_events)


def test_fire_needs_an_ignition_source(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    # Dry enough for fire (<45) but above the drought threshold; full fuel.
    w = _world(moisture=30.0, plant=100.0)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    for tick in range(100):
        w.update_events(tick, {}, CALM_WEATHER)

    assert not any(e["kind"] == "fire" for e in w.active_events), (
        "fuel + dryness alone must not ignite without a source"
    )


def test_agent_ember_ignites_wildfire_only_when_dry(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    w = _world(moisture=20.0, plant=100.0)
    w.get_cell(4, 3).setdefault("materials", {})["ember"] = 1.0
    w.update_events(ev.EMBER_CHECK_EVERY, {}, CALM_WEATHER)
    fires = [e for e in w.active_events if e["kind"] == "fire"]
    assert fires and (fires[0]["x"], fires[0]["y"]) == (4, 3), (
        "an ember in a dry fueled cell must be able to start a wildfire"
    )

    w2 = _world(moisture=80.0, plant=100.0)  # soaked: same ember, no fire
    w2.get_cell(4, 3).setdefault("materials", {})["ember"] = 1.0
    w2.update_events(ev.EMBER_CHECK_EVERY, {}, CALM_WEATHER)
    assert not any(e["kind"] == "fire" for e in w2.active_events)


def test_fire_starves_without_fuel(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    w = _world(moisture=0.0, plant=0.0, temperature=10.0)  # cold blocks drought noise
    w.active_events.append(
        {"kind": "fire", "x": 10, "y": 7, "radius": 3, "intensity": 1.0, "ttl": 100}
    )

    for tick in range(1, 25):
        w.update_events(tick, {}, CALM_WEATHER)

    assert not any(e["kind"] == "fire" for e in w.active_events), (
        "a fire without fuel must starve long before its ttl"
    )


def test_blight_needs_moist_dense_vegetation(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    lush = _world(moisture=70.0, plant=100.0)
    lush.update_events(ev.BLIGHT_CHECK_EVERY, {}, CALM_WEATHER)
    assert any(e["kind"] == "blight" for e in lush.active_events)

    dry = _world(moisture=30.0, plant=100.0)
    dry.update_events(ev.BLIGHT_CHECK_EVERY, {}, CALM_WEATHER)
    assert not any(e["kind"] == "blight" for e in dry.active_events)


def test_global_cap_holds_for_every_genesis_path(monkeypatch):
    monkeypatch.setattr(ev, "EVENT_WARMUP_TICKS", 0)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    w = _world(moisture=5.0, plant=100.0, temperature=25.0)  # drought+fire conditions
    w.get_cell(4, 3).setdefault("materials", {})["ember"] = 1.0
    for i in range(ev.MAX_ACTIVE_EVENTS):
        w.active_events.append(
            {"kind": "storm", "x": 2 + i, "y": 2, "radius": 4, "intensity": 1.0, "ttl": 50,
             "vx": 0, "vy": 0}
        )

    w.update_events(ev.DROUGHT_CHECK_EVERY, {}, RISKY_WEATHER)

    assert len(w.active_events) == ev.MAX_ACTIVE_EVENTS, (
        "no genesis path may exceed MAX_ACTIVE_EVENTS"
    )


def test_fire_burns_agents_and_storms_exhaust_them():
    seed_all(7)
    w = _world()
    burned = Agent.spawn_random(5, 5)
    soaked = Agent.spawn_random(10, 10)
    safe = Agent.spawn_random(0, 14)
    w.active_events.append(
        {"kind": "fire", "x": 5, "y": 5, "radius": 3, "intensity": 1.0, "ttl": 50}
    )
    w.active_events.append(
        {"kind": "storm", "x": 10, "y": 10, "radius": 3, "intensity": 1.0, "ttl": 50,
         "vx": 0, "vy": 0}
    )

    h0, e0 = burned.health, soaked.energy
    ev.apply_event_agent_effects(w, [burned, soaked, safe])

    assert burned.health < h0, "standing in a fire must hurt"
    assert soaked.energy < e0, "standing in a storm must cost energy"
    assert safe.health == 100.0 and safe.energy > 0, "agents outside events are untouched"


def test_camp_shelters_from_storms():
    seed_all(7)
    w = _world()
    exposed = Agent.spawn_random(10, 10)
    sheltered = Agent.spawn_random(4, 4)
    w.S["camp"][4, 4] = 1.0
    for x, y in ((10, 10), (4, 4)):
        w.active_events.append(
            {"kind": "storm", "x": x, "y": y, "radius": 3, "intensity": 1.0, "ttl": 50,
             "vx": 0, "vy": 0}
        )

    e_exposed, e_sheltered = exposed.energy, sheltered.energy
    ev.apply_event_agent_effects(w, [exposed, sheltered])

    loss_exposed = e_exposed - exposed.energy
    loss_sheltered = e_sheltered - sheltered.energy
    assert loss_sheltered < loss_exposed, "a camp must dampen storm exposure"
