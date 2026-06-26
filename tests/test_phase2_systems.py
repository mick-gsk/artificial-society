"""Phase 2 liveness: dormant built-in systems, once activated via the registry,
actually run and change the world.

These assert the *intended* new behaviour. The golden trajectory
(tests/test_regression_golden.py) is regenerated to match each activation, so it
keeps guarding against *unintended* drift on top of the new baseline.
"""

from artificial_society.simulation import Simulation
from artificial_society.systems import registry


def _fresh(**kw):
    params = dict(
        headless=True, seed=1, grid_w=20, grid_h=15, initial_population=8, load_checkpoint=False
    )
    params.update(kw)
    return Simulation(**params)


def test_environment_systems_registered():
    sim = _fresh()
    for name in ("seasons", "weather", "world_regrowth"):
        assert name in sim.systems, f"{name} not built from registry"
        assert hasattr(sim, name)


def test_world_regrowth_runs_each_step():
    sim = _fresh()
    # Fully deplete the edible layer across the whole map.
    for row in sim.world.cells:
        for cell in row:
            cell["plant_food"] = 0.0
            cell["meat_food"] = 0.0
            cell["food"] = 0.0
    sim.step()
    total_food = sum(c["food"] for row in sim.world.cells for c in row)
    assert total_food > 0.0, "world.update_environment did not regrow food after a step"


def test_births_occur_and_grow_population():
    # Agents reach MIN_REPRODUCTION_AGE (60) then gestate 40 ticks, so births need
    # ~100+ ticks; run long enough to observe several.
    sim = Simulation(
        headless=True, seed=3, grid_w=22, grid_h=16, initial_population=16, load_checkpoint=False
    )
    for _ in range(220):
        sim.step()
    alive = [a for a in sim.agents if a.alive]
    born_after_start = [a for a in alive if a.birth_tick > 0]
    assert born_after_start, "no child was ever born (reproduction not wired)"
    assert len(alive) > 16, "population did not grow via births"


def test_season_and_weather_state_published():
    sim = _fresh()
    sim.step()
    assert isinstance(getattr(sim, "_season_state", None), dict)
    assert "phase" in sim._season_state
    assert isinstance(getattr(sim, "_weather_state", None), dict)
    assert "rain_map" in sim._weather_state


# --- Society systems activated this phase (tribes / economy / technology / stats / disease) ---


def test_society_systems_registered():
    sim = _fresh()
    for name in ("tribes", "economy", "technology", "stats", "disease"):
        assert name in sim.systems, f"{name} not built from registry"
        assert hasattr(sim, name)


def test_tick_order_stats_last_and_disease_before_economy():
    """Order is a contract: stats must read post-update state (last), and disease must
    seed/spread before economy and stats so the same tick reflects new infections."""
    ticked = [s.name for s in registry.specs() if s.tick is not None]
    assert ticked[-1] == "stats", f"stats must tick last, got {ticked}"
    assert ticked.index("tribes") < ticked.index("disease") < ticked.index("economy")
    assert ticked.index("disease") < ticked.index("stats")


def test_life_stage_counts_are_collected_not_zero():
    """Regression for the latent bug: ``life_stage`` is a *method*; the old
    ``getattr(a, "life_stage", ...)`` returned the bound method, so every life-stage
    count was 0. After the fix the counts partition the living population."""
    sim = _fresh()
    for _ in range(5):
        sim.step()
    live = [a for a in sim.agents if a.alive]
    last = sim.stats.last
    total = last["n_child"] + last["n_adult"] + last["n_elder"]
    assert total == len(live), "life-stage counts do not sum to the living population"
    assert total > 0, "life-stage counts are all zero (the getattr-on-a-method bug)"
    # agents spawn at age 0, so early on every living agent is a child
    assert last["n_child"] >= 1


def test_stats_collected_each_step():
    sim = _fresh()
    sim.step()
    assert "tick" in sim.stats.last
    assert sim.stats.population_history, "stats.update never ran"


def test_tribes_form_from_trust():
    # Trust must build before sociable agents form a tribe; seed/pop/size chosen so a
    # tribe reliably forms within the horizon (first forms ~tick 72 at this seed).
    sim = _fresh(seed=1, grid_w=22, grid_h=16, initial_population=16)
    for _ in range(120):
        sim.step()
    assert sim.tribes.count() >= 1, "no tribe ever formed (tribes tick not wired)"
    assert any(a.tribe_id is not None for a in sim.agents if a.alive)


def test_economy_updates_prices():
    sim = _fresh()
    for _ in range(30):
        sim.step()
    assert set(sim.economy.prices) == {"wood", "stone", "fiber"}
    assert sim.economy.prices != {"wood": 1.0, "stone": 1.0, "fiber": 1.0}, (
        "economy.update never recomputed prices from the default"
    )


def test_technology_tracks_capabilities():
    sim = _fresh()
    for _ in range(60):
        sim.step()
    assert isinstance(sim.technology.capability_map, dict)
    assert len(sim.technology.capability_map) >= 1, "technology.update never populated capabilities"


def test_disease_environmental_infection_fires():
    """Before this phase no agent could ever be infected: the environmental trigger had no
    call site. Holding an agent at the wound-fever precondition (health < 35), the disease
    system must eventually infect someone. Asserts 'eventually', never a single RNG draw."""
    sim = _fresh()
    assert "disease" in sim.systems
    target = next((a for a in sim.agents if a.alive), None)
    infected = False
    for _ in range(300):
        if target is None or not target.alive:
            target = next((a for a in sim.agents if a.alive), None)
        if target is not None:
            target.health = 10.0  # keep the wound-fever window open against health regen
        sim.step()
        if any(getattr(a, "disease_id", None) is not None for a in sim.agents if a.alive):
            infected = True
            break
    assert infected, "disease system never produced an infection despite wound-fever conditions"
