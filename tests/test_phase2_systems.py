"""Phase 2 liveness: dormant built-in systems, once activated via the registry,
actually run and change the world.

These assert the *intended* new behaviour. The golden trajectory
(tests/test_regression_golden.py) is regenerated to match each activation, so it
keeps guarding against *unintended* drift on top of the new baseline.
"""

from artificial_society.simulation import Simulation


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
