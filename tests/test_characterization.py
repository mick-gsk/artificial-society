"""Phase 0/1 characterization: a single explicit, headless, deterministic loop.

These guard the move off the dual bootstrap/monkeypatch loop onto one explicit
``Simulation.step()``: the run must complete headless, be reproducible from a
seed, and honour a tick bound. They are the regression net for the structural
refactor in later phases.
"""

from artificial_society.simulation import Simulation

SEED = 42
TICKS = 40
GRID_W, GRID_H = 24, 16
POP = 16


def _run(seed=SEED, ticks=TICKS):
    sim = Simulation(
        headless=True,
        seed=seed,
        grid_w=GRID_W,
        grid_h=GRID_H,
        initial_population=POP,
        load_checkpoint=False,
    )
    trajectory = []
    for _ in range(ticks):
        sim.step()
        alive = [a for a in sim.agents if a.alive]
        total_energy = sum(a.energy for a in alive)
        trajectory.append((sim.tick, len(alive), round(total_energy, 3)))
    return sim, trajectory


def test_headless_run_completes_without_error():
    sim, trajectory = _run()
    assert sim.tick == TICKS
    assert len(trajectory) == TICKS
    assert all(pop > 0 for _, pop, _ in trajectory)


def test_run_is_deterministic_for_fixed_seed():
    _, t1 = _run()
    _, t2 = _run()
    assert t1 == t2


def test_run_respects_max_ticks_bound():
    sim = Simulation(
        headless=True,
        seed=SEED,
        grid_w=GRID_W,
        grid_h=GRID_H,
        initial_population=POP,
        load_checkpoint=False,
    )
    sim.run(max_ticks=10)
    assert sim.tick == 10


def test_step_emits_no_per_tick_debug_noise(capsys):
    sim = Simulation(
        headless=True,
        seed=7,
        grid_w=20,
        grid_h=15,
        initial_population=10,
        load_checkpoint=False,
    )
    for _ in range(15):
        sim.step()
    out = capsys.readouterr().out
    for marker in ("[BIRTH]", "[TRY]", "[PREGNANCY_START]", "[NEED-INVENTION]"):
        assert marker not in out, f"hot-loop debug print leaked: {marker}"
