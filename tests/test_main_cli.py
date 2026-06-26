"""Phase 1a: the app entry point drives the explicit loop, headless-capable.

main() must accept argv, run bounded + headless without opening a window, and
return 0 — proving the application path uses the same explicit Simulation.run
as the tests (no bootstrap loop override).
"""
from artificial_society import main as main_mod


def test_main_headless_bounded_returns_zero():
    rc = main_mod.main([
        "--headless", "--seed", "42", "--ticks", "12",
        "--grid-w", "20", "--grid-h", "15", "--pop", "10",
    ])
    assert rc == 0


def test_main_does_not_apply_bootstrap_run_override():
    # The explicit Simulation.run must be the one the app uses, not bootstrap's.
    from artificial_society.simulation import Simulation
    assert Simulation.run.__qualname__.startswith("Simulation.")
