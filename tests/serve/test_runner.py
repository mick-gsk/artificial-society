"""Tests for the dashboard SimulationRunner (CPU is sufficient)."""
from __future__ import annotations

import time

from artificial_society.serve import runner as runner_mod
from artificial_society.serve.runner import SimulationRunner

SMALL = {"seed": 1, "grid_w": 20, "grid_h": 15, "pop": 8}


def _wait(pred, timeout=30.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_bounded_run_finishes_with_stats_and_history():
    r = SimulationRunner()
    r.start({**SMALL, "ticks": 10})
    assert _wait(lambda: r.snapshot()["status"] == "finished"), r.snapshot()

    snap = r.snapshot()
    assert snap["tick"] == 10
    assert snap["stats"]["tick"] == 10
    assert snap["stats"]["population"] >= 0
    assert not r.is_running

    hist = r.history()
    assert len(hist["population_history"]) == 10
    # history entries are (tick, value) tuples
    assert hist["population_history"][0][0] == 1


def test_stop_halts_a_running_sim():
    r = SimulationRunner()
    r.start({**SMALL})  # unbounded
    assert _wait(lambda: r.snapshot()["tick"] >= 2), r.snapshot()
    r.stop()
    assert not r.is_running
    assert r.snapshot()["status"] == "stopped"


def test_start_replaces_a_previous_run():
    r = SimulationRunner()
    r.start({**SMALL})
    assert _wait(lambda: r.snapshot()["tick"] >= 1)
    r.start({**SMALL, "ticks": 5})
    assert _wait(lambda: r.snapshot()["status"] == "finished"), r.snapshot()
    assert r.snapshot()["tick"] == 5


def test_failed_run_sets_status_without_killing_runner(monkeypatch):
    class BoomSim:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(runner_mod, "Simulation", BoomSim)
    r = SimulationRunner()
    r.start({**SMALL, "ticks": 3})
    assert _wait(lambda: r.snapshot()["status"] == "failed"), r.snapshot()
    assert "boom" in (r.snapshot()["error"] or "")


def test_device_info_reports_a_type():
    info = SimulationRunner().device_info()
    assert info["type"] in {"cuda", "cpu", "unknown"}
