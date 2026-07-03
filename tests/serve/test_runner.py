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
    # stats are now collected inside step() (registry 'stats' tick) with the 0-indexed
    # in-loop tick, so after 10 ticks (0..9) the last sample is labelled 9.
    assert snap["stats"]["tick"] == 9
    assert snap["stats"]["population"] >= 0
    assert not r.is_running

    hist = r.history()
    assert len(hist["population_history"]) == 10
    # history entries are (tick, value) tuples; first sample is the 0-indexed tick 0
    assert hist["population_history"][0][0] == 0


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


def _inspect_union(r):
    """The set of agent ids the worker would pass to build_frame right now."""
    with r._lock:
        return frozenset(r._inspect.values())


def test_inspect_registry_union_and_clear():
    r = SimulationRunner()
    assert _inspect_union(r) == frozenset()

    r.set_inspect(1, 5)
    r.set_inspect(2, 9)
    assert _inspect_union(r) == {5, 9}

    # None removes just that token's entry, leaving the other client's intact.
    r.set_inspect(1, None)
    assert _inspect_union(r) == {9}

    # clear_client (disconnect) drops the remaining registration.
    r.clear_client(2)
    assert _inspect_union(r) == frozenset()

    # clear_client is idempotent — a second call on an unknown token is a no-op.
    r.clear_client(2)
    r.clear_client(999)
    assert _inspect_union(r) == frozenset()


def test_two_clients_inspecting_same_agent_collapse_in_union():
    r = SimulationRunner()
    r.set_inspect(1, 7)
    r.set_inspect(2, 7)
    assert _inspect_union(r) == {7}
    # one client leaving still leaves the agent inspected by the other
    r.clear_client(1)
    assert _inspect_union(r) == {7}


def test_start_preserves_inspect_registrations():
    r = SimulationRunner()
    r.set_inspect(1, 3)
    r.set_inspect(2, 4)
    # A fresh run (start clears run state) must NOT wipe connection-scoped
    # inspect registrations — a client keeps inspecting across a run restart.
    r.start({**SMALL, "ticks": 3})
    assert _wait(lambda: r.snapshot()["status"] == "finished"), r.snapshot()
    assert _inspect_union(r) == {3, 4}


def _layer_union(r):
    """The set of overlay names the worker would pass to build_frame right now."""
    with r._lock:
        return frozenset().union(*r._layers.values()) if r._layers else frozenset()


def test_layers_registry_union_and_clear():
    r = SimulationRunner()
    assert _layer_union(r) == frozenset()

    r.set_layers(1, ["temperature"])
    r.set_layers(2, ["danger", "disease"])
    assert _layer_union(r) == {"temperature", "danger", "disease"}

    # A new selection *replaces* a token's full set (not a merge).
    r.set_layers(1, ["disease"])
    assert _layer_union(r) == {"danger", "disease"}

    # An empty list removes that token's entry entirely.
    r.set_layers(2, [])
    assert _layer_union(r) == {"disease"}

    # clear_client (disconnect) drops the remaining registration.
    r.clear_client(1)
    assert _layer_union(r) == frozenset()

    # clear_client is idempotent on unknown/known tokens.
    r.clear_client(1)
    r.clear_client(999)
    assert _layer_union(r) == frozenset()


def test_two_clients_selecting_overlapping_layers_union():
    r = SimulationRunner()
    r.set_layers(1, ["danger", "temperature"])
    r.set_layers(2, ["danger", "disease"])
    assert _layer_union(r) == {"danger", "temperature", "disease"}
    # one client leaving still leaves the layers the other still wants
    r.clear_client(1)
    assert _layer_union(r) == {"danger", "disease"}


def test_clear_client_drops_both_inspect_and_layers():
    r = SimulationRunner()
    r.set_inspect(5, 3)
    r.set_layers(5, ["danger"])
    r.clear_client(5)
    assert _inspect_union(r) == frozenset()
    assert _layer_union(r) == frozenset()


def test_start_preserves_layer_registrations():
    r = SimulationRunner()
    r.set_layers(1, ["danger"])
    r.set_layers(2, ["temperature"])
    r.start({**SMALL, "ticks": 3})
    assert _wait(lambda: r.snapshot()["status"] == "finished"), r.snapshot()
    assert _layer_union(r) == {"danger", "temperature"}
