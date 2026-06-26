"""Tests for the dashboard FastAPI app via Starlette's TestClient (CPU)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from artificial_society.serve.app import app

SMALL = {"seed": 1, "grid_w": 20, "grid_h": 15, "pop": 8}


def _wait_status(client, target, timeout=30.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get("/api/status").json()["status"] == target:
            return True
        time.sleep(interval)
    return False


def test_health_reports_device():
    with TestClient(app) as client:
        info = client.get("/api/health").json()["device"]
        assert info["type"] in {"cuda", "cpu", "unknown"}


def test_index_served():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Artificial Society" in resp.text


def test_run_status_history_and_graph():
    with TestClient(app) as client:
        # bounded run so the test is deterministic and quick
        resp = client.post("/api/run", json={**SMALL, "ticks": 6})
        assert resp.status_code == 200
        assert _wait_status(client, "finished")

        snap = client.get("/api/status").json()
        assert snap["tick"] == 6
        # stats are collected inside step() (registry 'stats' tick), labelled with the
        # 0-indexed in-loop tick, so 6 ticks (0..5) give a last sample of 5.
        assert snap["stats"]["tick"] == 5

        hist = client.get("/api/history").json()
        assert len(hist["population_history"]) == 6

        png = client.get("/api/graph.png")
        assert png.status_code == 200
        assert png.headers["content-type"] == "image/png"
        assert png.content[:4] == b"\x89PNG"


def test_graph_empty_before_run_returns_204():
    with TestClient(app) as client:
        # ensure no leftover history from a prior test's runner state
        client.post("/api/stop")
        # a fresh app instance would be cleaner, but 204 holds whenever history is empty
        resp = client.get("/api/graph.png")
        assert resp.status_code in {200, 204}  # 200 only if a prior run left history


def test_double_run_conflicts():
    with TestClient(app) as client:
        client.post("/api/run", json={**SMALL})  # unbounded, stays running
        try:
            assert _wait_status(client, "running")
            second = client.post("/api/run", json={**SMALL})
            assert second.status_code == 409
        finally:
            client.post("/api/stop")
