"""WebSocket live-frame smoke test via Starlette's TestClient (CPU)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from artificial_society.serve.app import app

SMALL = {"seed": 1, "grid_w": 12, "grid_h": 8, "pop": 6}


def test_ws_streams_hello_then_frames():
    with TestClient(app) as client:
        # Unbounded run so frames keep flowing while we read a couple.
        assert client.post("/api/run", json={**SMALL}).status_code == 200
        try:
            with client.websocket_connect("/ws") as ws:
                hello = ws.receive_json()
                assert hello["type"] == "hello"
                assert hello["biomes"][0]["idx"] == 0
                assert len(hello["biomes"][0]["rgb"]) == 3

                frame = ws.receive_json()
                assert frame["type"] == "frame"
                assert frame["grid"] == {"w": 12, "h": 8}
                assert len(frame["cells"]["food"]) == 12 * 8
                assert "agents" in frame and "events" in frame
        finally:
            client.post("/api/stop")
