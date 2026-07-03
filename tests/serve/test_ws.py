"""WebSocket live-frame + back-channel tests via Starlette's TestClient (CPU)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from artificial_society.serve.app import app

SMALL = {"seed": 1, "grid_w": 12, "grid_h": 8, "pop": 6}


def _next_frame(ws):
    """Receive messages until a ``frame`` arrives, returning it (skips hello)."""
    while True:
        msg = ws.receive_json()
        if msg.get("type") == "frame":
            return msg


def _read_frames_until(ws, pred, budget=200):
    """Read up to ``budget`` frames, returning the first that satisfies ``pred``.

    Returns ``None`` if the budget is exhausted without a match. The budget is
    generous because frames are tick-deduplicated and only sampled ~20 Hz, so it
    can take several reads before a just-registered inspect request shows up in a
    freshly built frame.
    """
    for _ in range(budget):
        frame = _next_frame(ws)
        if pred(frame):
            return frame
    return None


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

                frame = _next_frame(ws)
                assert frame["type"] == "frame"
                assert frame["grid"] == {"w": 12, "h": 8}
                assert len(frame["cells"]["food"]) == 12 * 8
                assert "agents" in frame and "events" in frame
                # No inspection requested yet -> no detail blob on the wire.
                assert "detail" not in frame
        finally:
            client.post("/api/stop")


def test_ws_inspect_backchannel_toggles_detail():
    """Full L3/L4 flow: request a living agent -> detail appears with exactly
    that id; clear it -> detail disappears again; the sender keeps streaming."""
    with TestClient(app) as client:
        assert client.post("/api/run", json={**SMALL}).status_code == 200
        try:
            with client.websocket_connect("/ws") as ws:
                hello = ws.receive_json()
                assert hello["type"] == "hello"

                # (b) frames flow with no detail before any inspect request.
                frame = _read_frames_until(ws, lambda f: bool(f["agents"]))
                assert frame is not None, "no frame with agents arrived"
                assert "detail" not in frame
                target_id = frame["agents"][0]["id"]

                # (c) request inspection of a living agent -> detail appears,
                # keyed by exactly that id, within a few frames.
                ws.send_json({"type": "inspect", "id": target_id})
                got = _read_frames_until(ws, lambda f: "detail" in f)
                assert got is not None, "detail never appeared after inspect request"
                assert str(target_id) in got["detail"]
                blob = got["detail"][str(target_id)]
                assert blob["id"] == target_id  # the deep read is for the right agent
                assert "needs" in blob  # a real _agent_detail payload, not a stub

                # (d) clear the inspection -> detail disappears again.
                ws.send_json({"type": "inspect", "id": None})
                cleared = _read_frames_until(ws, lambda f: "detail" not in f)
                assert cleared is not None, "detail never cleared after id=null"

                # (e) an unknown message type must not raise or kill the stream.
                ws.send_json({"type": "quatsch", "foo": 42})
                still_alive = _next_frame(ws)
                assert still_alive["type"] == "frame"
                assert "detail" not in still_alive  # unknown type changed nothing
        finally:
            client.post("/api/stop")


def test_ws_layers_backchannel_toggles_overlay():
    """Full C3 flow: request a server overlay -> ``layers.temperature`` appears
    within the throttled cadence; clear it -> the layers stop arriving."""
    with TestClient(app) as client:
        assert client.post("/api/run", json={**SMALL}).status_code == 200
        try:
            with client.websocket_connect("/ws") as ws:
                hello = ws.receive_json()
                assert hello["type"] == "hello"

                # (a) no layers requested yet -> no layers key on the wire.
                frame = _read_frames_until(ws, lambda f: bool(f["agents"]))
                assert frame is not None
                assert "layers" not in frame

                # (b) request the temperature overlay -> within a few (throttled)
                # frames a frame arrives carrying layers.temperature. Budget is
                # generous: overlays ship only every 10th tick.
                ws.send_json({"type": "layers", "want": ["temperature"]})
                got = _read_frames_until(
                    ws, lambda f: "layers" in f and "temperature" in f["layers"]
                )
                assert got is not None, "temperature overlay never appeared"
                n = got["grid"]["w"] * got["grid"]["h"]
                temp = got["layers"]["temperature"]
                assert len(temp) == n
                assert all(0 <= v <= 9 for v in temp)

                # (c) clear the request -> layers stop arriving. We must see
                # several tick%10 frames with no layers key to be sure it's off,
                # so read a run of frames and assert none carry layers once
                # cleared (allowing a couple of in-flight frames to drain first).
                ws.send_json({"type": "layers", "want": []})
                cleared = _read_frames_until(
                    ws,
                    lambda f: f["tick"] % 10 == 0 and "layers" not in f,
                )
                assert cleared is not None, "layers never cleared after want=[]"
        finally:
            client.post("/api/stop")
