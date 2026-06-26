"""FastAPI app for the LAN dashboard.

Holds one :class:`SimulationRunner` and exposes start/stop/status/history plus a
server-rendered ecology PNG and a device-info health check. Serves the static
single-page dashboard from ``serve/static``.

Run with ``python -m artificial_society.serve`` (see ``__main__``).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless, thread-safe rendering — no display, no GUI backend

import matplotlib.pyplot as plt  # noqa: E402
from fastapi import FastAPI, Response  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from artificial_society.serve.runner import SimulationRunner  # noqa: E402
from artificial_society.visualization.ecology_graph import build_ecology_figure  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Artificial Society Dashboard")
runner = SimulationRunner()


class RunParams(BaseModel):
    seed: Optional[int] = None
    ticks: Optional[int] = None
    grid_w: Optional[int] = None
    grid_h: Optional[int] = None
    pop: Optional[int] = None


class _TrackerShim:
    """Minimal stand-in exposing the *_history attributes build_ecology_figure reads."""

    def __init__(self, history: dict) -> None:
        for key, series in history.items():
            setattr(self, key, series)


@app.get("/api/health")
def health() -> dict:
    return {"device": runner.device_info()}


@app.get("/api/status")
def status() -> dict:
    return runner.snapshot()


@app.get("/api/history")
def history() -> dict:
    return runner.history()


@app.post("/api/run")
def run(params: RunParams) -> JSONResponse:
    if runner.is_running:
        return JSONResponse(status_code=409, content={"detail": "a run is already active; stop it first"})
    runner.start(params.model_dump())
    return JSONResponse(runner.snapshot())


@app.post("/api/stop")
def stop() -> dict:
    runner.stop()
    return runner.snapshot()


@app.get("/api/graph.png")
def graph_png() -> Response:
    hist = runner.history()
    if not hist.get("population_history"):
        return Response(status_code=204)  # nothing to plot yet
    fig = build_ecology_figure(_TrackerShim(hist))
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
    finally:
        plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
