"""Drive a single headless :class:`Simulation` in a background thread.

The dashboard owns one :class:`SimulationRunner`. It steps the simulation
flat-out (or throttled) on its own thread and, after every tick, refreshes the
statistics tracker and copies a consistent snapshot under a lock so HTTP
handlers can read live state without tearing or racing the writer.

The simulation is always constructed ``headless=True`` (no pygame display) and
with ``load_checkpoint=False`` so seeded runs start fresh and reproducible. Stats
are collected inside ``Simulation.step`` (the registry ``stats`` tick, Phase 2), so
the runner just snapshots ``sim.stats`` after each tick rather than updating it.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any

from artificial_society.simulation import Simulation

# History buffers maintained by StatisticsTracker (each a list of (tick, value)).
HISTORY_KEYS = (
    "population_history",
    "knowledge_history",
    "cooperation_history",
    "food_history",
    "energy_history",
)

# Defaults mirror the headless CLI (artificial_society.main).
DEFAULTS = {"grid_w": 60, "grid_h": 40, "pop": 36, "seed": None, "ticks": None}


def _empty_history() -> dict[str, list[tuple[int, float]]]:
    return {k: [] for k in HISTORY_KEYS}


class SimulationRunner:
    """Owns one background simulation run at a time.

    States: ``idle`` (nothing has run), ``running``, ``stopped`` (halted by the
    user), ``finished`` (reached its ``ticks`` bound), ``failed`` (the loop
    raised — ``error`` holds the traceback; the server keeps serving).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sim: Simulation | None = None

        self._status = "idle"
        self._error: str | None = None
        self._params: dict[str, Any] = {}
        self._tick = 0
        self._last_stats: dict[str, Any] = {}
        self._history = _empty_history()

    # -- introspection ------------------------------------------------------

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def device_info(self) -> dict[str, str]:
        """Report the torch compute device so the dashboard can confirm the GPU
        is actually in use (or warn loudly when it falls back to CPU)."""
        try:
            import torch

            if torch.cuda.is_available():
                return {"type": "cuda", "name": torch.cuda.get_device_name(0)}
            return {"type": "cpu", "name": "CPU"}
        except Exception as exc:  # torch missing / driver error — never crash a request
            return {"type": "unknown", "name": str(exc)}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "tick": self._tick,
                "params": dict(self._params),
                "stats": dict(self._last_stats),
                "error": self._error,
                "device": self.device_info(),
            }

    def history(self) -> dict[str, list[tuple[int, float]]]:
        with self._lock:
            return {k: list(v) for k, v in self._history.items()}

    # -- control ------------------------------------------------------------

    def start(self, params: dict[str, Any] | None = None) -> None:
        """Stop any active run, then launch a fresh one with ``params``
        (``seed, ticks, grid_w, grid_h, pop`` — missing keys use DEFAULTS)."""
        self.stop()  # idempotent; joins a previous thread if alive
        merged = dict(DEFAULTS)
        if params:
            merged.update({k: v for k, v in params.items() if v is not None})

        self._stop.clear()
        with self._lock:
            self._status = "running"
            self._error = None
            self._params = merged
            self._tick = 0
            self._last_stats = {}
            self._history = _empty_history()

        self._thread = threading.Thread(
            target=self._run_loop, args=(merged,), name="sim-runner", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the loop to halt and wait for the thread to exit."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        with self._lock:
            if self._status == "running":
                self._status = "stopped"

    # -- worker -------------------------------------------------------------

    def _run_loop(self, params: dict[str, Any]) -> None:
        try:
            sim = Simulation(
                headless=True,
                seed=params.get("seed"),
                grid_w=int(params["grid_w"]),
                grid_h=int(params["grid_h"]),
                initial_population=int(params["pop"]),
                load_checkpoint=False,
            )
            with self._lock:
                self._sim = sim

            max_ticks = params.get("ticks")
            throttle = float(params.get("min_step_interval", 0.0) or 0.0)
            n = 0
            while not self._stop.is_set():
                sim.step()  # the registry 'stats' tick collects stats during the step
                with self._lock:
                    self._tick = sim.tick
                    self._last_stats = dict(sim.stats.last)
                    self._history = {k: list(getattr(sim.stats, k, [])) for k in HISTORY_KEYS}
                n += 1
                if max_ticks is not None and n >= int(max_ticks):
                    with self._lock:
                        if self._status == "running":
                            self._status = "finished"
                    return
                if throttle:
                    time.sleep(throttle)
        except Exception:
            with self._lock:
                self._status = "failed"
                self._error = traceback.format_exc()
