"""Shared helpers for deterministic, headless trajectory capture.

Used by both the golden-trajectory generator and the regression test so the
two can never drift apart.
"""

from __future__ import annotations

from artificial_society.simulation import Simulation

# Canonical, deliberately small run used as the behaviour fingerprint.
PARAMS = dict(seed=42, grid_w=30, grid_h=20, initial_population=24)
TICKS = 60


def compute_trajectory(ticks=TICKS, **overrides):
    """Run headless and return a per-tick fingerprint of the population state.

    The fingerprint (population, total energy, total health) is sensitive to
    essentially every agent/world interaction, so an unchanged trajectory is
    strong evidence a refactor preserved behaviour.
    """
    params = {**PARAMS, **overrides}
    sim = Simulation(headless=True, load_checkpoint=False, **params)
    trajectory = []
    for _ in range(ticks):
        sim.step()
        alive = [a for a in sim.agents if a.alive]
        trajectory.append(
            [
                sim.tick,
                len(alive),
                round(sum(a.energy for a in alive), 2),
                round(sum(a.health for a in alive), 2),
            ]
        )
    return trajectory
