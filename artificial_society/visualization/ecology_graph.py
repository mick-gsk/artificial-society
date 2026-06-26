"""Headless-safe time-series visualization of ecology/population.

This module is a *read-only observer* of the simulation. It reads the history
buffers that :class:`~artificial_society.visualization.statistics.StatisticsTracker`
accumulates and renders a matplotlib figure plotting population, mean world food
and mean agent energy over time.

It is fully headless: matplotlib is forced onto the non-interactive ``Agg``
backend at import time, so no display is ever required. Figures can be returned
to the caller or saved to a PNG file.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

# Force a non-interactive backend *before* importing pyplot so this module is
# safe to import in headless environments (tests run with SDL_VIDEODRIVER=dummy).
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
from matplotlib.figure import Figure  # noqa: E402

HistoryEntry = tuple[int, float]


def _split_history(history: Sequence | None) -> tuple[list[float], list[float]]:
    """Split a history buffer into parallel (ticks, values) lists.

    Entries are normally ``(tick, value)`` tuples (as stored by
    :class:`StatisticsTracker`), but plain scalars are tolerated and indexed
    positionally. Read-only: the input is never mutated.
    """
    ticks: list[float] = []
    values: list[float] = []
    if not history:
        return ticks, values
    for i, entry in enumerate(history):
        if isinstance(entry, (int, float)):
            ticks.append(float(i))
            values.append(float(entry))
            continue
        try:
            ticks.append(float(entry[0]))
            values.append(float(entry[1]))
        except (TypeError, IndexError, ValueError):
            continue
    return ticks, values


def build_ecology_figure(
    tracker,
    *,
    title: str = "Ecology & Population over time",
) -> Figure:
    """Build a matplotlib :class:`Figure` from a ``StatisticsTracker``.

    Plots population (left axis) together with mean world food and mean agent
    energy (right axis) over the recorded tick history. The tracker is only
    read from, never mutated.

    Returns a freshly created :class:`matplotlib.figure.Figure`; the caller is
    responsible for saving or closing it.
    """
    pop_ticks, pop_vals = _split_history(getattr(tracker, "population_history", None))
    food_ticks, food_vals = _split_history(getattr(tracker, "food_history", None))
    energy_ticks, energy_vals = _split_history(getattr(tracker, "energy_history", None))

    fig, ax_pop = plt.subplots(figsize=(8, 4.5))
    ax_env = ax_pop.twinx()

    ax_pop.set_xlabel("tick")
    ax_pop.set_ylabel("population", color="tab:blue")
    ax_env.set_ylabel("mean world food / mean energy")

    # Population on the primary axis.
    ax_pop.plot(
        pop_ticks,
        pop_vals,
        color="tab:blue",
        label="population",
        linewidth=2,
    )
    ax_pop.tick_params(axis="y", labelcolor="tab:blue")

    # Ecology signals on the secondary axis.
    ax_env.plot(
        food_ticks,
        food_vals,
        color="tab:green",
        label="mean world food",
        linewidth=1.5,
    )
    ax_env.plot(
        energy_ticks,
        energy_vals,
        color="tab:orange",
        label="mean agent energy",
        linewidth=1.5,
        linestyle="--",
    )

    # Single combined legend.
    lines_pop, labels_pop = ax_pop.get_legend_handles_labels()
    lines_env, labels_env = ax_env.get_legend_handles_labels()
    ax_pop.legend(
        lines_pop + lines_env,
        labels_pop + labels_env,
        loc="upper left",
        fontsize=8,
    )

    ax_pop.set_title(title)
    ax_pop.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def save_ecology_graph(tracker, path: str, **kwargs) -> str:
    """Render the ecology figure and write it to ``path`` as a PNG.

    Returns the path written. The figure is closed afterwards to avoid leaking
    matplotlib state across calls.
    """
    fig = build_ecology_figure(tracker, **kwargs)
    try:
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
    return path
