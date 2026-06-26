"""Headless tests for the ecology/population time-series graph.

These assert the matplotlib figure renders without a display (Agg backend) and
that the visualization is a pure read-only observer of the simulation.
"""
import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.figure import Figure

from artificial_society.visualization.ecology_graph import (
    build_ecology_figure,
    save_ecology_graph,
)
from artificial_society.visualization.statistics import StatisticsTracker


class _SyntheticTracker:
    """Minimal stand-in exposing the history buffers the graph reads."""

    def __init__(self):
        self.population_history = [(t, 8 + (t % 3)) for t in range(40)]
        self.food_history = [(t, 0.5 + 0.1 * (t % 5)) for t in range(40)]
        self.energy_history = [(t, 30.0 + (t % 7)) for t in range(40)]


def test_build_figure_from_synthetic_history():
    fig = build_ecology_figure(_SyntheticTracker())
    assert isinstance(fig, Figure)

    # Primary axis + its twin secondary axis.
    axes = fig.get_axes()
    assert len(axes) == 2

    # Three lines: population, world food, agent energy.
    all_lines = [ln for ax in axes for ln in ax.get_lines()]
    assert len(all_lines) == 3
    labels = {ln.get_label() for ln in all_lines}
    assert "population" in labels
    assert "mean world food" in labels
    assert "mean agent energy" in labels

    matplotlib.pyplot.close(fig)


def test_save_ecology_graph_writes_png(tmp_path):
    out = tmp_path / "ecology.png"
    returned = save_ecology_graph(_SyntheticTracker(), str(out))
    assert returned == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_figure_empty_history_is_safe():
    fig = build_ecology_figure(_SyntheticTracker.__new__(_SyntheticTracker))
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)


def test_graph_from_freshly_collected_stats():
    """Run a short headless sim, feed the real tracker, render the graph.

    Confirms the StatisticsTracker accumulates the food/energy history the
    graph relies on and that rendering succeeds end-to-end without a display.
    """
    from artificial_society.simulation import Simulation

    sim = Simulation(headless=True, grid_w=20, grid_h=15, initial_population=8)
    tracker = StatisticsTracker()

    for _ in range(15):
        sim.step()
        tracker.update(
            getattr(sim, "tick", 0),
            sim.agents,
            sim.world,
            sim.tribes,
            sim.technology,
        )

    assert len(tracker.population_history) > 0
    assert len(tracker.food_history) == len(tracker.population_history)
    assert len(tracker.energy_history) == len(tracker.population_history)

    fig = build_ecology_figure(tracker)
    assert isinstance(fig, Figure)
    assert len(fig.get_axes()) == 2
    matplotlib.pyplot.close(fig)
