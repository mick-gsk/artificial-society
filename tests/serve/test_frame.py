"""Tests for the live-frame serializer (serve/frame.py). CPU, deterministic."""

from __future__ import annotations

import re

from artificial_society.serve.frame import biome_legend, build_frame
from artificial_society.simulation import Simulation

HEX = re.compile(r"^#[0-9a-f]{6}$")


def _sim(ticks: int = 3) -> Simulation:
    sim = Simulation(
        headless=True,
        seed=1,
        grid_w=12,
        grid_h=8,
        initial_population=6,
        load_checkpoint=False,
    )
    for _ in range(ticks):
        sim.step()
    sim.stats.update(sim.tick, sim.agents, sim.world, sim.tribes, sim.technology)
    return sim


def test_frame_shape_and_quantized_cells():
    f = build_frame(_sim())
    assert f["type"] == "frame"
    assert f["grid"] == {"w": 12, "h": 8}
    n = 12 * 8
    for key in ("food", "water", "biome"):
        assert len(f["cells"][key]) == n, key
    # scalars are quantized to ints to keep frames small
    assert all(isinstance(v, int) for v in f["cells"]["food"])
    assert all(isinstance(v, int) for v in f["cells"]["water"])


def test_frame_agents_within_bounds_and_colored():
    f = build_frame(_sim())
    legend_size = len(biome_legend())
    assert all(0 <= b < legend_size for b in f["cells"]["biome"])
    for a in f["agents"]:
        assert 0 <= a["x"] < 12 and 0 <= a["y"] < 8
        assert HEX.match(a["col"]), a["col"]
        assert a["st"] in (0, 1, 2)
    # the cards still read the aggregate stats block
    assert f["stats"]["tick"] == f["tick"]


def test_biome_legend_stable_and_indexed():
    leg = biome_legend()
    assert [e["idx"] for e in leg] == list(range(len(leg)))
    assert all(len(e["rgb"]) == 3 for e in leg)
    # deterministic order across calls so the client can cache it
    assert [e["name"] for e in biome_legend()] == [e["name"] for e in leg]
