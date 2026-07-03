"""Tests for the analysis-overlay layers (Task C3).

CPU, deterministic. Complements ``test_frame.py`` / ``test_frame_v2.py`` /
``test_frame_moist_ash.py``.

Server-computed overlays (``temperature`` / ``danger`` / ``disease``) ship only
when a client requested them AND only every 10th tick — otherwise the ``layers``
key is absent so a frame with no request is byte-identical to a plain v2 frame
(the client caches the last array between refreshes).
"""

from __future__ import annotations

import json

import numpy as np

from artificial_society.serve.frame import build_frame
from artificial_society.simulation import Simulation


def _sim(ticks: int = 3, w: int = 12, h: int = 8, pop: int = 6) -> Simulation:
    sim = Simulation(
        headless=True,
        seed=1,
        grid_w=w,
        grid_h=h,
        initial_population=pop,
        load_checkpoint=False,
    )
    for _ in range(ticks):
        sim.step()
    sim.stats.update(sim.tick, sim.agents, sim.world, sim.tribes, sim.technology)
    return sim


# ---------------------------------------------------------------------------
# no request -> no key -> byte-identical to a plain frame
# ---------------------------------------------------------------------------


def test_no_layers_key_without_request():
    sim = _sim()
    sim.tick = 10  # a multiple of 10 (the gate is not the reason a key is absent)
    f = build_frame(sim)
    assert "layers" not in f


def test_frame_byte_identical_when_no_layer_names():
    """A frame built with the default empty ``layer_names`` must equal one built
    with no ``layer_names`` argument at all — the overlay path adds nothing."""
    sim = _sim()
    sim.tick = 20
    f_default = build_frame(sim)
    f_empty = build_frame(sim, layer_names=frozenset())
    assert f_default == f_empty
    assert "layers" not in f_default


# ---------------------------------------------------------------------------
# gate: names present but tick%10 != 0 -> still no key
# ---------------------------------------------------------------------------


def test_layers_absent_when_tick_not_multiple_of_10():
    sim = _sim()
    sim.tick = 13  # not a multiple of 10
    f = build_frame(sim, layer_names=frozenset({"temperature", "danger"}))
    assert "layers" not in f


def test_layers_gate_at_tick_zero_counts_as_multiple_of_10():
    sim = _sim()
    sim.tick = 0  # 0 % 10 == 0
    f = build_frame(sim, layer_names=frozenset({"danger"}))
    assert "layers" in f
    assert "danger" in f["layers"]


# ---------------------------------------------------------------------------
# gate open: quantized lists, length w*h, values 0..9
# ---------------------------------------------------------------------------


def test_layers_present_quantized_when_tick_multiple_of_10():
    sim = _sim()
    n = sim.world.width * sim.world.height
    sim.tick = 20
    f = build_frame(sim, layer_names=frozenset({"temperature", "danger", "disease"}))
    assert "layers" in f
    layers = f["layers"]
    assert set(layers.keys()) == {"temperature", "danger", "disease"}
    for name, arr in layers.items():
        assert len(arr) == n, name
        assert all(isinstance(v, int) for v in arr), name
        assert all(0 <= v <= 9 for v in arr), name


def test_danger_disease_quantization_matches_field_over_ten():
    sim = _sim()
    sim.tick = 30
    # Inject a known danger/disease pattern to pin the exact /10 formula.
    sim.world.F["danger"][:] = 0.0
    sim.world.F["danger"][0, 0] = 45.0  # rint(4.5) -> 4 (banker's rounding)
    sim.world.F["danger"][0, 1] = 100.0  # clipped to 9
    sim.world.F["disease"][:] = 0.0
    sim.world.F["disease"][1, 1] = 76.0  # rint(7.6) -> 8
    f = build_frame(sim, layer_names=frozenset({"danger", "disease"}))
    danger = f["layers"]["danger"]
    disease = f["layers"]["disease"]
    expected_danger = np.clip(np.rint(sim.world.F["danger"] / 10.0), 0, 9)
    expected_disease = np.clip(np.rint(sim.world.F["disease"] / 10.0), 0, 9)
    assert danger == expected_danger.astype(int).ravel().tolist()
    assert disease == expected_disease.astype(int).ravel().tolist()
    # Spot-check the pinned cells (row-major flatten: index = y*w + x).
    w = sim.world.width
    assert danger[0 * w + 0] == 4
    assert danger[0 * w + 1] == 9
    assert disease[1 * w + 1] == 8


def test_temperature_quantization_matches_shifted_formula():
    sim = _sim()
    sim.tick = 40
    sim.world.F["temperature"][:] = 20.0  # (20+20)/8 = 5
    sim.world.F["temperature"][0, 0] = -20.0  # clipped to 0
    sim.world.F["temperature"][0, 1] = 52.0  # (52+20)/8 = 9
    f = build_frame(sim, layer_names=frozenset({"temperature"}))
    temp = f["layers"]["temperature"]
    expected = np.clip(np.rint((sim.world.F["temperature"] + 20.0) / 8.0), 0, 9)
    assert temp == expected.astype(int).ravel().tolist()
    w = sim.world.width
    assert temp[0 * w + 0] == 0
    assert temp[0 * w + 1] == 9
    assert temp[2 * w + 2] == 5  # an untouched cell stays at the uniform 20 -> 5


# ---------------------------------------------------------------------------
# unknown names / missing fields
# ---------------------------------------------------------------------------


def test_unknown_layer_names_silently_ignored():
    sim = _sim()
    sim.tick = 20
    f = build_frame(sim, layer_names=frozenset({"temperature", "bogus", "food"}))
    assert set(f["layers"].keys()) == {"temperature"}


def test_layers_key_omitted_when_only_unknown_names_requested():
    sim = _sim()
    sim.tick = 20
    f = build_frame(sim, layer_names=frozenset({"bogus", "nonsense"}))
    assert "layers" not in f


# ---------------------------------------------------------------------------
# read-only / determinism / json-safety
# ---------------------------------------------------------------------------


def test_layers_read_only_double_call_identical():
    sim = _sim()
    sim.tick = 50
    f1 = build_frame(sim, layer_names=frozenset({"danger", "disease", "temperature"}))
    f2 = build_frame(sim, layer_names=frozenset({"danger", "disease", "temperature"}))
    assert f1 == f2


def test_frame_json_serializable_with_layers():
    sim = _sim()
    sim.tick = 60
    f = build_frame(sim, layer_names=frozenset({"temperature", "danger", "disease"}))
    assert "layers" in f
    json.dumps(f)  # no numpy scalars leaking through
