"""Tests for the moisture/ash cell fields (Frame-Schema v2, R6).

CPU, deterministic. Complements ``test_frame.py`` / ``test_frame_v2.py``.
``moist`` ships every frame; ``ash`` ships only every 5th tick AND only while
some cell still holds visible ash — otherwise the key is absent so the client
keeps painting its cached last array (falsy-omission, same idiom as ``detail``
and the compact agent behaviour fields).
"""

from __future__ import annotations

import json

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
# moist: present every frame, right shape/type/range
# ---------------------------------------------------------------------------


def test_moist_present_every_frame_with_right_length_and_values():
    sim = _sim()
    n = sim.world.width * sim.world.height
    for tick_offset in range(6):  # spans several tick%5 phases
        sim.step()
        f = build_frame(sim)
        assert "moist" in f["cells"]
        moist = f["cells"]["moist"]
        assert len(moist) == n
        assert all(isinstance(v, int) for v in moist)
        assert all(0 <= v <= 100 for v in moist), tick_offset


def test_moist_matches_world_field_rounded():
    sim = _sim()
    f = build_frame(sim)
    import numpy as np

    expected = np.rint(sim.world.F["moisture"]).astype(int).ravel().tolist()
    assert f["cells"]["moist"] == expected


# ---------------------------------------------------------------------------
# ash: gated on (tick % 5 == 0) AND (max ash > 0.5); else key absent
# ---------------------------------------------------------------------------


def test_ash_absent_when_tick_not_multiple_of_5():
    sim = _sim()
    sim.world.F["ash"][:] = 42.0  # plenty of ash, would pass the max() gate
    sim.tick = 7  # not a multiple of 5
    f = build_frame(sim)
    assert "ash" not in f["cells"]


def test_ash_absent_when_tick_multiple_of_5_but_max_below_threshold():
    sim = _sim()
    sim.world.F["ash"][:] = 0.2  # below the 0.5 threshold
    sim.tick = 10  # multiple of 5
    f = build_frame(sim)
    assert "ash" not in f["cells"]


def test_ash_present_when_tick_multiple_of_5_and_max_above_threshold():
    sim = _sim()
    sim.world.F["ash"][:] = 0.0
    sim.world.F["ash"][0, 0] = 12.0  # single hot cell pushes max() over 0.5
    sim.tick = 15  # multiple of 5 (and tick 0 also qualifies, tested separately)
    f = build_frame(sim)
    assert "ash" in f["cells"]
    n = sim.world.width * sim.world.height
    ash = f["cells"]["ash"]
    assert len(ash) == n
    assert all(isinstance(v, int) for v in ash)
    assert ash[0] == 12


def test_ash_gate_at_tick_zero_counts_as_multiple_of_5():
    sim = _sim()
    sim.world.F["ash"][:] = 5.0
    sim.tick = 0
    f = build_frame(sim)
    assert "ash" in f["cells"]


def test_ash_values_match_world_field_rounded_when_present():
    sim = _sim()
    sim.world.F["ash"][:] = 3.3
    sim.world.F["ash"][1, 1] = 77.6
    sim.tick = 20
    f = build_frame(sim)
    import numpy as np

    expected = np.rint(sim.world.F["ash"]).astype(int).ravel().tolist()
    assert f["cells"]["ash"] == expected


# ---------------------------------------------------------------------------
# read-only / determinism / json-safety
# ---------------------------------------------------------------------------


def test_moist_ash_read_only_double_call_identical():
    sim = _sim()
    sim.world.F["ash"][:] = 5.0
    sim.tick = 25  # multiple of 5, ash present
    f1 = build_frame(sim)
    f2 = build_frame(sim)
    assert f1 == f2


def test_frame_json_serializable_with_moist_and_ash():
    sim = _sim()
    sim.world.F["ash"][:] = 9.0
    sim.tick = 30
    f = build_frame(sim)
    assert "ash" in f["cells"]
    json.dumps(f)  # no numpy scalars leaking through


def test_steady_state_frames_without_ash_omit_key_across_ticks():
    """Steady state (no fire ever happened) -> ash key never appears."""
    sim = _sim(ticks=1)
    for _ in range(12):
        sim.step()
        f = build_frame(sim)
        if float(sim.world.F["ash"].max()) <= 0.5:
            assert "ash" not in f["cells"]
