"""A1: parallelising the pilot across seeds must not change a single byte.

Each seed is fully process-isolated and ``PYTHONHASHSEED``-pinned, so running the
seeds concurrently (``workers>1``) must produce per-seed exports byte-identical to
running them serially (``workers=1``) at the same per-worker thread count. This
guards the concurrent-session A1 implementation against any parallelism-induced
nondeterminism.
"""

from __future__ import annotations

import pytest

# This test spawns real run_single subprocesses (full sim). Skip cleanly rather than
# fail with an opaque FileNotFoundError when the sim's heavy deps are unavailable.
pytest.importorskip("torch")
pytest.importorskip("pygame")

from artificial_society.research.run_pilot import run_pilot

# tiny, fast config; threads pinned equal in both runs so the only difference is
# serial-vs-parallel scheduling.
_TINY = dict(ticks=6, grid_w=10, grid_h=8, pop=4, moisture=0.5, threads=1)


def test_parallel_pilot_is_byte_identical_to_serial(tmp_path):
    seeds = [1001, 1002]
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    run_pilot(seeds, outdir=str(serial), workers=1, **_TINY)
    run_pilot(seeds, outdir=str(parallel), workers=2, **_TINY)

    for s in seeds:
        for arm in ("learned", "recombiner"):
            a = (serial / f"{arm}_seed{s}.json").read_bytes()
            b = (parallel / f"{arm}_seed{s}.json").read_bytes()
            assert a == b, f"{arm} seed {s}: parallel output differs from serial"
