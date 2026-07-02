"""Pilot orchestrator — both arms over N paired seeds, optionally parallel across seeds.

For each seed: run the learned arm, read its measured ``n_attempts``, then run the
recombiner arm compute-matched to that count. Each run is a subprocess
(``PYTHONHASHSEED=0``, forced CPU) for clean isolation of the module-global
``DISCOVERY_REGISTRY`` / ``Agent.id_counter`` singletons (Spec M4 / §13).

``--workers N`` runs N seeds concurrently (each seed = a learned→recombiner chain).
The world-update loop dominates runtime and is single-threaded Python, so parallel
seeds scale near-linearly on a multi-core CPU; per-subprocess BLAS/torch threads are
capped (``--threads``) to avoid oversubscription. CPU is forced via
``CUDA_VISIBLE_DEVICES=-1`` (the GPU FP16 path crashes and CPU is faster for these
small nets anyway).

Defaults: n=12 seeds, 8000 ticks. ``--smoke`` runs a fast 1-seed / 500-tick check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_SEEDS = list(range(1001, 1013))  # 12 paired seeds
DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "out")


def _subproc_env(threads: int) -> dict:
    t = str(max(1, threads))
    return {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "CUDA_VISIBLE_DEVICES": "-1",  # force CPU (GPU FP16 bug; CPU faster for tiny nets)
        "OMP_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "NUMEXPR_NUM_THREADS": t,
    }


def run_one_seed(seed, ticks, grid_w, grid_h, pop, moisture, outdir, threads):
    """Run learned then recombiner for one seed. Returns (seed, ok, message)."""
    env = _subproc_env(threads)
    base = [sys.executable, "-m", "artificial_society.research.run_single"]

    learned_out = os.path.join(outdir, f"learned_seed{seed}.json")
    r1 = subprocess.run(
        base
        + [
            "--arm",
            "learned",
            "--seed",
            str(seed),
            "--ticks",
            str(ticks),
            "--grid-w",
            str(grid_w),
            "--grid-h",
            str(grid_h),
            "--pop",
            str(pop),
            "--out",
            learned_out,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if r1.returncode != 0:
        return seed, False, f"learned failed: {r1.stderr[-400:]}"

    with open(learned_out) as f:
        attempts = int(json.load(f)["meta"]["n_attempts"])

    recomb_out = os.path.join(outdir, f"recombiner_seed{seed}.json")
    r2 = subprocess.run(
        base
        + [
            "--arm",
            "recombiner",
            "--seed",
            str(seed),
            "--attempts",
            str(max(1, attempts)),
            "--moisture",
            str(moisture),
            "--out",
            recomb_out,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if r2.returncode != 0:
        return seed, False, f"recombiner failed: {r2.stderr[-400:]}"

    return seed, True, f"attempts={attempts}"


def run_pilot(seeds, ticks, grid_w, grid_h, pop, moisture, outdir, workers, threads):
    os.makedirs(outdir, exist_ok=True)
    print(
        f"[pilot] {len(seeds)} seeds x {ticks} ticks | workers={workers} threads/worker={threads} "
        f"| grid={grid_w}x{grid_h} pop={pop} -> {outdir}",
        flush=True,
    )
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(run_one_seed, s, ticks, grid_w, grid_h, pop, moisture, outdir, threads): s
            for s in seeds
        }
        for fut in as_completed(futs):
            seed, ok, msg = fut.result()
            done += 1
            print(
                f"[pilot] ({done}/{len(seeds)}) seed {seed}: {'OK' if ok else 'FAIL'} — {msg}",
                flush=True,
            )
    print(f"[pilot] done. {len(seeds)} paired seeds -> {outdir}", flush=True)
    print(f"[pilot] next: analyze_gate --outdir {outdir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the Stage-0a pilot (both arms, N seeds).")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--grid-w", type=int, default=30)
    p.add_argument("--grid-h", type=int, default=20)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--moisture", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--workers", type=int, default=1, help="seeds to run concurrently")
    p.add_argument("--threads", type=int, default=2, help="BLAS/torch threads per worker")
    p.add_argument("--smoke", action="store_true", help="1 seed, 500 ticks pipeline check")
    args = p.parse_args(argv)

    if args.smoke:
        seeds = [args.seeds[0]] if args.seeds else [1001]
        run_pilot(
            seeds, 500, 20, 15, 12, args.moisture, args.outdir, workers=1, threads=args.threads
        )
        return

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    run_pilot(
        seeds,
        args.ticks,
        args.grid_w,
        args.grid_h,
        args.pop,
        args.moisture,
        args.outdir,
        args.workers,
        args.threads,
    )


if __name__ == "__main__":
    main()
