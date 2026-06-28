"""Pilot orchestrator — runs both arms over N paired seeds as subprocesses.

For each seed:
  1. run the learned arm (subprocess, ``PYTHONHASHSEED=0``) -> ``learned_seed<S>.json``;
  2. read its measured ``n_attempts``;
  3. run the recombiner arm matched to that attempt count -> ``recombiner_seed<S>.json``.

Subprocess-per-run gives clean process isolation, which also resets the
module-global ``DISCOVERY_REGISTRY`` and ``Agent.id_counter`` singletons between
runs and pins ``PYTHONHASHSEED`` for byte-reproducible seeded runs (Spec M4 / §13).

Defaults: n=12 seeds, 8000 ticks (the confirmed pilot scope). ``--smoke`` runs a
fast 1-seed / 500-tick pipeline check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_SEEDS = list(range(1001, 1013))  # 12 paired seeds
DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "out")


def _run(cmd: list[str], env: dict) -> None:
    print(f"[pilot] $ {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def run_pilot(
    seeds: list[int],
    ticks: int,
    grid_w: int,
    grid_h: int,
    pop: int,
    moisture: float,
    outdir: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
    }
    base = [sys.executable, "-m", "artificial_society.research.run_single"]

    for s in seeds:
        learned_out = os.path.join(outdir, f"learned_seed{s}.json")
        _run(
            base
            + [
                "--arm",
                "learned",
                "--seed",
                str(s),
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
            env,
        )
        with open(learned_out) as f:
            attempts = int(json.load(f)["meta"]["n_attempts"])

        recombiner_out = os.path.join(outdir, f"recombiner_seed{s}.json")
        _run(
            base
            + [
                "--arm",
                "recombiner",
                "--seed",
                str(s),
                "--attempts",
                str(max(1, attempts)),
                "--moisture",
                str(moisture),
                "--out",
                recombiner_out,
            ],
            env,
        )

    print(f"[pilot] done. {len(seeds)} paired seeds -> {outdir}")
    print("[pilot] next: python -m artificial_society.research.analyze_gate --outdir " + outdir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the Stage-0a pilot (both arms, N seeds).")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--grid-w", type=int, default=30)
    p.add_argument("--grid-h", type=int, default=20)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--moisture", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--smoke", action="store_true", help="1 seed, 500 ticks pipeline check")
    args = p.parse_args(argv)

    if args.smoke:
        seeds = [args.seeds[0]] if args.seeds else [1001]
        run_pilot(
            seeds,
            ticks=500,
            grid_w=20,
            grid_h=15,
            pop=12,
            moisture=args.moisture,
            outdir=args.outdir,
        )
        return

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    run_pilot(seeds, args.ticks, args.grid_w, args.grid_h, args.pop, args.moisture, args.outdir)


if __name__ == "__main__":
    main()
