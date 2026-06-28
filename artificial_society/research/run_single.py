"""Run one arm × one seed and export it. Designed to be a subprocess entrypoint.

Usage::

    python -m artificial_society.research.run_single --arm learned \
        --seed 1001 --ticks 8000 --grid-w 30 --grid-h 20 --pop 24 --out learned_1001.json

    python -m artificial_society.research.run_single --arm recombiner \
        --seed 1001 --attempts 9500 --out recombiner_1001.json

The *learned* arm runs the unmodified ``Simulation`` (no source edits): it drives
``sim.step()`` directly (so the per-tick series is uncapped, unlike
``StatisticsTracker``), counts ``combine_vectors`` calls via
:func:`research.instrument.count_combine_calls`, and exports the global
``DISCOVERY_REGISTRY`` lineage. The *recombiner* arm is compute-matched to the
learned arm's measured attempt count (passed via ``--attempts``).
"""

from __future__ import annotations

import argparse
import os

# Headless safety for batch/remote hosts (no window/audio). Must precede the
# simulation import, which pulls in pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def run_learned(
    seed: int, ticks: int, grid_w: int, grid_h: int, pop: int
) -> tuple[dict, list, list]:
    from artificial_society.environment.materials import DISCOVERY_REGISTRY
    from artificial_society.research.instrument import count_combine_calls, quiet_stdout
    from artificial_society.simulation import Simulation

    with quiet_stdout():
        sim = Simulation(
            headless=True,
            load_checkpoint=False,
            seed=seed,
            grid_w=grid_w,
            grid_h=grid_h,
            initial_population=pop,
        )

    series: list[dict] = []
    with count_combine_calls() as cc, quiet_stdout():
        for t in range(ticks):
            sim.step()
            alive = [a for a in sim.agents if a.alive]
            series.append(
                {
                    "tick": t,
                    "n_attempts": cc.n,
                    "n_discoveries": len(DISCOVERY_REGISTRY.entries),
                    "pop": len(alive),
                    "energy": round(float(sum(a.energy for a in alive)), 3),
                }
            )

    entries = list(DISCOVERY_REGISTRY.entries)
    meta = {
        "arm": "learned",
        "seed": seed,
        "ticks": ticks,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "pop": pop,
        "n_attempts": cc.n,
        "n_discoveries": len(entries),
    }
    return meta, series, entries


def run_recombiner_arm(seed: int, attempts: int, moisture: float) -> tuple[dict, list, list]:
    from artificial_society.research.recombiner import run_recombiner

    checkpoint = max(1, attempts // 200)
    entries, series = run_recombiner(
        seed=seed, n_attempts=attempts, moisture=moisture, checkpoint_every=checkpoint
    )
    meta = {
        "arm": "recombiner",
        "seed": seed,
        "n_attempts": attempts,
        "moisture": moisture,
        "n_discoveries": len(entries),
    }
    return meta, series, entries


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run one research arm and export it.")
    p.add_argument("--arm", choices=["learned", "recombiner"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--ticks", type=int, default=8000, help="learned arm only")
    p.add_argument("--attempts", type=int, default=0, help="recombiner arm only")
    p.add_argument("--grid-w", type=int, default=30)
    p.add_argument("--grid-h", type=int, default=20)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--moisture", type=float, default=0.5)
    args = p.parse_args(argv)

    from artificial_society.research.export import dump_run

    if args.arm == "learned":
        meta, series, entries = run_learned(
            args.seed, args.ticks, args.grid_w, args.grid_h, args.pop
        )
    else:
        if args.attempts <= 0:
            p.error("--attempts must be > 0 for the recombiner arm")
        meta, series, entries = run_recombiner_arm(args.seed, args.attempts, args.moisture)

    dump_run(args.out, meta, series, entries)
    print(
        f"[run_single] arm={meta['arm']} seed={meta['seed']} "
        f"attempts={meta['n_attempts']} discoveries={meta['n_discoveries']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
