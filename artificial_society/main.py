"""Entry point for the Artificial Society simulation.

Run with a window (default) or headless for batch / reproducible runs:

    python -m artificial_society.main                          # GUI
    python -m artificial_society.main --headless --seed 42 --ticks 2000

The simulation uses the single explicit ``Simulation.run`` loop directly; the
old bootstrap/monkeypatch ``run`` override is no longer applied here.

Note: a fully reproducible run also needs a fixed hash seed, e.g.
``PYTHONHASHSEED=0 python -m artificial_society.main --headless --seed 42 ...``
(per-process hash randomization otherwise changes set/dict iteration order).
"""
from __future__ import annotations

import argparse

from artificial_society.simulation import Simulation


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="artificial_society")
    p.add_argument("--headless", action="store_true",
                   help="run without opening a window (batch / reproducible runs)")
    p.add_argument("--seed", type=int, default=None, help="seed for a reproducible run")
    p.add_argument("--ticks", type=int, default=None,
                   help="stop after N ticks (default: run until the window closes)")
    p.add_argument("--grid-w", type=int, default=60)
    p.add_argument("--grid-h", type=int, default=40)
    p.add_argument("--pop", type=int, default=36, help="initial population")
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=800)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if not args.headless:
        import pygame
        pygame.init()

    sim = Simulation(
        width=args.width, height=args.height,
        grid_w=args.grid_w, grid_h=args.grid_h,
        initial_population=args.pop,
        headless=args.headless, seed=args.seed,
        # Headless/seeded runs start fresh for reproducibility; GUI resumes a checkpoint.
        load_checkpoint=not args.headless,
    )
    sim.run(max_ticks=args.ticks)

    if not args.headless:
        import pygame
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
