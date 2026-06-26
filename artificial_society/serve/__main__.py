"""Launch the dashboard server: ``python -m artificial_society.serve``.

Binds 0.0.0.0:8000 so the PC is reachable from the MacBook over the LAN.
For reproducible seeded runs, set ``PYTHONHASHSEED=0`` *before* starting the
process (the run-dashboard.bat / .sh wrappers do this).
"""
from __future__ import annotations

import argparse
import os


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="artificial_society.serve")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if os.environ.get("PYTHONHASHSEED") != "0":
        print("[serve] warning: PYTHONHASHSEED != 0 — seeded runs may not be byte-reproducible.")

    import uvicorn

    uvicorn.run("artificial_society.serve.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
