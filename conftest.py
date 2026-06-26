"""Pytest bootstrap: run the simulation headless in tests.

- Forces SDL into dummy mode so pygame never opens a real window/audio device.
- Ensures the repo root (containing the ``artificial_society`` package) imports.

Note: cross-process reproducibility additionally requires a fixed
``PYTHONHASHSEED`` (per-process hash randomization changes set/dict iteration
order). Tests that compare against a committed trajectory pin it themselves via
a subprocess (see tests/test_regression_golden.py); within-process determinism
tests do not need it.
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(__file__))
