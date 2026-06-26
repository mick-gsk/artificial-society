# Tests

Two layers:

- **Determinism contract (core, 🔒 do not edit to pass):** `test_headless.py`
  (headless construction + seed digest), `test_regression_golden.py` (60-tick golden
  trajectory vs `golden_trajectory.json`), `test_characterization.py`, `_util.py`.
  These guard reproducibility across every parallel merge — see `docs/ownership.md`.

- **Per-domain unit tests (your lane):** `tests/agents/`, `tests/environment/`,
  `tests/systems/`, `tests/visualization/`. Add fast, isolated tests for your module
  here so you can verify locally without leaning on the slow full-sim run. Keep them
  deterministic: seed via `artificial_society.rng.seed_all`, run headless.

All tests run headless (`conftest.py` forces dummy SDL). Run: `pytest -q`.
