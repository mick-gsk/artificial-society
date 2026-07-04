"""The GUI/default path must NOT silently auto-load a stray checkpoint.pkl.

Regression guard for the 'stale root checkpoint auto-loads' footgun: a fresh
Simulation with default flags ignores an existing checkpoint file; loading
happens only on an explicit request (main --resume -> load_checkpoint=True).
"""

from __future__ import annotations

from artificial_society import simulation as sim_mod
from artificial_society.main import _parse_args
from artificial_society.simulation import Simulation

SMALL = dict(headless=True, grid_w=12, grid_h=8, initial_population=4)


def test_resume_flag_defaults_off_and_toggles():
    assert _parse_args([]).resume is False
    assert _parse_args(["--resume"]).resume is True


def test_default_construction_ignores_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(tmp_path / "ckpt.pkl"))
    seed = Simulation(load_checkpoint=False, **SMALL)
    seed.tick = 999
    seed._save_checkpoint()

    # Default flags must start fresh, NOT resume tick=999.
    fresh = Simulation(**SMALL)
    assert fresh.tick == 0


def test_explicit_load_still_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(tmp_path / "ckpt.pkl"))
    seed = Simulation(load_checkpoint=False, **SMALL)
    seed.tick = 777
    seed._save_checkpoint()

    resumed = Simulation(load_checkpoint=True, **SMALL)
    assert resumed.tick == 777
