"""C5 (3a-Anteil): physics_v2 im Checkpoint-Payload; Mismatch re-raised VOR dem broad-except."""

from __future__ import annotations

import pickle

import pytest

import artificial_society.simulation as sim_mod
from artificial_society.simulation import CheckpointIncompatibleError, Simulation

_PARAMS = dict(headless=True, grid_w=20, grid_h=15, initial_population=8)


@pytest.fixture()
def checkpoint_path(tmp_path, monkeypatch):
    pfad = str(tmp_path / "checkpoint.pkl")
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", pfad)
    return pfad


def test_roundtrip_v2_erhaelt_flag_objekte_und_embodiment(checkpoint_path):
    sim1 = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim1.step()
    sim1.step()
    masse = sim1.world.objects.total_mass()
    sim1._save_checkpoint()

    sim2 = Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)
    assert sim2.tick == sim1.tick
    assert sim2.world.objects.total_mass() == pytest.approx(masse)
    assert all(a.physics_v2 and a.body is not None for a in sim2.agents)


def test_flag_mismatch_re_raised_statt_still_verschluckt(checkpoint_path):
    sim1 = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim1._save_checkpoint()
    with pytest.raises(CheckpointIncompatibleError):
        Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)


def test_alt_checkpoint_ohne_key_laedt_nur_mit_flag_aus(checkpoint_path):
    """Legacy-Payload (kein physics_v2-Key) ⇒ implizit False: lädt mit Flag aus,
    kollidiert hart mit Flag an."""
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"agents": [], "tick": 5}, f)
    sim = Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)
    assert sim.tick == 5
    with pytest.raises(CheckpointIncompatibleError):
        Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)


def test_kaputter_checkpoint_startet_weiterhin_frisch(checkpoint_path):
    """Der broad-except-Pfad (korrupte Datei) bleibt erhalten: frisch starten."""
    with open(checkpoint_path, "wb") as f:
        f.write(b"kein pickle")
    sim = Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)
    assert sim.tick == 0 and len(sim.agents) == 8
