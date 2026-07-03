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


def test_kaputter_checkpoint_v2_seedet_objekt_schicht_frisch(checkpoint_path):
    """Broad-except-Pfad mit physics_v2=True muss auch das v2-Objekt-Seeding
    nachholen (sonst startet die Welt masselos)."""
    with open(checkpoint_path, "wb") as f:
        f.write(b"kein pickle")
    sim = Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)
    assert sim.tick == 0 and len(sim.agents) == 8
    assert sim.world.objects.total_mass() > 0


def test_unbekannte_format_version_wird_hart_abgewiesen(checkpoint_path):
    """C5 (3b): Version ∉ {1, 2} ⇒ harter Fehler VOR dem broad-except, bei
    beiden Flag-Stellungen — kein stiller Frisch-Start (Datenverlust).
    Version 1 (Legacy, auch ohne Key) lädt dagegen weiter mit Flag aus —
    das pinnt der unveränderte 3a-Test test_alt_checkpoint_ohne_key_...."""
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"format_version": 3, "agents": [], "tick": 5}, f)
    with pytest.raises(CheckpointIncompatibleError, match="format_version"):
        Simulation(seed=3, physics_v2=False, load_checkpoint=True, **_PARAMS)
    with pytest.raises(CheckpointIncompatibleError, match="format_version"):
        Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)


def test_3a_aera_v2_checkpoint_wird_klar_abgewiesen(checkpoint_path):
    """C5-Kern: v2-Checkpoint mit fremden Netz-Formen ⇒ klare Fehlermeldung
    statt stillem [compat]-Rebuild (Gewichtsverlust) oder Forward-Absturz.
    Konstruiert: Version-1-Payload MIT physics_v2=True, aber v1-Brains
    (3a-Ära-Fall) — der Flag-Guard allein würde ihn durchlassen."""
    quelle = Simulation(seed=3, physics_v2=False, load_checkpoint=False, **_PARAMS)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {"agents": quelle.agents, "tick": 3, "physics_v2": True, "world": quelle.world},
            f,
        )
    with pytest.raises(CheckpointIncompatibleError, match="Brain-Formen"):
        Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)


def test_payload_traegt_format_version_2(checkpoint_path):
    sim = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim._save_checkpoint()
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    assert data["format_version"] == 2
    assert data["physics_v2"] is True


def test_roundtrip_erhaelt_v2_brain_formen(checkpoint_path):
    """C5: v2-Brain-Formen überleben den Roundtrip (29 Köpfe, 192er GRU)."""
    sim1 = Simulation(seed=3, physics_v2=True, load_checkpoint=False, **_PARAMS)
    sim1.step()
    sim1._save_checkpoint()
    sim2 = Simulation(seed=3, physics_v2=True, load_checkpoint=True, **_PARAMS)
    assert all(a.brain.physics_v2 for a in sim2.agents)
    assert all(a.brain.gru.input_size == 192 for a in sim2.agents)
    assert all(a.brain.policy_mean.out_features == 29 for a in sim2.agents)
    assert all(a.causal_model is not None for a in sim2.agents)
