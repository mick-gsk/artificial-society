"""D2 Registry-Integrations-Smoke-Test: Objekt-Schicht läuft nachweislich im Sim-Tick."""

from __future__ import annotations

import pytest

from artificial_society.environment.physics.actions import DECAY_RATE
from artificial_society.environment.physics.objects import make_object
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=30, grid_h=20, initial_population=8)


def test_v2_sim_registriert_system_und_seedet_objekte():
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    assert "physics_v2_objects" in sim.systems
    assert sim.world.objects.total_mass() > 0.0, "Start-Seeding muss Objekte liefern"
    assert sim.world.objects.ledger["spawned"] == pytest.approx(sim.world.objects.total_mass())
    for _ in range(3):
        sim.step()  # Spawning/Verwesung laufen im Tick, ohne zu crashen


def test_verwesung_laeuft_ueber_den_sim_tick():
    """Kein direkter Mechanik-Aufruf: der Massenverlust muss aus sim.step() kommen."""
    sim = Simulation(seed=42, physics_v2=True, **_PARAMS)
    kadaver = make_object("carcass", 50.0)
    sim.world.objects.add(kadaver, (5, 5), source="from_carcass")
    sim.step()
    assert kadaver.mass == pytest.approx(50.0 * (1.0 - DECAY_RATE))


def test_gegenprobe_flag_aus_system_wirkt_nicht():
    sim = Simulation(seed=42, physics_v2=False, **_PARAMS)
    assert "physics_v2_objects" in sim.systems  # registriert, aber inert
    assert sim.world.objects.total_mass() == 0.0  # kein Seeding
    kadaver = make_object("carcass", 50.0)
    sim.world.objects.add(kadaver, (5, 5), source="from_carcass")
    for _ in range(3):
        sim.step()
    assert kadaver.mass == 50.0  # keine Verwesung
    boden = [o for o, _ in sim.world.objects.all_objects()]
    assert boden == [kadaver]  # kein Spawning
