"""Physik-v2-Objektschicht als registriertes System (Plan 3a, Spec A/B2/B3.4).

Der Pro-Tick-Anteil der Objekt-Schicht (Verwesung, dann Spawning) läuft über
den System-Registry-Seam statt als if-Verzweigung in der Tick-Schleife. Bei
physics_v2=False ist der Tick ein sofortiger No-op (kein RNG-Draw — Golden).
"""

from __future__ import annotations

from artificial_society.environment.phys_objects import tick_decay, tick_spawn
from artificial_society.systems.registry import register_system


class PhysicsV2ObjectsSystem:
    """Zustandsloses System; die Objekt-Schicht selbst lebt an der Welt."""

    def tick(self, sim, tick: int) -> None:
        if not getattr(sim, "physics_v2", False):
            return
        layer = sim.world.objects
        tick_decay(layer)
        tick_spawn(layer, sim.world.biomes)


def _tick(sim, tick: int) -> None:
    sim.physics_v2_objects.tick(sim, tick)


# order 27: nach world_regrowth (25), vor tribes (30) — Objekte altern/erscheinen
# im selben Tick, in dem auch die Zell-Ökologie fortgeschrieben wurde.
register_system("physics_v2_objects", lambda sim: PhysicsV2ObjectsSystem(), order=27, tick=_tick)
