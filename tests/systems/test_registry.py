"""Unit test for the system registry seam (systems/registry.py).

Locks the contract that lets systems be added without editing simulation.py:
registration order is irrelevant, tick order follows the integer `order`, and a
built system is exposed both as sim.<name> and in the returned bus. Also serves as
the template for per-domain unit tests (fast, isolated, no full Simulation run).
"""
from artificial_society.systems import registry


class _FakeSim:
    pass


def test_register_build_exposes_and_orders(monkeypatch):
    # Isolate the global registry so we don't disturb the built-in systems.
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    # Skip discover() so build_systems doesn't re-import the real modules.
    monkeypatch.setattr(registry, "_discovered", True)
    try:
        ticks = []
        # Register out of order; `order` (not registration/import order) decides ticks.
        registry.register_system("alpha", lambda sim: {"id": "A"}, order=5)
        registry.register_system(
            "beta", lambda sim: {"id": "B"}, order=1,
            tick=lambda sim, t: ticks.append(t),
        )

        sim = _FakeSim()
        systems = registry.build_systems(sim)

        assert set(systems) == {"alpha", "beta"}
        assert sim.alpha == {"id": "A"}       # exposed as attribute
        assert systems["beta"] is sim.beta    # same instance in the bus
        assert [s.name for s in registry.specs()] == ["beta", "alpha"]  # by order

        registry.tick_systems(sim, 7)
        assert ticks == [7]  # only beta opted into ticking
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)
