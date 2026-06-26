"""Smoke test: the package must import on the available interpreter.

Doubles as a guard against Python-3.10-only syntax (PEP 604 ``X | None``
annotations evaluated at runtime) creeping back into modules that omit
``from __future__ import annotations``.
"""


def test_simulation_package_imports():
    import artificial_society.simulation  # noqa: F401


def test_core_modules_import():
    import artificial_society.world  # noqa: F401
    import artificial_society.agents.agent  # noqa: F401
    import artificial_society.agents.brain  # noqa: F401
