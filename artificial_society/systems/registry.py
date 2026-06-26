"""System registry — add a simulation system without editing ``simulation.py``.

A system module registers itself at import time via the ``@register(...)``
decorator on a small factory. ``Simulation`` calls :func:`discover` (import every
managed module so registrations fire), then :func:`build_systems` (construct each
system, expose it as ``sim.<name>`` and in ``sim.systems``), and
:func:`tick_systems` from ``step()`` (tick the ones with a ``tick`` hook).

Why this exists: previously every new system had to be wired into ``simulation.py``
in four places (import, ``__init__``, the ``agent.update`` call, the tick loop), so
two agents adding systems in parallel always collided. With the registry, **adding a
system is one new file** under ``systems/`` — no shared edit, no merge conflict.

Determinism: systems are ordered solely by their integer ``order`` (ties broken by
``name``), never by import order, so a given seed always ticks them identically.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable

# A `tick` of None means "construct only" (dormant) — the system is available as
# sim.<name> / in sim.systems but is not ticked from the loop. Existing systems are
# dormant today; this preserves the current behaviour exactly (see TODO(phase2) in
# simulation.step). A system that wants to tick supplies tick=<callable(sim, tick)>.
TickHook = Callable[[Any, int], None]
Factory = Callable[[Any], Any]


@dataclass(frozen=True)
class SystemSpec:
    name: str
    factory: Factory
    order: int = 100
    tick: TickHook | None = None


_REGISTRY: dict[str, SystemSpec] = {}
_discovered = False


def register_system(
    name: str, factory: Factory, *, order: int = 100, tick: TickHook | None = None
) -> None:
    """Register (or replace) a system. Idempotent across module re-imports."""
    _REGISTRY[name] = SystemSpec(name=name, factory=factory, order=order, tick=tick)


def register(*, name: str, order: int = 100, tick: TickHook | None = None):
    """Decorator form. Decorate a factory ``f(sim) -> system_instance``::

    @register(name="trade", order=45)
    def _build(sim):
        return TradeSystem()
    """

    def decorator(factory: Factory) -> Factory:
        register_system(name, factory, order=order, tick=tick)
        return factory

    return decorator


def clear() -> None:
    """Reset the registry (tests only)."""
    global _discovered
    _REGISTRY.clear()
    _discovered = False


def specs() -> list[SystemSpec]:
    """Registered specs in deterministic tick order."""
    return sorted(_REGISTRY.values(), key=lambda s: (s.order, s.name))


# Packages auto-imported so a brand-new module's @register fires without anyone
# editing simulation.py. The registry module itself is skipped (it has nothing to
# register and importing it from within discovery would be circular).
_DISCOVER_PACKAGES = ("artificial_society.systems",)
_DISCOVER_SKIP = {
    "artificial_society.systems.registry",
}


def discover(force: bool = False) -> None:
    """Import every managed module once so its registration runs.

    A module that fails to import is skipped (never crash the whole simulation for
    one bad system). Already-imported modules are returned from the import cache,
    so this is cheap and side-effect-free to call repeatedly.
    """
    global _discovered
    if _discovered and not force:
        return
    for pkg_name in _DISCOVER_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:
            continue
        for mod in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            if mod.name in _DISCOVER_SKIP:
                continue
            # A system that fails to import simply isn't available; never crash the
            # whole simulation for one bad module.
            with contextlib.suppress(Exception):
                importlib.import_module(mod.name)
    _discovered = True


def build_systems(sim) -> dict:
    """Construct every registered system, set ``sim.<name>``, return the bus dict."""
    discover()
    systems: dict = {}
    for spec in specs():
        instance = spec.factory(sim)
        setattr(sim, spec.name, instance)
        systems[spec.name] = instance
    return systems


def tick_systems(sim, tick: int) -> None:
    """Tick every registered system that opted in, in ascending ``order``."""
    for spec in specs():
        if spec.tick is not None:
            spec.tick(sim, tick)
