"""Every-K-ticks centralized RSSM training hook (spec §4.5), via the system registry.

Mirrors the world_regrowth.py shape: a marker class + module-level ``_tick``
registered through ``@register``. The ``getattr`` guard makes this a total no-op
on v1/v2 arms (``sim.rssm_learner`` absent or None) — zero draws, golden-safe.
Ordered at 65: after every agent-affecting builtin (economy=40, technology=50,
evolution=60) but strictly before "stats" (order=70) — test_phase2_systems.py's
``test_tick_order_stats_last_and_disease_before_economy`` pins stats as the last
system ticked every tick, and training does not need to run after that snapshot.
"""

from __future__ import annotations

from artificial_society.systems.registry import register


class RssmLearning:
    """Marker system; all behaviour lives in the registered tick hook."""


def _tick(sim, tick: int) -> None:
    learner = getattr(sim, "rssm_learner", None)
    if learner is not None:
        learner.maybe_train(tick, sim.agents)


@register(name="rssm_learning", order=65, tick=_tick)
def _build(sim):
    return RssmLearning()
