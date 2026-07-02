"""Audit group 6 (fix 15): language convergence is a world-level census.

It used to hang off agent id==1 inside Agent.update — when that agent died,
check_convergence and tick_decay stopped globally for the rest of the run.
Token decay was also wrongly coupled to population >= 2.
"""

from __future__ import annotations

from artificial_society.simulation import Simulation
from artificial_society.systems.language import TOKEN_WORLD


def _sim(pop: int = 5):
    return Simulation(
        headless=True,
        seed=43,
        grid_w=15,
        grid_h=12,
        initial_population=pop,
        load_checkpoint=False,
    )


def test_convergence_runs_without_agent_one(monkeypatch):
    sim = _sim()
    sim.agents = [a for a in sim.agents if a.id != 1]  # agent 1 is gone
    assert len(sim.agents) >= 2

    called = {"conv": 0, "decay": 0}
    monkeypatch.setattr(
        TOKEN_WORLD,
        "check_convergence",
        lambda memories, tick: called.__setitem__("conv", called["conv"] + 1),
    )
    monkeypatch.setattr(
        TOKEN_WORLD, "tick_decay", lambda: called.__setitem__("decay", called["decay"] + 1)
    )

    sim.step()  # tick 0 hits the 90-tick cadence

    assert called["conv"] == 1, "convergence must not depend on agent id 1 being alive"
    assert called["decay"] == 1


def test_token_decay_independent_of_population(monkeypatch):
    sim = _sim()
    sim.agents = sim.agents[:1]  # below the >=2 memories needed for convergence

    called = {"conv": 0, "decay": 0}
    monkeypatch.setattr(
        TOKEN_WORLD,
        "check_convergence",
        lambda memories, tick: called.__setitem__("conv", called["conv"] + 1),
    )
    monkeypatch.setattr(
        TOKEN_WORLD, "tick_decay", lambda: called.__setitem__("decay", called["decay"] + 1)
    )

    sim.step()

    assert called["conv"] == 0
    assert called["decay"] == 1, "token decay must not be coupled to population size"


def test_cadence_is_90_ticks(monkeypatch):
    sim = _sim()
    called = {"decay": 0}
    monkeypatch.setattr(
        TOKEN_WORLD, "tick_decay", lambda: called.__setitem__("decay", called["decay"] + 1)
    )

    sim.step()  # tick 0: fires
    sim.step()  # tick 1: must not fire

    assert called["decay"] == 1
