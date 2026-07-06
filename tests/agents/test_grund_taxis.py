"""Grund-Taxis (Architektur A2) — angeborene symmetrische Nahrungs-/Partner-Taxis.

Deckt den Testplan aus `docs/superpowers/specs/2026-07-06-grund-taxis-design.md`
§5 (i)+(ii) + die beiden bindenden Review-Auflagen:
  (i)  Unit-Liveness: hungrig->Nahrung, satt+paarungsbereit->Partner, sonst Ruhe,
       Priorität Nahrung>Partner, Determinismus (kein RNG-Draw).
  (ii) Lern-Integrität: der PPO-Buffer speichert weiter die GESAMPELTE Move-Aktion
       + ihr log_prob (nicht die Taxis-Bewegung); Move-Dims 0/1 sind welt-inert.
  (iii) Whitebox: innate_locomotion liest weder `action` noch `rng`/`random`.
  (iv) v1-Pfad + Gate: bei physics_v2=False / taxis AUS bleibt primitive_move.
"""

from __future__ import annotations

import inspect
import random

import numpy as np
import torch

from artificial_society.agents.agent import (
    TAXIS_HUNGER_THRESHOLD,
    Agent,
)


# --------------------------------------------------------------------------- #
# Hilfen                                                                        #
# --------------------------------------------------------------------------- #
class FakeWorld:
    """Minimal-Welt: nur die von innate_locomotion genutzte Oberfläche."""

    def __init__(self, width, height, food=None, passable=None):
        self.width = width
        self.height = height
        self._food = dict(food or {})
        self._passable = dict(passable or {})

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x, y):
        return {
            "food": self._food.get((x, y), 0.0),
            "passable": self._passable.get((x, y), True),
        }


def _make_agent(x, y, sex="f", energy=100.0, age=100, sense_radius=3):
    a = Agent.spawn_random(x, y)
    a.sex = sex
    a.energy = energy
    a.age = age
    a.alive = True
    a.pregnant = False
    a.reproduction_cooldown = 0
    a.is_sleeping = False
    a.trust = {}
    a.genes["sense_radius"] = sense_radius
    return a


def _cheby(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


# --------------------------------------------------------------------------- #
# (i) Unit-Liveness                                                            #
# --------------------------------------------------------------------------- #
def test_food_taxis_step():
    """Hungriger Agent, EINE Food-Zelle NO im sense_radius -> Distanz sinkt strikt."""
    world = FakeWorld(20, 20, food={(7, 7): 25.0})
    a = _make_agent(5, 5, energy=50.0, sense_radius=3)  # 50 < TAXIS_HUNGER_THRESHOLD
    before = _cheby(a.pos, (7, 7))
    a.innate_locomotion(world, [a])
    after = _cheby(a.pos, (7, 7))
    assert after < before, f"erwartet Annäherung an Food, {before}->{after}"
    assert a.pos == (6, 6), "genau ein diagonaler Schritt Richtung Food"


def test_mate_taxis_step():
    """Satt + paarungsbereit, ein kompatibler Partner in MATE_SEEK_RADIUS -> Annäherung."""
    a = _make_agent(5, 5, sex="f", energy=100.0, age=100)  # >= TAXIS_HUNGER_THRESHOLD
    b = _make_agent(5, 9, sex="m", energy=100.0, age=100)
    world = FakeWorld(20, 20)  # keine Food-Zellen
    assert a.can_reproduce() and b.can_reproduce()
    before = _cheby(a.pos, b.pos)
    a.innate_locomotion(world, [a, b])
    after = _cheby(a.pos, b.pos)
    assert after <= before  # nicht-steigend (robust gg. Grenz-Oszillation)
    assert after < before  # hier strikt (Distanz 4 > 0)
    assert a.pos == (5, 6)


def test_sated_rests():
    """Satt (energy >= Schwelle) und NICHT paarungsbereit -> pos unverändert."""
    a = _make_agent(5, 5, energy=TAXIS_HUNGER_THRESHOLD + 10.0, age=10)  # age<repro
    assert not a.can_reproduce()
    world = FakeWorld(20, 20, food={(7, 7): 25.0})  # Food da, aber nicht hungrig
    a.innate_locomotion(world, [a])
    assert a.pos == (5, 5)


def test_priority_food_over_mate():
    """Hungrig UND paarungsbereit (energy in [60,80)) -> bewegt sich zur Nahrung."""
    # energy=70: hungrig (70 < 80) UND can_reproduce (70 >= 60)
    a = _make_agent(5, 5, sex="f", energy=70.0, age=100, sense_radius=3)
    b = _make_agent(5, 1, sex="m", energy=100.0, age=100)  # Partner im Süden
    world = FakeWorld(20, 20, food={(8, 5): 25.0})  # Food im Osten
    assert a.can_reproduce()
    a.innate_locomotion(world, [a, b])
    # Food-Priorität: Schritt nach Osten (+x), nicht nach Süden (-y) zum Partner
    assert a.pos == (6, 5), f"erwartet Food-Schritt (6,5), war {a.pos}"


def test_determinism_no_rng():
    """innate_locomotion zieht KEIN random.*/np.random und ist seed-reproduzierbar."""
    def run():
        # Agent/Welt VOR der RNG-Momentaufnahme bauen (spawn_random zieht RNG);
        # gemessen wird nur der innate_locomotion-Batch selbst.
        world = FakeWorld(20, 20, food={(9, 9): 30.0})
        a = _make_agent(5, 5, energy=40.0, sense_radius=6)
        py_state = random.getstate()
        np_state = np.random.get_state()
        traj = []
        for _ in range(4):
            a.innate_locomotion(world, [a])
            traj.append(a.pos)
        # RNG-Zustand nach dem Batch unverändert (kein Draw in der Taxis)
        assert random.getstate() == py_state
        assert np.array_equal(np.random.get_state()[1], np_state[1])
        return traj

    t1 = run()
    t2 = run()
    assert t1 == t2, "gleiche Startbedingung -> identische Trajektorie"
    assert t1[-1] == (9, 9)  # erreicht die Food-Zelle


def test_mate_tie_break_deterministic():
    """Bei Distanzgleichstand gewinnt der kleinere agent.id (deterministisch)."""
    a = _make_agent(5, 5, sex="f", energy=100.0, age=100)
    left = _make_agent(3, 5, sex="m", energy=100.0, age=100)  # dist 2
    right = _make_agent(7, 5, sex="m", energy=100.0, age=100)  # dist 2
    # gleiche Chebyshev-Distanz -> kleinere id gewinnt
    lo = left if left.id < right.id else right
    world = FakeWorld(20, 20)
    a.innate_locomotion(world, [a, left, right])
    step_x = a.pos[0] - 5
    assert step_x == (1 if lo.pos[0] > 5 else -1)


# --------------------------------------------------------------------------- #
# (ii) Lern-Integrität — Buffer bleibt on-policy                               #
# --------------------------------------------------------------------------- #
def _sim_v2(seed=42):
    from artificial_society.simulation import Simulation

    return Simulation(
        headless=True,
        load_checkpoint=False,
        physics_v2=True,
        grid_w=20,
        grid_h=15,
        initial_population=8,
        seed=seed,
    )


def test_buffer_records_sampled_move():
    """Der Rollout-Buffer trägt die GESAMPELTE Move-Aktion + log_prob, NICHT die Taxis.

    Beweist: ausgeführte Taxis-Bewegung wird nicht als Aktion verbucht — kein
    `ausgeführt != gespeichert`-Mismatch (der Option-B-Defekt)."""
    from artificial_society.agents.brain import Brain

    captured = {}
    orig = Brain.act_v2

    def spy(self, *a, **k):
        step = orig(self, *a, **k)
        captured[id(self)] = step
        return step

    Brain.act_v2 = spy
    Agent.taxis_enabled = True
    try:
        sim = _sim_v2()
        sim.step()
    finally:
        Brain.act_v2 = orig
        Agent.taxis_enabled = False

    checked = 0
    for agent in sim.agents:
        step = captured.get(id(agent.brain))
        if step is None or len(agent.brain.rollout.storage) == 0:
            continue
        stored = agent.brain.rollout.storage[-1]
        sampled_act = step["action_tensor"].detach().squeeze(0)
        # Move-Dims 0/1: gespeichert == gesampelt (nicht die Taxis-Richtung)
        assert torch.equal(stored["action"][:2], sampled_act[:2])
        assert torch.equal(stored["action"], sampled_act)
        assert torch.equal(stored["log_prob"], step["log_prob"].detach().squeeze(0))
        checked += 1
    assert checked > 0, "mindestens ein Agent muss eine Transition gespeichert haben"


def test_move_dims_have_no_world_effect():
    """Gegensätzliche gesampelte Move-Dims -> identische resultierende Positionen.

    Belegt die Inertheit der Move-Dims (Basis des Null-Bias-Beweises §2/A2)."""
    from artificial_society.agents.brain import Brain

    orig = Brain.act_v2
    forced = {"val": 0.9}

    def make_spy():
        def spy(self, *a, **k):
            step = orig(self, *a, **k)
            v = forced["val"]
            step["action_tensor"][0, 0] = v
            step["action_tensor"][0, 1] = v
            lst = list(step["action_list"])
            lst[0] = v
            lst[1] = v
            step["action_list"] = lst
            return step

        return spy

    def run(move_val):
        Brain.act_v2 = make_spy()
        forced["val"] = move_val
        Agent.taxis_enabled = True
        try:
            sim = _sim_v2(seed=123)
            sim.step()
            return [tuple(a.pos) for a in sim.agents]
        finally:
            Brain.act_v2 = orig
            Agent.taxis_enabled = False

    pos_plus = run(+0.9)  # Move-Dims -> +x,+y
    pos_minus = run(-0.9)  # Move-Dims -> -x,-y
    assert pos_plus == pos_minus, "Lokomotion hängt nur an der Taxis, nicht an Dims 0/1"


# --------------------------------------------------------------------------- #
# (iii) Whitebox — keine action-/RNG-Leserei                                   #
# --------------------------------------------------------------------------- #
def test_innate_locomotion_signature_and_no_rng():
    sig = inspect.signature(Agent.innate_locomotion)
    assert list(sig.parameters) == ["self", "world", "agents"], (
        "Signatur nimmt kein `action`/`brain_step` entgegen"
    )
    src = "".join(
        inspect.getsource(m)
        for m in (
            Agent.innate_locomotion,
            Agent._nearest_food_cell,
            Agent._nearest_compatible_mate,
        )
    )
    for forbidden in ("random", "np.random", "brain_step", "action_list", "action_tensor"):
        assert forbidden not in src, f"innate_locomotion darf `{forbidden}` nicht referenzieren"
    # liest die gesampelte Move-Aktion nicht
    assert "action[" not in src and '["move_x"]' not in src and '["move_y"]' not in src


# --------------------------------------------------------------------------- #
# (iv) v1-Pfad + Gate unberührt                                                #
# --------------------------------------------------------------------------- #
def test_gate_v1_uses_primitive_move():
    """physics_v2=False -> Gate ruft primitive_move, NICHT innate_locomotion."""
    calls = {"prim": 0, "taxis": 0}
    orig_prim = Agent.primitive_move
    orig_taxis = Agent.innate_locomotion

    def spy_prim(self, world, action):
        calls["prim"] += 1
        return orig_prim(self, world, action)

    def spy_taxis(self, world, agents):
        calls["taxis"] += 1
        return orig_taxis(self, world, agents)

    Agent.primitive_move = spy_prim
    Agent.innate_locomotion = spy_taxis
    Agent.taxis_enabled = True  # selbst bei taxis AN darf v1 sie nicht ziehen
    try:
        from artificial_society.simulation import Simulation

        sim = Simulation(
            headless=True,
            load_checkpoint=False,
            physics_v2=False,
            grid_w=20,
            grid_h=15,
            initial_population=6,
            seed=42,
        )
        sim.step()
    finally:
        Agent.primitive_move = orig_prim
        Agent.innate_locomotion = orig_taxis
        Agent.taxis_enabled = False
    assert calls["prim"] > 0, "v1-Pfad muss primitive_move nutzen"
    assert calls["taxis"] == 0, "v1-Pfad darf innate_locomotion nie aufrufen"


def test_gate_v2_taxis_off_uses_primitive_move():
    """physics_v2=True aber taxis AUS -> primitive_move (bisheriges v2-Verhalten)."""
    calls = {"prim": 0, "taxis": 0}
    orig_prim = Agent.primitive_move
    orig_taxis = Agent.innate_locomotion

    def spy_prim(self, world, action):
        calls["prim"] += 1
        return orig_prim(self, world, action)

    def spy_taxis(self, world, agents):
        calls["taxis"] += 1
        return orig_taxis(self, world, agents)

    Agent.primitive_move = spy_prim
    Agent.innate_locomotion = spy_taxis
    Agent.taxis_enabled = False
    try:
        sim = _sim_v2(seed=42)
        sim.step()
    finally:
        Agent.primitive_move = orig_prim
        Agent.innate_locomotion = orig_taxis
    assert calls["prim"] > 0
    assert calls["taxis"] == 0, "taxis AUS: innate_locomotion darf nicht feuern"


def test_gate_v2_taxis_on_uses_innate():
    """physics_v2=True und taxis AN -> innate_locomotion feuert, primitive_move nicht."""
    calls = {"prim": 0, "taxis": 0}
    orig_prim = Agent.primitive_move
    orig_taxis = Agent.innate_locomotion

    def spy_prim(self, world, action):
        calls["prim"] += 1
        return orig_prim(self, world, action)

    def spy_taxis(self, world, agents):
        calls["taxis"] += 1
        return orig_taxis(self, world, agents)

    Agent.primitive_move = spy_prim
    Agent.innate_locomotion = spy_taxis
    Agent.taxis_enabled = True
    try:
        sim = _sim_v2(seed=42)
        sim.step()
    finally:
        Agent.primitive_move = orig_prim
        Agent.innate_locomotion = orig_taxis
        Agent.taxis_enabled = False
    assert calls["taxis"] > 0, "taxis AN: innate_locomotion muss feuern"
    assert calls["prim"] == 0, "taxis AN im v2-Pfad: primitive_move darf nicht feuern"
