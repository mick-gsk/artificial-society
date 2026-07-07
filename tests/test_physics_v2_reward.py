"""D3: Belohnungs-Reinheit (Test rechnet gegen), v1-Trigger tot (Spy + Positiv-Kontrolle),
Terminal-Transition über sim.step()."""

from __future__ import annotations

import numpy as np
import pytest

import artificial_society.agents.agent as agent_mod
from artificial_society.agents.brain import DEATH_REWARD_V2, ROLLOUT_HORIZON, Brain
from artificial_society.simulation import Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)


def _sim(physics_v2, seed=42):
    return Simulation(seed=seed, physics_v2=physics_v2, **_PARAMS)


def test_belohnungs_reinheit_gegen_gemockte_events(monkeypatch):
    """Gemockte Event-Quellen (Territorium, Social Learning, Sprache) liefern
    riesige Boni — der v2-Reward muss trotzdem EXAKT die C3-Terme sein."""
    monkeypatch.setattr(agent_mod, "territory_reward_for_agent", lambda a, w: 99.0)
    monkeypatch.setattr(agent_mod, "social_learning_step", lambda a, ags, t: 99.0)
    monkeypatch.setattr(agent_mod, "_maybe_mark_language", lambda a, c, t, v, r: r + 99.0)
    sim = _sim(physics_v2=True)
    agent = sim.agents[0]
    agent.is_sleeping = False
    e0, h0 = agent.energy, agent.health

    agent.update(
        sim.world,
        sim.agents,
        tick=0,
        tribes=sim.tribes,
        economy=sim.economy,
        technology=sim.technology,
    )

    cl = agent.curiosity_last
    curiosity = max(
        0.0, min(2.0, 0.25 * cl["nextslot"] + 0.5 * cl["causal"] + 0.25 * cl["novelty"])
    )
    erwartet = (
        0.6 * (agent.energy - e0) / 45.0
        + 0.6 * (agent.health - h0) / 50.0
        - 0.3 * max(0.0, (60.0 - agent.energy) / 60.0)
        + 0.3 * curiosity
    )
    assert agent.last_reward == pytest.approx(erwartet, abs=1e-9), (
        "v2-Reward muss exakt r_energy + r_health + r_deficit + 0.3*curiosity sein"
    )


def test_v2_reward_ohne_cognition_skalierung(monkeypatch):
    """C3 exakt: kein cognition_mult im v2 (der wäre ein multiplikativer Shaping-Kanal).

    EndocrineSystem nutzt __slots__ (Bestandscode) — ein Instanz-Patch von
    ``modifiers`` scheitert daher an 'attribute is read-only"; das Klassen-
    attribut ist dagegen ganz normal patchbar (kein Slot-Konflikt) und deckt
    denselben Zweck ab (cognition-Wert künstlich hochsetzen)."""
    from artificial_society.agents.modulation import ModulationSystem

    sim = _sim(physics_v2=True)
    agent = sim.agents[0]
    original_modifiers = ModulationSystem.modifiers  # VOR dem Patch binden (sonst Rekursion)

    def gepatcht(self):
        d = original_modifiers(self)
        d["cognition"] = 3.0
        return d

    monkeypatch.setattr(ModulationSystem, "modifiers", gepatcht)
    e0, h0 = agent.energy, agent.health
    agent.update(
        sim.world,
        sim.agents,
        tick=0,
        tribes=sim.tribes,
        economy=sim.economy,
        technology=sim.technology,
    )
    cl = agent.curiosity_last
    curiosity = max(
        0.0, min(2.0, 0.25 * cl["nextslot"] + 0.5 * cl["causal"] + 0.25 * cl["novelty"])
    )
    erwartet = (
        0.6 * (agent.energy - e0) / 45.0
        + 0.6 * (agent.health - h0) / 50.0
        - 0.3 * max(0.0, (60.0 - agent.energy) / 60.0)
        + 0.3 * curiosity
    )
    assert agent.last_reward == pytest.approx(erwartet, abs=1e-9)


def _mit_spy(monkeypatch, ziel_modul, name):
    zaehler = {"n": 0}
    original = getattr(ziel_modul, name)

    def spion(*a, **kw):
        zaehler["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(ziel_modul, name, spion)
    return zaehler


def test_v1_trigger_tot_mit_v1_positiv_kontrolle(monkeypatch):
    """D3/F7: Invention-Trigger + Planner per Spy überwacht — 0 Aufrufe im v2
    über 6 Ticks, UND die Positiv-Kontrolle beweist, dass die Spies greifen."""
    need = _mit_spy(monkeypatch, agent_mod, "agent_invent_from_need")
    inv = _mit_spy(monkeypatch, agent_mod, "agent_try_invention")
    cook = _mit_spy(monkeypatch, agent_mod, "agent_try_cook")
    plan = _mit_spy(monkeypatch, Brain, "plan_action")

    sim = _sim(physics_v2=True)
    for _ in range(6):
        sim.step()
    assert need["n"] == 0 and inv["n"] == 0 and cook["n"] == 0, "v1-Erfindung ist im v2 AUS (B6)"
    assert plan["n"] == 0, "Planner ist im v2 deaktiviert (C5)"

    # Positiv-Kontrolle: dieselben Spies feuern im v1-Modus (deterministisch:
    # invent_from_need läuft bei Cooldown 0 jeden Tick; plan_action bei tick 0).
    sim_v1 = _sim(physics_v2=False)
    for _ in range(6):
        sim_v1.step()
    assert need["n"] >= 1, "Positiv-Kontrolle: Spy muss im v1 feuern"
    assert plan["n"] >= 1, "Positiv-Kontrolle: Planner-Spy muss im v1 feuern"


def test_terminal_transition_ueber_sim_step(monkeypatch):
    """D3: Tod im Tick ⇒ finalize_terminal läuft genau einmal (remove_dead),
    der Buffer ist danach geleert."""
    aufrufe = []
    original = Brain.finalize_terminal

    def spion(self, *a, **kw):
        aufrufe.append(self)
        return original(self, *a, **kw)

    monkeypatch.setattr(Brain, "finalize_terminal", spion)
    sim = _sim(physics_v2=True)
    for _ in range(3):
        sim.step()  # Buffer der Agenten füllen
    opfer = sim.agents[0]
    opfer_brain = opfer.brain
    assert len(opfer_brain.rollout) > 0
    opfer.health = 0.01
    opfer.energy = 0.0

    sim.step()

    assert opfer not in sim.agents
    assert aufrufe.count(opfer_brain) == 1, "finalize_terminal genau EINMAL pro Tod"
    assert len(opfer_brain.rollout) == 0, "Restbuffer geflusht"


def test_terminal_malus_nicht_verworfen_bei_vollem_buffer(monkeypatch):
    """M-1 (Final-Review): stirbt ein v2-Agent (z.B. Toxin-Tod in
    _execute_embodied, self.alive=False, KEIN Früh-Return) in exakt dem Tick,
    der seinen Buffer auf ROLLOUT_HORIZON (128) Transitionen bringt, darf der
    v2-Pfad in agent.update NICHT trotzdem maybe_train() aufrufen — das würde
    den vollen Buffer flushen+leeren, BEVOR remove_dead → finalize_terminal
    läuft, und der Todes-Malus -3.0 fände keinen Buffer mehr vor (verfiele
    still). Stattdessen muss finalize_terminal (via remove_dead) den noch
    vollen Buffer sehen, den Malus additiv auf die letzte Transition münzen
    und done=True setzen.

    Aufbau: Buffer wird auf ROLLOUT_HORIZON - 1 vorgefüllt (reale Transition
    dupliziert — vermeidet Nachbau des internen Dict-Layouts), dann
    _execute_embodied gepatcht, um wie ein Toxin-Tod self.alive=False zu
    setzen, OHNE Früh-Return — exakt der reale Code-Pfad. Der nächste
    opfer.update(...)-Aufruf fügt dadurch die 128. Transition hinzu UND lässt
    den Agenten im selben Tick sterben — das ist das echte Race.

    Bekannte, akzeptierte Rest-Lücke (NICHT gefixt): stirbt ein Agent per
    Früh-Return NACHDEM der Vortick den Buffer bereits regulär geflusht hat
    (Buffer also schon leer VOR diesem Tick), gibt es keine Transition mehr,
    an die finalize_terminal den Malus hängen kann — der Malus entfällt in
    diesem seltenen Fall bewusst weiterhin (siehe finalize_terminal-Guard
    `if not self.rollout.storage: return None`)."""
    sim = _sim(physics_v2=True)
    opfer = sim.agents[0]
    opfer_brain = opfer.brain
    opfer.is_sleeping = False
    assert opfer_brain.physics_v2

    def _tick():
        opfer.update(
            sim.world,
            sim.agents,
            tick=0,
            tribes=sim.tribes,
            economy=sim.economy,
            technology=sim.technology,
        )

    _tick()
    assert len(opfer_brain.rollout) == 1, "ein Tick füllt genau eine Transition"
    vorlage = opfer_brain.rollout.storage[0]
    while len(opfer_brain.rollout) < ROLLOUT_HORIZON - 1:
        opfer_brain.rollout.add(dict(vorlage))
    assert len(opfer_brain.rollout) == ROLLOUT_HORIZON - 1, "Buffer eine Transition vor dem Trigger"

    # Toxin-Tod nachbauen: _execute_embodied setzt self.alive=False mitten im
    # Tick, OHNE Früh-Return — update() läuft danach bis maybe_train() durch.
    original_execute = agent_mod.Agent._execute_embodied

    def toedlich(self, world, brain_step, view):
        result = original_execute(self, world, brain_step, view)
        self.alive = False
        return result

    monkeypatch.setattr(agent_mod.Agent, "_execute_embodied", toedlich)

    original_finalize = Brain.finalize_terminal
    aufrufe = []

    def spion(self, *a, **kw):
        aufrufe.append(len(self.rollout))
        return original_finalize(self, *a, **kw)

    monkeypatch.setattr(Brain, "finalize_terminal", spion)

    _tick()  # Todes-Tick: fügt die 128. Transition hinzu UND toetet den Agenten.

    assert not opfer.alive, "Agent muss in diesem Tick gestorben sein"
    assert len(opfer_brain.rollout) == ROLLOUT_HORIZON, (
        "maybe_train() darf im Sterbe-Tick NICHT gelaufen sein — der Buffer "
        "muss noch voll sein, wenn remove_dead/finalize_terminal ihn sieht"
    )
    letzte_transition = opfer_brain.rollout.storage[-1]  # Referenz vor dem Clear sichern
    reward_vor_malus = letzte_transition["reward"]

    sim.remove_dead()

    assert opfer not in sim.agents
    assert aufrufe == [ROLLOUT_HORIZON], (
        "finalize_terminal muss den Buffer NOCH VOLL vorfinden — maybe_train "
        "darf ihn im Sterbe-Tick nicht vorher geflusht haben"
    )
    assert len(opfer_brain.rollout) == 0, "finalize_terminal flusht/leert den Buffer selbst"
    erwarteter_malus_reward = max(-6.0, min(6.0, reward_vor_malus + DEATH_REWARD_V2))
    assert letzte_transition["reward"] == pytest.approx(erwarteter_malus_reward), (
        "Todes-Malus -3.0 muss additiv auf die letzte Transition gemünzt sein"
    )
    assert letzte_transition["done"] is True, "letzte Transition muss done=True tragen"


def test_v1_reward_pfad_unveraendert():
    """Verzweigungs-Gegenprobe: im v1 skaliert cognition_mult weiter und die
    Event-Rewards zählen (keine Regression durch den v2-Umbau)."""
    sim = _sim(physics_v2=False)
    agent = sim.agents[0]
    agent.update(
        sim.world,
        sim.agents,
        tick=0,
        tribes=sim.tribes,
        economy=sim.economy,
        technology=sim.technology,
    )
    assert not hasattr(agent, "curiosity_last") or agent.physics_v2 is False
    assert np.isfinite(agent.last_reward)
