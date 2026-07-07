"""Plan 4: v2 vererbt kein Gelerntes bei Geburt — nur Trait. v1 byte-identisch (Golden)."""

from __future__ import annotations

import torch

from artificial_society.simulation import CausalMemory, Simulation


def _v2_sim():
    return Simulation(
        headless=True,
        load_checkpoint=False,
        physics_v2=True,
        seed=7,
        grid_w=20,
        grid_h=15,
        initial_population=8,
    )


def _brain_weight_snapshot(brain):
    return [p.detach().clone() for _, p in brain.named_parameters()]


def test_v2_kind_erbt_keine_brain_gewichte():
    sim = _v2_sim()
    parent = sim.agents[0]
    # Elter-Netz von der Zufallsinit wegtrainieren, damit Elter ≠ frisch
    with torch.no_grad():
        for _, p in parent.brain.named_parameters():
            p.add_(torch.randn_like(p) * 0.5)
    parent_w = _brain_weight_snapshot(parent.brain)
    spawn = sim.spawn_agent_from_parent(parent, dict(parent.traits))
    spawn_w = _brain_weight_snapshot(spawn.brain)
    paare = [(c, p) for c, p in zip(spawn_w, parent_w) if c.shape == p.shape]
    assert paare, "keine form-gleichen Tensoren gefunden"
    spawn_flat = torch.cat([c.flatten() for c, _ in paare])
    parent_flat = torch.cat([p.flatten() for _, p in paare])
    cos = float(torch.dot(spawn_flat, parent_flat) / (spawn_flat.norm() * parent_flat.norm()))
    # Kopiertes Netz (A1, strength≈0.5) korreliert stark mit dem gestörten Elter
    # (gemessen cos≈0.99); ein frisches Netz ist unkorreliert (cos≈0.00).
    # max-abs-Abweichung trennt NICHT (beide Fälle > 0.1) — Kosinus schon.
    assert cos < 0.5, f"Kind-Brain korreliert mit Elter (cos={cos:.3f}) — A1 nicht gegated"


def test_v2_kind_erbt_keine_lern_stores():
    sim = _v2_sim()
    parent = sim.agents[0]
    parent.causal_memory = CausalMemory(capacity=32)
    parent.causal_memory.receive_transmitted(("strike", "mat_flint", "mat_stone"), fidelity=0.9)
    parent.remedy_knowledge = {"cough": ["herb_b"]}
    parent.material_inventory = {"mat_flint": 2.0, "water": 1.0}
    spawn = sim.spawn_agent_from_parent(parent, dict(parent.traits))
    assert getattr(spawn, "causal_memory", None) is None or not spawn.causal_memory.sequences
    assert not spawn.remedy_knowledge
    spawn_mats = getattr(spawn, "material_inventory", {})
    assert not any(m.startswith("mat_") for m in spawn_mats), "mat_-Discovery vererbt (A9)"


def test_v2_kind_erbt_kein_resource_memory():
    sim = _v2_sim()
    parent = sim.agents[0]
    parent.memory.resource_memory = [(2, 2), (3, 3), (4, 4)]
    spawn = sim.spawn_agent_from_parent(parent, dict(parent.traits))
    assert spawn.memory.resource_memory == [], "resource_memory vererbt (A10 nicht gegated)"


import ast
import inspect
import textwrap

from artificial_society.systems.adaptation import AdaptationSystem


def test_spawn_agent_wird_nie_mit_parent_aufgerufen():
    """A3–A6 (ToM/Knowledge/EmotionalMemory/world_memory) stehen in
    Agent.spawn_agent unter `if parent is not None`; make_spawn ruft es ohne
    parent. Schlüge jemand parent= durch, würden diese Pfade schlagartig live
    und bräuchten ein parent.physics_v2-Gate. (Substring-Check scheidet aus:
    die Signatur enthält `other_parent=None`, und "parent=" ist Substring von
    "other_parent=" → AST-basiert prüfen.)"""
    src = textwrap.dedent(inspect.getsource(AdaptationSystem.make_spawn))
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "spawn_agent"
    ]
    assert calls, "make_spawn ruft Agent.spawn_agent nicht mehr auf?"
    for call in calls:
        assert "parent" not in {kw.arg for kw in call.keywords}, (
            "make_spawn reicht parent= durch — A3–A6 wären live!"
        )
        assert len(call.args) <= 6  # parent ist der 7. Positionsparameter


def test_latente_derive_from_methoden_feuern_nicht_bei_geburt():
    """StrategySystem.derive_from / EpisodicStrategyMemory.derive_from sind
    uncalled — Agenten tragen die Attribute nicht. Wächter gegen versehentliches
    Verdrahten."""
    sim = _v2_sim()
    a = sim.agents[0]
    assert not hasattr(a, "strategy")
    assert not hasattr(a, "strategy_memory")
    assert not hasattr(a, "episodic_strategy")


def test_v2_kind_erbt_weiter_trait_und_trust_prior():
    sim = _v2_sim()
    parent = sim.agents[0]
    parent.tribe_id = 1  # frisches Sim hat tribe_id=None → trust-Zweig feuerte nie
    # strength aus den übergebenen Genen entfernen → prüft den v2-Pfad
    # (derive_strength_trait) echt, statt tautologisch die Dict-Kopie
    traits = dict(parent.traits)
    traits.pop("strength", None)
    spawn = sim.spawn_agent_from_parent(parent, traits)
    assert "strength" in spawn.traits, "strength-Gen wurde nicht via v2-Pfad ergänzt"
    # Verwandtschafts-Prior bleibt (fester Wert, kein Lamarck)
    assert spawn.trust.get(parent.id) == 0.4


def test_lebzeit_imitation_bleibt_intakt():
    """A2 (Brain.imitate_from) ist der Lebzeit-Kanal — vom A1-Gating unberührt."""
    sim = _v2_sim()
    a, b = sim.agents[0], sim.agents[1]
    before = [p.detach().clone() for _, p in a.brain.named_parameters()]
    a.brain.imitate_from(b.brain, strength=0.5)
    after = [p.detach().clone() for _, p in a.brain.named_parameters()]
    diffs = [float((x - y).abs().max()) for x, y in zip(after, before) if x.shape == y.shape]
    assert max(diffs) > 0.0, "imitate_from wirkt nicht mehr — Lebzeit-Kanal beschädigt"


import math


def test_v2_geburten_smoke_nan_frei():
    """Kurzer v2-Lauf mit garantierter Geburt bleibt NaN-frei (Sanity nach dem Umbau)."""
    sim = _v2_sim()
    # Geburt erzwingen (der reguläre Caller extendet sim.agents selbst; hier manuell)
    sim.agents.append(sim.spawn_agent_from_parent(sim.agents[0], dict(sim.agents[0].traits)))
    for _ in range(40):
        sim.step()
    assert len(sim.agents) > 0
    assert all(math.isfinite(float(a.last_reward)) for a in sim.agents)
    assert all(not torch.isnan(a.hidden_state).any() for a in sim.agents)
