"""Plan 4: v2 vererbt kein Gelerntes bei Geburt — nur Gene. v1 byte-identisch (Golden)."""

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
    child = sim.spawn_child_from_parent(parent, dict(parent.genes))
    child_w = _brain_weight_snapshot(child.brain)
    paare = [(c, p) for c, p in zip(child_w, parent_w) if c.shape == p.shape]
    assert paare, "keine form-gleichen Tensoren gefunden"
    child_flat = torch.cat([c.flatten() for c, _ in paare])
    parent_flat = torch.cat([p.flatten() for _, p in paare])
    cos = float(torch.dot(child_flat, parent_flat) / (child_flat.norm() * parent_flat.norm()))
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
    child = sim.spawn_child_from_parent(parent, dict(parent.genes))
    assert getattr(child, "causal_memory", None) is None or not child.causal_memory.sequences
    assert not child.remedy_knowledge
    child_mats = getattr(child, "material_inventory", {})
    assert not any(m.startswith("mat_") for m in child_mats), "mat_-Discovery vererbt (A9)"


def test_v2_kind_erbt_kein_resource_memory():
    sim = _v2_sim()
    parent = sim.agents[0]
    parent.memory.resource_memory = [(2, 2), (3, 3), (4, 4)]
    child = sim.spawn_child_from_parent(parent, dict(parent.genes))
    assert child.memory.resource_memory == [], "resource_memory vererbt (A10 nicht gegated)"
