"""C5: strength-Gen (16.) — v2-only-Draws, v1-RNG-Strom bleibt unberührt."""

from __future__ import annotations

import random
from types import SimpleNamespace

from artificial_society.agents.agent import Agent, attach_body
from artificial_society.agents.genetics import (
    GENE_RANGES,
    STRENGTH_MUTATION_SIGMA,
    ensure_strength_gene,
    inherit_genes,
    inherit_strength,
    random_genes,
)


def test_gene_ranges_hat_strength_als_16_gen():
    assert GENE_RANGES["strength"] == (0.1, 0.9)
    assert len(GENE_RANGES) == 16
    assert STRENGTH_MUTATION_SIGMA == 0.012  # 0.012-Klasse (Spec C5)


def test_random_genes_zieht_kein_strength():
    """v1-RNG-Strom byte-identisch: random_genes darf den neuen Key NICHT ziehen."""
    random.seed(5)
    genes = random_genes()
    assert "strength" not in genes
    assert len(genes) == 15


def test_inherit_genes_ueberspringt_strength_auch_bei_v2_eltern():
    """Der GENE_RANGES-Loop in inherit_genes darf für strength keinen gauss-Draw
    machen (v1-Strom!) und keinen Key erzeugen — Vererbung läuft über inherit_strength."""
    random.seed(6)
    parent = SimpleNamespace(genes={**random_genes(), "strength": 0.7}, learning_score=1.0)
    child = inherit_genes(parent, parent)
    assert "strength" not in child


def test_ensure_strength_gene_zieht_einmal_und_ist_idempotent():
    random.seed(7)
    genes = {}
    ensure_strength_gene(genes)
    erster = genes["strength"]
    assert 0.1 <= erster <= 0.9
    ensure_strength_gene(genes)  # idempotent: kein zweiter Draw
    assert genes["strength"] == erster


def test_inherit_strength_fitness_gewichtet_mit_sigma_0012():
    random.seed(8)
    a = SimpleNamespace(genes={"strength": 0.3}, learning_score=1.0)
    b = SimpleNamespace(genes={"strength": 0.7}, learning_score=1.0)
    kinder = []
    for _ in range(200):
        child = {}
        inherit_strength(child, a, b)
        kinder.append(child["strength"])
    mittel = sum(kinder) / len(kinder)
    streuung = (sum((k - mittel) ** 2 for k in kinder) / len(kinder)) ** 0.5
    assert abs(mittel - 0.5) < 0.01  # gleiche Fitness ⇒ Mittelwert der Eltern
    assert 0.006 < streuung < 0.020  # σ ≈ 0.012, NICHT 0.25-Klasse


def test_attach_body_speist_body_aus_dem_gen():
    random.seed(9)
    agent = Agent.spawn_random(0, 0)
    assert "strength" not in agent.genes  # v1-Spawn zieht es nicht
    attach_body(agent)
    assert "strength" in agent.genes
    assert agent.body.strength == agent.genes["strength"]
    assert 0.1 <= agent.body.strength <= 0.9


def test_spawn_child_from_parent_v2_laeuft_ohne_namenskollision():
    """Regression (Review F1): spawn_child_from_parent hat eine LOKALE Variable
    `inherit_strength` (Gewichts-Vererbungsstärke für inherit_weights_from) —
    der genetics-Import MUSS aliased sein (inherit_strength_gene), sonst
    UnboundLocalError beim ersten v2-Kind-Spawn (latent bis Task 14)."""
    from artificial_society.simulation import Simulation

    sim = Simulation(
        seed=11,
        physics_v2=True,
        headless=True,
        load_checkpoint=False,
        grid_w=20,
        grid_h=15,
        initial_population=8,
    )
    eltern = sim.agents[0]
    kind = sim.spawn_child_from_parent(eltern, dict(eltern.genes))
    assert "strength" in kind.genes
    assert kind.body is not None
    assert kind.body.strength == kind.genes["strength"]
