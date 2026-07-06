"""C5: strength-Gen (16.) — v2-only-Draws, v1-RNG-Strom bleibt unberührt."""

from __future__ import annotations

import random
from types import SimpleNamespace

from artificial_society.agents.agent import Agent, attach_body
from artificial_society.agents.traits import (
    STRENGTH_PERTURBATION_SIGMA,
    TRAIT_RANGES,
    derive_strength,
    derive_traits,
    ensure_strength_trait,
    random_traits,
)


def test_trait_ranges_hat_strength_als_16_gen():
    assert TRAIT_RANGES["strength"] == (0.1, 0.9)
    assert len(TRAIT_RANGES) == 16
    assert STRENGTH_PERTURBATION_SIGMA == 0.012  # 0.012-Klasse (Spec C5)


def test_random_traits_zieht_kein_strength():
    """v1-RNG-Strom byte-identisch: random_genes darf den neuen Key NICHT ziehen."""
    random.seed(5)
    traits = random_traits()
    assert "strength" not in traits
    assert len(traits) == 15


def test_derive_traits_ueberspringt_strength_auch_bei_v2_eltern():
    """Der GENE_RANGES-Loop in inherit_genes darf für strength keinen gauss-Draw
    machen (v1-Strom!) und keinen Key erzeugen — Vererbung läuft über inherit_strength."""
    random.seed(6)
    parent = SimpleNamespace(traits={**random_traits(), "strength": 0.7}, learning_score=1.0)
    child = derive_traits(parent, parent)
    assert "strength" not in child


def test_ensure_strength_trait_zieht_einmal_und_ist_idempotent():
    random.seed(7)
    traits = {}
    ensure_strength_trait(traits)
    erster = traits["strength"]
    assert 0.1 <= erster <= 0.9
    ensure_strength_trait(traits)  # idempotent: kein zweiter Draw
    assert traits["strength"] == erster


def test_derive_strength_fitness_gewichtet_mit_sigma_0012():
    random.seed(8)
    a = SimpleNamespace(traits={"strength": 0.3}, learning_score=1.0)
    b = SimpleNamespace(traits={"strength": 0.7}, learning_score=1.0)
    kinder = []
    for _ in range(200):
        child = {}
        derive_strength(child, a, b)
        kinder.append(child["strength"])
    mittel = sum(kinder) / len(kinder)
    streuung = (sum((k - mittel) ** 2 for k in kinder) / len(kinder)) ** 0.5
    assert abs(mittel - 0.5) < 0.01  # gleiche Fitness ⇒ Mittelwert der Eltern
    assert 0.006 < streuung < 0.020  # σ ≈ 0.012, NICHT 0.25-Klasse


def test_attach_body_speist_body_aus_dem_gen():
    random.seed(9)
    agent = Agent.spawn_random(0, 0)
    assert "strength" not in agent.traits  # v1-Spawn zieht es nicht
    attach_body(agent)
    assert "strength" in agent.traits
    assert agent.body.strength == agent.traits["strength"]
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
    kind = sim.spawn_child_from_parent(eltern, dict(eltern.traits))
    assert "strength" in kind.traits
    assert kind.body is not None
    assert kind.body.strength == kind.traits["strength"]
