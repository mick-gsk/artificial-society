"""TDD for M1 (Path-A): policy-coupled selective generation.

M1 replaces the random WHAT-to-combine choice with a seeded-softmax + value-of-
information IMAGINED means-ends search. It is flag-gated by ``AS_PATHA_M1`` (OFF is
byte-identical to the live system, so the determinism golden stays green) and scores
candidates with an UNCOUNTED ``_combine_pure`` twin so the learned arm stays
compute-matched to the recombiner (the imagined search must not inflate the counted
``combine_vectors`` attempt total).
"""

from __future__ import annotations

import numpy as np

from artificial_society.environment.materials import IDX, MATERIALS, N_PROPS, combine_vectors
from artificial_society.research.instrument import count_combine_calls
from artificial_society.rng import seed_all
from artificial_society.systems import need_driven_invention as ndi
from artificial_society.systems.invention import PRIMITIVE_ACTIONS

_BASE = list(MATERIALS.keys())[:5]


class _FakeAgent:
    def __init__(self, inv):
        self.material_inventory = dict(inv)
        self.genes = {"curiosity": 0.5}
        self.causal_memory = None


def _sharp_need():
    need = np.zeros(N_PROPS, dtype=np.float32)
    need[IDX["sharpness"]] = 1.0
    need[IDX["hardness"]] = 0.5
    return need


def test_m1_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AS_PATHA_M1", raising=False)
    assert ndi._m1_enabled() is False


def test_m1_enabled_by_flag(monkeypatch):
    monkeypatch.setenv("AS_PATHA_M1", "1")
    assert ndi._m1_enabled() is True


def test_imagined_combine_twin_is_uncounted():
    # The imagined means-ends search scores candidates with _combine_pure, which the
    # instrument's identity-based counter never sees -> compute-match to the recombiner
    # holds. Only the single real combine_vectors call is counted.
    va, vb = MATERIALS[_BASE[0]], MATERIALS[_BASE[1]]
    env = {"moisture": 0.5}
    with count_combine_calls() as cc:
        for _ in range(25):
            ndi._combine_pure(va, vb, "strike", env)  # imagined -> uncounted
        assert cc.n == 0
        combine_vectors(va, vb, "strike", env)  # real -> counted
        assert cc.n == 1


def test_m1_selection_is_pure_and_returns_valid_triple(monkeypatch):
    monkeypatch.setenv("AS_PATHA_M1", "1")
    seed_all(123)
    agent = _FakeAgent({m: 1.0 for m in _BASE})
    cell = {"materials": {m: 1.0 for m in _BASE}}
    env = {"moisture": 0.5}
    with count_combine_calls() as cc:
        mat_a, mat_b, action = ndi._select_combination_m1(agent, cell, _sharp_need(), None, env)
        assert cc.n == 0  # the whole imagined search is uncounted
    assert action in PRIMITIVE_ACTIONS
    assert mat_a is not None


def test_m1_selection_is_reproducible(monkeypatch):
    monkeypatch.setenv("AS_PATHA_M1", "1")
    agent = _FakeAgent({m: 1.0 for m in _BASE})
    cell = {"materials": {m: 1.0 for m in _BASE}}
    env = {"moisture": 0.5}
    seed_all(7)
    pick1 = ndi._select_combination_m1(agent, cell, _sharp_need(), None, env)
    seed_all(7)
    pick2 = ndi._select_combination_m1(agent, cell, _sharp_need(), None, env)
    assert pick1 == pick2
