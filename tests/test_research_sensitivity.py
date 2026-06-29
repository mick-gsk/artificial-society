"""B4/B5: matched-moisture and matched-ingredient sensitivity nulls.

These are *secondary* nulls (the fixed-0.5-moisture, full-24-ingredient recombiner
stays the conservative primary). B5 restricts the recombiner's seed pool to the
ingredients the learned arm actually encountered (derived from its export); B4
replays the moisture distribution the learned arm experienced. Both reuse the
*identical* recombiner machinery — only the inputs change.
"""

from __future__ import annotations

import numpy as np

from artificial_society.environment.materials import MATERIALS, combine_vectors
from artificial_society.research import sensitivity
from artificial_society.research.instrument import count_combine_calls
from artificial_society.research.recombiner import run_recombiner


def _base_inputs(entries):
    """All base-material names appearing as recipe inputs in a registry."""
    used = set()
    for e in entries:
        for mat in e["recipe"][1:3]:
            if mat in MATERIALS:
                used.add(mat)
    return used


# --- B5: matched ingredient ----------------------------------------------------


def test_base_materials_used_extracts_encountered_ingredients():
    entries = [
        {"id": "mat_0000", "recipe": ("strike", "stone", "flint"), "vector": [0.5] * 12},
        {"id": "mat_0001", "recipe": ("bundle", "mat_0000", "dry_grass"), "vector": [0.5] * 12},
        {"id": "mat_0002", "recipe": ("blow", "mat_0001", None), "vector": [0.5] * 12},
    ]
    used = sensitivity.base_materials_used(entries)
    assert set(used) == {"stone", "flint", "dry_grass"}
    # mat_XXXX ids and None are excluded; order follows MATERIALS
    assert used == [m for m in MATERIALS if m in {"stone", "flint", "dry_grass"}]


def test_recombiner_seed_pool_restricts_base_ingredients():
    pool = ["stone", "flint"]
    entries, _ = run_recombiner(seed=3, n_attempts=2000, seed_pool=pool)
    assert _base_inputs(entries) <= set(pool)
    # reproducible
    entries2, _ = run_recombiner(seed=3, n_attempts=2000, seed_pool=pool)
    assert [e["recipe"] for e in entries] == [e["recipe"] for e in entries2]


def test_matched_ingredient_null_uses_only_encountered_ingredients():
    learned = [
        {"id": "mat_0000", "recipe": ("strike", "stone", "flint"), "vector": [0.5] * 12},
        {"id": "mat_0001", "recipe": ("bundle", "mat_0000", "stone"), "vector": [0.6] * 12},
    ]
    entries, meta = sensitivity.run_matched_ingredient_null(learned, seed=5, n_attempts=1500)
    assert set(meta["seed_pool"]) == {"stone", "flint"}
    assert _base_inputs(entries) <= {"stone", "flint"}


# --- B4: matched moisture ------------------------------------------------------


def test_instrument_records_env_moisture():
    va, vb = MATERIALS["stone"], MATERIALS["flint"]
    with count_combine_calls(record_moisture=True) as cc:
        combine_vectors(va, vb, "strike", {"moisture": 0.3})
        combine_vectors(va, vb, "strike", {"moisture": 0.7})
    assert cc.n == 2
    assert cc.moisture == [0.3, 0.7]


def test_recombiner_moisture_samples_reproducible():
    samples = [0.1, 0.5, 0.9]
    e1, _ = run_recombiner(seed=8, n_attempts=1500, moisture_samples=samples)
    e2, _ = run_recombiner(seed=8, n_attempts=1500, moisture_samples=samples)
    assert [e["recipe"] for e in e1] == [e["recipe"] for e in e2]


def test_recombiner_moisture_value_changes_outcomes_with_rng_held():
    """Same RNG draws (equal-length sample lists) but different moisture -> different run."""
    dry, wet = [0.05] * 4, [0.95] * 4  # same len -> identical index draws
    e_dry, _ = run_recombiner(seed=2, n_attempts=3000, moisture_samples=dry)
    e_wet, _ = run_recombiner(seed=2, n_attempts=3000, moisture_samples=wet)
    assert [e["recipe"] for e in e_dry] != [e["recipe"] for e in e_wet]


def test_thin_distribution_is_deterministic_subsample():
    vals = list(np.arange(1000.0))
    thin = sensitivity.thin_distribution(vals, max_n=100)
    assert len(thin) == 100
    assert thin == sensitivity.thin_distribution(vals, max_n=100)  # deterministic
    assert sensitivity.thin_distribution(vals, max_n=5000) == vals  # no-op when small enough
