"""B4/B5 sensitivity nulls — matched moisture & matched ingredient.

The conservative *primary* null is the fixed-moisture (0.5), full-24-ingredient
recombiner. These secondary nulls tighten the effect estimate by removing two
confounds while keeping the identical recombination machinery:

* **B5 matched ingredient** — restrict the recombiner's starting materials to the
  base ingredients the learned arm actually encountered (derived from its exported
  discovery recipes), isolating "better *search*" from "more *access*".
* **B4 matched moisture** — replay the moisture distribution the learned arm
  experienced (captured at the ``combine_vectors`` call site via
  :func:`instrument.count_combine_calls(record_moisture=True)`), instead of the
  fixed 0.5 the recombiner otherwise uses.

Both keep the fixed-0.5 / full-pool arm as the pre-registered primary; results are
reported as sensitivity (design §9: sensitivity ≠ confirmatory).
"""

from __future__ import annotations

from artificial_society.environment.materials import MATERIALS
from artificial_society.research.recombiner import run_recombiner


def base_materials_used(entries: list[dict]) -> list[str]:
    """Base ``MATERIALS`` that appear as recipe inputs in a learned export.

    Order follows ``MATERIALS`` for determinism; ``mat_XXXX`` discoveries and
    ``None`` slots are excluded. This is the ingredient set for the B5 null.
    """
    used = set()
    for e in entries:
        recipe = e.get("recipe")
        if not recipe:
            continue
        for mat in recipe[1:3]:
            if mat in MATERIALS:
                used.add(mat)
    return [m for m in MATERIALS if m in used]


def thin_distribution(values, max_n: int = 2000) -> list[float]:
    """Deterministically down-sample ``values`` to at most ``max_n`` (evenly spaced).

    Used to keep an exported empirical moisture distribution compact while
    preserving its shape. No RNG; returns the input unchanged when already small.
    """
    vals = [float(x) for x in values]
    n = len(vals)
    if n <= max_n:
        return vals
    step = n / max_n
    return [vals[int(i * step)] for i in range(max_n)]


def run_matched_ingredient_null(
    learned_entries: list[dict], seed: int, n_attempts: int, moisture: float = 0.5
):
    """B5: recombiner restricted to the ingredients the learned arm encountered."""
    pool = base_materials_used(learned_entries)
    entries, series = run_recombiner(
        seed=seed, n_attempts=n_attempts, moisture=moisture, seed_pool=pool
    )
    meta = {
        "arm": "recombiner_matched_ingredient",
        "seed": seed,
        "n_attempts": n_attempts,
        "seed_pool": pool,
        "n_discoveries": len(entries),
    }
    return entries, meta


def run_matched_moisture_null(
    moisture_samples, seed: int, n_attempts: int, seed_pool: list[str] | None = None
):
    """B4: recombiner replaying the learned arm's empirical moisture distribution."""
    entries, series = run_recombiner(
        seed=seed,
        n_attempts=n_attempts,
        moisture_samples=moisture_samples,
        seed_pool=seed_pool,
    )
    meta = {
        "arm": "recombiner_matched_moisture",
        "seed": seed,
        "n_attempts": n_attempts,
        "n_moisture_samples": len(moisture_samples),
        "n_discoveries": len(entries),
    }
    return entries, meta
