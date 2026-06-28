"""Standalone, compute-matched random-recombiner null.

This is the primary baseline for the gate. It replicates the *identical* invention
machinery (``materials.combine_vectors`` + a fresh ``DiscoveryRegistry`` with the
same 0.08 dedup) but with **uniform** input/action selection and **no** policy, no
memory, no social learning, and no embodiment — exactly the process that the
adversarial review found reproduces the headline signal (~9.8k discoveries, DAG
depth 22). It is the disembodied operator, isolated from any learning.

Fidelity / fairness choices (documented simplifications for the pilot):

* **Seed pool** = all base ``MATERIALS`` (24). A superset of what agents typically
  encounter, so the recombiner is a *conservative* (strong) null: if the learned
  arm still wins against a recombiner with full ingredient access, the result is
  robust.
* **env** = ``{"moisture": moisture}`` (default 0.5). ``combine_vectors`` reads no
  other env key. A sensitivity check with sampled moisture is a follow-up.
* **Compute matching** = run exactly ``n_attempts`` combinations, where
  ``n_attempts`` is the learned arm's measured ``combine_vectors`` call count.

``combine_vectors`` uses the *global* ``random`` module in its stochastic "rub"
branch, so we seed and then restore global RNG state around the loop to stay both
reproducible and side-effect-free for in-process callers.
"""

from __future__ import annotations

import random

import numpy as np

from artificial_society.environment.materials import MATERIALS, DiscoveryRegistry, combine_vectors
from artificial_society.research.instrument import quiet_stdout
from artificial_society.systems.invention import PRIMITIVE_ACTIONS


def run_recombiner(
    seed: int,
    n_attempts: int,
    moisture: float = 0.5,
    checkpoint_every: int | None = None,
    quiet: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Run the standalone recombiner for ``n_attempts`` combinations.

    Returns ``(entries, series)`` where ``entries`` is the fresh registry's
    append-only discovery list and ``series`` is per-checkpoint
    ``{attempt, n_discoveries}`` snapshots.
    """
    env = {"moisture": moisture}
    saved_state = random.getstate()
    random.seed(seed)
    try:
        reg = DiscoveryRegistry()
        pool: list[str] = list(MATERIALS.keys())
        vecs: dict[str, np.ndarray] = {name: MATERIALS[name].copy() for name in MATERIALS}
        series: list[dict] = []

        with quiet_stdout(quiet):
            for i in range(n_attempts):
                mat_a = random.choice(pool)
                mat_b = None
                if len(pool) > 1:
                    mat_b = random.choice([m for m in pool if m != mat_a])
                action = random.choice(PRIMITIVE_ACTIONS)

                va = vecs[mat_a]
                vb = vecs[mat_b] if mat_b is not None else None
                new_vec = combine_vectors(va, vb, action, env)
                if new_vec is not None and float(new_vec.sum()) > 0.1:
                    before = len(reg.entries)
                    mat_id = reg.register(
                        new_vec, discoverer_id=-1, tick=i, recipe=(action, mat_a, mat_b)
                    )
                    if len(reg.entries) > before:  # genuinely new (not a re-discovery)
                        vecs[mat_id] = reg.entries[-1]["vector"]
                        pool.append(mat_id)

                if checkpoint_every and (i + 1) % checkpoint_every == 0:
                    series.append({"attempt": i + 1, "n_discoveries": len(reg.entries)})

        series.append({"attempt": n_attempts, "n_discoveries": len(reg.entries)})
        return list(reg.entries), series
    finally:
        random.setstate(saved_state)
