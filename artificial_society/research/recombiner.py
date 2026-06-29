"""Standalone, compute-matched random-recombiner null.

This is the primary baseline for the gate. It replicates the *identical* invention
machinery (``materials.combine_vectors`` with the same 0.08 dedup radius) but with
**uniform** input/action selection and **no** policy, no memory, no social learning,
and no embodiment — the disembodied operator the adversarial review found reproduces
the headline signal (~9.8k discoveries, DAG depth 22).

Performance note: the simulation's ``DiscoveryRegistry`` dedups with a pure-Python
``for entry in entries`` scan (O(#discoveries) per attempt) which becomes the
bottleneck when matching ~10^5 attempts. This module therefore keeps its **own**
vectorised dedup (one numpy distance op against a growable array) — same 0.08
semantics, orders of magnitude faster, and it touches no hot file.

Fidelity / fairness choices (documented simplifications for the pilot):
* **Seed pool** = all base ``MATERIALS`` (24) — a superset of what agents usually
  encounter, so this is a *conservative* (strong) null.
* **env** = ``{"moisture": moisture}`` (default 0.5); ``combine_vectors`` reads no
  other env key.
* **Compute matching** = run exactly ``n_attempts`` combinations (the learned arm's
  measured ``combine_vectors`` call count).

``combine_vectors`` uses the *global* ``random`` module in its stochastic "rub"
branch, so we seed and restore global RNG state around the loop to stay both
reproducible and side-effect-free for in-process callers.
"""

from __future__ import annotations

import random

import numpy as np

from artificial_society.environment.materials import MATERIALS, N_PROPS, combine_vectors
from artificial_society.systems.invention import PRIMITIVE_ACTIONS

DEDUP_TAU = 0.08  # matches DiscoveryRegistry's default similarity threshold


def run_recombiner(
    seed: int,
    n_attempts: int,
    moisture: float = 0.5,
    checkpoint_every: int | None = None,
    quiet: bool = True,  # kept for signature compat; no prints to suppress here
    seed_pool: list[str] | None = None,
    moisture_samples=None,
) -> tuple[list[dict], list[dict]]:
    """Run the standalone recombiner for ``n_attempts`` combinations.

    Returns ``(entries, series)`` where ``entries`` is the append-only discovery
    list (``{id, recipe, vector, discovered_by, tick, uses}``) and ``series`` is
    per-checkpoint ``{attempt, n_discoveries}`` snapshots.

    Sensitivity-null knobs (both keep the *identical* recombination machinery):
    * ``seed_pool`` (B5, matched ingredient) — restrict the starting base materials
      to this subset (intersected with ``MATERIALS``, order preserved) instead of
      all 24. Discovered materials are still added as they appear.
    * ``moisture_samples`` (B4, matched moisture) — if given, each attempt draws its
      ``env["moisture"]`` from this empirical sample (one seeded draw per attempt)
      instead of the fixed ``moisture`` scalar.
    """
    env = {"moisture": moisture}
    use_msamples = moisture_samples is not None and len(moisture_samples) > 0
    msamples = [float(x) for x in moisture_samples] if use_msamples else None
    saved_state = random.getstate()
    random.seed(seed)
    try:
        # Parallel lists: index i -> (id/name, vector). Base materials first.
        if seed_pool is not None:
            wanted = set(seed_pool)
            pool_ids: list[str] = [m for m in MATERIALS if m in wanted]
            if not pool_ids:
                raise ValueError("seed_pool does not overlap MATERIALS")
        else:
            pool_ids = list(MATERIALS.keys())
        pool_vecs: list[np.ndarray] = [MATERIALS[m].astype(np.float32) for m in pool_ids]

        # Vectorised dedup store for DISCOVERED materials only (growable, preallocated).
        cap = 4096
        store = np.zeros((cap, N_PROPS), dtype=np.float32)
        count = 0  # number of discovered materials

        entries: list[dict] = []
        series: list[dict] = []

        for i in range(n_attempts):
            n = len(pool_ids)
            ai = random.randrange(n)
            bi = ai
            if n > 1:
                while bi == ai:
                    bi = random.randrange(n)
            mat_a, va = pool_ids[ai], pool_vecs[ai]
            mat_b, vb = (pool_ids[bi], pool_vecs[bi]) if bi != ai else (None, None)
            action = random.choice(PRIMITIVE_ACTIONS)
            if use_msamples:
                env["moisture"] = msamples[random.randrange(len(msamples))]

            new_vec = combine_vectors(va, vb, action, env)
            if new_vec is not None and float(new_vec.sum()) > 0.1:
                vec = np.clip(new_vec, 0.0, 1.0).astype(np.float32)
                # one vectorised distance scan against all discovered vectors
                is_new = not (
                    count > 0 and np.linalg.norm(store[:count] - vec, axis=1).min() < DEDUP_TAU
                )
                if is_new:
                    if count == cap:
                        store = np.concatenate([store, np.zeros_like(store)])
                        cap = store.shape[0]
                    store[count] = vec
                    mat_id = f"mat_{count:04d}"
                    count += 1
                    entries.append(
                        {
                            "id": mat_id,
                            "recipe": (action, mat_a, mat_b),
                            "vector": vec,
                            "discovered_by": -1,
                            "tick": i,
                            "uses": 0,
                        }
                    )
                    pool_ids.append(mat_id)
                    pool_vecs.append(vec)

            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                series.append({"attempt": i + 1, "n_discoveries": count})

        series.append({"attempt": n_attempts, "n_discoveries": count})
        return entries, series
    finally:
        random.setstate(saved_state)
