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
) -> tuple[list[dict], list[dict]]:
    """Run the standalone recombiner for ``n_attempts`` combinations.

    Returns ``(entries, series)`` where ``entries`` is the append-only discovery
    list (``{id, recipe, vector, discovered_by, tick, uses}``) and ``series`` is
    per-checkpoint ``{attempt, n_discoveries}`` snapshots.
    """
    env = {"moisture": moisture}
    saved_state = random.getstate()
    random.seed(seed)
    try:
        # Parallel lists: index i -> (id/name, vector). Base materials first.
        pool_ids: list[str] = list(MATERIALS.keys())
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


def run_embodied_recombiner(
    seed: int,
    n_attempts: int,
    n_agents: int = 8,
    share_fidelity: float = 0.72,
    moisture: float = 0.5,
    checkpoint_every: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Embodiment-matched recombiner null (E0 / M2).

    Identical blind uniform combine machinery as :func:`run_recombiner`, but the
    ``n_attempts`` budget is split round-robin over ``n_agents`` agents, each holding
    its **own** local material pool (fragmentation). A new discovery is shared into one
    random other agent's pool only with probability ``share_fidelity`` (imperfect
    transmission, the social channel's documented default 0.72). Discoveries register
    in ONE global, globally-deduped store (the metric's registry, exactly like the
    learned arm's ``DISCOVERY_REGISTRY``) and carry ``discovered_by = agent index``.

    This isolates the disembodied recombiner's structural advantage — a single
    perfect-memory monopolist pool with the full discovery set always available — from
    any learning, so the gate can test whether the -9.4 deficit is an embodiment /
    population-fragmentation confound rather than a failure of the learned policy.

    Documented simplification (pilot): re-deriving an already-globally-known material
    does NOT add it to the agent's pool — pools grow only via an agent's own *novel*
    discoveries plus ``share_fidelity`` transmission. ``n_agents=1`` is a single-pool
    null (≈ the disembodied operator, minus the no-op sharing branch).
    """
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    env = {"moisture": moisture}
    saved_state = random.getstate()
    random.seed(seed)
    try:
        base_ids: list[str] = list(MATERIALS.keys())
        base_vecs: list[np.ndarray] = [MATERIALS[m].astype(np.float32) for m in base_ids]
        # per-agent local pools (parallel id/vec lists); each starts with all base materials
        pools_ids: list[list[str]] = [list(base_ids) for _ in range(n_agents)]
        pools_vecs: list[list[np.ndarray]] = [list(base_vecs) for _ in range(n_agents)]

        # ONE global vectorised dedup store for DISCOVERED materials (the metric's view).
        cap = 4096
        store = np.zeros((cap, N_PROPS), dtype=np.float32)
        count = 0

        entries: list[dict] = []
        series: list[dict] = []

        for i in range(n_attempts):
            a = i % n_agents
            pid, pvec = pools_ids[a], pools_vecs[a]
            n = len(pid)
            ai = random.randrange(n)
            bi = ai
            if n > 1:
                while bi == ai:
                    bi = random.randrange(n)
            mat_a, va = pid[ai], pvec[ai]
            mat_b, vb = (pid[bi], pvec[bi]) if bi != ai else (None, None)
            action = random.choice(PRIMITIVE_ACTIONS)

            new_vec = combine_vectors(va, vb, action, env)
            if new_vec is not None and float(new_vec.sum()) > 0.1:
                vec = np.clip(new_vec, 0.0, 1.0).astype(np.float32)
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
                            "discovered_by": a,
                            "tick": i,
                            "uses": 0,
                        }
                    )
                    # discoverer gains it locally
                    pid.append(mat_id)
                    pvec.append(vec)
                    # imperfect transmission into one random OTHER agent's pool
                    if n_agents > 1 and random.random() < share_fidelity:
                        other = random.randrange(n_agents)
                        if other != a:
                            pools_ids[other].append(mat_id)
                            pools_vecs[other].append(vec)

            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                series.append({"attempt": i + 1, "n_discoveries": count})

        series.append({"attempt": n_attempts, "n_discoveries": count})
        return entries, series
    finally:
        random.setstate(saved_state)
