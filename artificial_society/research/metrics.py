"""Functional / irreducible innovation complexity (offline, arm-agnostic).

Why not raw DAG depth? The adversarial review showed that the raw
dependency-DAG depth is a *counting artifact*: a non-learning random recombiner
reaches DAG depth ~22 while the artifacts' functional content (active property
dimensions, vector norm) saturates much earlier. Chain length is not "standing on
shoulders".

This module computes complexity that is robust to that artifact. It operates only
on the exported registry of a single run — a list of entries
``{id, recipe=(action, mat_a, mat_b), vector[12]}`` — plus the base ``MATERIALS``
(depth 0).

Definitions
-----------
* **Structural depth** ``sd(M)`` — the classic longest path from seeds:
  ``sd(seed) = 0``; ``sd(M) = 1 + max(sd(inputs))``. Kept only as a sanity/baseline.

* **Functional depth** ``fd(M)`` — the *minimum* structural depth over all
  artifacts whose property vector lies within ``func_tau`` of M's vector. The
  discovery registry dedups at 0.08, so distinct entries are ≥0.08 apart; choosing
  ``func_tau`` larger than the dedup radius (default 0.15) clusters
  *functionally equivalent* artifacts, so a deep-but-redundant artifact inherits
  the shallow depth of an equivalent one reachable by a short recipe. This is the
  metric that distinguishes genuine cumulative innovation from redundant churn.

* **Knock-out validation** — for a sample of deterministic two-input recipes,
  remove an input and recompute ``combine_vectors``; if the product changes by
  ≥ the dedup radius, the input was functionally *required*. Confirms that recipe
  inputs actually drive the effect (rather than being decorative lineage).

``func_tau`` is a free observer parameter; report a sensitivity sweep (the gate
analysis does this).
"""

from __future__ import annotations

import random

import numpy as np

from artificial_society.environment.materials import MATERIALS, combine_vectors

DEDUP_TAU = 0.08
FUNC_TAU = 0.15
ACTIVE_DIM_THRESHOLD = 0.1
DETERMINISTIC_ACTIONS = (
    "strike",
    "bind",
    "bundle",
    "place_on_heat",
    "blow",
    "carry",
    "eat",
)  # "rub" is stochastic (uses global random), excluded from knock-out


def _is_base(mat: str | None) -> bool:
    return mat is not None and mat in MATERIALS


def _index(entries: list[dict]) -> dict[str, dict]:
    return {e["id"]: e for e in entries}


def _vec(mat: str | None, idx: dict[str, dict]) -> np.ndarray | None:
    """Resolve a material name/id to its property vector, or None if unknown."""
    if mat is None:
        return None
    if mat in MATERIALS:
        return MATERIALS[mat].copy()
    e = idx.get(mat)
    return np.asarray(e["vector"], dtype=np.float32).copy() if e is not None else None


def _recipe_inputs(recipe) -> tuple[str, str | None, str | None]:
    """Normalise a recipe (tuple or json list) to (action, mat_a, mat_b)."""
    action = recipe[0]
    a = recipe[1] if len(recipe) > 1 else None
    b = recipe[2] if len(recipe) > 2 else None
    return action, a, b


def structural_depths(entries: list[dict]) -> dict[str, int]:
    """Longest-path depth from seeds for every entry. Cycle-guarded."""
    idx = _index(entries)
    depth: dict[str, int] = {}
    visiting: set[str] = set()

    def d(mat: str | None) -> int:
        if mat is None or _is_base(mat):
            return 0
        if mat not in idx:
            return 0  # unknown reference -> treat as a leaf
        if mat in depth:
            return depth[mat]
        if mat in visiting:
            return 0  # defensive: registry is append-only, cycles shouldn't occur
        visiting.add(mat)
        recipe = idx[mat].get("recipe")
        if not recipe:
            depth[mat] = 0
        else:
            _, a, b = _recipe_inputs(recipe)
            depth[mat] = 1 + max(d(a), d(b))
        visiting.discard(mat)
        return depth[mat]

    for e in entries:
        d(e["id"])
    return depth


def functional_depths(
    entries: list[dict],
    struct: dict[str, int] | None = None,
    func_tau: float = FUNC_TAU,
) -> dict[str, int]:
    """Min structural depth over the func_tau-neighbourhood of each entry's vector."""
    if not entries:
        return {}
    if struct is None:
        struct = structural_depths(entries)
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    sd = np.array([struct[e["id"]] for e in entries], dtype=np.int64)
    fd: dict[str, int] = {}
    for i in range(len(entries)):
        dist = np.linalg.norm(V - V[i], axis=1)
        fd[entries[i]["id"]] = int(sd[dist <= func_tau].min())
    return fd


def n_functional_clusters(entries: list[dict], func_tau: float = FUNC_TAU) -> int:
    """Greedy count of functionally distinct artifacts (vector clustering)."""
    if not entries:
        return 0
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    reps: list[np.ndarray] = []
    for i in range(len(V)):
        if not any(np.linalg.norm(V[i] - r) <= func_tau for r in reps):
            reps.append(V[i])
    return len(reps)


def mean_active_dims(entries: list[dict], threshold: float = ACTIVE_DIM_THRESHOLD) -> float:
    if not entries:
        return 0.0
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    return float((threshold < V).sum(axis=1).mean())


def knockout_validate(
    entries: list[dict],
    sample: int = 80,
    seed: int = 0,
    dedup_tau: float = DEDUP_TAU,
    moisture: float = 0.5,
) -> dict:
    """Sample deterministic 2-input recipes; check inputs are functionally required.

    For each sampled recipe we (1) reproduce the product from its inputs and skip
    it if the deterministic recompute does not land within ``dedup_tau`` of the
    stored vector (state/env drift), then (2) drop input ``mat_b`` and require the
    product to change by ≥ ``dedup_tau`` — i.e. the input mattered.
    """
    idx = _index(entries)
    env = {"moisture": moisture}
    cand = [
        e
        for e in entries
        if e.get("recipe")
        and _recipe_inputs(e["recipe"])[0] in DETERMINISTIC_ACTIONS
        and _recipe_inputs(e["recipe"])[2] is not None
    ]
    rng = random.Random(seed)
    rng.shuffle(cand)
    cand = cand[:sample]

    tested = 0
    required = 0
    for e in cand:
        action, a, b = _recipe_inputs(e["recipe"])
        va, vb = _vec(a, idx), _vec(b, idx)
        if va is None or vb is None:
            continue
        repro = combine_vectors(va, vb, action, env)
        if repro is None:
            continue
        repro = np.clip(repro, 0.0, 1.0)
        if np.linalg.norm(repro - np.asarray(e["vector"], dtype=np.float32)) > dedup_tau:
            continue  # not deterministically reproducible -> skip
        tested += 1
        ko = combine_vectors(va, None, action, env)
        if ko is None:
            required += 1
            continue
        if np.linalg.norm(np.clip(ko, 0.0, 1.0) - repro) >= dedup_tau:
            required += 1

    return {
        "tested": tested,
        "required": required,
        "required_frac": (required / tested) if tested else None,
    }


def analyze_registry(entries: list[dict], func_tau: float = FUNC_TAU) -> dict:
    """One run -> the scalar DVs used by the gate."""
    struct = structural_depths(entries)
    fd = functional_depths(entries, struct, func_tau)
    sd_vals = np.array(list(struct.values()), dtype=float) if struct else np.zeros(1)
    fd_vals = np.array(list(fd.values()), dtype=float) if fd else np.zeros(1)
    return {
        "n_entries": len(entries),
        "func_tau": func_tau,
        "max_structural_depth": int(sd_vals.max()),
        "max_functional_depth": int(fd_vals.max()),
        "p95_functional_depth": float(np.percentile(fd_vals, 95)),
        "mean_functional_depth": float(fd_vals.mean()),
        "n_functional_clusters": n_functional_clusters(entries, func_tau),
        "mean_active_dims": mean_active_dims(entries),
    }
