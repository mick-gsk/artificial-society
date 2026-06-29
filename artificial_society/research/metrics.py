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

# A2: a cKDTree radius query turns the func_tau neighbourhood scan from O(n^2)
# into ~O(n log n). scipy is optional — without it we fall back to the
# brute-force scan, which is bit-identical (just slower).
try:
    from scipy.spatial import cKDTree as _cKDTree

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - only hit in a scipy-less environment
    _cKDTree = None
    _HAS_SCIPY = False

DEDUP_TAU = 0.08
FUNC_TAU = 0.15
ACTIVE_DIM_THRESHOLD = 0.1
# Radius inflation for the KDTree candidate query. cKDTree measures distances in
# float64 on the float32 vectors, while the exact filter compares a float32 norm
# to func_tau; this margin (>> the ~1e-7 float32/float64 gap, << the 0.08 dedup
# spacing) guarantees the candidate set is a superset of the true float32-norm
# neighbourhood, so the exact re-filter is bit-identical to brute force.
_KDTREE_RADIUS_MARGIN = 1e-4
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


def _functional_depths_bruteforce(
    V: np.ndarray, sd: np.ndarray, ids: list[str], func_tau: float
) -> dict[str, int]:
    """O(n^2) reference: for each entry, the min struct depth over its func_tau ball."""
    fd: dict[str, int] = {}
    for i in range(len(ids)):
        dist = np.linalg.norm(V - V[i], axis=1)
        fd[ids[i]] = int(sd[dist <= func_tau].min())
    return fd


def _functional_depths_kdtree(
    V: np.ndarray, sd: np.ndarray, ids: list[str], func_tau: float
) -> dict[str, int]:
    """O(n log n) cKDTree path — bit-identical to :func:`_functional_depths_bruteforce`.

    The tree query returns a *superset* of each entry's func_tau neighbourhood
    (radius inflated by ``_KDTREE_RADIUS_MARGIN``); we then re-apply the *exact*
    float32 ``norm <= func_tau`` test to those candidates, so the depths selected
    — and thus their min — are identical to the brute-force scan.
    """
    if len(ids) == 0:
        return {}
    V64 = V.astype(np.float64)
    tree = _cKDTree(V64)
    neighbours = tree.query_ball_point(V64, func_tau + _KDTREE_RADIUS_MARGIN)
    fd: dict[str, int] = {}
    for i in range(len(ids)):
        cand = np.asarray(neighbours[i], dtype=np.int64)
        dist = np.linalg.norm(V[cand] - V[i], axis=1)
        fd[ids[i]] = int(sd[cand][dist <= func_tau].min())
    return fd


def functional_depths(
    entries: list[dict],
    struct: dict[str, int] | None = None,
    func_tau: float = FUNC_TAU,
) -> dict[str, int]:
    """Min structural depth over the func_tau-neighbourhood of each entry's vector.

    Uses the cKDTree radius query when scipy is available (A2), else a brute-force
    O(n^2) scan. Both paths return identical values (see the bit-identity tests).
    """
    if not entries:
        return {}
    if struct is None:
        struct = structural_depths(entries)
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    sd = np.array([struct[e["id"]] for e in entries], dtype=np.int64)
    ids = [e["id"] for e in entries]
    if _HAS_SCIPY:
        return _functional_depths_kdtree(V, sd, ids, func_tau)
    return _functional_depths_bruteforce(V, sd, ids, func_tau)


def _n_functional_clusters_bruteforce(entries: list[dict], func_tau: float) -> int:
    """O(n^2) greedy leader-clustering: count points not within func_tau of an earlier rep."""
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    reps: list[np.ndarray] = []
    for i in range(len(V)):
        if not any(np.linalg.norm(V[i] - r) <= func_tau for r in reps):
            reps.append(V[i])
    return len(reps)


def _n_functional_clusters_kdtree(entries: list[dict], func_tau: float) -> int:
    """O(n log n) leader-clustering — bit-identical to the brute-force greedy.

    Same index-order leader rule: point ``i`` is a new cluster rep iff no earlier rep
    lies within ``func_tau``. When ``i`` becomes a rep we mark every point within
    ``func_tau`` of it as covered (cKDTree candidate query at radius
    ``func_tau+_KDTREE_RADIUS_MARGIN``, then the exact float32 ``norm<=func_tau``
    re-filter), so a later point is skipped iff some earlier rep already covers it —
    identical to the brute-force condition.
    """
    if not entries:
        return 0
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    n = len(V)
    V64 = V.astype(np.float64)
    tree = _cKDTree(V64)
    covered = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if covered[i]:
            continue
        count += 1
        cand = np.asarray(
            tree.query_ball_point(V64[i], func_tau + _KDTREE_RADIUS_MARGIN), dtype=np.int64
        )
        dist = np.linalg.norm(V[cand] - V[i], axis=1)
        covered[cand[dist <= func_tau]] = True
    return count


def n_functional_clusters(entries: list[dict], func_tau: float = FUNC_TAU) -> int:
    """Greedy count of functionally distinct artifacts (vector clustering).

    cKDTree-accelerated when scipy is available (A2), else brute-force O(n^2).
    Both paths return the identical count (see the bit-identity tests).
    """
    if not entries:
        return 0
    if _HAS_SCIPY:
        return _n_functional_clusters_kdtree(entries, func_tau)
    return _n_functional_clusters_bruteforce(entries, func_tau)


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


def analyze_registry(
    entries: list[dict],
    func_tau: float = FUNC_TAU,
    struct: dict[str, int] | None = None,
) -> dict:
    """One run -> the scalar DVs used by the gate.

    ``struct`` (structural depths) is tau-independent; callers sweeping several
    ``func_tau`` values should compute it once and pass it in to avoid recomputing
    it per tau (A2 caching). Passing it in does not change any returned value.
    """
    if struct is None:
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
