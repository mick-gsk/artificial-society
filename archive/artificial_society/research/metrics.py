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

from artificial_society.environment.materials import IDX, MATERIALS, combine_vectors

DEDUP_TAU = 0.08
FUNC_TAU = 0.15
ACTIVE_DIM_THRESHOLD = 0.1
ADVANCE_MARGIN = 0.02  # task-utility a deeper artifact must beat to count as a genuine advance
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
    sd = np.array([struct[e["id"]] for e in entries], dtype=np.int32)
    n = len(entries)
    # Chunked min-structural-depth over the func_tau ball of each vector.
    # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b keeps memory at O(chunk * n), not O(n^2 * dims),
    # so this scales to the ~30k-entry recombiner registries.
    Vn2 = (V * V).sum(axis=1)
    big = np.int32(np.iinfo(np.int32).max)
    out = np.empty(n, dtype=np.int32)
    tau2 = float(func_tau) ** 2
    chunk = 256
    for s in range(0, n, chunk):
        b = V[s : s + chunk]
        d2 = (b * b).sum(axis=1)[:, None] + Vn2[None, :] - 2.0 * (b @ V.T)
        masked = np.where(d2 <= tau2 + 1e-6, sd[None, :], big)
        out[s : s + chunk] = masked.min(axis=1)
    return {entries[i]["id"]: int(out[i]) for i in range(n)}


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


# ---------------------------------------------------------------------------
# Value-based DVs (Path-A retrofit, Schritt A)
# ---------------------------------------------------------------------------
# Why these exist: the original gate DV (``max_functional_depth``) reads only the
# *geometry* of the global discovery registry. It is blind to reward, adoption
# (``uses``), agent attribution (``discovered_by``) and transmission — so four of
# the five retrofit mechanisms cannot move it *by construction*. The adversarial
# review forced a measurement reform: DVs that score the **realised, rebuilt
# frontier** — function-advancing, adopted, transmitted value — not artifact count.
#
# The task basis B operationalises the design's pre-registered goal set
# {edibility-given-hunger, heat, light, sharpness, low-toxicity, scent}. Each task
# maps the (n,12) property matrix to an (n,) scalar utility in [0,1]; conjunctive
# goals fold "low-toxicity" in as a ``(1 - toxicity)`` safety factor and reduce the
# single-dimension base-saturation of this world's rich seed set (fire already
# saturates heat/light, cooked_* saturate edibility). ``func_tau``/``k``/the basis
# are observer parameters — sweep + pre-register before any unblinded re-pilot.


def _u_edible_safe(V):
    return V[:, IDX["edibility"]] * (1.0 - V[:, IDX["toxicity"]])


def _u_warm_meal(V):
    return V[:, IDX["edibility"]] * V[:, IDX["heat_emission"]] * (1.0 - V[:, IDX["toxicity"]])


def _u_portable_light(V):
    return V[:, IDX["light_emission"]] * (1.0 - V[:, IDX["mass"]])


def _u_cutting_tool(V):
    return V[:, IDX["sharpness"]] * V[:, IDX["hardness"]]


def _u_fragrant(V):
    return V[:, IDX["scent"]]


def _u_heat(V):
    return V[:, IDX["heat_emission"]]


TASK_BASIS = {
    "edible_safe": _u_edible_safe,
    "warm_meal": _u_warm_meal,
    "portable_light": _u_portable_light,
    "cutting_tool": _u_cutting_tool,
    "fragrant": _u_fragrant,
    "heat": _u_heat,
}


def _task_matrix(V: np.ndarray, task_basis: dict) -> np.ndarray:
    """(n,12) property matrix -> (n,T) task-utility matrix in basis order."""
    names = list(task_basis)
    if V.shape[0] == 0 or not names:
        return np.zeros((V.shape[0], len(names)), dtype=float)
    return np.column_stack([np.asarray(task_basis[t](V), dtype=float) for t in names])


def _base_frontier(task_basis: dict, base_vectors) -> np.ndarray:
    """Per-task ceiling reachable from base materials (depth 0)."""
    base = (
        np.array(list(MATERIALS.values()), dtype=np.float32)
        if base_vectors is None
        else np.array(base_vectors, dtype=np.float32)
    )
    Ub = _task_matrix(base, task_basis)
    return Ub.max(axis=0) if Ub.shape[0] else np.zeros(len(task_basis))


def accumulated_useful_depth(
    entries: list[dict],
    struct: dict[str, int] | None = None,
    task_basis: dict = TASK_BASIS,
    margin: float = ADVANCE_MARGIN,
    base_vectors=None,
) -> dict:
    """DV2 — cumulative *useful* depth: an artifact earns depth credit only if it
    advances a task-frontier no shallower artifact (or base material) reached.

    Churn-immune (redundant deep artifacts that don't beat the frontier score 0)
    and arm-symmetric (pure function of vectors + recipes + task basis), so it is
    the DV that can be validated on the existing pilot data without instrumentation.
    """
    names = list(task_basis)
    empty = {
        "accumulated_useful_depth": 0,
        "n_useful_advances": 0,
        "useful_depth_max": 0,
        "per_task": {t: 0 for t in names},
        "advancing_ids": [],
    }
    if not entries:
        return empty
    if struct is None:
        struct = structural_depths(entries)
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    U = _task_matrix(V, task_basis)
    frontier = _base_frontier(task_basis, base_vectors)
    sd = np.array([struct[e["id"]] for e in entries], dtype=int)

    per_task_maxdepth = {t: 0 for t in names}
    advancing_ids: list[str] = []
    for d in sorted(set(sd.tolist())):
        rows = np.where(sd == d)[0]
        layer_adv = U[rows] > (frontier[None, :] + margin)  # (k,T) vs strictly-shallower frontier
        for ti, t in enumerate(names):
            if layer_adv[:, ti].any():
                per_task_maxdepth[t] = d
        advancing_ids.extend(entries[int(r)]["id"] for r in rows[layer_adv.any(axis=1)])
        if rows.size:  # fold this layer in only after scoring it
            frontier = np.maximum(frontier, U[rows].max(axis=0))

    advancing = set(advancing_ids)
    useful_depth_max = max(
        (int(struct[e["id"]]) for e in entries if e["id"] in advancing), default=0
    )
    return {
        "accumulated_useful_depth": int(sum(per_task_maxdepth.values())),
        "n_useful_advances": len(advancing_ids),
        "useful_depth_max": useful_depth_max,
        "per_task": per_task_maxdepth,
        "advancing_ids": advancing_ids,
    }


def graded_useful_advance(
    entries: list[dict],
    struct: dict[str, int] | None = None,
    task_basis: dict = TASK_BASIS,
    margin: float = ADVANCE_MARGIN,
    base_vectors=None,
    depth_weight: bool = False,
) -> dict:
    """M5(a) — non-floored, arm-symmetric, churn-immune functional score.

    Like :func:`accumulated_useful_depth` it walks structural layers shallow->deep
    and only credits an artifact that pushes a task-frontier no shallower artifact
    (or base material) reached. But instead of recording the *depth* at which the
    last advance happened — which saturates (with ``len(task_basis)`` tasks the score
    is capped, and empirically pins at 2 for the embodied arm in both arms) — it sums
    the **margin-normalised magnitude** of every frontier advance over all layers::

        sum_layers sum_tasks  (layer_best - frontier) / margin   for advances > margin

    This has unbounded headroom above the floor yet stays churn-immune (a structurally
    deep but non-advancing artifact contributes 0) and arm-symmetric (a pure function of
    vectors + recipes + task basis; it ignores ``uses``/``discovered_by``), so the
    disembodied recombiner can be neither flattered nor floored by instrumentation.

    ``depth_weight=True`` multiplies each layer's advance by its structural depth ``d``,
    so cumulative ("standing on shoulders") advances dominate. The unweighted form is
    deliberately depth-blind — the E0 embodiment experiment showed it saturates near a
    task-ceiling magnitude regardless of fragmentation, so use ``depth_weight=True`` as
    the discriminating gate DV and the unweighted form only as a saturation diagnostic.
    """
    names = list(task_basis)
    if not entries:
        return {"graded_useful_advance": 0.0, "per_task": {t: 0.0 for t in names}, "n_advancing_layers": 0}
    if struct is None:
        struct = structural_depths(entries)
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    U = _task_matrix(V, task_basis)
    frontier = _base_frontier(task_basis, base_vectors)
    sd = np.array([struct[e["id"]] for e in entries], dtype=int)

    m = float(margin) if margin else 1.0
    per_task = {t: 0.0 for t in names}
    n_layers = 0
    for d in sorted(set(sd.tolist())):
        rows = np.where(sd == d)[0]
        if not rows.size:
            continue
        layer_best = U[rows].max(axis=0)  # (T,) best task-utility on this structural layer
        advance = layer_best - frontier  # vs the strictly-shallower frontier
        w = float(d) if depth_weight else 1.0  # reward advances made at deeper layers
        gained = False
        for ti, t in enumerate(names):
            if advance[ti] > margin:
                per_task[t] += (float(advance[ti]) / m) * w
                gained = True
        if gained:
            n_layers += 1
        frontier = np.maximum(frontier, layer_best)  # fold this layer in AFTER scoring it
    return {
        "graded_useful_advance": float(sum(per_task.values())),
        "per_task": per_task,
        "n_advancing_layers": n_layers,
    }


def population_functional_value(
    entries: list[dict],
    func_tau: float = FUNC_TAU,
    task_basis: dict = TASK_BASIS,
    weight: str = "uses",
) -> dict:
    """DV1 — Σ over functional clusters of (adoption weight · best task-utility).

    On the *current* pilot export the adoption weight ``uses`` is polluted
    (incremented on every ``get_vector`` lookup, not on real homeostatic use) and
    is structurally 0 for the disembodied recombiner. So this DV is returned with
    ``provisional=True``: it floors the recombiner for an *instrumentation* reason,
    not a behavioural one, and must NOT be used for the fairness proof until ``uses``
    is replaced by a clean ``record_use`` signal (Step-1 instrumentation).
    """
    if not entries:
        return {
            "population_functional_value": 0.0,
            "weight_source": weight,
            "provisional": weight == "uses",
            "n_clusters": 0,
            "total_weight": 0.0,
        }
    w = np.array([float(e.get("uses", 0)) for e in entries], dtype=float)
    total_w = float(w.sum())
    if total_w == 0.0:  # recombiner / un-instrumented: value collapses, skip O(n^2) clustering
        return {
            "population_functional_value": 0.0,
            "weight_source": weight,
            "provisional": weight == "uses",
            "n_clusters": 0,
            "total_weight": 0.0,
        }
    V = np.array([e["vector"] for e in entries], dtype=np.float32)
    U = _task_matrix(V, task_basis)
    util = (U.max(axis=1) if U.shape[1] else np.zeros(len(entries))).astype(np.float32)

    # Greedy func_tau clustering with a preallocated, vectorised nearest-rep search
    # (assign each artifact to its closest existing cluster within func_tau, else
    # open a new one) — O(n · n_clusters) in numpy, not an O(n^2) Python loop.
    cap = len(V)
    reps = np.empty((cap, V.shape[1]), dtype=np.float32)
    cl_w = np.zeros(cap, dtype=np.float64)
    cl_u = np.zeros(cap, dtype=np.float64)
    nrep = 0
    tau2 = float(func_tau) ** 2
    for i in range(cap):
        if nrep:
            d2 = ((reps[:nrep] - V[i]) ** 2).sum(axis=1)
            j = int(d2.argmin())
            if d2[j] <= tau2:
                cl_w[j] += w[i]
                cl_u[j] = max(cl_u[j], float(util[i]))
                continue
        reps[nrep] = V[i]
        cl_w[nrep] = w[i]
        cl_u[nrep] = float(util[i])
        nrep += 1

    value = float((cl_w[:nrep] * cl_u[:nrep]).sum())
    return {
        "population_functional_value": value,
        "weight_source": weight,
        "provisional": weight == "uses",
        "n_clusters": nrep,
        "total_weight": total_w,
    }


def transmitted_frontier_advances(
    entries: list[dict],
    struct: dict[str, int] | None = None,
    task_basis: dict = TASK_BASIS,
    margin: float = ADVANCE_MARGIN,
    base_vectors=None,
    k: int = 2,
) -> dict:
    """DV3 — frontier-advancing artifacts that were *also* adopted/transmitted.

    Requires per-agent attribution (``discovered_by``) which the current export
    does not carry (all -1). Returns ``computable=False`` in that case so the gate
    treats it as instrumentation-blocked rather than a real zero. When attribution
    exists, an advance counts if it was created by a real agent and adopted ≥ k
    times (``uses`` as the best available adoption proxy).
    """
    computable = any(int(e.get("discovered_by", -1)) >= 0 for e in entries)
    if not computable:
        return {
            "transmitted_frontier_advances": 0,
            "computable": False,
            "reason": "no per-agent discovered_by/adopter attribution in export",
        }
    aud = accumulated_useful_depth(
        entries, struct=struct, task_basis=task_basis, margin=margin, base_vectors=base_vectors
    )
    idx = _index(entries)
    count = 0
    for mid in aud["advancing_ids"]:
        e = idx[mid]
        if int(e.get("discovered_by", -1)) >= 0 and int(e.get("uses", 0)) >= k:
            count += 1
    return {"transmitted_frontier_advances": count, "computable": True, "k": k}


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
