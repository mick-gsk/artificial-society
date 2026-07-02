"""Observability for the causal-model prototype (flag ``AS_CAUSAL_MODEL``).

Run from the repo root:
    ../venv/bin/python docs/research/causal-model-proto/observe_causal_model.py

PROTOTYPE / illustrative only — NOT wired into the M1–M6 experiment matrix. Surfaces
three things the design promised:
  * transfer: does an agent's learned model beat a predict-the-mean baseline on material
    pairs (mostly) never combined?  (the causal-theory test)
  * L2 readability: a finite-difference sensitivity "theory" table from a black-box MLP.
  * behaviour: discovery count with the flag ON vs OFF (effect of L3 active experimentation).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from artificial_society.environment.materials import (  # noqa: E402
    DISCOVERY_REGISTRY,
    MATERIALS,
    PROP_DIMS,
    combine_vectors,
    get_vector,
)
from artificial_society.rng import seed_all  # noqa: E402
from artificial_society.systems.invention import PRIMITIVE_ACTIONS  # noqa: E402

ENV = {"wind": 0.3, "moisture": 0.5, "temperature": 15}
N_PROPS = len(PROP_DIMS)


def run_sim(enabled: bool, seed: int = 1234, ticks: int = 400, pop: int = 12):
    if enabled:
        os.environ["AS_CAUSAL_MODEL"] = "1"
    else:
        os.environ.pop("AS_CAUSAL_MODEL", None)
    DISCOVERY_REGISTRY.reset()
    seed_all(seed)
    from artificial_society.simulation import Simulation

    sim = Simulation(headless=True, grid_w=20, grid_h=15, initial_population=pop)
    for _ in range(ticks):
        sim.step()
    n_disc = len(DISCOVERY_REGISTRY.entries)
    models = [m for m in (getattr(a, "causal_model", None) for a in sim.agents) if m is not None]
    # materials the agents actually experience (their plausible input distribution) — a
    # fair, in-distribution transfer probe instead of uniform-random over ALL materials
    experienced = set()
    for a in sim.agents:
        experienced.update(k for k in getattr(a, "material_inventory", {}) if not k.startswith("_"))
    experienced.update(m for m in MATERIALS if not m.startswith("mat_"))
    return n_disc, models, sorted(experienced)


def transfer_eval(model, names, n_pairs: int = 300, seed: int = 0):
    """Model error vs predict-the-mean baseline on pairs drawn from the agents' experienced
    material distribution -> beating the mean requires generalizing within that manifold."""
    rng = np.random.default_rng(seed)
    if len(names) < 2:
        names = list(MATERIALS.keys())
    targets, model_errs = [], []
    for _ in range(n_pairs):
        na, nb = names[rng.integers(len(names))], names[rng.integers(len(names))]
        ai = int(rng.integers(len(PRIMITIVE_ACTIONS)))
        va, vb = get_vector(na), get_vector(nb)
        out = combine_vectors(va, vb, PRIMITIVE_ACTIONS[ai], ENV)
        y = out if out is not None else np.zeros(N_PROPS, dtype=np.float32)
        targets.append(np.asarray(y, dtype=np.float64))
        model_errs.append(float(np.linalg.norm(model.predict(va, vb, ai) - y)))
    targets = np.array(targets)
    mean_y = targets.mean(axis=0)
    base_errs = [float(np.linalg.norm(mean_y - y)) for y in targets]
    return float(np.mean(model_errs)), float(np.mean(base_errs))


def print_theory(model, action: str = "rub", topk: int = 6):
    """L2: read out the strongest input-property -> output-property couplings the MLP
    learned, around a mid-range operating point."""
    a = np.full(N_PROPS, 0.5, dtype=np.float32)
    b = np.full(N_PROPS, 0.5, dtype=np.float32)
    jac = model.sensitivity(a, b, PRIMITIVE_ACTIONS.index(action))  # (2*N_PROPS, N_PROPS)
    flat = []
    for inp in range(2 * N_PROPS):
        side = "a" if inp < N_PROPS else "b"
        in_name = PROP_DIMS[inp % N_PROPS]
        for out in range(N_PROPS):
            flat.append((abs(jac[inp, out]), f"{in_name}({side})", PROP_DIMS[out], jac[inp, out]))
    flat.sort(reverse=True)
    print(f"  learned 'theory' for action={action!r} (top {topk} couplings):")
    for _, inp, outp, val in flat[:topk]:
        print(f"    {inp:>16} --> {outp:<14} d={val:+.3f}")


def main():
    print("== causal-model prototype — observability ==\n")
    off_disc, _, _ = run_sim(enabled=False)
    on_disc, models, exp_names = run_sim(enabled=True)
    print("\n--- behaviour (L3 active experimentation) ---")
    print(f"  discoveries  OFF={off_disc}   ON={on_disc}")
    print(f"  agents carrying a causal model after ON run: {len(models)}")

    if not models:
        print("\n(no agent built a model this run — raise ticks/pop and retry)")
        return
    model = max(models, key=lambda m: sum(m._bucket_counts.values()))
    obs = sum(model._bucket_counts.values())
    print(f"\n--- transfer (causal-theory test) — best agent, {obs} observations ---")
    m_err, b_err = transfer_eval(model, exp_names)
    verdict = "GENERALIZES" if m_err < b_err else "no transfer"
    print(f"  model error={m_err:.4f}   mean-baseline error={b_err:.4f}   -> {verdict}")
    print("\n--- L2 readability (sensitivity probe) ---")
    print_theory(model, action="rub")


if __name__ == "__main__":
    main()
