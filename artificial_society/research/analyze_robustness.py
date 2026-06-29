"""Robustness of the Stage-0a gate to the exploration-volume confound.

``max_functional_depth`` is an extreme-value statistic that grows with sample size.
At equal #attempts the disembodied recombiner makes ~5-6x more discoveries than the
embodied learned arm, so its higher max functional depth could be a *sample-size*
artifact rather than genuinely deeper functional structure. A methodology paper that
indicts confounds must not contain one, so this script controls for it two ways:

1. **Discovery-matched gate** — the registry is append-only, so a prefix ``entries[:k]``
   is exactly its state after ``k`` discoveries. We truncate each recombiner registry to
   the paired learned arm's discovery count and recompute the gate.
2. **Rarefaction curves** — max functional depth vs #discoveries for both arms over the
   common range, averaged across seeds with bootstrap CIs. If the curves coincide, the
   recombiner's apparent win was pure exploration volume; if the recombiner stays above
   at equal #discoveries, it genuinely builds deeper functional structure.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from artificial_society.research.export import load_run
from artificial_society.research.metrics import functional_depths

FUNC_TAU = 0.15


def _collect(outdir: str) -> dict[int, dict]:
    paired: dict[int, dict] = {}
    for arm in ("learned", "recombiner"):
        for p in sorted(glob.glob(os.path.join(outdir, f"{arm}_seed*.json"))):
            seed = int(load_run(p)["meta"]["seed"])
            paired.setdefault(seed, {})[arm] = p
    return {s: v for s, v in paired.items() if "learned" in v and "recombiner" in v}


def _max_fd(entries: list[dict], func_tau: float) -> int:
    if not entries:
        return 0
    return max(functional_depths(entries, func_tau=func_tau).values())


def _bootstrap_ci(vals, n_boot=10000, seed=0):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    rng = np.random.default_rng(seed)
    boots = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _verdict(lv, rv):
    lm, llo, lhi = _bootstrap_ci(lv, seed=1)
    rm, rlo, rhi = _bootstrap_ci(rv, seed=2)
    diff = np.asarray(lv, dtype=float) - np.asarray(rv, dtype=float)
    dm, dlo, dhi = _bootstrap_ci(diff, seed=3)
    sep = llo > rhi
    pos = dlo > 0
    v = "PATH_A" if (sep and pos) else ("BORDERLINE_LEAN_A" if pos else "PATH_B_OR_RETROFIT")
    return (lm, llo, lhi), (rm, rlo, rhi), (dm, dlo, dhi), v


def _print(name, lv, rv):
    (lm, llo, lhi), (rm, rlo, rhi), (dm, dlo, dhi), v = _verdict(lv, rv)
    print(
        f"[{name}] learned {lm:5.2f} [{llo:.2f},{lhi:.2f}] | "
        f"recombiner {rm:5.2f} [{rlo:.2f},{rhi:.2f}] | "
        f"diff {dm:6.2f} [{dlo:.2f},{dhi:.2f}] -> {v}"
    )
    return v


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Confound checks for the Stage-0a gate.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--func-tau", type=float, default=FUNC_TAU)
    args = ap.parse_args(argv)
    tau = args.func_tau

    paired = _collect(args.outdir)
    seeds = sorted(paired)
    learned_entries = {s: load_run(paired[s]["learned"])["entries"] for s in seeds}
    recomb_entries = {s: load_run(paired[s]["recombiner"])["entries"] for s in seeds}

    learned_full = [_max_fd(learned_entries[s], tau) for s in seeds]
    recomb_full = [_max_fd(recomb_entries[s], tau) for s in seeds]
    recomb_matched = [_max_fd(recomb_entries[s][: len(learned_entries[s])], tau) for s in seeds]

    print("=" * 78)
    print(f"ROBUSTNESS — n={len(seeds)} seeds, func_tau={tau}")
    print(
        "discoveries: learned mean="
        f"{np.mean([len(learned_entries[s]) for s in seeds]):.0f} "
        f"recombiner mean={np.mean([len(recomb_entries[s]) for s in seeds]):.0f}"
    )
    print("-" * 78)
    _print("attempt-matched (gate)", learned_full, recomb_full)
    v_disc = _print("discovery-matched     ", learned_full, recomb_matched)
    print("=" * 78)

    # Rarefaction over the common discovery range.
    kmax = min(len(learned_entries[s]) for s in seeds)
    ks = [int(x) for x in np.linspace(max(50, kmax // 12), kmax, 12)]
    L = np.zeros((len(seeds), len(ks)))
    R = np.zeros((len(seeds), len(ks)))
    for i, s in enumerate(seeds):
        for j, k in enumerate(ks):
            L[i, j] = _max_fd(learned_entries[s][:k], tau)
            R[i, j] = _max_fd(recomb_entries[s][:k], tau)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for arr, lab, c in ((R, "random-recombiner", "tab:red"), (L, "learned/social", "tab:blue")):
        m = arr.mean(axis=0)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        ax.plot(ks, m, color=c, label=lab, marker="o", ms=3)
        ax.fill_between(ks, lo, hi, color=c, alpha=0.2)
    ax.set_xlabel("# discoveries (rarefaction, common range)")
    ax.set_ylabel(f"max functional depth (func_tau={tau})")
    ax.set_title(f"Discovery-matched rarefaction (discovery-matched gate: {v_disc})")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(args.outdir, "rarefaction_figure.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
