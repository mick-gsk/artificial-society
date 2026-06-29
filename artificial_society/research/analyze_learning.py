"""Does the learned arm *invent* any better than the random operator, or is its
innovation just the recombination operator plus demography?

Uses only data already exported by the pilot (no re-run):

1. **Discovery efficiency vs cumulative attempts**, learned vs recombiner, overlaid.
   New-discoveries-per-attempt declines in *any* cumulative process (dedup saturation),
   so a decline alone proves nothing — the informative comparison is whether the
   embodied learned policy converts attempts into (functional) discoveries any better
   than uniform random recombination. If the curves coincide / learned is below, the
   learned machinery adds no inventive efficiency.
2. **Cumulative discoveries vs attempts** — the same, integrated.
3. **Population** and **energy/agent** over ticks (learned) — the demographic/ecological
   dynamics the innovation is embedded in.

Writes ``learning_figure.png``.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from artificial_society.research.export import load_run

GRID_N = 20


def _collect(outdir: str) -> dict[int, dict]:
    paired: dict[int, dict] = {}
    for arm in ("learned", "recombiner"):
        for p in sorted(glob.glob(os.path.join(outdir, f"{arm}_seed*.json"))):
            seed = int(load_run(p)["meta"]["seed"])
            paired.setdefault(seed, {})[arm] = p
    return {s: v for s, v in paired.items() if "learned" in v and "recombiner" in v}


def _cum(series, xkey):
    """Return (attempts_cumulative, discoveries_cumulative) arrays from a run's series."""
    att = np.array([row[xkey] for row in series], dtype=float)
    disc = np.array([row["n_discoveries"] for row in series], dtype=float)
    return att, disc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)
    paired = _collect(args.outdir)
    seeds = sorted(paired)

    learned = {s: load_run(paired[s]["learned"]) for s in seeds}
    recomb = {s: load_run(paired[s]["recombiner"]) for s in seeds}

    # common cumulative-attempt grid (min over all runs so every curve is defined)
    amax = min(
        min(learned[s]["meta"]["n_attempts"] for s in seeds),
        min(recomb[s]["meta"]["n_attempts"] for s in seeds),
    )
    grid = np.linspace(0, amax, GRID_N + 1)
    mids = 0.5 * (grid[1:] + grid[:-1])

    def curves(runs, xkey):
        cum = np.zeros((len(seeds), len(grid)))
        for i, s in enumerate(seeds):
            att, disc = _cum(runs[s]["series"], xkey)
            cum[i] = np.interp(grid, att, disc)
        eff = np.diff(cum, axis=1) / np.diff(grid)[None, :]
        return cum, eff

    l_cum, l_eff = curves(learned, "n_attempts")
    r_cum, r_eff = curves(recomb, "attempt")

    print("=" * 70)
    print("Discovery efficiency (new discoveries / attempt), mean over the common range:")
    print(f"  learned    : {l_eff.mean():.3f}")
    print(f"  recombiner : {r_eff.mean():.3f}  (ratio {r_eff.mean() / max(l_eff.mean(), 1e-9):.2f}x)")
    print("=" * 70)

    # learned per-tick demography
    ls = [learned[s]["series"] for s in seeds]
    ticks = np.array([row["tick"] for row in ls[0]])
    P = np.array([[row["pop"] for row in s] for s in ls], dtype=float)
    E = np.array([[row["energy"] for row in s] for s in ls], dtype=float)
    epa = E / np.maximum(P, 1.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def band(ax, x, M, c, label):
        ax.plot(x, M.mean(0), color=c, label=label, marker="o", ms=3)
        ax.fill_between(x, np.percentile(M, 2.5, 0), np.percentile(M, 97.5, 0), color=c, alpha=0.2)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    band(ax[0, 0], mids, r_eff, "tab:red", "random-recombiner")
    band(ax[0, 0], mids, l_eff, "tab:blue", "learned/social")
    ax[0, 0].set_xlabel("# combination attempts (cumulative)")
    ax[0, 0].set_ylabel("new discoveries / attempt")
    ax[0, 0].set_title("Inventive efficiency: learned vs random operator")
    ax[0, 0].legend(fontsize=8)

    band(ax[0, 1], grid, r_cum, "tab:red", "random-recombiner")
    band(ax[0, 1], grid, l_cum, "tab:blue", "learned/social")
    ax[0, 1].set_xlabel("# combination attempts (cumulative)")
    ax[0, 1].set_ylabel("# discoveries")
    ax[0, 1].set_title("Discoveries vs attempts (same operator budget)")
    ax[0, 1].legend(fontsize=8)

    band(ax[1, 0], ticks, P, "tab:green", "population")
    ax[1, 0].set_xlabel("tick")
    ax[1, 0].set_ylabel("population")
    ax[1, 0].set_title("Population (demography)")

    band(ax[1, 1], ticks, epa, "tab:orange", "energy/agent")
    ax[1, 1].set_xlabel("tick")
    ax[1, 1].set_ylabel("energy / agent")
    ax[1, 1].set_title("Energy per agent (homeostasis proxy)")

    fig.tight_layout()
    out = os.path.join(args.outdir, "learning_figure.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
