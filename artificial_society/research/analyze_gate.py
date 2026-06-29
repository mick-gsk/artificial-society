"""Gate analysis — apply the pre-registered decision rule to a pilot run.

Loads paired ``learned_seed<S>.json`` / ``recombiner_seed<S>.json`` exports from a
pilot output directory, computes the functional-complexity DV per run, and applies
the pre-fixed gate rule with bootstrap confidence intervals.

Gate rule (Spec §11 / §14)
--------------------------
Primary DV: ``max_functional_depth`` (functional, *not* raw DAG depth).

* **PATH_A** — the learned/social arm beats the random-recombiner null with
  *separated* per-arm bootstrap CIs **and** a paired-difference CI strictly > 0.
* **BORDERLINE_LEAN_A** — paired-difference CI > 0 but per-arm CIs overlap.
* **PATH_B_OR_RETROFIT** — otherwise: either retrofit P-A (policy-coupling) and
  repeat the pilot, or pivot to the methodology/null-result paper (Path B).

Verdict stability is reported across a small ``func_tau`` sensitivity sweep.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from artificial_society.research import stats
from artificial_society.research.export import load_run
from artificial_society.research.metrics import analyze_registry, structural_depths

# B3: pre-registered significance threshold for the exact paired permutation test.
ALPHA = 0.05

PRIMARY_DV = "max_functional_depth"
# B2: candidate primary DVs whose between-seed variance is compared on the pilot.
CANDIDATE_DVS = ("max_functional_depth", "p95_functional_depth", "mean_functional_depth")
FUNC_TAU_SWEEP = (0.10, 0.15, 0.20)
N_BOOTSTRAP = 10000


def _bootstrap_mean_ci(vals, n_boot=N_BOOTSTRAP, seed=0, lo=2.5, hi=97.5):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, lo)), float(np.percentile(boots, hi))


def _collect(outdir: str) -> dict[int, dict]:
    """Map seed -> {'learned': path, 'recombiner': path} for paired exports."""
    paired: dict[int, dict] = {}
    for arm in ("learned", "recombiner"):
        for path in sorted(glob.glob(os.path.join(outdir, f"{arm}_seed*.json"))):
            meta = load_run(path)["meta"]
            paired.setdefault(int(meta["seed"]), {})[arm] = path
    return {s: v for s, v in paired.items() if "learned" in v and "recombiner" in v}


def _load_cache(paired: dict[int, dict]) -> dict[int, dict]:
    """Load each run's entries + structural depths ONCE (A2 caching).

    ``structural_depths`` is tau-independent, so we compute it a single time per
    ``(seed, arm)`` and reuse it across the whole func_tau sweep instead of
    reloading the JSON and recomputing it once per tau. Returns
    ``{seed: {arm: (entries, struct)}}``.
    """
    cache: dict[int, dict] = {}
    for seed in sorted(paired):
        per_arm = {}
        for arm in ("learned", "recombiner"):
            entries = load_run(paired[seed][arm])["entries"]
            per_arm[arm] = (entries, structural_depths(entries))
        cache[seed] = per_arm
    return cache


def _dv_for_tau(cache: dict[int, dict], func_tau: float, dv: str) -> tuple[list, list, list]:
    """Return (learned_vals, recombiner_vals, seeds) for one func_tau.

    Reuses the cached ``(entries, struct)`` so the only per-tau work is the
    func_tau-dependent neighbourhood scan. DVs are bit-identical to recomputing
    structural depths from scratch each tau.
    """
    learned_vals, recomb_vals, seeds = [], [], []
    for seed in sorted(cache):
        le, ls = cache[seed]["learned"]
        re, rs = cache[seed]["recombiner"]
        learned_vals.append(analyze_registry(le, func_tau, struct=ls)[dv])
        recomb_vals.append(analyze_registry(re, func_tau, struct=rs)[dv])
        seeds.append(seed)
    return learned_vals, recomb_vals, seeds


def _verdict(learned_vals, recomb_vals) -> dict:
    l_mean, l_lo, l_hi = _bootstrap_mean_ci(learned_vals, seed=1)
    r_mean, r_lo, r_hi = _bootstrap_mean_ci(recomb_vals, seed=2)
    diff = np.asarray(learned_vals, dtype=float) - np.asarray(recomb_vals, dtype=float)
    d_mean, d_lo, d_hi = _bootstrap_mean_ci(diff, seed=3)

    separated = l_lo > r_hi
    paired_positive = d_lo > 0
    if separated and paired_positive:
        verdict = "PATH_A"
    elif paired_positive:
        verdict = "BORDERLINE_LEAN_A"
    else:
        verdict = "PATH_B_OR_RETROFIT"

    # B3: exact paired permutation test + standardised effect size + BCa CIs.
    perm = stats.exact_sign_flip_test(diff)
    dz = stats.cohens_dz(diff)
    wilcox_p = stats.wilcoxon_p(diff)
    l_bca = stats.bca_mean_ci(learned_vals, seed=11)
    r_bca = stats.bca_mean_ci(recomb_vals, seed=12)
    d_bca = stats.bca_mean_ci(diff, seed=13)
    perm_sig = bool(np.isfinite(perm["p_value"]) and perm["p_value"] < ALPHA)

    return {
        "verdict": verdict,
        "learned": {"mean": l_mean, "ci": [l_lo, l_hi], "bca_ci": [l_bca[1], l_bca[2]]},
        "recombiner": {"mean": r_mean, "ci": [r_lo, r_hi], "bca_ci": [r_bca[1], r_bca[2]]},
        "paired_diff": {"mean": d_mean, "ci": [d_lo, d_hi], "bca_ci": [d_bca[1], d_bca[2]]},
        "ci_separated": bool(separated),
        "paired_ci_positive": bool(paired_positive),
        # B3 additions (pre-registered) — reported alongside the bootstrap-CI rule.
        "paired_permutation": {"p_value": perm["p_value"], "method": perm["method"]},
        "permutation_significant": perm_sig,
        "cohens_dz": dz,
        "wilcoxon_p": wilcox_p,
        # Pre-registered confirmatory rule: CI-separation AND paired-CI>0 AND exact
        # permutation test significant at ALPHA (see the Stage-0a pre-registration).
        "confirmatory_path_a": bool(separated and paired_positive and perm_sig),
    }


def compare_dvs(outdir: str, func_tau: float | None = None, dvs=CANDIDATE_DVS) -> dict:
    """B2: between-seed variance of each candidate DV on the pilot, ranked by decisiveness.

    Run this on the pilot to choose the lowest-variance / most-decisive primary DV
    and pre-register it for the confirmatory run. The others are reported as
    sensitivity DVs.
    """
    func_tau = FUNC_TAU_SWEEP[1] if func_tau is None else func_tau
    paired = _collect(outdir)
    if not paired:
        raise SystemExit(f"No paired learned/recombiner exports found in {outdir}")
    cache = _load_cache(paired)
    # One analyze_registry per (seed, arm) — all candidate DVs come from the same call
    # (they are just different summaries of the same functional depths). Avoids the
    # 3x-redundant recompute of computing each DV via a separate analyze_registry pass.
    vals = {arm: {dv: [] for dv in dvs} for arm in ("learned", "recombiner")}
    for seed in sorted(cache):
        for arm in ("learned", "recombiner"):
            entries, struct = cache[seed][arm]
            res = analyze_registry(entries, func_tau, struct=struct)
            for dv in dvs:
                vals[arm][dv].append(res[dv])
    summaries = {
        dv: stats.summarize_paired_dv(vals["learned"][dv], vals["recombiner"][dv]) for dv in dvs
    }
    return {
        "func_tau": func_tau,
        "n_seeds": len(paired),
        "summaries": summaries,
        "ranking": stats.rank_dvs(summaries),
    }


def analyze(outdir: str, dv: str = PRIMARY_DV, plot: bool = True) -> dict:
    paired = _collect(outdir)
    if not paired:
        raise SystemExit(f"No paired learned/recombiner exports found in {outdir}")

    cache = _load_cache(paired)  # A2: load entries + structural depths once, reuse across taus
    sweep = {}
    for tau in FUNC_TAU_SWEEP:
        lv, rv, seeds = _dv_for_tau(cache, tau, dv)
        sweep[tau] = {"result": _verdict(lv, rv), "learned": lv, "recombiner": rv, "seeds": seeds}

    primary = sweep[FUNC_TAU_SWEEP[1]]  # func_tau = 0.15
    report = {
        "n_seeds": len(paired),
        "dv": dv,
        "primary_func_tau": FUNC_TAU_SWEEP[1],
        "primary": primary["result"],
        "sensitivity": {tau: sweep[tau]["result"]["verdict"] for tau in FUNC_TAU_SWEEP},
    }

    if plot:
        try:
            _plot(primary, dv, os.path.join(outdir, "gate_figure.png"))
            report["figure"] = os.path.join(outdir, "gate_figure.png")
        except Exception as exc:  # plotting must never break the verdict
            report["figure_error"] = str(exc)

    return report


def _plot(primary: dict, dv: str, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lv = np.asarray(primary["learned"], dtype=float)
    rv = np.asarray(primary["recombiner"], dtype=float)
    res = primary["result"]

    fig, ax = plt.subplots(figsize=(6, 5))
    # paired lines
    for a, b in zip(lv, rv):
        ax.plot([0, 1], [b, a], color="0.8", zorder=1)
    ax.scatter(np.zeros_like(rv), rv, color="tab:red", label="random-recombiner", zorder=2)
    ax.scatter(np.ones_like(lv), lv, color="tab:blue", label="learned/social", zorder=2)
    # means + CI
    for x, arm, color in ((0, "recombiner", "tab:red"), (1, "learned", "tab:blue")):
        m = res[arm]["mean"]
        lo, hi = res[arm]["ci"]
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="D", color=color, capsize=5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["random-\nrecombiner", "learned/\nsocial"])
    ax.set_ylabel(dv)
    ax.set_title(
        f"Gate: {res['verdict']}  (paired diff CI {np.round(res['paired_diff']['ci'], 2)})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _print(report: dict) -> None:
    p = report["primary"]
    print("=" * 64)
    print(
        f"GATE VERDICT: {p['verdict']}   (n={report['n_seeds']} seeds, DV={report['dv']},"
        f" func_tau={report['primary_func_tau']})"
    )
    print("-" * 64)
    print(f"  learned     mean={p['learned']['mean']:.2f}  CI={np.round(p['learned']['ci'], 2)}")
    print(
        f"  recombiner  mean={p['recombiner']['mean']:.2f}  CI={np.round(p['recombiner']['ci'], 2)}"
    )
    print(
        f"  paired diff mean={p['paired_diff']['mean']:.2f}  CI={np.round(p['paired_diff']['ci'], 2)}"
    )
    print(f"  CI separated={p['ci_separated']}  paired CI>0={p['paired_ci_positive']}")
    perm = p["paired_permutation"]
    dz = p["cohens_dz"]
    print(
        f"  exact perm p={perm['p_value']:.4g} ({perm['method']})  Wilcoxon p={p['wilcoxon_p']:.4g}"
    )
    print(f"  Cohen's dz={dz['dz']:.2f}  CI={np.round(dz['ci'], 2)}")
    print(f"  paired diff BCa CI={np.round(p['paired_diff']['bca_ci'], 2)}")
    print(f"  CONFIRMATORY Path A (sep & paired>0 & perm sig)={p['confirmatory_path_a']}")
    print(f"  func_tau sensitivity: {report['sensitivity']}")
    if report.get("figure"):
        print(f"  figure -> {report['figure']}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Apply the Stage-0a gate rule to a pilot run.")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--dv", type=str, default=PRIMARY_DV)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--compare-dvs",
        action="store_true",
        help="B2: report between-seed variance of each candidate DV and exit (no gate)",
    )
    args = p.parse_args(argv)
    if args.compare_dvs:
        rep = compare_dvs(args.outdir)
        print(f"DV variance comparison (func_tau={rep['func_tau']}, n={rep['n_seeds']} seeds)")
        for dv in rep["ranking"]:
            s = rep["summaries"][dv]
            print(
                f"  {dv:24s} mean_diff={s['mean_diff']:.3f} sd_diff={s['sd_diff']:.3f} "
                f"dz={s['dz']:.2f} ci_halfwidth={s['ci_halfwidth']:.3f}"
            )
        print(f"  recommended primary DV (most decisive): {rep['ranking'][0]}")
        return
    report = analyze(args.outdir, dv=args.dv, plot=not args.no_plot)
    _print(report)


if __name__ == "__main__":
    main()
