"""Small-n paired inference for the Stage-0a gate (B3).

The pilot is a *paired* design (same seed, both arms), so all inference is on the
per-seed differences ``d_i = learned_i - recombiner_i``. For the small n typical
of a pilot/confirmatory run, the percentile bootstrap CI the gate started with is
optimistic; this module adds the pre-registered, more defensible tools:

* :func:`exact_sign_flip_test` — the exact paired permutation (sign-flip) test.
  With n seeds there are 2**n sign assignments; we enumerate all of them for an
  *exact* p-value when feasible, falling back to seeded Monte-Carlo sign-flips for
  large n. (Feeding it signed ranks reproduces ``scipy.stats.wilcoxon(mode='exact')``
  exactly — see the tests — so the same engine yields the rank-based variant.)
* :func:`cohens_dz` — the standardised paired effect size with a bootstrap CI.
* :func:`bca_mean_ci` — a bias-corrected-and-accelerated bootstrap CI for the mean
  (materially better than the percentile bootstrap at small n).
* :func:`wilcoxon_p` — the rank-based nonparametric p-value (scipy), reported as a
  pre-registered secondary alongside the sign-flip test.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import bootstrap, nct, norm, t, wilcoxon

_TOL = 1e-9


def exact_sign_flip_test(diffs, max_exact: int = 20, n_mc: int = 20000, seed: int = 0) -> dict:
    """Two-sided exact paired sign-flip permutation test on the mean difference.

    Under the null (the per-seed differences are symmetric about 0) each ``d_i`` is
    equally likely to be ``+d_i`` or ``-d_i``. The statistic is ``T = sum(d_i)``;
    the two-sided p-value is the fraction of the 2**n sign assignments with
    ``|T| >= |T_observed|``. Enumerated exactly for ``n <= max_exact``, otherwise
    estimated from ``n_mc`` seeded random sign-flips via the unbiased
    ``(1 + hits) / (1 + n_mc)`` estimator.
    """
    d = np.asarray(diffs, dtype=float).ravel()
    n = d.size
    if n == 0:
        return {
            "p_value": float("nan"),
            "method": "empty",
            "n": 0,
            "n_perms": 0,
            "observed": float("nan"),
            "statistic": "mean_diff",
        }
    obs = abs(float(d.sum()))

    if n <= max_exact:
        total = 1 << n
        hits = 0
        chunk = 1 << min(n, 18)  # bound peak memory of the sign matrix
        bitpos = np.arange(n, dtype=np.uint64)[None, :]
        for start in range(0, total, chunk):
            rows = np.arange(start, min(start + chunk, total), dtype=np.uint64)[:, None]
            bits = ((rows >> bitpos) & np.uint64(1)).astype(np.int8)
            signs = 1 - 2 * bits  # bit 0 -> +1, bit 1 -> -1
            t = signs @ d
            hits += int((np.abs(t) >= obs - _TOL).sum())
        return {
            "p_value": hits / total,
            "method": "exact",
            "n": n,
            "n_perms": total,
            "observed": float(d.sum()),
            "statistic": "mean_diff",
        }

    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_mc, n), dtype=np.int8) * 2 - 1
    t = signs @ d
    hits = int((np.abs(t) >= obs - _TOL).sum())
    return {
        "p_value": (hits + 1) / (n_mc + 1),
        "method": "monte_carlo",
        "n": n,
        "n_perms": n_mc,
        "observed": float(d.sum()),
        "statistic": "mean_diff",
    }


def cohens_dz(diffs, n_boot: int = 10000, seed: int = 0, confidence: float = 0.95) -> dict:
    """Standardised paired effect size ``dz = mean(d) / sd(d)`` with a bootstrap CI.

    The CI is a seeded percentile bootstrap that discards degenerate resamples (zero
    within-resample SD → ``±inf`` dz). Those infinities carry the sign of the mean, so
    dropping them very slightly narrows the CI on the mean's side — anti-conservative,
    but only material at tiny n (P(all-identical resample) ``= n**(1-n)``: ~1.6% at
    n=4, ~1.5e-12 at n=12). ``n_dropped`` is reported so the effect is visible. The dz
    CI is descriptive only; the gate verdict never depends on it.
    """
    d = np.asarray(diffs, dtype=float).ravel()
    n = d.size
    if n == 0:
        return {"dz": float("nan"), "ci": [float("nan"), float("nan")], "n": 0, "n_dropped": 0}
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        dz = 0.0 if mean == 0.0 else float("inf") * (1.0 if mean > 0 else -1.0)
        return {"dz": float(dz), "ci": [float(dz), float(dz)], "n": n, "n_dropped": 0}

    dz = mean / sd
    rng = np.random.default_rng(seed)
    samp = d[rng.integers(0, n, size=(n_boot, n))]
    with np.errstate(divide="ignore", invalid="ignore"):
        bz = samp.mean(axis=1) / samp.std(axis=1, ddof=1)
    finite = np.isfinite(bz)
    n_dropped = int((~finite).sum())
    bz = bz[finite]
    alpha = (1.0 - confidence) / 2.0 * 100.0
    lo, hi = (
        (float(np.percentile(bz, alpha)), float(np.percentile(bz, 100 - alpha)))
        if bz.size
        else (float("nan"), float("nan"))
    )
    return {"dz": float(dz), "ci": [lo, hi], "n": n, "n_dropped": n_dropped}


def bca_mean_ci(vals, seed: int = 0, confidence: float = 0.95, n_boot: int = 10000):
    """Bias-corrected-and-accelerated (BCa) bootstrap CI for the mean.

    Returns ``(point, lo, hi)``. Deterministic given ``seed``. Constant or
    single-sample inputs collapse to a degenerate ``(point, point, point)``.
    """
    v = np.asarray(vals, dtype=float).ravel()
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(v.mean())
    if n == 1 or np.allclose(v, v[0]):
        return (point, point, point)
    res = bootstrap(
        (v,),
        np.mean,
        method="BCa",
        n_resamples=n_boot,
        confidence_level=confidence,
        random_state=np.random.default_rng(seed),
    )
    return (point, float(res.confidence_interval.low), float(res.confidence_interval.high))


def wilcoxon_p(diffs) -> float:
    """Two-sided Wilcoxon signed-rank p-value (rank-based nonparametric secondary)."""
    d = np.asarray(diffs, dtype=float).ravel()
    if d.size == 0 or np.all(d == 0):
        return float("nan")
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


# --- B2: choose the lowest-variance / most-decisive primary DV ------------------


def summarize_paired_dv(learned_vals, recomb_vals, confidence: float = 0.95) -> dict:
    """Between-seed dispersion of a candidate DV (paired learned vs recombiner).

    Returns the paired-difference mean, its between-seed SD, the standardised
    effect ``dz`` (the scale-free decisiveness used to rank DVs), and the t-based
    CI half-width for the current n. p95 should show a smaller ``sd_diff`` /
    larger ``dz`` than the noisy ``max`` order statistic.
    """
    learned = np.asarray(learned_vals, dtype=float).ravel()
    recomb = np.asarray(recomb_vals, dtype=float).ravel()
    diff = learned - recomb
    n = diff.size
    mean_diff = float(diff.mean()) if n else float("nan")
    sd_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")
    if not n or sd_diff == 0.0:
        dz = 0.0 if (n and mean_diff == 0.0) else float("inf") * (1.0 if mean_diff > 0 else -1.0)
        halfwidth = 0.0 if sd_diff == 0.0 else float("nan")
    else:
        dz = mean_diff / sd_diff
        halfwidth = float(t.ppf(1 - (1 - confidence) / 2, n - 1) * sd_diff / math.sqrt(n))
    return {
        "n": n,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "dz": float(dz),
        "ci_halfwidth": halfwidth,
        "sd_learned": float(learned.std(ddof=1)) if n > 1 else float("nan"),
        "sd_recombiner": float(recomb.std(ddof=1)) if n > 1 else float("nan"),
    }


def rank_dvs(summaries: dict) -> list:
    """Rank candidate DVs by decisiveness — largest effect *magnitude* ``|dz|`` first.

    Magnitude, not signed dz: the most decisive DV is the one that best separates the
    arms regardless of *which* arm wins (the recombiner out-performing the learned arm
    is the very confound this study tests, so negative dz must rank by |dz| too). ``inf``
    (zero-variance perfect separation) ranks first; ``nan`` (undefined, n<2) ranks last.
    """

    def key(name):
        dz = summaries[name]["dz"]
        if np.isnan(dz):
            return float("-inf")
        if np.isinf(dz):
            return float("inf")
        return abs(dz)

    return sorted(summaries, key=key, reverse=True)


# --- B1: required n for a target CI half-width / power --------------------------


def required_n_for_halfwidth(
    sd_diff: float, target_halfwidth: float, confidence: float = 0.95, min_n: int = 2
) -> int:
    """Smallest n so a normal-approx CI on the mean difference has the target half-width.

    ``n = ceil((z * sd_diff / target_halfwidth) ** 2)`` with ``z`` the two-sided
    normal critical value. Used to pre-register the confirmatory sample size from
    the pilot's estimated ``sd_diff``.
    """
    if target_halfwidth <= 0 or sd_diff <= 0:
        raise ValueError("sd_diff and target_halfwidth must be positive")
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    return max(min_n, math.ceil((z * sd_diff / target_halfwidth) ** 2))


def required_n_for_power(
    dz: float,
    power: float = 0.8,
    alpha: float = 0.05,
    min_n: int = 2,
    max_n: int = 100000,
) -> int:
    """Smallest n giving a two-sided paired t-test the target power at effect ``dz``.

    Exact via the noncentral-t distribution: noncentrality ``dz * sqrt(n)``,
    ``df = n - 1``.
    """
    if dz <= 0:
        raise ValueError("dz must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in range(max(min_n, 2), max_n + 1):
            df = n - 1
            ncp = dz * math.sqrt(n)
            tcrit = t.ppf(1 - alpha / 2, df)
            achieved = float(nct.sf(tcrit, df, ncp) + nct.cdf(-tcrit, df, ncp))
            if achieved >= power:
                return n
    return max_n
