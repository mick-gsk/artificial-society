"""B3: exact paired permutation test + effect size + BCa bootstrap.

These pin the small-n inference the gate will pre-register: an *exact* paired
sign-flip permutation test (validated both against an independent enumeration and
against ``scipy.stats.wilcoxon(mode='exact')``), Cohen's dz, and a BCa CI.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import rankdata, wilcoxon

from artificial_society.research import stats


def _brute_sign_flip_p(diffs) -> float:
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    obs = abs(d.sum())
    count = 0
    for s in itertools.product((1, -1), repeat=n):
        if abs((np.asarray(s) * d).sum()) >= obs - 1e-9:
            count += 1
    return count / 2**n


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_exact_sign_flip_matches_independent_enumeration(seed):
    rng = np.random.default_rng(seed)
    diffs = rng.normal(size=9)
    res = stats.exact_sign_flip_test(diffs)
    assert res["method"] == "exact"
    assert res["n_perms"] == 2**9
    assert abs(res["p_value"] - _brute_sign_flip_p(diffs)) < 1e-12


def test_exact_sign_flip_all_positive_is_two_over_2n():
    diffs = [0.1, 0.5, 0.9, 1.3, 2.0]  # all same sign -> only all-+/all-- are as extreme
    res = stats.exact_sign_flip_test(diffs)
    assert res["p_value"] == pytest.approx(2 / 2 ** len(diffs))


def test_exact_permutation_matches_scipy_wilcoxon_exact():
    """The user-required reference check: exact sign-flip perm == exact Wilcoxon."""
    rng = np.random.default_rng(123)
    d = rng.normal(size=11)
    d = d[np.abs(d) > 1e-3]
    # keep |d| tie-free and zero-free so scipy's exact mode is unambiguous
    assert len(np.unique(np.round(np.abs(d), 6))) == len(d)
    signed_ranks = np.sign(d) * rankdata(np.abs(d))
    p_perm = stats.exact_sign_flip_test(signed_ranks)["p_value"]
    p_wilcox = wilcoxon(d, mode="exact").pvalue
    assert abs(p_perm - p_wilcox) < 1e-9


def test_sign_flip_falls_back_to_monte_carlo_for_large_n():
    rng = np.random.default_rng(5)
    diffs = rng.normal(size=30)  # 2^30 is too many -> sampled
    res = stats.exact_sign_flip_test(diffs, max_exact=20)
    assert res["method"] == "monte_carlo"
    assert 0.0 <= res["p_value"] <= 1.0
    # determinism: same seed -> same p
    assert res["p_value"] == stats.exact_sign_flip_test(diffs, max_exact=20)["p_value"]


def test_cohens_dz_value():
    diffs = [1.0, 2.0, 3.0, 4.0]  # mean 2.5, sd(ddof=1) = sqrt(5/3)
    res = stats.cohens_dz(diffs)
    assert res["dz"] == pytest.approx(2.5 / np.sqrt(5.0 / 3.0))
    lo, hi = res["ci"]
    assert lo <= res["dz"] <= hi


def test_cohens_dz_zero_variance():
    res = stats.cohens_dz([2.0, 2.0, 2.0])  # constant positive diff
    assert np.isinf(res["dz"]) and res["dz"] > 0


def test_bca_ci_deterministic_and_brackets_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=3.0, scale=1.0, size=40)
    point, lo, hi = stats.bca_mean_ci(vals, seed=42)
    assert point == pytest.approx(float(np.mean(vals)))
    assert lo < point < hi
    # deterministic given the seed
    assert (point, lo, hi) == stats.bca_mean_ci(vals, seed=42)


def test_bca_ci_degenerate_constant():
    point, lo, hi = stats.bca_mean_ci([4.0, 4.0, 4.0, 4.0], seed=1)
    assert point == lo == hi == 4.0


# --- B2: pick the lowest-variance / most-decisive primary DV --------------------


def test_summarize_paired_dv_values():
    from scipy.stats import t

    learned = [3.0, 4.0, 5.0]
    recomb = [1.0, 1.0, 2.0]
    s = stats.summarize_paired_dv(learned, recomb)
    diff = np.array([2.0, 3.0, 3.0])
    assert s["mean_diff"] == pytest.approx(diff.mean())
    assert s["sd_diff"] == pytest.approx(diff.std(ddof=1))
    assert s["dz"] == pytest.approx(diff.mean() / diff.std(ddof=1))
    n = 3
    expect_hw = t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    assert s["ci_halfwidth"] == pytest.approx(expect_hw)


def test_compare_dvs_picks_most_decisive():
    """Two DVs, SAME mean effect (2.4) but different between-seed SD -> tight wins."""
    tight = stats.summarize_paired_dv([3, 3, 4, 3, 4], [1, 1, 1, 1, 1])  # diff [2,2,3,2,3]
    noisy = stats.summarize_paired_dv([2, 5, 1, 6, 3], [1, 1, 1, 1, 1])  # diff [1,4,0,5,2]
    assert tight["mean_diff"] == pytest.approx(noisy["mean_diff"])  # same effect
    assert tight["sd_diff"] < noisy["sd_diff"]  # tighter dispersion
    ranking = stats.rank_dvs({"noisy": noisy, "tight": tight})
    assert ranking[0] == "tight"  # higher dz (more decisive) ranks first


def test_rank_dvs_uses_effect_magnitude_not_sign():
    """When the recombiner WINS (negative dz), the largest-|dz| DV must still rank first."""
    tight = stats.summarize_paired_dv(
        [1, 1, 1, 1, 1], [3, 3, 4, 3, 4]
    )  # diff very negative, low SD
    noisy = stats.summarize_paired_dv(
        [1, 1, 1, 1, 1], [2, 5, 1, 6, 3]
    )  # diff less negative, high SD
    assert tight["dz"] < 0 and noisy["dz"] < 0
    assert abs(tight["dz"]) > abs(noisy["dz"])
    assert stats.rank_dvs({"noisy": noisy, "tight": tight})[0] == "tight"


# --- B1: required-n for a target CI half-width / power --------------------------


def test_required_n_for_halfwidth_closed_form():
    # n = ceil((z * sd / halfwidth)^2); z_0.975 ~ 1.95996
    assert stats.required_n_for_halfwidth(sd_diff=1.0, target_halfwidth=0.5) == 16
    assert stats.required_n_for_halfwidth(sd_diff=2.0, target_halfwidth=0.5) == 62


def test_required_n_for_halfwidth_monotone():
    n_wide = stats.required_n_for_halfwidth(1.0, 1.0)
    n_narrow = stats.required_n_for_halfwidth(1.0, 0.25)
    assert n_narrow > n_wide >= 2


def test_required_n_for_power_properties():
    n_small_effect = stats.required_n_for_power(dz=0.5)
    n_large_effect = stats.required_n_for_power(dz=1.0)
    assert n_large_effect < n_small_effect
    assert stats.required_n_for_power(dz=0.8, power=0.9) >= stats.required_n_for_power(
        dz=0.8, power=0.8
    )
    # a large effect needs only a handful of pairs
    assert 2 <= stats.required_n_for_power(dz=1.5) <= 12
