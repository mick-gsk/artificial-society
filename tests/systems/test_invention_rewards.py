"""Unit test for the invention reward rebalance (systems/invention.py).

Replaces the flat novelty-pump assertions with value-gated marginal reward checks.
The ratchet module (_lineage_frontier) is now the signal: reward is proportional
to the marginal gain over the tribe-level frontier, not a flat per-discovery bonus.
"""

from artificial_society.systems import invention as inv
from artificial_society.systems import _lineage_frontier as lf


def test_no_flat_novelty_pump_constant():
    """The flat, value-blind bonus constant has been removed; the ratchet gain is present."""
    assert not hasattr(inv, "NEW_DISCOVERY_BONUS")
    assert hasattr(inv, "RATCHET_GAIN")


def test_marginal_value_is_the_reward_signal(monkeypatch):
    """A higher-value discovery earns positive marginal; repeating the same value earns 0."""
    lf.reset_frontiers()
    # First discovery at value 0.9 → the frontier moves from 0 to 0.9, marginal = 0.9
    m1 = lf.update_frontier(5, 0.9)
    # Same value again → frontier does not move, marginal = 0
    m2 = lf.update_frontier(5, 0.9)
    assert m1 > 0 and m2 == 0.0


def test_lower_value_earns_zero_marginal():
    """A discovery below the current frontier earns zero marginal."""
    lf.reset_frontiers()
    lf.update_frontier(7, 0.8)
    m = lf.update_frontier(7, 0.5)
    assert m == 0.0


def test_ratchet_gain_scale():
    """RATCHET_GAIN must be positive and in a reasonable range to serve as a scaling factor."""
    assert inv.RATCHET_GAIN > 0
    assert inv.RATCHET_GAIN >= 1.0  # must meaningfully amplify genuine progress


def test_legacy_path_still_rewarded_not_removed():
    """Legacy reward constants stay positive (bootstrap survival)."""
    assert inv.COOK_REWARD > 0.0
    assert inv.SHARP_REWARD > 0.0
    assert inv.WARMTH_REWARD > 0.0
    assert inv.LIGHT_REWARD > 0.0
