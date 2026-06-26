"""Unit test for the invention reward rebalance (systems/invention.py).

Locks the intended reward relationship after rebalancing: genuine emergent
vector discoveries must be rewarded *competitively with or above* the legacy
scripted recipes (sharp_stone, cooked_meat, fire/warmth, light, tinder), so the
simulation funnels agents toward open-ended emergence rather than the 5 hardcoded
outcomes — while keeping the legacy path alive for early-survival bootstrap.

Fast and deterministic: asserts directly on the reward constants, no Simulation run.
"""
from artificial_society.systems import invention as inv
from artificial_society.systems import need_driven_invention as ndi


def test_new_discovery_dominates_every_legacy_outcome():
    """A genuinely-new discovery's bonus must exceed every legacy reward."""
    legacy_rewards = [
        inv.COOK_REWARD,
        inv.SHARP_REWARD,
        inv.WARMTH_REWARD,
        inv.LIGHT_REWARD,
        inv.LEGACY_MAX_REWARD,
    ]
    for r in legacy_rewards:
        assert r < inv.NEW_DISCOVERY_BONUS
    # Comfortable margin: a new discovery is the single most valuable event.
    assert inv.NEW_DISCOVERY_BONUS >= 2.0 * inv.LEGACY_MAX_REWARD


def test_rediscovery_is_competitive_with_strongest_legacy():
    """Repeating a self-discovered emergent recipe is never worse than a script recipe."""
    # Re-discovery reward must be >= the strongest reliable legacy outcome,
    # so an agent has no incentive to abandon emergent recipes for scripted ones.
    assert inv.REDISCOVERY_REWARD >= inv.LEGACY_MAX_REWARD
    assert inv.REDISCOVERY_REWARD >= inv.COOK_REWARD
    assert inv.REDISCOVERY_REWARD >= inv.SHARP_REWARD


def test_legacy_path_still_rewarded_not_removed():
    """Legacy recipes stay positive (bootstrap survival), just no longer dominant."""
    assert inv.COOK_REWARD > 0.0
    assert inv.SHARP_REWARD > 0.0
    assert inv.WARMTH_REWARD > 0.0
    assert inv.LIGHT_REWARD > 0.0


def test_need_driven_emergent_path_uses_full_weight_and_bonus():
    """Need-driven invention must apply the same discovery bonus and full weight."""
    # Emergent contribution is no longer down-weighted relative to the legacy path.
    assert ndi.EMERGENT_WEIGHT >= 1.0
    # The need-driven path imports the same discovery economics as invention.py.
    assert ndi.NEW_DISCOVERY_BONUS == inv.NEW_DISCOVERY_BONUS
    assert ndi.REDISCOVERY_REWARD == inv.REDISCOVERY_REWARD
