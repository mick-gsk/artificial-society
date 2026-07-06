"""B1 regression: the plant carrying-capacity knob must be a REAL, live lever.

The vectorized regrow path reads ``world._bio["plant_ceiling"]`` — a per-biome
array frozen at world construction — so patching ``SCARCITY_CEILING_FACTOR``
after a world is built is a silent no-op, and scaling the *inflow* rate
``FOOD_SCARCITY_FACTOR`` only changes regrowth SPEED, not the equilibrium
standing stock (which is pinned by ``plant_headroom -> 0`` at
``plant_ceiling * capacity``). ``PLANT_CEILING_SCALE`` exposes the standing
ceiling itself as a call-time-patchable module-level knob.
"""

import artificial_society.environment.resources as R
from artificial_society.world import World


def _equilibrium_mean_plant(ticks=3000):
    """Fresh world, pure regrowth (no consumption) to equilibrium; mean plant_food
    over land cells."""
    w = World(24, 18)
    ss = {"food_factor": 1.0, "temperature_shift": 0.0}
    ws = {"rain_map": 0.0, "temperature_shift": 0.0, "wind": 0.0, "storm_risk": 0.0}
    ef = w.event_fields_grid()
    for t in range(ticks):
        R.regrow_grid(w, ss, ws, t, ef)
    land = ~w._bio["is_water"]
    return float(w.F["plant_food"][land].mean())


def test_plant_ceiling_scale_raises_equilibrium_standing_stock():
    """Scaling PLANT_CEILING_SCALE measurably raises equilibrium standing plant_food."""
    saved = R.PLANT_CEILING_SCALE
    try:
        R.PLANT_CEILING_SCALE = 1.0
        base = _equilibrium_mean_plant()
        R.PLANT_CEILING_SCALE = 2.0
        scaled = _equilibrium_mean_plant()
    finally:
        R.PLANT_CEILING_SCALE = saved
    # Doubling the ceiling should lift the equilibrium standing stock substantially
    # (near-linear in the ceiling; allow slack for stress/temperature coupling).
    assert scaled > base * 1.5, f"ceiling knob did not bind: base={base}, scaled={scaled}"


def test_inflow_rate_alone_does_not_move_carrying_capacity():
    """Documents the B1 no-op: tripling the inflow rate barely moves equilibrium."""
    saved_food = R.FOOD_SCARCITY_FACTOR
    saved_ceil = R.PLANT_CEILING_SCALE
    try:
        R.PLANT_CEILING_SCALE = 1.0
        R.FOOD_SCARCITY_FACTOR = 0.50
        base = _equilibrium_mean_plant()
        R.FOOD_SCARCITY_FACTOR = 0.50 * 3.0
        faster = _equilibrium_mean_plant()
    finally:
        R.FOOD_SCARCITY_FACTOR = saved_food
        R.PLANT_CEILING_SCALE = saved_ceil
    # Equilibrium is set by the ceiling, not the inflow rate: <5% change.
    assert faster < base * 1.05, (
        f"inflow rate unexpectedly moved equilibrium: base={base}, faster={faster}"
    )
