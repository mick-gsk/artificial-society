"""Life-stage definitions and stat modifiers.

Tick scale (1 season = 900 ticks, AGE_LIMIT = 5000):
  CHILD  :   0 –  500   (~0–15 years)  — dependent, high learning, no reproduction
  ADULT  : 500 – 3200   (~15–80 years) — full capabilities
  ELDER  : 3200– 5000   (~80+ years)   — wisdom bonus, physical decline
"""

CHILD_MAX  = 120
ADULT_MAX  = 3500
# anything above ADULT_MAX is ELDER

STAGE_CHILD  = 'child'
STAGE_ADULT  = 'adult'
STAGE_ELDER  = 'elder'


def get_life_stage(age: int) -> str:
    if age < CHILD_MAX:
        return STAGE_CHILD
    if age < ADULT_MAX:
        return STAGE_ADULT
    return STAGE_ELDER


# ---------------------------------------------------------------------------
# Per-stage multipliers applied in agent.update()
# ---------------------------------------------------------------------------

STAGE_STATS = {
    STAGE_CHILD: {
        # Physical
        'max_energy_mult':     0.60,   # smaller body
        'move_cost_mult':      0.70,   # lighter
        'hydration_loss_mult': 0.75,
        'health_regen_mult':   1.40,   # children heal fast
        'attack_damage_mult':  0.20,   # can barely hurt anyone
        # Cognitive
        'learning_mult':       2.20,   # fast learner
        'sense_radius_mult':   0.60,   # limited awareness
        'memory_retention_mult': 0.90,
        # Gates
        'can_reproduce':       False,
        'can_attack':          False,
        'can_build':           False,
        'foraging_mult':       0.55,   # dependent on adults
    },
    STAGE_ADULT: {
        'max_energy_mult':     1.00,
        'move_cost_mult':      1.00,
        'hydration_loss_mult': 1.00,
        'health_regen_mult':   1.00,
        'attack_damage_mult':  1.00,
        'learning_mult':       1.00,
        'sense_radius_mult':   1.00,
        'memory_retention_mult': 1.00,
        'can_reproduce':       True,
        'can_attack':          True,
        'can_build':           True,
        'foraging_mult':       1.00,
    },
    STAGE_ELDER: {
        # Physical decline
        'max_energy_mult':     0.80,
        'move_cost_mult':      1.35,   # slower
        'hydration_loss_mult': 1.20,   # dehydrates faster
        'health_regen_mult':   0.55,   # slower healing
        'attack_damage_mult':  0.55,
        # Wisdom bonus
        'learning_mult':       0.50,   # learning slows
        'sense_radius_mult':   1.25,   # experienced, better awareness
        'memory_retention_mult': 1.10, # long memory
        'can_reproduce':       False,  # post-reproductive
        'can_attack':          True,
        'can_build':           True,
        'foraging_mult':       0.75,
    },
}


def get_stage_stats(age: int) -> dict:
    return STAGE_STATS[get_life_stage(age)]
