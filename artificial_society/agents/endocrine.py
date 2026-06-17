"""
Endocrine System
----------------
Agents do NOT perceive the world directly. They perceive their own
internal chemical state, which is CAUSED by world events.

This mirrors how biological organisms work:
  - Light does not "tell" the brain it's daytime.
    Light suppresses melatonin -> melatonin drop wakes the organism.
  - Danger does not "tell" the agent to flee.
    Danger triggers cortisol/adrenaline -> those chemicals change behaviour.
  - A herb does not "cure" a disease.
    A herb shifts chemical levels -> those levels happen to counteract illness.

The Brain receives only the 8 hormone levels (float 0..1 each).
It NEVER receives raw labels like 'is_night', 'light', 'disease_level'.
All semantic meaning must be inferred by the agent from correlations
between its own body chemistry and outcomes over time.

Hormones
--------
CORTISOL      Stress / threat response. High -> more energy use, health cost.
              Triggered by: danger, disturbance, starvation, disease.
ADRENALINE   Acute threat. High -> speed boost short term, crash afterward.
              Triggered by: sudden danger spike, attack, predators at night.
MELATONIN    Sleep onset. High -> fatigue, reduced cognition.
              Triggered by: darkness (low light), circadian timer.
SEROTONIN    Wellbeing / satiety. High -> cooperation, reduced aggression.
              Triggered by: good food, social bonding, warmth, sunlight.
DOPAMINE     Reward anticipation. High -> explore / experiment drive.
              Triggered by: novelty, successful foraging, discovery.
OXYTOCIN     Social bonding. High -> increased trust, reduced attack urge.
              Triggered by: proximity to known agents, tribe membership.
INFLAMMATION Disease / injury marker. High -> health drain, sick amplifier.
              Triggered by: active disease, wounds, pollution.
METABOLISM   Energy regulation. High -> efficient digestion, low hunger.
              Triggered by: good hydration, recent food, body heat.

All hormones decay toward a baseline each tick.
External inputs shift them up or down; the agent's body handles the rest.
"""

from __future__ import annotations
import math

# Hormone indices (used as list positions, not labels for the brain)
CORTISOL     = 0
ADRENALINE   = 1
MELATONIN    = 2
SEROTONIN    = 3
DOPAMINE     = 4
OXYTOCIN     = 5
INFLAMMATION = 6
METABOLISM   = 7

N_HORMONES = 8

# Baseline resting levels (0..1)
BASELINE = [0.20, 0.05, 0.10, 0.45, 0.30, 0.25, 0.05, 0.50]

# Decay rate per tick toward baseline (higher = faster return)
DECAY = [0.04, 0.08, 0.03, 0.025, 0.035, 0.020, 0.015, 0.030]

# Clamp bounds
MIN_H = 0.0
MAX_H = 1.0


class EndocrineSystem:
    """
    Maintains 8 hormone floats for one agent.
    Call update() each tick to apply decay + world-driven inputs.
    Call apply_substance() when the agent consumes something.
    """

    __slots__ = ('h',)

    def __init__(self):
        self.h: list[float] = list(BASELINE)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, agent, world):
        """
        Apply all automatic tick-by-tick hormonal influences.
        agent: the Agent dataclass (for energy, health, sick, pos, etc.)
        world: the World object (for cell data and day_state)
        """
        h = self.h
        cell = world.get_cell(*agent.pos)
        dn   = world.day_state

        # ---- inputs from world state (all indirect) ----

        # Light drives melatonin INVERSELY (darkness = high melatonin)
        light = dn.get('light', 1.0)
        target_melatonin = BASELINE[MELATONIN] + 0.75 * (1.0 - light)
        h[MELATONIN] = _nudge(h[MELATONIN], target_melatonin, rate=0.04)

        # Danger / disturbance drives cortisol and adrenaline
        threat = (
            cell.get('danger', 0.0) * 0.008
            + cell.get('disturbance', 0.0) * 0.005
            + dn.get('danger_mult', 1.0) * 0.02
        )
        h[CORTISOL]   = _clamp(h[CORTISOL]   + threat)
        h[ADRENALINE] = _clamp(h[ADRENALINE] + threat * 0.6)

        # Starvation / low energy drives cortisol
        from artificial_society.agents.agent import MAX_ENERGY
        hunger_stress = max(0.0, 0.8 - agent.energy / MAX_ENERGY) * 0.06
        h[CORTISOL] = _clamp(h[CORTISOL] + hunger_stress)

        # Disease / inflammation
        sick_drive = agent.sick / 100.0
        pollution_drive = cell.get('pollution', 0.0) / 100.0
        h[INFLAMMATION] = _clamp(h[INFLAMMATION] + 0.05 * sick_drive + 0.01 * pollution_drive)

        # Inflammation feeds back into cortisol (sickness stress)
        h[CORTISOL] = _clamp(h[CORTISOL] + 0.02 * h[INFLAMMATION])

        # Warmth + sunlight boost serotonin
        warmth = cell.get('warmth', 0.0)
        h[SEROTONIN] = _clamp(h[SEROTONIN] + 0.015 * warmth + 0.01 * light)

        # Good hydration and food boosts metabolism
        fed_score = (
            (agent.hydration / 100.0) * 0.5
            + min(1.0, agent.energy / MAX_ENERGY) * 0.5
        )
        h[METABOLISM] = _nudge(h[METABOLISM], BASELINE[METABOLISM] + 0.4 * fed_score, rate=0.03)

        # Metabolism regulates energy efficiency
        # (handled in agent via endocrine_modifiers)

        # High cortisol suppresses serotonin (stress kills wellbeing)
        if h[CORTISOL] > 0.6:
            h[SEROTONIN] = _clamp(h[SEROTONIN] - 0.02 * (h[CORTISOL] - 0.6))

        # High serotonin suppresses aggression signal via cortisol reduction
        if h[SEROTONIN] > 0.65:
            h[CORTISOL] = _clamp(h[CORTISOL] - 0.01 * (h[SEROTONIN] - 0.65))

        # Social proximity raises oxytocin
        # (caller passes nearby count via apply_social_signal)

        # Dopamine: novelty decay — caller boosts on discovery

        # ---- decay all hormones toward baseline ----
        for i in range(N_HORMONES):
            h[i] = _nudge(h[i], BASELINE[i], rate=DECAY[i])
            h[i] = _clamp(h[i])

    # ------------------------------------------------------------------
    # External signals (called by agent logic)
    # ------------------------------------------------------------------

    def apply_social_signal(self, nearby_count: int, same_tribe: bool):
        """Proximity to known agents raises oxytocin."""
        bond_boost = min(0.15, nearby_count * 0.025)
        if same_tribe:
            bond_boost += 0.04
        self.h[OXYTOCIN] = _clamp(self.h[OXYTOCIN] + bond_boost)
        self.h[SEROTONIN] = _clamp(self.h[SEROTONIN] + bond_boost * 0.3)

    def apply_discovery(self, novelty: float):
        """Successful invention or new causal sequence raises dopamine."""
        self.h[DOPAMINE] = _clamp(self.h[DOPAMINE] + 0.12 * novelty)

    def apply_attack_received(self):
        """Being attacked spikes adrenaline and cortisol."""
        self.h[ADRENALINE] = _clamp(self.h[ADRENALINE] + 0.35)
        self.h[CORTISOL]   = _clamp(self.h[CORTISOL]   + 0.20)

    def apply_successful_forage(self, gain: float):
        """Eating well raises serotonin and metabolism."""
        boost = min(0.12, gain * 0.04)
        self.h[SEROTONIN] = _clamp(self.h[SEROTONIN] + boost)
        self.h[METABOLISM] = _clamp(self.h[METABOLISM] + boost * 0.5)
        self.h[DOPAMINE] = _clamp(self.h[DOPAMINE] + boost * 0.3)

    def apply_substance(self, tag: str, amount: float = 1.0):
        """
        Apply the chemical effects of a consumed substance.
        The agent does NOT know these mappings — it must discover correlations.
        Tags are herb names, food types, or material names.

        Effects are physiological, not semantic:
          willow  -> anti-inflammatory (reduces INFLAMMATION, CORTISOL)
          garlic  -> immune stimulant (reduces INFLAMMATION, boosts METABOLISM)
          elderberry -> antioxidant (boosts SEROTONIN, reduces INFLAMMATION)
          mushroom -> psychoactive (spikes DOPAMINE, can raise or lower CORTISOL)
          moss    -> calming/sleep aid (raises MELATONIN, lowers ADRENALINE)
          raw_meat  -> energy metabolite (boosts METABOLISM, ADRENALINE)
          cooked_meat -> efficient fuel (boosts METABOLISM, SEROTONIN)
          cooked_root -> steady fuel (boosts METABOLISM moderately)
          plant_food  -> light boost SEROTONIN
        """
        a = min(amount, 3.0)  # cap effect
        h = self.h
        if tag == 'herb_willow':
            h[INFLAMMATION] = _clamp(h[INFLAMMATION] - 0.18 * a)
            h[CORTISOL]     = _clamp(h[CORTISOL]     - 0.12 * a)
        elif tag == 'herb_garlic':
            h[INFLAMMATION] = _clamp(h[INFLAMMATION] - 0.14 * a)
            h[METABOLISM]   = _clamp(h[METABOLISM]   + 0.10 * a)
        elif tag == 'herb_elderberry':
            h[SEROTONIN]    = _clamp(h[SEROTONIN]    + 0.12 * a)
            h[INFLAMMATION] = _clamp(h[INFLAMMATION] - 0.10 * a)
        elif tag == 'herb_mushroom':
            # Unpredictable: dopamine spike, cortisol may go either way
            h[DOPAMINE]   = _clamp(h[DOPAMINE]   + 0.20 * a)
            h[CORTISOL]   = _clamp(h[CORTISOL]   + (0.10 - 0.20 * (a % 1.0)) * a)
        elif tag == 'herb_moss':
            h[MELATONIN]  = _clamp(h[MELATONIN]  + 0.15 * a)
            h[ADRENALINE] = _clamp(h[ADRENALINE] - 0.10 * a)
            h[CORTISOL]   = _clamp(h[CORTISOL]   - 0.08 * a)
        elif tag == 'raw_meat':
            h[METABOLISM]  = _clamp(h[METABOLISM]  + 0.08 * a)
            h[ADRENALINE]  = _clamp(h[ADRENALINE]  + 0.05 * a)
        elif tag == 'cooked_meat':
            h[METABOLISM]  = _clamp(h[METABOLISM]  + 0.12 * a)
            h[SEROTONIN]   = _clamp(h[SEROTONIN]   + 0.06 * a)
        elif tag == 'cooked_root':
            h[METABOLISM]  = _clamp(h[METABOLISM]  + 0.08 * a)
        elif tag == 'plant_food':
            h[SEROTONIN]   = _clamp(h[SEROTONIN]   + 0.04 * a)
        elif tag == 'water':
            h[CORTISOL]    = _clamp(h[CORTISOL]    - 0.05 * a)
            h[METABOLISM]  = _clamp(h[METABOLISM]  + 0.06 * a)
        # Unknown substances: no effect (agent must discover via trial)

    # ------------------------------------------------------------------
    # Output: what the Brain actually sees
    # ------------------------------------------------------------------

    def as_features(self) -> list[float]:
        """Return the 8 hormone levels as brain input features."""
        return list(self.h)

    def modifiers(self) -> dict:
        """
        Translate hormone levels into physiological modifiers.
        These are applied by the agent body, NOT visible to the brain.
        The brain sees hormones; the body translates hormones to effects.

        Returns dict with:
          energy_regen   : float multiplier on energy gain
          move_cost_mult : float multiplier on movement cost
          health_drain   : float extra health drain per tick
          sleep_drive    : float 0..1 (high melatonin = sleep pressure)
          forage_eff     : float multiplier on foraging yield
          social_bias    : float added to cooperation signal
          aggression_bias: float shift in attack threshold
          cognition      : float 0..1 multiplier on reward learning rate
        """
        h = self.h
        return {
            # Adrenaline gives short burst but costs extra energy
            'energy_regen':    1.0 + 0.3 * h[METABOLISM] - 0.15 * h[CORTISOL],
            'move_cost_mult':  1.0 + 0.3 * h[ADRENALINE] - 0.1 * h[METABOLISM],
            'health_drain':    0.05 * h[INFLAMMATION] + 0.03 * h[CORTISOL],
            # Melatonin drives sleep; cortisol suppresses it
            'sleep_drive':     max(0.0, h[MELATONIN] - 0.5 * h[CORTISOL]),
            'forage_eff':      0.7 + 0.6 * h[METABOLISM] - 0.2 * h[MELATONIN],
            # Oxytocin and serotonin bias toward social
            'social_bias':     0.3 * h[OXYTOCIN] + 0.2 * h[SEROTONIN],
            # High cortisol/adrenaline lowers attack threshold
            'aggression_bias': 0.4 * h[CORTISOL] + 0.3 * h[ADRENALINE] - 0.3 * h[OXYTOCIN],
            # Dopamine and low cortisol = better learning
            'cognition':       max(0.2, 0.5 + 0.5 * h[DOPAMINE] - 0.4 * h[CORTISOL] - 0.3 * h[MELATONIN]),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _clamp(v: float, lo: float = MIN_H, hi: float = MAX_H) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _nudge(current: float, target: float, rate: float) -> float:
    """Exponential smoothing toward target."""
    return current + rate * (target - current)
