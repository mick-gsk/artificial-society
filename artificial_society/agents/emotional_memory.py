"""
Emotional Memory
----------------
Humans do not remember facts neutrally. Emotional events are encoded
stronger, recalled more easily, and shape future behaviour durably.

This module implements a biologically-grounded emotional memory system
with the following human-analogous properties:

1. VALENCE + AROUSAL ENCODING
   Every experience is tagged with:
     valence  (-1..1)  : bad <-> good
     arousal  ( 0..1)  : calm <-> intense
   High arousal events (fear, joy, pain) are encoded more strongly.
   This mirrors the amygdala's role in emotional memory consolidation.

2. FLASHBULB MEMORY
   Extremely arousing events (arousal > 0.75) decay very slowly and
   are almost perfectly consolidated. Analogous to traumatic or
   euphoric flashbulb memories in humans.

3. TRAUMA THRESHOLD
   If cortisol > 0.80 at encoding time, the trace is marked as
   'traumatic'. Traumatic traces never fully decay (floor at 0.20)
   and strongly bias future cortisol responses to similar stimuli.
   Analogous to PTSD-like re-experiencing.

4. EXTINCTION LEARNING
   Safe re-exposure to a feared stimulus gradually reduces fear
   (valence moves toward 0, arousal decays faster).
   This mirrors cognitive-behavioural extinction of conditioned fear.
   Extinction is slow -- requires repeated exposure without negative
   outcome, just as in human therapy.

5. MOOD BASELINE
   A persistent mood float (-1..1) is computed as the EMA of recent
   trace valences, weighted by arousal. This mood:
     - Biases serotonin and dopamine baseline
     - Colours new experience encoding (negative mood = more negative
       encoding of ambiguous events -- depressive realism)
   Analogous to affective priming and mood-congruent memory.

6. CONTEXT-DEPENDENT RETRIEVAL
   Traces stored with similar hormonal context (cortisol, serotonin
   levels at encoding time) are more easily retrieved in matching
   states. Analogous to state-dependent memory in humans.

7. GENERALISATION
   Fear/joy of one stimulus spreads weakly to similar stimuli
   (same category: 'predator', 'herb', 'agent', 'place').
   Analogous to stimulus generalisation in fear conditioning.

8. RECONSOLIDATION
   When a trace is retrieved and the outcome differs from expectation,
   the trace is updated (partially rewritten). Analogous to memory
   reconsolidation in human recall.

Integration
-----------
Called from agent.update() after each tick:
  emotional_memory.encode_experience(stimulus, valence, arousal,
                                     context_hormones, tick)

Endocrine modulation (applied each tick):
  mods = emotional_memory.endocrine_modulations()
  -> cortisol_delta, serotonin_delta, dopamine_delta, adrenaline_delta

Reward signal for brain:
  reward_delta = emotional_memory.reward_signal()

Decay:
  emotional_memory.tick_decay()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants (biologically motivated)
# ---------------------------------------------------------------------------

# Decay rate per tick for a normal (non-traumatic) trace
BASE_DECAY_RATE         = 0.0015   # full trace lasts ~667 ticks
FLASHBULB_DECAY_RATE    = 0.00025  # ~4000 ticks (years in agent time)
TRAUMA_FLOOR            = 0.20     # traumatic traces never drop below this strength
TRAUMA_CORTISOL_THRESH  = 0.78

# Extinction per safe re-exposure
EXTINCTION_RATE         = 0.06
EXTINCTION_MIN_EXPOSURE = 3        # exposures needed before extinction kicks in

# Mood EMA rate
MOOD_EMA_RATE           = 0.012

# Generalisation spread to same-category stimuli
GENERALISATION_STRENGTH = 0.25

# Max traces stored
MAX_TRACES              = 64

# Arousal threshold for flashbulb encoding
FLASHBULB_AROUSAL       = 0.72

# Context similarity threshold for state-dependent retrieval bonus
CONTEXT_MATCH_THRESHOLD = 0.25

# Stimulus categories (used for generalisation)
_CATEGORIES = {
    'predator': {'predator', 'wolf', 'bear', 'snake', 'attack'},
    'food':     {'plant_food', 'meat', 'cooked_meat', 'cooked_root', 'water', 'berry'},
    'herb':     {'herb_willow', 'herb_garlic', 'herb_elderberry', 'herb_mushroom', 'herb_moss'},
    'agent':    {'agent_trust', 'agent_attack', 'agent_share', 'agent_mate'},
    'place':    {'home', 'camp', 'fire', 'territory'},
    'weather':  {'cold', 'heat', 'rain', 'drought'},
}

def _category(stimulus: str) -> Optional[str]:
    for cat, members in _CATEGORIES.items():
        if stimulus in members or stimulus.startswith(cat):
            return cat
    return None


# ---------------------------------------------------------------------------
# Single emotional trace
# ---------------------------------------------------------------------------
@dataclass
class EmotionalTrace:
    """
    One emotional memory trace.

    stimulus        : str   -- what caused this emotion
    valence         : float -- -1 (bad) to +1 (good)
    arousal         : float -- 0 (calm) to 1 (intense)
    strength        : float -- 0..1, decays over time
    decay_rate      : float -- ticks-based decay
    traumatic       : bool  -- never fully fades
    consolidation   : float -- 0..1, how well consolidated
    tick_encoded    : int
    context_hormones: list  -- [cortisol, serotonin, dopamine, adrenaline] at encoding
    extinction_count: int   -- safe re-exposures so far
    category        : str
    last_retrieved  : int
    """
    stimulus: str
    valence: float
    arousal: float
    strength: float
    decay_rate: float
    traumatic: bool
    consolidation: float
    tick_encoded: int
    context_hormones: list  # [cortisol, serotonin, dopamine, adrenaline]
    extinction_count: int = 0
    category: str = 'unknown'
    last_retrieved: int = 0

    def effective_valence(self) -> float:
        """Valence weighted by current strength and consolidation."""
        return self.valence * self.strength * self.consolidation

    def effective_arousal(self) -> float:
        return self.arousal * self.strength


# ---------------------------------------------------------------------------
# EmotionalMemory
# ---------------------------------------------------------------------------
class EmotionalMemory:
    """
    Maintains a bank of EmotionalTraces for one agent.
    """

    def __init__(self):
        self.traces: list[EmotionalTrace] = []
        self.mood: float = 0.0          # persistent affective baseline -1..1
        self._tick: int  = 0

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_experience(
        self,
        stimulus: str,
        valence: float,
        arousal: float,
        context_hormones: list,  # [cortisol, serotonin, dopamine, adrenaline]
        tick: int,
        mood_bias: bool = True,
    ) -> EmotionalTrace:
        """
        Encode a new emotional experience.

        Flashbulb: high arousal -> slow decay, high consolidation.
        Trauma:    cortisol > threshold -> traumatic flag.
        Mood bias: negative mood slightly darkens ambiguous valence.
        Generalisation: spread weak trace to category siblings.

        If an existing trace for same stimulus exists, reconsolidate
        instead of creating duplicate.
        """
        cortisol = context_hormones[0] if len(context_hormones) > 0 else 0.2

        # Mood-congruent encoding bias
        if mood_bias and abs(valence) < 0.4:
            valence += 0.15 * self.mood  # ambiguous events coloured by mood
        valence = max(-1.0, min(1.0, valence))

        # Arousal amplifies encoding strength
        base_strength   = 0.40 + 0.60 * arousal
        consolidation   = 0.50 + 0.50 * arousal

        # Flashbulb
        if arousal >= FLASHBULB_AROUSAL:
            decay_rate    = FLASHBULB_DECAY_RATE
            consolidation = min(1.0, consolidation + 0.20)
        else:
            decay_rate    = BASE_DECAY_RATE * (1.0 - 0.5 * arousal)

        # Trauma
        traumatic = cortisol >= TRAUMA_CORTISOL_THRESH and arousal >= 0.55
        if traumatic:
            decay_rate  = FLASHBULB_DECAY_RATE * 0.5
            base_strength = 1.0
            consolidation = 1.0

        cat = _category(stimulus) or 'unknown'

        # Reconsolidation: update existing trace if same stimulus
        existing = self._find_trace(stimulus)
        if existing is not None:
            # Blend new experience into existing trace
            blend = 0.35
            existing.valence      = (1-blend)*existing.valence + blend*valence
            existing.arousal      = max(existing.arousal, arousal)
            existing.strength     = min(1.0, existing.strength + 0.20 * base_strength)
            existing.consolidation= min(1.0, (existing.consolidation + consolidation) * 0.5 + 0.1)
            existing.last_retrieved = tick
            if traumatic:
                existing.traumatic = True
                existing.decay_rate = FLASHBULB_DECAY_RATE * 0.5
            self._update_mood(valence, arousal)
            return existing

        trace = EmotionalTrace(
            stimulus=stimulus,
            valence=valence,
            arousal=arousal,
            strength=base_strength,
            decay_rate=decay_rate,
            traumatic=traumatic,
            consolidation=consolidation,
            tick_encoded=tick,
            context_hormones=list(context_hormones[:4]),
            category=cat,
        )
        self._add_trace(trace)
        self._update_mood(valence, arousal)

        # Generalisation: spread weakly to same-category stimuli
        if cat != 'unknown':
            self._generalise(cat, valence, arousal * GENERALISATION_STRENGTH, tick, context_hormones)

        return trace

    # ------------------------------------------------------------------
    # Extinction
    # ------------------------------------------------------------------
    def safe_exposure(
        self,
        stimulus: str,
        tick: int,
    ) -> bool:
        """
        Record a safe re-exposure to a previously feared stimulus.
        Returns True if extinction reduced fear.

        Must be called when agent encounters stimulus without negative
        outcome (e.g. approaches predator territory safely, eats herb
        without illness).
        """
        trace = self._find_trace(stimulus)
        if trace is None or trace.valence >= 0.0:
            return False  # not feared

        trace.extinction_count += 1
        if trace.extinction_count < EXTINCTION_MIN_EXPOSURE:
            return False

        # Gradual extinction: valence moves toward 0, arousal decays faster
        trace.valence  = min(0.0, trace.valence + EXTINCTION_RATE)
        trace.arousal  = max(0.0, trace.arousal  - EXTINCTION_RATE * 0.5)
        trace.strength = max(
            TRAUMA_FLOOR if trace.traumatic else 0.0,
            trace.strength - EXTINCTION_RATE * 0.3,
        )
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        stimulus: str,
        current_hormones: list,
        tick: int,
    ) -> Optional[EmotionalTrace]:
        """
        Retrieve trace for stimulus, boosted if current hormonal context
        matches encoding context (state-dependent memory).

        Side-effect: retrieval strengthens the trace slightly
        (retrieval-induced potentiation).
        """
        trace = self._find_trace(stimulus)
        if trace is None:
            return None

        # State-dependent boost
        context_sim = self._context_similarity(trace.context_hormones, current_hormones)
        if context_sim > CONTEXT_MATCH_THRESHOLD:
            trace.strength = min(1.0, trace.strength + 0.02)

        trace.last_retrieved = tick
        return trace

    def fear_of(
        self,
        stimulus: str,
        current_hormones: list,
        tick: int,
    ) -> float:
        """
        Return fear level 0..1 for a stimulus.
        0 = no fear / positive association.
        """
        trace = self.retrieve(stimulus, current_hormones, tick)
        if trace is None:
            return 0.0
        fear = max(0.0, -trace.effective_valence())
        return min(1.0, fear)

    def desire_of(
        self,
        stimulus: str,
        current_hormones: list,
        tick: int,
    ) -> float:
        """
        Return desire level 0..1 for a stimulus.
        0 = no desire / negative association.
        """
        trace = self.retrieve(stimulus, current_hormones, tick)
        if trace is None:
            return 0.0
        desire = max(0.0, trace.effective_valence())
        return min(1.0, desire)

    def most_feared(self, current_hormones: list, tick: int) -> Optional[str]:
        """Return stimulus with highest current fear."""
        if not self.traces:
            return None
        fears = [
            (t.stimulus, self.fear_of(t.stimulus, current_hormones, tick))
            for t in self.traces if t.valence < 0
        ]
        if not fears:
            return None
        return max(fears, key=lambda x: x[1])[0]

    def most_desired(self, current_hormones: list, tick: int) -> Optional[str]:
        """Return stimulus with highest current desire."""
        if not self.traces:
            return None
        desires = [
            (t.stimulus, self.desire_of(t.stimulus, current_hormones, tick))
            for t in self.traces if t.valence > 0
        ]
        if not desires:
            return None
        return max(desires, key=lambda x: x[1])[0]

    # ------------------------------------------------------------------
    # Endocrine modulations
    # ------------------------------------------------------------------
    def endocrine_modulations(self) -> dict:
        """
        Compute per-tick hormone deltas driven by emotional memory state.

        Returns dict with small float deltas to add to hormone levels.
        These are applied by agent.py each tick.

        Biological analogues:
          Traumatic traces -> elevated cortisol baseline (PTSD-like)
          Positive mood    -> serotonin + dopamine boost
          Negative mood    -> cortisol elevation, serotonin suppression
          High fear traces -> adrenaline priming near fear stimuli
          Flashbulb traces -> contextual adrenaline when retrieved
        """
        cortisol_delta   = 0.0
        serotonin_delta  = 0.0
        dopamine_delta   = 0.0
        adrenaline_delta = 0.0

        for t in self.traces:
            ev = t.effective_valence()
            ea = t.effective_arousal()

            if t.traumatic and t.strength > TRAUMA_FLOOR:
                # Chronic low-level cortisol elevation from trauma
                cortisol_delta   += 0.004 * t.strength
                adrenaline_delta += 0.002 * t.strength

            if ev < -0.2:
                # Fear/pain memory -> cortisol up, serotonin down
                cortisol_delta  += 0.003 * abs(ev)
                serotonin_delta -= 0.002 * abs(ev)
            elif ev > 0.2:
                # Positive memory -> serotonin + dopamine up
                serotonin_delta += 0.002 * ev
                dopamine_delta  += 0.001 * ev

        # Mood effect on baseline
        if self.mood > 0.1:
            serotonin_delta += 0.005 * self.mood
            dopamine_delta  += 0.003 * self.mood
        elif self.mood < -0.1:
            cortisol_delta  += 0.005 * abs(self.mood)
            serotonin_delta -= 0.004 * abs(self.mood)

        # Clamp deltas to avoid runaway
        def _c(v): return max(-0.04, min(0.04, v))
        return {
            'cortisol':   _c(cortisol_delta),
            'serotonin':  _c(serotonin_delta),
            'dopamine':   _c(dopamine_delta),
            'adrenaline': _c(adrenaline_delta),
        }

    # ------------------------------------------------------------------
    # Reward signal for brain
    # ------------------------------------------------------------------
    def reward_signal(self) -> float:
        """
        Small reward/penalty based on emotional memory state.
        Positive mood contributes a small positive reward each tick.
        Traumatic active traces contribute a penalty.
        This biases the brain toward mood-improving behaviours.
        """
        r = 0.015 * self.mood
        trauma_penalty = sum(
            0.008 * t.strength
            for t in self.traces
            if t.traumatic and t.strength > TRAUMA_FLOOR + 0.05
        )
        return r - trauma_penalty

    # ------------------------------------------------------------------
    # Decay (called every tick)
    # ------------------------------------------------------------------
    def tick_decay(self, tick: int) -> None:
        """
        Decay all traces. Remove fully decayed non-traumatic traces.
        Traumatic traces floor at TRAUMA_FLOOR.
        """
        self._tick = tick
        surviving = []
        for t in self.traces:
            t.strength -= t.decay_rate
            if t.traumatic:
                t.strength = max(TRAUMA_FLOOR, t.strength)
                surviving.append(t)
            elif t.strength > 0.01:
                surviving.append(t)
        self.traces = surviving

        # Mood decays slowly toward 0 (hedonic adaptation)
        self.mood *= (1.0 - 0.003)

    # ------------------------------------------------------------------
    # Inheritance (called at birth from parent)
    # ------------------------------------------------------------------
    def inherit_from(
        self,
        parent_em: 'EmotionalMemory',
        strength_factor: float = 0.30,
    ) -> None:
        """
        Child inherits a blurred copy of parent's strongest emotional
        traces. This implements:
          - Epigenetic trauma transmission (parent's fear -> child's
            elevated baseline cortisol for feared stimuli)
          - Positive cultural priming (parent's joy -> child's affinity)

        Only highly consolidated, strong parent traces are transmitted.
        Strength is multiplied by strength_factor (default 0.30).
        """
        for t in sorted(parent_em.traces, key=lambda x: -x.strength * x.consolidation)[:8]:
            if t.strength * t.consolidation < 0.35:
                continue
            child_trace = EmotionalTrace(
                stimulus=t.stimulus,
                valence=t.valence * 0.6,   # muted but directional
                arousal=t.arousal * 0.5,
                strength=t.strength * strength_factor,
                decay_rate=BASE_DECAY_RATE,  # not inherited as flashbulb
                traumatic=False,             # trauma not directly inherited
                consolidation=t.consolidation * 0.5,
                tick_encoded=self._tick,
                context_hormones=[0.2, 0.45, 0.3, 0.05],  # default baseline
                category=t.category,
            )
            self._add_trace(child_trace)
        # Mood inheritance: very weak
        self.mood = parent_em.mood * 0.15

    # ------------------------------------------------------------------
    # Summary / debug
    # ------------------------------------------------------------------
    def summary(self) -> list[dict]:
        return [
            {
                'stimulus':      t.stimulus,
                'valence':       round(t.valence, 2),
                'arousal':       round(t.arousal, 2),
                'strength':      round(t.strength, 2),
                'traumatic':     t.traumatic,
                'consolidation': round(t.consolidation, 2),
                'category':      t.category,
                'extinctions':   t.extinction_count,
            }
            for t in sorted(self.traces, key=lambda x: -x.strength)
        ]

    @property
    def mood_label(self) -> str:
        if self.mood >  0.35: return 'content'
        if self.mood >  0.10: return 'neutral_positive'
        if self.mood < -0.35: return 'distressed'
        if self.mood < -0.10: return 'neutral_negative'
        return 'neutral'

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _add_trace(self, trace: EmotionalTrace) -> None:
        self.traces.append(trace)
        if len(self.traces) > MAX_TRACES:
            # Evict weakest non-traumatic trace
            non_trauma = [t for t in self.traces if not t.traumatic]
            if non_trauma:
                weakest = min(non_trauma, key=lambda t: t.strength * t.consolidation)
                self.traces.remove(weakest)

    def _find_trace(self, stimulus: str) -> Optional[EmotionalTrace]:
        for t in self.traces:
            if t.stimulus == stimulus:
                return t
        return None

    def _update_mood(self, valence: float, arousal: float) -> None:
        """EMA update of mood baseline, weighted by arousal."""
        weight = 0.3 + 0.7 * arousal
        effective_rate = MOOD_EMA_RATE * weight
        self.mood = self.mood * (1.0 - effective_rate) + valence * effective_rate
        self.mood = max(-1.0, min(1.0, self.mood))

    def _generalise(
        self,
        category: str,
        valence: float,
        weak_arousal: float,
        tick: int,
        context_hormones: list,
    ) -> None:
        """
        Apply weak generalisation: all existing traces of the same
        category shift slightly toward the new valence.
        """
        for t in self.traces:
            if t.category == category and t.stimulus != 'generalisation':
                blend = GENERALISATION_STRENGTH * 0.15
                t.valence = (1 - blend) * t.valence + blend * valence

    def _context_similarity(
        self,
        h1: list,
        h2: list,
    ) -> float:
        """
        Cosine-like similarity between two hormone context vectors.
        Both are [cortisol, serotonin, dopamine, adrenaline].
        """
        if not h1 or not h2:
            return 0.0
        n = min(len(h1), len(h2))
        dot  = sum(h1[i] * h2[i] for i in range(n))
        mag1 = math.sqrt(sum(x*x for x in h1[:n])) or 1e-9
        mag2 = math.sqrt(sum(x*x for x in h2[:n])) or 1e-9
        return dot / (mag1 * mag2)

    def __repr__(self) -> str:
        return (
            f"EmotionalMemory(traces={len(self.traces)}, "
            f"mood={self.mood:.2f} [{self.mood_label}])"
        )
