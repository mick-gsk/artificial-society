"""
KnowledgeGraph + EpisodicMemory
--------------------------------
EpisodicMemory  – NGU-style k-NN novelty score (state-space distance,
                   not activation magnitude).
KnowledgeGraph  – Persistent causal fact store.  Agents record which
                   (action, mat_a, mat_b) combinations succeeded or
                   failed, accumulate confidence, and share knowledge
                   via inheritance and imitation.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque

import torch


# ---------------------------------------------------------------------------
# EpisodicMemory – novelty via k-NN distance in observation space
# ---------------------------------------------------------------------------
class EpisodicMemory:
    """
    Stores a rolling window of past observations and returns a novelty score
    for any new observation as the mean distance to its k nearest neighbours.

    Based on: Badia et al. "Never Give Up" (DeepMind, 2020).
    """

    def __init__(self, capacity: int = 500, k: int = 10, epsilon: float = 1e-3):
        self.capacity = capacity
        self.k = k
        self.epsilon = epsilon
        self.buffer: Deque[torch.Tensor] = deque(maxlen=capacity)

    def novelty(self, obs: torch.Tensor) -> float:
        """
        Returns novelty in [0, 1).
        High  = obs is far from everything seen so far.
        Low   = obs resembles many past states.
        """
        obs = obs.detach().float()
        if len(self.buffer) < self.k:
            self.buffer.append(obs)
            return 1.0

        stack = torch.stack(list(self.buffer))          # (N, D)
        dists = torch.norm(stack - obs.unsqueeze(0), dim=-1)  # (N,)
        k_actual = min(self.k, len(dists))
        knn_dist = dists.topk(k_actual, largest=False).values.mean()
        score = float(knn_dist / (knn_dist + self.epsilon))

        self.buffer.append(obs)
        return score

    def reset(self) -> None:
        self.buffer.clear()


# ---------------------------------------------------------------------------
# KnowledgeGraph – persistent causal fact store
# ---------------------------------------------------------------------------
class CausalFact:
    """One learned causal association: (action, mat_a, mat_b) -> outcome."""

    __slots__ = ('key', 'outcome_ids', 'confidence', 'tries', 'successes')

    def __init__(self, key: tuple):
        self.key: tuple = key              # (action, mat_a, mat_b | None)
        self.outcome_ids: list[str] = []   # material IDs produced
        self.confidence: float = 0.0       # in [-1, 1]
        self.tries: int = 0
        self.successes: int = 0

    def update(self, outcomes: list[str], success: bool) -> None:
        self.tries += 1
        if success:
            self.successes += 1
            self.confidence = min(1.0, self.confidence + 0.12)
            for o in outcomes:
                if o not in self.outcome_ids:
                    self.outcome_ids.append(o)
        else:
            self.confidence = max(-1.0, self.confidence - 0.05)

    @property
    def success_rate(self) -> float:
        return self.successes / self.tries if self.tries > 0 else 0.0

    def __repr__(self) -> str:
        return (f"CausalFact({self.key}, conf={self.confidence:.2f}, "
                f"tries={self.tries}, outcomes={self.outcome_ids})")


class KnowledgeGraph:
    """
    Stores causal facts an agent has learned through experimentation.

    Usage
    -----
    kg = KnowledgeGraph()
    kg.record(('rub', 'dry_wood', 'flint'), ['ember'], success=True)
    kg.record(('rub', 'dry_wood', 'flint'), [], success=False)

    best = kg.best_untested_hypothesis(known_materials)
    # -> tuple like ('bind', 'fiber', 'sharp_stone') or None

    Inheritance / Imitation
    -----------------------
    child_kg.inherit_from(parent_kg)  – copies high-confidence facts with noise
    child_kg.imitate_from(other_kg)   – blends in another agent's knowledge
    """

    ACTIONS = ('rub', 'strike', 'bind', 'bundle', 'place_on_heat', 'blow', 'eat')

    def __init__(self):
        self.facts: dict[tuple, CausalFact] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def record(self, key: tuple, outcomes: list[str], success: bool) -> None:
        """Record the result of one interaction attempt."""
        if key not in self.facts:
            self.facts[key] = CausalFact(key)
        self.facts[key].update(outcomes, success)

    def get(self, key: tuple) -> CausalFact | None:
        return self.facts.get(key)

    def confident_facts(self, threshold: float = 0.3) -> list[CausalFact]:
        """Return facts with confidence above threshold."""
        return [f for f in self.facts.values() if f.confidence >= threshold]

    def best_untested_hypothesis(
        self,
        available_materials: list[str],
        min_confidence_to_skip: float = 0.4,
    ) -> tuple | None:
        """
        Suggest the most promising (action, mat_a, mat_b) to try next.
        Prefers combinations that are:
          1. Completely unknown (never tried)
          2. Low-confidence but promising (tried < 3 times)
        Returns None if nothing interesting can be suggested.
        """
        candidates: list[tuple[float, tuple]] = []

        for mat_a in available_materials:
            for action in self.ACTIONS:
                for mat_b in [None] + available_materials:  # type: ignore[list-item]
                    if mat_b == mat_a:
                        continue
                    key = (action, mat_a, mat_b)
                    fact = self.facts.get(key)
                    if fact is None:
                        # Unknown – high priority
                        candidates.append((1.0, key))
                    elif fact.tries < 3 and fact.confidence < min_confidence_to_skip:
                        # Under-explored
                        candidates.append((0.5 - fact.confidence * 0.1, key))

        if not candidates:
            return None

        # Sort descending by priority, pick best
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    def known_material_uses(self, mat_id: str) -> list[CausalFact]:
        """All facts involving a given material."""
        return [
            f for f in self.facts.values()
            if mat_id in (f.key[1], f.key[2])
        ]

    # ------------------------------------------------------------------
    # Cultural transmission
    # ------------------------------------------------------------------
    def inherit_from(
        self,
        parent: 'KnowledgeGraph',
        strength: float = 0.7,
        confidence_threshold: float = 0.25,
    ) -> None:
        """
        Copy high-confidence facts from parent with slight noise.
        Biological analogue: innate knowledge / prepared learning.
        """
        for key, fact in parent.facts.items():
            if fact.confidence < confidence_threshold:
                continue
            if key not in self.facts:
                self.facts[key] = CausalFact(key)
            child_fact = self.facts[key]
            # Blend confidence toward parent's with small noise
            noise = (torch.randn(1).item() * 0.05)
            child_fact.confidence = (
                strength * fact.confidence
                + (1.0 - strength) * child_fact.confidence
                + noise
            )
            child_fact.confidence = max(-1.0, min(1.0, child_fact.confidence))
            # Copy outcome knowledge
            for oid in fact.outcome_ids:
                if oid not in child_fact.outcome_ids:
                    child_fact.outcome_ids.append(oid)

    def imitate_from(
        self,
        other: 'KnowledgeGraph',
        strength: float = 0.15,
    ) -> None:
        """
        Blend in another agent's knowledge during social observation.
        Weaker than inheritance – cultural learning is noisier.
        """
        for key, fact in other.facts.items():
            if fact.confidence < 0.2:
                continue
            if key not in self.facts:
                self.facts[key] = CausalFact(key)
            my_fact = self.facts[key]
            my_fact.confidence = (
                (1.0 - strength) * my_fact.confidence
                + strength * fact.confidence
            )
            my_fact.confidence = max(-1.0, min(1.0, my_fact.confidence))
            for oid in fact.outcome_ids:
                if oid not in my_fact.outcome_ids:
                    my_fact.outcome_ids.append(oid)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def summary(self) -> list[dict]:
        return [
            {
                'key': f.key,
                'confidence': round(f.confidence, 3),
                'tries': f.tries,
                'success_rate': round(f.success_rate, 3),
                'outcomes': f.outcome_ids,
            }
            for f in sorted(self.facts.values(), key=lambda x: -x.confidence)
        ]

    def __len__(self) -> int:
        return len(self.facts)

    def __repr__(self) -> str:
        return f"KnowledgeGraph({len(self.facts)} facts)"
