"""
KnowledgeGraph + NoveltyMemory
--------------------------------
NoveltyMemory   – NGU-style k-NN novelty score (state-space distance,
                   not activation magnitude). Formerly named EpisodicMemory;
                   a pickle-compat alias keeps old checkpoints loadable.
KnowledgeGraph  – Persistent causal fact store.  Agents record which
                   (action, mat_a, mat_b) combinations succeeded or
                   failed, accumulate confidence, and share knowledge
                   via inheritance and imitation.

Emergenz-Erweiterungen (v3):
  - CompositeAction: Erfolgreiche Aktionssequenzen werden als neue Makro-
    Aktionen gespeichert. Agenten können damit neue Handlungsklassen
    entdecken, die nicht explizit programmiert wurden.
  - Prerequisites in CausalFact: Multi-Step-Technologiebäume. Eine Erfindung
    kann Voraussetzungen haben – Agenten ohne Feuer-Wissen "denken" nicht
    an Töpfern.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch


# ---------------------------------------------------------------------------
# NoveltyMemory – novelty via k-NN distance in observation space
# ---------------------------------------------------------------------------
class NoveltyMemory:
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
        # --- Tier-3 perf: cache the stacked buffer; rebuild only when it changes.
        # `_version` bumps on every mutation; `stacked()` rebuilds lazily on a miss,
        # so the per-rollout re-stack (and, on GPU, the host->device upload) happens
        # at most once per buffer change instead of on every read.
        self._version: int = 0
        self._cache: torch.Tensor | None = None
        self._cache_version: int = -1

    def _remember(self, obs: torch.Tensor) -> None:
        """Append to the rolling buffer and invalidate the stack cache."""
        self.buffer.append(obs)
        self._version += 1

    def stacked(self, device: torch.device | None = None) -> torch.Tensor | None:
        """Buffer stacked into one ``(N, D)`` tensor, cached across reads.

        Identical to ``torch.stack(list(self.buffer))`` (optionally moved to
        ``device``); rebuilt only when the buffer mutates. Returns ``None`` for
        an empty buffer. When the cache already lives on ``device`` the ``.to``
        is a no-op, which is what removes the repeated GPU re-upload.
        """
        if not self.buffer:
            return None
        if self._cache is None or self._cache_version != self._version:
            self._cache = torch.stack(list(self.buffer))
            self._cache_version = self._version
        stack = self._cache
        if device is not None:
            stack = stack.to(device)
        return stack

    def novelty(self, obs: torch.Tensor) -> float:
        """
        Returns novelty in [0, 1).
        High  = obs is far from everything seen so far.
        Low   = obs resembles many past states.
        """
        obs = obs.detach().float()
        if len(self.buffer) < self.k:
            self._remember(obs)
            return 1.0

        stack = self.stacked(obs.device)                # (N, D)
        dists = torch.norm(stack - obs.unsqueeze(0), dim=-1)  # (N,)
        k_actual = min(self.k, len(dists))
        knn_dist = dists.topk(k_actual, largest=False).values.mean()
        score = float(knn_dist / (knn_dist + self.epsilon))

        self._remember(obs)
        return score

    def reset(self) -> None:
        self.buffer.clear()
        self._version += 1

    # ------------------------------------------------------------------
    # Pickle support (checkpoints serialize whole Agent/Brain graphs)
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict:
        # The stacked cache is a derived tensor (and may live on GPU); never
        # persist it -- it bloats checkpoints and breaks cross-device loads.
        state = self.__dict__.copy()
        state["_cache"] = None
        state["_cache_version"] = -1
        return state

    def __setstate__(self, state: dict) -> None:
        # pickle restores objects WITHOUT calling __init__, so back-fill the
        # cache attributes. Checkpoints written before the cache existed have
        # none of them; restore safe defaults so _remember()/stacked() never
        # AttributeError on the first call after load.
        self.__dict__.update(state)
        if not hasattr(self, "_version"):
            self._version = 0
        self._cache = None
        self._cache_version = -1


# ---------------------------------------------------------------------------
# CompositeAction – dynamisch entdeckte Makro-Aktionen (NEU)
# ---------------------------------------------------------------------------
@dataclass
class CompositeAction:
    """
    Eine neu entdeckte Handlung = gespeicherte Sequenz primitiver Aktionen.

    Biologisches Vorbild: Menschen kombinieren bekannte Handlungen zu neuen
    Werkzeugnutzungen (Bogen = bind + rub + bundle in Sequenz). Diese
    Makro-Aktionen entstehen nicht durch explizite Programmierung, sondern
    durch wiederholte erfolgreiche Verkettung.
    """
    action_id: str                            # z.B. "macro_0007"
    steps: list[str]                          # ['rub', 'blow', 'bundle']
    context_materials: list[str] = field(default_factory=list)  # Materialkontext
    confidence: float = 0.0
    uses: int = 0
    total_reward: float = 0.0

    def update(self, reward: float) -> None:
        self.uses += 1
        self.total_reward += reward
        self.confidence = min(1.0, self.confidence + 0.08 * max(0.0, reward))

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.uses if self.uses > 0 else 0.0

    def __repr__(self) -> str:
        return (f"CompositeAction({self.action_id}, steps={self.steps}, "
                f"conf={self.confidence:.2f}, uses={self.uses})")


# ---------------------------------------------------------------------------
# KnowledgeGraph – persistent causal fact store
# ---------------------------------------------------------------------------
class CausalFact:
    """One learned causal association: (action, mat_a, mat_b) -> outcome."""

    __slots__ = ('key', 'outcome_ids', 'confidence', 'tries', 'successes', 'prerequisites')

    def __init__(self, key: tuple):
        self.key: tuple = key              # (action, mat_a, mat_b | None)
        self.outcome_ids: list[str] = []   # material IDs produced
        self.confidence: float = 0.0       # in [-1, 1]
        self.tries: int = 0
        self.successes: int = 0
        # NEU: Voraussetzungen (andere CausalFact-Keys die bekannt sein müssen)
        # Biologisch: Agenten ohne Feuer-Wissen "denken" nicht an Töpfern.
        # ACHTUNG: wird derzeit nirgends aus Erster-Hand-Erfahrung befüllt (nur
        # via inherit_from kopiert) — eine Tech-Tree-Quelle zu designen ist ein
        # getrackter Follow-up; bis dahin ist prerequisites_met() immer True.
        self.prerequisites: list[tuple] = []

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

    def prerequisites_met(self, known_facts: dict) -> bool:
        """True wenn alle Voraussetzungen im eigenen KnowledgeGraph bekannt sind."""
        for prereq_key in self.prerequisites:
            fact = known_facts.get(prereq_key)
            if fact is None or fact.confidence < 0.2:
                return False
        return True

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

    Composite Actions (NEU)
    -----------------------
    kg.record_macro(['rub', 'blow', 'bundle'], reward=1.5, materials=['dry_wood', 'fiber'])
    # Speichert eine neue Makro-Aktion falls sie noch unbekannt ist.

    best_macro = kg.best_macro_action(min_confidence=0.3)
    # -> CompositeAction or None

    Inheritance / Imitation
    -----------------------
    child_kg.inherit_from(parent_kg)  – copies high-confidence facts with noise
    child_kg.imitate_from(other_kg)   – blends in another agent's knowledge
    """

    ACTIONS = ('rub', 'strike', 'bind', 'bundle', 'place_on_heat', 'blow', 'eat')

    def __init__(self):
        self.facts: dict[tuple, CausalFact] = {}
        # NEU: Dynamisch entdeckte Makro-Aktionen
        self.macro_actions: dict[tuple, CompositeAction] = {}

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
        Beruecksichtigt jetzt Prerequisites: Kombis deren Voraussetzungen
        nicht bekannt sind, werden nicht vorgeschlagen.
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
                        candidates.append((1.0, key))
                    elif fact.tries < 3 and fact.confidence < min_confidence_to_skip:
                        # Prüfe ob Voraussetzungen erfüllt sind (NEU)
                        if fact.prerequisites_met(self.facts):
                            candidates.append((0.5 - fact.confidence * 0.1, key))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    def known_material_uses(self, mat_id: str) -> list[CausalFact]:
        """All facts involving a given material."""
        return [
            f for f in self.facts.values()
            if mat_id in (f.key[1], f.key[2])
        ]

    # ------------------------------------------------------------------
    # CompositeAction API (NEU)
    # ------------------------------------------------------------------
    def record_macro(
        self,
        steps: list[str],
        reward: float,
        materials: list[str] | None = None,
    ) -> CompositeAction:
        """
        Speichert eine erfolgreiche Aktionssequenz als Makro-Aktion.
        Falls sie bereits bekannt ist, wird nur ihr Reward-Tracking aktualisiert.

        Emergenz-Mechanismus: Makro-Aktionen entstehen bottom-up aus
        beobachteten Erfolgen – nicht top-down durch Rezept-Vorgaben.
        """
        key = tuple(steps)
        if key not in self.macro_actions:
            action_id = f"macro_{len(self.macro_actions):04d}"
            self.macro_actions[key] = CompositeAction(
                action_id=action_id,
                steps=list(steps),
                context_materials=materials or [],
            )
        self.macro_actions[key].update(reward)
        return self.macro_actions[key]

    def best_macro_action(self, min_confidence: float = 0.25) -> CompositeAction | None:
        """
        Gibt die best bewertete Makro-Aktion zurueck, die zuverlaessig genug ist.
        Wird vom Brain genutzt um neue zusammengesetzte Aktionen auszuführen.
        """
        candidates = [
            m for m in self.macro_actions.values()
            if m.confidence >= min_confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.avg_reward * m.confidence)

    def all_macro_actions(self, min_confidence: float = 0.0) -> list[CompositeAction]:
        """Alle bekannten Makro-Aktionen, optional gefiltert nach Confidence."""
        return [
            m for m in self.macro_actions.values()
            if m.confidence >= min_confidence
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
        Überträgt jetzt auch Makro-Aktionen (NEU).
        """
        for key, fact in parent.facts.items():
            if fact.confidence < confidence_threshold:
                continue
            if key not in self.facts:
                self.facts[key] = CausalFact(key)
            child_fact = self.facts[key]
            noise = (torch.randn(1).item() * 0.05)
            child_fact.confidence = (
                strength * fact.confidence
                + (1.0 - strength) * child_fact.confidence
                + noise
            )
            child_fact.confidence = max(-1.0, min(1.0, child_fact.confidence))
            for oid in fact.outcome_ids:
                if oid not in child_fact.outcome_ids:
                    child_fact.outcome_ids.append(oid)
            # Prerequisites mitübertragen (NEU)
            child_fact.prerequisites = list(fact.prerequisites)

        # Makro-Aktionen vererben (NEU): Nur sehr sichere werden übertragen
        for key, macro in parent.macro_actions.items():
            if macro.confidence >= confidence_threshold + 0.1:
                if key not in self.macro_actions:
                    self.macro_actions[key] = CompositeAction(
                        action_id=f"macro_{len(self.macro_actions):04d}",
                        steps=list(macro.steps),
                        context_materials=list(macro.context_materials),
                        confidence=macro.confidence * strength * 0.8,
                        uses=0,
                    )

    def imitate_from(
        self,
        other: 'KnowledgeGraph',
        strength: float = 0.15,
    ) -> None:
        """
        Blend in another agent's knowledge during social observation.
        Weaker than inheritance – cultural learning is noisier.
        Überträgt jetzt auch Makro-Aktionen (NEU).
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

        # Makro-Aktionen imitieren (NEU)
        for key, macro in other.macro_actions.items():
            if macro.confidence >= 0.3 and key not in self.macro_actions:
                self.macro_actions[key] = CompositeAction(
                    action_id=f"macro_{len(self.macro_actions):04d}",
                    steps=list(macro.steps),
                    context_materials=list(macro.context_materials),
                    confidence=macro.confidence * strength,
                    uses=0,
                )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def summary(self) -> list[dict]:
        facts_summary = [
            {
                'key': f.key,
                'confidence': round(f.confidence, 3),
                'tries': f.tries,
                'success_rate': round(f.success_rate, 3),
                'outcomes': f.outcome_ids,
                'prerequisites': f.prerequisites,
            }
            for f in sorted(self.facts.values(), key=lambda x: -x.confidence)
        ]
        macros_summary = [
            {
                'action_id': m.action_id,
                'steps': m.steps,
                'confidence': round(m.confidence, 3),
                'uses': m.uses,
                'avg_reward': round(m.avg_reward, 3),
            }
            for m in sorted(self.macro_actions.values(), key=lambda x: -x.confidence)
        ]
        return facts_summary + macros_summary

    def __len__(self) -> int:
        return len(self.facts)

    def __repr__(self) -> str:
        return f"KnowledgeGraph({len(self.facts)} facts, {len(self.macro_actions)} macros)"


# Pickle-compat alias: checkpoints written before the rename reference
# artificial_society.agents.knowledge.EpisodicMemory; unpickling resolves the
# class via module attribute lookup, so this alias keeps them loadable.
# (Checkpoints written by NEW code serialize as NoveltyMemory and are not
# loadable by pre-rename code.)
EpisodicMemory = NoveltyMemory
