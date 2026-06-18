"""
Theory of Mind (ToM)
---------------------
Agents maintain a model of what other agents know, believe, and intend.
This enables:
  - Teaching  : share knowledge only when the other lacks it
  - Deception : act as if you don't know something to gain advantage
  - Inference : predict another's next action from their inferred state
  - Role recognition : learn that agent X is the fire-maker / hunter

Architecture
------------
Each agent owns a TheoryOfMind instance which holds one AgentModel
per known agent. An AgentModel is a lightweight belief structure updated
through observation (position, action_mode, inventory signals, messages).

Crucially: the agent never receives ground-truth about another's internal
state. It must INFER from observable behaviour -- exactly as humans do.

Integration points in agent.py
-------------------------------
  1. _ensure_new_fields()  -> agent.tom = TheoryOfMind(agent.id)
  2. update_social()       -> agent.tom.observe_agent(other, tick)
  3. spawn_child()         -> child.tom.inherit_from(parent.tom)
  4. KnowledgeGraph share  -> gated by tom.should_teach(other_id)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports; Agent type used as string annotation only


# ---------------------------------------------------------------------------
# Belief about another agent's knowledge / capabilities
# ---------------------------------------------------------------------------
@dataclass
class AgentModel:
    """
    What agent A believes about agent B.

    Fields
    ------
    agent_id        : int   -- ID of the modelled agent (B)
    inferred_energy : float -- estimated energy level 0..1
    inferred_knowledge: set -- material IDs / causal keys B is believed to know
    role            : str   -- inferred social role ('unknown','hunter','maker','elder','scout')
    last_seen_tick  : int
    last_seen_pos   : tuple
    trust_estimate  : float -- our estimate of how much B trusts us (-1..1)
    deception_count : int   -- how many times B has apparently misled us
    observation_count: int  -- total observations of B
    intent_vector   : list  -- smoothed action-mode distribution (6 floats)
    """
    agent_id: int
    inferred_energy: float = 0.5
    inferred_knowledge: set = field(default_factory=set)
    role: str = 'unknown'
    last_seen_tick: int = 0
    last_seen_pos: tuple = (0, 0)
    trust_estimate: float = 0.0
    deception_count: int = 0
    observation_count: int = 0
    intent_vector: list = field(default_factory=lambda: [1/6]*6)

    # Action mode index map (must match last_action_mode strings used in agent.py)
    _MODE_IDX: dict = field(default_factory=lambda: {
        'idle': 0, 'move': 0, 'gather': 1, 'forage_herb': 1,
        'sleep': 2, 'invent': 3, 'craft': 3, 'experiment': 3,
        'share': 4, 'signal': 4, 'mate': 4,
        'attack': 5,
    })

    def update_intent(self, action_mode: str, alpha: float = 0.15) -> None:
        """Exponential moving average over observed action modes."""
        idx = self._MODE_IDX.get(action_mode, 0)
        for i in range(len(self.intent_vector)):
            self.intent_vector[i] *= (1.0 - alpha)
        self.intent_vector[idx] += alpha
        # Normalise
        total = sum(self.intent_vector) or 1.0
        self.intent_vector = [v / total for v in self.intent_vector]

    def infer_role(self) -> str:
        """Infer role from observed action distribution."""
        iv = self.intent_vector
        if iv[5] > 0.30:           return 'warrior'
        if iv[3] > 0.25:           return 'maker'
        if iv[1] > 0.30:           return 'hunter'
        if iv[4] > 0.25:           return 'elder'
        if iv[2] > 0.25:           return 'sleeper'
        return 'scout'

    def predicted_next_action(self) -> str:
        """Return most likely next action mode."""
        modes = ['idle', 'gather', 'sleep', 'invent', 'share', 'attack']
        return modes[self.intent_vector.index(max(self.intent_vector))]


# ---------------------------------------------------------------------------
# TheoryOfMind -- one per agent
# ---------------------------------------------------------------------------
class TheoryOfMind:
    """
    Maintains AgentModels for all observed agents.

    Key capabilities
    ----------------
    observe_agent()         -- update model from current observation
    should_teach()          -- decide whether to share knowledge with B
    should_deceive()        -- decide whether to suppress a signal toward B
    infer_knowledge_gap()   -- estimate what B doesn't know that we do
    inherit_from()          -- cultural transmission at birth
    """

    def __init__(self, owner_id: int):
        self.owner_id = owner_id
        self.models: dict[int, AgentModel] = {}
        # Cache of our own known causal keys (filled externally)
        self._own_knowledge_keys: set = set()

    # ------------------------------------------------------------------
    # Core observation update
    # ------------------------------------------------------------------
    def observe_agent(
        self,
        other,          # Agent instance
        tick: int,
        own_trust: float = 0.0,
    ) -> AgentModel:
        """
        Update belief model for `other` based on observable signals.

        Observable signals (no ground truth):
          - other.pos
          - other.last_action_mode
          - other.message_vector  (public signal)
          - other.tribe_id
          - other.tool (visible if adjacent)
          - other.life_stage()    (age group roughly visible)
          - other.sick > 50       (visibly ill)
        """
        oid = other.id
        if oid not in self.models:
            self.models[oid] = AgentModel(agent_id=oid)

        m = self.models[oid]
        m.observation_count += 1
        m.last_seen_tick = tick
        m.last_seen_pos  = other.pos

        # Infer energy from action mode (sick/slow = low energy)
        if other.last_action_mode == 'sleep':
            m.inferred_energy = max(0.0, m.inferred_energy - 0.05)
        elif other.last_action_mode in ('gather', 'hunt'):
            m.inferred_energy = min(1.0, m.inferred_energy + 0.03)
        elif other.last_action_mode == 'idle':
            m.inferred_energy *= 0.97  # slight decay -- assume idle = depleted

        # Infer knowledge from observed behaviour
        if other.last_action_mode in ('invent', 'craft', 'experiment'):
            # Agent is doing research -- probably knows more than us
            if other.tool:
                m.inferred_knowledge.add(f'tool:{other.tool}')
        if other.last_action_mode == 'forage_herb':
            m.inferred_knowledge.add('herb_use')
        if other.tool:
            m.inferred_knowledge.add(f'tool:{other.tool}')

        # Material inventory signals (visible items)
        inv = getattr(other, 'material_inventory', {})
        for mat in inv:
            if inv.get(mat, 0) > 0.1:
                m.inferred_knowledge.add(f'has:{mat}')

        # Trust estimate: adjust from message content
        msg = getattr(other, 'message_vector', [0.0]*4)
        if len(msg) >= 2 and msg[0] > 0.3:
            m.trust_estimate = min(1.0, m.trust_estimate + 0.02)
        elif len(msg) >= 2 and msg[0] < -0.3:
            m.trust_estimate = max(-1.0, m.trust_estimate - 0.03)

        # Update intent distribution
        m.update_intent(other.last_action_mode)

        # Infer role
        m.role = m.infer_role()

        return m

    # ------------------------------------------------------------------
    # Teaching decision
    # ------------------------------------------------------------------
    def should_teach(
        self,
        other_id: int,
        knowledge_key: str,
        own_trust_of_other: float = 0.0,
        tribe_match: bool = False,
    ) -> bool:
        """
        Return True if we should share `knowledge_key` with other_id.

        Teaching is favoured when:
          - Other probably doesn't know it yet (knowledge gap)
          - We trust them (same tribe or high trust)
          - We are not under survival pressure (checked externally via cortisol)
          - Knowledge is useful (not just noise)

        This is a probabilistic decision, not a deterministic rule.
        """
        m = self.models.get(other_id)
        if m is None:
            return False

        # Does other likely lack this knowledge?
        other_probably_knows = knowledge_key in m.inferred_knowledge
        if other_probably_knows:
            return False

        # Base probability from trust
        p = 0.15 + 0.40 * max(0.0, own_trust_of_other)
        if tribe_match:
            p += 0.25

        # Role bonus: elders teach more
        if m.role == 'elder':
            p += 0.10

        # Deception penalty: if B has deceived us before, share less
        p -= 0.10 * min(1.0, m.deception_count * 0.2)

        return random.random() < min(0.95, p)

    # ------------------------------------------------------------------
    # Deception decision
    # ------------------------------------------------------------------
    def should_deceive(
        self,
        other_id: int,
        own_aggression: float = 0.0,
        competition_pressure: float = 0.0,
    ) -> bool:
        """
        Decide whether to suppress / falsify signal toward other_id.

        Deception is favoured when:
          - Low trust of other
          - High resource competition
          - High aggression gene
          - Other is not tribe member (checked externally)

        Biological analogue: signalling false resource locations to
        outcompete rivals (observed in some primates and corvids).
        """
        m = self.models.get(other_id)
        if m is None:
            return False

        p = 0.02 + 0.15 * own_aggression + 0.20 * competition_pressure
        p -= 0.30 * max(0.0, m.trust_estimate)  # trust suppresses deception
        p  = max(0.0, min(0.60, p))
        return random.random() < p

    # ------------------------------------------------------------------
    # Knowledge gap inference
    # ------------------------------------------------------------------
    def infer_knowledge_gap(
        self,
        other_id: int,
        own_kg_keys: set,
    ) -> set:
        """
        Return set of causal/material keys we know that other probably lacks.
        Used to select what to teach.
        """
        m = self.models.get(other_id)
        if m is None:
            return set()
        known_by_other = m.inferred_knowledge
        return {k for k in own_kg_keys if k not in known_by_other}

    # ------------------------------------------------------------------
    # Cultural transmission at birth
    # ------------------------------------------------------------------
    def inherit_from(
        self,
        parent_tom: 'TheoryOfMind',
        strength: float = 0.4,
    ) -> None:
        """
        Child inherits a blurred copy of parent's agent models.
        Child has never met these agents but has a prior disposition
        (prepared trust/distrust) based on parent's experience.

        Biological analogue: attachment theory, in-group familiarity bias.
        """
        for oid, parent_model in parent_tom.models.items():
            if parent_model.observation_count < 3:
                continue  # don't inherit shallow observations
            child_model = AgentModel(agent_id=oid)
            # Inherit trust estimate with noise
            noise = random.gauss(0, 0.10)
            child_model.trust_estimate = max(-1.0, min(1.0,
                strength * parent_model.trust_estimate + noise
            ))
            # Inherit role belief
            child_model.role = parent_model.role
            # Inherit partial knowledge inference (with forgetting)
            for k in parent_model.inferred_knowledge:
                if random.random() < strength:
                    child_model.inferred_knowledge.add(k)
            child_model.intent_vector = [
                strength * v + (1 - strength) * (1/6)
                for v in parent_model.intent_vector
            ]
            self.models[oid] = child_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_model(self, other_id: int) -> AgentModel | None:
        return self.models.get(other_id)

    def known_roles(self) -> dict[int, str]:
        return {oid: m.role for oid, m in self.models.items()}

    def most_trusted_id(self) -> int | None:
        if not self.models:
            return None
        return max(self.models, key=lambda k: self.models[k].trust_estimate)

    def most_skilled_maker(self) -> int | None:
        makers = {oid: m for oid, m in self.models.items() if m.role == 'maker'}
        if not makers:
            return None
        return max(makers, key=lambda k: makers[k].observation_count)

    def update_own_knowledge(
        self,
        kg_keys: set,
    ) -> None:
        """Sync internal cache of own KnowledgeGraph keys."""
        self._own_knowledge_keys = set(kg_keys)

    def summary(self) -> list[dict]:
        return [
            {
                'id':           m.agent_id,
                'role':         m.role,
                'trust_est':    round(m.trust_estimate, 2),
                'observations': m.observation_count,
                'inferred_knowledge_count': len(m.inferred_knowledge),
                'predicted_next': m.predicted_next_action(),
            }
            for m in sorted(self.models.values(), key=lambda x: -x.observation_count)
        ]

    def __repr__(self) -> str:
        return f"TheoryOfMind(owner={self.owner_id}, models={len(self.models)})"
