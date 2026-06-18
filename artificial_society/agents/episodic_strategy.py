"""
EpisodicStrategyMemory – Episodisches Gedächtnis für Handlungsstrategien
------------------------------------------------------------------------
Speichert erfolgreiche Handlungssequenzen als wiederverwendbare Episoden.

Nicht nur: 'Wo war Nahrung?'
Sondern:   'Welche Sequenz hat zum Erfolg geführt?'

Format:
  Episode = {
      'goal':    str,          # z.B. 'EAT'
      'actions': list[str],    # ['forage', 'forage', 'cooperate']
      'outcome': str,          # 'success' | 'partial' | 'failure'
      'reward':  float,
      'tick':    int,
      'context': dict,         # energy, health, etc. bei Start
  }

Biologisches Vorbild: Hippocampus-basiertes episodisches Gedächtnis.
Menschen erinnern sich an konkrete Ereignisse, nicht nur an abstrakte Fakten.
"""
from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


@dataclass
class Episode:
    goal: str
    actions: List[str]
    outcome: str   # 'success' | 'partial' | 'failure'
    reward: float
    tick: int
    context: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.outcome == 'success'


class EpisodicStrategyMemory:
    """
    Fixed-capacity rolling buffer of episodes.
    Provides retrieval by goal similarity and outcome.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.buffer: Deque[Episode] = deque(maxlen=capacity)
        # Running stats per goal
        self._goal_success: dict = {}
        self._goal_count: dict = {}

    def record(self, goal: str, actions: list, reward: float, tick: int,
               context: dict = None) -> None:
        outcome = 'success' if reward > 0.5 else ('partial' if reward > 0.1 else 'failure')
        ep = Episode(
            goal=goal,
            actions=list(actions[-4:]),  # last 4 actions
            outcome=outcome,
            reward=reward,
            tick=tick,
            context=context or {},
        )
        self.buffer.append(ep)
        self._goal_count[goal] = self._goal_count.get(goal, 0) + 1
        if ep.is_success:
            self._goal_success[goal] = self._goal_success.get(goal, 0) + 1

    def best_actions_for_goal(self, goal: str) -> Optional[List[str]]:
        """
        Retrieve the action sequence from the most successful episode
        with the given goal. Returns None if no successful episode found.
        """
        successes = [
            ep for ep in self.buffer
            if ep.goal == goal and ep.is_success
        ]
        if not successes:
            return None
        best = max(successes, key=lambda e: e.reward)
        return list(best.actions)

    def goal_success_rate(self, goal: str) -> float:
        count = self._goal_count.get(goal, 0)
        if count == 0:
            return 0.0
        return self._goal_success.get(goal, 0) / count

    def recent_avg_reward(self, n: int = 10) -> float:
        recent = list(self.buffer)[-n:]
        if not recent:
            return 0.0
        return sum(e.reward for e in recent) / len(recent)

    def inherit_from(self, parent: 'EpisodicStrategyMemory',
                     strength: float = 0.4, n: int = 16) -> None:
        """
        Child inherits some successful episodes from parent.
        Biological analogue: learned behavioral templates passed down.
        """
        successes = [ep for ep in parent.buffer if ep.is_success]
        # Sort by reward, take top n
        successes.sort(key=lambda e: e.reward, reverse=True)
        for ep in successes[:n]:
            if random.random() < strength:
                # Add with slight noise on reward
                self.record(
                    goal=ep.goal,
                    actions=ep.actions,
                    reward=ep.reward * (0.8 + random.random() * 0.4),
                    tick=0,
                    context={},
                )

    def __len__(self) -> int:
        return len(self.buffer)
