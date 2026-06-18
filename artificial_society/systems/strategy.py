"""
StrategySystem – Emergente Verhaltensstrategien
------------------------------------------------
Agenten wählen nicht direkt Aktionen, sondern übergeordnete Strategien.
Strategien werden bewertet und verstärkt (Reinforcement auf Strategie-Ebene).

Strategien:
  Explorer   – Erkundet die Welt, entdeckt neue Ressourcen
  Builder    – Baut persistente Objekte, investiert in Infrastruktur
  Hoarder    – Sammelt und lagert Ressourcen
  Trader     – Spezialisiert auf Austausch und Kooperation
  Inventor   – Fokussiert auf Erfindungen und Technologie
  Caretaker  – Kümmert sich um Kinder und schwache Agenten

Biologisches Vorbild: Individuelle Nischen in sozialen Gruppen.
Nicht jedes Mitglied macht dasselbe — Arbeitsteilung entsteht emergent.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, Optional


STRATEGIES = ('Explorer', 'Builder', 'Hoarder', 'Trader', 'Inventor', 'Caretaker')

STRATEGY_GOAL_BIAS: Dict[str, Dict[str, float]] = {
    'Explorer':  {'EXPLORE': 2.0, 'EAT': 0.8, 'REPRODUCE': 0.8},
    'Builder':   {'BUILD': 3.0,   'EXPLORE': 0.6, 'EAT': 1.0},
    'Hoarder':   {'EAT': 1.5,    'BUILD': 1.5, 'EXPLORE': 0.5},
    'Trader':    {'COOPERATE': 2.5, 'EAT': 1.0, 'EXPLORE': 1.0},
    'Inventor':  {'RESEARCH': 3.0, 'EXPLORE': 1.2, 'EAT': 0.9},
    'Caretaker': {'COOPERATE': 2.0, 'REPRODUCE': 1.5, 'EAT': 1.0},
}

# Welche Rewards stärken welche Strategie
STRATEGY_REWARD_SOURCE: Dict[str, list] = {
    'Explorer':  ['explore', 'forage'],
    'Builder':   ['build', 'structure'],
    'Hoarder':   ['forage', 'storage'],
    'Trader':    ['cooperate', 'trade'],
    'Inventor':  ['research', 'invention'],
    'Caretaker': ['cooperate', 'birth', 'kin'],
}


@dataclass
class StrategyRecord:
    name: str
    score: float = 0.5
    uses: int = 0
    total_reward: float = 0.0
    last_used_tick: int = 0

    def update(self, reward: float):
        self.uses += 1
        self.total_reward += reward
        # Exponential moving average
        self.score = 0.92 * self.score + 0.08 * max(0.0, reward)

    def decay(self):
        """Slight decay toward neutral — prevents lock-in."""
        self.score = 0.998 * self.score + 0.001 * 0.5

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.uses if self.uses > 0 else 0.0


class StrategySystem:
    """
    Tracks which strategies an agent has tried and how well they worked.
    The agent consults this to bias its goal selection.
    """

    def __init__(self):
        self.records: Dict[str, StrategyRecord] = {
            s: StrategyRecord(name=s) for s in STRATEGIES
        }
        self.current: str = 'Explorer'
        self._switch_cooldown: int = 0
        self._ticks_on_current: int = 0

    def update(self, reward: float, mode: str, tick: int):
        """Update the current strategy's score based on what just happened."""
        # Determine which strategy benefits from this reward source
        for strat, sources in STRATEGY_REWARD_SOURCE.items():
            if mode in sources:
                self.records[strat].update(reward * 0.5)

        self.records[self.current].update(reward)
        self._ticks_on_current += 1

        for rec in self.records.values():
            rec.decay()

        self._switch_cooldown = max(0, self._switch_cooldown - 1)

    def maybe_switch(self, genes: dict, tick: int) -> str:
        """
        Decide whether to switch strategy.
        Explorers and Inventors switch more often (curiosity).
        Returns current strategy name.
        """
        if self._switch_cooldown > 0:
            return self.current
        # Switch if another strategy looks significantly better
        best_score = self.records[self.current].score
        best_name = self.current
        for name, rec in self.records.items():
            if rec.score > best_score * 1.15:  # 15% better threshold
                best_score = rec.score
                best_name = name

        if best_name != self.current:
            self.current = best_name
            # Cooldown depends on plasticity gene
            plasticity = genes.get('plasticity', 1.0)
            self._switch_cooldown = max(20, int(60 / plasticity))
            self._ticks_on_current = 0

        # Occasional random exploration of strategies (epsilon-greedy)
        curiosity = genes.get('curiosity', 0.5)
        if random.random() < 0.02 * curiosity:
            self.current = random.choice(STRATEGIES)
            self._switch_cooldown = 15

        return self.current

    def goal_weights(self) -> Dict[str, float]:
        """Return goal priority weights for current strategy."""
        return STRATEGY_GOAL_BIAS.get(self.current, {})

    def inherit_from(self, parent_strategy: 'StrategySystem', strength: float = 0.4):
        """Child inherits strategy preferences with noise."""
        for name, rec in parent_strategy.records.items():
            self.records[name].score = (
                strength * rec.score + (1.0 - strength) * 0.5
            )
        # Start with parent's best strategy
        best = max(parent_strategy.records.values(), key=lambda r: r.score)
        self.current = best.name

    def summary(self) -> dict:
        return {
            'current': self.current,
            'scores': {n: round(r.score, 3) for n, r in self.records.items()}
        }
