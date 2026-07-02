"""
Reproduction Metrics – Vollständige Tracking-Schicht
------------------------------------------------------
Verfolgt den gesamten Reproduktionsprozess:
  reproduction_attempts     – wie oft ein Agent Partner gesucht hat
  successful_pairings       – erfolgreiche Paarungen
  pregnancies_started       – gestartete Schwangerschaften
  births                    – abgeschlossene Geburten
  offspring_survival_rate   – Anteil Kinder die >200 Ticks überleben

Diese Metriken sind essentiell um Reproduktions-Engpässe zu debuggen.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class AgentReproStats:
    agent_id: int
    attempts: int = 0
    pairings: int = 0
    pregnancies: int = 0
    births: int = 0
    children_alive: int = 0   # tracked externally
    children_survived_200: int = 0


class ReproductionMetrics:
    """
    Singleton-style registry (one per Simulation).
    The simulation calls these methods at key lifecycle events.
    """

    def __init__(self):
        self.agents: Dict[int, AgentReproStats] = {}
        # Global counters
        self.total_attempts = 0
        self.total_pairings = 0
        self.total_pregnancies = 0
        self.total_births = 0
        self.total_survived_200 = 0
        # Track birth tick per child: child_id -> (birth_tick, parent_ids)
        self._births_log: Dict[int, Tuple[int, int, int]] = {}  # child_id -> (tick, mother_id, father_id)

    def _get(self, agent_id: int) -> AgentReproStats:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentReproStats(agent_id)
        return self.agents[agent_id]

    def record_attempt(self, agent_id: int):
        self._get(agent_id).attempts += 1
        self.total_attempts += 1

    def record_pairing(self, mother_id: int, father_id: int):
        self._get(mother_id).pairings += 1
        self._get(father_id).pairings += 1
        self.total_pairings += 1

    def record_pregnancy(self, mother_id: int):
        self._get(mother_id).pregnancies += 1
        self.total_pregnancies += 1

    def record_birth(self, mother_id: int, father_id: int, child_id: int, tick: int):
        self._get(mother_id).births += 1
        self.total_births += 1
        self._births_log[child_id] = (tick, mother_id, father_id)

    def check_survival_200(self, child_id: int, current_tick: int, alive: bool) -> bool:
        """
        Call this every tick for living children.
        Returns True if the 200-tick milestone was just reached.
        """
        if child_id not in self._births_log:
            return False
        birth_tick, mother_id, father_id = self._births_log[child_id]
        if current_tick - birth_tick == 200 and alive:
            self.total_survived_200 += 1
            self._get(mother_id).children_survived_200 += 1
            if father_id:
                self._get(father_id).children_survived_200 += 1
            return True
        return False

    @property
    def survival_rate(self) -> float:
        if self.total_births == 0:
            return 0.0
        return self.total_survived_200 / self.total_births

    def summary(self) -> dict:
        return {
            'total_attempts':    self.total_attempts,
            'total_pairings':    self.total_pairings,
            'total_pregnancies': self.total_pregnancies,
            'total_births':      self.total_births,
            'survived_200':      self.total_survived_200,
            'survival_rate':     round(self.survival_rate, 3),
        }

    def log_status(self, tick: int):
        s = self.summary()
        print(
            f"[REPRO_METRICS] tick={tick}",
            f"attempts={s['total_attempts']}",
            f"pairings={s['total_pairings']}",
            f"pregnancies={s['total_pregnancies']}",
            f"births={s['total_births']}",
            f"survived200={s['survived_200']}",
            f"rate={s['survival_rate']:.2%}",
        )
