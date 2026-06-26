"""
Goal Stack Extensions: Komplexe Sequenzen & Kulturelle Transmission
----------------------------------------------------------------------
Erweiterung von goal_stack.py um:

  GoalSequence  -- Mehrstufige Aktion als wiederholbare Vorlage.
                   Agenten koennen Sequenzen speichern und abrufen.
                   Basis fuer 'Wissen' das ueber Generationen
                   weitergegeben wird.

  SequenceLibrary -- Gemeinsame Bibliothek erfolgreicher Sequenzen.
                    Agenten koennen Sequenzen 'beobachten' und kopieren.
                    Kulturelle Transmission: gute Methoden verbreiten sich.

  RecipeDiscovery -- Wenn eine Sequenz wiederholt zu einem neuen Material
                    fuehrt, wird sie als 'Rezept' registriert.
                    Nicht hardcodiert -- nur wenn die Sequenz konsistent
                    dasselbe Ergebnis erzeugt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from artificial_society.environment.materials import DISCOVERY_REGISTRY
from artificial_society.systems.goal_stack import GOAL_PLANNER, GoalStack, SubGoal


# ---------------------------------------------------------------------------
# GoalSequence: eine Folge von SubGoals als Einheit
# ---------------------------------------------------------------------------
@dataclass
class GoalSequence:
    """
    Eine benannte Folge von SubGoals.
    Kann gespeichert, kopiert und weitergegeben werden.
    Das ist die Basis fuer kulturell transmittiertes Wissen.
    """

    seq_id: str
    label: str
    steps: list[SubGoal]
    creator_id: int
    times_used: int = 0
    total_reward: float = 0.0
    times_shared: int = 0

    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.times_used)

    def push_to_stack(self, stack: GoalStack):
        """Schreibt alle Steps (umgekehrt) in den Stack."""
        for step in reversed(self.steps):
            stack.push(step)


# ---------------------------------------------------------------------------
# Vordefinierte Sequenz-Templates (als Heuristiken, nicht Hardcode)
# ---------------------------------------------------------------------------
def _pottery_sequence(creator_id: int) -> GoalSequence:
    """Keramik herstellen: Ton form + trocknen + brennen."""
    return GoalSequence(
        seq_id="seq_pottery",
        label="pottery",
        creator_id=creator_id,
        steps=[
            SubGoal(
                action="collect",
                target_mat="clay",
                max_ticks=15,
                label="collect_clay",
                reward_pred=0.1,
            ),
            SubGoal(
                action="form",
                target_mat=None,
                max_ticks=10,
                label="form_clay",
                reward_pred=0.3,
                done_fn=lambda a, c: any(
                    getattr(o, "shape", "") == "hollow" for o in c.get("objects", [])
                ),
            ),
            SubGoal(
                action="place_on_heat",
                target_mat=None,
                max_ticks=20,
                label="fire_pottery",
                reward_pred=0.5,
                done_fn=lambda a, c: any(
                    getattr(o, "durability", 0) > 0.7 for o in c.get("objects", [])
                ),
            ),
        ],
    )


def _farming_sequence(seed_mat: str, creator_id: int) -> GoalSequence:
    """Feld anlegen: Boden lockern + saeen + bewaessern."""
    return GoalSequence(
        seq_id=f"seq_farm_{seed_mat}",
        label=f"farm_{seed_mat}",
        creator_id=creator_id,
        steps=[
            SubGoal(
                action="collect",
                target_mat=seed_mat,
                max_ticks=20,
                label=f"get_{seed_mat}",
                reward_pred=0.2,
            ),
            SubGoal(
                action="plant", target_mat=seed_mat, max_ticks=5, label="plant", reward_pred=0.4
            ),
            SubGoal(
                action="wait",
                target_mat=seed_mat,
                max_ticks=60,
                label="wait_growth",
                reward_pred=0.0,
                done_fn=lambda a, c: c.get("materials", {}).get(seed_mat, 0) > 0.5,
            ),
            SubGoal(
                action="collect",
                target_mat=seed_mat,
                max_ticks=10,
                label="harvest",
                reward_pred=1.2,
            ),
        ],
    )


def _shelter_sequence(creator_id: int) -> GoalSequence:
    """Huette bauen: Holz/Stein sammeln + arch-Action."""
    return GoalSequence(
        seq_id="seq_shelter",
        label="build_shelter",
        creator_id=creator_id,
        steps=[
            SubGoal(
                action="collect",
                target_mat="dry_wood",
                max_ticks=20,
                label="collect_wood",
                reward_pred=0.1,
            ),
            SubGoal(
                action="collect",
                target_mat="stone",
                max_ticks=20,
                label="collect_stone",
                reward_pred=0.1,
            ),
            SubGoal(
                action="arch",
                target_mat=None,
                max_ticks=15,
                label="build_dome",
                reward_pred=1.5,
                done_fn=lambda a, c: any(
                    getattr(o, "shelter", 0) > 0.3 for o in c.get("objects", [])
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# SequenceLibrary: geteilte Wissensbasis
# ---------------------------------------------------------------------------
class SequenceLibrary:
    """
    Gemeinsame Bibliothek aller bekannten Sequenzen.
    Agenten koennen Sequenzen hinzufuegen und abrufen.
    Wenn eine Sequenz haeufig genutzt und erfolgreich ist,
    verbreitet sie sich in der Population.
    """

    def __init__(self):
        self.sequences: dict[str, GoalSequence] = {}
        self._add_defaults()

    def _add_defaults(self):
        """Einige Basis-Sequenzen als Startpunkt."""
        for seq in [
            _pottery_sequence(creator_id=-1),
            _farming_sequence("seed_grain", creator_id=-1),
            _shelter_sequence(creator_id=-1),
        ]:
            self.sequences[seq.seq_id] = seq

    def register(self, seq: GoalSequence):
        """Neues Wissen hinzufuegen."""
        if seq.seq_id not in self.sequences:
            self.sequences[seq.seq_id] = seq
            print(f"[SEQUENCE] New sequence learned: {seq.label} by agent_{seq.creator_id}")
        else:
            # Reward-Statistik aktualisieren
            existing = self.sequences[seq.seq_id]
            existing.times_used += seq.times_used
            existing.total_reward += seq.total_reward

    def best_sequence_for_context(
        self,
        agent_state: dict,
        world_context: dict,
        top_k: int = 3,
    ) -> list[GoalSequence]:
        """
        Gibt die K besten Sequenzen fuer den aktuellen Kontext zurueck.
        Sortiert nach avg_reward * recency.
        """
        scored = [
            (seq, seq.avg_reward())
            for seq in self.sequences.values()
            if seq.times_used > 0 or seq.creator_id == -1  # defaults immer verfuegbar
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in scored[:top_k]]

    def observe_and_learn(
        self,
        observer_agent,
        model_agent,
        trust: float,
        tick: int,
    ):
        """
        Kulturelle Transmission: Observer kopiert Sequenzen von Model
        wenn Trust hoch genug und Modells Sequenzen erfolgreicher sind.
        """
        if trust < 0.4:
            return
        model_lib = getattr(model_agent, "known_sequences", set())
        for seq_id in model_lib:
            if seq_id in self.sequences and seq_id not in getattr(
                observer_agent, "known_sequences", set()
            ):
                seq = self.sequences[seq_id]
                if seq.avg_reward() > 0.3:  # Nur gute Sequenzen uebertragen
                    if not hasattr(observer_agent, "known_sequences"):
                        observer_agent.known_sequences = set()
                    observer_agent.known_sequences.add(seq_id)
                    seq.times_shared += 1
                    print(
                        f"[CULTURE] tick={tick} agent_{observer_agent.id} "
                        f"learned {seq.label} from agent_{model_agent.id}"
                    )


# ---------------------------------------------------------------------------
# RecipeDiscovery: wenn Sequenz konsistent neues Material erzeugt
# ---------------------------------------------------------------------------
class RecipeDiscovery:
    """
    Tracked Sequenz-Outcomes.
    Wenn dieselbe Sequenz > 3x dasselbe neue Material erzeugt:
    Sequenz wird als 'Rezept' in der DiscoveryRegistry verknuepft.
    """

    def __init__(self):
        # seq_id -> [(result_mat_id, reward), ...]
        self.outcomes: dict[str, list[tuple]] = {}

    def record_outcome(
        self,
        seq_id: str,
        result_mat_id: str,
        reward: float,
    ):
        if seq_id not in self.outcomes:
            self.outcomes[seq_id] = []
        self.outcomes[seq_id].append((result_mat_id, reward))

        # Konsistenz-Check
        outcomes = self.outcomes[seq_id]
        if len(outcomes) >= 3:
            mat_ids = [o[0] for o in outcomes[-5:]]
            most_common = max(set(mat_ids), key=mat_ids.count)
            count = mat_ids.count(most_common)
            if count >= 3:
                # Konsistentes Rezept gefunden
                avg_r = float(np.mean([o[1] for o in outcomes if o[0] == most_common]))
                print(
                    f"[RECIPE] Sequence {seq_id} consistently produces "
                    f"{most_common} (reward={avg_r:.2f})"
                )
                # Im DiscoveryRegistry als 'recipe' markieren
                for entry in DISCOVERY_REGISTRY.entries:
                    if entry["id"] == most_common:
                        entry["recipe"] = (seq_id, most_common, None)
                        break


SEQUENCE_LIBRARY = SequenceLibrary()
RECIPE_DISCOVERY = RecipeDiscovery()


def agent_tick_with_goals(
    agent,
    cell: dict,
    world_context: dict,
    tick: int,
) -> tuple[str | None, float]:
    if not hasattr(agent, "goal_stack") or agent.goal_stack is None:
        agent.goal_stack = GoalStack()

    action_from_stack, shaping = agent.goal_stack.tick(agent, cell)

    import random

    if agent.goal_stack.is_empty() and random.random() < 0.30:
        suggestions = GOAL_PLANNER.suggest_goals(agent, cell, world_context)
        for goal in suggestions[:2]:
            agent.goal_stack.push(goal)

        if not agent.goal_stack.is_empty():
            action_from_stack, shaping = agent.goal_stack.tick(agent, cell)

    return action_from_stack, shaping
