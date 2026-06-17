"""
Goal Stack: Vorausplanung ueber mehrere Ticks
----------------------------------------------
Agenten reagieren nicht nur reaktiv auf den aktuellen Tick.
Sie koennen Zwischenziele stapeln und ueber mehrere Ticks
verfolgen. Das ist die Grundlage fuer:

  - Keramik herstellen (form -> trocknen -> brennen -> befuellen)
  - Feld anlegen     (raeumen -> saeen -> bewaessern -> ernten)
  - Huette bauen     (material sammeln -> form -> arch -> versiegeln)
  - Handel vorbereiten (inventar aufbauen -> zu Agent gehen -> tauschen)

Die Ziele werden NICHT von aussen vorgegeben.
Sie entstehen emergent wenn der Brain lernt, dass bestimmte
Aktionssequenzen mehr Reward bringen als Einzelaktionen.

Architektur:
  GoalStack    -- LIFO-Stack von SubGoals pro Agent
  SubGoal      -- Ein atomares Zwischenziel mit Abbruchbedingung
  GoalPlanner  -- Leichtgewichtiger Heuristik-Planer (kein MCTS)
                  Ergaenzt den PPO-Brain um kurzfristige Planung
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# SubGoal
# ---------------------------------------------------------------------------
@dataclass
class SubGoal:
    """
    Ein atomares Zwischenziel.
    action:      primitive Action die ausgefuehrt werden soll
    target_mat:  Material das benoetigt wird (oder None)
    target_x/y:  Zielposition (oder None = egal)
    reward_pred: Erwarteter Reward wenn Goal erreicht
    done_fn:     Callable(agent, cell) -> bool -- Abbruchbedingung
    max_ticks:   Nach wieviel Ticks das Goal aufgegeben wird
    ticks_spent: Wie viele Ticks bereits verwendet
    """
    action:      str
    target_mat:  Optional[str]  = None
    target_x:    Optional[int]  = None
    target_y:    Optional[int]  = None
    reward_pred: float          = 0.0
    done_fn:     Optional[Callable] = None
    max_ticks:   int            = 20
    ticks_spent: int            = 0
    label:       str            = ''   # human-readable fuer Logging

    def is_done(self, agent, cell: dict) -> bool:
        if self.done_fn is not None:
            return self.done_fn(agent, cell)
        # Default: Material im Inventar?
        if self.target_mat:
            inv = getattr(agent, 'material_inventory', {})
            return inv.get(self.target_mat, 0.0) > 0.1
        return False

    def is_expired(self) -> bool:
        return self.ticks_spent >= self.max_ticks


# ---------------------------------------------------------------------------
# GoalStack
# ---------------------------------------------------------------------------
class GoalStack:
    """
    LIFO-Stack von SubGoals fuer einen Agenten.
    Der Agent verfolgt immer das oberste (aktuellste) Ziel.
    """
    def __init__(self, max_depth: int = 6):
        self.stack:     list[SubGoal] = []
        self.max_depth: int           = max_depth
        self.completed: list[dict]    = []  # Log abgeschlossener Goals
        self.failed:    list[dict]    = []  # Log fehlgeschlagener Goals

    def push(self, goal: SubGoal):
        if len(self.stack) < self.max_depth:
            self.stack.append(goal)

    def peek(self) -> Optional[SubGoal]:
        return self.stack[-1] if self.stack else None

    def pop(self) -> Optional[SubGoal]:
        return self.stack.pop() if self.stack else None

    def tick(self, agent, cell: dict) -> tuple[Optional[str], float]:
        """
        Prueft ob das aktuelle Ziel erreicht/abgelaufen ist.
        Gibt (action, reward_shaping) zurueck.
        """
        if not self.stack:
            return None, 0.0

        goal = self.stack[-1]
        goal.ticks_spent += 1

        # Ziel erreicht?
        if goal.is_done(agent, cell):
            self.stack.pop()
            self.completed.append({
                'label':  goal.label,
                'ticks':  goal.ticks_spent,
                'reward': goal.reward_pred,
            })
            return goal.action, goal.reward_pred  # Bonus-Reward

        # Abgelaufen?
        if goal.is_expired():
            self.stack.pop()
            self.failed.append({
                'label': goal.label,
                'ticks': goal.ticks_spent,
            })
            return None, -0.05  # kleiner Penalty

        # Noch aktiv: naechste Action ist die des aktuellen Goals
        return goal.action, 0.0

    def depth(self) -> int:
        return len(self.stack)

    def is_empty(self) -> bool:
        return len(self.stack) == 0


# ---------------------------------------------------------------------------
# GoalPlanner: erstellt Goal-Sequenzen aus bekannten Heuristiken
# ---------------------------------------------------------------------------
class GoalPlanner:
    """
    Leichtgewichtiger Planer. Kein MCTS, kein A*.
    Erkennt Opportunitaeten und schreibt SubGoal-Sequenzen in den Stack.

    Regeln sind NICHT hardcodiert als Rezepte.
    Sie sind Heuristiken der Form:
      'wenn Bedingung X, dann versuche Aktion Y um Ziel Z zu erreichen'
    Der Brain entscheidet ob er dem Planer folgt oder nicht.
    """

    def suggest_goals(
        self,
        agent,
        cell: dict,
        world_context: dict,
    ) -> list[SubGoal]:
        """
        Gibt Liste von empfohlenen SubGoals zurueck.
        Agent kann diese in seinen GoalStack pushen.
        """
        suggestions = []
        inv    = getattr(agent, 'material_inventory', {})
        energy = getattr(agent, 'energy', 100) / 240.0
        cold   = world_context.get('temperature', 20) < 8
        dark   = world_context.get('light', 1.0) < 0.3

        # --- Hunger: Nahrung beschaffen ---
        if energy < 0.4:
            if inv.get('raw_meat', 0) > 0.1 or inv.get('raw_root', 0) > 0.1:
                # Feuer suchen und kochen
                mat = 'raw_meat' if inv.get('raw_meat',0) > inv.get('raw_root',0) else 'raw_root'
                suggestions.append(SubGoal(
                    action     = 'place_on_heat',
                    target_mat = 'cooked_meat' if mat == 'raw_meat' else 'cooked_root',
                    reward_pred = 0.8,
                    max_ticks   = 15,
                    label       = f'cook_{mat}',
                    done_fn     = lambda a, c: (
                        a.material_inventory.get('cooked_meat', 0) > 0.1
                        or a.material_inventory.get('cooked_root', 0) > 0.1
                    ),
                ))
            else:
                # Nahrung sammeln
                suggestions.append(SubGoal(
                    action      = 'eat',
                    target_mat  = 'raw_meat',
                    reward_pred = 0.5,
                    max_ticks   = 30,
                    label       = 'forage_food',
                ))

        # --- Kalt: Feuer machen ---
        if cold:
            if inv.get('dry_grass', 0) > 0.2 or inv.get('dry_wood', 0) > 0.2:
                suggestions.append(SubGoal(
                    action      = 'rub',
                    target_mat  = 'ember',
                    reward_pred = 1.2,
                    max_ticks   = 25,
                    label       = 'make_fire',
                    done_fn     = lambda a, c: c.get('materials', {}).get('fire', 0) > 0.3,
                ))

        # --- Ton vorhanden: Keramik-Sequenz ---
        clay_in_cell = cell.get('materials', {}).get('clay', 0.0)
        if clay_in_cell > 0.3 and not any(g.label == 'make_pottery' for g in agent.goal_stack.stack
                                          if hasattr(agent, 'goal_stack')):
            suggestions.append(SubGoal(
                action      = 'form',
                target_mat  = None,
                reward_pred = 0.6,
                max_ticks   = 10,
                label       = 'make_pottery',
                done_fn     = lambda a, c: any(
                    getattr(o, 'shape', '') == 'hollow'
                    for o in c.get('objects', [])
                ),
            ))

        # --- Samen vorhanden + guter Boden: Pflanzen ---
        plantable = [m for m, q in inv.items()
                     if q > 0.1 and m in ('seed_grain','seed_herb','seed_fiber','root_cut')]
        good_soil = world_context.get('soil', 'rock') in ('loam', 'clay')
        season    = world_context.get('season', 'summer')
        if plantable and good_soil and season in ('spring', 'summer'):
            suggestions.append(SubGoal(
                action      = 'plant',
                target_mat  = plantable[0],
                reward_pred = 1.0,
                max_ticks   = 5,
                label       = f'plant_{plantable[0]}',
            ))

        # --- Werkzeug fehlt: binden ---
        has_tool = getattr(agent, 'tool', None) is not None
        has_fiber = inv.get('fiber', 0) > 0.1
        has_flint = inv.get('flint', 0) > 0.1 or inv.get('sharp_stone', 0) > 0.1
        if not has_tool and has_fiber and has_flint:
            suggestions.append(SubGoal(
                action      = 'bind',
                reward_pred = 0.9,
                max_ticks   = 8,
                label       = 'craft_tool',
                done_fn     = lambda a, c: (
                    getattr(a, 'tool', None) is not None
                ),
            ))

        return suggestions


GOAL_PLANNER = GoalPlanner()


# ---------------------------------------------------------------------------
# Integration: Agent-Tick mit Goal Stack
# ---------------------------------------------------------------------------
def agent_tick_with_goals(
    agent,
    cell: dict,
    world_context: dict,
    tick: int,
) -> tuple[str, float]:
    """
    Kombiniert GoalStack + GoalPlanner.
    Gibt (action, reward_shaping) zurueck fuer diesen Tick.

    Der PPO-Brain bekommt reward_shaping als zusaetzliches Signal.
    Er lernt selbst ob Planung besser ist als reaktives Handeln.
    """
    # GoalStack initialisieren wenn noetig
    if not hasattr(agent, 'goal_stack'):
        agent.goal_stack = GoalStack()

    # Stack-Tick: pruefen ob aktuelles Ziel erledigt/abgelaufen
    action_from_stack, shaping = agent.goal_stack.tick(agent, cell)

    # Wenn Stack leer: Planer konsultieren (nicht immer -- 30% Wahrscheinlichkeit)
    import random
    if agent.goal_stack.is_empty() and random.random() < 0.30:
        suggestions = GOAL_PLANNER.suggest_goals(agent, cell, world_context)
        for goal in suggestions[:2]:  # max 2 neue Ziele auf einmal
            agent.goal_stack.push(goal)
        # Neue Action vom frisch gefuellten Stack
        if not agent.goal_stack.is_empty():
            action_from_stack, shaping = agent.goal_stack.tick(agent, cell)

    return action_from_stack, shaping
