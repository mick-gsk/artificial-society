"""
Goal Stack: Vorausplanung ueber mehrere Ticks
----------------------------------------------
Agenten reagieren nicht nur reaktiv auf den aktuellen Tick.
Sie koennen Zwischenziele stapeln und ueber mehrere Ticks
verfolgen.

Die Ziele werden NICHT von aussen vorgegeben.
Sie entstehen emergent wenn der Brain lernt, dass bestimmte
Aktionssequenzen mehr Reward bringen als Einzelaktionen.

Architektur:
  GoalStack    -- LIFO-Stack von SubGoals pro Agent
  SubGoal      -- Ein atomares Zwischenziel mit Abbruchbedingung
  GoalPlanner  -- Emergenter Planer basierend auf Need-Vektoren
                  (keine hardcodierten Rezepte)

NEU: GoalPlanner nutzt compute_need_vector aus need_driven_invention.
Ziele werden rein aus physikalischen Eigenschaften und aktuellen Beduerfnissen
abgeleitet, nicht aus fest programmierten if-cold-then-fire Regeln.
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
    label:       str            = ''

    def is_done(self, agent, cell: dict) -> bool:
        if self.done_fn is not None:
            return self.done_fn(agent, cell)
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
    def __init__(self, max_depth: int = 6):
        self.stack:     list[SubGoal] = []
        self.max_depth: int           = max_depth
        self.completed: list[dict]    = []
        self.failed:    list[dict]    = []

    def push(self, goal: SubGoal):
        if len(self.stack) < self.max_depth:
            self.stack.append(goal)

    def peek(self) -> Optional[SubGoal]:
        return self.stack[-1] if self.stack else None

    def pop(self) -> Optional[SubGoal]:
        return self.stack.pop() if self.stack else None

    def tick(self, agent, cell: dict) -> tuple[Optional[str], float]:
        if not self.stack:
            return None, 0.0
        goal = self.stack[-1]
        goal.ticks_spent += 1
        if goal.is_done(agent, cell):
            self.stack.pop()
            self.completed.append({
                'label':  goal.label,
                'ticks':  goal.ticks_spent,
                'reward': goal.reward_pred,
            })
            return goal.action, goal.reward_pred
        if goal.is_expired():
            self.stack.pop()
            self.failed.append({
                'label': goal.label,
                'ticks': goal.ticks_spent,
            })
            return None, -0.05
        return goal.action, 0.0

    def depth(self) -> int:
        return len(self.stack)

    def is_empty(self) -> bool:
        return len(self.stack) == 0


# ---------------------------------------------------------------------------
# GoalPlanner: vollstaendig emergent, keine hardcodierten Rezepte
# ---------------------------------------------------------------------------
class GoalPlanner:
    """
    Emergenter Planer basierend auf Need-Vektoren.

    KEIN hardcodiertes if-cold-then-fire.
    KEIN hardcodiertes if-hungry-then-cook.

    Stattdessen:
      1. Berechne Need-Vektor des Agenten
      2. Finde Materialien im Inventar/Zelle die den Need erfuellen koennen
      3. Schlage SubGoal vor das diese Materialien kombiniert
      4. Der Agent entscheidet selbst ob er dem Vorschlag folgt

    Der Planer hat kein Wissen ueber Rezepte. Er weiss nur:
    - Was der Agent braucht (Need-Vektor)
    - Was verfuegbar ist (Inventar + Zelle)
    - Welche Actions es gibt (PRIMITIVE_ACTIONS)
    Der Rest wird durch Trial-and-Error und CausalMemory gelernt.
    """

    def suggest_goals(
        self,
        agent,
        cell: dict,
        world_context: dict,
    ) -> list[SubGoal]:
        from artificial_society.systems.need_driven_invention import (
            compute_need_vector, _select_materials_by_need,
            _select_action_by_need, NEED_THRESHOLD, PROP_DIMS,
        )
        from artificial_society.environment.materials import IDX, N_PROPS, get_vector
        import numpy as np

        suggestions = []
        need = compute_need_vector(agent, cell)
        import numpy as _np
        need_magnitude = float(_np.linalg.norm(_np.maximum(need, 0.0)))

        # Nur Ziele vorschlagen wenn echter Need besteht
        if need_magnitude < NEED_THRESHOLD:
            return []

        # Top-Need identifizieren (welche physikalische Eigenschaft wird gebraucht?)
        positive_need = np.maximum(need, 0.0)
        top_prop_idx  = int(np.argmax(positive_need))
        top_prop      = PROP_DIMS[top_prop_idx]
        top_need_val  = float(positive_need[top_prop_idx])

        if top_need_val < 0.2:
            return []

        # Materialien auswaehlen die den Need am besten erfuellen
        mat_a, mat_b = _select_materials_by_need(agent, cell, need)
        if mat_a is None:
            return []

        vec_a = get_vector(mat_a)
        vec_b = get_vector(mat_b) if mat_b else None
        causal_mem = getattr(agent, 'causal_memory', None)
        action = _select_action_by_need(need, vec_a, vec_b, causal_mem)

        # done_fn: Ziel erreicht wenn Need-Vektor in Inventar befriedigt wird
        # (d.h. irgendein Material mit der benoetigten Eigenschaft > Schwellenwert)
        needed_prop_idx = top_prop_idx
        needed_threshold = 0.3

        def need_fulfilled(a, c, prop_idx=needed_prop_idx, threshold=needed_threshold):
            inv = getattr(a, 'material_inventory', {})
            for mat_id, qty in inv.items():
                if qty > 0.1:
                    v = get_vector(mat_id)
                    if float(v[prop_idx]) > threshold:
                        return True
            # Auch Zelle pruefen (z.B. Feuer in Zelle loest Kaelte-Need)
            slot = c.get('materials', {})
            for mat_id, qty in slot.items():
                if qty > 0.1:
                    v = get_vector(mat_id)
                    if float(v[prop_idx]) > threshold:
                        return True
            return False

        reward_pred = top_need_val * 0.8  # proportional zum Need

        suggestions.append(SubGoal(
            action      = action,
            target_mat  = None,   # kein hardcodiertes Ziel-Material
            reward_pred = reward_pred,
            max_ticks   = 20,
            label       = f'need_{top_prop}',
            done_fn     = need_fulfilled,
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
    if not hasattr(agent, 'goal_stack'):
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
