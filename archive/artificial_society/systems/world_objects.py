"""
WorldObjectRegistry – Persistente Objekte
-------------------------------------------
Objekte die Agenten erschaffen, überleben deren Tod.
Das ist die Grundlage für kumulative Technologie und Zivilisationsbildung.

Objekttypen:
  campfire    – Wärmequelle, Sozialtreffpunkt, Wissenstransfer-Hub
  storage     – Ressourcenlager (Grundlage für Besitz, Handel, Diebstahl)
  marker      – Territoriumsmarkierung, Warnsignal
  workshop    – Erfindungsbonus für Agenten in der Nähe
  knowledge_stone – Gespeichertes Wissen (Rezepte, Heilmittel)

Alle Objekte:
  - altern (durability sinkt pro Tick)
  - können genutzt werden (apply_effect)
  - können verbessert werden (upgrade)
  - produzieren Effekte auf umliegende Agenten
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


OBJECT_TYPES = ('campfire', 'storage', 'marker', 'workshop', 'knowledge_stone')

BUILD_COSTS = {
    'campfire':       {'wood': 2, 'stone': 0, 'fiber': 0},
    'storage':        {'wood': 2, 'stone': 1, 'fiber': 1},
    'marker':         {'wood': 0, 'stone': 1, 'fiber': 0},
    'workshop':       {'wood': 3, 'stone': 2, 'fiber': 1},
    'knowledge_stone':{'wood': 0, 'stone': 2, 'fiber': 0},
}

BASE_DURABILITY = {
    'campfire':        200,
    'storage':         800,
    'marker':          400,
    'workshop':        600,
    'knowledge_stone': 1200,
}

DECAY_RATE = {
    'campfire':        1.5,   # burns out fast
    'storage':         0.2,
    'marker':          0.3,
    'workshop':        0.2,
    'knowledge_stone': 0.1,   # very persistent
}


@dataclass
class WorldObject:
    obj_id: int
    obj_type: str
    pos: Tuple[int, int]
    creator_id: int
    tribe_id: Optional[int]
    created_tick: int
    durability: float = 0.0
    level: int = 1
    stored_resources: Dict[str, float] = field(default_factory=dict)
    stored_knowledge: List[tuple] = field(default_factory=list)  # CausalFact keys
    stored_recipes: Dict[str, list] = field(default_factory=dict)  # disease -> herbs
    uses: int = 0
    last_used_tick: int = 0

    def __post_init__(self):
        if self.durability == 0.0:
            self.durability = BASE_DURABILITY.get(self.obj_type, 500) * (1.0 + 0.1 * (self.level - 1))

    @property
    def alive(self) -> bool:
        return self.durability > 0

    def tick(self):
        """Age the object each simulation tick."""
        decay = DECAY_RATE.get(self.obj_type, 0.3)
        self.durability = max(0.0, self.durability - decay)

    def upgrade(self, resources: dict) -> bool:
        """Spend resources to improve the object."""
        costs = BUILD_COSTS.get(self.obj_type, {})
        for res, amount in costs.items():
            if resources.get(res, 0) < amount:
                return False
        for res, amount in costs.items():
            resources[res] = resources.get(res, 0) - amount
        self.level += 1
        self.durability += BASE_DURABILITY.get(self.obj_type, 500) * 0.5
        return True

    def apply_effect(self, agent, tick: int) -> float:
        """
        Apply the object's effect to a nearby agent.
        Returns reward bonus.
        """
        if not self.alive:
            return 0.0
        self.uses += 1
        self.last_used_tick = tick
        reward = 0.0

        if self.obj_type == 'campfire':
            # Warmth + social bonus
            agent.health = min(100.0, agent.health + 0.5 * self.level)
            agent.energy = min(240.0, agent.energy + 0.3 * self.level)
            # Slow decay when used
            self.durability = min(
                BASE_DURABILITY['campfire'] * self.level,
                self.durability + 0.2
            )
            reward = 0.05

        elif self.obj_type == 'storage':
            # Agent can deposit or withdraw resources
            # Deposit: if energy > 180, add surplus food token
            if agent.energy > 180.0 and self.stored_resources.get('food', 0) < 50.0 * self.level:
                deposit = min(10.0, agent.energy - 160.0)
                agent.energy -= deposit
                self.stored_resources['food'] = self.stored_resources.get('food', 0) + deposit
            # Withdraw: if energy < 80
            elif agent.energy < 80.0 and self.stored_resources.get('food', 0) > 0:
                take = min(20.0, self.stored_resources['food'])
                self.stored_resources['food'] -= take
                agent.energy = min(240.0, agent.energy + take)
                reward = 0.3

        elif self.obj_type == 'workshop':
            # Invention bonus — applied externally in agent._try_invent
            reward = 0.0  # handled elsewhere

        elif self.obj_type == 'knowledge_stone':
            # Transfer stored knowledge to agent
            if self.stored_knowledge:
                kg = getattr(agent, 'knowledge', None)
                if kg is not None:
                    for key in self.stored_knowledge[:3]:  # read 3 facts per visit
                        if key not in kg.facts:
                            from artificial_society.agents.knowledge import CausalFact
                            kg.facts[key] = CausalFact(key)
                            kg.facts[key].confidence = 0.3
                            reward += 0.1
            if self.stored_recipes:
                for disease, herbs in self.stored_recipes.items():
                    if disease not in getattr(agent, 'remedy_knowledge', {}):
                        if not hasattr(agent, 'remedy_knowledge'):
                            agent.remedy_knowledge = {}
                        agent.remedy_knowledge[disease] = list(herbs)
                        reward += 0.2

        return reward

    def store_knowledge(self, kg, remedy_knowledge: dict = None):
        """Engrave knowledge into this stone/workshop."""
        if self.obj_type not in ('knowledge_stone', 'workshop'):
            return
        if kg is not None:
            for key, fact in kg.facts.items():
                if fact.confidence > 0.5 and key not in self.stored_knowledge:
                    self.stored_knowledge.append(key)
                    if len(self.stored_knowledge) > 50 * self.level:
                        break
        if remedy_knowledge:
            for disease, herbs in remedy_knowledge.items():
                if disease not in self.stored_recipes:
                    self.stored_recipes[disease] = list(herbs)


class WorldObjectRegistry:
    """
    Global registry of all persistent world objects.
    Survives agent death — this is the civilisation layer.
    """

    def __init__(self):
        self._objects: Dict[int, WorldObject] = {}
        self._counter: int = 0
        # Spatial index: pos -> list of obj_ids
        self._spatial: Dict[Tuple[int, int], List[int]] = {}

    def create(self, obj_type: str, pos: Tuple[int, int], creator_id: int,
               tribe_id: Optional[int], tick: int) -> Optional[WorldObject]:
        if obj_type not in OBJECT_TYPES:
            return None
        self._counter += 1
        obj = WorldObject(
            obj_id=self._counter,
            obj_type=obj_type,
            pos=pos,
            creator_id=creator_id,
            tribe_id=tribe_id,
            created_tick=tick,
        )
        self._objects[self._counter] = obj
        self._spatial.setdefault(pos, []).append(self._counter)
        return obj

    def get_at(self, pos: Tuple[int, int], obj_type: str = None) -> List[WorldObject]:
        ids = self._spatial.get(pos, [])
        objs = [self._objects[i] for i in ids if i in self._objects and self._objects[i].alive]
        if obj_type:
            objs = [o for o in objs if o.obj_type == obj_type]
        return objs

    def get_nearby(self, pos: Tuple[int, int], radius: int = 2,
                   obj_type: str = None) -> List[WorldObject]:
        px, py = pos
        result = []
        for (ox, oy), ids in self._spatial.items():
            if abs(ox - px) <= radius and abs(oy - py) <= radius:
                for oid in ids:
                    obj = self._objects.get(oid)
                    if obj and obj.alive:
                        if obj_type is None or obj.obj_type == obj_type:
                            result.append(obj)
        return result

    def tick_all(self):
        """Age all objects. Remove dead ones from spatial index."""
        dead = []
        for obj in self._objects.values():
            obj.tick()
            if not obj.alive:
                dead.append(obj.obj_id)
        for oid in dead:
            obj = self._objects.pop(oid)
            ids = self._spatial.get(obj.pos, [])
            if oid in ids:
                ids.remove(oid)

    def can_build(self, obj_type: str, resources: dict) -> bool:
        costs = BUILD_COSTS.get(obj_type, {})
        return all(resources.get(r, 0) >= a for r, a in costs.items())

    def spend_resources(self, obj_type: str, resources: dict) -> bool:
        if not self.can_build(obj_type, resources):
            return False
        for r, a in BUILD_COSTS[obj_type].items():
            resources[r] = resources.get(r, 0) - a
        return True

    @property
    def count(self) -> int:
        return sum(1 for o in self._objects.values() if o.alive)

    def summary(self) -> dict:
        counts = {t: 0 for t in OBJECT_TYPES}
        for o in self._objects.values():
            if o.alive:
                counts[o.obj_type] += 1
        return counts
