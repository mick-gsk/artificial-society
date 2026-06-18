from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch

from artificial_society.agents import agent as agent_mod
from artificial_society.agents.agent import Agent
from artificial_society.agents.brain import Brain
from artificial_society.agents.life_stage import CHILD_MAX, ADULT_MAX, STAGE_CHILD, STAGE_ADULT, STAGE_ELDER, get_stage_stats
from artificial_society.environment.resources import maybe_build_structure
from artificial_society.environment.structures import BUILD_ENERGY_COST, apply_structure_effects
from artificial_society.environment.territory import territory_reward_for_agent, get_home_forage_bonus
from artificial_society.systems.goal_stack import GoalStack
from artificial_society.systems.goal_stack_ext import agent_tick_with_goals
from artificial_society.systems.invention import agent_try_cook, agent_try_invention
from artificial_society.systems.language import TOKEN_WORLD, TokenMemory, agent_mark
from artificial_society.systems.need_driven_invention import agent_invent_from_need, compute_need_vector
from artificial_society.systems.social_learning import social_learning_step
from artificial_society.systems.remedy import (
    evaluate_remedy,
    record_cure_discovery,
    share_remedy_knowledge,
)

_PATCHED = False


_RESOURCE_ALIASES = {
    'wood': ('dry_wood', 'wet_wood', 'wood'),
    'stone': ('stone', 'flint'),
    'fiber': ('fiber', 'dry_grass', 'crushed_herb', 'leaf'),
}


def _compact_material_inventory(agent: Agent, max_entries: int = 24) -> None:
    inv = getattr(agent, 'material_inventory', None)
    if not inv:
        return

    cleaned = {k: float(v) for k, v in inv.items() if float(v) > 0.01}
    if len(cleaned) <= max_entries:
        agent.material_inventory = cleaned
        return

    protected = {
        k: v for k, v in cleaned.items()
        if not k.startswith('mat_')
    }
    discovered = sorted(
        ((k, v) for k, v in cleaned.items() if k.startswith('mat_')),
        key=lambda item: (-item[1], item[0]),
    )

    remaining = max_entries - len(protected)
    kept = dict(protected)
    if remaining > 0:
        for mat_id, qty in discovered[:remaining]:
            kept[mat_id] = qty

    # If protected already exceeds the cap, keep the most important essentials.
    if len(kept) > max_entries:
        essentials = [
            ('sharp_stone', kept.get('sharp_stone', 0.0)),
            ('fire', kept.get('fire', 0.0)),
            ('ember', kept.get('ember', 0.0)),
            ('cooked_meat', kept.get('cooked_meat', 0.0)),
            ('raw_meat', kept.get('raw_meat', 0.0)),
            ('raw_root', kept.get('raw_root', 0.0)),
            ('stone', kept.get('stone', 0.0)),
            ('wood', kept.get('wood', 0.0)),
            ('fiber', kept.get('fiber', 0.0)),
        ]
        essentials = [(k, v) for k, v in essentials if k in kept and v > 0]
        kept = dict(essentials[:max_entries])

    agent.material_inventory = kept


def _ensure_runtime_fields(agent: Agent) -> None:
    if not hasattr(agent, 'goal_stack') or agent.goal_stack is None:
        agent.goal_stack = GoalStack()
    if not hasattr(agent, 'token_memory') or agent.token_memory is None:
        agent.token_memory = TokenMemory()
    if not hasattr(agent, '_next_planning_tick'):
        agent._next_planning_tick = 0
    if not hasattr(agent, '_planning_stride'):
        agent._planning_stride = 4
    if not hasattr(agent, '_last_goal_action'):
        agent._last_goal_action = None
    if not hasattr(agent, '_language_retry_tick'):
        agent._language_retry_tick = 0
    if not hasattr(agent, '_inventory_cap'):
        agent._inventory_cap = 24
    if not hasattr(agent, 'x'):
        type(agent).x = property(lambda self: self.pos[0])  # type: ignore[attr-defined]
    if not hasattr(agent, 'y'):
        type(agent).y = property(lambda self: self.pos[1])  # type: ignore[attr-defined]


# Keep the class-level properties in place for the language system.
if not hasattr(Agent, 'x'):
    Agent.x = property(lambda self: self.pos[0])  # type: ignore[attr-defined]
if not hasattr(Agent, 'y'):
    Agent.y = property(lambda self: self.pos[1])  # type: ignore[attr-defined]


def _patch_discovery_registry() -> None:
    from artificial_society.environment.materials import DiscoveryRegistry

    if getattr(DiscoveryRegistry, '_runtime_patch_applied', False):
        return

    original_register = DiscoveryRegistry.register

    def known_ids(self) -> list[str]:
        cached = getattr(self, '_known_ids_cache', None)
        if cached is None:
            cached = [e['id'] for e in self.entries]
            self._known_ids_cache = cached
        return list(cached)

    def register(self, vector, discoverer_id: int = -1, tick: int = 0, recipe=None) -> str:
        result = original_register(self, vector, discoverer_id=discoverer_id, tick=tick, recipe=recipe)
        self._known_ids_cache = None
        return result

    DiscoveryRegistry.known_ids = known_ids  # type: ignore[assignment]
    DiscoveryRegistry.register = register  # type: ignore[assignment]
    DiscoveryRegistry._runtime_patch_applied = True


def _collect_resources_from_materials(self: Agent, world) -> None:
    x, y = self.pos
    cell = world.get_cell(x, y)
    slot = cell.setdefault('materials', {})
    tool_bonus = 0.20 if self.tool == 'sharp_stone' else 0.0

    for resource, aliases in _RESOURCE_ALIASES.items():
        collected = 0.0
        target_take = 1.0 + tool_bonus
        for material_name in aliases:
            qty = float(slot.get(material_name, 0.0))
            if qty <= 0:
                continue
            take = min(qty, max(0.0, target_take - collected))
            if take <= 0:
                continue
            slot[material_name] = max(0.0, qty - take)
            if slot[material_name] <= 0.01:
                slot.pop(material_name, None)
            collected += take
            if collected >= target_take:
                break

        if collected > 0:
            self.resources[resource] = float(self.resources.get(resource, 0.0)) + collected


def _build_from_resources(self: Agent, world) -> None:
    x, y = self.pos
    cell = world.get_cell(x, y)
    result = maybe_build_structure(cell, self.resources)
    if not result:
        return

    if isinstance(BUILD_ENERGY_COST, dict):
        cost = float(BUILD_ENERGY_COST.get(result, 10.0))
    else:
        cost = float(BUILD_ENERGY_COST)
    self.energy = max(0.0, self.energy - cost)



def _maybe_mark_language(agent: Agent, cell: dict, tick: int, context: np.ndarray, reward: float) -> float:
    if reward < 0.9 or tick < agent._language_retry_tick:
        return reward
    if random.random() > 0.12:
        return reward

    token_id = agent_mark(agent, cell, context, tick)
    if token_id:
        agent._language_retry_tick = tick + 12
        return reward + 0.1
    return reward



def _maybe_collect_language_convergence(agent: Agent, agents: list[Agent], tick: int) -> None:
    if tick % 75 != 0 or getattr(agent, 'id', 0) != 1:
        return
    memories = [getattr(a, 'token_memory', None) for a in agents]
    memories = [m for m in memories if m is not None]
    if len(memories) >= 2:
        TOKEN_WORLD.check_convergence(memories, tick)



def patched_update(
    self: Agent,
    world,
    agents,
    tick: int,
    season_state=None,
    weather_state=None,
    tribes=None,
    economy=None,
    technology=None,
):
    if not self.alive:
        return None

    _ensure_runtime_fields(self)

    from artificial_society.agents import agent as agent_module

    # Ensure the life-stage and age constants match the separate life_stage module.
    agent_module.MIN_REPRODUCTION_AGE = CHILD_MAX
    agent_module.ELDER_AGE = ADULT_MAX
    agent_module.STAGE_CHILD = CHILD_MAX
    agent_module.STAGE_ELDER = ADULT_MAX
    agent_module.AGE_LIMIT = 5000
    agent_module.AGE_HEALTH_DECAY_START = ADULT_MAX
    agent_module.AGE_HEALTH_DECAY_HARD = 4500

    # Make invention less noisy and less expensive computationally.
    agent_module.INVENTION_BASE_PROB = 0.08
    agent_module.INVENTION_CURIOSITY_MULT = 0.05
    agent_module.NEED_INVENTION_INTERVAL = 12

    from artificial_society.agents import brain as brain_module
    brain_module.PLAN_CANDIDATES = 8
    brain_module.PLAN_HORIZON = 2
    brain_module.PLAN_HORIZON_RESEARCH = 6

    self.endocrine.update(self, world)
    mods = self.endocrine.modifiers()
    stage = get_stage_stats(self.age)

    self._age_tick()
    self._disease_tick(world)
    if not self.alive:
        return None

    self._sleep_tick(mods)

    move_cost = 0.5 * mods.get('move_cost_mult', 1.0) * stage['move_cost_mult']
    hydration_loss = 0.3 * mods.get('hydration_loss_mult', 1.0) * stage['hydration_loss_mult']
    self.energy = max(0.0, self.energy - move_cost)
    self.hydration = max(0.0, self.hydration - hydration_loss)
    self.health = max(0.0, self.health - mods.get('health_drain', 0.0))

    if self.energy <= 0:
        self.health = max(0.0, self.health - 1.5)
    if self.hydration <= 0:
        self.health = max(0.0, self.health - 1.0)
    if self.health <= 0:
        self.alive = False
        return None

    features = self.local_features(world, agents)
    if self.hidden_state is None:
        self.hidden_state = self.brain.initial_hidden()

    self._planning_stride = 2 if self.goal_stack and not self.goal_stack.is_empty() else 4
    use_planning = tick >= getattr(self, '_next_planning_tick', 0)
    if use_planning:
        self._next_planning_tick = tick + self._planning_stride

    research_mode = self._need_inv_cooldown <= 0 or (self.goal_stack and not self.goal_stack.is_empty())
    brain_step = self.brain.act(
        features,
        self.hidden_state,
        use_planning=use_planning,
        research_mode=research_mode,
    )
    self.hidden_state = brain_step['next_hidden']
    action_list = brain_step['action_list']
    action = {
        'move_x':    action_list[0],
        'move_y':    action_list[1],
        'forage':    action_list[2],
        'cooperate': action_list[3],
        'attack':    action_list[4],
        'build':     action_list[5],
    }

    reward = 0.0
    mode = 'idle'

    if self.goal_stack is not None:
        goal_action, goal_shaping = agent_tick_with_goals(
            self,
            world.get_cell(*self.pos),
            {
                'tick': tick,
                'season_state': season_state or {},
                'weather_state': weather_state or {},
                'tribes': tribes,
                'economy': economy,
                'technology': technology,
            },
            tick,
        )
        self._last_goal_action = goal_action
        reward += 0.6 * goal_shaping
        if goal_action is not None:
            self._next_planning_tick = tick
            if goal_action in {'collect', 'harvest'}:
                action['forage'] = max(action['forage'], 0.7)
            elif goal_action in {'place_on_heat', 'arch', 'build_dome', 'fire_pottery', 'form'}:
                action['build'] = max(action['build'], 0.7)
            elif goal_action == 'wait':
                action['move_x'] = 0.0
                action['move_y'] = 0.0
                action['forage'] = 0.0
                action['attack'] = 0.0

    self.primitive_move(world, action)
    apply_structure_effects(self, world.get_cell(*self.pos))

    stage_foraging = stage.get('foraging_mult', 1.0)
    if not self.is_sleeping:
        if action['forage'] > 0.0:
            gained = self._forage(world, mods)
            reward += gained * 0.05 * stage_foraging
            if gained > 0:
                mode = 'forage'
                self._collect_herbs(world)

        if action['cooperate'] > 0.2:
            reward += self._cooperate(agents, mods, tick)
            mode = 'cooperate'

        if action['attack'] > 0.5 and stage.get('can_attack', True):
            reward += self._attack(agents, mods)
            mode = 'attack'

        if action['build'] > 0.4 and stage.get('can_build', True):
            self._collect_resources(world)
            self._build(world)
            mode = 'build'

        self._maybe_craft_tool()
        self._try_remedy()
        self._share_remedy(agents)

    self.last_action_mode = mode

    reward += territory_reward_for_agent(self, world)

    if stage.get('can_reproduce', True):
        self._try_reproduce(agents)
    child_genes = self.progress_pregnancy()

    if tick % 3 == 0:
        reward += social_learning_step(self, agents, tick)

    if self._need_inv_cooldown <= 0:
        compute_need_vector(self, world.get_cell(*self.pos))
        inv_result = agent_invent_from_need(self, world.get_cell(*self.pos), world.get_cell(*self.pos), tick)
        if inv_result:
            reward += 0.5
            self.endocrine.apply_discovery(1.0)
        self._need_inv_cooldown = agent_module.NEED_INVENTION_INTERVAL
    else:
        self._need_inv_cooldown -= 1

    inv_prob = agent_module.INVENTION_BASE_PROB + agent_module.INVENTION_CURIOSITY_MULT * self.genes.get('curiosity', 0.5)
    if tick % 3 == 0 and random.random() < inv_prob:
        invented = agent_try_invention(self, world.get_cell(*self.pos), world.get_cell(*self.pos))
        if invented:
            reward += 1.0
            self.endocrine.apply_discovery(1.0)

    if tick % 4 == 0 and random.random() < 0.18:
        cooked = agent_try_cook(self, world.get_cell(*self.pos))
        if cooked:
            reward += 0.3
            self.endocrine.apply_substance('cooked_meat', 1.0)

    if economy is not None:
        economy.maybe_trade(self, agents)

    next_features_raw = self.local_features(world, agents)
    intrinsic = self.brain.intrinsic_reward(
        brain_step['hidden_in'],
        brain_step['action_tensor'],
        next_features_raw,
    )
    reward += 0.3 * intrinsic

    next_obs_t = torch.tensor(
        next_features_raw,
        dtype=torch.float32,
        device=brain_step['hidden_in'].device,
    )
    self.brain.episodic_memory.novelty(next_obs_t)

    reward = _maybe_mark_language(self, world.get_cell(*self.pos), tick, np.asarray(next_features_raw, dtype=np.float32), reward)
    _maybe_collect_language_convergence(self, agents, tick)
    _compact_material_inventory(self, getattr(self, '_inventory_cap', 24))

    cognition_mult = mods.get('cognition', 1.0)
    effective_reward = reward * cognition_mult
    self.brain.store_transition(
        brain_step['obs_tensor'],
        brain_step['hidden_in'],
        brain_step['action_tensor'],
        brain_step['log_prob'],
        brain_step['value'],
        effective_reward,
        not self.alive,
        next_features_raw,
    )

    loss = self.brain.maybe_train()
    if loss is not None:
        self.last_loss = loss

    self.last_reward = effective_reward
    self.reproduction_cooldown = max(0, self.reproduction_cooldown - 1)

    x, y = self.pos
    nearby_for_tom = [
        a for a in agents
        if a is not self and a.alive
        and abs(a.pos[0] - x) <= 2
        and abs(a.pos[1] - y) <= 2
    ]
    for other in nearby_for_tom:
        self.tom.observe_agent(other, tick)

    h = self.endocrine.h
    arousal = min(1.0, max(0.0, (h[0] + h[1]) / 2))
    context_hormones = [h[0], h[3], h[4], h[1]]
    self.emotional_memory.encode_experience(
        stimulus=mode,
        valence=min(1.0, max(-1.0, reward * 0.1)),
        arousal=arousal,
        context_hormones=context_hormones,
        tick=tick,
    )

    return child_genes


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _patch_discovery_registry()
    Agent._collect_resources = _collect_resources_from_materials  # type: ignore[assignment]
    Agent._build = _build_from_resources  # type: ignore[assignment]
    Agent.update = patched_update  # type: ignore[assignment]

    # Lower system-wide load a bit while keeping the emergent model intact.
    agent_mod.INVENTION_BASE_PROB = 0.08
    agent_mod.INVENTION_CURIOSITY_MULT = 0.05
    agent_mod.NEED_INVENTION_INTERVAL = 12
    agent_mod.MIN_REPRODUCTION_AGE = CHILD_MAX
    agent_mod.ELDER_AGE = ADULT_MAX
    agent_mod.STAGE_CHILD = CHILD_MAX
    agent_mod.STAGE_ELDER = ADULT_MAX
    agent_mod.AGE_LIMIT = 5000
    agent_mod.AGE_HEALTH_DECAY_START = ADULT_MAX
    agent_mod.AGE_HEALTH_DECAY_HARD = 4500

    brain_mod = __import__('artificial_society.agents.brain', fromlist=['dummy'])
    brain_mod.PLAN_CANDIDATES = 8
    brain_mod.PLAN_HORIZON = 2
    brain_mod.PLAN_HORIZON_RESEARCH = 6

    _PATCHED = True


apply_patches()
