from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch

from artificial_society.agents import agent as agent_mod
from artificial_society.agents.agent import Agent
from artificial_society.agents.brain import Brain
from artificial_society.agents.life_stage import CHILD_MAX, ADULT_MAX, get_stage_stats
from artificial_society.environment.materials import DiscoveryRegistry
from artificial_society.environment.resources import maybe_build_structure
from artificial_society.environment.structures import BUILD_ENERGY_COST, apply_structure_effects
from artificial_society.environment.territory import territory_reward_for_agent
from artificial_society.systems.goal_stack import GoalStack
from artificial_society.systems.goal_stack_ext import agent_tick_with_goals
from artificial_society.systems.invention import agent_try_cook, agent_try_invention
from artificial_society.systems.language import (
    TOKEN_WORLD,
    TokenMemory,
    agent_mark,
    agent_observe_token,
)
from artificial_society.systems.need_driven_invention import agent_invent_from_need, compute_need_vector
from artificial_society.systems.remedy import evaluate_remedy, record_cure_discovery, share_remedy_knowledge
from artificial_society.systems import economy as economy_mod
from artificial_society.systems import social_learning as social_learning_mod

_PATCHED = False
_ORIGINAL_SPAWN_RANDOM = None
_ORIGINAL_SPAWN_CHILD = None
_ORIGINAL_TRADE = None
_ORIGINAL_SOCIAL_LEARNING = None

_RESOURCE_ALIASES = {
    'wood': ('dry_wood', 'wet_wood', 'wood'),
    'stone': ('stone', 'flint'),
    'fiber': ('fiber', 'dry_grass', 'crushed_herb', 'leaf'),
}


def _ensure_class_properties() -> None:
    if not hasattr(Agent, 'x'):
        Agent.x = property(lambda self: self.pos[0])  # type: ignore[attr-defined]
    if not hasattr(Agent, 'y'):
        Agent.y = property(lambda self: self.pos[1])  # type: ignore[attr-defined]


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
    if not hasattr(agent, '_cached_nearby_agents'):
        agent._cached_nearby_agents = []
    if not hasattr(agent, '_cached_nearby_radius'):
        agent._cached_nearby_radius = 2
    if not hasattr(agent, '_disease_immunity'):
        agent._disease_immunity = {}


def _compact_material_inventory(agent: Agent, max_entries: int = 24) -> None:
    inv = getattr(agent, 'material_inventory', None)
    if not inv:
        return

    cleaned = {k: float(v) for k, v in inv.items() if float(v) > 0.01}
    if len(cleaned) <= max_entries:
        agent.material_inventory = cleaned
        return

    protected = {k: v for k, v in cleaned.items() if not k.startswith('mat_')}
    discovered = sorted(
        ((k, v) for k, v in cleaned.items() if k.startswith('mat_')),
        key=lambda item: (-item[1], item[0]),
    )

    remaining = max_entries - len(protected)
    kept = dict(protected)
    if remaining > 0:
        for mat_id, qty in discovered[:remaining]:
            kept[mat_id] = qty

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


def _patch_discovery_registry() -> None:
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
    tool_bonus = 0.20 if getattr(self, 'tool', None) == 'sharp_stone' else 0.0

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


def _cached_nearby_agents(agent: Agent, agents: list[Agent], radius: int = 2) -> list[Agent]:
    cached = getattr(agent, '_cached_nearby_agents', None)
    cached_radius = getattr(agent, '_cached_nearby_radius', None)
    if cached is not None and cached_radius == radius:
        return cached
    x, y = agent.pos
    nearby = [
        other for other in agents
        if other is not agent and other.alive
        and abs(other.pos[0] - x) <= radius
        and abs(other.pos[1] - y) <= radius
    ]
    agent._cached_nearby_agents = nearby
    agent._cached_nearby_radius = radius
    return nearby


def _social_learning_step_cached(agent: Agent, agents: list[Agent], tick: int) -> float:
    causal_mem = getattr(agent, 'causal_memory', None)
    if causal_mem is None:
        return 0.0

    nearby = _cached_nearby_agents(agent, agents, radius=2)
    if not nearby:
        return 0.0

    reward = 0.0
    agent_reward = getattr(agent, 'last_reward', 0.0)

    for other in nearby:
        other_mem = getattr(other, 'causal_memory', None)
        if other_mem is None:
            continue

        other_reward = getattr(other, 'last_reward', 0.0)
        curiosity = agent.genes.get('curiosity', 0.5)
        trust = agent.trust.get(other.id, 0.0)

        observe_chance = social_learning_mod.OBSERVE_PROB + 0.15 * curiosity
        if other_reward > 0.5 and random.random() < observe_chance:
            seq = other_mem.sample_for_transmission()
            if seq:
                fidelity = social_learning_mod.FIDELITY_BASE + social_learning_mod.FIDELITY_TRUST_BONUS * max(0.0, trust)
                causal_mem.receive_transmitted(seq, fidelity=fidelity)
                reward += 0.05

        if trust > 0.5 and random.random() < social_learning_mod.TEACH_PROB:
            seq = causal_mem.sample_for_transmission()
            other_causal = getattr(other, 'causal_memory', None)
            if seq and other_causal:
                other_causal.receive_transmitted(seq, fidelity=social_learning_mod.FIDELITY_BASE + social_learning_mod.FIDELITY_TRUST_BONUS * trust)
                reward += 0.06

        if (
            other_reward > agent_reward * social_learning_mod.IMITATION_SUCCESS_RATIO
            and other_reward > 0.3
            and trust >= social_learning_mod.IMITATION_MIN_TRUST
            and random.random() < 0.15 + 0.20 * curiosity
        ):
            agent_brain = getattr(agent, 'brain', None)
            other_brain = getattr(other, 'brain', None)
            if agent_brain is not None and other_brain is not None:
                agent_brain.imitate_from(other_brain)
                reward += 0.08

    return reward


def _maybe_trade_cached(self, agent: Agent, agents: list[Agent]) -> None:
    nearby = _cached_nearby_agents(agent, agents, radius=2)
    if not nearby:
        return

    for other in nearby:
        if agent.trust.get(other.id, 0.0) < 0.1:
            continue
        for give_res, want_res in [('wood', 'stone'), ('stone', 'fiber'), ('fiber', 'wood')]:
            if agent.resources[give_res] > 1 and other.resources[want_res] > 1:
                agent.resources[give_res] -= 1
                other.resources[give_res] += 1
                other.resources[want_res] -= 1
                agent.resources[want_res] += 1
                self.trade_count += 1
                trust_gain = 0.03 + 0.02 * self.prices[give_res]
                agent.trust[other.id] = min(1.0, agent.trust.get(other.id, 0.0) + trust_gain)
                other.trust[agent.id] = min(1.0, other.trust.get(agent.id, 0.0) + trust_gain)
                break


def _maybe_mark_language(agent: Agent, cell: dict, tick: int, context_vec: np.ndarray, reward: float) -> float:
    if reward < 0.9 or tick < getattr(agent, '_language_retry_tick', 0):
        return reward
    if random.random() > 0.12:
        return reward

    token_id = agent_mark(agent, cell, context_vec, tick)
    if token_id:
        agent._language_retry_tick = tick + 12
        return reward + 0.1
    return reward


def _observe_tokens(agent: Agent, cell: dict, context_vec: np.ndarray, reward: float) -> None:
    x, y = agent.pos
    tokens = TOKEN_WORLD.tokens_at(x, y, radius=1)
    if not tokens:
        return
    signal = max(0.0, reward)
    for token in tokens:
        agent_observe_token(agent, token, context_vec, reward_signal=signal)


def _maybe_collect_language_convergence(agent: Agent, agents: list[Agent], tick: int) -> None:
    if tick % 90 != 0 or getattr(agent, 'id', 0) != 1:
        return
    memories = [getattr(a, 'token_memory', None) for a in agents]
    memories = [m for m in memories if m is not None]
    if len(memories) >= 2:
        TOKEN_WORLD.check_convergence(memories, tick)
        TOKEN_WORLD.tick_decay()


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

    self.endocrine.update(self, world)
    mods = self.endocrine.modifiers()
    stage = get_stage_stats(self.age)

    self._age_tick()
    self._disease_tick(world)
    if not self.alive:
        return None

    self._sleep_tick(mods)

    current_cell = world.get_cell(*self.pos)
    structure_mods = apply_structure_effects(self, current_cell)

    move_cost = 0.5 * mods.get('move_cost_mult', 1.0) * stage['move_cost_mult'] * structure_mods.get('cold_factor', 1.0)
    hydration_loss = 0.3 * mods.get('hydration_loss_mult', 1.0) * stage['hydration_loss_mult']
    self.energy = max(0.0, self.energy - move_cost)
    self.hydration = max(0.0, min(100.0, self.hydration - hydration_loss + structure_mods.get('hydration_bonus', 0.0)))
    self.health = max(0.0, self.health - mods.get('health_drain', 0.0) * structure_mods.get('disease_factor', 1.0))

    if self.energy <= 0:
        self.health = max(0.0, self.health - 1.5)
    if self.hydration <= 0:
        self.health = max(0.0, self.health - 1.0)
    if self.health <= 0:
        self.alive = False
        return None

    nearby_agents = _cached_nearby_agents(self, agents, radius=2)
    features = self.local_features(world, agents)
    if self.hidden_state is None:
        self.hidden_state = self.brain.initial_hidden()

    self._planning_stride = 2 if getattr(self, 'goal_stack', None) and not self.goal_stack.is_empty() else 4
    use_planning = tick >= getattr(self, '_next_planning_tick', 0)
    if use_planning:
        self._next_planning_tick = tick + self._planning_stride

    research_mode = self._need_inv_cooldown <= 0 or (getattr(self, 'goal_stack', None) is not None and not self.goal_stack.is_empty())
    brain_step = self.brain.act(
        features,
        self.hidden_state,
        use_planning=use_planning,
        research_mode=research_mode,
    )
    self.hidden_state = brain_step['next_hidden']
    action_list = brain_step['action_list']
    action = {
        'move_x': action_list[0],
        'move_y': action_list[1],
        'forage': action_list[2],
        'cooperate': action_list[3],
        'attack': action_list[4],
        'build': action_list[5],
    }

    reward = 0.0
    mode = 'idle'

    if getattr(self, 'goal_stack', None) is not None:
        goal_action, goal_shaping = agent_tick_with_goals(
            self,
            current_cell,
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
    current_cell = world.get_cell(*self.pos)
    structure_mods = apply_structure_effects(self, current_cell)

    if not self.is_sleeping:
        if action['forage'] > 0.0:
            gained = self._forage(world, {**mods, 'forage_eff': mods.get('forage_eff', 1.0) * (1.0 + structure_mods.get('forage_bonus', 0.0))})
            reward += gained * 0.05 * stage.get('foraging_mult', 1.0)
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
        reward += _social_learning_step_cached(self, agents, tick)

    if self._need_inv_cooldown <= 0:
        compute_need_vector(self, current_cell)
        inv_result = agent_invent_from_need(self, current_cell, current_cell, tick)
        if inv_result:
            reward += 0.5
            self.endocrine.apply_discovery(1.0)
        self._need_inv_cooldown = agent_mod.NEED_INVENTION_INTERVAL
    else:
        self._need_inv_cooldown -= 1

    inv_prob = agent_mod.INVENTION_BASE_PROB + agent_mod.INVENTION_CURIOSITY_MULT * self.genes.get('curiosity', 0.5)
    if tick % 3 == 0 and random.random() < inv_prob:
        invented = agent_try_invention(self, current_cell, current_cell)
        if invented:
            reward += 1.0
            self.endocrine.apply_discovery(1.0)

    if tick % 4 == 0 and random.random() < 0.18:
        cooked = agent_try_cook(self, current_cell)
        if cooked:
            reward += 0.3
            self.endocrine.apply_substance('cooked_meat', 1.0)

    if economy is not None:
        _maybe_trade_cached(economy, self, agents)

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

    context_vec = np.asarray(next_features_raw, dtype=np.float32)
    reward = _maybe_mark_language(self, current_cell, tick, context_vec, reward)
    _observe_tokens(self, current_cell, context_vec, reward)
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

    for other in nearby_agents:
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


def _wrap_classmethods() -> None:
    global _ORIGINAL_SPAWN_RANDOM, _ORIGINAL_SPAWN_CHILD

    if _ORIGINAL_SPAWN_RANDOM is None:
        _ORIGINAL_SPAWN_RANDOM = Agent.spawn_random.__func__

        def spawn_random(cls, x, y):
            agent = _ORIGINAL_SPAWN_RANDOM(cls, x, y)
            _ensure_runtime_fields(agent)
            return agent

        Agent.spawn_random = classmethod(spawn_random)  # type: ignore[assignment]

    if _ORIGINAL_SPAWN_CHILD is None:
        _ORIGINAL_SPAWN_CHILD = Agent.spawn_child.__func__

        def spawn_child(cls, x, y, genes, generation=1, parent_id=None, tribe_id=None, parent=None):
            agent = _ORIGINAL_SPAWN_CHILD(cls, x, y, genes, generation=generation, parent_id=parent_id, tribe_id=tribe_id, parent=parent)
            _ensure_runtime_fields(agent)
            return agent

        Agent.spawn_child = classmethod(spawn_child)  # type: ignore[assignment]


def apply_emergence_integration() -> None:
    global _PATCHED, _ORIGINAL_TRADE, _ORIGINAL_SOCIAL_LEARNING
    if _PATCHED:
        return

    _ensure_class_properties()
    _patch_discovery_registry()
    _wrap_classmethods()

    Agent._collect_resources = _collect_resources_from_materials  # type: ignore[assignment]
    Agent._build = _build_from_resources  # type: ignore[assignment]
    Agent.update = patched_update  # type: ignore[assignment]

    if _ORIGINAL_TRADE is None:
        _ORIGINAL_TRADE = economy_mod.EconomySystem.maybe_trade
        economy_mod.EconomySystem.maybe_trade = _maybe_trade_cached  # type: ignore[assignment]

    if _ORIGINAL_SOCIAL_LEARNING is None:
        _ORIGINAL_SOCIAL_LEARNING = social_learning_mod.social_learning_step
        social_learning_mod.social_learning_step = _social_learning_step_cached  # type: ignore[assignment]

    # Agent- and brain-level tuning is now baked into agents/agent.py and
    # agents/brain.py (the source of truth) instead of mutated here at import.

    _PATCHED = True


apply_emergence_integration()
