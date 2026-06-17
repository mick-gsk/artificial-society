import random
from dataclasses import dataclass, field

from artificial_society.agents.brain import Brain, INPUT_SIZE
from artificial_society.agents.memory import EpisodicMemory
from artificial_society.agents.genetics import random_genes, inherit_genes
from artificial_society.agents.communication import CommunicationSystem
from artificial_society.environment.resources import apply_consumption, maybe_build_structure
from artificial_society.environment.herbs import available_herbs, collect_herb, regrow_herbs
from artificial_society.systems.remedy import (
    evaluate_remedy,
    record_cure_discovery,
    share_remedy_knowledge,
    try_infect_agent,
    REMEDY_REGISTRY,
)
from artificial_society.systems.culture import CausalMemory
from artificial_society.systems.invention import agent_try_invention
from artificial_society.systems.social_learning import social_learning_step

MAX_ENERGY = 240.0
INITIAL_ENERGY = 120.0
CHILD_START_ENERGY = 100.0
REPRODUCTION_ENERGY = 115.0
REPRODUCTION_COST = 32.0
REPRODUCTION_COOLDOWN = 300
MIN_REPRODUCTION_AGE = 140
GESTATION_TIME = 190
AGE_LIMIT = 5000
PLANT_ENERGY = 28.0
MEAT_ENERGY = 40.0
CORPSE_ENERGY = 36.0

SHARP_STONE_FORAGE_BONUS = 0.30
SHARP_STONE_COLLECT_BONUS = 0.20

# Sleeping thresholds
SLEEP_PRESSURE_THRESHOLD = 0.55   # agent tries to sleep above this
SLEEP_ENERGY_REGEN = 0.40         # energy/tick while sleeping
SLEEP_HEALTH_REGEN = 0.12         # health/tick while sleeping


def _ensure_new_fields(agent):
    """Backward-compat: inject missing fields and rebuild brain if input size changed."""
    if not hasattr(agent, 'causal_memory') or agent.causal_memory is None:
        agent.causal_memory = CausalMemory(capacity=32)
    if not hasattr(agent, 'material_inventory') or agent.material_inventory is None:
        agent.material_inventory = {}
    if not hasattr(agent, 'sleep_debt') or agent.sleep_debt is None:
        agent.sleep_debt = 0.0
    if not hasattr(agent, 'is_sleeping'):
        agent.is_sleeping = False
    if not hasattr(agent, 'brain') or agent.brain is None:
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    elif agent.brain.input_size != INPUT_SIZE:
        print(f'[compat] Agent {agent.id}: brain input_size={agent.brain.input_size} != {INPUT_SIZE}, rebuilding.')
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()


@dataclass
class Agent:
    id_counter: int = 0
    id: int = 0
    energy: float = INITIAL_ENERGY
    health: float = 100.0
    hydration: float = 100.0
    age: int = 0
    pos: tuple = (0, 0)
    genes: dict = field(default_factory=random_genes)
    memory: EpisodicMemory = field(default_factory=lambda: EpisodicMemory(10))
    brain: Brain = field(default_factory=Brain)
    communication: CommunicationSystem = field(default_factory=CommunicationSystem)
    message_vector: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    trust: dict = field(default_factory=dict)
    tribe_id: int | None = None
    resources: dict = field(default_factory=lambda: {'wood': 0, 'stone': 0, 'fiber': 0})
    tool: str | None = None
    alive: bool = True
    learning_score: float = 0.0
    reproduction_cooldown: int = 0
    hidden_state: object = None
    last_reward: float = 0.0
    last_loss: float = 0.0
    sex: str = 'm'
    pregnant: bool = False
    gestation: float = 0.0
    stored_child_genes: dict | None = None
    children: int = 0
    generation: int = 0
    parent_id: int | None = None
    plant_eaten: int = 0
    meat_eaten: int = 0
    birth_tick: int = 0
    sick: float = 0.0
    last_action_mode: str = 'idle'
    disease_id: str | None = None
    remedy_knowledge: dict = field(default_factory=dict)
    herbs_carried: dict = field(default_factory=dict)
    causal_memory: CausalMemory = field(default_factory=lambda: CausalMemory(capacity=32))
    material_inventory: dict = field(default_factory=dict)
    # Sleep system
    sleep_debt: float = 0.0
    is_sleeping: bool = False

    @classmethod
    def spawn_random(cls, x, y):
        cls.id_counter += 1
        genes = random_genes()
        memory_capacity = genes['memory_capacity']
        brain = Brain()
        agent = cls(
            id=cls.id_counter,
            pos=(x, y),
            genes=genes,
            memory=EpisodicMemory(memory_capacity),
            brain=brain,
            sex=random.choice(['m', 'f']),
        )
        agent.hidden_state = brain.initial_hidden()
        return agent

    @classmethod
    def spawn_child(cls, x, y, genes, generation=1, parent_id=None, tribe_id=None):
        cls.id_counter += 1
        brain = Brain()
        agent = cls(
            id=cls.id_counter,
            pos=(x, y),
            genes=genes,
            memory=EpisodicMemory(genes['memory_capacity']),
            brain=brain,
            energy=CHILD_START_ENERGY,
            sex=random.choice(['m', 'f']),
            generation=generation,
            parent_id=parent_id,
            tribe_id=tribe_id,
        )
        agent.hidden_state = brain.initial_hidden()
        agent.reproduction_cooldown = REPRODUCTION_COOLDOWN
        return agent

    def display_color(self):
        meat_bias = max(0.0, min(1.0, (self.genes['diet_preference'] + 1.0) * 0.5))
        plant_bias = max(0.0, min(1.0, (-self.genes['diet_preference'] + 1.0) * 0.5))
        g = max(70, min(255, int(80 + 120 * self.genes['cooperation'] + 35 * self.genes['memory_retention'])))
        r = max(70, min(255, int(80 + 100 * meat_bias + 60 * self.genes['aggression'])))
        b = max(70, min(255, int(80 + 100 * plant_bias + 50 * self.genes['plasticity'] / 1.8)))
        return (r, g, b)

    def can_reproduce(self):
        return self.alive and self.age >= MIN_REPRODUCTION_AGE and self.energy >= REPRODUCTION_ENERGY and self.reproduction_cooldown <= 0 and not self.pregnant

    def local_features(self, world, agents):
        _ensure_new_fields(self)
        x, y = self.pos
        cell = world.get_cell(x, y)
        near = [a for a in agents if a is not self and abs(a.pos[0] - x) <= int(self.genes['sense_radius']) and abs(a.pos[1] - y) <= int(self.genes['sense_radius']) and a.alive]
        friends = sum(1 for a in near if self.trust.get(a.id, 0.0) > 0.2)
        retrieval = self.memory.retrieval_features(x, y, cell['tick'], world.width, world.height)
        avg_trust = sum(self.trust.values()) / len(self.trust) if self.trust else 0.0
        herb_presence = min(1.0, len(available_herbs(cell)) / 5.0)
        warmth = min(1.0, cell.get('warmth', 0.0))
        mat_count = min(1.0, len(cell.get('materials', {})) / 6.0)
        causal_feats = self.causal_memory.feature_vector()
        inv_size = min(1.0, sum(self.material_inventory.values()) / 5.0)

        # --- Day/Night features (3 new inputs) ---
        dn = world.day_state
        light = dn.get('light', 1.0)
        sleep_pressure = dn.get('sleep_pressure', 0.0)
        is_night = 1.0 if dn.get('phase') == 'night' else 0.0

        # --- Tech features: how many unique causal sequences this agent knows ---
        # Normalized: 0..1 over an expected max of 16 sequences
        tech_knowledge = min(1.0, len(self.causal_memory.sequences) / 16.0)
        # Does this agent know any remedy?
        has_remedy = 1.0 if self.remedy_knowledge else 0.0
        # Sleep debt (0..1)
        sleep_debt_feat = min(1.0, self.sleep_debt / 80.0)

        return [
            self.energy / MAX_ENERGY,
            self.health / 100.0,
            self.hydration / 100.0,
            min(1.0, self.age / AGE_LIMIT),
            cell['food'] / 180.0,
            cell['water'] / 100.0,
            (cell['temperature'] + 20) / 72.0,
            cell['danger'] / 100.0,
            cell['disease'] / 100.0,
            cell['soil_fertility'] / 100.0,
            cell['pollution'] / 100.0,
            cell['carrying_capacity'] / 100.0,
            cell['moisture'] / 100.0,
            cell['ash'] / 100.0,
            cell['disturbance'] / 100.0,
            min(1.0, len(near) / 10.0),
            min(1.0, friends / 6.0),
            self.genes['curiosity'],
            self.genes['aggression'],
            self.genes['cooperation'],
            self.genes['sociality'],
            1.0 if self.tool else 0.0,
            avg_trust * 0.5 + 0.5,
            min(1.0, self.resources['wood'] / 4.0),
            min(1.0, self.resources['stone'] / 4.0),
            min(1.0, self.resources['fiber'] / 4.0),
            max(-1.0, min(1.0, self.last_reward / 10.0)),
            max(0.0, min(1.0, self.sick / 100.0)),
            herb_presence,
            warmth,
            mat_count,
            inv_size,
            # Day/Night
            light,
            sleep_pressure,
            is_night,
            # Tech / knowledge state
            tech_knowledge,
            has_remedy,
            sleep_debt_feat,
            *causal_feats,
            *retrieval,
        ]

    def visible_cells(self, world):
        x, y = self.pos
        # Night vision penalty: sense radius halved at full night
        light = world.day_state.get('light', 1.0)
        vision_mult = 0.5 + 0.5 * light
        radius = max(1, int(round((self.genes['vision'] + 0.35 * self.genes['sense_radius']) * vision_mult)))
        return world.neighbors(x, y, radius)

    def progress_pregnancy(self):
        child = None
        if self.pregnant:
            self.gestation -= 1
            self.energy -= 0.03
            if self.gestation <= 0 and self.stored_child_genes is not None:
                child = self.stored_child_genes
                self.pregnant = False
                self.gestation = 0
                self.stored_child_genes = None
        return child

    def primitive_move(self, world, action):
        # Sleeping agents don't move
        if self.is_sleeping:
            return
        # Night: slower movement (higher energy cost, lower explore probability)
        dx = 1 if action['move_x'] > 0.33 else -1 if action['move_x'] < -0.33 else 0
        dy = 1 if action['move_y'] > 0.33 else -1 if action['move_y'] < -0.33 else 0
        if dx == 0 and dy == 0 and action['explore'] > 0.3 and random.random() < 0.35:
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if world.in_bounds(nx, ny) and world.get_biome(nx, ny) != 'water':
            self.pos = (nx, ny)
            if dx or dy:
                self.last_action_mode = 'move'

    def apply_sleep(self, world):
        """
        Sleep system:
        - Sleep debt builds during the night phase proportional to sleep_pressure.
        - Above SLEEP_PRESSURE_THRESHOLD, agent may fall asleep.
        - Sleeping agents regenerate energy and health, but cannot act.
        - Dawn clears sleep debt and wakes the agent.
        """
        _ensure_new_fields(self)
        dn = world.day_state
        phase = dn.get('phase', 'day')
        sleep_pressure = dn.get('sleep_pressure', 0.0)

        if phase == 'day':
            # Wake up and clear debt
            self.is_sleeping = False
            self.sleep_debt = max(0.0, self.sleep_debt - 2.0)
            return

        # Build sleep debt at night
        self.sleep_debt = min(100.0, self.sleep_debt + sleep_pressure * 0.8)

        # Decision to sleep: high sleep pressure + not in danger
        cell = world.get_cell(*self.pos)
        danger_here = cell.get('danger', 0.0) + cell.get('disturbance', 0.0)
        safe_enough = danger_here < 25.0

        if not self.is_sleeping and self.sleep_debt > SLEEP_PRESSURE_THRESHOLD * 80 and safe_enough:
            # Prefer to sleep if also near warmth (fire) or in shelter
            warmth = cell.get('warmth', 0.0)
            shelter = cell.get('structures', {}).get('shelter', False)
            sleep_chance = 0.25 + 0.35 * (warmth > 0.3) + 0.25 * bool(shelter)
            if random.random() < sleep_chance:
                self.is_sleeping = True
                self.last_action_mode = 'sleep'

        if self.is_sleeping:
            # Regen while asleep
            self.energy = min(MAX_ENERGY, self.energy + SLEEP_ENERGY_REGEN)
            self.health = min(100.0, self.health + SLEEP_HEALTH_REGEN)
            self.sleep_debt = max(0.0, self.sleep_debt - 3.0)
            self.last_action_mode = 'sleep'
            # Wake if danger arrives
            if danger_here > 40.0 or self.sleep_debt <= 0.0:
                self.is_sleeping = False

    def forage(self, world, intensity):
        _ensure_new_fields(self)
        if self.is_sleeping:
            return 0.0
        cell = world.get_cell(*self.pos)
        if intensity < -0.25:
            return 0.0
        # Night foraging penalty: reduced by light level
        light = world.day_state.get('light', 1.0)
        night_penalty = 0.5 + 0.5 * light   # 0.5 at full dark, 1.0 at full day
        plant_need = max(0.2, 1.0 - self.energy / MAX_ENERGY)
        meat_bias = max(0.0, self.genes['diet_preference'])
        plant_bias = max(0.0, -self.genes['diet_preference'])
        plant_take = min(cell['plant_food'], 2.0 + 8.0 * intensity * (0.5 + plant_need + 0.4 * plant_bias))
        meat_take = min(cell['meat_food'], 1.0 + 5.5 * intensity * (0.35 + plant_need + 0.5 * meat_bias))
        water_need = max(0.3, 1.0 - self.hydration / 100.0)
        water_take = min(cell['water'], 1.0 + 8.0 * intensity * water_need)
        efficiency = 0.75 + 0.35 * self.genes['efficiency']
        tool_mult = (1.0 + SHARP_STONE_FORAGE_BONUS) if self.tool == 'sharp_stone' else 1.0
        cook_bonus = 0.0
        for mat in ('cooked_meat', 'cooked_root'):
            if self.material_inventory.get(mat, 0) > 0.1:
                val = min(self.material_inventory[mat], 1.0)
                self.material_inventory[mat] -= val
                self.energy = min(MAX_ENERGY, self.energy + val * 22.0)
                cook_bonus += 0.25
        plant_gain = plant_take * (PLANT_ENERGY / 8.0) * (1.0 + 0.55 * plant_bias) * efficiency * tool_mult * night_penalty
        meat_gain = meat_take * (MEAT_ENERGY / 6.0) * (1.0 + 0.55 * meat_bias) * efficiency * tool_mult * night_penalty
        apply_consumption(cell, plant=plant_take, meat=meat_take, water=water_take)
        self.energy = min(MAX_ENERGY, self.energy + plant_gain + meat_gain)
        self.hydration = min(100.0, self.hydration + water_take * 8.5)
        self.health = min(100.0, self.health + water_take * 0.16)
        if plant_take > 0.5:
            self.plant_eaten += 1
        if meat_take > 0.3:
            self.meat_eaten += 1
        if plant_take + meat_take + water_take > 0.8:
            self.last_action_mode = 'gather'
        self.memory.remember_resource(self.pos, cell['food'], cell['water'], cell['tick'])
        consumed_tags: list[str] = []
        if plant_take > 0.3:
            consumed_tags.append('plant_food')
        if meat_take > 0.3:
            consumed_tags.append('meat')
        if water_take > 0.3:
            consumed_tags.append('water')
        forage_reward = 0.025 * (plant_gain + meat_gain) + 0.18 * water_take + cook_bonus
        forage_reward += self._try_use_herbs(cell, consumed_tags)
        return forage_reward

    def _try_use_herbs(self, cell: dict, consumed_tags: list[str]) -> float:
        herbs_here = available_herbs(cell)
        if not herbs_here:
            return 0.0
        reward = 0.0
        curiosity = self.genes.get('curiosity', 0.5)
        sick_drive = min(1.0, self.sick / 40.0)
        sample_prob = 0.15 + 0.30 * curiosity + 0.40 * sick_drive
        for tag in herbs_here:
            if random.random() < sample_prob:
                taken = collect_herb(cell, tag, amount=1.0)
                if taken > 0:
                    self.herbs_carried[tag] = self.herbs_carried.get(tag, 0.0) + taken
                    consumed_tags.append(tag)
                    self.last_action_mode = 'forage_herb'
        if self.disease_id and consumed_tags:
            prev_disease = self.disease_id
            cure_bonus = evaluate_remedy(self, consumed_tags)
            if cure_bonus > 0:
                record_cure_discovery(self, prev_disease, consumed_tags)
                reward += cure_bonus * 2.5
        return reward

    def collect_materials(self, world, intensity):
        _ensure_new_fields(self)
        if self.is_sleeping or intensity < 0.15:
            return 0.0
        cell = world.get_cell(*self.pos)
        gain = 0.0
        tool_bonus = SHARP_STONE_COLLECT_BONUS if self.tool == 'sharp_stone' else 0.0
        if cell['biome'] == 'forest' and random.random() < 0.08 + 0.16 * intensity + tool_bonus:
            self.resources['wood'] += 1
            gain += 0.15
        if cell['biome'] == 'mountain' and random.random() < 0.06 + 0.14 * intensity + tool_bonus:
            self.resources['stone'] += 1
            gain += 0.15
        if cell['biome'] in ('grassland', 'swamp') and random.random() < 0.08 + 0.18 * intensity + tool_bonus:
            self.resources['fiber'] += 1
            gain += 0.12
        slot = cell.get('materials', {})
        for mat, qty in list(slot.items()):
            if qty > 0.1 and random.random() < 0.15 * intensity:
                take = min(qty, 0.5)
                slot[mat] -= take
                self.material_inventory[mat] = self.material_inventory.get(mat, 0.0) + take
                gain += 0.05
        return gain

    def maybe_craft(self, technology, intensity):
        return 0.0

    def maybe_build(self, world, intensity):
        if self.is_sleeping or intensity < 0.55:
            return None
        cell = world.get_cell(*self.pos)
        built = maybe_build_structure(cell, self.resources)
        if built is not None:
            self.last_action_mode = f'build:{built}'
        return built

    def maybe_signal(self, features, action_values):
        if self.is_sleeping:
            return
        self.communication.emit(self, features, action_values)
        if abs(sum(self.message_vector)) > 0.35:
            self.last_action_mode = 'signal'

    def update_social(self, agents, tribes, tick, action, features_before):
        _ensure_new_fields(self)
        if self.is_sleeping:
            return 0.0
        nearby = [a for a in agents if a is not self and a.alive and abs(a.pos[0] - self.pos[0]) <= 1 and abs(a.pos[1] - self.pos[1]) <= 1]
        social_reward = 0.0
        for other in nearby:
            helpful = other.message_vector[0] > -0.15
            prior = self.trust.get(other.id, 0.0)
            delta = 0.015 if helpful else -0.008
            delta += 0.018 * self.genes['sociality']
            if action['communicate'] > 0.1:
                before_food = features_before[4]
                before_danger = features_before[7]
                after_food = other.local_features_cache[4] if hasattr(other, 'local_features_cache') else before_food
                after_danger = other.local_features_cache[7] if hasattr(other, 'local_features_cache') else before_danger
                info_bonus = self.communication.evaluate_message_usefulness(other, self, before_food, after_food, before_danger, after_danger)
                delta += info_bonus * 0.18
                social_reward += info_bonus
                if self.tribe_id is not None and other.tribe_id == self.tribe_id:
                    social_reward += self._share_tribe_resources(other)
                if self.trust.get(other.id, 0.0) > 0.3:
                    if share_remedy_knowledge(self, other):
                        social_reward += 0.15
            self.trust[other.id] = max(-1.0, min(1.0, prior + delta))
            self.memory.remember_social(other.id, self.trust[other.id], helpful, tick)
        tribes.consider_join(self, nearby)
        social_reward += social_learning_step(self, agents, tick)
        return social_reward

    def _share_tribe_resources(self, other) -> float:
        reward = 0.0
        for res in ('wood', 'stone', 'fiber'):
            if self.resources[res] >= 3 and other.resources[res] == 0:
                self.resources[res] -= 1
                other.resources[res] += 1
                reward += 0.08
                break
        if self.energy > 180.0 and other.energy < 80.0:
            transfer = min(20.0, self.energy - 160.0)
            self.energy -= transfer
            other.energy += transfer
            reward += 0.10
        return reward

    def maybe_reproduce(self, agents, action):
        if self.is_sleeping:
            return False
        if not self.can_reproduce() or action['communicate'] < 0.15 or action['explore'] < 0.05:
            return False
        nearby = [a for a in agents if a is not self and a.alive and abs(a.pos[0] - self.pos[0]) <= 1 and abs(a.pos[1] - self.pos[1]) <= 1]
        random.shuffle(nearby)
        for other in nearby:
            if not other.can_reproduce() or other.sex == self.sex or other.pregnant:
                continue
            mother = self if self.sex == 'f' else other
            father = other if mother is self else self
            mother.pregnant = True
            mother.gestation = GESTATION_TIME / max(0.5, mother.genes['gestation_efficiency'])
            mother.stored_child_genes = inherit_genes(mother, father)
            mother.energy -= REPRODUCTION_COST
            father.energy -= REPRODUCTION_COST
            mother.reproduction_cooldown = REPRODUCTION_COOLDOWN
            father.reproduction_cooldown = REPRODUCTION_COOLDOWN
            mother.children += 1
            father.children += 1
            self.last_action_mode = 'mate'
            other.last_action_mode = 'mate'
            return True
        return False

    def maybe_attack(self, agents, intensity):
        if self.is_sleeping or intensity < 0.55:
            return 0.0
        nearby = [a for a in agents if a is not self and a.alive and abs(a.pos[0] - self.pos[0]) <= 1 and abs(a.pos[1] - self.pos[1]) <= 1]
        if not nearby:
            return -0.05
        target = max(nearby, key=lambda a: self.trust.get(a.id, 0.0) * -1 + random.random() * 0.1)
        damage = 1.0 + 3.5 * self.genes['aggression']
        target.health -= damage
        target.trust[self.id] = max(-1.0, min(1.0, target.trust.get(self.id, 0.0) - 0.18))
        self.energy -= 0.5
        self.last_action_mode = 'attack'
        return 0.12 if target.health < 30 else -0.04

    def apply_disease(self, world):
        cell = world.get_cell(*self.pos)
        exposure = 0.004 * cell['disease'] + 0.002 * cell['pollution'] + 0.001 * cell['disturbance']
        if random.random() < max(0.0, exposure - 0.30):
            self.sick = min(100.0, self.sick + 2.0 + 0.02 * cell['disease'])
            if self.disease_id is None and self.sick > 15.0:
                self.disease_id = random.choice(list(REMEDY_REGISTRY.keys()))
        recovery = 0.15 * (self.health / 100.0) + 0.05 * (self.hydration / 100.0)
        self.sick = max(0.0, self.sick - recovery)
        if self.sick <= 2.0:
            self.disease_id = None
        if self.sick > 0:
            self.health -= 0.010 * self.sick
            self.energy -= 0.006 * self.sick
        warmth = world.get_cell(*self.pos).get('warmth', 0.0)
        if warmth > 0.2:
            self.sick = max(0.0, self.sick - 0.15 * warmth)

    def apply_environmental_effects(self, world):
        cell = world.get_cell(*self.pos)
        biome_cost = world.biome_move_cost(*self.pos)
        move_cost = (0.22 + (1.0 / max(0.6, self.genes['speed'])) * 0.10) * biome_cost
        move_cost *= (1.75 - min(1.5, self.genes['efficiency']))
        self.energy -= move_cost
        self.hydration -= 0.30 + 0.008 * cell['temperature'] + 0.010 * biome_cost + 0.012 * cell['disturbance']
        self.energy -= 0.003 * cell['danger'] + 0.003 * cell['disturbance']
        self.health -= max(0, abs(cell['temperature'] - 20) - 14) * 0.05
        self.health -= 0.005 * cell['pollution'] + 0.006 * cell['ash']
        warmth = cell.get('warmth', 0.0)
        if warmth > 0.3:
            cold_exposure = max(0, abs(cell['temperature'] - 20) - 14)
            self.health += cold_exposure * 0.03 * warmth
            self.hydration += 0.05 * warmth
        if cell['biome'] == 'desert':
            self.energy -= 0.18
            self.hydration -= 0.35
        if cell['biome'] == 'swamp':
            self.health -= 0.05
        if self.hydration <= 0:
            self.health -= 0.6
            self.energy -= 0.25
        if self.energy <= 0:
            self.health -= 0.6
        if self.age > AGE_LIMIT:
            self.energy = 0
            self.health -= 2.0
        # Sleep debt penalty: cognitive degradation
        if self.sleep_debt > 40.0:
            perf_penalty = (self.sleep_debt - 40.0) / 120.0
            self.health -= 0.02 * perf_penalty
            self.energy -= 0.04 * perf_penalty
        # Night danger: predator pressure increases at night
        danger_mult = world.day_state.get('danger_mult', 1.0)
        if danger_mult > 1.0:
            extra_danger = cell.get('danger', 0.0) * (danger_mult - 1.0) * 0.004
            self.health -= extra_danger
        if self.health <= 0:
            self.alive = False

    def update(self, world, agents, tick, season_state, weather_state, tribes, economy, technology):
        _ensure_new_fields(self)
        self.age += 1
        self.last_action_mode = 'idle'
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        child_genes = self.progress_pregnancy()

        # Sleep must happen before perception/action
        self.apply_sleep(world)

        if not self.is_sleeping:
            for xx, yy, cell, biome in self.visible_cells(world):
                if cell['food'] > 50 or cell['water'] > 55:
                    self.memory.remember_resource((xx, yy), cell['food'], cell['water'], tick)
                if cell['danger'] > 60 or cell['disease'] > 35:
                    self.memory.remember_danger((xx, yy), max(cell['danger'], cell['disease']), tick)

        features = self.local_features(world, agents)
        self.local_features_cache = features
        prev_energy = self.energy
        prev_hydration = self.hydration
        prev_health = self.health

        if self.is_sleeping:
            # Minimal brain pass — no action taken, just store idle transition
            brain_step = self.brain.act(features, self.hidden_state)
            self.apply_disease(world)
            self.apply_environmental_effects(world)
            reward = 0.05 if self.alive else -4.5
            d_energy = (self.energy - prev_energy) / 30.0
            d_health = (self.health - prev_health) / 20.0
            reward += 0.45 * d_energy + 0.42 * d_health
            self.brain.store_transition(
                obs_tensor=brain_step['obs_tensor'],
                hidden_in=brain_step['hidden_in'],
                action_tensor=brain_step['action_tensor'],
                log_prob=brain_step['log_prob'],
                value=brain_step['value'],
                reward=reward,
                done=not self.alive,
                next_obs=features,
            )
            loss = self.brain.maybe_train()
            if loss is not None:
                self.last_loss = loss
            self.hidden_state = self.brain.initial_hidden() if not self.alive else brain_step['next_hidden'] * self.genes['memory_retention']
            self.energy = max(0.0, min(MAX_ENERGY, self.energy))
            self.hydration = max(0.0, min(100.0, self.hydration))
            self.health = max(0.0, min(100.0, self.health))
            self.last_reward = reward
            return child_genes

        brain_step = self.brain.act(features, self.hidden_state)
        values = brain_step['action_list']
        action = {
            'move_x': values[0],
            'move_y': values[1],
            'eat': values[2],
            'explore': values[3],
            'communicate': values[4],
            'attack': values[5],
        }

        self.primitive_move(world, action)
        reward = 0.0
        reward += self.forage(world, max(0.0, (action['eat'] + 1.0) * 0.5))
        reward += self.collect_materials(world, max(0.0, (action['explore'] + 1.0) * 0.5))
        if action['explore'] > 0.2 and random.random() < 0.25 + 0.2 * self.genes.get('curiosity', 0.5):
            cell = world.get_cell(*self.pos)
            env = {
                'wind': cell.get('disturbance', 0) / 100.0,
                'moisture': cell.get('moisture', 50) / 100.0,
                'temperature': cell.get('temperature', 20),
            }
            reward += agent_try_invention(self, cell, env)
            self.last_action_mode = 'experiment'
        built = self.maybe_build(world, action['explore'])
        if built is not None:
            reward += 1.2
        self.maybe_signal(features, values)
        reward += self.update_social(agents, tribes, tick, action, features)
        economy.maybe_trade(self, agents)
        if self.maybe_reproduce(agents, action):
            reward += 2.2 * self.genes['fertility']
        reward += self.maybe_attack(agents, action['attack'])
        self.apply_disease(world)
        self.apply_environmental_effects(world)

        d_energy = (self.energy - prev_energy) / 30.0
        d_hydration = (self.hydration - prev_hydration) / 30.0
        d_health = (self.health - prev_health) / 20.0
        reward += 0.45 * d_energy + 0.38 * d_hydration + 0.42 * d_health
        reward += 0.05 if self.alive else -4.5
        if child_genes is not None:
            reward += 2.5
        if self.pregnant:
            reward -= 0.01
        if self.sick > 60:
            self.last_action_mode = 'sick'
        # Sleep debt degrades reward signal (cognitive impairment)
        if self.sleep_debt > 40.0:
            reward *= max(0.5, 1.0 - (self.sleep_debt - 40.0) / 120.0)

        next_features = self.local_features(world, agents)
        intrinsic = self.brain.intrinsic_reward(brain_step['hidden_in'], brain_step['action_tensor'], next_features)
        reward += 0.12 * intrinsic
        self.brain.store_transition(
            obs_tensor=brain_step['obs_tensor'],
            hidden_in=brain_step['hidden_in'],
            action_tensor=brain_step['action_tensor'],
            log_prob=brain_step['log_prob'],
            value=brain_step['value'],
            reward=reward,
            done=not self.alive,
            next_obs=next_features,
        )
        loss = self.brain.maybe_train()
        if loss is not None:
            self.last_loss = loss
        self.hidden_state = self.brain.initial_hidden() if not self.alive else brain_step['next_hidden'] * self.genes['memory_retention']
        self.energy = max(0.0, min(MAX_ENERGY, self.energy))
        self.hydration = max(0.0, min(100.0, self.hydration))
        self.health = max(0.0, min(100.0, self.health))
        self.last_reward = reward
        self.learning_score += reward * 0.02 * self.genes['plasticity']
        return child_genes
