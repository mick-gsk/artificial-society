import random
from dataclasses import dataclass, field

from artificial_society.agents.brain import Brain, INPUT_SIZE
from artificial_society.agents.memory import EpisodicMemory
from artificial_society.agents.genetics import random_genes, inherit_genes
from artificial_society.agents.communication import CommunicationSystem
from artificial_society.agents.endocrine import EndocrineSystem
from artificial_society.environment.resources import apply_consumption, maybe_build_structure
from artificial_society.environment.herbs import available_herbs, collect_herb, regrow_herbs
from artificial_society.environment.territory import territory_reward_for_agent, get_home_forage_bonus
from artificial_society.environment.structures import apply_structure_effects, structure_feature_vector, BUILD_ENERGY_COST
from artificial_society.systems.remedy import (
    evaluate_remedy,
    record_cure_discovery,
    share_remedy_knowledge,
    try_infect_agent,
    REMEDY_REGISTRY,
)
from artificial_society.systems.culture import CausalMemory
from artificial_society.systems.invention import agent_try_invention, agent_try_cook
from artificial_society.systems.social_learning import social_learning_step

MAX_ENERGY = 240.0
INITIAL_ENERGY = 120.0
CHILD_START_ENERGY = 100.0
REPRODUCTION_ENERGY = 100.0
REPRODUCTION_COST = 28.0
REPRODUCTION_COOLDOWN = 120
MIN_REPRODUCTION_AGE = 80
GESTATION_TIME = 80
AGE_LIMIT = 800
ELDER_AGE = 600
AGE_HEALTH_DECAY_START = 600
AGE_HEALTH_DECAY_HARD  = 750
PLANT_ENERGY = 28.0
MEAT_ENERGY = 40.0
CORPSE_ENERGY = 36.0

SHARP_STONE_FORAGE_BONUS   = 0.30
SHARP_STONE_COLLECT_BONUS  = 0.20

SLEEP_DRIVE_THRESHOLD = 0.45
SLEEP_ENERGY_REGEN    = 0.40
SLEEP_HEALTH_REGEN    = 0.12

STAGE_CHILD = MIN_REPRODUCTION_AGE
STAGE_ELDER = ELDER_AGE

COOP_FORAGE_BONUS_PER_MEMBER = 0.14
COOP_FORAGE_MAX_BONUS        = 0.65
COOP_DEFENSE_HEALTH_BONUS    = 0.08
COOP_SHARE_THRESHOLD_DONOR   = 160.0
COOP_SHARE_THRESHOLD_RECV    = 60.0
COOP_SHARE_AMOUNT            = 18.0
COOP_SHARE_REWARD            = 0.25
COOP_RECV_REWARD             = 0.30
COOP_PROXIMITY_RADIUS        = 2

INVENTION_BASE_PROB      = 0.55
INVENTION_CURIOSITY_MULT = 0.35


def _ensure_new_fields(agent):
    if not hasattr(agent, 'causal_memory') or agent.causal_memory is None:
        agent.causal_memory = CausalMemory(capacity=32)
    if not hasattr(agent, 'material_inventory') or agent.material_inventory is None:
        agent.material_inventory = {}
    if not hasattr(agent, 'endocrine') or agent.endocrine is None:
        agent.endocrine = EndocrineSystem()
    if not hasattr(agent, 'is_sleeping'):
        agent.is_sleeping = False
    if not hasattr(agent, 'brain') or agent.brain is None:
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    elif agent.brain.input_size != INPUT_SIZE:
        print(f'[compat] Agent {agent.id}: brain input_size={agent.brain.input_size} != {INPUT_SIZE}, rebuilding.')
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    if not hasattr(agent, 'tool'):
        agent.tool = None
    if agent.tool is None and getattr(agent, 'material_inventory', {}).get('sharp_stone', 0) > 0.1:
        agent.tool = 'sharp_stone'
    if not hasattr(agent, '_last_mate_id'):
        agent._last_mate_id = None


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
    endocrine: EndocrineSystem = field(default_factory=EndocrineSystem)
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
    is_sleeping: bool = False
    _last_mate_id: int | None = None   # NEU: speichert ID des Partners fuer Geburts-Kontext

    @classmethod
    def spawn_random(cls, x, y):
        cls.id_counter += 1
        genes  = random_genes()
        brain  = Brain(plasticity=genes.get('plasticity', 1.0))
        agent  = cls(
            id=cls.id_counter, pos=(x, y), genes=genes,
            memory=EpisodicMemory(genes['memory_capacity']),
            brain=brain, sex=random.choice(['m', 'f']),
        )
        agent.hidden_state = brain.initial_hidden()
        return agent

    @classmethod
    def spawn_child(cls, x, y, genes, generation=1, parent_id=None, tribe_id=None):
        cls.id_counter += 1
        brain  = Brain(plasticity=genes.get('plasticity', 1.0))
        agent  = cls(
            id=cls.id_counter, pos=(x, y), genes=genes,
            memory=EpisodicMemory(genes['memory_capacity']),
            brain=brain, energy=CHILD_START_ENERGY,
            sex=random.choice(['m', 'f']),
            generation=generation, parent_id=parent_id, tribe_id=tribe_id,
        )
        agent.hidden_state = brain.initial_hidden()
        agent.reproduction_cooldown = REPRODUCTION_COOLDOWN
        return agent

    def life_stage(self):
        if self.age < STAGE_CHILD:  return 'child'
        if self.age >= STAGE_ELDER: return 'elder'
        return 'adult'

    def display_color(self):
        meat_bias  = max(0.0, min(1.0, (self.genes['diet_preference'] + 1.0) * 0.5))
        plant_bias = max(0.0, min(1.0, (-self.genes['diet_preference'] + 1.0) * 0.5))
        g = max(70, min(255, int(80 + 120*self.genes['cooperation'] + 35*self.genes['memory_retention'])))
        r = max(70, min(255, int(80 + 100*meat_bias + 60*self.genes['aggression'])))
        b = max(70, min(255, int(80 + 100*plant_bias + 50*self.genes['plasticity']/1.8)))
        return (r, g, b)

    def can_reproduce(self):
        return (
            self.alive and self.age >= MIN_REPRODUCTION_AGE
            and self.age < ELDER_AGE
            and self.energy >= REPRODUCTION_ENERGY
            and self.reproduction_cooldown <= 0
            and not self.pregnant
        )

    def local_features(self, world, agents):
        _ensure_new_fields(self)
        x, y  = self.pos
        cell  = world.get_cell(x, y)
        near  = [a for a in agents if a is not self
                 and abs(a.pos[0]-x) <= int(self.genes['sense_radius'])
                 and abs(a.pos[1]-y) <= int(self.genes['sense_radius'])
                 and a.alive]
        friends    = sum(1 for a in near if self.trust.get(a.id, 0.0) > 0.2)
        retrieval  = self.memory.retrieval_features(x, y, cell['tick'], world.width, world.height)
        avg_trust  = sum(self.trust.values()) / len(self.trust) if self.trust else 0.0
        herb_pres  = min(1.0, len(available_herbs(cell)) / 5.0)
        warmth     = min(1.0, cell.get('warmth', 0.0))
        mat_count  = min(1.0, len(cell.get('materials', {})) / 6.0)
        causal_f   = self.causal_memory.feature_vector()
        inv_size   = min(1.0, sum(self.material_inventory.values()) / 5.0)
        hormones   = self.endocrine.as_features()
        struct_f   = structure_feature_vector(cell)
        return [
            self.energy / MAX_ENERGY,
            self.health / 100.0,
            self.hydration / 100.0,
            min(1.0, self.age / AGE_LIMIT),
            cell['food']  / 180.0,
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
            min(1.0, self.resources['wood']  / 4.0),
            min(1.0, self.resources['stone'] / 4.0),
            min(1.0, self.resources['fiber'] / 4.0),
            max(-1.0, min(1.0, self.last_reward / 10.0)),
            herb_pres,
            warmth,
            mat_count,
            inv_size,
            *struct_f,
            *causal_f,
            *retrieval,
            *hormones,
        ]

    def visible_cells(self, world):
        x, y = self.pos
        melatonin   = self.endocrine.h[2]
        vision_mult = max(0.4, 1.0 - 0.6 * melatonin)
        radius = max(1, int(round((self.genes['vision'] + 0.35*self.genes['sense_radius']) * vision_mult)))
        return world.neighbors(x, y, radius)

    def progress_pregnancy(self):
        child = None
        if self.pregnant:
            self.gestation -= 1
            self.energy    -= 0.03
            if self.gestation <= 0 and self.stored_child_genes is not None:
                child = self.stored_child_genes
                self.pregnant = False
                self.gestation = 0
                self.stored_child_genes = None
        return child

    def primitive_move(self, world, action):
        if self.is_sleeping:
            return
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
        _ensure_new_fields(self)
        mods        = self.endocrine.modifiers()
        sleep_drive = mods['sleep_drive']
        cell        = world.get_cell(*self.pos)
        danger_here = cell.get('danger', 0.0) + cell.get('disturbance', 0.0)
        safe_enough = danger_here < 25.0
        if not self.is_sleeping and sleep_drive > SLEEP_DRIVE_THRESHOLD and safe_enough:
            warmth  = cell.get('warmth', 0.0)
            shelter = cell.get('structures', {}).get('camp', 0.0)
            sleep_chance = 0.20 + 0.30*(warmth > 0.3) + 0.25*min(1.0, shelter)
            if random.random() < sleep_chance:
                self.is_sleeping   = True
                self.last_action_mode = 'sleep'
        if self.is_sleeping:
            self.energy   = min(MAX_ENERGY, self.energy + SLEEP_ENERGY_REGEN)
            self.health   = min(100.0,      self.health + SLEEP_HEALTH_REGEN)
            self.endocrine.h[2] = max(0.0, self.endocrine.h[2] - 0.05)
            self.last_action_mode = 'sleep'
            if danger_here > 40.0 or self.endocrine.h[2] < 0.12 or self.endocrine.h[1] > 0.55:
                self.is_sleeping = False

    def forage(self, world, intensity):
        _ensure_new_fields(self)
        if self.is_sleeping:
            return 0.0
        cell       = world.get_cell(*self.pos)
        mods       = self.endocrine.modifiers()
        forage_eff = mods['forage_eff']
        plant_need = max(0.2, 1.0 - self.energy / MAX_ENERGY)
        meat_bias  = max(0.0, self.genes['diet_preference'])
        plant_bias = max(0.0, -self.genes['diet_preference'])
        plant_take = min(cell['plant_food'], 2.0 + 8.0*intensity*(0.5 + plant_need + 0.4*plant_bias))
        meat_take  = min(cell['meat_food'],  1.0 + 5.5*intensity*(0.35 + plant_need + 0.5*meat_bias))
        water_need = max(0.3, 1.0 - self.hydration / 100.0)
        water_take = min(cell['water'], 1.0 + 8.0*intensity*water_need)
        efficiency = (0.75 + 0.35*self.genes['efficiency']) * forage_eff
        tool_mult  = (1.0 + SHARP_STONE_FORAGE_BONUS) if self.tool == 'sharp_stone' else 1.0
        home_bonus = get_home_forage_bonus(self, world)
        struct_mods = apply_structure_effects(self, cell)
        efficiency *= (1.0 + home_bonus + struct_mods['forage_bonus'])
        cook_bonus_health = 0.0
        for mat in ('cooked_meat', 'cooked_root'):
            if self.material_inventory.get(mat, 0) > 0.1:
                val = min(self.material_inventory[mat], 1.0)
                self.material_inventory[mat] -= val
                self.energy = min(MAX_ENERGY, self.energy + val * 22.0)
                cook_bonus_health += val * 0.5
                self.endocrine.apply_substance(mat, val)
        plant_gain = plant_take * (PLANT_ENERGY/8.0) * (1.0 + 0.55*plant_bias) * efficiency * tool_mult
        meat_gain  = meat_take  * (MEAT_ENERGY /6.0) * (1.0 + 0.55*meat_bias)  * efficiency * tool_mult
        apply_consumption(cell, plant=plant_take, meat=meat_take, water=water_take)
        self.energy    = min(MAX_ENERGY, self.energy + plant_gain + meat_gain)
        self.hydration = min(100.0, self.hydration + water_take * 8.5 + struct_mods['hydration_bonus'])
        self.health    = min(100.0, self.health + water_take * 0.16 + cook_bonus_health)
        if plant_take > 0.5:
            self.plant_eaten += 1
            self.endocrine.apply_substance('plant_food', plant_take * 0.3)
        if meat_take > 0.3:
            self.meat_eaten += 1
            self.endocrine.apply_substance('raw_meat', meat_take * 0.3)
        if water_take > 0.3:
            self.endocrine.apply_substance('water', water_take * 0.2)
        if plant_take + meat_take + water_take > 0.8:
            self.last_action_mode = 'gather'
        total_gain = plant_gain + meat_gain
        if total_gain > 0.5:
            self.endocrine.apply_successful_forage(total_gain)
        # Jahreszeit aus world.day_state oder seasons -- fuer Gedaechtnis-Tag
        season_id = getattr(world, 'current_season', None)
        self.memory.remember_resource(self.pos, cell['food'], cell['water'], cell['tick'], season_id=season_id)
        consumed_tags: list[str] = []
        if plant_take > 0.3:  consumed_tags.append('plant_food')
        if meat_take  > 0.3:  consumed_tags.append('meat')
        if water_take > 0.3:  consumed_tags.append('water')
        return self._try_use_herbs(cell, consumed_tags)

    def _try_use_herbs(self, cell, consumed_tags):
        herbs_here = available_herbs(cell)
        if not herbs_here:
            return 0.0
        reward       = 0.0
        curiosity    = self.genes.get('curiosity', 0.5)
        dopamine     = self.endocrine.h[4]
        inflammation = self.endocrine.h[6]
        sample_prob  = 0.10 + 0.25*curiosity + 0.20*dopamine + 0.30*inflammation
        for tag in herbs_here:
            if random.random() < sample_prob:
                taken = collect_herb(cell, tag, amount=1.0)
                if taken > 0:
                    self.herbs_carried[tag] = self.herbs_carried.get(tag, 0.0) + taken
                    consumed_tags.append(tag)
                    self.last_action_mode = 'forage_herb'
                    self.endocrine.apply_substance(tag, taken)
        if self.disease_id and consumed_tags:
            prev_disease = self.disease_id
            cure_bonus   = evaluate_remedy(self, consumed_tags)
            if cure_bonus > 0:
                record_cure_discovery(self, prev_disease, consumed_tags)
                reward += cure_bonus * 2.5
        return reward

    def collect_materials(self, world, intensity):
        _ensure_new_fields(self)
        if self.is_sleeping:
            return
        cell       = world.get_cell(*self.pos)
        tool_bonus = SHARP_STONE_COLLECT_BONUS if self.tool == 'sharp_stone' else 0.0
        if cell['biome'] == 'forest'    and random.random() < 0.08 + 0.16*intensity + tool_bonus:
            self.resources['wood']  += 1
        if cell['biome'] == 'mountain'  and random.random() < 0.06 + 0.14*intensity + tool_bonus:
            self.resources['stone'] += 1
        if cell['biome'] in ('grassland','swamp') and random.random() < 0.08 + 0.18*intensity + tool_bonus:
            self.resources['fiber'] += 1
        slot = cell.get('materials', {})
        for mat, qty in list(slot.items()):
            if qty > 0.1 and random.random() < 0.15 * intensity:
                take = min(qty, 0.5)
                slot[mat] -= take
                self.material_inventory[mat] = self.material_inventory.get(mat, 0.0) + take
                if mat == 'sharp_stone' and self.tool is None:
                    self.tool = 'sharp_stone'

    def maybe_craft(self, technology, intensity):
        if self.is_sleeping:
            return 0.0
        return 0.0

    def maybe_build(self, world, intensity):
        if self.is_sleeping:
            return None
        cell  = world.get_cell(*self.pos)
        built = maybe_build_structure(cell, self.resources)
        if built is not None:
            cost = BUILD_ENERGY_COST.get(built, 8.0)
            self.energy = max(0.0, self.energy - cost)
            self.last_action_mode = f'build:{built}'
        return built

    def maybe_signal(self, features, action_values):
        if self.is_sleeping:
            return
        self.communication.emit(self, features, action_values)
        if abs(sum(self.message_vector)) > 0.35:
            self.last_action_mode = 'signal'

    def cooperative_forage_bonus(self, agents) -> float:
        if self.tribe_id is None:
            return 1.0
        close_tribe = sum(
            1 for a in agents
            if a is not self and a.alive
            and a.tribe_id == self.tribe_id
            and abs(a.pos[0]-self.pos[0]) <= COOP_PROXIMITY_RADIUS
            and abs(a.pos[1]-self.pos[1]) <= COOP_PROXIMITY_RADIUS
        )
        return 1.0 + min(COOP_FORAGE_MAX_BONUS, close_tribe * COOP_FORAGE_BONUS_PER_MEMBER)

    def cooperative_defense_bonus(self, agents) -> float:
        if self.tribe_id is None:
            return 0.0
        close_tribe = sum(
            1 for a in agents
            if a is not self and a.alive
            and a.tribe_id == self.tribe_id
            and abs(a.pos[0]-self.pos[0]) <= COOP_PROXIMITY_RADIUS
            and abs(a.pos[1]-self.pos[1]) <= COOP_PROXIMITY_RADIUS
        )
        return min(0.40, close_tribe * COOP_DEFENSE_HEALTH_BONUS)

    def maybe_share_food(self, agents) -> float:
        if self.tribe_id is None or self.energy < COOP_SHARE_THRESHOLD_DONOR:
            return 0.0
        reward = 0.0
        nearby_tribe = [
            a for a in agents
            if a is not self and a.alive
            and a.tribe_id == self.tribe_id
            and abs(a.pos[0]-self.pos[0]) <= COOP_PROXIMITY_RADIUS
            and abs(a.pos[1]-self.pos[1]) <= COOP_PROXIMITY_RADIUS
            and a.energy < COOP_SHARE_THRESHOLD_RECV
        ]
        for recipient in nearby_tribe[:2]:
            transfer = min(COOP_SHARE_AMOUNT, self.energy - COOP_SHARE_THRESHOLD_RECV)
            if transfer < 5.0:
                break
            self.energy     -= transfer
            recipient.energy = min(MAX_ENERGY, recipient.energy + transfer)
            reward          += COOP_SHARE_REWARD
            if hasattr(recipient.brain, 'rollout') and recipient.brain.rollout.storage:
                recipient.brain.rollout.storage[-1]['reward'] += COOP_RECV_REWARD
            self.last_action_mode       = 'share'
            self.endocrine.h[5]         = min(1.0, self.endocrine.h[5]      + 0.12)
            recipient.endocrine.h[5]    = min(1.0, recipient.endocrine.h[5] + 0.18)
        return reward

    def update_social(self, agents, tribes, tick, action, features_before):
        _ensure_new_fields(self)
        if self.is_sleeping:
            return 0.0
        nearby = [a for a in agents if a is not self and a.alive
                  and abs(a.pos[0]-self.pos[0]) <= 1
                  and abs(a.pos[1]-self.pos[1]) <= 1]
        same_tribe_nearby = sum(1 for a in nearby if a.tribe_id == self.tribe_id and self.tribe_id is not None)
        self.endocrine.apply_social_signal(len(nearby), same_tribe_nearby > 0)
        mods = self.endocrine.modifiers()
        for other in nearby:
            helpful = other.message_vector[0] > -0.15
            prior   = self.trust.get(other.id, 0.0)
            delta   = 0.015 if helpful else -0.008
            delta  += 0.018*self.genes['sociality'] + 0.010*mods['social_bias']
            if action['communicate'] > 0.1:
                before_food   = features_before[4]
                before_danger = features_before[7]
                after_food    = other.local_features_cache[4]  if hasattr(other, 'local_features_cache') else before_food
                after_danger  = other.local_features_cache[7]  if hasattr(other, 'local_features_cache') else before_danger
                info_bonus = self.communication.evaluate_message_usefulness(other, self, before_food, after_food, before_danger, after_danger)
                delta += info_bonus * 0.18
                if self.tribe_id is not None and other.tribe_id == self.tribe_id:
                    self._share_tribe_resources(other)
                if self.trust.get(other.id, 0.0) > 0.3:
                    share_remedy_knowledge(self, other)
            self.trust[other.id] = max(-1.0, min(1.0, prior + delta))
            self.memory.remember_social(other.id, self.trust[other.id], helpful, tick)
        tribes.consider_join(self, nearby)
        social_learning_step(self, agents, tick)
        return 0.0

    def _share_tribe_resources(self, other):
        for res in ('wood', 'stone', 'fiber'):
            if self.resources[res] >= 3 and other.resources[res] == 0:
                self.resources[res] -= 1
                other.resources[res] += 1
                break
        if self.energy > 180.0 and other.energy < 80.0:
            transfer = min(20.0, self.energy - 160.0)
            self.energy  -= transfer
            other.energy += transfer

    def maybe_reproduce(self, agents, action):
        if self.is_sleeping or not self.can_reproduce():
            return False
        nearby = [a for a in agents if a is not self and a.alive
                  and abs(a.pos[0]-self.pos[0]) <= 2
                  and abs(a.pos[1]-self.pos[1]) <= 2]
        random.shuffle(nearby)
        for other in nearby:
            if not other.can_reproduce() or other.sex == self.sex or other.pregnant:
                continue
            mother = self if self.sex == 'f' else other
            father = other if mother is self else self
            mother.pregnant           = True
            mother.gestation          = GESTATION_TIME / max(0.5, mother.genes['gestation_efficiency'])
            mother.stored_child_genes = inherit_genes(mother, father)
            mother.energy            -= REPRODUCTION_COST
            father.energy            -= REPRODUCTION_COST
            mother.reproduction_cooldown = REPRODUCTION_COOLDOWN
            father.reproduction_cooldown = REPRODUCTION_COOLDOWN
            mother.children += 1
            father.children += 1
            # NEU: Partner-ID fuer Geburtskontext speichern
            mother._last_mate_id = father.id
            father._last_mate_id = mother.id
            mother.endocrine.h[5] = min(1.0, mother.endocrine.h[5] + 0.25)
            father.endocrine.h[5] = min(1.0, father.endocrine.h[5] + 0.15)
            self.last_action_mode  = 'mate'
            other.last_action_mode = 'mate'
            return True
        return False

    def maybe_attack(self, agents, intensity):
        if self.is_sleeping:
            return
        mods = self.endocrine.modifiers()
        if intensity < 0.55 - 0.15*mods['aggression_bias']:
            return
        nearby = [a for a in agents if a is not self and a.alive
                  and abs(a.pos[0]-self.pos[0]) <= 1
                  and abs(a.pos[1]-self.pos[1]) <= 1]
        if not nearby:
            return
        target = max(nearby, key=lambda a: self.trust.get(a.id, 0.0)*-1 + random.random()*0.1)
        damage = 1.0 + 3.5*self.genes['aggression']
        target.health -= damage
        target.trust[self.id] = max(-1.0, min(1.0, target.trust.get(self.id, 0.0) - 0.18))
        target.endocrine.apply_attack_received()
        self.energy -= 0.5
        self.last_action_mode = 'attack'

    def apply_disease(self, world):
        cell     = world.get_cell(*self.pos)
        struct_mods = apply_structure_effects(self, cell)
        disease_factor = struct_mods['disease_factor']
        exposure = (0.004*cell['disease'] + 0.002*cell['pollution'] + 0.001*cell['disturbance']) * disease_factor
        if random.random() < max(0.0, exposure - 0.30):
            self.sick = min(100.0, self.sick + 2.0 + 0.02*cell['disease'])
            if self.disease_id is None and self.sick > 15.0:
                self.disease_id = random.choice(list(REMEDY_REGISTRY.keys()))
        inflammation = self.endocrine.h[6]
        recovery = 0.15*(self.health/100.0) + 0.05*(self.hydration/100.0)
        recovery *= max(0.3, 1.0 - 0.5*inflammation)
        self.sick = max(0.0, self.sick - recovery)
        if self.sick <= 2.0:
            self.disease_id = None
        if self.sick > 0:
            self.health -= 0.010*self.sick
            self.energy -= 0.006*self.sick
        if cell.get('warmth', 0.0) > 0.2:
            self.sick = max(0.0, self.sick - 0.15*cell['warmth'])

    def apply_environmental_effects(self, world):
        _ensure_new_fields(self)
        cell       = world.get_cell(*self.pos)
        biome_cost = world.biome_move_cost(*self.pos)
        mods       = self.endocrine.modifiers()
        struct_mods = apply_structure_effects(self, cell)
        move_cost  = (0.22 + (1.0/max(0.6, self.genes['speed']))*0.10) * biome_cost
        move_cost *= (1.75 - min(1.5, self.genes['efficiency']))
        move_cost *= mods['move_cost_mult']
        self.energy    -= move_cost
        self.hydration -= 0.30 + 0.008*cell['temperature'] + 0.010*biome_cost + 0.012*cell['disturbance']
        self.energy    -= 0.003*cell['danger'] + 0.003*cell['disturbance']
        cold_dmg = max(0, abs(cell['temperature']-20) - 14) * 0.05
        self.health    -= cold_dmg * struct_mods['cold_factor']
        self.health    -= 0.005*cell['pollution'] + 0.006*cell['ash']
        self.health    -= mods['health_drain']
        if self.age >= AGE_HEALTH_DECAY_START:
            decay = 0.3 + max(0.0, (self.age-AGE_HEALTH_DECAY_HARD)*1.5/(AGE_LIMIT-AGE_HEALTH_DECAY_HARD))
            self.health -= decay
        warmth = cell.get('warmth', 0.0)
        if warmth > 0.3:
            cold_exp = max(0, abs(cell['temperature']-20) - 14)
            self.health    += cold_exp * 0.03 * warmth
            self.hydration += 0.05 * warmth
        if cell['biome'] == 'desert':
            self.energy    -= 0.18
            self.hydration -= 0.35
        if cell['biome'] == 'swamp':
            self.health -= 0.05
        if self.hydration <= 0:
            self.health -= 0.6
            self.energy -= 0.25
        if self.energy <= 0:
            self.health -= 0.6
        if self.age > AGE_LIMIT:
            self.energy  = 0
            self.health -= 5.0
        if self.endocrine.h[1] > 0.5:
            self.energy -= 0.02 * (self.endocrine.h[1] - 0.5)
        if self.health <= 0:
            self.alive = False

    def update(self, world, agents, tick, season_state, weather_state, tribes, economy, technology):
        _ensure_new_fields(self)
        self.age += 1
        self.last_action_mode = 'idle'
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        child_genes = self.progress_pregnancy()

        self.endocrine.update(self, world)
        self.apply_sleep(world)

        if not self.is_sleeping:
            season_id = getattr(world, 'current_season', None)
            for xx, yy, cell, biome in self.visible_cells(world):
                if cell['food'] > 30 or cell['water'] > 40:
                    self.memory.remember_resource((xx, yy), cell['food'], cell['water'], tick, season_id=season_id)
                if cell['danger'] > 50 or cell['disease'] > 30:
                    self.memory.remember_danger((xx, yy), max(cell['danger'], cell['disease']), tick, season_id=season_id)

        features = self.local_features(world, agents)
        self.local_features_cache = features

        prev_energy    = self.energy
        prev_hydration = self.hydration
        prev_health    = self.health

        if self.is_sleeping:
            brain_step = self.brain.act(features, self.hidden_state)
            self.apply_disease(world)
            self.apply_environmental_effects(world)
            reward = _homeostasis_reward(self.energy, prev_energy, self.hydration, prev_hydration, self.health, prev_health, self.alive)
            self.brain.store_transition(
                obs_tensor=brain_step['obs_tensor'], hidden_in=brain_step['hidden_in'],
                action_tensor=brain_step['action_tensor'], log_prob=brain_step['log_prob'],
                value=brain_step['value'], reward=reward, done=not self.alive, next_obs=features,
            )
            loss = self.brain.maybe_train()
            if loss is not None: self.last_loss = loss
            self.hidden_state = self.brain.initial_hidden() if not self.alive else brain_step['next_hidden'] * self.genes['memory_retention']
            self.energy    = max(0.0, min(MAX_ENERGY, self.energy))
            self.hydration = max(0.0, min(100.0, self.hydration))
            self.health    = max(0.0, min(100.0, self.health))
            self.last_reward = reward
            return child_genes

        brain_step = self.brain.act(features, self.hidden_state)
        values = brain_step['action_list']
        action = {
            'move_x':      values[0],
            'move_y':      values[1],
            'eat':         values[2],
            'explore':     values[3],
            'communicate': values[4],
            'attack':      values[5],
        }

        self.primitive_move(world, action)

        coop_mult        = self.cooperative_forage_bonus(agents)
        herb_heal_reward = self.forage(world, max(0.0, (action['eat'] + 1.0) * 0.5) * coop_mult)
        self.collect_materials(world, max(0.0, (action['explore'] + 1.0) * 0.5))

        cell = world.get_cell(*self.pos)
        if random.random() < 0.35 + 0.30 * self.genes.get('curiosity', 0.5):
            agent_try_cook(self, cell)

        share_reward  = self.maybe_share_food(agents)

        defense_bonus = self.cooperative_defense_bonus(agents)
        if defense_bonus > 0:
            self.health = min(100.0, self.health + defense_bonus)

        if action['explore'] > 0.1 and random.random() < INVENTION_BASE_PROB + INVENTION_CURIOSITY_MULT * self.genes.get('curiosity', 0.5):
            env = {
                'wind':        cell.get('disturbance', 0) / 100.0,
                'moisture':    cell.get('moisture', 50)   / 100.0,
                'temperature': cell.get('temperature', 20),
            }
            inv_reward = agent_try_invention(self, cell, env)
            if inv_reward > 0:
                self.endocrine.apply_discovery(min(1.0, inv_reward))
            self.last_action_mode = 'experiment'

        self.maybe_build(world, action['explore'])
        self.maybe_signal(features, values)
        self.update_social(agents, tribes, tick, action, features)
        economy.maybe_trade(self, agents)
        self.maybe_reproduce(agents, action)
        self.maybe_attack(agents, action['attack'])
        self.apply_disease(world)
        self.apply_environmental_effects(world)

        if self.sick > 60:
            self.last_action_mode = 'sick'

        reward  = _homeostasis_reward(self.energy, prev_energy, self.hydration, prev_hydration, self.health, prev_health, self.alive)
        reward += herb_heal_reward + share_reward

        # position_reward mit world und aktueller Jahreszeit
        current_season = getattr(world, 'current_season', None)
        mem_reward  = self.memory.position_reward(self.pos, tick, world=world, current_season=current_season)
        reward     += mem_reward

        terr_reward = territory_reward_for_agent(self, world)
        reward     += terr_reward

        next_features = self.local_features(world, agents)
        intrinsic = self.brain.intrinsic_reward(brain_step['hidden_in'], brain_step['action_tensor'], next_features)
        reward   += 0.10 * intrinsic

        mods    = self.endocrine.modifiers()
        reward *= mods['cognition']

        self.brain.store_transition(
            obs_tensor=brain_step['obs_tensor'],   hidden_in=brain_step['hidden_in'],
            action_tensor=brain_step['action_tensor'], log_prob=brain_step['log_prob'],
            value=brain_step['value'], reward=reward, done=not self.alive, next_obs=next_features,
        )
        loss = self.brain.maybe_train()
        if loss is not None: self.last_loss = loss
        self.hidden_state = self.brain.initial_hidden() if not self.alive else brain_step['next_hidden'] * self.genes['memory_retention']
        self.energy    = max(0.0, min(MAX_ENERGY, self.energy))
        self.hydration = max(0.0, min(100.0, self.hydration))
        self.health    = max(0.0, min(100.0, self.health))
        self.last_reward = reward
        self.learning_score += reward * 0.02 * self.genes['plasticity']
        return child_genes


def _homeostasis_reward(energy, prev_energy, hydration, prev_hydration, health, prev_health, alive):
    d_energy    = (energy    - prev_energy)    / MAX_ENERGY * 3.0
    d_hydration = (hydration - prev_hydration) / 100.0      * 2.5
    d_health    = (health    - prev_health)    / 100.0      * 4.0
    alive_bonus = 0.04 if alive else -5.0
    return d_energy + d_hydration + d_health + alive_bonus
