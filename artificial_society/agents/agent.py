import random
from dataclasses import dataclass, field

from artificial_society.agents.brain import Brain, INPUT_SIZE
from artificial_society.agents.memory import EpisodicMemory
from artificial_society.agents.genetics import random_genes, inherit_genes
from artificial_society.agents.communication import CommunicationSystem
from artificial_society.agents.endocrine import EndocrineSystem
from artificial_society.agents.theory_of_mind import TheoryOfMind
from artificial_society.agents.knowledge import KnowledgeGraph
from artificial_society.agents.emotional_memory import EmotionalMemory
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
from artificial_society.systems.need_driven_invention import agent_invent_from_need, compute_need_vector
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
NEED_INVENTION_INTERVAL  = 8


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
    if not hasattr(agent, '_need_inv_cooldown'):
        agent._need_inv_cooldown = 0
    if not hasattr(agent, 'tom') or agent.tom is None:
        agent.tom = TheoryOfMind(agent.id)
    if not hasattr(agent, 'knowledge') or agent.knowledge is None:
        agent.knowledge = KnowledgeGraph()
    if not hasattr(agent, 'emotional_memory') or agent.emotional_memory is None:
        agent.emotional_memory = EmotionalMemory()


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
    _last_mate_id: int | None = None
    _need_inv_cooldown: int = 0
    tom: TheoryOfMind = field(default_factory=lambda: TheoryOfMind(0))
    knowledge: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    emotional_memory: EmotionalMemory = field(default_factory=EmotionalMemory)

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
        agent.tom = TheoryOfMind(agent.id)
        agent.knowledge = KnowledgeGraph()
        agent.emotional_memory = EmotionalMemory()
        return agent

    @classmethod
    def spawn_child(cls, x, y, genes, generation=1, parent_id=None, tribe_id=None, parent=None):
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
        agent.tom = TheoryOfMind(agent.id)
        agent.knowledge = KnowledgeGraph()
        agent.emotional_memory = EmotionalMemory()
        if parent is not None:
            agent.tom.inherit_from(parent.tom, strength=0.4)
            agent.knowledge.inherit_from(parent.knowledge, strength=0.7)
            agent.emotional_memory.inherit_from(parent.emotional_memory, strength_factor=0.30)
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
        x, y = self.pos
        nx = max(0, min(world.width  - 1, x + dx))
        ny = max(0, min(world.height - 1, y + dy))
        cell = world.get_cell(nx, ny)
        if cell.get('passable', True):
            self.pos = (nx, ny)

    def _forage(self, world, mods):
        x, y  = self.pos
        cell  = world.get_cell(x, y)
        gain  = 0.0
        tool_bonus = SHARP_STONE_FORAGE_BONUS if self.tool == 'sharp_stone' else 0.0
        home_bonus = get_home_forage_bonus(world, self)
        eff   = mods.get('forage_eff', 1.0) * (1.0 + tool_bonus + home_bonus)
        # food
        food_available = cell.get('food', 0.0)
        if food_available > 0:
            diet = self.genes.get('diet_preference', 0.0)  # -1=plant, +1=meat
            if diet < 0:  # plant preference
                take = min(food_available, PLANT_ENERGY * eff)
                cell['food'] = max(0.0, food_available - take)
                self.energy  = min(MAX_ENERGY, self.energy + take)
                self.plant_eaten += 1
                gain += take
                self.endocrine.apply_substance('plant_food', take / PLANT_ENERGY)
                self.endocrine.apply_successful_forage(take)
            else:  # meat / omnivore
                carcass = cell.get('carcass', 0.0)
                if carcass > 0:
                    take = min(carcass, MEAT_ENERGY * eff)
                    cell['carcass'] = max(0.0, carcass - take)
                    self.energy = min(MAX_ENERGY, self.energy + take)
                    self.meat_eaten += 1
                    gain += take
                    self.endocrine.apply_substance('raw_meat', take / MEAT_ENERGY)
                    self.endocrine.apply_successful_forage(take)
                else:
                    take = min(food_available, PLANT_ENERGY * eff)
                    cell['food'] = max(0.0, food_available - take)
                    self.energy  = min(MAX_ENERGY, self.energy + take)
                    self.plant_eaten += 1
                    gain += take
                    self.endocrine.apply_substance('plant_food', take / PLANT_ENERGY)
                    self.endocrine.apply_successful_forage(take)
        # water
        water_available = cell.get('water', 0.0)
        if water_available > 0 and self.hydration < 100.0:
            take = min(water_available, 8.0 * eff)
            cell['water']   = max(0.0, water_available - take)
            self.hydration  = min(100.0, self.hydration + take)
            gain += take * 0.3
            self.endocrine.apply_substance('water', take / 8.0)
        return gain

    def _collect_herbs(self, world):
        x, y = self.pos
        cell = world.get_cell(x, y)
        herbs = available_herbs(cell)
        if not herbs:
            return
        herb = random.choice(herbs)
        if collect_herb(cell, herb):
            self.herbs_carried[herb] = self.herbs_carried.get(herb, 0) + 1
            self.endocrine.apply_substance(f'herb_{herb}', 1.0)

    def _try_remedy(self):
        if self.disease_id is None or not self.herbs_carried:
            return
        result = evaluate_remedy(self, self.disease_id)
        if result == 'cured':
            record_cure_discovery(self, self.disease_id)
            self.disease_id = None
            self.sick = max(0.0, self.sick - 40.0)
        elif result == 'partial':
            self.sick = max(0.0, self.sick - 15.0)

    def _share_remedy(self, agents):
        if not self.remedy_knowledge:
            return
        x, y = self.pos
        near = [
            a for a in agents
            if a is not self and a.alive
            and abs(a.pos[0]-x) <= 2 and abs(a.pos[1]-y) <= 2
        ]
        if near:
            share_remedy_knowledge(self, random.choice(near))

    def _attack(self, agents, mods):
        x, y = self.pos
        agg_bias = mods.get('aggression_bias', 0.0)
        threshold = max(0.05, self.genes['aggression'] + agg_bias - 0.3)
        if random.random() > threshold:
            return 0.0
        targets = [
            a for a in agents
            if a is not self and a.alive
            and abs(a.pos[0]-x) <= 1 and abs(a.pos[1]-y) <= 1
            and self.trust.get(a.id, 0.0) < 0.0
        ]
        if not targets:
            return 0.0
        target = random.choice(targets)
        dmg = max(1.0, 8.0 * self.genes['aggression'] + agg_bias * 5.0)
        target.health -= dmg
        target.endocrine.apply_attack_received()
        if target.health <= 0:
            target.alive = False
            loot = target.energy * 0.3
            self.energy = min(MAX_ENERGY, self.energy + loot)
            return loot
        self.endocrine.apply_attack_received()  # counter-stress
        return 0.5

    def _cooperate(self, agents, mods):
        x, y = self.pos
        social_bias = mods.get('social_bias', 0.0)
        nearby = [
            a for a in agents
            if a is not self and a.alive
            and abs(a.pos[0]-x) <= COOP_PROXIMITY_RADIUS
            and abs(a.pos[1]-y) <= COOP_PROXIMITY_RADIUS
        ]
        if not nearby:
            return 0.0
        reward = 0.0
        same_tribe = [a for a in nearby if a.tribe_id == self.tribe_id and self.tribe_id is not None]
        self.endocrine.apply_social_signal(len(nearby), bool(same_tribe))
        # foraging bonus from group
        forage_bonus = min(COOP_FORAGE_MAX_BONUS, len(nearby) * COOP_FORAGE_BONUS_PER_MEMBER)
        self.energy = min(MAX_ENERGY, self.energy + forage_bonus)
        reward += forage_bonus * 0.1
        # energy sharing
        if self.energy >= COOP_SHARE_THRESHOLD_DONOR:
            for partner in nearby:
                if partner.energy < COOP_SHARE_THRESHOLD_RECV:
                    self.energy   -= COOP_SHARE_AMOUNT
                    partner.energy = min(MAX_ENERGY, partner.energy + COOP_SHARE_AMOUNT)
                    self.trust[partner.id]  = min(1.0, self.trust.get(partner.id, 0.0) + 0.05)
                    partner.trust[self.id]  = min(1.0, partner.trust.get(self.id, 0.0) + 0.08)
                    reward += COOP_SHARE_REWARD
                    break
        # receive sharing
        for partner in nearby:
            if (partner.energy >= COOP_SHARE_THRESHOLD_DONOR
                    and self.energy < COOP_SHARE_THRESHOLD_RECV):
                partner.energy -= COOP_SHARE_AMOUNT
                self.energy = min(MAX_ENERGY, self.energy + COOP_SHARE_AMOUNT)
                reward += COOP_RECV_REWARD
                self.trust[partner.id] = min(1.0, self.trust.get(partner.id, 0.0) + 0.08)
                break
        # trust update with ToM
        for partner in nearby:
            if self.trust.get(partner.id, 0.0) > 0.1:
                self.tom.observe_action(partner.id, 'cooperate', outcome=1.0)
            delta = 0.01 * (1.0 + social_bias)
            self.trust[partner.id] = min(1.0, self.trust.get(partner.id, 0.0) + delta)
        return reward

    def _try_reproduce(self, agents):
        if not self.can_reproduce() or self.sex != 'f':
            return None
        x, y = self.pos
        males = [
            a for a in agents
            if a is not self and a.alive and a.sex == 'm'
            and a.can_reproduce()
            and abs(a.pos[0]-x) <= 2 and abs(a.pos[1]-y) <= 2
            and self.trust.get(a.id, 0.0) >= -0.2
        ]
        if not males:
            return None
        mate = max(males, key=lambda a: (
            a.genes.get('cooperation', 0.5)
            + a.genes.get('plasticity', 1.0) / 1.8
            + self.trust.get(a.id, 0.0)
        ))
        child_genes = inherit_genes(self.genes, mate.genes)
        self.energy  -= REPRODUCTION_COST
        mate.energy  -= REPRODUCTION_COST * 0.5
        self.reproduction_cooldown = REPRODUCTION_COOLDOWN
        mate.reproduction_cooldown = REPRODUCTION_COOLDOWN
        self.pregnant = True
        self.gestation = GESTATION_TIME
        self.stored_child_genes = child_genes
        self._last_mate_id = mate.id
        mate._last_mate_id  = self.id
        self.children += 1
        mate.children  += 1
        return None  # child born via progress_pregnancy

    def _collect_resources(self, world):
        x, y = self.pos
        cell = world.get_cell(x, y)
        tool_bonus = SHARP_STONE_COLLECT_BONUS if self.tool == 'sharp_stone' else 0.0
        for res in ('wood', 'stone', 'fiber'):
            available = cell.get(res, 0)
            if available > 0:
                amount = min(available, 1 + tool_bonus)
                cell[res] = max(0, available - int(amount))
                self.resources[res] = self.resources.get(res, 0) + int(amount)

    def _build(self, world):
        x, y = self.pos
        cell = world.get_cell(x, y)
        result = maybe_build_structure(self, cell)
        if result:
            self.energy -= BUILD_ENERGY_COST

    def _maybe_craft_tool(self):
        if self.tool is None:
            stone = self.material_inventory.get('sharp_stone', 0)
            if stone < 1:
                stone = self.resources.get('stone', 0)
            if stone >= 1:
                self.tool = 'sharp_stone'
                if 'sharp_stone' in self.material_inventory:
                    self.material_inventory['sharp_stone'] -= 1
                else:
                    self.resources['stone'] = max(0, self.resources.get('stone', 0) - 1)

    def _disease_tick(self, world):
        if self.disease_id is None:
            return
        rec = REMEDY_REGISTRY.get(self.disease_id, {})
        severity   = rec.get('severity', 0.5)
        biome      = world.get_biome(*self.pos)
        biome_mult = 1.3 if biome in rec.get('worse_in', []) else 1.0
        drain      = severity * biome_mult
        self.health -= drain
        self.sick    = min(100.0, self.sick + drain)
        if self.health <= 0:
            self.alive = False

    def _age_tick(self):
        self.age += 1
        if self.age > AGE_HEALTH_DECAY_START:
            rate = 0.04 if self.age < AGE_HEALTH_DECAY_HARD else 0.12
            self.health = max(0.0, self.health - rate)
        if self.age >= AGE_LIMIT:
            self.alive = False

    def _sleep_tick(self, mods):
        sleep_drive = mods.get('sleep_drive', 0.0)
        if not self.is_sleeping and sleep_drive > SLEEP_DRIVE_THRESHOLD:
            self.is_sleeping = True
        if self.is_sleeping:
            self.energy   = min(MAX_ENERGY, self.energy   + SLEEP_ENERGY_REGEN)
            self.health   = min(100.0,      self.health   + SLEEP_HEALTH_REGEN)
            self.hydration = max(0.0,       self.hydration - 0.1)
            # wake when sleep drive dissipates
            if sleep_drive < 0.20:
                self.is_sleeping = False

    def update(
        self,
        world,
        agents,
        tick: int,
        season_state=None,
        weather_state=None,
        tribes=None,
        economy=None,
        technology=None,
    ):
        """
        Main per-tick update. Returns child_genes dict if reproduction occurs,
        otherwise None.
        """
        if not self.alive:
            return None

        _ensure_new_fields(self)

        # --- endocrine update (must come first) ---
        self.endocrine.update(self, world)
        mods = self.endocrine.modifiers()

        # --- age & disease ---
        self._age_tick()
        self._disease_tick(world)
        if not self.alive:
            return None

        # --- sleep ---
        self._sleep_tick(mods)

        # --- passive drains ---
        move_cost = 0.5 * mods.get('move_cost_mult', 1.0)
        self.energy    = max(0.0, self.energy    - move_cost)
        self.hydration = max(0.0, self.hydration - 0.3)
        health_drain   = mods.get('health_drain', 0.0)
        self.health    = max(0.0, self.health    - health_drain)

        # starvation / dehydration
        if self.energy <= 0:
            self.health = max(0.0, self.health - 1.5)
        if self.hydration <= 0:
            self.health = max(0.0, self.health - 1.0)
        if self.health <= 0:
            self.alive = False
            return None

        # --- brain: observe & act ---
        features = self.local_features(world, agents)
        if self.hidden_state is None:
            self.hidden_state = self.brain.initial_hidden()

        brain_step = self.brain.act(
            features,
            self.hidden_state,
            use_planning=True,
        )
        self.hidden_state = brain_step['next_hidden']
        action_list = brain_step['action_list']

        # map 6 continuous outputs to named action dict
        action = {
            'move_x':     action_list[0],
            'move_y':     action_list[1],
            'forage':     action_list[2],
            'cooperate':  action_list[3],
            'attack':     action_list[4],
            'build':      action_list[5],
        }

        # --- execute actions ---
        reward = 0.0

        self.primitive_move(world, action)
        apply_structure_effects(self, world.get_cell(*self.pos))

        mode = 'idle'
        if not self.is_sleeping:
            if action['forage'] > 0.0:
                gained = self._forage(world, mods)
                reward += gained * 0.05
                if gained > 0:
                    mode = 'forage'
                    self._collect_herbs(world)

            if action['cooperate'] > 0.2:
                reward += self._cooperate(agents, mods)
                mode = 'cooperate'

            if action['attack'] > 0.5:
                reward += self._attack(agents, mods)
                mode = 'attack'

            if action['build'] > 0.4:
                self._collect_resources(world)
                self._build(world)
                mode = 'build'

            self._maybe_craft_tool()
            self._try_remedy()
            self._share_remedy(agents)

        self.last_action_mode = mode

        # --- territory reward ---
        territory_r = territory_reward_for_agent(world, self)
        reward += territory_r

        # --- reproduction ---
        self._try_reproduce(agents)
        child_genes = self.progress_pregnancy()

        # --- social learning (Schicht 2) ---
        social_learning_step(self, agents)

        # --- need-driven invention (Schicht 1) ---
        if self._need_inv_cooldown <= 0:
            need_vec = compute_need_vector(self)
            inv_result = agent_invent_from_need(self, world.get_cell(*self.pos), need_vec)
            if inv_result:
                reward += 0.5
                self.endocrine.apply_discovery(1.0)
            self._need_inv_cooldown = NEED_INVENTION_INTERVAL
        else:
            self._need_inv_cooldown -= 1

        # --- standard invention & cooking ---
        inv_prob = INVENTION_BASE_PROB + INVENTION_CURIOSITY_MULT * self.genes.get('curiosity', 0.5)
        if random.random() < inv_prob:
            invented = agent_try_invention(self, world.get_cell(*self.pos))
            if invented:
                reward += 1.0
                self.endocrine.apply_discovery(1.0)
        if random.random() < 0.3:
            cooked = agent_try_cook(self, world.get_cell(*self.pos))
            if cooked:
                reward += 0.3
                self.endocrine.apply_substance('cooked_meat', 1.0)

        # --- economy participation ---
        if economy is not None:
            economy.agent_trade(self, agents)

        # --- causal memory update ---
        next_features = self.local_features(world, agents)
        delta_energy = (self.energy - features[0] * MAX_ENERGY) / MAX_ENERGY
        if abs(delta_energy) > 0.02:
            seq_key = (mode, round(delta_energy, 1))
            self.causal_memory.record(seq_key, delta_energy)

        # --- intrinsic reward & brain training ---
        next_features_raw = self.local_features(world, agents)
        intrinsic = self.brain.intrinsic_reward(
            brain_step['hidden_in'],
            brain_step['action_tensor'],
            next_features_raw,
        )
        reward += 0.3 * intrinsic

        # update episodic memory with current observation
        import torch
        next_obs_t = torch.tensor(next_features_raw, dtype=torch.float32,
                                  device=self.brain.initial_hidden().device)
        self.brain.episodic_memory.add(next_obs_t)

        # store transition and maybe train
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

        # ToM & emotional memory update
        self.tom.update_beliefs(agents)
        self.emotional_memory.record_event(
            pos=self.pos,
            emotion_tag=mode,
            valence=min(1.0, max(-1.0, reward * 0.1)),
            tick=tick,
        )

        return child_genes
