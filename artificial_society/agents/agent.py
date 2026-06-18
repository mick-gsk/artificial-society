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
        dy = 1 if action['mov