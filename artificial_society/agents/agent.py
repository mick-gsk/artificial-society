from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from artificial_society.agents.brain import INPUT_SIZE, Brain
from artificial_society.agents.communication import CommunicationSystem
from artificial_society.agents.emotional_memory import EmotionalMemory
from artificial_society.agents.endocrine import EndocrineSystem
from artificial_society.agents.genetics import inherit_genes, random_genes
from artificial_society.agents.knowledge import KnowledgeGraph
from artificial_society.agents.life_stage import get_stage_stats
from artificial_society.agents.memory import EpisodicMemory
from artificial_society.agents.theory_of_mind import TheoryOfMind
from artificial_society.environment.herbs import available_herbs, collect_herb
from artificial_society.environment.resources import maybe_build_structure
from artificial_society.environment.structures import (
    BUILD_ENERGY_COST,
    apply_structure_effects,
    structure_feature_vector,
)
from artificial_society.environment.territory import (
    get_home_forage_bonus,
    territory_reward_for_agent,
)
from artificial_society.systems.culture import CausalMemory
from artificial_society.systems.goal_stack import GoalStack
from artificial_society.systems.goal_stack_ext import agent_tick_with_goals
from artificial_society.systems.invention import agent_try_cook, agent_try_invention
from artificial_society.systems.language import (
    TOKEN_WORLD,
    TokenMemory,
    agent_mark,
    agent_observe_token,
)
from artificial_society.systems.need_driven_invention import (
    agent_invent_from_need,
    compute_need_vector,
)
from artificial_society.systems.remedy import (
    REMEDY_REGISTRY,
    evaluate_remedy,
    record_cure_discovery,
    share_remedy_knowledge,
)
from artificial_society.systems.social_learning import social_learning_step

MAX_ENERGY = 240.0
INITIAL_ENERGY = 120.0
CHILD_START_ENERGY = 100.0
REPRODUCTION_ENERGY = 60.0
REPRODUCTION_COST = 20.0
REPRODUCTION_COOLDOWN = 100
# Tuned values previously applied at import by emergence_runtime; now the source of truth.
MIN_REPRODUCTION_AGE = 60  # int(life_stage.CHILD_MAX * 0.5)
GESTATION_TIME = 40
AGE_LIMIT = 5000
ELDER_AGE = 3500  # life_stage.ADULT_MAX
AGE_HEALTH_DECAY_START = 3500  # life_stage.ADULT_MAX
AGE_HEALTH_DECAY_HARD = 4500
PLANT_ENERGY = 30.0
MEAT_ENERGY = 45.0
CORPSE_ENERGY = 36.0

SHARP_STONE_FORAGE_BONUS = 0.30
SHARP_STONE_COLLECT_BONUS = 0.20

SLEEP_DRIVE_THRESHOLD = 0.45
SLEEP_ENERGY_REGEN = 0.40
SLEEP_HEALTH_REGEN = 0.25

STAGE_CHILD = 120  # life_stage.CHILD_MAX
STAGE_ELDER = ELDER_AGE  # life_stage.ADULT_MAX (3500)

COOP_FORAGE_BONUS_PER_MEMBER = 0.14
COOP_FORAGE_MAX_BONUS = 0.65
COOP_DEFENSE_HEALTH_BONUS = 0.08
COOP_SHARE_THRESHOLD_DONOR = 160.0
COOP_SHARE_THRESHOLD_RECV = 60.0
COOP_SHARE_AMOUNT = 18.0
COOP_SHARE_REWARD = 0.25
COOP_RECV_REWARD = 0.30
COOP_PROXIMITY_RADIUS = 2

INVENTION_BASE_PROB = 0.08
INVENTION_CURIOSITY_MULT = 0.05
NEED_INVENTION_INTERVAL = 12

# Emergenz v3: Macro-Aktion Reward-Bonus wenn eine bekannte Sequenz ausgefuehrt wird
MACRO_ACTION_REWARD_BONUS = 0.4

_RESOURCE_ALIASES = {
    "wood": ("dry_wood", "wet_wood", "wood"),
    "stone": ("stone", "flint"),
    "fiber": ("fiber", "dry_grass", "crushed_herb", "leaf"),
}


def ensure_fields(agent) -> None:
    """Single source of truth for agent field initialisation (Phase 3 / 3c).

    Collapses the historical ``_ensure_new_fields`` / ``_ensure_runtime_fields``
    (agent.py) and ``_migrate_agent`` (simulation.py) paths into one idempotent
    helper. Every assignment is ``hasattr``-guarded, so calling it on a
    fully-constructed agent is a no-op that draws no RNG (preserving the golden
    trajectory), while an agent deserialised from an old checkpoint — missing newer
    fields — is fully reinstated.
    """
    # --- brain + per-agent core objects ---
    if not hasattr(agent, "brain") or agent.brain is None:
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    elif agent.brain.input_size != INPUT_SIZE:
        print(
            f"[compat] Agent {agent.id}: brain input_size={agent.brain.input_size} != {INPUT_SIZE}, rebuilding."
        )
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    if not hasattr(agent, "hidden_state") or agent.hidden_state is None:
        agent.hidden_state = agent.brain.initial_hidden()
    if not hasattr(agent, "_brain_device"):
        agent._brain_device = next(agent.brain.parameters()).device
    if not hasattr(agent, "causal_memory") or agent.causal_memory is None:
        agent.causal_memory = CausalMemory(capacity=32)
    if not hasattr(agent, "material_inventory") or agent.material_inventory is None:
        agent.material_inventory = {}
    if not hasattr(agent, "endocrine") or agent.endocrine is None:
        agent.endocrine = EndocrineSystem()
    if not hasattr(agent, "is_sleeping"):
        agent.is_sleeping = False
    if not hasattr(agent, "tool"):
        agent.tool = None
    if agent.tool is None and getattr(agent, "material_inventory", {}).get("sharp_stone", 0) > 0.1:
        agent.tool = "sharp_stone"
    if not hasattr(agent, "_last_mate_id"):
        agent._last_mate_id = None
    if not hasattr(agent, "_need_inv_cooldown"):
        agent._need_inv_cooldown = 0
    if not hasattr(agent, "tom") or agent.tom is None:
        agent.tom = TheoryOfMind(agent.id)
    if not hasattr(agent, "knowledge") or agent.knowledge is None:
        agent.knowledge = KnowledgeGraph()
    if not hasattr(agent, "emotional_memory") or agent.emotional_memory is None:
        agent.emotional_memory = EmotionalMemory()
    if not hasattr(agent, "remedy_knowledge"):
        agent.remedy_knowledge = {}
    if not hasattr(agent, "herbs_carried"):
        agent.herbs_carried = {}
    # Emergenz v3: recent_action_sequence fuer Makro-Aktion-Erkennung
    if not hasattr(agent, "_recent_action_seq"):
        agent._recent_action_seq = []
    # --- planning / runtime caches ---
    if not hasattr(agent, "goal_stack") or agent.goal_stack is None:
        agent.goal_stack = GoalStack()
    if not hasattr(agent, "token_memory") or agent.token_memory is None:
        agent.token_memory = TokenMemory()
    if not hasattr(agent, "_next_planning_tick"):
        agent._next_planning_tick = 0
    if not hasattr(agent, "_planning_stride"):
        agent._planning_stride = 4
    if not hasattr(agent, "_last_goal_action"):
        agent._last_goal_action = None
    if not hasattr(agent, "_language_retry_tick"):
        agent._language_retry_tick = 0
    if not hasattr(agent, "_inventory_cap"):
        agent._inventory_cap = 24
    if not hasattr(agent, "_cached_nearby_agents"):
        agent._cached_nearby_agents = []
    if not hasattr(agent, "_cached_nearby_radius"):
        agent._cached_nearby_radius = 2
    if not hasattr(agent, "_disease_immunity"):
        agent._disease_immunity = {}


def _compact_material_inventory(agent, max_entries: int = 24) -> None:
    inv = getattr(agent, "material_inventory", None)
    if not inv:
        return

    cleaned = {k: float(v) for k, v in inv.items() if float(v) > 0.01}
    if len(cleaned) <= max_entries:
        agent.material_inventory = cleaned
        return

    protected = {k: v for k, v in cleaned.items() if not k.startswith("mat_")}
    discovered = sorted(
        ((k, v) for k, v in cleaned.items() if k.startswith("mat_")),
        key=lambda item: (-item[1], item[0]),
    )

    remaining = max_entries - len(protected)
    kept = dict(protected)
    if remaining > 0:
        for mat_id, qty in discovered[:remaining]:
            kept[mat_id] = qty

    if len(kept) > max_entries:
        essentials = [
            ("sharp_stone", kept.get("sharp_stone", 0.0)),
            ("fire", kept.get("fire", 0.0)),
            ("ember", kept.get("ember", 0.0)),
            ("cooked_meat", kept.get("cooked_meat", 0.0)),
            ("raw_meat", kept.get("raw_meat", 0.0)),
            ("raw_root", kept.get("raw_root", 0.0)),
            ("stone", kept.get("stone", 0.0)),
            ("wood", kept.get("wood", 0.0)),
            ("fiber", kept.get("fiber", 0.0)),
        ]
        essentials = [(k, v) for k, v in essentials if k in kept and v > 0]
        kept = dict(essentials[:max_entries])

    agent.material_inventory = kept


def _maybe_mark_language(agent, cell: dict, tick: int, context_vec, reward: float) -> float:
    if reward < 0.9 or tick < getattr(agent, "_language_retry_tick", 0):
        return reward
    if random.random() > 0.12:
        return reward

    token_id = agent_mark(agent, cell, context_vec, tick)
    if token_id:
        agent._language_retry_tick = tick + 12
        return reward + 0.1
    return reward


def _observe_tokens(agent, cell: dict, context_vec, reward: float) -> None:
    x, y = agent.pos
    tokens = TOKEN_WORLD.tokens_at(x, y, radius=1)
    if not tokens:
        return
    signal = max(0.0, reward)
    for token in tokens:
        agent_observe_token(agent, token, context_vec, reward_signal=signal)


def _maybe_collect_language_convergence(agent, agents: list, tick: int) -> None:
    if tick % 90 != 0 or getattr(agent, "id", 0) != 1:
        return
    memories = [getattr(a, "token_memory", None) for a in agents]
    memories = [m for m in memories if m is not None]
    if len(memories) >= 2:
        TOKEN_WORLD.check_convergence(memories, tick)
        TOKEN_WORLD.tick_decay()


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
    resources: dict = field(default_factory=lambda: {"wood": 0, "stone": 0, "fiber": 0})
    tool: str | None = None
    alive: bool = True
    learning_score: float = 0.0
    reproduction_cooldown: int = 0
    hidden_state: object = None
    last_reward: float = 0.0
    last_loss: float = 0.0
    sex: str = "m"
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
    last_action_mode: str = "idle"
    disease_id: str | None = None
    remedy_knowledge: dict = field(default_factory=dict)
    herbs_carried: dict = field(default_factory=dict)
    causal_memory: CausalMemory = field(default_factory=lambda: CausalMemory(capacity=32))
    material_inventory: dict = field(default_factory=dict)
    world_memory: dict = field(default_factory=dict)
    current_goal: str = "SURVIVE"
    goal_target: tuple | None = None
    last_goal_change: int = 0
    goal_commitment: int = 0
    is_sleeping: bool = False
    _last_mate_id: int | None = None
    _need_inv_cooldown: int = 0
    tom: TheoryOfMind = field(default_factory=lambda: TheoryOfMind(0))
    knowledge: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    emotional_memory: EmotionalMemory = field(default_factory=EmotionalMemory)
    # Emergenz v3: Kurzzeitgedaechtnis fuer ausgefuehrte Aktionssequenzen
    _recent_action_seq: list = field(default_factory=list)

    @classmethod
    def spawn_random(cls, x, y):
        cls.id_counter += 1
        genes = random_genes()
        brain = Brain(plasticity=genes.get("plasticity", 1.0))
        agent = cls(
            id=cls.id_counter,
            pos=(x, y),
            genes=genes,
            memory=EpisodicMemory(genes["memory_capacity"]),
            brain=brain,
            sex=random.choice(["m", "f"]),
        )
        agent.hidden_state = brain.initial_hidden()
        agent.tom = TheoryOfMind(agent.id)
        agent.knowledge = KnowledgeGraph()
        agent.emotional_memory = EmotionalMemory()
        agent._recent_action_seq = []
        ensure_fields(agent)
        return agent

    @classmethod
    def spawn_child(cls, x, y, genes, generation=1, parent_id=None, tribe_id=None, parent=None):
        cls.id_counter += 1
        brain = Brain(plasticity=genes.get("plasticity", 1.0))
        agent = cls(
            id=cls.id_counter,
            pos=(x, y),
            genes=genes,
            memory=EpisodicMemory(genes["memory_capacity"]),
            brain=brain,
            energy=CHILD_START_ENERGY,
            sex=random.choice(["m", "f"]),
            generation=generation,
            parent_id=parent_id,
            tribe_id=tribe_id,
        )
        agent.hidden_state = brain.initial_hidden()
        agent.reproduction_cooldown = REPRODUCTION_COOLDOWN
        agent.tom = TheoryOfMind(agent.id)
        agent.knowledge = KnowledgeGraph()
        agent.emotional_memory = EmotionalMemory()
        agent._recent_action_seq = []
        if parent is not None:
            agent.tom.inherit_from(parent.tom, strength=0.4)
            agent.knowledge.inherit_from(parent.knowledge, strength=0.7)
            agent.emotional_memory.inherit_from(parent.emotional_memory, strength_factor=0.30)
            known_places = list(parent.world_memory.items())
            random.shuffle(known_places)
            for pos, info in known_places[:50]:
                agent.world_memory[pos] = dict(info)
        ensure_fields(agent)
        return agent

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def _nearby_cached(self, agents, radius=2):
        cached = getattr(self, "_cached_nearby_agents", None)
        cached_radius = getattr(self, "_cached_nearby_radius", None)
        if cached is not None and cached_radius == radius:
            return cached
        x, y = self.pos
        nearby = [
            other
            for other in agents
            if other is not self
            and other.alive
            and abs(other.pos[0] - x) <= radius
            and abs(other.pos[1] - y) <= radius
        ]
        self._cached_nearby_agents = nearby
        self._cached_nearby_radius = radius
        return nearby

    def life_stage(self):
        if self.age < STAGE_CHILD:
            return "child"
        if self.age >= STAGE_ELDER:
            return "elder"
        return "adult"

    def display_color(self):
        meat_bias = max(0.0, min(1.0, (self.genes["diet_preference"] + 1.0) * 0.5))
        plant_bias = max(0.0, min(1.0, (-self.genes["diet_preference"] + 1.0) * 0.5))
        g = max(
            70,
            min(
                255, int(80 + 120 * self.genes["cooperation"] + 35 * self.genes["memory_retention"])
            ),
        )
        r = max(70, min(255, int(80 + 100 * meat_bias + 60 * self.genes["aggression"])))
        b = max(70, min(255, int(80 + 100 * plant_bias + 50 * self.genes["plasticity"] / 1.8)))
        return (r, g, b)

    def can_reproduce(self):
        return (
            self.alive
            and self.age >= MIN_REPRODUCTION_AGE
            and self.age < ELDER_AGE
            and self.energy >= REPRODUCTION_ENERGY
            and self.reproduction_cooldown <= 0
            and not self.pregnant
        )

    def local_features(self, world, agents):
        ensure_fields(self)
        x, y = self.pos
        cell = world.get_cell(x, y)
        near = [
            a
            for a in agents
            if a is not self
            and abs(a.pos[0] - x) <= int(self.genes["sense_radius"])
            and abs(a.pos[1] - y) <= int(self.genes["sense_radius"])
            and a.alive
        ]
        friends = sum(1 for a in near if self.trust.get(a.id, 0.0) > 0.2)
        retrieval = self.memory.retrieval_features(x, y, cell["tick"], world.width, world.height)
        avg_trust = sum(self.trust.values()) / len(self.trust) if self.trust else 0.0
        herb_pres = min(1.0, len(available_herbs(cell)) / 5.0)
        warmth = min(1.0, cell.get("warmth", 0.0))
        mat_count = min(1.0, len(cell.get("materials", {})) / 6.0)
        causal_f = self.causal_memory.feature_vector()
        inv_size = min(1.0, sum(self.material_inventory.values()) / 5.0)
        hormones = self.endocrine.as_features()
        struct_f = structure_feature_vector(cell)
        return [
            self.energy / MAX_ENERGY,
            self.health / 100.0,
            self.hydration / 100.0,
            min(1.0, self.age / AGE_LIMIT),
            cell["food"] / 180.0,
            cell["water"] / 100.0,
            (cell["temperature"] + 20) / 72.0,
            cell["danger"] / 100.0,
            cell["disease"] / 100.0,
            cell["soil_fertility"] / 100.0,
            cell["pollution"] / 100.0,
            cell["carrying_capacity"] / 100.0,
            cell["moisture"] / 100.0,
            cell["ash"] / 100.0,
            cell["disturbance"] / 100.0,
            min(1.0, len(near) / 10.0),
            min(1.0, friends / 6.0),
            self.genes["curiosity"],
            self.genes["aggression"],
            self.genes["cooperation"],
            self.genes["sociality"],
            1.0 if self.tool else 0.0,
            avg_trust * 0.5 + 0.5,
            min(1.0, self.resources["wood"] / 4.0),
            min(1.0, self.resources["stone"] / 4.0),
            min(1.0, self.resources["fiber"] / 4.0),
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
        melatonin = self.endocrine.h[2]
        vision_mult = max(0.4, 1.0 - 0.6 * melatonin)
        radius = max(
            1, int(round((self.genes["vision"] + 0.35 * self.genes["sense_radius"]) * vision_mult))
        )
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
        if self.is_sleeping:
            return
        dx = 1 if action["move_x"] > 0.33 else -1 if action["move_x"] < -0.33 else 0
        dy = 1 if action["move_y"] > 0.33 else -1 if action["move_y"] < -0.33 else 0
        x, y = self.pos
        nx = max(0, min(world.width - 1, x + dx))
        ny = max(0, min(world.height - 1, y + dy))
        cell = world.get_cell(nx, ny)
        if cell.get("passable", True):
            self.pos = (nx, ny)

    def _forage(self, world, mods):
        x, y = self.pos
        cell = world.get_cell(x, y)
        gain = 0.0
        tool_bonus = SHARP_STONE_FORAGE_BONUS if self.tool == "sharp_stone" else 0.0
        home_bonus = get_home_forage_bonus(self, world)
        eff = mods.get("forage_eff", 1.0) * (1.0 + tool_bonus + home_bonus)
        food_available = cell.get("food", 0.0)
        if food_available > 0:
            diet = self.genes.get("diet_preference", 0.0)
            if diet < 0:
                take = min(food_available, PLANT_ENERGY * eff)
                cell["food"] = max(0.0, food_available - take)
                self.energy = min(MAX_ENERGY, self.energy + take)
                self.plant_eaten += 1
                gain += take
                self.endocrine.apply_substance("plant_food", take / PLANT_ENERGY)
                self.endocrine.apply_successful_forage(take)
            else:
                carcass = cell.get("carcass", 0.0)
                if carcass > 0:
                    take = min(carcass, MEAT_ENERGY * eff)
                    cell["carcass"] = max(0.0, carcass - take)
                    self.energy = min(MAX_ENERGY, self.energy + take)
                    self.meat_eaten += 1
                    gain += take
                    self.endocrine.apply_substance("raw_meat", take / MEAT_ENERGY)
                    self.endocrine.apply_successful_forage(take)
                else:
                    take = min(food_available, PLANT_ENERGY * eff)
                    cell["food"] = max(0.0, food_available - take)
                    self.energy = min(MAX_ENERGY, self.energy + take)
                    self.plant_eaten += 1
                    gain += take
                    self.endocrine.apply_substance("plant_food", take / PLANT_ENERGY)
                    self.endocrine.apply_successful_forage(take)
        water_available = cell.get("water", 0.0)
        if water_available > 0 and self.hydration < 100.0:
            take = min(water_available, 8.0 * eff)
            cell["water"] = max(0.0, water_available - take)
            self.hydration = min(100.0, self.hydration + take)
            gain += take * 0.3
            self.endocrine.apply_substance("water", take / 8.0)
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
            self.endocrine.apply_substance(f"herb_{herb}", 1.0)

    def _try_remedy(self):
        if self.disease_id is None or not self.herbs_carried:
            return
        result = evaluate_remedy(self, self.disease_id)
        if result == "cured":
            record_cure_discovery(self, self.disease_id)
            self.disease_id = None
            self.sick = max(0.0, self.sick - 40.0)
        elif result == "partial":
            self.sick = max(0.0, self.sick - 15.0)

    def _share_remedy(self, agents):
        if not self.remedy_knowledge:
            return
        x, y = self.pos
        near = [
            a
            for a in agents
            if a is not self and a.alive and abs(a.pos[0] - x) <= 2 and abs(a.pos[1] - y) <= 2
        ]
        if near:
            share_remedy_knowledge(self, random.choice(near))

    def _attack(self, agents, mods):
        x, y = self.pos
        agg_bias = mods.get("aggression_bias", 0.0)
        threshold = max(0.05, self.genes["aggression"] + agg_bias - 0.3)
        if random.random() > threshold:
            return 0.0
        targets = [
            a
            for a in agents
            if a is not self
            and a.alive
            and abs(a.pos[0] - x) <= 1
            and abs(a.pos[1] - y) <= 1
            and self.trust.get(a.id, 0.0) < 0.0
        ]
        if not targets:
            return 0.0
        target = random.choice(targets)
        dmg = max(1.0, 8.0 * self.genes["aggression"] + agg_bias * 5.0)
        target.health -= dmg
        target.endocrine.apply_attack_received()
        if target.health <= 0:
            target.alive = False
            loot = target.energy * 0.3
            self.energy = min(MAX_ENERGY, self.energy + loot)
            return loot
        self.endocrine.apply_attack_received()
        return 0.5

    def _cooperate(self, agents, mods, tick):
        x, y = self.pos
        social_bias = mods.get("social_bias", 0.0)
        nearby = [
            a
            for a in agents
            if a is not self
            and a.alive
            and abs(a.pos[0] - x) <= COOP_PROXIMITY_RADIUS
            and abs(a.pos[1] - y) <= COOP_PROXIMITY_RADIUS
        ]
        if not nearby:
            return 0.0
        reward = 0.0
        same_tribe = [
            a for a in nearby if a.tribe_id == self.tribe_id and self.tribe_id is not None
        ]
        self.endocrine.apply_social_signal(len(nearby), bool(same_tribe))
        forage_bonus = min(COOP_FORAGE_MAX_BONUS, len(nearby) * COOP_FORAGE_BONUS_PER_MEMBER)
        self.energy = min(MAX_ENERGY, self.energy + forage_bonus)
        reward += forage_bonus * 0.1
        if self.energy >= COOP_SHARE_THRESHOLD_DONOR:
            for partner in nearby:
                if partner.energy < COOP_SHARE_THRESHOLD_RECV:
                    self.energy -= COOP_SHARE_AMOUNT
                    partner.energy = min(MAX_ENERGY, partner.energy + COOP_SHARE_AMOUNT)
                    self.trust[partner.id] = min(1.0, self.trust.get(partner.id, 0.0) + 0.05)
                    partner.trust[self.id] = min(1.0, partner.trust.get(self.id, 0.0) + 0.08)
                    reward += COOP_SHARE_REWARD
                    break
        for partner in nearby:
            if (
                partner.energy >= COOP_SHARE_THRESHOLD_DONOR
                and self.energy < COOP_SHARE_THRESHOLD_RECV
            ):
                partner.energy -= COOP_SHARE_AMOUNT
                self.energy = min(MAX_ENERGY, self.energy + COOP_SHARE_AMOUNT)
                reward += COOP_RECV_REWARD
                self.trust[partner.id] = min(1.0, self.trust.get(partner.id, 0.0) + 0.08)
                break
        for partner in nearby:
            if self.trust.get(partner.id, 0.0) > 0.1:
                self.tom.observe_agent(partner, tick, own_trust=self.trust.get(partner.id, 0.0))
            delta = 0.01 * (1.0 + social_bias)
            self.trust[partner.id] = min(1.0, self.trust.get(partner.id, 0.0) + delta)
        return reward

    def _try_reproduce(self, agents):
        if not self.can_reproduce() or self.sex != "f":
            return None
        x, y = self.pos
        males = [
            a
            for a in agents
            if a is not self
            and a.alive
            and a.sex == "m"
            and a.can_reproduce()
            and abs(a.pos[0] - x) <= 5
            and abs(a.pos[1] - y) <= 5
            and self.trust.get(a.id, 0.0) >= -0.2
        ]
        if not males:
            return None
        mate = max(
            males,
            key=lambda a: (
                a.genes.get("cooperation", 0.5)
                + a.genes.get("plasticity", 1.0) / 1.8
                + self.trust.get(a.id, 0.0)
            ),
        )
        child_genes = inherit_genes(self, mate)
        self.energy -= REPRODUCTION_COST
        mate.energy -= REPRODUCTION_COST * 0.5
        self.reproduction_cooldown = REPRODUCTION_COOLDOWN
        mate.reproduction_cooldown = REPRODUCTION_COOLDOWN
        self.pregnant = True
        eff = self.genes.get("gestation_efficiency", 1.0)

        self.gestation = max(20, int(GESTATION_TIME / eff))
        self.stored_child_genes = child_genes
        self._last_mate_id = mate.id
        mate._last_mate_id = self.id
        self.children += 1
        mate.children += 1
        return None

    def _collect_resources(self, world):
        x, y = self.pos
        cell = world.get_cell(x, y)
        slot = cell.setdefault("materials", {})
        tool_bonus = 0.20 if getattr(self, "tool", None) == "sharp_stone" else 0.0

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

    def _build(self, world):
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

    def _maybe_craft_tool(self):
        if self.tool is None:
            stone = self.material_inventory.get("sharp_stone", 0)
            if stone < 1:
                stone = self.resources.get("stone", 0)
            if stone >= 1:
                self.tool = "sharp_stone"
                if "sharp_stone" in self.material_inventory:
                    self.material_inventory["sharp_stone"] -= 1
                else:
                    self.resources["stone"] = max(0, self.resources.get("stone", 0) - 1)

    def _disease_tick(self, world):
        if self.disease_id is None:
            return
        rec = REMEDY_REGISTRY.get(self.disease_id, {})
        severity = rec.get("severity", 0.5)
        biome = world.get_biome(*self.pos)
        biome_mult = 1.3 if biome in rec.get("worse_in", []) else 1.0
        drain = severity * biome_mult
        self.health -= drain
        self.sick = min(100.0, self.sick + drain)
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
        sleep_drive = mods.get("sleep_drive", 0.0)
        if not self.is_sleeping and sleep_drive > SLEEP_DRIVE_THRESHOLD:
            self.is_sleeping = True
        if self.is_sleeping:
            self.energy = min(MAX_ENERGY, self.energy + SLEEP_ENERGY_REGEN)
            self.health = min(100.0, self.health + SLEEP_HEALTH_REGEN)
            self.hydration = max(0.0, self.hydration - 0.1)
            if sleep_drive < 0.20:
                self.is_sleeping = False

    def _record_macro_if_successful(self, mode: str, reward: float) -> float:
        """
        Emergenz v3: Verfolgt Aktionssequenzen und speichert erfolgreiche
        als CompositeAction im KnowledgeGraph.

        Funktionsweise:
        - Jede ausgefuehrte nicht-idle Aktion wird in _recent_action_seq gemerkt.
        - Wenn reward > 0.5, wird die letzte 2-3 Aktionen als Makro-Aktion registriert.
        - Das Gehirn erhaelt einen Bonus-Reward wenn es eine bekannte Makro-Aktion
          reproduziert (Reinforcement des erlernten Verhaltensmusters).

        Biologisches Vorbild: Menschen automatisieren erfolgreiche Verhaltenssequenzen
        (Motorische Schemata, Prozedurales Gedaechtnis).
        """
        if mode == "idle":
            return 0.0

        seq = getattr(self, "_recent_action_seq", [])
        seq.append(mode)
        # Fenster: letzte 3 Aktionen
        if len(seq) > 3:
            seq.pop(0)
        self._recent_action_seq = seq

        macro_bonus = 0.0
        if reward > 0.5 and len(seq) >= 2:
            # Neue oder bekannte Makro-Aktion registrieren
            macro = self.knowledge.record_macro(
                steps=list(seq),
                reward=reward,
                materials=list(self.material_inventory.keys()),
            )
            # Bonus wenn diese Sequenz bereits bekannt und bestaetigt ist
            if macro.uses > 3 and macro.confidence > 0.3:
                macro_bonus = MACRO_ACTION_REWARD_BONUS * macro.confidence

        return macro_bonus

    def update_memory(self, world):

        MEMORY_DECAY = 500
        MAX_MEMORY_CELLS = 200

        x, y = self.pos

        for agent in world.agents:
            if agent is self:
                continue

            if not agent.alive:
                continue

            if abs(agent.pos[0] - x) <= 8 and abs(agent.pos[1] - y) <= 8:
                self.world_memory[("agent", agent.id)] = {
                    "type": "agent",
                    "value": 1,
                    "tick": self.age,
                    "pos": agent.pos,
                }

        for dx in range(-4, 5):
            for dy in range(-4, 5):
                nx = x + dx
                ny = y + dy

                if not world.in_bounds(nx, ny):
                    continue

                cell = world.get_cell(nx, ny)

                if cell.get("food", 0) > 15:
                    self.world_memory[(nx, ny)] = {
                        "type": "food",
                        "value": cell["food"],
                        "tick": self.age,
                    }

                if cell.get("water", 0) > 15:
                    self.world_memory[(nx, ny)] = {
                        "type": "water",
                        "value": cell["water"],
                        "tick": self.age,
                    }

                if cell.get("danger", 0) > 30:
                    self.world_memory[(nx, ny)] = {
                        "type": "danger",
                        "value": cell["danger"],
                        "tick": self.age,
                    }

                if cell.get("disease", 0) > 20:
                    self.world_memory[(nx, ny)] = {
                        "type": "disease",
                        "value": cell["disease"],
                        "tick": self.age,
                    }

                if cell.get("warmth", 0) > 0.2:
                    self.world_memory[(nx, ny)] = {
                        "type": "warmth",
                        "value": cell["warmth"],
                        "tick": self.age,
                    }

        self.world_memory = {
            pos: info
            for pos, info in self.world_memory.items()
            if self.age - info["tick"] < MEMORY_DECAY
        }

        if len(self.world_memory) > MAX_MEMORY_CELLS:
            newest = sorted(self.world_memory.items(), key=lambda x: x[1]["tick"])[
                -MAX_MEMORY_CELLS:
            ]

            self.world_memory = dict(newest)

    def choose_goal(self):

        if self.goal_commitment > 0:
            self.goal_commitment -= 1
            return self.current_goal

        if self.energy < 60:
            self.current_goal = "EAT"
            self.goal_commitment = 30

        elif self.hydration < 40:
            self.current_goal = "DRINK"
            self.goal_commitment = 30

        elif (
            self.energy > REPRODUCTION_ENERGY
            and self.health > 70
            and self.age > MIN_REPRODUCTION_AGE
        ):
            self.current_goal = "REPRODUCE"
            self.goal_commitment = 40

        else:
            self.current_goal = "EXPLORE"
            self.goal_commitment = 20

        return self.current_goal

    def select_goal_target(self):

        candidates = []

        for pos, info in self.world_memory.items():
            score = info["value"]

            if self.current_goal == "EAT":
                if info["type"] == "food":
                    candidates.append((score, pos))

            elif self.current_goal == "DRINK":
                if info["type"] == "water":
                    candidates.append((score, pos))

            elif self.current_goal == "REPRODUCE":
                if info["type"] == "agent":
                    candidates.append((1, info["pos"]))

            elif self.current_goal == "EXPLORE" and info["type"] in (
                "food",
                "water",
                "warmth",
            ):
                candidates.append((score * 0.5, pos))

        if candidates:
            candidates.sort(reverse=True)
            self.goal_target = candidates[0][1]
        else:
            self.goal_target = None

    def goal_move(self, world):

        if self.goal_target is None:
            return None

        tx, ty = self.goal_target

        if not world.in_bounds(tx, ty):
            self.goal_target = None
            return None

        target_cell = world.get_cell(tx, ty)

        if self.current_goal == "EAT" and target_cell.get("food", 0) <= 0:
            self.goal_target = None
            return None

        if self.current_goal == "DRINK" and target_cell.get("water", 0) <= 0:
            self.goal_target = None
            return None

        x, y = self.pos

        dx = 0
        dy = 0

        if tx > x:
            dx = 1
        elif tx < x:
            dx = -1

        if ty > y:
            dy = 1
        elif ty < y:
            dy = -1

        return dx, dy

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
        if not self.alive:
            return None

        ensure_fields(self)

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

        move_cost = (
            0.5
            * mods.get("move_cost_mult", 1.0)
            * stage["move_cost_mult"]
            * structure_mods.get("cold_factor", 1.0)
        )
        hydration_loss = 0.3 * mods.get("hydration_loss_mult", 1.0) * stage["hydration_loss_mult"]
        self.energy = max(0.0, self.energy - move_cost)
        self.hydration = max(
            0.0,
            min(
                100.0, self.hydration - hydration_loss + structure_mods.get("hydration_bonus", 0.0)
            ),
        )
        self.health = max(
            0.0,
            self.health - mods.get("health_drain", 0.0) * structure_mods.get("disease_factor", 1.0),
        )

        if self.energy <= 0:
            self.health = max(0.0, self.health - 1.5)
        if self.hydration <= 0:
            self.health = max(0.0, self.health - 1.0)
        if self.health <= 0:
            self.alive = False
            return None

        nearby_agents = self._nearby_cached(agents, 2)
        features = self.local_features(world, agents)
        if self.hidden_state is None:
            self.hidden_state = self.brain.initial_hidden()

        self._planning_stride = (
            2 if getattr(self, "goal_stack", None) and not self.goal_stack.is_empty() else 4
        )
        use_planning = tick >= getattr(self, "_next_planning_tick", 0)
        if use_planning:
            self._next_planning_tick = tick + self._planning_stride

        research_mode = self._need_inv_cooldown <= 0 or (
            getattr(self, "goal_stack", None) is not None and not self.goal_stack.is_empty()
        )
        brain_step = self.brain.act(
            features,
            self.hidden_state,
            use_planning=use_planning,
            research_mode=research_mode,
        )
        self.hidden_state = brain_step["next_hidden"]
        action_list = brain_step["action_list"]
        action = {
            "move_x": action_list[0],
            "move_y": action_list[1],
            "forage": action_list[2],
            "cooperate": action_list[3],
            "attack": action_list[4],
            "build": action_list[5],
        }

        reward = 0.0
        mode = "idle"

        if getattr(self, "goal_stack", None) is not None:
            goal_action, goal_shaping = agent_tick_with_goals(
                self,
                current_cell,
                {
                    "tick": tick,
                    "season_state": season_state or {},
                    "weather_state": weather_state or {},
                    "tribes": tribes,
                    "economy": economy,
                    "technology": technology,
                },
                tick,
            )
            self._last_goal_action = goal_action
            reward += 0.6 * goal_shaping
            if goal_action is not None:
                self._next_planning_tick = tick
                if goal_action in {"collect", "harvest"}:
                    action["forage"] = max(action["forage"], 0.7)
                elif goal_action in {"place_on_heat", "arch", "build_dome", "fire_pottery", "form"}:
                    action["build"] = max(action["build"], 0.7)
                elif goal_action == "wait":
                    action["move_x"] = 0.0
                    action["move_y"] = 0.0
                    action["forage"] = 0.0
                    action["attack"] = 0.0

        self.primitive_move(world, action)
        current_cell = world.get_cell(*self.pos)
        structure_mods = apply_structure_effects(self, current_cell)

        if not self.is_sleeping:
            if action["forage"] > 0.0:
                gained = self._forage(
                    world,
                    {
                        **mods,
                        "forage_eff": mods.get("forage_eff", 1.0)
                        * (1.0 + structure_mods.get("forage_bonus", 0.0)),
                    },
                )
                reward += gained * 0.05 * stage.get("foraging_mult", 1.0)
                if gained > 0:
                    mode = "forage"
                    self._collect_herbs(world)

            if action["cooperate"] > 0.2:
                reward += self._cooperate(agents, mods, tick)
                mode = "cooperate"

            if action["attack"] > 0.5 and stage.get("can_attack", True):
                reward += self._attack(agents, mods)
                mode = "attack"

            if action["build"] > 0.4 and stage.get("can_build", True):
                self._collect_resources(world)
                self._build(world)
                mode = "build"

            self._maybe_craft_tool()
            self._try_remedy()
            self._share_remedy(agents)

        self.last_action_mode = mode
        reward += territory_reward_for_agent(self, world)

        if stage.get("can_reproduce", True):
            self._try_reproduce(agents)
        child_genes = self.progress_pregnancy()

        if tick % 3 == 0:
            reward += social_learning_step(self, agents, tick)

        if self._need_inv_cooldown <= 0:
            compute_need_vector(self, current_cell)
            inv_result = agent_invent_from_need(self, current_cell, current_cell, tick)
            if inv_result:
                reward += 0.5
                self.endocrine.apply_discovery(1.0)
            self._need_inv_cooldown = NEED_INVENTION_INTERVAL
        else:
            self._need_inv_cooldown -= 1

        inv_prob = INVENTION_BASE_PROB + INVENTION_CURIOSITY_MULT * self.genes.get("curiosity", 0.5)
        if tick % 3 == 0 and random.random() < inv_prob:
            invented = agent_try_invention(self, current_cell, current_cell)
            if invented:
                reward += 1.0
                self.endocrine.apply_discovery(1.0)

        if tick % 4 == 0 and random.random() < 0.18:
            cooked = agent_try_cook(self, current_cell)
            if cooked:
                reward += 0.3
                self.endocrine.apply_substance("cooked_meat", 1.0)

        if economy is not None:
            economy.maybe_trade(self, agents)

        next_features_raw = self.local_features(world, agents)
        intrinsic = self.brain.intrinsic_reward(
            brain_step["hidden_in"],
            brain_step["action_tensor"],
            next_features_raw,
        )
        reward += 0.3 * intrinsic

        next_obs_t = torch.tensor(
            next_features_raw,
            dtype=torch.float32,
            device=brain_step["hidden_in"].device,
        )
        self.brain.episodic_memory.novelty(next_obs_t)

        context_vec = np.asarray(next_features_raw, dtype=np.float32)
        reward = _maybe_mark_language(self, current_cell, tick, context_vec, reward)
        _observe_tokens(self, current_cell, context_vec, reward)
        _maybe_collect_language_convergence(self, agents, tick)
        _compact_material_inventory(self, getattr(self, "_inventory_cap", 24))

        cognition_mult = mods.get("cognition", 1.0)
        effective_reward = reward * cognition_mult
        self.brain.store_transition(
            brain_step["obs_tensor"],
            brain_step["hidden_in"],
            brain_step["action_tensor"],
            brain_step["log_prob"],
            brain_step["value"],
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
