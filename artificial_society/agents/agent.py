from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch

from artificial_society.agents.brain import INPUT_SIZE, V1_HEAD_DIMS, Brain
from artificial_society.agents.communication import CommunicationSystem
from artificial_society.agents.emotional_memory import EmotionalMemory
from artificial_society.agents.endocrine import EndocrineSystem
from artificial_society.agents.genetics import ensure_strength_gene, inherit_genes, random_genes
from artificial_society.agents.knowledge import KnowledgeGraph
from artificial_society.agents.life_stage import get_stage_stats
from artificial_society.agents.memory import EpisodicMemory
from artificial_society.agents.perception_v2 import (
    NoveltyBuckets,
    admissible_masks,
    build_slots,
    resolve_slot_of,
)
from artificial_society.agents.theory_of_mind import TheoryOfMind
from artificial_society.environment.herbs import available_herbs, collect_herb
from artificial_society.environment.materials import get_vector, material_reward
from artificial_society.environment.physics.actions import (
    do_cut,
    do_eat,
    do_grasp,
    do_release,
    do_strike,
    enforce_carry_budget,
)
from artificial_society.environment.physics.body import BODY_MASS_DEFAULT_KG, Body, Hands
from artificial_society.environment.resources import apply_consumption, clamp, maybe_build_structure
from artificial_society.environment.structures import (
    BUILD_ENERGY_COST,
    apply_structure_effects,
    structure_feature_vector,
)
from artificial_society.environment.territory import (
    get_home_forage_bonus,
    territory_reward_for_agent,
)
from artificial_society.systems.causal_model import CausalModelV2
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
# A newborn's start energy is TRANSFERRED from the mother at birth (capped so
# she keeps this floor), not minted: the old +70 net mint per birth (100 start
# vs ~30 parental conception cost) subsidised population overshoot far past
# the world's food-regrowth carrying capacity, ending in mass starvation.
BIRTH_ENERGY_FLOOR = 10.0
REPRODUCTION_ENERGY = 60.0
REPRODUCTION_COST = 20.0
REPRODUCTION_COOLDOWN = 100
# Density-dependent fertility. Personal energy alone (>= REPRODUCTION_ENERGY) is
# a poor breeding cue: agents hoard up to MAX_ENERGY (240) and so breed off fat
# reserves banked when food was plentiful, blind to how crowded the ground has
# become. The population then overshoots the world's food-regrowth carrying
# capacity and mass-starves (measured: pop 36 -> 64 -> 9 on a 60x40 world). A
# mother now also requires the local food standing stock, shared across the
# mouths already nearby, to clear a floor — the logistic negative feedback of a
# real ecosystem: as local density rises, per-capita food falls and fertility
# drops, so the population settles near carrying capacity instead of crashing.
REPRODUCTION_SENSE_RADIUS = 2
REPRODUCTION_MIN_FOOD_PER_CAPITA = 6.0
# Demografie-Stabilisierung C2 (Option A, nur physics_v2): das Fruchtbarkeits-Gate
# hängt im v2-Pfad an der nachhaltigen ERNEUERUNG pro Mund (Σ plant_renewal_ema
# über die (2R+1)²-Box / Münder), NICHT am momentanen Bestand. Das momentane-
# Bestand-Gate (REPRODUCTION_MIN_FOOD_PER_CAPITA) wird bei voller t0-Welt vom
# nicht erneuerbaren Startpuffer getäuscht → Malthus-Overshoot 30→~55→Crash.
# Der EMA-Fluss dämpft dagegen doppelt bei niedriger Dichte (Zähler steigt via
# stress_factor-Erholung, Nenner sinkt) und erhält so die Erholung.
# SKALA: EMA-Fluss (food/tick), NICHT die alte 6/8-Bestandsskala. Der Default
# 0.20 wurde per Instrumentierung (gate-off, 4 Seeds, grid 24x18) in die Lücke
# der renewal-pro-Mund-Verteilung gelegt: crowded (pop>=45) p75=0.12 / recovered
# (pop 8-15) p75=0.21. 0.20 verwirft ~88% der Overshoot-Checks (> crowded-p75) und
# öffnet ~27% der Erholungs-Checks (< recovered-p75). Freier, sweepbarer Parameter
# (Harness --min-renewal-per-capita).
REPRODUCTION_MIN_RENEWAL_PER_CAPITA = 0.20
# Angeborene Grund-Taxis (Chemotaxis-Analogon, Architektur A2): symmetrischer
# Nahrungs- & Partner-GRUNDTRIEB, der im physics_v2-Pfad die Welt-Wirkung des
# Move-Kopfs (Dims 0/1) ersetzt. Rein deterministisch, KEIN random.* — der
# zentrale seed-getriebene RNG-Strom wird nicht verschoben. Als eingebauter
# Trieb gedacht (wie Metabolismus/Konzeption-bei-Ko-Lokation); Werkzeuge/
# Sprache/Bauen bleiben gelernt. Kalibrierung (Schwelle/Radius) ist offen und
# wird im Piloten sweepbar in den meta-Record geloggt.
#
# Schwellen-Ordnung (Review-Auflage A1): REPRODUCTION_ENERGY (60) <
# TAXIS_HUNGER_THRESHOLD (80). Der effektive Mate-Seek-Floor ist damit = 80:
# unter 80 sucht ein Agent Nahrung (`hungry`), ab 80 und repro-fähig einen
# Partner. 80 ist BEWUSST niedrig gewählt — der Review warnte, dass 120 die
# Partnersuche erst ab E>=120 zulässt und die Allee-Falle nur langsam bricht.
# 80 (~0.33*MAX_ENERGY=240) hält einen echten Hunger-Boden und lässt zugleich
# das breite Energieband [80,240] aktiv Partner suchen, sodass paarungsbereite
# Agenten (energy>=60) realistisch zur Partnersuche kommen.
TAXIS_HUNGER_THRESHOLD = 80.0
# Partner-Wahrnehmungsradius (Chebyshev), > ±5-Konzeptionsbox in _try_reproduce
# (die NICHT geändert wird), damit Agenten die letzte Lücke ins Konzeptions-
# fenster aktiv schließen. Bewusst flacher Konstant-Radius (andere Modalität als
# der per-Agent-`sense_radius` der Nahrungs-Taxis) — im Review als vertretbare
# Asymmetrie benannt.
MATE_SEEK_RADIUS = 8
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
        # None = "no snapshot yet"; update() resets this every tick. (The old []
        # default was a permanently-valid empty cache that silently disabled
        # social learning, trade and ToM for the agent's whole life.)
        agent._cached_nearby_agents = None
    if not hasattr(agent, "_cached_nearby_radius"):
        agent._cached_nearby_radius = 2
    if not hasattr(agent, "_disease_immunity"):
        agent._disease_immunity = {}
    # --- Physik v2 (Plan 3a) ---
    if not hasattr(agent, "physics_v2"):
        agent.physics_v2 = False
    if not hasattr(agent, "body"):
        agent.body = None
    if not hasattr(agent, "hands"):
        agent.hands = None
    if agent.physics_v2 and agent.body is None:
        agent.body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=agent.genes.get("strength", 0.5))
    if agent.physics_v2 and agent.hands is None:
        agent.hands = Hands()
    if agent.physics_v2:
        # Plan 3b: v2-Brain + Neugier-Apparat idempotent nachrüsten
        # (Checkpoint-geladene Agenten; frisch gebaute sind schon komplett).
        if not getattr(agent.brain, "physics_v2", False):
            print(f"[compat] Agent {agent.id}: v1-Brain im v2-Modus, rebuilding.")
            agent.brain = Brain(plasticity=agent.genes.get("plasticity", 1.0), physics_v2=True)
            agent.hidden_state = agent.brain.initial_hidden()
        if getattr(agent, "causal_model", None) is None:
            agent.causal_model = CausalModelV2(plasticity=agent.genes.get("plasticity", 1.0))
        if getattr(agent, "_novelty_buckets", None) is None:
            agent._novelty_buckets = NoveltyBuckets()
        if not hasattr(agent, "_causal_pending"):
            agent._causal_pending = None
        if not hasattr(agent, "_causal_epistemic_pending"):
            agent._causal_epistemic_pending = 0.0
        if not hasattr(agent, "curiosity_last"):
            agent.curiosity_last = {"nextslot": 0.0, "causal": 0.0, "novelty": 0.0}


def attach_body(agent) -> None:
    """Physik-v2-Embodiment: Body + Hände aus dem strength-Gen, v2-Brain,
    Causal Model und Novelty-Buckets (Plan 3b). Zieht RNG nur im v2-Pfad.

    Das v2-Brain wird VOR inherit_weights_from gebaut (spawn_child_from_parent
    ruft attach_body zuerst) — Eltern- und Kind-Brain sind dann form-gleich.
    """
    agent.physics_v2 = True
    ensure_strength_gene(agent.genes)
    agent.body = Body(body_mass=BODY_MASS_DEFAULT_KG, strength=agent.genes["strength"])
    agent.hands = Hands()
    plast = agent.genes.get("plasticity", 1.0)
    agent.brain = Brain(plasticity=plast, physics_v2=True)
    agent.hidden_state = agent.brain.initial_hidden()
    agent.causal_model = CausalModelV2(plasticity=plast)
    agent._novelty_buckets = NoveltyBuckets()
    agent._causal_pending = None
    agent._causal_epistemic_pending = 0.0
    agent.curiosity_last = {"nextslot": 0.0, "causal": 0.0, "novelty": 0.0}


def _inventory_value_state(agent) -> dict:
    """Minimaler homeostatischer Zustand für die Inventar-Bewertung via material_reward."""
    return {
        "energy": getattr(agent, "energy", MAX_ENERGY) / MAX_ENERGY,
        "cold": False,
        "dark": False,
    }


def _compact_material_inventory(agent, max_entries: int = 24) -> None:
    inv = getattr(agent, "material_inventory", None)
    if not inv:
        return

    cleaned = {k: float(v) for k, v in inv.items() if float(v) > 0.01}
    if len(cleaned) <= max_entries:
        agent.material_inventory = cleaned
        return

    # Phase 5 de-scripting: keine privilegierte "essentials"-Liste und kein Vorrang
    # scripted Items (wood/fire/...) vor entdeckten mat_*. Jeder Eintrag konkurriert
    # nach seinem emergenten Wert für DIESEN Agenten (material_reward über den
    # Eigenschaftsvektor); Menge dient als Tiebreak. Die wertvollsten max_entries bleiben.
    state = _inventory_value_state(agent)

    def _value(item):
        mat, qty = item
        try:
            worth = material_reward(get_vector(mat), state)
        except Exception:
            worth = 0.0
        return (worth, qty, mat)

    ranked = sorted(cleaned.items(), key=_value, reverse=True)
    agent.material_inventory = dict(ranked[:max_entries])


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
    # Physik v2 (Plan 3a): Flag + Embodiment. Default False/None ⇒ v1 byte-gleich.
    physics_v2: bool = False
    body: object = None
    hands: object = None
    # Angeborene Grund-Taxis (A2) — KLASSEN-Konfiguration (ClassVar, kein
    # dataclass-Feld). Default AUS ⇒ Verhalten byte-gleich zum bisherigen
    # v2-/v1-Pfad (primitive_move). Der Harness (scripts/m1_pilot.py --taxis)
    # setzt diese Klassen-Attribute vor dem Sim-Bau. Die Schwellen sind zur
    # Laufzeit über `self.<attr>` lesbar (Sweep-Knobs).
    taxis_enabled: ClassVar[bool] = False
    taxis_hunger_threshold: ClassVar[float] = TAXIS_HUNGER_THRESHOLD
    mate_seek_radius: ClassVar[int] = MATE_SEEK_RADIUS

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
        """Per-tick neighbor snapshot, lazily computed by the first consumer.

        update() invalidates the snapshot at the start of every tick (both the
        v1 and the physics_v2 branch, since the reset sits before the branch);
        all consumers (social_learning, economy.maybe_trade, the ToM loop) run
        after primitive_move, so they share one post-move snapshot per tick.
        """
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

    def innate_locomotion(self, world, agents):
        """Angeborene Grund-Taxis (A2): EIN gerichteter Grundschritt/Tick.

        Ersetzt im physics_v2-Pfad die Welt-Wirkung des Move-Kopfs. Rein,
        RNG-frei und deterministisch — liest WEDER die gesampelte Move-Aktion
        (Dims 0/1) NOCH zieht sie einen Zufallswert (Null-Bias-Prämisse §2/A2).
        Priorität survival-first: hungrig -> Nahrungs-Taxis; sonst paarungsbereit
        -> Partner-Taxis; sonst Ruhe. Gleiche Klemm-/`passable`-Physik wie
        `primitive_move`, ein Schritt (dx,dy ∈ {-1,0,+1}), kein Teleport.
        """
        if self.is_sleeping:
            return
        target = None
        if self.energy < self.taxis_hunger_threshold:
            target = self._nearest_food_cell(world)
        elif self.can_reproduce():
            target = self._nearest_compatible_mate(agents)
            if target is None:
                # Allee-Rettung (Populations-Kollaps-Fix). Ohne Suchverhalten
                # friert ein paarungsbereiter Agent OHNE Partner im
                # MATE_SEEK_RADIUS an Ort und Stelle ein — die Move-Kopf-Dims
                # sind welt-inert, also ist innate_locomotion die EINZIGE
                # Lokomotion. Zwei verstreute Fruchtbare (Abstand > 8) fanden
                # sich bei geringer Dichte nie -> births_cum gefror, Population
                # blieb bei 2 gepinnt. Deterministischer Rendezvous-Grundtrieb
                # (Lek-/Sammelplatz-Analogon): driftet zum Weltzentrum, bis der
                # Partner in Reichweite kommt. Die EIGENTLICHE Paarung bleibt
                # perzeptions-/ko-lokations-gebunden (MATE_SEEK_RADIUS=8 und die
                # ±5-Konzeptionsbox in _try_reproduce sind unverändert) — KEIN
                # globaler Konzeptionsradius, KEIN Teleport, RNG-frei.
                target = self._mate_rendezvous_cell(world)
        if target is None:
            return  # gesättigt / kein Gradient im Radius -> Ruhe (kein Schritt)
        x, y = self.pos
        dx = (target[0] > x) - (target[0] < x)  # sign, ∈ {-1,0,+1}
        dy = (target[1] > y) - (target[1] < y)
        nx = max(0, min(world.width - 1, x + dx))
        ny = max(0, min(world.height - 1, y + dy))
        if world.get_cell(nx, ny).get("passable", True):
            self.pos = (nx, ny)

    def _nearest_food_cell(self, world):
        """Zelle mit max `food` (>0) im per-Agent-Wahrnehmungsradius.

        Tie-Breaks deterministisch: höchstes `food`; bei Gleichstand geringste
        Chebyshev-Distanz; dann lexikografisch `(x, y)`. Ist die beste Zelle die
        Standzelle -> None (bleiben & foragen). Kein Zufall.
        """
        x, y = self.pos
        r = max(1, int(self.genes["sense_radius"]))
        best_key = None
        best_pos = None
        for tx in range(x - r, x + r + 1):
            for ty in range(y - r, y + r + 1):
                if not world.in_bounds(tx, ty):
                    continue
                food = world.get_cell(tx, ty)["food"]
                if food <= 0:
                    continue
                dist = max(abs(tx - x), abs(ty - y))
                key = (-food, dist, tx, ty)  # min -> max food, min dist, lexi (x,y)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (tx, ty)
        if best_pos is None or best_pos == (x, y):
            return None
        return best_pos

    def _nearest_compatible_mate(self, agents):
        """Nächster kompatibler fruchtbarer Gegen-Sex-Partner in MATE_SEEK_RADIUS.

        Kompatibilität spiegelt `_try_reproduce`: lebendig, anderes Geschlecht,
        `can_reproduce()`, Trust >= -0.2. Tie-Breaks: geringste Chebyshev-Distanz;
        dann kleinste `agent.id`. Iteration über die ordnungsstabile `agents`-Liste,
        kein Zufall.
        """
        x, y = self.pos
        r = self.mate_seek_radius
        best_key = None
        best_pos = None
        for a in agents:
            if a is self or not a.alive or a.sex == self.sex:
                continue
            if not a.can_reproduce():
                continue
            if self.trust.get(a.id, 0.0) < -0.2:
                continue
            ax, ay = a.pos
            dist = max(abs(ax - x), abs(ay - y))
            if dist > r:
                continue
            key = (dist, a.id)
            if best_key is None or key < best_key:
                best_key = key
                best_pos = (ax, ay)
        if best_pos is None or best_pos == (x, y):
            return None
        return best_pos

    def _mate_rendezvous_cell(self, world):
        """Deterministischer Sammelpunkt für die Partnersuche (Allee-Rettung).

        Feuert nur als Fallback, wenn ein paarungsbereiter Agent KEINEN Partner
        im MATE_SEEK_RADIUS wahrnimmt. Ziel ist die Weltmitte — ein fixer,
        rein geometrischer Treffpunkt, sodass verstreute Fruchtbare deterministisch
        zusammenfinden (Brut-/Sammelplatz-Trieb), ohne dass ein spezifischer
        Partner ‚global gesehen' werden müsste. Steht der Agent schon in der Mitte
        -> None (Warten, kein Schritt). Kein Zufall, ko-lokation/Konzeption
        (±5-Box) unverändert.
        """
        cx = world.width // 2
        cy = world.height // 2
        if (cx, cy) == self.pos:
            return None
        return (cx, cy)

    def _forage(self, world, mods):
        x, y = self.pos
        cell = world.get_cell(x, y)
        gain = 0.0
        tool_bonus = SHARP_STONE_FORAGE_BONUS if self.tool == "sharp_stone" else 0.0
        home_bonus = get_home_forage_bonus(self, world)
        eff = mods.get("forage_eff", 1.0) * (1.0 + tool_bonus + home_bonus)
        diet = self.genes.get("diet_preference", 0.0)
        # Energy conservation (Phase 4): consumption debits the cell's *source*
        # pools (plant_food / meat_food / carcasses) through apply_consumption /
        # the World façade, not the derived `food` aggregate, which regrow_cell
        # recomputes from the sources every tick -- so debiting `food` directly was
        # wiped out next tick and effectively minted energy. The energy gained now
        # equals the food removed (1:1), a genuine transfer from the world.
        if diet < 0 or self.physics_v2:
            # Herbivore (v1) bzw. Physik-v2-Modus: nur Pflanzen-Zell-Foraging.
            # v2 (B6): Zell-Fleisch/Aas-Pools sind AUS — Fleisch existiert nur
            # noch als Kadaver-OBJEKT (B3); ein einziger Pfad je Kalorienquelle.
            # Herbivore: plant-food only.
            plant_available = cell.get("plant_food", 0.0)
            if plant_available > 0:
                take = min(plant_available, PLANT_ENERGY * eff)
                apply_consumption(world, x, y, plant=take)
                self.energy = min(MAX_ENERGY, self.energy + take)
                self.plant_eaten += 1
                gain += take
                self.endocrine.apply_substance("plant_food", take / PLANT_ENERGY)
                self.endocrine.apply_successful_forage(take)
            # v2-Kalorien-Fix (Option A): innater roher Handbiss aus einem
            # ko-lokierten Kadaver-Objekt am selben automatischen forage-Skalar.
            # Stellt die v1-Fleisch-Zugänglichkeit (Tod→Nahrung) wieder her, ohne
            # den Brain-Aktionsraum zu berühren, massenerhaltend über do_eat.
            # G1: STRIKT auf physics_v2 gegatet — der umgebende Zweig dient auch
            #     v1-Herbivoren (diet<0, physics_v2=False), die kein body/hands
            #     haben und deren Golden byte-identisch bleiben muss.
            # G2: nur bei Hunger (energy < MAX) — sonst mahlt do_eat das knappe,
            #     nicht-nachwachsende Kadaver-Objekt sinnlos in den eaten-Ledger.
            if self.physics_v2 and self.energy < MAX_ENERGY:
                layer = world.objects
                for obj in layer.objects_at((x, y)):
                    # C3: innater Biss STRIKT auf `carcass` (v1-Aas-Ersatz, toxin-
                    # frei). `raw_meat` ist Produkt GELERNTER Zerlegung + trägt Toxin,
                    # das der gelernte eat-Verb zahlt — es MUSS durch diesen Pfad, sonst
                    # dominiert der toxin-freie innate Biss ihn und verflacht den
                    # learn>nolearn-Gradient des M1-A/B.
                    if obj.kind not in ("carcass",):
                        continue
                    result = do_eat(self.body, self.hands, layer, self.pos, obj)
                    if not result.ok:
                        continue
                    # G3: Energie AUSSCHLIESSLICH über den do_eat-Rückgabewert
                    #     (eine Quelle der Wahrheit für kcal↔Energie, keine zweite
                    #     Rechnung → keine Doppelgutschrift).
                    # BEWUSST KEIN Toxin-health_delta: der innate Gnaw-Floor
                    #     spiegelt die v1-Aas-Zugänglichkeit (Zellpool gab Energie
                    #     OHNE Toxin-Strafe). Ein health_delta hier vergiftet jeden
                    #     auf einem Kadaver campenden Agenten langsam (gemessen
                    #     −2620 Health/1500 Ticks) → Demografie-Kollaps (mean_age
                    #     61 statt 138). Die Toxin-Kopplung bleibt dem GELERNTEN
                    #     verkörperten eat-Verb vorbehalten (agent.py:1118-1123).
                    if result.energy_delta_sim:
                        gained = result.energy_delta_sim
                        self.energy = max(0.0, min(MAX_ENERGY, self.energy + gained))
                        self.meat_eaten += 1
                        gain += gained
                        self.endocrine.apply_substance("raw_meat", gained / MEAT_ENERGY)
                        self.endocrine.apply_successful_forage(gained)
                    break  # ein Biss pro Tick, wie der verkörperte eat-Pfad
        else:
            # Carnivore/omnivore: carcasses first (now correctly keyed on the
            # stored `carcasses` field, not the never-present `carcass`), then the
            # meat source pool, then fall back to plants.
            carcass = cell.get("carcasses", 0.0)
            meat_available = cell.get("meat_food", 0.0)
            plant_available = cell.get("plant_food", 0.0)
            if carcass > 0:
                take = min(carcass, MEAT_ENERGY * eff)
                world.set_cell(x, y, "carcasses", clamp(carcass - take, 0.0, 140.0))
                self.energy = min(MAX_ENERGY, self.energy + take)
                self.meat_eaten += 1
                gain += take
                self.endocrine.apply_substance("raw_meat", take / MEAT_ENERGY)
                self.endocrine.apply_successful_forage(take)
            elif meat_available > 0:
                take = min(meat_available, MEAT_ENERGY * eff)
                apply_consumption(world, x, y, meat=take)
                self.energy = min(MAX_ENERGY, self.energy + take)
                self.meat_eaten += 1
                gain += take
                self.endocrine.apply_substance("raw_meat", take / MEAT_ENERGY)
                self.endocrine.apply_successful_forage(take)
            elif plant_available > 0:
                take = min(plant_available, PLANT_ENERGY * eff)
                apply_consumption(world, x, y, plant=take)
                self.energy = min(MAX_ENERGY, self.energy + take)
                self.plant_eaten += 1
                gain += take
                self.endocrine.apply_substance("plant_food", take / PLANT_ENERGY)
                self.endocrine.apply_successful_forage(take)
        water_available = cell.get("water", 0.0)
        if water_available > 0 and self.hydration < 100.0:
            take = min(water_available, 8.0 * eff)
            world.set_cell(x, y, "water", max(0.0, water_available - take))
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
        if collect_herb(world, x, y, herb):
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
            if self.physics_v2:
                # v2 (Spec B3): Energie aus einem Kill gibt es AUSSCHLIESSLICH
                # über den Kadaver — Loot wäre Energie ohne Massen-Gegenwert
                # (Doppel-Münzung, für den Ledger unsichtbar).
                return 0.0
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
        # Redistributive group-foraging bonus (Phase 4): instead of minting
        # `forage_bonus` energy from nothing, the better-off nearby members pool a
        # little energy for the active cooperator. Zero-sum: self gains exactly what
        # the donors actually give (the MAX_ENERGY clamp can only lose energy, never
        # create it).
        forage_bonus = min(COOP_FORAGE_MAX_BONUS, len(nearby) * COOP_FORAGE_BONUS_PER_MEMBER)
        donors = [a for a in nearby if a.energy > self.energy]
        if forage_bonus > 0.0 and donors:
            per = forage_bonus / len(donors)
            pooled = 0.0
            for donor in donors:
                contrib = min(per, donor.energy)
                donor.energy -= contrib
                pooled += contrib
            self.energy = min(MAX_ENERGY, self.energy + pooled)
            reward += pooled * 0.1
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

    def _local_food_per_capita(self, world, agents):
        """Local food standing stock shared over the mouths already nearby.

        Sums ``food`` over the (2R+1)^2 cell box around the mother and divides by
        the neighbour count (+1 for the mother herself). Reuses the per-tick
        neighbour snapshot at the shared radius 2 so it costs no extra scan.
        """
        x, y = self.pos
        r = REPRODUCTION_SENSE_RADIUS
        total_food = 0.0
        for cx in range(x - r, x + r + 1):
            for cy in range(y - r, y + r + 1):
                if world.in_bounds(cx, cy):
                    total_food += world.get_cell(cx, cy)["food"]
        nearby = len(self._nearby_cached(agents, r))
        return total_food / (nearby + 1.0)

    def _local_renewal_per_capita(self, world, agents):
        """Sustainable plant RENEWAL (flux) shared over the mouths already nearby.

        Demografie-Stabilisierung C2 (Option A). Sums the realized-regrowth EMA
        ``plant_renewal_ema`` over the (2R+1)^2 cell box around the mother and
        divides by the neighbour count (+1 for the mother herself). Unlike
        ``_local_food_per_capita`` (which sums the momentary standing stock and is
        fooled by the non-renewable t0 buffer), this reads the *sustainable inflow*
        and is suppressed under crowding via the regrow ``stress_factor`` channel.
        Reuses the shared radius-2 neighbour snapshot so it costs no extra scan.
        """
        x, y = self.pos
        r = REPRODUCTION_SENSE_RADIUS
        total_renewal = 0.0
        for cx in range(x - r, x + r + 1):
            for cy in range(y - r, y + r + 1):
                if world.in_bounds(cx, cy):
                    total_renewal += world.get_cell(cx, cy)["plant_renewal_ema"]
        nearby = len(self._nearby_cached(agents, r))
        return total_renewal / (nearby + 1.0)

    def _try_reproduce(self, world, agents):
        if not self.can_reproduce() or self.sex != "f":
            return None
        # C2 (KRITISCH, physics_v2-gegated): v2 hängt die Fertilität an die
        # nachhaltige Erneuerung pro Mund; v1 behält das momentane-Bestand-Gate
        # BYTE-FÜR-BYTE (Golden-Schutz) und liest plant_renewal_ema NIE.
        if self.physics_v2:
            if (
                self._local_renewal_per_capita(world, agents)
                < REPRODUCTION_MIN_RENEWAL_PER_CAPITA
            ):
                # The renewable supply here can't sustain another mouth — hold off.
                return None
        else:
            if self._local_food_per_capita(world, agents) < REPRODUCTION_MIN_FOOD_PER_CAPITA:
                # The ground here can't feed another mouth right now — hold off.
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

    def _resolve_causal_pending(self, view) -> None:
        """C4: löst das Causal-Pending des Vortricks gegen die aktuelle
        Wahrnehmung auf — dasselbe Objekt (id-basiert) einen Tick später.
        Aus der Wahrnehmung gefallen ⇒ maskiert, kein Loss, Epistemik 0."""
        self._causal_epistemic_pending = 0.0
        pending = getattr(self, "_causal_pending", None)
        if pending is None:
            return
        self._causal_pending = None
        slot = resolve_slot_of(view, pending["target"])
        if slot < 0:
            return
        out = self.causal_model.observe(
            pending["h"],
            pending["act"],
            pending["embed"],
            torch.as_tensor(view.feats[slot]),
        )
        self._causal_epistemic_pending = out["epistemic"]

    def _execute_embodied(self, world, brain_step, view):
        """C2/B4: Mapping des aktiven Verbs auf do_grasp/do_release/do_strike/
        do_cut/do_eat — EINE verkörperte Aktion pro Tick, an der Position der
        Wahrnehmung (vor primitive_move). ActionResult-Deltas gehen auf
        energy/health; D4-Metriken (Verb-Raten, DiscoveryV2 je Agent) werden
        an der ObjectLayer gepflegt. Kein Reward hier (C3 zahlt Physiologie)."""
        self._causal_next_target = None
        verb = brain_step["verb"]
        if verb is None:
            return None
        layer = world.objects
        metrics = layer.metrics
        target_idx = brain_step["target_idx"]
        tool_idx = brain_step["tool_idx"]
        target = view.objs[target_idx] if target_idx >= 0 else None
        tool = view.objs[tool_idx] if tool_idx >= 0 else None
        if target is None:
            noop = metrics.setdefault("verbs_noop", {})
            noop[verb] = noop.get(verb, 0) + 1
            return None
        effort = brain_step["effort"]

        if verb == "grasp":
            result = do_grasp(self.body, self.hands, layer, self.pos, target)
        elif verb == "release":
            result = do_release(self.body, self.hands, layer, self.pos, target)
        elif verb == "strike":
            if tool is None:
                return None  # act_v2 verhindert das; defensiv trotzdem No-op
            n_discovery = len(layer.discovery.entries)
            result = do_strike(self.body, self.hands, layer, self.pos, tool, target, effort, random)
        elif verb == "cut":
            n_discovery = len(layer.discovery.entries)
            result = do_cut(self.body, self.hands, layer, self.pos, tool, target, effort)
        elif verb == "eat":
            result = do_eat(self.body, self.hands, layer, self.pos, target)
        else:
            return None

        if result.energy_delta_sim:
            self.energy = max(0.0, min(MAX_ENERGY, self.energy + result.energy_delta_sim))
        if result.health_delta:
            self.health = max(0.0, self.health + result.health_delta)
            if self.health <= 0:
                self.alive = False  # Toxin-Tod: Terminal via remove_dead (Task 15)

        # D4 (Review F4): verbs_fired zählt Ausführungs-VERSUCHE — auch ok=False
        # (z. B. target_out_of_reach nach Überlast-Drop). verbs_failed zählt die
        # ok=False-Teilmenge separat: die Pilot-Diagnostik braucht Versuchsrate
        # UND Erfolgsrate der Policy (Erfolge = fired − failed).
        fired = metrics.setdefault("verbs_fired", {})
        fired[verb] = fired.get(verb, 0) + 1
        if not result.ok:
            failed = metrics.setdefault("verbs_failed", {})
            failed[verb] = failed.get(verb, 0) + 1
        if verb in ("strike", "cut"):
            gewachsen = len(layer.discovery.entries) - n_discovery
            if gewachsen:
                je_agent = metrics.setdefault("discovery_events_by_agent", {})
                je_agent[self.id] = je_agent.get(self.id, 0) + gewachsen

        # C4: das Causal-Target folgt der physischen Fortsetzung des Objekts —
        # strike-Zerstörung: massereichstes Fragment; cut: remainder.
        naechstes = target
        if verb == "strike" and result.fragments:
            naechstes = max(result.fragments, key=lambda f: f.mass)
        elif verb == "cut" and result.remainder is not None:
            naechstes = result.remainder
        self._causal_next_target = naechstes
        return result

    def _assemble_curiosity_v2(self, brain_step, next_features_raw, next_view, world):
        """C4: curiosity = 0.25·nextslot_err + 0.50·causal_epistemic +
        0.25·novelty, geclampt [0, 2]. Loggt die Zerlegung (D4: drei Quellen
        separat) und legt das Causal-Pending für den nächsten Tick an."""
        nextslot_err = self.brain.nextslot_error(
            brain_step, next_features_raw, next_view.feats, next_view.mask
        )
        causal_epistemic = self._causal_epistemic_pending
        novelty = self._novelty_buckets.observe_view(next_view)
        curiosity = max(
            0.0, min(2.0, 0.25 * nextslot_err + 0.50 * causal_epistemic + 0.25 * novelty)
        )
        self.curiosity_last = {
            "nextslot": nextslot_err,
            "causal": causal_epistemic,
            "novelty": novelty,
        }
        sums = world.objects.metrics.setdefault(
            "curiosity_sums", {"nextslot": 0.0, "causal": 0.0, "novelty": 0.0}
        )
        sums["nextslot"] += nextslot_err
        sums["causal"] += causal_epistemic
        sums["novelty"] += novelty
        target = getattr(self, "_causal_next_target", None)
        if target is not None and brain_step["target_idx"] >= 0:
            # C4: Input = detach(gru_h) ⊕ verkörperter Aktions-Vektor (22)
            # ⊕ detach(Slot-Embed des gewählten Ziels); Target = dasselbe
            # Objekt im nächsten Tick (Auflösung: _resolve_causal_pending).
            self._causal_pending = {
                "h": brain_step["next_hidden"],
                "act": brain_step["action_tensor"].squeeze(0)[V1_HEAD_DIMS:],
                "embed": brain_step["slot_embeds"][brain_step["target_idx"]],
                "target": target,
            }
        return curiosity

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
        # One neighbor snapshot per tick: drop last tick's snapshot; the first
        # consumer after primitive_move recomputes it (see _nearby_cached). This
        # reset sits BEFORE the physics_v2 branch, so it invalidates the snapshot
        # in both the v1 and the v2 path.
        self._cached_nearby_agents = None

        if self.physics_v2:
            # C3: ΔE/ΔH über den GANZEN Tick (Metabolik, Arbeit, Essen, Toxin,
            # Schlaf-Regeneration, Koop-Transfers — alles Physiologie).
            e_start, h_start = self.energy, self.health

        self.endocrine.update(self, world)
        mods = self.endocrine.modifiers()
        stage = get_stage_stats(self.age)

        self._age_tick()
        self._disease_tick(world)
        if not self.alive:
            return None

        self._sleep_tick(mods)

        if self.physics_v2 and self.body is not None:
            # Körper-Mechanik pro Tick (B4): Tragen ermüdet, Ruhe erholt;
            # Überlast (Ermüdung senkt die Kapazität) wirft zu Tick-Beginn das
            # jeweils schwerste gehaltene Objekt ab. Rein mechanisch, kein Reward.
            if self.hands.held:
                self.body.carry_tick(self.hands.carried_mass_kg())
            else:
                self.body.rest_tick()
            enforce_carry_budget(self.body, self.hands, world.objects, self.pos)

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

        features = self.local_features(world, agents)
        if self.hidden_state is None:
            self.hidden_state = self.brain.initial_hidden()

        if self.physics_v2:
            # v2 (C1/C2/C5): Objekt-Slots wahrnehmen, Causal-Pending des
            # Vortricks auflösen, dann Policy-Sampling OHNE Planner —
            # plan_action/imagine_rollout sind mit der v2-Architektur
            # inkompatibel (encoder(pred_next_obs) ohne obj_ctx; argmax).
            view = build_slots(self, world.objects)
            self._resolve_causal_pending(view)
            brain_step = self.brain.act_v2(
                features, self.hidden_state, view.feats, view.mask, admissible_masks(view)
            )
        else:
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

        if not self.physics_v2 and getattr(self, "goal_stack", None) is not None:
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

        if self.physics_v2:
            self._causal_next_target = None  # pro Tick frisch (auch im Schlaf)
            if not self.is_sleeping:
                # EINE verkörperte Aktion pro Tick (B4), ausgeführt an der
                # Wahrnehmungs-Position (vor primitive_move — die gesampelten
                # Zulässigkeits-Masken bleiben konsistent zur Ausführung).
                self._execute_embodied(world, brain_step, view)

        if self.physics_v2 and self.taxis_enabled:
            # Angeborene Grund-Taxis (A2): der gerichtete Grundtrieb ersetzt im
            # v2-Pilot die Welt-Wirkung des Move-Kopfs. Die gesampelte Move-Aktion
            # (Dims 0/1) bleibt physisch im action_tensor & im PPO-Buffer (on-
            # policy), wird aber verhaltens-inert — store_transition_v2 (brain.py)
            # bleibt unberührt. RNG-frei/deterministisch. v1-Pfad + taxis-AUS
            # nutzen weiter primitive_move (Verhalten byte-gleich, Golden grün).
            self.innate_locomotion(world, agents)
        else:
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
            self._try_reproduce(world, agents)
        child_genes = self.progress_pregnancy()

        if tick % 3 == 0:
            reward += social_learning_step(self, agents, tick)

        if not self.physics_v2:
            # v2 (B6): v1-Erfindung AUS — beide Trigger-Pfade entfallen mitsamt
            # ihren Boni; Entdecken läuft künftig über die Objekt-Physik (3b).
            if self._need_inv_cooldown <= 0:
                compute_need_vector(self, current_cell)
                inv_result = agent_invent_from_need(self, world, *self.pos, tick)
                if inv_result:
                    reward += 0.5
                    self.endocrine.apply_discovery(1.0)
                self._need_inv_cooldown = NEED_INVENTION_INTERVAL
            else:
                self._need_inv_cooldown -= 1

        inv_prob = INVENTION_BASE_PROB + INVENTION_CURIOSITY_MULT * self.genes.get("curiosity", 0.5)
        if not self.physics_v2 and tick % 3 == 0 and random.random() < inv_prob:
            invented = agent_try_invention(self, world, *self.pos)
            if invented:
                reward += 1.0
                self.endocrine.apply_discovery(1.0)

        if not self.physics_v2 and tick % 4 == 0 and random.random() < 0.18:
            cooked = agent_try_cook(self, world, *self.pos)
            if cooked:
                reward += 0.3
                self.endocrine.apply_substance("cooked_meat", 1.0)

        if economy is not None:
            economy.maybe_trade(self, agents)

        next_features_raw = self.local_features(world, agents)
        if not self.physics_v2:
            # v1-Intrinsic (inkl. rew_err-Mechanik) und NGU-Episodic laufen NUR
            # im v1 — der v2 ersetzt beides durch die C4-Neugier (rew_err ist
            # ersatzlos gestrichen; Planner/NGU sind im v2 aus).
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
        if self.physics_v2:
            # C3: nur Überleben + Neugier. Der bis hier akkumulierte v1-Reward
            # (Forage-Event, Koop, Attack, Territorium, Sprache, Social
            # Learning, Trade) wird bewusst VERWORFEN — die Mechanik lief, sie
            # zahlt nur nicht (B6). Keine cognition-Skalierung: C3 exakt.
            # r_death (−3.0) kommt NICHT hier, sondern genau einmal in
            # Brain.finalize_terminal am Todes-Aggregationspunkt remove_dead.
            next_view = build_slots(self, world.objects)
            curiosity = self._assemble_curiosity_v2(brain_step, next_features_raw, next_view, world)
            reward = (
                0.6 * (self.energy - e_start) / 45.0
                + 0.6 * (self.health - h_start) / 50.0
                - 0.3 * max(0.0, (60.0 - self.energy) / 60.0)
                + 0.3 * curiosity
            )
            effective_reward = reward
        else:
            effective_reward = reward * cognition_mult
        if self.physics_v2:
            self.brain.store_transition_v2(
                brain_step,
                effective_reward,
                not self.alive,
                next_features_raw,
                next_view.feats,
                next_view.mask,
            )
        else:
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

        # M-1 (Final-Review): im v2-Pfad NICHT trainieren, wenn der Agent in
        # diesem Tick gestorben ist. Sonst kann maybe_train einen exakt
        # gefüllten 128er-Buffer VOR remove_dead/finalize_terminal flushen —
        # der Todes-Malus (-3.0) haette dann keinen Buffer mehr zum Anhängen
        # und verfiele still. finalize_terminal (via remove_dead) übernimmt
        # Flush + Malus + Training für gestorbene v2-Agenten selbst. Der
        # v1-Pfad bleibt unveraendert (kein finalize_terminal-Aequivalent).
        loss = None if self.physics_v2 and not self.alive else self.brain.maybe_train()
        if loss is not None:
            self.last_loss = loss

        self.last_reward = effective_reward
        self.reproduction_cooldown = max(0, self.reproduction_cooldown - 1)

        for other in self._nearby_cached(agents, 2):
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
