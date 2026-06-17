import os
import random
import pickle
import pygame

from artificial_society.world import World
from artificial_society.renderer import Renderer
from artificial_society.agents.agent import Agent, CORPSE_ENERGY
from artificial_society.agents.brain import Brain, INPUT_SIZE
from artificial_society.agents.endocrine import EndocrineSystem
from artificial_society.environment.resources import add_carcass
from artificial_society.environment.seasons import SeasonCycle
from artificial_society.environment.weather import WeatherSystem
from artificial_society.environment.daynight import day_phase
from artificial_society.systems.tribes import TribeSystem
from artificial_society.systems.economy import EconomySystem
from artificial_society.systems.technology import TechnologySystem
from artificial_society.systems.evolution import EvolutionSystem
from artificial_society.visualization.statistics import StatisticsTracker
from artificial_society.systems.remedy import try_infect_agent, REMEDY_REGISTRY
from artificial_society.systems.invention import tick_materials, seed_world_materials
from artificial_society.systems.culture import CausalMemory

EVENT_WARMUP_TICKS = 600
MIN_POPULATION = 8
RESPAWN_COUNT = 6
CHECKPOINT_INTERVAL = 500
CHECKPOINT_PATH = 'checkpoint.pkl'

IMMUNITY_WINDOW_DEFAULT = 200

INHERIT_SEQUENCES = 10
INHERIT_FIDELITY = 0.70
DEATH_BROADCAST_FIDELITY = 0.45
DEATH_BROADCAST_RADIUS = 4


def _migrate_agent(agent):
    """
    Bring a pickled agent up to the current code version.
    Called for every agent after checkpoint load.
    """
    # Brain: rebuild if missing or wrong input size
    if not hasattr(agent, 'brain') or agent.brain is None:
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    elif getattr(agent.brain, 'input_size', None) != INPUT_SIZE:
        print(f'[migrate] agent {agent.id}: brain input_size '
              f'{agent.brain.input_size} -> {INPUT_SIZE}, rebuilding')
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()

    if not hasattr(agent, 'hidden_state') or agent.hidden_state is None:
        agent.hidden_state = agent.brain.initial_hidden()

    # Endocrine: inject if missing (old checkpoint)
    if not hasattr(agent, 'endocrine') or agent.endocrine is None:
        agent.endocrine = EndocrineSystem()

    # CausalMemory
    if not hasattr(agent, 'causal_memory') or agent.causal_memory is None:
        agent.causal_memory = CausalMemory(capacity=32)

    # Material inventory
    if not hasattr(agent, 'material_inventory') or agent.material_inventory is None:
        agent.material_inventory = {}

    # Sleep flag
    if not hasattr(agent, 'is_sleeping'):
        agent.is_sleeping = False

    # Remedy knowledge
    if not hasattr(agent, 'remedy_knowledge'):
        agent.remedy_knowledge = {}

    # Herbs carried
    if not hasattr(agent, 'herbs_carried'):
        agent.herbs_carried = {}


class Simulation:
    def __init__(self, width=1200, height=800, grid_w=60, grid_h=40, initial_population=36):
        self.width = width
        self.height = height
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_px = min((width - 300) // grid_w, height // grid_h)
        self.world = World(grid_w, grid_h)
        self.seasons = SeasonCycle()
        self.weather = WeatherSystem()
        self.tribes = TribeSystem()
        self.economy = EconomySystem()
        self.technology = TechnologySystem()
        self.evolution = EvolutionSystem()
        self.stats = StatisticsTracker()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Artificial Society v3.2')
        self.renderer = Renderer(width, height, self.cell_px)
        self.font = pygame.font.SysFont('consolas', 16)
        self.running = True
        self.tick = 0
        self.agents = []
        if os.path.exists(CHECKPOINT_PATH):
            self._load_checkpoint()
        else:
            self.spawn_initial_population(initial_population)
        seed_world_materials(self.world)

    def spawn_initial_population(self, n):
        for _ in range(n):
            x, y = self.world.random_land_position()
            self.agents.append(Agent.spawn_random(x, y))

    def spawn_child_from_parent(self, parent, genes):
        x, y = self.world.find_free_neighbor(parent.pos)
        if x is None:
            x, y = parent.pos
        child = self.evolution.make_child(parent, x, y, genes=genes)
        child.hidden_state = child.brain.initial_hidden()
        child.birth_tick = self.tick
        parent_mem = getattr(parent, 'causal_memory', None)
        if parent_mem:
            child.causal_memory = CausalMemory(capacity=32)
            seqs = list(parent_mem.sequences.keys())[:INHERIT_SEQUENCES]
            for seq in seqs:
                child.causal_memory.receive_transmitted(seq, fidelity=INHERIT_FIDELITY)
        if parent.remedy_knowledge:
            for disease, herbs in parent.remedy_knowledge.items():
                if random.random() < INHERIT_FIDELITY:
                    child.remedy_knowledge[disease] = list(herbs)
        return child

    def _broadcast_death_knowledge(self, agent):
        causal_mem = getattr(agent, 'causal_memory', None)
        if causal_mem is None and not agent.remedy_knowledge:
            return
        ax, ay = agent.pos
        recipients = [
            a for a in self.agents
            if a is not agent
            and a.alive
            and abs(a.pos[0] - ax) <= DEATH_BROADCAST_RADIUS
            and abs(a.pos[1] - ay) <= DEATH_BROADCAST_RADIUS
            and (a.tribe_id == agent.tribe_id or a.parent_id == agent.id or agent.parent_id == a.id)
        ]
        if not recipients:
            return
        seqs = list(causal_mem.sequences.keys()) if causal_mem else []
        for recipient in recipients:
            if not hasattr(recipient, 'causal_memory') or recipient.causal_memory is None:
                recipient.causal_memory = CausalMemory(capacity=32)
            for seq in seqs:
                if random.random() < DEATH_BROADCAST_FIDELITY:
                    recipient.causal_memory.receive_transmitted(seq, fidelity=DEATH_BROADCAST_FIDELITY)
            for disease, herbs in agent.remedy_knowledge.items():
                if disease not in recipient.remedy_knowledge and random.random() < DEATH_BROADCAST_FIDELITY:
                    recipient.remedy_knowledge[disease] = list(herbs)

    def emergency_respawn(self):
        for _ in range(RESPAWN_COUNT):
            x, y = self.world.random_land_position()
            a = Agent.spawn_random(x, y)
            a.birth_tick = self.tick
            self.agents.append(a)

    def remove_dead(self):
        survivors = []
        for agent in self.agents:
            if agent.alive:
                survivors.append(agent)
                continue
            self._broadcast_death_knowledge(agent)
            add_carcass(self.world.get_cell(*agent.pos), CORPSE_ENERGY)
        self.agents = survivors

    def _is_immune(self, agent, disease_id: str) -> bool:
        immunity_map = getattr(agent, '_disease_immunity', {})
        return self.tick < immunity_map.get(disease_id, 0)

    def _grant_immunity(self, agent, disease_id: str):
        if not hasattr(agent, '_disease_immunity'):
            agent._disease_immunity = {}
        window = REMEDY_REGISTRY.get(disease_id, {}).get('immunity_after', IMMUNITY_WINDOW_DEFAULT)
        agent._disease_immunity[disease_id] = self.tick + window

    def tick_immunity_and_recovery(self):
        for agent in self.agents:
            if not agent.alive:
                continue
            prev = getattr(agent, '_prev_disease_id', None)
            curr = getattr(agent, 'disease_id', None)
            if prev is not None and curr is None:
                self._grant_immunity(agent, prev)
            agent._prev_disease_id = curr

    def spread_diseases(self):
        infectious = [
            a for a in self.agents
            if a.alive and getattr(a, 'disease_id', None) is not None
        ]
        for carrier in infectious:
            disease_id = carrier.disease_id
            rec = REMEDY_REGISTRY.get(disease_id, {})
            if rec.get('spread_rate', 0) == 0:
                continue
            radius = 2 if rec.get('vector') == 'airborne' else 1
            cx, cy = carrier.pos
            biome = self.world.get_biome(cx, cy)
            neighbours = [
                a for a in self.agents
                if a is not carrier
                and a.alive
                and abs(a.pos[0] - cx) <= radius
                and abs(a.pos[1] - cy) <= radius
                and getattr(a, 'disease_id', None) is None
                and not self._is_immune(a, disease_id)
            ]
            for neighbour in neighbours:
                try_infect_agent(neighbour, disease_id, biome=biome)

    def _save_checkpoint(self):
        try:
            data = {
                'tick': self.tick,
                'agents': self.agents,
                'tribes': self.tribes,
                'technology': self.technology,
                'economy': self.economy,
            }
            with open(CHECKPOINT_PATH, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f'[checkpoint] save failed: {e}')

    def _load_checkpoint(self):
        try:
            with open(CHECKPOINT_PATH, 'rb') as f:
                data = pickle.load(f)
            self.tick = data['tick']
            self.agents = data['agents']
            self.tribes = data['tribes']
            self.technology = data['technology']
            self.economy = data['economy']
            # Migrate every agent to the current code version
            for agent in self.agents:
                _migrate_agent(agent)
            print(f'[checkpoint] loaded tick={self.tick}, agents={len(self.agents)}')
        except Exception as e:
            print(f'[checkpoint] load failed: {e}, starting fresh')
            self.spawn_initial_population(36)

    def step(self):
        self.tick += 1

        dn = day_phase(self.tick)
        self.world.day_state = dn

        season_state = self.seasons.update(self.tick)
        weather_state = self.weather.update(self.world, season_state, self.tick)
        if self.tick < EVENT_WARMUP_TICKS:
            self.world.active_events = [
                e for e in self.world.active_events
                if e.get('kind') not in ('drought', 'fire', 'blight', 'storm')
            ]
        self.world.update_environment(season_state, weather_state, self.tick)
        tick_materials(self.world)
        self.tribes.update_membership(self.agents)
        births = []
        for agent in list(self.agents):
            child_genes = agent.update(
                world=self.world,
                agents=self.agents,
                tick=self.tick,
                season_state=season_state,
                weather_state=weather_state,
                tribes=self.tribes,
                economy=self.economy,
                technology=self.technology,
            )
            if agent.alive and child_genes is not None:
                births.append(self.spawn_child_from_parent(agent, child_genes))
        self.agents.extend(births)
        self.tick_immunity_and_recovery()
        self.spread_diseases()
        self.remove_dead()
        self.tribes.cleanup(self.agents)
        self.technology.update(self.agents, self.tribes)
        self.economy.update(self.agents, self.tribes)
        self.stats.update(self.tick, self.agents, self.world, self.tribes, self.technology)
        if len(self.agents) < MIN_POPULATION:
            self.emergency_respawn()
        if self.tick % CHECKPOINT_INTERVAL == 0:
            self._save_checkpoint()

    def draw(self):
        self.renderer.draw(self.screen, self.world, self.agents, self.stats, self.tribes, self.technology)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self._save_checkpoint()
                        print('[checkpoint] manual save')
                    elif event.key == pygame.K_DELETE:
                        if os.path.exists(CHECKPOINT_PATH):
                            os.remove(CHECKPOINT_PATH)
                            print('[checkpoint] deleted')
            self.step()
            self.draw()
            self.clock.tick(30)
