import os
import random
import pickle
import pygame

from artificial_society.world import World
from artificial_society.renderer import Renderer
from artificial_society.agents.agent import Agent, CORPSE_ENERGY
from artificial_society.environment.resources import add_carcass
from artificial_society.environment.seasons import SeasonCycle
from artificial_society.environment.weather import WeatherSystem
from artificial_society.systems.tribes import TribeSystem
from artificial_society.systems.economy import EconomySystem
from artificial_society.systems.technology import TechnologySystem
from artificial_society.systems.evolution import EvolutionSystem
from artificial_society.visualization.statistics import StatisticsTracker
from artificial_society.systems.remedy import try_infect_agent

# Events only start after this tick to let agents stabilise first
EVENT_WARMUP_TICKS = 600
# Minimum alive population before emergency respawn kicks in
MIN_POPULATION = 8
# How many agents to respawn when below floor
RESPAWN_COUNT = 6
# Save checkpoint every N ticks
CHECKPOINT_INTERVAL = 500
CHECKPOINT_PATH = 'checkpoint.pkl'


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
        pygame.display.set_caption('Artificial Society v3.1')
        self.renderer = Renderer(width, height, self.cell_px)
        self.font = pygame.font.SysFont('consolas', 16)
        self.running = True
        self.tick = 0
        self.agents = []
        if os.path.exists(CHECKPOINT_PATH):
            self._load_checkpoint()
        else:
            self.spawn_initial_population(initial_population)

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

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
        return child

    def emergency_respawn(self):
        """If population drops too low, inject fresh random agents to prevent extinction."""
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
            add_carcass(self.world.get_cell(*agent.pos), CORPSE_ENERGY)
        self.agents = survivors

    # ------------------------------------------------------------------
    # Disease spreading between nearby agents
    # ------------------------------------------------------------------

    def spread_diseases(self):
        """Infected agents try to spread their disease to adjacent alive neighbours."""
        infectious = [a for a in self.agents if a.alive and getattr(a, 'disease_id', None) is not None]
        for carrier in infectious:
            cx, cy = carrier.pos
            neighbours = [
                a for a in self.agents
                if a is not carrier and a.alive
                and abs(a.pos[0] - cx) <= 1 and abs(a.pos[1] - cy) <= 1
                and getattr(a, 'disease_id', None) is None
            ]
            for neighbour in neighbours:
                try_infect_agent(neighbour, carrier.disease_id)

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

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
            print(f'[checkpoint] loaded tick={self.tick}, agents={len(self.agents)}')
        except Exception as e:
            print(f'[checkpoint] load failed: {e}, starting fresh')
            self.spawn_initial_population(36)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self):
        self.tick += 1
        season_state = self.seasons.update(self.tick)
        weather_state = self.weather.update(self.world, season_state, self.tick)
        if self.tick < EVENT_WARMUP_TICKS:
            self.world.active_events = [
                e for e in self.world.active_events
                if e.get('kind') not in ('drought', 'fire', 'blight', 'storm')
            ]
        self.world.update_environment(season_state, weather_state, self.tick)
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
        # FIX 1: spread diseases between neighbouring agents each tick
        self.spread_diseases()
        self.remove_dead()
        self.tribes.cleanup(self.agents)
        self.technology.update(self.agents, self.tribes)
        self.economy.update(self.agents, self.tribes)
        self.stats.update(self.tick, self.agents, self.world, self.tribes, self.technology)
        # FIX 3: population floor guard
        if len(self.agents) < MIN_POPULATION:
            self.emergency_respawn()
        # FIX 6: periodic checkpoint
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
