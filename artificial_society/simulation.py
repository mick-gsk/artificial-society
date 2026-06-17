import random
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

# Events only start after this tick to let agents stabilise first
EVENT_WARMUP_TICKS = 600
EVENT_SPAWN_RATE = 0.003


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
        pygame.display.set_caption('Artificial Society v3.0 - Dynamic Learning World')
        self.renderer = Renderer(width, height, self.cell_px)
        self.font = pygame.font.SysFont('consolas', 16)
        self.running = True
        self.tick = 0
        self.agents = []
        self.spawn_initial_population(initial_population)

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

    def remove_dead(self):
        survivors = []
        for agent in self.agents:
            if agent.alive:
                survivors.append(agent)
                continue
            add_carcass(self.world.get_cell(*agent.pos), CORPSE_ENERGY)
        self.agents = survivors

    def step(self):
        self.tick += 1
        season_state = self.seasons.update(self.tick)
        weather_state = self.weather.update(self.world, season_state, self.tick)
        # Suppress disturbance events during warmup period
        if self.tick < EVENT_WARMUP_TICKS:
            self.world.active_events = [e for e in self.world.active_events if e.get('type') not in ('drought', 'fire', 'blight', 'storm')]
        elif random.random() < EVENT_SPAWN_RATE:
            self.world.maybe_spawn_event(self.tick)
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
        self.remove_dead()
        self.tribes.cleanup(self.agents)
        self.technology.update(self.agents, self.tribes)
        self.economy.update(self.agents, self.tribes)
        self.stats.update(self.tick, self.agents, self.world, self.tribes, self.technology)

    def draw(self):
        self.renderer.draw(self.screen, self.world, self.agents, self.stats, self.tribes, self.technology)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.step()
            self.draw()
            self.clock.tick(30)
