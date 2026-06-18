import os
import random
import pickle
import pygame

import artificial_society.runtime_patches  # noqa: F401

from artificial_society.world import World
from artificial_society.renderer import Renderer
from artificial_society.agents.agent import Agent, CORPSE_ENERGY, MAX_ENERGY
from artificial_society.agents.brain import Brain, INPUT_SIZE
from artificial_society.agents.endocrine import EndocrineSystem
from artificial_society.environment.resources import add_carcass
from artificial_society.environment.seasons import SeasonCycle
from artificial_society.environment.weather import WeatherSystem
from artificial_society.environment.daynight import day_phase
from artificial_society.environment.territory import update_territory_claims
from artificial_society.systems.tribes import TribeSystem
from artificial_society.systems.economy import EconomySystem
from artificial_society.systems.technology import TechnologySystem
from artificial_society.systems.evolution import EvolutionSystem
from artificial_society.visualization.statistics import StatisticsTracker
from artificial_society.systems.remedy import try_infect_agent, REMEDY_REGISTRY
from artificial_society.systems.invention import tick_materials, seed_world_materials, share_discovery
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

HAMILTON_CHILD_R       = 0.50
HAMILTON_TRIBE_R       = 0.10
HAMILTON_REWARD_SCALE  = 0.08
HAMILTON_TICK_INTERVAL = 20

# --- SCHICHT 3: Soziales Wissens-Bootstrapping ---
SOCIAL_SHARING_INTERVAL      = 15
SOCIAL_SHARING_RADIUS        = 2
SOCIAL_SHARING_ENERGY_REWARD = 1.5

# --- SCHICHT 4: Generationen-Wissenstransfer ---
DEATH_MATERIAL_TRANSFER_FIDELITY = 0.45
DEATH_MATERIAL_MIN_QTY           = 0.3
DEATH_MATERIAL_TRANSFER_RATIO    = 0.4

SIDEBAR_W = 300


def _migrate_agent(agent):
    if not hasattr(agent, 'brain') or agent.brain is None:
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    elif getattr(agent.brain, 'input_size', None) != INPUT_SIZE:
        agent.brain = Brain()
        agent.hidden_state = agent.brain.initial_hidden()
    if not hasattr(agent, 'hidden_state') or agent.hidden_state is None:
        agent.hidden_state = agent.brain.initial_hidden()
    if not hasattr(agent, 'endocrine') or agent.endocrine is None:
        agent.endocrine = EndocrineSystem()
    if not hasattr(agent, 'causal_memory') or agent.causal_memory is None:
        agent.causal_memory = CausalMemory(capacity=32)
    if not hasattr(agent, 'material_inventory') or agent.material_inventory is None:
        agent.material_inventory = {}
    if not hasattr(agent, 'is_sleeping'):
        agent.is_sleeping = False
    if not hasattr(agent, 'remedy_knowledge'):
        agent.remedy_knowledge = {}
    if not hasattr(agent, 'herbs_carried'):
        agent.herbs_carried = {}
    if not hasattr(agent, '_last_mate_id'):
        agent._last_mate_id = None


class Simulation:
    def __init__(self, width=1200, height=800, grid_w=60, grid_h=40, initial_population=36):
        self.width    = width
        self.height   = height
        self.grid_w   = grid_w
        self.grid_h   = grid_h
        self.cell_px  = min((width - SIDEBAR_W) // grid_w, height // grid_h)
        self.world    = World(grid_w, grid_h)
        self.seasons  = SeasonCycle()
        self.weather  = WeatherSystem()
        self.tribes   = TribeSystem()
        self.economy  = EconomySystem()
        self.technology = TechnologySystem()
        self.evolution  = EvolutionSystem()
        self.stats      = StatisticsTracker()
        # Fenster explizit mit voller Breite (Spielfeld + Sidebar)
        self.screen = pygame.display.set_mode((width, height), 0, 32)
        self.clock  = pygame.time.Clock()
        pygame.display.set_caption('Artificial Society v3.5')
        self.renderer = Renderer(width, height, self.cell_px)
        self.font     = pygame.font.SysFont('consolas', 16)
        self.running  = True
        self.tick     = 0
        self.agents   = []
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
        other_parent = None
        agent_by_id  = {a.id: a for a in self.agents if a.alive}
        mate_id = getattr(parent, '_last_mate_id', None)
        if mate_id is not None:
            other_parent = agent_by_id.get(mate_id)
        child = self.evolution.make_child(parent, x, y, genes=genes, other_parent=other_parent)
        child.hidden_state = child.brain.initial_hidden()
        child.birth_tick   = self.tick
        inherit_strength = max(0.20, min(0.75, 0.75 - (child.genes['plasticity'] - 0.3) / (1.8 - 0.3) * 0.55))
        child.brain.inherit_weights_from(parent.brain, strength=inherit_strength)
        if other_parent is not None:
            child.brain.inherit_weights_from(other_parent.brain, strength=inherit_strength * 0.4)
        parent_mem = getattr(parent, 'causal_memory', None)
        if parent_mem:
            child.causal_memory = CausalMemory(capacity=32)
            for seq in list(parent_mem.sequences.keys())[:INHERIT_SEQUENCES]:
                child.causal_memory.receive_transmitted(seq, fidelity=INHERIT_FIDELITY)
        if parent.remedy_knowledge:
            for disease, herbs in parent.remedy_knowledge.items():
                if random.random() < INHERIT_FIDELITY:
                    child.remedy_knowledge[disease] = list(herbs)
        parent_inv = getattr(parent, 'material_inventory', {})
        parent_discoveries = {
            m: q for m, q in parent_inv.items()
            if m.startswith('mat_') and q >= DEATH_MATERIAL_MIN_QTY
        }
        if parent_discoveries:
            child_inv = getattr(child, 'material_inventory', {})
            for mat_id, qty in parent_discoveries.items():
                if random.random() < INHERIT_FIDELITY:
                    child_inv[mat_id] = child_inv.get(mat_id, 0.0) + qty * DEATH_MATERIAL_TRANSFER_RATIO
            child.material_inventory = child_inv
        return child

    def _broadcast_death_knowledge(self, agent):
        causal_mem  = getattr(agent, 'causal_memory', None)
        inv         = getattr(agent, 'material_inventory', {})
        discoveries = {
            m: q for m, q in inv.items()
            if m.startswith('mat_') and q >= DEATH_MATERIAL_MIN_QTY
        }
        if causal_mem is None and not agent.remedy_knowledge and not discoveries:
            return
        ax, ay = agent.pos
        recipients = [
            a for a in self.agents
            if a is not agent and a.alive
            and abs(a.pos[0]-ax) <= DEATH_BROADCAST_RADIUS
            and abs(a.pos[1]-ay) <= DEATH_BROADCAST_RADIUS
            and (a.tribe_id == agent.tribe_id or a.parent_id == agent.id or agent.parent_id == a.id)
        ]
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
            r_inv = getattr(recipient, 'material_inventory', {})
            for mat_id, qty in discoveries.items():
                if random.random() < DEATH_MATERIAL_TRANSFER_FIDELITY:
                    r_inv[mat_id] = r_inv.get(mat_id, 0.0) + qty * DEATH_MATERIAL_TRANSFER_RATIO
            recipient.material_inventory = r_inv

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

    def _is_immune(self, agent, disease_id):
        return self.tick < getattr(agent, '_disease_immunity', {}).get(disease_id, 0)

    def _grant_immunity(self, agent, disease_id):
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
        infectious = [a for a in self.agents if a.alive and getattr(a, 'disease_id', None) is not None]
        for carrier in infectious:
            disease_id = carrier.disease_id
            rec = REMEDY_REGISTRY.get(disease_id, {})
            if rec.get('spread_rate', 0) == 0:
                continue
            radius = 2 if rec.get('vector') == 'airborne' else 1
            cx, cy = carrier.pos
            biome   = self.world.get_biome(cx, cy)
            for a in self.agents:
                if (a is not carrier and a.alive
                        and abs(a.pos[0]-cx) <= radius
                        and abs(a.pos[1]-cy) <= radius
                        and getattr(a, 'disease_id', None) is None
                        and not self._is_immune(a, disease_id)):
                    try_infect_agent(a, disease_id, biome=biome)

    def _apply_hamilton_rewards(self):
        if self.tick % HAMILTON_TICK_INTERVAL != 0:
            return
        tribe_members = {}
        for a in self.agents:
            if a.alive and a.tribe_id is not None:
                tribe_members.setdefault(a.tribe_id, []).append(a)
        for agent in self.agents:
            if not agent.alive:
                continue
            tribe = tribe_members.get(agent.tribe_id, [])
            self_reward = 0.0
            if tribe:
                self_reward = sum(1.0 for m in tribe if m is not agent) * HAMILTON_TRIBE_R
            child_r = getattr(agent, 'children', 0) * HAMILTON_CHILD_R
            agent.energy = min(MAX_ENERGY, agent.energy + (self_reward + child_r) * HAMILTON_REWARD_SCALE)

    def _save_checkpoint(self):
        try:
            with open(CHECKPOINT_PATH, 'wb') as f:
                pickle.dump({
                    'agents': self.agents,
                    'tick':   self.tick,
                    'world':  self.world,
                    'stats':  self.stats,
                    'tribes': self.tribes,
                    'technology': self.technology,
                }, f)
        except Exception as e:
            print(f'[checkpoint] save failed: {e}')

    def _load_checkpoint(self):
        try:
            with open(CHECKPOINT_PATH, 'rb') as f:
                data = pickle.load(f)
            self.agents     = data.get('agents', [])
            self.tick       = data.get('tick', 0)
            self.world      = data.get('world', self.world)
            self.stats      = data.get('stats', self.stats)
            self.tribes     = data.get('tribes', self.tribes)
            self.technology = data.get('technology', self.technology)
            for agent in self.agents:
                _migrate_agent(agent)
            print(f'[checkpoint] loaded tick={self.tick}, agents={len(self.agents)}')
        except Exception as e:
            print(f'[checkpoint] load failed: {e} — starting fresh')
            self.spawn_initial_population(36)

    def _collect_stats(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return
        from artificial_society.agents.life_stage import STAGE_CHILD, STAGE_ADULT, STAGE_ELDER
        n_child = sum(1 for a in alive if getattr(a, 'life_stage', STAGE_ADULT) == STAGE_CHILD)
        n_adult = sum(1 for a in alive if getattr(a, 'life_stage', STAGE_ADULT) == STAGE_ADULT)
        n_elder = sum(1 for a in alive if getattr(a, 'life_stage', STAGE_ADULT) == STAGE_ELDER)
        avg_age  = sum(self.tick - getattr(a, 'birth_tick', self.tick) for a in alive) / len(alive)
        avg_hyd  = sum(getattr(a, 'hydration',  50.0) for a in alive) / len(alive)
        avg_sick = sum(getattr(a, 'sick',        0.0)  for a in alive) / len(alive)
        avg_rew  = sum(getattr(a, 'last_reward', 0.0)  for a in alive) / len(alive)
        n_preg   = sum(1 for a in alive if getattr(a, 'pregnant', False))
        n_tribes = len({a.tribe_id for a in alive if a.tribe_id is not None})
        food_vals  = [self.world.get_cell(x, y)['food']  for y in range(self.world.height) for x in range(self.world.width)]
        water_vals = [self.world.get_cell(x, y)['water'] for y in range(self.world.height) for x in range(self.world.width)]
        dis_vals   = [self.world.get_cell(x, y)['disease']     for y in range(self.world.height) for x in range(self.world.width)]
        pol_vals   = [self.world.get_cell(x, y)['pollution']   for y in range(self.world.height) for x in range(self.world.width)]
        dist_vals  = [self.world.get_cell(x, y)['disturbance'] for y in range(self.world.height) for x in range(self.world.width)]
        self.stats.record({
            'population':       len(alive),
            'n_child':          n_child,
            'n_adult':          n_adult,
            'n_elder':          n_elder,
            'average_age':      avg_age,
            'avg_hydration':    avg_hyd,
            'avg_sick':         avg_sick,
            'avg_reward':       avg_rew,
            'pregnant':         n_preg,
            'tribes':           n_tribes,
            'active_events':    0,
            'world_food':       sum(food_vals)  / max(1, len(food_vals)),
            'world_water':      sum(water_vals) / max(1, len(water_vals)),
            'world_disease':    sum(dis_vals)   / max(1, len(dis_vals)),
            'world_pollution':  sum(pol_vals)   / max(1, len(pol_vals)),
            'world_disturbance':sum(dist_vals)  / max(1, len(dist_vals)),
        })

    def step(self):
        self.tick += 1
        season = self.seasons.current_season(self.tick)
        weather = self.weather.current(self.tick)
        phase   = day_phase(self.tick)

        # Agenten updaten
        new_children = []
        for agent in self.agents:
            if not agent.alive:
                continue
            result = agent.update(
                self.world, self.agents,
                season=season, weather=weather,
                day_phase=phase,
                tribes=self.tribes,
                economy=self.economy,
                technology=self.technology,
                tick=self.tick,
            )
            if result and result.get('birth'):
                child = self.spawn_child_from_parent(agent, result.get('child_genes'))
                new_children.append(child)
            if result and result.get('discovery'):
                share_discovery(agent, self.agents, result['discovery'])

        if self.tick % 100 == 0:

            alive = [a for a in self.agents if a.alive]

            repro_ready = sum(
                1 for a in alive
                if hasattr(a, "can_reproduce") and a.can_reproduce()
            )

            pregnant = sum(
                1 for a in alive
                if getattr(a, "pregnant", False)
            )

            print(
                f"[REPRO]",
                f"tick={self.tick}",
                f"alive={len(alive)}",
                f"ready={repro_ready}",
                f"pregnant={pregnant}",
            )

        self.agents.extend(new_children)

        # Welt-Systeme
        tick_materials(self.world)
        self.spread_diseases()
        self.tick_immunity_and_recovery()
        self._apply_hamilton_rewards()

        if self.tick % SOCIAL_SHARING_INTERVAL == 0:
            try:
                from artificial_society.systems.social_learning import social_learning_step
                social_learning_step(self.agents, self.world, radius=SOCIAL_SHARING_RADIUS,
                                     energy_reward=SOCIAL_SHARING_ENERGY_REWARD)
            except Exception:
                pass

        if self.tick % 10 == 0:
            update_territory_claims(self.agents, self.world)

        self.remove_dead()

        if len(self.agents) < MIN_POPULATION:
            self.emergency_respawn()

        if self.tick % 5 == 0:
            self._collect_stats()

        if self.tick % CHECKPOINT_INTERVAL == 0:
            self._save_checkpoint()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_s:
                        self._save_checkpoint()
                        print('[checkpoint] manually saved')
                    elif event.key == pygame.K_DELETE:
                        if os.path.exists(CHECKPOINT_PATH):
                            os.remove(CHECKPOINT_PATH)
                            print('[checkpoint] deleted')

            self.step()
            self.renderer.draw(
                self.screen,
                self.world,
                self.agents,
                self.stats,
                self.tribes,
                self.technology,
            )
            self.clock.tick(30)
