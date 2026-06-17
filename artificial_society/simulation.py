import os
import random
import pickle
import pygame

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
# Wie oft pro Tick wird social sharing aktiviert
SOCIAL_SHARING_INTERVAL      = 15   # Alle 15 Ticks
SOCIAL_SHARING_RADIUS        = 2    # Maximale Entfernung fuer Wissensaustausch
SOCIAL_SHARING_ENERGY_REWARD = 1.5  # Energie-Bonus fuers Lehren

# --- SCHICHT 4: Generationen-Wissenstransfer ---
DEATH_MATERIAL_TRANSFER_FIDELITY = 0.45
DEATH_MATERIAL_MIN_QTY           = 0.3
DEATH_MATERIAL_TRANSFER_RATIO    = 0.4


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
        self.cell_px  = min((width - 300) // grid_w, height // grid_h)
        self.world    = World(grid_w, grid_h)
        self.seasons  = SeasonCycle()
        self.weather  = WeatherSystem()
        self.tribes   = TribeSystem()
        self.economy  = EconomySystem()
        self.technology = TechnologySystem()
        self.evolution  = EvolutionSystem()
        self.stats      = StatisticsTracker()
        self.screen = pygame.display.set_mode((width, height))
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

        # --- SCHICHT 4 (Eltern -> Kind): Material-Discoveries weitervererben ---
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
        """
        Stirbt ein Agent, gibt er sein Wissen an nahestehende Stammesmitglieder
        und Verwandte weiter.

        SCHICHT 4 - Papier-Prinzip: Material-Discoveries werden jetzt ebenfalls
        uebertragen, sodass Wissen ueber Generationen akkumuliert statt verloren geht.
        """
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

            # --- NEU: Material-Discoveries weitergeben (Papier-Prinzip) ---
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
            bonus = 0.0
            for other in self.agents:
                if not other.alive:
                    continue
                if other.parent_id == agent.id or agent.parent_id == other.id:
                    fitness_b = (other.energy/MAX_ENERGY)*0.5 + (other.health/100.0)*0.5
                    bonus += HAMILTON_CHILD_R * fitness_b
            if agent.tribe_id is not None:
                for other in tribe_members.get(agent.tribe_id, []):
                    if other is agent:
                        continue
                    fitness_b = (other.energy/MAX_ENERGY)*0.5 + (other.health/100.0)*0.5
                    bonus += HAMILTON_TRIBE_R * fitness_b
            if bonus > 0 and agent.brain.rollout.storage:
                agent.brain.rollout.storage[-1]['reward'] += bonus * HAMILTON_REWARD_SCALE
                agent.last_reward += bonus * HAMILTON_REWARD_SCALE

    def _apply_social_knowledge_sharing(self):
        """
        SCHICHT 3: Soziales Wissens-Bootstrapping.
        Agenten die entdeckte Materialien besitzen, teilen ihr Wissen mit
        nahen Stammesmitgliedern. Lehren wird mit Energie belohnt, damit
        Spezialisierung sich evolutionary lohnt.

        Dadurch entsteht spontan: Einer entdeckt Feuer, gibt es weiter,
        andere bauen darauf auf und erfinden Kochen, Keramik etc.
        """
        if self.tick % SOCIAL_SHARING_INTERVAL != 0:
            return

        for agent in self.agents:
            if not agent.alive:
                continue
            inv = getattr(agent, 'material_inventory', {})
            discoveries = [m for m in inv if m.startswith('mat_') and inv[m] > 0.5]
            if not discoveries:
                continue

            ax, ay = agent.pos
            nearby = [
                a for a in self.agents
                if a is not agent and a.alive
                and abs(a.pos[0]-ax) <= SOCIAL_SHARING_RADIUS
                and abs(a.pos[1]-ay) <= SOCIAL_SHARING_RADIUS
            ]
            if not nearby:
                continue

            mat_to_share = random.choice(discoveries)
            recipient    = random.choice(nearby)
            if share_discovery(agent, recipient, mat_to_share):
                # Lehrer erhaelt Energie-Bonus -> Spezialisierung lohnt sich
                agent.energy = min(agent.energy + SOCIAL_SHARING_ENERGY_REWARD, MAX_ENERGY)

    def _save_checkpoint(self):
        try:
            data = {'tick': self.tick, 'agents': self.agents, 'tribes': self.tribes,
                    'technology': self.technology, 'economy': self.economy}
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
            for agent in self.agents:
                _migrate_agent(agent)
            print(f'[checkpoint] loaded tick={self.tick}, agents={len(self.agents)}')
        except Exception as e:
            print(f'[checkpoint] load failed: {e}, starting fresh')
            self.spawn_initial_population(36)

    def step(self):
        self.tick += 1
        dn           = day_phase(self.tick)
        self.world.day_state = dn
        season_state = self.seasons.update(self.tick)
        weather_state= self.weather.update(self.world, season_state, self.tick)
        if isinstance(season_state, dict):
            self.world.current_season = season_state.get('season', None)
        else:
            self.world.current_season = getattr(season_state, 'name', None)
        if self.tick < EVENT_WARMUP_TICKS:
            self.world.active_events = [e for e in self.world.active_events
                                         if e.get('kind') not in ('drought', 'fire', 'blight', 'storm')]
        self.world.update_environment(season_state, weather_state, self.tick)
        tick_materials(self.world)
        update_territory_claims(self.world, self.agents)
        self.tribes.update_membership(self.agents)
        births = []
        for agent in list(self.agents):
            child_genes = agent.update(
                world=self.world, agents=self.agents, tick=self.tick,
                season_state=season_state, weather_state=weather_state,
                tribes=self.tribes, economy=self.economy, technology=self.technology,
            )
            if agent.alive and child_genes is not None:
                births.append(self.spawn_child_from_parent(agent, child_genes))
        self.agents.extend(births)
        self.tick_immunity_and_recovery()
        self.spread_diseases()
        self._apply_hamilton_rewards()

        # --- SCHICHT 3: Soziales Bootstrapping ---
        self._apply_social_knowledge_sharing()

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
