import os
import pickle
import random

import pygame

import artificial_society.systems._builtins  # noqa: F401  (registers built-in systems)
from artificial_society.agents.agent import CORPSE_ENERGY, MAX_ENERGY, Agent, ensure_fields
from artificial_society.environment.materials import DISCOVERY_REGISTRY
from artificial_society.environment.resources import add_carcass
from artificial_society.environment.territory import update_territory_claims
from artificial_society.renderer import Renderer
from artificial_society.rng import seed_all
from artificial_society.systems import registry
from artificial_society.systems.culture import CausalMemory
from artificial_society.systems.goal_stack_ext import RECIPE_DISCOVERY, SEQUENCE_LIBRARY
from artificial_society.systems.invention import (
    seed_world_materials,
    tick_materials,
)
from artificial_society.systems.language import TOKEN_WORLD
from artificial_society.systems.remedy import REMEDY_REGISTRY, try_infect_agent
from artificial_society.world import World

EVENT_WARMUP_TICKS = 600
MIN_POPULATION = 8
RESPAWN_COUNT = 6
CHECKPOINT_INTERVAL = 500
CHECKPOINT_PATH = "checkpoint.pkl"

IMMUNITY_WINDOW_DEFAULT = 200

INHERIT_SEQUENCES = 10
INHERIT_FIDELITY = 0.70
DEATH_BROADCAST_FIDELITY = 0.45
DEATH_BROADCAST_RADIUS = 4

HAMILTON_CHILD_R = 0.50
HAMILTON_TRIBE_R = 0.10
HAMILTON_REWARD_SCALE = 0.08
HAMILTON_TICK_INTERVAL = 20

# --- SCHICHT 3: Soziales Wissens-Bootstrapping ---
SOCIAL_SHARING_INTERVAL = 15
SOCIAL_SHARING_RADIUS = 2
SOCIAL_SHARING_ENERGY_REWARD = 1.5

# --- SCHICHT 4: Generationen-Wissenstransfer ---
DEATH_MATERIAL_TRANSFER_FIDELITY = 0.45
DEATH_MATERIAL_MIN_QTY = 0.3
DEATH_MATERIAL_TRANSFER_RATIO = 0.4

SIDEBAR_W = 300


def _reset_accumulating_singletons() -> None:
    """Reset every global registry that accumulates across in-process runs.

    A freshly-constructed Simulation must be independent of any prior one in the
    same process (reproducibility). This is the single source of truth for which
    globals accumulate — add new accumulating singletons here, never inline in
    ``__init__``.
    """
    DISCOVERY_REGISTRY.reset()
    TOKEN_WORLD.reset()
    SEQUENCE_LIBRARY.reset()
    RECIPE_DISCOVERY.reset()


def _capture_singletons() -> dict:
    """Snapshot every accumulating singleton for a full, reproducible checkpoint.

    Restores the emergent state a save->load must not lose: discovered materials,
    language tokens, and recipe outcomes. SEQUENCE_LIBRARY is reset-only (its default
    sequences hold unpicklable lambdas) — see SequenceLibrary.reset.
    """
    return {
        "discovery": DISCOVERY_REGISTRY.state_dict(),
        "tokens": TOKEN_WORLD.state_dict(),
        "recipes": RECIPE_DISCOVERY.state_dict(),
    }


def _restore_singletons(data: dict) -> None:
    """Restore singleton snapshots produced by _capture_singletons.

    Missing keys are left at their freshly-reset state, so a pre-Phase-3 checkpoint
    (no "registries" block) still loads cleanly.
    """
    if not data:
        return
    if "discovery" in data:
        DISCOVERY_REGISTRY.load_state_dict(data["discovery"])
    if "tokens" in data:
        TOKEN_WORLD.load_state_dict(data["tokens"])
    if "recipes" in data:
        RECIPE_DISCOVERY.load_state_dict(data["recipes"])


class Simulation:
    def __init__(
        self,
        width=1200,
        height=800,
        grid_w=60,
        grid_h=40,
        initial_population=36,
        headless=False,
        seed=None,
        load_checkpoint=True,
    ):
        # Seed first, before anything stochastic (biome grid, population) is built.
        self.headless = headless
        self.seed = seed
        if seed is not None:
            seed_all(seed)
            # Reset the agent id sequence so a seed reproduces the same ids too.
            Agent.id_counter = 0
        # Each simulation starts with fresh emergent-discovery / language / sequence
        # state, so a run is reproducible and independent of any prior simulation in
        # this process. See _reset_accumulating_singletons for the full list.
        _reset_accumulating_singletons()
        self.width = width
        self.height = height
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_px = min((width - SIDEBAR_W) // grid_w, height // grid_h)
        self.world = World(grid_w, grid_h)
        # Systems are built from the registry (systems/registry.py) instead of being
        # hard-coded here, so adding a system is one new file and never edits this
        # method. Each becomes sim.<name> (e.g. sim.tribes) and an entry in
        # sim.systems. Built-ins are registered dormant, preserving today's loop.
        self.systems = registry.build_systems(self)
        if headless:
            # No window, font, or renderer in headless mode (tests / batch runs).
            self.screen = None
            self.clock = None
            self.renderer = None
            self.font = None
        else:
            # Fenster explizit mit voller Breite (Spielfeld + Sidebar)
            self.screen = pygame.display.set_mode((width, height), 0, 32)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Artificial Society v3.5")
            self.renderer = Renderer(width, height, self.cell_px)
            self.font = pygame.font.SysFont("consolas", 16)
        self.running = True
        self.tick = 0
        self.agents = []
        self._initial_population = initial_population
        if load_checkpoint and os.path.exists(CHECKPOINT_PATH):
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
        agent_by_id = {a.id: a for a in self.agents if a.alive}
        mate_id = getattr(parent, "_last_mate_id", None)
        if mate_id is not None:
            other_parent = agent_by_id.get(mate_id)
        child = self.evolution.make_child(parent, x, y, genes=genes, other_parent=other_parent)
        child.hidden_state = child.brain.initial_hidden()
        child.birth_tick = self.tick
        inherit_strength = max(
            0.20, min(0.75, 0.75 - (child.genes["plasticity"] - 0.3) / (1.8 - 0.3) * 0.55)
        )
        child.brain.inherit_weights_from(parent.brain, strength=inherit_strength)
        if other_parent is not None:
            child.brain.inherit_weights_from(other_parent.brain, strength=inherit_strength * 0.4)
        parent_mem = getattr(parent, "causal_memory", None)
        if parent_mem:
            child.causal_memory = CausalMemory(capacity=32)
            for seq in list(parent_mem.sequences.keys())[:INHERIT_SEQUENCES]:
                child.causal_memory.receive_transmitted(seq, fidelity=INHERIT_FIDELITY)
        if parent.remedy_knowledge:
            for disease, herbs in parent.remedy_knowledge.items():
                if random.random() < INHERIT_FIDELITY:
                    child.remedy_knowledge[disease] = list(herbs)
        parent_inv = getattr(parent, "material_inventory", {})
        parent_discoveries = {
            m: q
            for m, q in parent_inv.items()
            if m.startswith("mat_") and q >= DEATH_MATERIAL_MIN_QTY
        }
        if parent_discoveries:
            child_inv = getattr(child, "material_inventory", {})
            for mat_id, qty in parent_discoveries.items():
                if random.random() < INHERIT_FIDELITY:
                    child_inv[mat_id] = (
                        child_inv.get(mat_id, 0.0) + qty * DEATH_MATERIAL_TRANSFER_RATIO
                    )
            child.material_inventory = child_inv
        return child

    def _broadcast_death_knowledge(self, agent):
        causal_mem = getattr(agent, "causal_memory", None)
        inv = getattr(agent, "material_inventory", {})
        discoveries = {
            m: q for m, q in inv.items() if m.startswith("mat_") and q >= DEATH_MATERIAL_MIN_QTY
        }
        if causal_mem is None and not agent.remedy_knowledge and not discoveries:
            return
        ax, ay = agent.pos
        recipients = [
            a
            for a in self.agents
            if a is not agent
            and a.alive
            and abs(a.pos[0] - ax) <= DEATH_BROADCAST_RADIUS
            and abs(a.pos[1] - ay) <= DEATH_BROADCAST_RADIUS
            and (a.tribe_id == agent.tribe_id or a.parent_id == agent.id or agent.parent_id == a.id)
        ]
        seqs = list(causal_mem.sequences.keys()) if causal_mem else []
        for recipient in recipients:
            if not hasattr(recipient, "causal_memory") or recipient.causal_memory is None:
                recipient.causal_memory = CausalMemory(capacity=32)
            for seq in seqs:
                if random.random() < DEATH_BROADCAST_FIDELITY:
                    recipient.causal_memory.receive_transmitted(
                        seq, fidelity=DEATH_BROADCAST_FIDELITY
                    )
            for disease, herbs in agent.remedy_knowledge.items():
                if (
                    disease not in recipient.remedy_knowledge
                    and random.random() < DEATH_BROADCAST_FIDELITY
                ):
                    recipient.remedy_knowledge[disease] = list(herbs)
            r_inv = getattr(recipient, "material_inventory", {})
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
            add_carcass(self.world, *agent.pos, CORPSE_ENERGY)
        self.agents = survivors

    def _is_immune(self, agent, disease_id):
        return self.tick < getattr(agent, "_disease_immunity", {}).get(disease_id, 0)

    def _grant_immunity(self, agent, disease_id):
        if not hasattr(agent, "_disease_immunity"):
            agent._disease_immunity = {}
        window = REMEDY_REGISTRY.get(disease_id, {}).get("immunity_after", IMMUNITY_WINDOW_DEFAULT)
        agent._disease_immunity[disease_id] = self.tick + window

    def tick_immunity_and_recovery(self):
        for agent in self.agents:
            if not agent.alive:
                continue
            prev = getattr(agent, "_prev_disease_id", None)
            curr = getattr(agent, "disease_id", None)
            if prev is not None and curr is None:
                self._grant_immunity(agent, prev)
            agent._prev_disease_id = curr

    def spread_diseases(self):
        infectious = [
            a for a in self.agents if a.alive and getattr(a, "disease_id", None) is not None
        ]
        for carrier in infectious:
            disease_id = carrier.disease_id
            rec = REMEDY_REGISTRY.get(disease_id, {})
            if rec.get("spread_rate", 0) == 0:
                continue
            radius = 2 if rec.get("vector") == "airborne" else 1
            cx, cy = carrier.pos
            biome = self.world.get_biome(cx, cy)
            for a in self.agents:
                if (
                    a is not carrier
                    and a.alive
                    and abs(a.pos[0] - cx) <= radius
                    and abs(a.pos[1] - cy) <= radius
                    and getattr(a, "disease_id", None) is None
                    and not self._is_immune(a, disease_id)
                ):
                    try_infect_agent(a, disease_id, biome=biome)

    def _apply_hamilton_rewards(self):
        # Kin selection, made energy-conservative (Phase 4): instead of minting a
        # per-agent reward from nothing, each tribe taxes a small fraction of every
        # member's energy into a pool and redistributes it weighted by kin
        # investment (tribe ties + number of children). Energy only ever moves
        # *within* a tribe, so total energy is conserved (the MAX_ENERGY clamp can
        # only sink energy, never create it).
        if self.tick % HAMILTON_TICK_INTERVAL != 0:
            return
        tribe_members = {}
        for a in self.agents:
            if a.alive and a.tribe_id is not None:
                tribe_members.setdefault(a.tribe_id, []).append(a)
        for members in tribe_members.values():
            if len(members) < 2:
                continue
            pool = 0.0
            for m in members:
                # Clamp the taxable amount to be non-negative: a member that has
                # somehow gone negative must contribute 0, never be silently lifted
                # to 0 (which would mint energy and break the zero-sum invariant).
                # For energy >= 0 this is identical to the old min(...) form.
                tax = max(0.0, m.energy) * HAMILTON_REWARD_SCALE
                m.energy -= tax
                pool += tax
            if pool <= 0.0:
                continue
            weights = [
                (len(members) - 1) * HAMILTON_TRIBE_R + getattr(m, "children", 0) * HAMILTON_CHILD_R
                for m in members
            ]
            total_w = sum(weights)
            if total_w <= 0.0:
                share = pool / len(members)
                for m in members:
                    m.energy = min(MAX_ENERGY, m.energy + share)
            else:
                for m, w in zip(members, weights):
                    m.energy = min(MAX_ENERGY, m.energy + pool * (w / total_w))

    def _save_checkpoint(self):
        try:
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump(
                    {
                        "agents": self.agents,
                        "tick": self.tick,
                        "world": self.world,
                        "stats": self.stats,
                        "tribes": self.tribes,
                        "technology": self.technology,
                        # Full emergent state so a save->load is reproducible.
                        "registries": _capture_singletons(),
                    },
                    f,
                )
        except Exception as e:
            print(f"[checkpoint] save failed: {e}")

    def _load_checkpoint(self):
        try:
            with open(CHECKPOINT_PATH, "rb") as f:
                data = pickle.load(f)
            self.agents = data.get("agents", [])
            self.tick = data.get("tick", 0)
            self.world = data.get("world", self.world)
            self.stats = data.get("stats", self.stats)
            self.tribes = data.get("tribes", self.tribes)
            self.technology = data.get("technology", self.technology)
            # Restore the global emergent registries; absent in pre-Phase-3 saves,
            # in which case the freshly-reset singletons are kept.
            _restore_singletons(data.get("registries", {}))
            for agent in self.agents:
                ensure_fields(agent)
            print(f"[checkpoint] loaded tick={self.tick}, agents={len(self.agents)}")
        except Exception as e:
            print(f"[checkpoint] load failed: {e} — starting fresh")
            self.spawn_initial_population(self._initial_population)

    def _collect_stats(self):
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return
        from artificial_society.agents.life_stage import STAGE_ADULT, STAGE_CHILD, STAGE_ELDER

        n_child = sum(1 for a in alive if getattr(a, "life_stage", STAGE_ADULT) == STAGE_CHILD)
        n_adult = sum(1 for a in alive if getattr(a, "life_stage", STAGE_ADULT) == STAGE_ADULT)
        n_elder = sum(1 for a in alive if getattr(a, "life_stage", STAGE_ADULT) == STAGE_ELDER)
        avg_age = sum(self.tick - getattr(a, "birth_tick", self.tick) for a in alive) / len(alive)
        avg_hyd = sum(getattr(a, "hydration", 50.0) for a in alive) / len(alive)
        avg_sick = sum(getattr(a, "sick", 0.0) for a in alive) / len(alive)
        avg_rew = sum(getattr(a, "last_reward", 0.0) for a in alive) / len(alive)
        n_preg = sum(1 for a in alive if getattr(a, "pregnant", False))
        n_tribes = len({a.tribe_id for a in alive if a.tribe_id is not None})
        food_vals = [
            self.world.get_cell(x, y)["food"]
            for y in range(self.world.height)
            for x in range(self.world.width)
        ]
        water_vals = [
            self.world.get_cell(x, y)["water"]
            for y in range(self.world.height)
            for x in range(self.world.width)
        ]
        dis_vals = [
            self.world.get_cell(x, y)["disease"]
            for y in range(self.world.height)
            for x in range(self.world.width)
        ]
        pol_vals = [
            self.world.get_cell(x, y)["pollution"]
            for y in range(self.world.height)
            for x in range(self.world.width)
        ]
        dist_vals = [
            self.world.get_cell(x, y)["disturbance"]
            for y in range(self.world.height)
            for x in range(self.world.width)
        ]
        self.stats.record(
            {
                "population": len(alive),
                "n_child": n_child,
                "n_adult": n_adult,
                "n_elder": n_elder,
                "average_age": avg_age,
                "avg_hydration": avg_hyd,
                "avg_sick": avg_sick,
                "avg_reward": avg_rew,
                "pregnant": n_preg,
                "tribes": n_tribes,
                "active_events": 0,
                "world_food": sum(food_vals) / max(1, len(food_vals)),
                "world_water": sum(water_vals) / max(1, len(water_vals)),
                "world_disease": sum(dis_vals) / max(1, len(dis_vals)),
                "world_pollution": sum(pol_vals) / max(1, len(pol_vals)),
                "world_disturbance": sum(dist_vals) / max(1, len(dist_vals)),
            }
        )

    def step(self):
        """Advance the simulation by exactly one tick.

        Single, explicit per-tick pipeline. It reproduces only the operations
        that are *effective* in today's live loop, so moving off the
        bootstrap/monkeypatch loop is behaviour-preserving. Operations the old
        live loop silently dropped — world regrowth, births, disease spread,
        society systems, statistics — are re-wired in Phase 2 and marked
        TODO(phase2) below.
        """
        tick = self.tick

        # --- world systems effective today ---
        update_territory_claims(self.world, self.agents)
        tick_materials(self.world)
        # TODO(phase2): self.world.update_environment(season_state, weather_state, tick)

        # --- per-agent update ---
        # A non-None return from Agent.update is a completed pregnancy: spawn the
        # child with full genetic / brain / knowledge inheritance (spawn_child_from_parent).
        new_children = []
        for agent in list(self.agents):
            if not agent.alive:
                continue
            child_genes = agent.update(
                self.world,
                self.agents,
                tick,
                season_state=None,
                weather_state=None,
                tribes=self.tribes,
                economy=self.economy,
                technology=self.technology,
            )
            if child_genes is not None:
                new_children.append(self.spawn_child_from_parent(agent, child_genes))
        self.agents.extend(new_children)

        # Drop agents that died during their own update. As in the current live
        # loop, pre-filtering means remove_dead() finds no bodies (no carcass or
        # death-knowledge broadcast yet).
        # TODO(phase2): route deaths through remove_dead() for carcass + broadcast.
        self.agents = [a for a in self.agents if a.alive]
        self.remove_dead()

        self.tick_immunity_and_recovery()
        self._apply_hamilton_rewards()

        # Registered systems with a tick hook run here in ascending `order`. The
        # built-ins are dormant (tick=None) so this is a no-op today — it is the
        # seam a newly-added system ticks through without editing step(). Re-wiring
        # the dormant built-ins (disease spread, society systems, stats) is the
        # separate, intentional Phase 2 work.
        registry.tick_systems(self, tick)

        if len(self.agents) < MIN_POPULATION:
            self.emergency_respawn()

        if CHECKPOINT_INTERVAL and tick > 0 and tick % CHECKPOINT_INTERVAL == 0:
            self._save_checkpoint()

        self.tick = tick + 1

    def run(self, max_ticks=None):
        """Drive the simulation loop.

        Headless when the simulation was constructed with ``headless=True``.
        ``max_ticks`` bounds the number of ticks (tests / batch runs); ``None``
        runs until the window is closed (GUI mode only).
        """
        self.running = True
        n = 0
        while self.running:
            if not self.headless:
                self._handle_events()

            self.step()

            if not self.headless and self.renderer is not None:
                self.renderer.draw(
                    self.screen,
                    self.world,
                    self.agents,
                    self.stats,
                    self.tribes,
                    self.technology,
                )
                if self.clock is not None:
                    self.clock.tick(30)

            n += 1
            if max_ticks is not None and n >= max_ticks:
                break

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_s:
                    self._save_checkpoint()
                    print("[checkpoint] manually saved")
                elif event.key == pygame.K_DELETE and os.path.exists(CHECKPOINT_PATH):
                    os.remove(CHECKPOINT_PATH)
                    print("[checkpoint] deleted")
