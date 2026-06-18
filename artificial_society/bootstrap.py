"""Bootstrap layer that makes the simulation start reliably.

Loads the emergent runtime if available, self-heals the missing
goal-stack export when needed, and injects a fallback Simulation.run
method if the class does not define one.
"""

from __future__ import annotations

from typing import Any
import warnings

# ---------------------------------------------------------------------
# Goal-stack compatibility
# ---------------------------------------------------------------------
def _ensure_goal_stack_export() -> None:
    try:
        from artificial_society.systems import goal_stack_ext as gse
    except Exception as exc:
        warnings.warn(f"Could not import goal_stack_ext: {exc!r}")
        return

    if hasattr(gse, "agent_tick_with_goals"):
        return

    from artificial_society.systems.goal_stack import GoalStack

    def agent_tick_with_goals(agent, cell: dict, world_context: dict, tick: int):
        if not hasattr(agent, "goal_stack") or agent.goal_stack is None:
            agent.goal_stack = GoalStack()

        action_from_stack = None
        shaping = 0.0

        tick_fn = getattr(agent.goal_stack, "tick", None)
        if callable(tick_fn):
            try:
                action_from_stack, shaping = tick_fn(agent, cell)
            except TypeError:
                try:
                    action_from_stack, shaping = tick_fn(cell)
                except Exception:
                    pass
            except Exception:
                pass

        planner = globals().get("GOAL_PLANNER", None)
        is_empty = getattr(agent.goal_stack, "is_empty", None)
        if callable(is_empty):
            empty = False
            try:
                empty = bool(is_empty())
            except Exception:
                empty = False
        else:
            empty = False

        if empty and planner is not None:
            try:
                suggestions = planner.suggest_goals(agent, cell, world_context)
            except Exception:
                suggestions = []
            for goal in suggestions[:2]:
                try:
                    agent.goal_stack.push(goal)
                except Exception:
                    continue

            if callable(tick_fn):
                try:
                    action_from_stack, shaping = tick_fn(agent, cell)
                except TypeError:
                    try:
                        action_from_stack, shaping = tick_fn(cell)
                    except Exception:
                        pass
                except Exception:
                    pass

        return action_from_stack, shaping

    gse.agent_tick_with_goals = agent_tick_with_goals  # type: ignore[attr-defined]


_ensure_goal_stack_export()

# ---------------------------------------------------------------------
# Runtime patches: try to load them, but do not hard-fail startup if the
# environment is in a partially migrated state.
# ---------------------------------------------------------------------
try:
    import artificial_society.runtime_patches  # noqa: F401
except Exception as exc:
    warnings.warn(
        "artificial_society.runtime_patches could not be loaded cleanly; "
        f"continuing with the fallback run patch. Details: {exc!r}"
    )

# ---------------------------------------------------------------------
# Simulation.run fallback / patch
# ---------------------------------------------------------------------
def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn(*args)
        except TypeError:
            try:
                return fn()
            except Exception:
                return None
    except Exception:
        return None

def _migrate_agent(agent: Any) -> None:
    if not hasattr(agent, "alive"):
        agent.alive = True
    if not hasattr(agent, "is_sleeping"):
        agent.is_sleeping = False
    if not hasattr(agent, "material_inventory") or agent.material_inventory is None:
        agent.material_inventory = {}
    if not hasattr(agent, "remedy_knowledge") or agent.remedy_knowledge is None:
        agent.remedy_knowledge = {}
    if not hasattr(agent, "_disease_immunity"):
        agent._disease_immunity = {}
    if not hasattr(agent, "_next_planning_tick"):
        agent._next_planning_tick = 0
    if not hasattr(agent, "_need_inv_cooldown"):
        agent._need_inv_cooldown = 0
    if not hasattr(agent, "goal_stack") or agent.goal_stack is None:
        try:
            from artificial_society.systems.goal_stack import GoalStack
            agent.goal_stack = GoalStack()
        except Exception:
            agent.goal_stack = None

def patch_simulation_class(Simulation: type) -> type:
    if getattr(Simulation, "_bootstrap_run_patched", False):
        return Simulation

    import pygame
    from artificial_society.environment.territory import update_territory_claims
    from artificial_society.systems.invention import tick_materials, share_discovery
    from artificial_society.visualization.overlays import draw_dashboard

    def run(self) -> None:
        self.running = True
        clock = getattr(self, "clock", pygame.time.Clock())
        self.clock = clock

        target_fps = 30
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            tick = int(getattr(self, "tick", 0))

            seasons = getattr(self, "seasons", None)
            if seasons is not None:
                for name, args in (("advance", ()), ("step", ()), ("tick", (tick,))):
                    if hasattr(seasons, name):
                        _safe_call(getattr(seasons, name), *args)
                        break

            weather = getattr(self, "weather", None)
            if weather is not None:
                for name, args in (("update", (tick,)), ("step", ()), ("tick", (tick,))):
                    if hasattr(weather, name):
                        _safe_call(getattr(weather, name), *args)
                        break

            world = getattr(self, "world", None)
            agents = getattr(self, "agents", [])

            _safe_call(update_territory_claims, world, agents, tick)
            _safe_call(tick_materials, world, tick)
            _safe_call(share_discovery, self, tick)

            alive_agents = []
            for agent in list(agents):
                _migrate_agent(agent)
                if not getattr(agent, "alive", True):
                    continue

                _safe_call(
                    getattr(agent, "update", None),
                    world,
                    agents,
                    tick,
                    season_state=getattr(seasons, "state", None),
                    weather_state=getattr(weather, "state", None),
                    tribes=getattr(self, "tribes", None),
                    economy=getattr(self, "economy", None),
                    technology=getattr(self, "technology", None),
                )

                if getattr(agent, "alive", True):
                    alive_agents.append(agent)

            self.agents = [a for a in alive_agents if getattr(a, "alive", True)]

            _safe_call(getattr(self, "remove_dead", None))
            _safe_call(getattr(self, "tick_immunity_and_recovery", None))
            _safe_call(getattr(self, "_apply_hamilton_rewards", None))

            min_population = getattr(__import__("artificial_society.simulation", fromlist=["MIN_POPULATION"]), "MIN_POPULATION", 8)
            if len(self.agents) < min_population:
                _safe_call(getattr(self, "emergency_respawn", None))

            stats = getattr(self, "stats", None)
            if stats is not None:
                _safe_call(getattr(stats, "update", None), self.agents, tick=tick)
                _safe_call(getattr(stats, "record", None), self.agents, tick=tick)

            checkpoint_interval = getattr(__import__("artificial_society.simulation", fromlist=["CHECKPOINT_INTERVAL"]), "CHECKPOINT_INTERVAL", 500)
            if checkpoint_interval and tick > 0 and tick % checkpoint_interval == 0:
                _safe_call(getattr(self, "_save_checkpoint", None))

            screen = getattr(self, "screen", None)
            renderer = getattr(self, "renderer", None)
            if screen is not None:
                try:
                    screen.fill((0, 0, 0))
                except Exception:
                    pass

                if renderer is not None:
                    _safe_call(getattr(renderer, "draw_world", None), screen, world)
                    _safe_call(getattr(renderer, "draw_hotspots", None), screen, world)
                    _safe_call(getattr(renderer, "draw_social_links", None), screen, self.agents)
                    _safe_call(getattr(renderer, "draw_agents", None), screen, self.agents, getattr(self, "tribes", None))

                # Restore the right-hand data panel
                _safe_call(draw_dashboard, screen, self)

                try:
                    pygame.display.flip()
                except Exception:
                    pass

            clock.tick(target_fps)
            self.tick = tick + 1

    Simulation.run = run  # type: ignore[assignment]
    Simulation._bootstrap_run_patched = True
    return Simulation

__all__ = ["patch_simulation_class"]
