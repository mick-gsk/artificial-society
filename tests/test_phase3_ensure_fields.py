"""Phase 3 (3c) — agent field init is a single source of truth.

The three historical init paths (`_ensure_new_fields`, `_ensure_runtime_fields`,
`_migrate_agent`) are collapsed into one `ensure_fields`. These tests pin that
contract: one entry point, and it fully initialises both freshly-spawned agents and
agents deserialised from an old checkpoint (missing newer fields).
"""

from __future__ import annotations

import artificial_society.agents.agent as agent_mod
import artificial_society.simulation as sim_mod
from artificial_society.agents.agent import Agent, ensure_fields
from artificial_society.rng import seed_all

# Union of every field the three old helpers used to guarantee.
RUNTIME_FIELDS = (
    "goal_stack",
    "token_memory",
    "_next_planning_tick",
    "_planning_stride",
    "_last_goal_action",
    "_language_retry_tick",
    "_inventory_cap",
    "_cached_nearby_agents",
    "_cached_nearby_radius",
    "_disease_immunity",
)
NEW_FIELDS = (
    "_brain_device",
    "causal_memory",
    "material_inventory",
    "endocrine",
    "is_sleeping",
    "brain",
    "tool",
    "_last_mate_id",
    "_need_inv_cooldown",
    "tom",
    "knowledge",
    "emotional_memory",
    "_recent_action_seq",
)
MIGRATE_EXTRAS = ("hidden_state", "remedy_knowledge", "herbs_carried")
ALL_FIELDS = RUNTIME_FIELDS + NEW_FIELDS + MIGRATE_EXTRAS


def test_single_entry_point_old_helpers_gone():
    """`ensure_fields` is the only init path; the three old symbols are removed."""
    assert callable(agent_mod.ensure_fields)
    assert not hasattr(agent_mod, "_ensure_new_fields")
    assert not hasattr(agent_mod, "_ensure_runtime_fields")
    assert not hasattr(sim_mod, "_migrate_agent")


def test_spawned_agent_fully_initialised():
    """A freshly spawned agent has every field the old paths guaranteed."""
    seed_all(0)
    agent = Agent.spawn_random(0, 0)
    for fname in ALL_FIELDS:
        assert hasattr(agent, fname), f"spawned agent missing {fname}"


def test_ensure_fields_restores_old_checkpoint_agent():
    """An agent missing newer fields (old pickle) is fully restored by ensure_fields.

    Covers the union: fields that historically only `_ensure_runtime_fields` or only
    `_migrate_agent` set must all be reinstated through the single path.
    """
    seed_all(1)
    agent = Agent.spawn_random(0, 0)
    # Simulate a stale checkpoint: drop a representative field from each old group.
    for fname in ("goal_stack", "_disease_immunity", "_brain_device", "remedy_knowledge"):
        if hasattr(agent, fname):
            delattr(agent, fname)

    ensure_fields(agent)

    for fname in ALL_FIELDS:
        assert hasattr(agent, fname), f"ensure_fields did not restore {fname}"
