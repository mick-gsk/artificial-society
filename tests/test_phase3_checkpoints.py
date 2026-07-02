"""Phase 3: checkpoints must round-trip the full emergent state.

Before Phase 3 a checkpoint pickled only agents/world/stats/tribes/technology, so a
save->load silently dropped every discovered material and language token. These tests
lock in that the global emergent registries survive a checkpoint round-trip, and that
old checkpoints written without registry data still load (graceful back-compat).
"""

from __future__ import annotations

import dataclasses
import io
import pickle

import numpy as np
import torch

from artificial_society import simulation as sim_mod
from artificial_society.agents.agent import Agent
from artificial_society.agents.knowledge import EpisodicMemory, NoveltyMemory
from artificial_society.environment.materials import DISCOVERY_REGISTRY, N_PROPS
from artificial_society.simulation import Simulation
from artificial_society.systems.goal_stack_ext import RECIPE_DISCOVERY
from artificial_society.systems.language import TOKEN_WORLD, Token

SMALL = dict(headless=True, grid_w=12, grid_h=8, initial_population=4)


def test_checkpoint_roundtrips_emergent_registries(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(tmp_path / "ckpt.pkl"))
    sim = Simulation(load_checkpoint=False, **SMALL)

    # Emergent state that previously did NOT survive a checkpoint.
    vec = np.zeros(N_PROPS, dtype=np.float32)
    vec[0] = 0.9
    mat_id = DISCOVERY_REGISTRY.register(vec, discoverer_id=1, tick=5)
    TOKEN_WORLD.place_token(Token(token_id="tok_test", creator_id=1, tick_made=5, x=2, y=3))
    RECIPE_DISCOVERY.outcomes["seq_test"] = [(mat_id, 1.5)]
    sim._save_checkpoint()

    # A fresh construction resets the singletons, then load must restore them.
    Simulation(load_checkpoint=True, **SMALL)
    assert mat_id in DISCOVERY_REGISTRY.known_ids()
    assert np.allclose(DISCOVERY_REGISTRY.get_vector(mat_id), vec)
    assert "tok_test" in TOKEN_WORLD.tokens
    assert RECIPE_DISCOVERY.outcomes.get("seq_test") == [(mat_id, 1.5)]


def test_checkpoint_without_registries_key_still_loads(tmp_path, monkeypatch):
    path = tmp_path / "old.pkl"
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(path))
    seed = Simulation(load_checkpoint=False, **SMALL)

    # Emulate a pre-Phase-3 checkpoint: no "registries" key at all.
    with open(path, "wb") as f:
        pickle.dump(
            {
                "agents": seed.agents,
                "tick": seed.tick,
                "world": seed.world,
                "stats": seed.stats,
                "tribes": seed.tribes,
                "technology": seed.technology,
            },
            f,
        )

    fresh = Simulation(load_checkpoint=True, **SMALL)  # must not raise
    assert fresh.tick == seed.tick


def test_checkpoint_restores_id_counter(tmp_path, monkeypatch):
    """Newborn ids after a load must not collide with loaded agents' ids."""
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(tmp_path / "ckpt.pkl"))
    sim = Simulation(load_checkpoint=False, **SMALL)
    loaded_ids = {a.id for a in sim.agents}
    sim._save_checkpoint()

    # Simulate a fresh process: the class-level counter starts back at 0.
    Agent.id_counter = 0
    Simulation(load_checkpoint=True, **SMALL)
    assert Agent.id_counter >= max(loaded_ids)
    newborn = Agent.spawn_random(1, 1)
    assert newborn.id not in loaded_ids


def test_id_counter_is_class_level_not_a_field():
    assert "id_counter" not in {f.name for f in dataclasses.fields(Agent)}
    assert "id_counter" not in Agent(id=999).__dict__


def test_novelty_memory_pickle_alias():
    """Old checkpoints reference knowledge.EpisodicMemory; the alias must resolve."""
    assert EpisodicMemory is NoveltyMemory
    # Exactly the lookup unpickling performs for pre-rename checkpoints:
    found = pickle.Unpickler(io.BytesIO(b"")).find_class(
        "artificial_society.agents.knowledge", "EpisodicMemory"
    )
    assert found is NoveltyMemory
    mem = NoveltyMemory(capacity=8, k=2)
    mem.novelty(torch.zeros(4))
    restored = pickle.loads(pickle.dumps(mem))
    assert len(restored.buffer) == 1
