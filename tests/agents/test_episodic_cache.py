"""Regression guards for the Tier-3 brain act-path optimizations.

Locks the three behaviours the 60-tick golden trajectory does NOT cover:

* TC-3  -- EpisodicMemory.stacked() is a value-identical, correctly-invalidated
           cache of torch.stack(list(buffer)).
* BC-1  -- a checkpoint pickled before the cache existed (no _version/_cache/
           _cache_version attrs) still loads and runs (pickle bypasses __init__).
* TC-1  -- act() runs under no_grad (not inference_mode), so its stored tensors
           re-feed into the grad-enabled PPO forward in maybe_train() without an
           "inference tensors cannot be saved for backward" error.
"""
from __future__ import annotations

import pickle
from collections import deque

import torch

from artificial_society.agents.brain import INPUT_SIZE, ROLLOUT_HORIZON, Brain
from artificial_society.agents.knowledge import EpisodicMemory

DIM = 8


def _fill(em: EpisodicMemory, n: int) -> None:
    for i in range(n):
        em.novelty(torch.full((DIM,), float(i)))


# --- TC-3: cache is value-identical and correctly invalidated ----------------
def test_stacked_matches_plain_stack_across_mutations():
    em = EpisodicMemory(capacity=32, k=4)
    _fill(em, 20)
    assert torch.equal(em.stacked(), torch.stack(list(em.buffer)))

    # adding more (incl. maxlen eviction once past capacity) keeps it identical
    _fill(em, 40)
    assert len(em.buffer) == 32  # evicted down to capacity
    assert torch.equal(em.stacked(), torch.stack(list(em.buffer)))


def test_stacked_cache_reused_until_mutation():
    em = EpisodicMemory(capacity=32, k=4)
    _fill(em, 10)
    first = em.stacked()
    # no mutation -> same cached object returned (the whole point of the cache)
    assert em.stacked() is first
    em.novelty(torch.zeros(DIM))  # mutation bumps _version
    rebuilt = em.stacked()
    assert rebuilt is not first
    assert torch.equal(rebuilt, torch.stack(list(em.buffer)))


def test_reset_invalidates_cache():
    em = EpisodicMemory(capacity=32, k=4)
    _fill(em, 10)
    em.stacked()
    em.reset()
    assert em.stacked() is None  # empty buffer
    _fill(em, 6)
    assert torch.equal(em.stacked(), torch.stack(list(em.buffer)))


# --- BC-1: old checkpoints (pre-cache) still load and run ---------------------
def test_legacy_pickle_without_cache_attrs_loads_and_runs():
    # Exactly what pickle.load does for an object stored before the cache existed:
    # build the instance via __new__ (no __init__) and feed it the legacy state.
    legacy_state = {
        "capacity": 500,
        "k": 10,
        "epsilon": 1e-3,
        "buffer": deque((torch.zeros(DIM) for _ in range(12)), maxlen=500),
    }
    em = EpisodicMemory.__new__(EpisodicMemory)
    em.__setstate__(legacy_state)
    # Would AttributeError on _version / _cache before the BC-1 fix:
    em.novelty(torch.ones(DIM))
    stack = em.stacked()
    assert stack is not None and stack.shape[0] >= 12


def test_pickle_roundtrip_drops_cache_and_rebuilds():
    em = EpisodicMemory(capacity=64, k=4)
    _fill(em, 20)
    em.stacked()  # populate the cache so __getstate__ has something to drop
    restored = pickle.loads(pickle.dumps(em))
    # cache is derived state -> not persisted; rebuilt lazily, value-identical
    assert restored._cache is None
    assert torch.equal(restored.stacked(), torch.stack(list(restored.buffer)))


# --- TC-1: no_grad act tensors must survive the PPO grad forward -------------
def test_act_then_maybe_train_no_inference_error():
    torch.manual_seed(0)
    b = Brain()
    h = b.initial_hidden()
    feats = [0.05] * INPUT_SIZE
    for i in range(ROLLOUT_HORIZON + 3):
        step = b.act(feats, h, use_planning=(i % 4 == 0))
        # stored tensors must NOT be inference tensors (inference_mode would set this)
        assert step["obs_tensor"].is_inference() is False
        b.store_transition(
            step["obs_tensor"],
            step["hidden_in"],
            step["action_tensor"],
            step["log_prob"],
            step["value"],
            reward=0.1,
            done=False,
            next_obs=[0.06] * INPUT_SIZE,
        )
        h = step["next_hidden"]
    loss = b.maybe_train()  # raises under inference_mode; must return a finite loss
    assert loss is not None
    assert abs(loss) < 1e6
