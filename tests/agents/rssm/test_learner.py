from __future__ import annotations

import dataclasses
import types

import torch

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.agents.rssm.learner import SharedLearner

CFG = dataclasses.replace(
    RSSMConfig(),
    max_slots=8,
    horizon=4,
    batch_size=4,
    seq_len=16,
    prefill_transitions=64,
    wm_warmup_updates=1,
    train_device="cpu",
)


def _agent(aid):
    return types.SimpleNamespace(id=aid, alive=True, hidden_state=None, birth_tick=0)


def _feats(v=0.5):
    return [v] * 57


def test_act_contract_and_determinism():
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    step = ln.act(ag, _feats())
    assert set(step) >= {"action_list", "next_hidden", "obs_tensor", "action_tensor"}
    assert len(step["action_list"]) == 7
    assert all(-1.0 <= a <= 1.0 for a in step["action_list"])
    h, z = step["next_hidden"]
    assert h.shape == (1, 256) and z.shape == (1, 32, 32)
    # same seed twice → identical trajectory of draws
    ln2 = SharedLearner(CFG, 42)
    ag2 = _agent(1)
    ln2.on_spawn(ag2, "initial", 0)
    step2 = ln2.act(ag2, _feats())
    assert step["action_list"] == step2["action_list"]


def test_global_rng_untouched_by_full_cycle():
    torch.manual_seed(123)
    before = torch.get_rng_state()
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    for t in range(80):
        step = ln.act(ag, _feats())
        ag.hidden_state = step["next_hidden"]
        ln.store_transition(ag, step, 0.1, done=(t == 79))
    ln.on_death(ag)
    assert torch.equal(before, torch.get_rng_state())


def test_train_cadence_and_warmup_gate():
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    for _t in range(70):
        step = ln.act(ag, _feats())
        ag.hidden_state = step["next_hidden"]
        ln.store_transition(ag, step, 0.0, done=False)
    assert ln.maybe_train(3, [ag]) is None  # 3 % train_every != 0
    m1 = ln.maybe_train(8, [ag])  # replay ≥ prefill → WM trains
    assert m1 is not None and ln.wm.wm_updates == 1
    assert "actor_loss" not in m1  # warm-up gate: no imagination yet
    m2 = ln.maybe_train(16, [ag])
    assert "actor_loss" in m2  # gate open after warmup updates


def test_spawn_death_slot_lifecycle():
    ln = SharedLearner(CFG, 42)
    a, b = _agent(1), _agent(2)
    ln.on_spawn(a, "initial", 0)
    ln.on_spawn(b, "birth", 5)
    assert a.rssm_slot != b.rssm_slot and a.spawn_origin == "initial"
    ln.on_death(a)
    c = _agent(3)
    ln.on_spawn(c, "respawn", 9)
    assert c.rssm_slot == a.rssm_slot  # freed slot reused


def test_act_actor_mirror_disabled_when_train_device_is_cpu():
    """GPU-pilot blocker regression: the CPU actor mirror is only needed (and only
    allocated) when the slab itself doesn't already live on CPU."""
    ln = SharedLearner(CFG, 42)
    assert ln._act_actor is None


def test_checkpoint_roundtrip():
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    step = ln.act(ag, _feats())
    payload = ln.checkpoint_payload()
    ln2 = SharedLearner(CFG, 42)
    ag2 = _agent(1)
    ag2.rssm_slot = ag.rssm_slot
    ln2.load_checkpoint_payload(payload, [ag2])
    step2 = ln2.act(ag2, _feats())
    assert step["action_list"] == step2["action_list"]


def test_checkpoint_payload_wm_opt_is_cpu_tree():
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    for _t in range(70):
        step = ln.act(ag, _feats())
        ag.hidden_state = step["next_hidden"]
        ln.store_transition(ag, step, 0.0, done=False)
    ln.maybe_train(8, [ag])  # ensures the WM optimizer has real state tensors
    payload = ln.checkpoint_payload()

    def _assert_cpu(obj):
        if torch.is_tensor(obj):
            assert obj.device.type == "cpu"
        elif isinstance(obj, dict):
            for v in obj.values():
                _assert_cpu(v)
        elif isinstance(obj, list):
            for v in obj:
                _assert_cpu(v)

    _assert_cpu(payload["wm_opt"])
    assert any(
        torch.is_tensor(v) for s in payload["wm_opt"].get("state", {}).values() for v in s.values()
    )  # optimizer state is non-trivial, so the check above proved something
