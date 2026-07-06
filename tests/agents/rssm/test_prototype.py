from __future__ import annotations

import dataclasses

import torch

from artificial_society.agents.rssm.actor_critic import ActorCriticSlab
from artificial_society.agents.rssm.config import RSSMConfig, make_generator
from artificial_society.agents.rssm.prototype import PrototypeActor

CFG = dataclasses.replace(RSSMConfig(), max_slots=4)


def test_newborn_equals_prototype_then_diverges():
    slab = ActorCriticSlab(CFG, make_generator(0, "s"))
    proto = PrototypeActor(CFG)
    assert proto.template() is None
    _s0 = slab.acquire_slot(proto.template())  # fresh init (no template yet)
    proto.update(slab)
    s1 = slab.acquire_slot(proto.template())  # warm-start from prototype
    assert torch.equal(slab.params["a_w1"][s1], proto.state_dict()["a_w1"])
    slab.params["a_w1"][s1] += 1.0  # fine-tuning diverges
    assert not torch.equal(slab.params["a_w1"][s1], proto.state_dict()["a_w1"])


def test_ema_moves_toward_population():
    slab = ActorCriticSlab(CFG, make_generator(0, "s"))
    proto = PrototypeActor(CFG)
    a = slab.acquire_slot(None)
    proto.update(slab)
    before = proto.state_dict()["a_w1"].clone()
    slab.params["a_w1"][a] += 10.0
    proto.update(slab)
    after = proto.state_dict()["a_w1"]
    assert not torch.equal(before, after)
    assert torch.all((after - before).abs() <= 10.0 * (1 - CFG.prototype_decay) + 1e-6)
