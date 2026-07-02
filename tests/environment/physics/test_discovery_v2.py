"""DiscoveryV2: stabile IDs für neuartige v2-Eigenschaftsvektoren."""

from __future__ import annotations

import random

import numpy as np

from artificial_society.environment.physics.discovery import DiscoveryV2
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.processes import strike


def test_same_vector_same_id_new_vector_new_id():
    reg = DiscoveryV2()
    v1 = np.zeros(13, dtype=np.float32)
    v2 = v1.copy()
    v2[0] = 0.5
    id_a = reg.register(v1)
    assert reg.register(v1.copy()) == id_a  # identisch → gleiche ID
    assert reg.register(v2) != id_a  # deutlich anders → neue ID
    assert reg.known_ids() == [id_a, "pmat_0001"]


def test_fragments_from_strike_get_registered():
    reg = DiscoveryV2()
    result = strike(make_object("flint", 0.8), make_object("granite", 1.0), 20.0, random.Random(42))
    ids = [reg.register(f.props, discoverer_id=7, tick=100) for f in result.fragments]
    assert all(i.startswith("pmat_") for i in ids)
    vec = reg.get_vector(ids[0])
    assert vec is not None and vec.shape == (13,)


def test_state_roundtrip_and_reset():
    reg = DiscoveryV2()
    reg.register(np.ones(13, dtype=np.float32))
    snapshot = reg.state_dict()
    reg2 = DiscoveryV2()
    reg2.load_state_dict(snapshot)
    assert reg2.known_ids() == reg.known_ids()
    reg2.reset()
    assert reg2.known_ids() == []
    assert reg2.get_vector("pmat_0000") is None
