"""C1: Objekt-Slots — Features, Maskierung, nächste-8-Auswahl, zulässige Mengen (C2)."""

from __future__ import annotations

import math
import pickle
import random
from types import SimpleNamespace

import numpy as np

from artificial_society.agents.perception_v2 import (
    K_GROUND_SLOTS,
    N_SLOTS,
    NOVELTY_LRU_CAP,
    PERCEPTION_RADIUS,
    SLOT_FEATS,
    SLOT_MASS_NORM_KG,
    NoveltyBuckets,
    admissible_masks,
    build_slots,
    resolve_slot_of,
)
from artificial_society.environment.phys_objects import ObjectLayer
from artificial_society.environment.physics.body import Hands
from artificial_society.environment.physics.objects import make_object
from artificial_society.environment.physics.props import IDX2


def _agent(pos=(5, 5), held=()):
    hands = Hands()
    hands.held.extend(held)
    return SimpleNamespace(pos=pos, hands=hands)


def _layer():
    return ObjectLayer(12, 12, rng=random.Random(3))


def test_konstanten_aus_der_spec():
    assert (K_GROUND_SLOTS, N_SLOTS, SLOT_FEATS, PERCEPTION_RADIUS) == (8, 10, 17, 4)
    assert SLOT_MASS_NORM_KG == 25.0


def test_slot_features_layout_boden():
    layer = _layer()
    stein = make_object("granite", 50.0)  # > 25 kg → mass-Feature gekappt bei 1
    layer.add(stein, (7, 4), source="spawned")  # dx=+2, dy=−1
    view = build_slots(_agent(), layer)
    assert view.mask[0] and view.objs[0] is stein
    f = view.feats[0]
    np.testing.assert_allclose(f[:13], stein.props)
    assert f[13] == 1.0  # min(mass/25, 1)
    assert f[14] == 2 / 4 and f[15] == -1 / 4  # dx/4, dy/4
    assert f[16] == 0.0  # held-Flag
    assert view.dists[0] == 2 and not view.at_own_pos[0] and not view.held_flags[0]


def test_hand_slots_8_und_9():
    fleisch = make_object("raw_meat", 1.0)
    klinge = make_object("flint", 0.3)
    view = build_slots(_agent(held=(fleisch, klinge)), _layer())
    assert view.mask[8] and view.mask[9]
    assert view.objs[8] is fleisch and view.objs[9] is klinge
    assert view.feats[8][16] == 1.0 and view.feats[9][16] == 1.0  # held-Flag
    assert view.feats[8][14] == 0.0 and view.feats[8][15] == 0.0  # dx=dy=0
    assert view.held_flags[8] and view.held_flags[9]
    assert not view.mask[:8].any()


def test_leere_slots_sind_nullvektor_plus_maske():
    view = build_slots(_agent(), _layer())
    assert not view.mask.any()
    np.testing.assert_array_equal(view.feats, np.zeros((N_SLOTS, SLOT_FEATS), dtype=np.float32))
    assert view.objs == [None] * N_SLOTS


def test_naechste_8_nach_chebyshev_r4():
    layer = _layer()
    fern = make_object("granite", 1.0)
    layer.add(fern, (10, 10), source="spawned")  # Chebyshev 5 > 4 → unsichtbar
    for i in range(9):  # 9 Objekte in Reichweite → nur die nächsten 8
        layer.add(make_object("granite", 1.0), (5 + min(i, 4), 5), source="spawned")
    view = build_slots(_agent(), layer)
    assert view.mask[:8].all()
    assert fern not in view.objs
    assert sorted(view.dists[:8].tolist()) == view.dists[:8].tolist()  # nach Distanz sortiert


def test_admissible_masks_c2():
    layer = _layer()
    hier = make_object("carcass", 25.0)
    nah = make_object("granite", 1.0)
    fern = make_object("granite", 1.0)
    layer.add(hier, (5, 5), source="from_carcass")  # eigene Position
    layer.add(nah, (6, 5), source="spawned")  # r=1
    layer.add(fern, (5, 8), source="spawned")  # r=3
    klinge = make_object("flint", 0.3)
    view = build_slots(_agent(held=(klinge,)), layer)
    masks = admissible_masks(view)
    i_hier, i_nah, i_fern = view.objs.index(hier), view.objs.index(nah), view.objs.index(fern)
    # grasp: Boden r ≤ 1
    assert masks["grasp"][i_hier] and masks["grasp"][i_nah] and not masks["grasp"][i_fern]
    assert not masks["grasp"][8]  # Gehaltenes ist kein grasp-Ziel
    # held: nur Hand-Slots
    assert masks["held"][8] and masks["held"][:8].sum() == 0
    # target (strike/cut/eat): Boden an eigener Position ∪ gehalten
    assert masks["target"][i_hier] and masks["target"][8]
    assert not masks["target"][i_nah] and not masks["target"][i_fern]
    # Nutzlast-Check: 13 Props der Slot-Features stimmen mit IDX2-Layout überein
    assert view.feats[i_hier][IDX2["nutrition"]] == hier.props[IDX2["nutrition"]]


def test_resolve_slot_of_ist_id_basiert():
    layer = _layer()
    a = make_object("granite", 1.0)
    b = make_object("granite", 1.0)  # wertgleich, andere Identität
    layer.add(a, (5, 5), source="spawned")
    layer.add(b, (5, 6), source="spawned")
    view = build_slots(_agent(), layer)
    assert view.objs[resolve_slot_of(view, a)] is a
    assert view.objs[resolve_slot_of(view, b)] is b
    assert resolve_slot_of(view, make_object("granite", 1.0)) == -1  # nie gesehen


def test_novelty_faellt_mit_1_durch_wurzel_n():
    """D3 (c): Bucket-Novelty fällt mit 1/√n über wiederholte Wahrnehmung."""
    layer = _layer()
    layer.add(make_object("granite", 1.0), (5, 5), source="spawned")
    view = build_slots(_agent(), layer)
    buckets = NoveltyBuckets()
    werte = [buckets.observe_view(view) for _ in range(4)]
    assert werte == [1.0, 1.0 / math.sqrt(2), 1.0 / math.sqrt(3), 1.0 / math.sqrt(4)]


def test_novelty_neuer_eigenschaftspunkt_zahlt_wieder_voll():
    layer = _layer()
    layer.add(make_object("granite", 1.0), (5, 5), source="spawned")
    agent = _agent()
    buckets = NoveltyBuckets()
    buckets.observe_view(build_slots(agent, layer))
    layer.add(make_object("carcass", 25.0), (5, 5), source="from_carcass")
    assert buckets.observe_view(build_slots(agent, layer)) == 1.0  # neuer Bucket → 1/√1


def test_novelty_dedupe_innerhalb_eines_ticks():
    """Zwei wertgleiche Steine im selben Tick zählen den Bucket nur EINMAL."""
    layer = _layer()
    layer.add(make_object("granite", 1.0), (5, 5), source="spawned")
    layer.add(make_object("granite", 1.0), (5, 6), source="spawned")
    buckets = NoveltyBuckets()
    buckets.observe_view(build_slots(_agent(), layer))
    assert list(buckets.counts.values()) == [1]


def test_novelty_lru_kappt_bei_4096_und_ist_picklebar():
    buckets = NoveltyBuckets(cap=3)
    for i in range(5):
        buckets.counts[("k", i)] = 1
        buckets._enforce_cap()
    assert len(buckets.counts) == 3
    assert ("k", 0) not in buckets.counts  # ältester Key flog raus
    assert NOVELTY_LRU_CAP == 4096
    wieder = pickle.loads(pickle.dumps(buckets))
    assert dict(wieder.counts) == dict(buckets.counts)
