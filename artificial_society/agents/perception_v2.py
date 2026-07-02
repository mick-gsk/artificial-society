"""Objekt-Wahrnehmung der Physik v2 (Plan 3b, Spec C1/C2).

10 Slots: die K=8 nächsten Boden-Objekte im Chebyshev-Radius 4 + 2 Hand-Slots.
17 Features je Slot: 13 Props + mass_kg/25 (gekappt bei 1; Anker 25-kg-Kadaver)
+ dx/4 + dy/4 (relativ, normiert) + held-Flag. Leere Slots = Null-Vektor + Maske.

Wahrnehmung ist reine Eigenschafts-Wahrnehmung: kein Objekt hat eine ID im
Gehirn — das id-basierte Slot-Tracking (Task 7) dient NUR dem Causal-Target
und dem Logging. Dieses Modul kennt weder torch noch das Gehirn.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from artificial_society.environment.physics.props import N_PROPS_V2

K_GROUND_SLOTS = 8
N_HAND_SLOTS = 2
N_SLOTS = K_GROUND_SLOTS + N_HAND_SLOTS  # 10
SLOT_FEATS = 17  # 13 Props + mass + dx + dy + held
PERCEPTION_RADIUS = 4  # Chebyshev (Spec C1)
SLOT_MASS_NORM_KG = 25.0  # Anker: 25-kg-Kadaver (Spec C1)

_MASS_IDX = N_PROPS_V2  # 13
_DX_IDX = N_PROPS_V2 + 1  # 14
_DY_IDX = N_PROPS_V2 + 2  # 15
_HELD_IDX = N_PROPS_V2 + 3  # 16


@dataclass
class SlotView:
    """Eine Wahrnehmungs-Momentaufnahme; objs/dists/at_own_pos/held_flags sind
    Metadaten für Zulässigkeit + Tracking — das Gehirn sieht nur feats+mask."""

    feats: np.ndarray  # (10, 17) float32
    mask: np.ndarray  # (10,) bool — True = Slot belegt
    objs: list  # len 10; PhysObject | None
    dists: np.ndarray  # (10,) int32 — Chebyshev; Hand-Slots = 0
    at_own_pos: np.ndarray  # (10,) bool — Boden-Objekt an eigener Position
    held_flags: np.ndarray  # (10,) bool


def _slot_features(obj, dx: int, dy: int, held: bool) -> np.ndarray:
    f = np.zeros(SLOT_FEATS, dtype=np.float32)
    f[:N_PROPS_V2] = obj.props
    f[_MASS_IDX] = min(obj.mass / SLOT_MASS_NORM_KG, 1.0)
    f[_DX_IDX] = dx / PERCEPTION_RADIUS
    f[_DY_IDX] = dy / PERCEPTION_RADIUS
    f[_HELD_IDX] = 1.0 if held else 0.0
    return f


def build_slots(agent, layer) -> SlotView:
    """Slots 0..7: nächste Boden-Objekte (Chebyshev ≤ 4, stabil nach Distanz
    sortiert — objects_near ist insertion-ordered, sorted() ist stabil ⇒
    deterministisch). Slots 8..9: gehaltene Objekte."""
    feats = np.zeros((N_SLOTS, SLOT_FEATS), dtype=np.float32)
    mask = np.zeros(N_SLOTS, dtype=bool)
    objs: list = [None] * N_SLOTS
    dists = np.zeros(N_SLOTS, dtype=np.int32)
    at_own = np.zeros(N_SLOTS, dtype=bool)
    held_flags = np.zeros(N_SLOTS, dtype=bool)

    x, y = agent.pos
    near = layer.objects_near((x, y), PERCEPTION_RADIUS)
    near.sort(key=lambda op: max(abs(op[1][0] - x), abs(op[1][1] - y)))
    for i, (obj, (ox, oy)) in enumerate(near[:K_GROUND_SLOTS]):
        d = max(abs(ox - x), abs(oy - y))
        feats[i] = _slot_features(obj, ox - x, oy - y, held=False)
        mask[i] = True
        objs[i] = obj
        dists[i] = d
        at_own[i] = d == 0

    held = list(agent.hands.held) if getattr(agent, "hands", None) is not None else []
    for j, obj in enumerate(held[:N_HAND_SLOTS]):
        s = K_GROUND_SLOTS + j
        feats[s] = _slot_features(obj, 0, 0, held=True)
        mask[s] = True
        objs[s] = obj
        held_flags[s] = True

    return SlotView(
        feats=feats, mask=mask, objs=objs, dists=dists, at_own_pos=at_own, held_flags=held_flags
    )


def admissible_masks(view: SlotView) -> dict:
    """Zulässige Slots je Verb-Rolle (Spec C2):
    grasp  — Boden-Objekt im Chebyshev-Radius 1;
    held   — gehaltene Objekte (release-Ziel; Werkzeug für strike/cut);
    target — Boden an eigener Position ∪ gehalten (strike/cut/eat-Ziel)."""
    ground = view.mask & ~view.held_flags
    held = view.mask & view.held_flags
    return {
        "grasp": ground & (view.dists <= 1),
        "held": held,
        "target": (ground & view.at_own_pos) | held,
    }


def resolve_slot_of(view: SlotView, obj) -> int:
    """Slot-Index eines Objekts per Identität (id-basiertes Tracking, Spec C1).
    NUR fürs Causal-Target/Logging — das Gehirn sieht nie Identitäten.
    -1 = aus der Wahrnehmung gefallen (Causal Model: maskiert, kein Loss)."""
    for i, o in enumerate(view.objs):
        if o is obj:
            return i
    return -1


NOVELTY_BUCKET_WIDTH = 0.25  # key = floor(p/0.25) über 13 Props (Spec C4.3)
NOVELTY_LRU_CAP = 4096


class NoveltyBuckets:
    """Count-based Novelty über Eigenschafts-Buckets, pro Agent (Spec C4.3).

    novelty = 1/√n(key) beim Wahrnehmen/Halten/Erzeugen; LRU-gekappt (4096
    Keys). Ideen (Bucket-Counts, 1/√n) aus archive/.../causal_model.py
    (_bucket_counts/info_gain); zustandsbasiert, keine designer-gewählten
    Events. Ein OrderedDict ist picklebar (Checkpoint-Verträglichkeit).
    """

    def __init__(self, cap: int = NOVELTY_LRU_CAP):
        self.cap = cap
        self.counts: OrderedDict = OrderedDict()

    def _key(self, props) -> tuple:
        return tuple(int(min(float(p), 1.0) // NOVELTY_BUCKET_WIDTH) for p in props)

    def _enforce_cap(self) -> None:
        while len(self.counts) > self.cap:
            self.counts.popitem(last=False)

    def observe_view(self, view: SlotView) -> float:
        """Novelty dieses Ticks: max über alle belegten Slots; jeder distinkte
        Bucket-Key zählt pro Tick genau einmal (zwei wertgleiche Steine sind
        EIN Punkt im Eigenschaftsraum)."""
        best = 0.0
        seen: set = set()
        for i in range(N_SLOTS):
            if not view.mask[i]:
                continue
            key = self._key(view.objs[i].props)
            if key in seen:
                continue
            seen.add(key)
            n = self.counts.get(key, 0) + 1
            self.counts[key] = n
            self.counts.move_to_end(key)
            self._enforce_cap()
            best = max(best, 1.0 / math.sqrt(n))
        return best
