"""D3-Neugier: (a) selbstbezügliche Dims nicht im Target, per-Dim-Normalisierung, Maskierung."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.brain import (
    CURIO_TARGET_DIM,
    OBS_TARGET_INCLUDED_IDX,
    Brain,
)


def _brain():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def _step(brain):
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    masks = {k: np.zeros(10, dtype=bool) for k in ("grasp", "held", "target")}
    return brain.act_v2([0.1] * 57, brain.initial_hidden(), feats, mask, masks)


def test_predict_world_v2_formen():
    brain = _brain()
    h = torch.zeros(1, 96)
    a = torch.zeros(1, 29)
    next_obs, next_slots, reward = brain.predict_world_v2(h, a)
    assert next_obs.shape == (1, 57)
    assert next_slots.shape == (1, 170)  # 10 × 17 rohe Slot-Features
    assert reward.shape == (1,)


def test_selbstbezuegliche_dims_nicht_im_nextslot_target():
    """D3 (a): last_reward (26), Causal (34–36), Episodic (37–48), Hormone (49–56)
    ändern den Fehler NICHT — sie sind nachweislich nicht im Target."""
    brain = _brain()
    step = _step(brain)
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    basis = [0.3] * 57
    gestoert = list(basis)
    for dim in [26, *range(34, 57)]:
        gestoert[dim] = 0.9  # massive Störung NUR auf ausgeschlossenen Dims
    b1 = Brain(physics_v2=True)
    b1.load_state_dict(brain.state_dict())
    e1 = brain.nextslot_error(step, basis, feats, mask)
    e2 = b1.nextslot_error(step, gestoert, feats, mask)
    assert e1 == e2
    # Gegenprobe: eine EINGESCHLOSSENE Dim ändert den Fehler
    b2 = Brain(physics_v2=True)
    b2.load_state_dict(brain.state_dict())
    anders = list(basis)
    anders[OBS_TARGET_INCLUDED_IDX[0]] = 0.9
    assert b2.nextslot_error(step, anders, feats, mask) != e1


def test_maskierte_slots_zaehlen_nicht():
    """Nur belegte Slots tragen zum Slot-Anteil des Fehlers bei."""
    brain = _brain()
    step = _step(brain)
    leer = np.zeros((10, 17), dtype=np.float32)
    voll_aber_maskiert = np.full((10, 17), 0.9, dtype=np.float32)
    maske_aus = np.zeros(10, dtype=bool)
    b1 = Brain(physics_v2=True)
    b1.load_state_dict(brain.state_dict())
    e1 = brain.nextslot_error(step, [0.3] * 57, leer, maske_aus)
    e2 = b1.nextslot_error(step, [0.3] * 57, voll_aber_maskiert, maske_aus)
    assert e1 == e2  # maskierte Features sind unsichtbar


def test_running_stats_normalisieren_wiederholten_fehler():
    """Per-Dim Mean+Std: ein konstanter Fehler wird über die Zeit wegnormalisiert
    (z → 0), statt ewige Rausch-Rente zu zahlen."""
    brain = _brain()
    step = _step(brain)
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    werte = [brain.nextslot_error(step, [0.5] * 57, feats, mask) for _ in range(600)]
    assert werte[-1] < werte[0] * 0.2, "konstanter Fehler muss wegnormalisiert werden"
    assert brain.curio_err_mean.shape == (CURIO_TARGET_DIM,)
