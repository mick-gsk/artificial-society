"""D3: Permutations-Invarianz, All-Masked-No-op, kategoriale Auswahl, log-prob-Struktur."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.brain import (
    RESEARCH_DRIVE_DIM,
    VERB_SLICE,
    VERBS_V2,
    Brain,
)


def _brain():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def _leere_masken():
    return {
        "grasp": np.zeros(10, dtype=bool),
        "held": np.zeros(10, dtype=bool),
        "target": np.zeros(10, dtype=bool),
    }


def _volle_szene():
    """2 Boden-Objekte an eigener Position (Slots 0, 1), 2 gehalten (8, 9)."""
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    for s in (0, 1, 8, 9):
        feats[s] = np.random.RandomState(s).rand(17).astype(np.float32)
        mask[s] = True
    feats[8][16] = feats[9][16] = 1.0
    masks = {
        "grasp": np.array([True, True] + [False] * 8),
        "held": np.array([False] * 8 + [True, True]),
        "target": np.array([True, True] + [False] * 6 + [True, True]),
    }
    return feats, mask, masks


def test_permutations_invarianz_vor_dem_sampling():
    """D3: Slot-Shuffle ⇒ identische Verteilungsparameter (μ, log_std,
    zurücksortierte attn) — Vergleich VOR dem Sampling (Sample-Vergleiche sind
    durch RNG-Konsum flaky)."""
    brain = _brain()
    hidden = torch.randn(1, 96)
    obs = torch.rand(1, 57)
    feats = torch.rand(1, 10, 17)
    mask = torch.ones(1, 10, dtype=torch.bool)
    perm = torch.tensor([3, 1, 4, 0, 7, 5, 2, 6, 8, 9])  # Boden-Slots permutiert
    m1, s1, v1, h1, _, a1 = brain.forward_v2(obs, hidden, feats, mask)
    m2, s2, v2, h2, _, a2 = brain.forward_v2(obs, hidden, feats[:, perm], mask[:, perm])
    assert torch.allclose(m1, m2, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)
    assert torch.allclose(h1, h2, atol=1e-5)
    assert torch.allclose(a1[:, perm], a2, atol=1e-5)  # attn zurücksortiert identisch


def test_all_masked_noop_ohne_nan_und_ohne_slot_logprob():
    brain = _brain()
    feats = np.zeros((10, 17), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    for _ in range(20):  # mehrere Samples: Verben feuern oft (Init-Bias +0.2)
        step = brain.act_v2([0.0] * 57, brain.initial_hidden(), feats, mask, _leere_masken())
        assert step["target_idx"] == -1 and step["tool_idx"] == -1
        assert not step["target_mask"].any() and not step["tool_mask"].any()
        assert torch.isfinite(step["log_prob"]).all()
        assert not torch.isnan(step["action_tensor"]).any()
        # log_prob == reiner kontinuierlicher Anteil (kein Kategorial-Term)
        mean, std, _, _, _, _ = brain.forward_v2(
            step["obs_tensor"], step["hidden_in"], step["slot_feats"], step["slot_mask"]
        )
        lp_cont, _ = brain._continuous_log_prob(mean, std, step["action_tensor"])
        assert torch.equal(step["log_prob"], lp_cont)


def test_research_drive_fehlt_in_der_logprob_summe():
    """D3: Dim 6 ist im v2 totes Rauschen — nicht im PPO-Ratio."""
    brain = _brain()
    mean = torch.zeros(1, 29)
    std = torch.ones(1, 29)
    a = torch.zeros(1, 29)
    lp0, _ = brain._continuous_log_prob(mean, std, a)
    a2 = a.clone()
    a2[0, RESEARCH_DRIVE_DIM] = 0.9  # nur research_drive ändern
    lp1, _ = brain._continuous_log_prob(mean, std, a2)
    assert torch.equal(lp0, lp1)
    a3 = a.clone()
    a3[0, 0] = 0.9  # Kontrolle: andere Dim ändert die log-prob sehr wohl
    lp2, _ = brain._continuous_log_prob(mean, std, a3)
    assert not torch.equal(lp0, lp2)


def test_kategoriale_auswahl_zieht_nur_zulaessige_slots():
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(3)
    verbs_gesehen = set()
    for _ in range(300):
        step = brain.act_v2([0.1] * 57, brain.initial_hidden(), feats, mask, masks)
        if step["verb"] is None:
            assert step["target_idx"] == -1
            continue
        verbs_gesehen.add(step["verb"])
        assert step["target_idx"] >= 0
        assert bool(step["target_mask"][0, step["target_idx"]])  # nur zulässige Ziele
        if step["verb"] in ("strike", "cut"):
            assert step["tool_idx"] in (8, 9)  # Werkzeug nur aus der Hand
            assert step["target_idx"] != step["tool_idx"]  # Ziel ≠ Werkzeug
        else:
            assert step["tool_idx"] == -1
        if step["verb"] == "grasp":
            assert step["target_idx"] in (0, 1)
        if step["verb"] == "release":
            assert step["target_idx"] in (8, 9)
    assert {"grasp", "release", "strike", "cut", "eat"} & verbs_gesehen, (
        "mit Init-Bias +0.2 müssen Verben in 300 Samples feuern"
    )


def test_strike_ohne_gehaltenes_objekt_ist_noop_ohne_slot_terme():
    brain = _brain()
    feats, mask, _ = _volle_szene()
    mask[8] = mask[9] = False  # Hände leer
    feats[8] = feats[9] = 0.0
    masks = {
        "grasp": np.array([True, True] + [False] * 8),
        "held": np.zeros(10, dtype=bool),
        "target": np.array([True, True] + [False] * 8),
    }
    torch.manual_seed(4)
    for _ in range(200):
        step = brain.act_v2([0.1] * 57, brain.initial_hidden(), feats, mask, masks)
        if step["verb"] == "strike":
            raise AssertionError("strike ohne Schläger muss als No-op aufgelöst werden")
        if step["verb"] == "cut":
            assert step["tool_idx"] == -1  # bloße Hand: kein Tool-Term
            assert step["target_idx"] in (0, 1)


def test_verbs_v2_reihenfolge():
    assert VERBS_V2 == ("grasp", "release", "strike", "cut", "eat")
    assert slice(7, 12) == VERB_SLICE
