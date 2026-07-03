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


def _act_und_speichere(brain, feats, mask, masks, n=40):
    """n Ticks act_v2 + store_transition_v2 mit synthetischen next-Werten."""
    hidden = brain.initial_hidden()
    steps = []
    for _ in range(n):
        step = brain.act_v2([0.1] * 57, hidden, feats, mask, masks)
        hidden = step["next_hidden"]
        brain.store_transition_v2(step, 0.5, False, [0.2] * 57, feats, mask)
        steps.append(step)
    return steps


def test_transition_speichert_maske_und_slot_index():
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(5)
    _act_und_speichere(brain, feats, mask, masks, n=40)
    assert len(brain.rollout) == 40
    t = brain.rollout.storage[0]
    for key, shape in (
        ("obs", (57,)),
        ("hidden", (96,)),
        ("action", (29,)),
        ("slot_feats", (10, 17)),
        ("slot_mask", (10,)),
        ("target_mask", (10,)),
        ("tool_mask", (10,)),
        ("next_slot_feats", (10, 17)),
        ("next_slot_mask", (10,)),
    ):
        assert tuple(t[key].shape) == shape, key
    assert isinstance(t["target_idx"], int) and isinstance(t["tool_idx"], int)
    mit_slot = [t for t in brain.rollout.storage if t["target_idx"] >= 0]
    assert mit_slot, "in 40 Ticks muss mindestens ein Verb gefeuert haben (Init-Bias)"
    for t in mit_slot:
        assert bool(t["target_mask"][t["target_idx"]])


def test_retraining_reproduktion_bitgleiche_logprobs():
    """D3: evaluate_actions_v2 liefert mit Maske+Slot-Index BITGLEICHE log-probs
    wie zur Sampling-Zeit (identischer Code-Pfad, unveränderte Gewichte).

    Review F6: scheitert torch.equal NUR an Batch-Numerik (B=1 vs. B=40), ist
    der Degrade auf allclose() der NORMALE Ausgang — s. Robustheits-Hinweis im
    Plan, keine Debug-Schleife. Verifiziert (Task 10): pro Zeile einzeln durch
    evaluate_actions_v2 (B=1) gejagt reproduziert JEDE Zeile bitgleich (Diff <
    1e-8) — die Abweichung entsteht ausschließlich beim gebatchten BLAS-Pfad
    (B=40), nicht durch falsche Maskierung/Konditionierung. Beobachteter max.
    Diff = 1.907e-6 bei log-prob-Beträgen ~15-22; float32-eps (1.19e-7) ×
    Betrag ≈ 2.38e-6 — die Abweichung liegt im Bereich einer einzelnen
    float32-ULP für diese Größenordnung, nicht im geplanten 1e-7 (der für
    kleinere Beträge kalibriert war). Toleranz daher auf atol=2e-6 gesetzt
    (weiterhin ULP-eng, keine pauschal gelockerte Toleranz)."""
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(6)
    _act_und_speichere(brain, feats, mask, masks, n=40)
    batch = brain.rollout.storage
    obs = torch.stack([t["obs"] for t in batch])
    hid = torch.stack([t["hidden"] for t in batch])
    sf = torch.stack([t["slot_feats"] for t in batch])
    sm = torch.stack([t["slot_mask"] for t in batch])
    act = torch.stack([t["action"] for t in batch])
    tm = torch.stack([t["target_mask"] for t in batch])
    om = torch.stack([t["tool_mask"] for t in batch])
    ti = torch.tensor([t["target_idx"] for t in batch], dtype=torch.long)
    oi = torch.tensor([t["tool_idx"] for t in batch], dtype=torch.long)
    alt = torch.stack([t["log_prob"] for t in batch])

    neu, ent_old, ent_new, value, next_hidden = brain.evaluate_actions_v2(
        obs, hid, sf, sm, act, tm, ti, om, oi
    )
    assert torch.allclose(neu, alt, atol=2e-6, rtol=0.0), (
        "Retraining-Reproduktion muss bitgleich sein (D3, bis auf Batch-BLAS-ULP-Rauschen)"
    )
    # Kontrakt-Kern (D3): Konditionierung + Term-Reihenfolge (tool→target) EXAKT —
    # zeilenweise B=1-Re-Evaluation über denselben Pfad ist BITGLEICH zur
    # Sampling-log-prob (das kann die 2e-6-Toleranz oben nicht garantieren).
    with torch.no_grad():
        for t in batch:
            einzeln, _, _, _, _ = brain.evaluate_actions_v2(
                t["obs"].unsqueeze(0),
                t["hidden"].unsqueeze(0),
                t["slot_feats"].unsqueeze(0),
                t["slot_mask"].unsqueeze(0),
                t["action"].unsqueeze(0),
                t["target_mask"].unsqueeze(0),
                torch.tensor([t["target_idx"]], dtype=torch.long),
                t["tool_mask"].unsqueeze(0),
                torch.tensor([t["tool_idx"]], dtype=torch.long),
            )
            assert torch.equal(einzeln.squeeze(0), t["log_prob"]), (
                "B=1-Re-Evaluation muss bitgleich sein — Konditionierungs-Bug (D3)"
            )
    assert next_hidden.shape == (40, 96)
    # Kategorial-Entropie zählt zur NEUEN Gruppe: Zeilen mit Slot-Ziehung haben mehr ent_new
    mit = torch.tensor([t["target_idx"] >= 0 for t in batch])
    if mit.any() and (~mit).any():
        assert ent_new[mit].mean() > ent_new[~mit].mean()
    assert ent_old.shape == (40,) and torch.isfinite(ent_old).all()


def test_evaluate_v2_gradient_erreicht_w_sel_und_slot_embed():
    """C2 (a): echter Gradientenpfad in Query UND Slot-Embeddings."""
    brain = _brain()
    feats, mask, masks = _volle_szene()
    torch.manual_seed(7)
    _act_und_speichere(brain, feats, mask, masks, n=40)
    batch = [t for t in brain.rollout.storage if t["target_idx"] >= 0]
    assert batch
    obs = torch.stack([t["obs"] for t in batch])
    hid = torch.stack([t["hidden"] for t in batch])
    sf = torch.stack([t["slot_feats"] for t in batch])
    sm = torch.stack([t["slot_mask"] for t in batch])
    act = torch.stack([t["action"] for t in batch])
    tm = torch.stack([t["target_mask"] for t in batch])
    om = torch.stack([t["tool_mask"] for t in batch])
    ti = torch.tensor([t["target_idx"] for t in batch], dtype=torch.long)
    oi = torch.tensor([t["tool_idx"] for t in batch], dtype=torch.long)
    lp, _, _, _, _ = brain.evaluate_actions_v2(obs, hid, sf, sm, act, tm, ti, om, oi)
    lp.sum().backward()
    assert brain.w_sel.weight.grad is not None and brain.w_sel.weight.grad.abs().sum() > 0
    assert brain.slot_embed[0].weight.grad is not None
    assert brain.slot_embed[0].weight.grad.abs().sum() > 0
