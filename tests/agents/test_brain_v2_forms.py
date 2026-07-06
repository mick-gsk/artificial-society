"""D3-Form-Tests: 29 kont. Dims, GRU-Input 192, v1-Formen unverändert, All-Masked ⇒ obj_ctx=0."""

from __future__ import annotations

import torch

from artificial_society.agents.brain import (
    ACTION_SIZE_V2,
    CURIO_TARGET_DIM,
    LOGSTD_FLOOR_NEW,
    OBS_TARGET_EXCLUDED,
    OBS_TARGET_INCLUDED_IDX,
    VERB_INIT_BIAS,
    Brain,
)


def _v2():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def test_v1_formen_unveraendert():
    """Golden-/Checkpoint-Garantie: das v1-Brain hat exakt die heutigen Module."""
    torch.manual_seed(0)
    brain = Brain()
    assert brain.physics_v2 is False
    assert brain.gru.input_size == 128 and brain.gru.hidden_size == 96
    assert brain.policy_mean.out_features == 7
    assert brain.policy_logstd.shape == (7,)
    assert brain.world_fc[0].in_features == 96 + 7
    for verboten in ("slot_embed", "attn_query", "w_sel", "next_slots_head"):
        assert not hasattr(brain, verboten), f"v1-Brain darf kein {verboten} haben"
    assert "curio_err_mean" not in dict(brain.named_buffers())


def test_v2_formen_nach_spec():
    brain = _v2()
    assert brain.physics_v2 is True
    assert brain.action_size == ACTION_SIZE_V2 == 29
    assert brain.gru.input_size == 192 and brain.gru.hidden_size == 96  # 128 + 64
    assert brain.policy_mean.out_features == 29
    assert brain.policy_logstd.shape == (29,)
    assert brain.slot_embed[0].in_features == 17 and brain.slot_embed[0].out_features == 32
    assert isinstance(brain.slot_embed[1], torch.nn.LayerNorm)
    assert brain.attn_query.in_features == 96 and brain.attn_query.out_features == 32
    assert brain.w_sel.in_features == 32 and brain.w_sel.out_features == 8
    assert brain.w_sel.bias is None
    assert brain.next_slots_head.out_features == 170  # 10 × 17
    assert brain.world_fc[0].in_features == 96 + 29
    assert brain.curio_err_mean.shape == (CURIO_TARGET_DIM,) == (203,)
    assert LOGSTD_FLOOR_NEW == -0.8


def test_verb_init_bias_und_logstd_floor():
    brain = _v2()
    assert torch.all(brain.policy_mean.bias[7:12] == VERB_INIT_BIAS)  # +0.2 Babbling-Prior
    with torch.no_grad():
        brain.policy_logstd.fill_(-5.0)
    ls = brain._clamped_logstd()
    assert torch.all(ls[:7] == -2.0)  # alte Gruppe: Floor −2.0 (wie heute)
    assert torch.all(ls[7:] == -0.8)  # neue Gruppe: Floor −0.8 (Spec C2)


def test_neugier_target_ausschluss_dims():
    """D3 Neugier (a): last_reward (26), Causal (34–36), Episodic (37–48), Modulator (49–56)."""
    assert frozenset({26} | set(range(34, 57))) == OBS_TARGET_EXCLUDED
    assert len(OBS_TARGET_INCLUDED_IDX) == 33
    assert 26 not in OBS_TARGET_INCLUDED_IDX
    assert all(d not in OBS_TARGET_INCLUDED_IDX for d in range(34, 57))


def test_forward_v2_formen_und_all_masked():
    brain = _v2()
    obs = torch.zeros(1, 57)
    hidden = torch.zeros(1, 96)
    feats = torch.zeros(1, 10, 17)
    mask = torch.zeros(1, 10, dtype=torch.bool)  # kein Objekt in Sicht, leere Hände
    mean, std, value, next_hidden, embeds, attn = brain.forward_v2(obs, hidden, feats, mask)
    assert mean.shape == (1, 29) and std.shape == (1, 29)
    assert next_hidden.shape == (1, 96) and embeds.shape == (1, 10, 32)
    for t in (mean, std, value, next_hidden, embeds, attn):
        assert not torch.isnan(t).any(), "All-Masked darf kein NaN erzeugen (Spec C1)"
    obj_ctx, _, attn0 = brain._encode_slots(feats, mask, hidden)
    assert torch.all(obj_ctx == 0.0), "All-Masked ⇒ obj_ctx = 0"
    assert torch.all(attn0 == 0.0)


def test_forward_v2_teilmaske_nutzt_nur_belegte_slots():
    brain = _v2()
    hidden = torch.zeros(1, 96)
    feats = torch.rand(1, 10, 17)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, 3] = True
    obj_ctx, embeds, attn = brain._encode_slots(feats, mask, hidden)
    assert attn[0, 3] == 1.0 and attn[0].sum() == 1.0  # einziger Slot trägt alles
    assert not torch.isnan(obj_ctx).any()
    # Max-Pool = Embedding des einzigen Slots
    assert torch.allclose(obj_ctx[0, 32:], embeds[0, 3])


def test_derive_weights_ueberspringt_shape_mismatch_v1_v2():
    """C5: derive_weights_from überspringt Mismatches still (gewollt beim
    Architektur-Wechsel) — v2-Kind von v1-Eltern crasht nicht."""
    torch.manual_seed(1)
    v1, v2 = Brain(), Brain(physics_v2=True)
    v2.derive_weights_from(v1)  # darf nicht werfen
