"""D3: PPO-v2-Parameter, KL-Early-Stop, Terminal-Transition + Buffer-Flush."""

from __future__ import annotations

import numpy as np
import torch

from artificial_society.agents.brain import (
    GAMMA_V2,
    KL_EARLY_STOP_V2,
    MINIBATCH_SIZE_V2,
    N_MINIBATCHES_V2,
    PPO_EPOCHS_V2,
    REWARD_CLAMP,
    ROLLOUT_HORIZON,
    Brain,
)


def _brain():
    torch.manual_seed(0)
    return Brain(physics_v2=True)


def _fuelle_buffer(brain, n, reward=0.5):
    feats = np.zeros((10, 17), dtype=np.float32)
    feats[0] = 0.3
    mask = np.zeros(10, dtype=bool)
    mask[0] = True
    masks = {
        "grasp": mask.copy(),
        "held": np.zeros(10, dtype=bool),
        "target": np.zeros(10, dtype=bool),
    }
    hidden = brain.initial_hidden()
    for _ in range(n):
        step = brain.act_v2([0.1] * 57, hidden, feats, mask, masks)
        hidden = step["next_hidden"]
        brain.store_transition_v2(step, reward, False, [0.2] * 57, feats, mask)


def test_ppo_v2_konstanten():
    assert GAMMA_V2 == 0.99
    assert PPO_EPOCHS_V2 == 4
    assert (N_MINIBATCHES_V2, MINIBATCH_SIZE_V2) == (4, 32)
    assert KL_EARLY_STOP_V2 == 0.02


def test_maybe_train_v2_trainiert_bei_128_und_leert_buffer():
    brain = _brain()
    _fuelle_buffer(brain, ROLLOUT_HORIZON - 1)
    assert brain.maybe_train() is None  # unterhalb des Horizonts: nichts
    _fuelle_buffer(brain, 1)
    vorher = [p.detach().clone() for p in brain.parameters()]
    loss = brain.maybe_train()
    assert loss is not None and np.isfinite(loss)
    assert len(brain.rollout) == 0
    geaendert = any(
        not torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert geaendert, "Training muss Gewichte bewegen"


def test_kl_early_stop_bricht_training_ab():
    """KL > 0.02 ⇒ Abbruch VOR dem Optimizer-Schritt des Minibatches."""
    brain = _brain()
    _fuelle_buffer(brain, ROLLOUT_HORIZON)
    # Alte log-probs künstlich massiv verschieben ⇒ approx-KL riesig
    for t in brain.rollout.storage:
        t["log_prob"] = t["log_prob"] + 10.0
    vorher = [p.detach().clone() for p in brain.parameters()]
    brain.maybe_train()
    unveraendert = all(
        torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert unveraendert, "KL-Stop muss greifen, bevor der erste Schritt appliziert wird"


def test_finalize_terminal_setzt_done_und_r_death_und_flusht():
    """D3: sterbender Agent schreibt done=True-Transition, Buffer wird trainiert."""
    brain = _brain()
    _fuelle_buffer(brain, 10, reward=0.5)
    letzte = brain.rollout.storage[-1]
    assert letzte["done"] is False
    erwartet = max(-REWARD_CLAMP, min(REWARD_CLAMP, letzte["reward"] - 3.0))
    vorher = [p.detach().clone() for p in brain.parameters()]

    loss = brain.finalize_terminal()

    assert letzte["done"] is True
    assert letzte["reward"] == erwartet  # r_death = −3.0 genau einmal
    assert loss is not None and np.isfinite(loss)
    assert len(brain.rollout) == 0  # Restbuffer geflusht
    geaendert = any(
        not torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert geaendert, "Terminal-Flush muss trainieren"


def test_finalize_terminal_randfaelle():
    brain = _brain()
    assert brain.finalize_terminal() is None  # leerer Buffer: No-op
    _fuelle_buffer(brain, 1)
    assert brain.finalize_terminal() is None  # 1 Transition: kein Training (GAE/Norm.)
    assert len(brain.rollout) == 0  # aber geleert
    # F2 (Review): kleine Buffer MÜSSEN trainieren — n=4 macht Gradientenschritte
    # (mb_size = max(2, n // 4); mit max(1, …) würde der 1er-Skip alles überspringen).
    _fuelle_buffer(brain, 4)
    vorher = [p.detach().clone() for p in brain.parameters()]
    loss = brain.finalize_terminal()
    assert loss is not None and np.isfinite(loss)
    geaendert = any(
        not torch.equal(a, b) for a, b in zip(vorher, [p.detach() for p in brain.parameters()])
    )
    assert geaendert, "Terminal-Flush bei n=4 muss trainieren (Spec C2: geflusht UND trainiert)"
    assert len(brain.rollout) == 0
