"""Unit tests for the agent-internal causal forward model (PROTOTYPE).

These exercise the standalone ``CausalModel`` learning machinery. The
determinism golden trajectory (flag OFF = byte-identical) is guarded separately
by ``tests/test_regression_golden.py``; here we only test the module in
isolation with the flag conceptually ON.
"""

import numpy as np

from artificial_society.systems.causal_model import (
    CausalModel,
    _cm_enabled,
    causal_model_step,
    get_or_create_model,
    select_by_info_gain,
)

N_PROPS = 12
N_ACTIONS = 8


class _FakeAgent:
    """Minimal stand-in: the helpers only touch ``.id`` and ``.causal_model``."""

    def __init__(self, agent_id):
        self.id = agent_id


def _toy_physics(props_a, props_b, action_idx):
    """A deterministic, smooth, input-dependent target — a learnable stand-in for
    ``combine_vectors``. Includes one multiplicative interaction so beating a
    constant-mean baseline genuinely requires using the inputs (= generalization).
    """
    y = 0.5 * (props_a + props_b)
    y[4] = props_a[1] * props_b[7]  # interaction: hardness-ish × dryness-ish
    y[0] = 0.3 * props_a[0] + 0.2 * action_idx / N_ACTIONS
    return np.clip(y, 0.0, 1.0)


def test_cm_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv("AS_CAUSAL_MODEL", raising=False)
    assert _cm_enabled() is False
    monkeypatch.setenv("AS_CAUSAL_MODEL", "1")
    assert _cm_enabled() is True
    monkeypatch.setenv("AS_CAUSAL_MODEL", "0")
    assert _cm_enabled() is False


def test_predict_shape_and_range():
    m = CausalModel(seed=1)
    a = np.full(N_PROPS, 0.5, dtype=np.float32)
    b = np.full(N_PROPS, 0.2, dtype=np.float32)
    pred = m.predict(a, b, 3)
    assert pred.shape == (N_PROPS,)
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_model_learns_repeated_pair():
    """Observing the same combination repeatedly drives prediction error down."""
    m = CausalModel(seed=7)
    rng = np.random.default_rng(0)
    a = rng.random(N_PROPS).astype(np.float32)
    b = rng.random(N_PROPS).astype(np.float32)
    action = 2
    y = _toy_physics(a, b, action)
    surprises = [m.observe(a, b, action, y) for _ in range(80)]
    assert surprises[-1] < surprises[0] * 0.5  # error at least halved


def test_reproducible_with_fixed_seed():
    """Same seed + same observation sequence => identical predictions (isolated RNG)."""
    rng = np.random.default_rng(3)
    obs = [
        (rng.random(N_PROPS).astype(np.float32), rng.random(N_PROPS).astype(np.float32), i % N_ACTIONS)
        for i in range(20)
    ]
    m1, m2 = CausalModel(seed=99), CausalModel(seed=99)
    for a, b, act in obs:
        m1.observe(a, b, act, _toy_physics(a, b, act))
        m2.observe(a, b, act, _toy_physics(a, b, act))
    qa, qb = np.full(N_PROPS, 0.4, dtype=np.float32), np.full(N_PROPS, 0.6, dtype=np.float32)
    assert np.array_equal(m1.predict(qa, qb, 1), m2.predict(qa, qb, 1))


def test_transfer_beats_mean_baseline():
    """The defining causal-theory test: on material pairs the model NEVER trained
    on, its prediction error must beat a constant predict-the-training-mean baseline.
    Beating the mean requires generalizing over latent input properties."""
    train_rng = np.random.default_rng(11)
    train = [
        (train_rng.random(N_PROPS).astype(np.float32), train_rng.random(N_PROPS).astype(np.float32), train_rng.integers(N_ACTIONS))
        for _ in range(250)
    ]
    held_rng = np.random.default_rng(777)
    held = [
        (held_rng.random(N_PROPS).astype(np.float32), held_rng.random(N_PROPS).astype(np.float32), held_rng.integers(N_ACTIONS))
        for _ in range(60)
    ]

    m = CausalModel(seed=5)
    for _ in range(40):  # epochs of online SGD
        for a, b, act in train:
            m.observe(a, b, int(act), _toy_physics(a, b, int(act)))

    train_mean = np.mean([_toy_physics(a, b, int(act)) for a, b, act in train], axis=0)
    model_err = np.mean([
        np.linalg.norm(m.predict(a, b, int(act)) - _toy_physics(a, b, int(act))) for a, b, act in held
    ])
    base_err = np.mean([
        np.linalg.norm(train_mean - _toy_physics(a, b, int(act))) for a, b, act in held
    ])
    assert model_err < base_err * 0.75  # clear generalization margin on unseen pairs


def test_sensitivity_probe_shape():
    """L2 readability: finite-difference Jacobian d(output)/d(input)."""
    m = CausalModel(seed=2)
    a = np.full(N_PROPS, 0.5, dtype=np.float32)
    b = np.full(N_PROPS, 0.5, dtype=np.float32)
    jac = m.sensitivity(a, b, 0)
    # inputs = props_a(12) + props_b(12) = 24 ; outputs = 12
    assert jac.shape == (2 * N_PROPS, N_PROPS)


def test_info_gain_drops_with_observation():
    """L3: expected information gain is higher for an unseen region than for a
    region the model has already explored many times."""
    m = CausalModel(seed=4)
    a = np.full(N_PROPS, 0.5, dtype=np.float32)
    b = np.full(N_PROPS, 0.5, dtype=np.float32)
    fresh = np.full(N_PROPS, 0.1, dtype=np.float32)
    ig_before = m.info_gain(a, b, 0)
    for _ in range(30):
        m.observe(a, b, 0, _toy_physics(a, b, 0))
    ig_after = m.info_gain(a, b, 0)
    ig_fresh = m.info_gain(fresh, fresh, 5)
    assert ig_after < ig_before
    assert ig_fresh > ig_after


# --- integration helpers (systems-lane glue, no hot-file edits) -------------


def test_get_or_create_model_off(monkeypatch):
    monkeypatch.delenv("AS_CAUSAL_MODEL", raising=False)
    agent = _FakeAgent(3)
    assert get_or_create_model(agent) is None
    assert getattr(agent, "causal_model", None) is None


def test_get_or_create_model_on_lazy_seeded_idempotent(monkeypatch):
    monkeypatch.setenv("AS_CAUSAL_MODEL", "1")
    a1 = _FakeAgent(42)
    m = get_or_create_model(a1)
    assert isinstance(m, CausalModel)
    assert a1.causal_model is m
    assert get_or_create_model(a1) is m  # idempotent
    # seeded from agent.id -> two agents with same id get identical initial weights
    a2 = _FakeAgent(42)
    assert np.array_equal(get_or_create_model(a2).W1, m.W1)
    a3 = _FakeAgent(43)
    assert not np.array_equal(get_or_create_model(a3).W1, m.W1)


def test_causal_model_step_off_is_noop(monkeypatch):
    monkeypatch.delenv("AS_CAUSAL_MODEL", raising=False)
    agent = _FakeAgent(1)
    a = np.full(N_PROPS, 0.5, dtype=np.float32)
    r = causal_model_step(agent, a, a, 0, a)
    assert r == 0.0
    assert getattr(agent, "causal_model", None) is None


def test_causal_model_step_on_rewards_and_scales_with_surprise(monkeypatch):
    monkeypatch.setenv("AS_CAUSAL_MODEL", "1")
    a = np.full(N_PROPS, 0.5, dtype=np.float32)
    near = np.full(N_PROPS, 0.5, dtype=np.float32)  # fresh model predicts ~0.5
    far = np.full(N_PROPS, 1.0, dtype=np.float32)
    # same id => identical fresh model => comparable surprise
    r_near = causal_model_step(_FakeAgent(8), a, a, 0, near)
    r_far = causal_model_step(_FakeAgent(8), a, a, 0, far)
    assert r_near >= 0.0
    assert r_far > r_near  # epistemic reward grows with prediction error
    # observing increments the model's record for that region
    agent = _FakeAgent(8)
    causal_model_step(agent, a, a, 0, far)
    assert agent.causal_model._bucket_counts[agent.causal_model._bucket_key(a, a, 0)] == 1


def test_select_by_info_gain_prefers_unexplored(monkeypatch):
    monkeypatch.setenv("AS_CAUSAL_MODEL", "1")
    m = CausalModel(seed=5)
    explored = (np.full(N_PROPS, 0.5, dtype=np.float32), np.full(N_PROPS, 0.5, dtype=np.float32), 0)
    fresh1 = (np.full(N_PROPS, 0.0, dtype=np.float32), np.full(N_PROPS, 0.0, dtype=np.float32), 1)
    fresh2 = (np.full(N_PROPS, 1.0, dtype=np.float32), np.full(N_PROPS, 1.0, dtype=np.float32), 2)
    for _ in range(40):
        m.observe(*explored, _toy_physics(*explored))
    cands = [explored, fresh1, fresh2]

    gen = np.random.default_rng(0)
    picks = [select_by_info_gain(m, cands, gen) for _ in range(300)]  # picks are indices
    assert all(0 <= p < len(cands) for p in picks)
    explored_share = sum(1 for p in picks if p == 0) / len(picks)  # index 0 == explored
    assert explored_share < 1.0 / 3.0  # chosen less than uniform -> info-gain biased

    # reproducible for a fixed generator seed
    g1, g2 = np.random.default_rng(7), np.random.default_rng(7)
    assert select_by_info_gain(m, cands, g1) == select_by_info_gain(m, cands, g2)
