"""Agent-internal causal forward model (PROTOTYPE — flag-gated ``AS_CAUSAL_MODEL``).

Each agent owns a tiny MLP that predicts the *outcome* property-vector of combining
two materials under an action — ``f(props_a, props_b, action) -> predicted result vec``.
Unlike :class:`~artificial_society.systems.culture.CausalMemory` (a flat contingency
table over surface material labels), this model is parametrised over the **latent
property dimensions** (``PROP_DIMS``), so what it learns generalises to material pairs
it has never combined. That generalisation IS the causal-theory test (see
``docs/research/causal-model-proto/DESIGN.md``).

Determinism: the MLP is seeded from a local ``np.random.Generator`` (seeded per agent
from ``agent.id``), kept isolated from the global RNG stream so it never perturbs the
golden trajectory. The whole module is dormant unless ``AS_CAUSAL_MODEL`` is set —
``_cm_enabled()`` mirrors ``need_driven_invention._m1_enabled()``.
"""

from __future__ import annotations

import os

import numpy as np

N_PROPS = 12
N_ACTIONS = 8
_INPUT_DIM = 2 * N_PROPS + N_ACTIONS  # [props_a | props_b | action_onehot] = 32


def _cm_enabled() -> bool:
    """True only when ``AS_CAUSAL_MODEL`` is set to a truthy value.

    OFF (default) means agents never instantiate a model, so no extra global RNG is
    drawn and the determinism golden trajectory + headless digest stay byte-identical.
    """
    return os.environ.get("AS_CAUSAL_MODEL", "") not in ("", "0", "false", "False")


# Weight of the L5 epistemic (prediction-error) term in the invention reward.
EPISTEMIC_REWARD_SCALE = 0.5


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class CausalModel:
    """A per-agent online-learned forward model of material combination outcomes."""

    def __init__(
        self,
        n_props: int = N_PROPS,
        n_actions: int = N_ACTIONS,
        hidden: int = 32,
        lr: float = 0.05,
        ema_alpha: float = 0.1,
        seed: int = 0,
    ):
        self.n_props = n_props
        self.n_actions = n_actions
        self.lr = lr
        self.ema_alpha = ema_alpha
        in_dim = 2 * n_props + n_actions
        # local generator, isolated from the global RNG stream; persists for L3 selection
        self.gen = np.random.default_rng(seed)
        self.W1 = (self.gen.standard_normal((in_dim, hidden)) / np.sqrt(in_dim)).astype(np.float64)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = (self.gen.standard_normal((hidden, n_props)) / np.sqrt(hidden)).astype(np.float64)
        self.b2 = np.zeros(n_props, dtype=np.float64)
        self.err_ema = 1.0  # running prediction-error magnitude
        self._bucket_counts: dict[tuple, int] = {}

    # -- feature encoding -------------------------------------------------
    def _featurize(self, props_a: np.ndarray, props_b: np.ndarray | None, action_idx: int) -> np.ndarray:
        phi = np.zeros(2 * self.n_props + self.n_actions, dtype=np.float64)
        phi[: self.n_props] = np.asarray(props_a, dtype=np.float64)[: self.n_props]
        if props_b is not None:
            phi[self.n_props : 2 * self.n_props] = np.asarray(props_b, dtype=np.float64)[: self.n_props]
        a = int(action_idx) % self.n_actions
        phi[2 * self.n_props + a] = 1.0
        return phi

    def _forward(self, phi: np.ndarray):
        z1 = phi @ self.W1 + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        out = _sigmoid(z2)
        return out, h

    # -- public API -------------------------------------------------------
    def predict(self, props_a: np.ndarray, props_b: np.ndarray | None, action_idx: int) -> np.ndarray:
        out, _ = self._forward(self._featurize(props_a, props_b, action_idx))
        return out.astype(np.float32)

    def observe(self, props_a: np.ndarray, props_b: np.ndarray | None, action_idx: int, y: np.ndarray) -> float:
        """One online SGD step toward observed outcome ``y``. Returns the pre-update
        prediction-error magnitude — the L5 "surprise" / epistemic signal."""
        phi = self._featurize(props_a, props_b, action_idx)
        out, h = self._forward(phi)
        target = np.asarray(y, dtype=np.float64)[: self.n_props]
        err = out - target
        surprise = float(np.linalg.norm(err))

        # backprop of 0.5 * ||out - y||^2 through sigmoid -> tanh -> linear
        dz2 = err * out * (1.0 - out)
        dW2 = np.outer(h, dz2)
        db2 = dz2
        dh = dz2 @ self.W2.T
        dz1 = dh * (1.0 - h * h)
        dW1 = np.outer(phi, dz1)
        db1 = dz1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        self.err_ema = (1.0 - self.ema_alpha) * self.err_ema + self.ema_alpha * surprise
        key = self._bucket_key(props_a, props_b, action_idx)
        self._bucket_counts[key] = self._bucket_counts.get(key, 0) + 1
        return surprise

    def sensitivity(self, props_a: np.ndarray, props_b: np.ndarray | None, action_idx: int, eps: float = 1e-2) -> np.ndarray:
        """L2 readability: finite-difference Jacobian d(output)/d(input prop), shape
        ``(2*n_props, n_props)`` — a readable "which input property drives which outcome"
        table even though the model itself is a black-box MLP."""
        base = self.predict(props_a, props_b, action_idx).astype(np.float64)
        a = np.asarray(props_a, dtype=np.float64)[: self.n_props].copy()
        b = (
            np.asarray(props_b, dtype=np.float64)[: self.n_props].copy()
            if props_b is not None
            else np.zeros(self.n_props)
        )
        jac = np.zeros((2 * self.n_props, self.n_props), dtype=np.float64)
        for i in range(self.n_props):
            ap = a.copy()
            ap[i] += eps
            jac[i] = (self.predict(ap, b, action_idx).astype(np.float64) - base) / eps
        for i in range(self.n_props):
            bp = b.copy()
            bp[i] += eps
            jac[self.n_props + i] = (self.predict(a, bp, action_idx).astype(np.float64) - base) / eps
        return jac

    def info_gain(self, props_a: np.ndarray, props_b: np.ndarray | None, action_idx: int) -> float:
        """L3: expected information gain of trying this combination — a count-based
        novelty bonus over the coarse property region plus the model's current error
        EMA. High for unexplored regions, decays as a region is revisited."""
        key = self._bucket_key(props_a, props_b, action_idx)
        count = self._bucket_counts.get(key, 0)
        novelty = 1.0 / np.sqrt(1.0 + count)
        return float(novelty + 0.1 * self.err_ema)

    # -- internal ---------------------------------------------------------
    def _bucket_key(self, props_a: np.ndarray, props_b: np.ndarray | None, action_idx: int) -> tuple:
        a = np.asarray(props_a, dtype=np.float64)[: self.n_props]
        b = (
            np.asarray(props_b, dtype=np.float64)[: self.n_props]
            if props_b is not None
            else np.zeros(self.n_props)
        )
        qa = tuple(np.round(a * 2.0) / 2.0)
        qb = tuple(np.round(b * 2.0) / 2.0)
        return (qa, qb, int(action_idx) % self.n_actions)


# ---------------------------------------------------------------------------
# Integration helpers (systems-lane glue — no hot-file edits)
# ---------------------------------------------------------------------------


def get_or_create_model(agent) -> CausalModel | None:
    """Return the agent's causal model, lazily creating it on first use.

    Returns ``None`` when the flag is OFF — so no model is attached, no RNG is drawn,
    and the agent dataclass is untouched (we never edit the hot ``agents/agent.py``).
    The model is seeded from ``agent.id`` via a local generator, keeping it reproducible
    and isolated from the global RNG stream.
    """
    if not _cm_enabled():
        return None
    model = getattr(agent, "causal_model", None)
    if model is None:
        model = CausalModel(seed=abs(int(getattr(agent, "id", 0))))
        agent.causal_model = model
    return model


def causal_model_step(agent, props_a, props_b, action_idx: int, observed_vec) -> float:
    """L4 + L5: predict the outcome, observe the real one, learn, and return the
    epistemic (prediction-error) reward. No-op returning ``0.0`` when the flag is OFF.
    """
    model = get_or_create_model(agent)
    if model is None:
        return 0.0
    surprise = model.observe(props_a, props_b, action_idx, observed_vec)
    return EPISTEMIC_REWARD_SCALE * surprise


def select_by_info_gain(
    model: CausalModel, candidates: list, gen: np.random.Generator, temp: float = 1.0
) -> int:
    """L3: pick the INDEX of a ``(props_a, props_b, action_idx)`` candidate via a SEEDED
    softmax over expected information gain — deliberately not argmax (greedy collapses
    diversity, the documented C+D failure mode). Returns ``-1`` for no candidates.
    ``gen`` is the agent's local generator, so the choice is reproducible and isolated
    from the global RNG stream. An index (not the object) is returned so callers can map
    it onto a parallel list of material names/actions.
    """
    if not candidates:
        return -1
    scores = np.array([model.info_gain(*c) for c in candidates], dtype=np.float64)
    scores -= scores.max()
    expv = np.exp(scores / max(1e-6, temp))
    probs = expv / expv.sum()
    return int(gen.choice(len(candidates), p=probs))
