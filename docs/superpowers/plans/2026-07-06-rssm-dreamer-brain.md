# RSSM (Dreamer-v3) Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the flag-gated `brain_arch="rssm"` experimental lane — one shared Dreamer-v3 RSSM world model trained on pooled population experience, per-agent actor-critics trained in latent imagination — per the reviewed spec `docs/superpowers/specs/2026-07-04-rssm-dreamer-brain-design.md` (@ 652bf6b).

**Architecture:** New isolated module tree `artificial_society/agents/rssm/` (config, util, world_model, replay, actor_critic, prototype, learner). Actors live in a pre-allocated max-population **slab** of stacked tensors with hand-vectorized batched linears and elementwise slab-Adam. Integration touches hot files only at 4 small seams (construction, `Agent.update` kwarg+dispatch, `ensure_fields` branch, checkpoint); the every-K-steps training hook is a **registered system** (`systems/rssm_learning.py`) via `registry.tick_systems` — no `step()` edit.

**Tech Stack:** Python 3.9, torch 2.8.0 (CPU act path; FP32 CUDA for WM-train + imagination), pytest. Repo conventions: `from __future__ import annotations`, dataclasses, typing; ruff auto-format hook.

## Global Constraints

- **Determinism is sacred:** rssm code draws ONLY from rssm-owned `torch.Generator`s (spec §6); never `random.*`, `np.random.*`, or global-torch sampling. Golden trajectory + headless digest must stay green after every task.
- **Zero-draw-when-off:** an off-arm (`brain_arch != "rssm"`) run must consume zero extra torch RNG and leave `torch.get_rng_state()` unchanged through `Simulation.__init__`.
- **Hot files** (`simulation.py`, `agents/agent.py`, `agents/brain.py`): edit only where a task explicitly says so, minimal diffs; `agents/brain.py` is NOT edited at all in this plan. Never import `brain.py`'s `USE_FP16`/`device`; never call `imagine_rollout`.
- **v1/v2 untouched:** all existing tests keep passing (`venv/bin/python -m pytest -q` from repo root, 356+ green).
- Test runner: `venv/bin/python -m pytest -q` (conftest forces headless SDL). Headless sim pattern: `Simulation(headless=True, load_checkpoint=False, seed=42, grid_w=20, grid_h=15, initial_population=8)`.
- All rssm hyperparameters live ONLY in `RSSMConfig` (spec §9); no magic numbers in module code.
- FP32 everywhere; act path pinned to CPU regardless of config device.
- Commit after every task (lane-prefixed branch `experiment/rssm-dreamer-brain` is already checked out; commit directly).

## File Structure

```
artificial_society/agents/rssm/
  __init__.py        # empty marker
  config.py          # RSSMConfig dataclass (all spec-§9 values) + make_generator()
  util.py            # symlog/symexp, twohot encode/decode (fixed 255 bins), MLP factory, zero-init
  world_model.py     # RSSMWorldModel: encoder/posterior, prior, GRUCell, grouped decoder, reward/continue heads, train_batch
  replay.py          # SharedReplay: per-life episodes, B×L padded+masked sampling, stratified deaths, per-agent starts
  actor_critic.py    # ActorCriticSlab: stacked params, batched linears, slab-Adam, imagination_update (reinforce default)
  prototype.py       # PrototypeActor: EMA of alive slots, newborn warm-start source
  learner.py         # SharedLearner: act path, store, spawn/death hooks, train cadence, checkpoint payload
artificial_society/systems/rssm_learning.py   # @register'd system: ticks learner.maybe_train (order after builtins)
scripts/rssm_ab.py                            # 3-arm A/B runner + RMST/KM analysis
docs/experiments/rssm-ab-prereg.md            # pre-registration file (Task 10)
tests/agents/rssm/                            # unit tests per module
tests/test_rssm_integration.py                # gating/golden/smoke/checkpoint tests
```

**Interface conventions used throughout (memorize):**
- Model state: `h ∈ (B,256)` float32, `z ∈ (B,32,32)` one-hot float32; flattened `z_flat ∈ (B,1024)`; `s = cat(h, z_flat) ∈ (B,1280)`; head input `s_g = cat(s, genes) ∈ (B,1284)`.
- Genes: the 4-dim slice `obs[..., 17:21]` (static per life; `cfg.gene_slice`); verified in Task 3 against `brain.py:24-41`.
- Agent-side carried state: `agent.hidden_state = (h(1,256), z(1,32,32))` CPU tensors; `agent.rssm_slot: int`; `agent.spawn_origin: str` in `{"initial","birth","respawn"}`; `agent.brain_arch: str`.
- `learner.act(agent, features)` returns the v1-compatible contract dict: keys `action_list` (list of 7 floats), `next_hidden` ((h,z) tuple), `obs_tensor` ((57,) float32), `action_tensor` ((7,) float32).

---

### Task 1: Scaffold + RSSMConfig + RNG isolation helper

**Files:**
- Create: `artificial_society/agents/rssm/__init__.py` (empty)
- Create: `artificial_society/agents/rssm/config.py`
- Test: `tests/agents/rssm/test_config.py` (+ empty `tests/agents/rssm/__init__.py` NOT needed — pytest uses rootdir conftest)

**Interfaces:**
- Produces: `RSSMConfig` (frozen dataclass, defaults below — later tasks import these exact field names), `make_generator(global_seed: int, key: object) -> torch.Generator`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rssm/test_config.py
from __future__ import annotations

import torch

from artificial_society.agents.rssm.config import RSSMConfig, make_generator


def test_defaults_match_spec():
    cfg = RSSMConfig()
    assert (cfg.obs_dim, cfg.action_dim) == (57, 7)
    assert (cfg.deter, cfg.stoch, cfg.classes) == (256, 32, 32)
    assert (cfg.beta_pred, cfg.beta_dyn, cfg.beta_rep) == (1.0, 0.5, 0.1)
    assert cfg.unimix == 0.01
    assert (cfg.num_bins, cfg.bin_low, cfg.bin_high) == (255, -20.0, 20.0)
    assert (cfg.wm_lr, cfg.ac_lr, cfg.adam_eps) == (1e-4, 3e-5, 1e-8)
    assert (cfg.wm_clip, cfg.ac_clip) == (1000.0, 100.0)
    assert (cfg.batch_size, cfg.seq_len) == (16, 64)
    assert (cfg.horizon, cfg.gamma, cfg.lam) == (15, 0.997, 0.95)
    assert cfg.entropy_eta == 3e-4
    assert cfg.actor_grad == "reinforce"
    assert (cfg.retnorm_decay, cfg.slow_critic_decay) == (0.99, 0.98)
    assert cfg.replay_critic_scale == 0.3
    assert (cfg.train_every, cfg.young_ratio_cap) == (8, 4)
    assert cfg.prototype_decay == 0.995
    assert cfg.gene_slice == (17, 21)
    assert cfg.max_slots >= 128


def test_generator_isolated_from_global():
    torch.manual_seed(0)
    before = torch.get_rng_state()
    g = make_generator(42, "learner")
    _ = torch.rand(1000, generator=g)
    assert torch.equal(before, torch.get_rng_state())  # global untouched


def test_generator_deterministic_and_keyed():
    a1 = torch.rand(8, generator=make_generator(42, ("agent", 7)))
    a2 = torch.rand(8, generator=make_generator(42, ("agent", 7)))
    b = torch.rand(8, generator=make_generator(42, ("agent", 8)))
    c = torch.rand(8, generator=make_generator(43, ("agent", 7)))
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b)
    assert not torch.equal(a1, c)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_config.py -q`
Expected: FAIL (`ModuleNotFoundError: artificial_society.agents.rssm`)

- [ ] **Step 3: Implement**

```python
# artificial_society/agents/rssm/config.py
"""All RSSM hyperparameters (spec §9) + the RNG-isolation helper (spec §6)."""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch


def make_generator(global_seed: int, key: object) -> torch.Generator:
    """CPU torch.Generator seeded from (global_seed, key), independent of global RNG state.

    key is any repr-stable object, e.g. ("agent", agent_id) or "learner".
    """
    mix = zlib.crc32(repr(key).encode("utf-8"))
    gen = torch.Generator(device="cpu")
    gen.manual_seed((int(global_seed) * 0x9E3779B1 + mix) % (2**63))
    return gen


@dataclass(frozen=True)
class RSSMConfig:
    # substrate
    obs_dim: int = 57
    action_dim: int = 7
    gene_slice: tuple = (17, 21)          # obs dims carrying genes (verified Task 3)
    # latent sizes
    deter: int = 256
    stoch: int = 32
    classes: int = 32
    mlp_hidden: int = 256                  # WM MLP width
    ac_hidden: int = 128                   # actor/critic MLP width
    # WM loss (spec §4.1)
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    unimix: float = 0.01
    # decoder grouping (dims down-weighted in recon target; filled in Task 3)
    irreducible_dims: tuple = ()
    irreducible_weight: float = 0.05
    # twohot (fixed support, reward + value)
    num_bins: int = 255
    bin_low: float = -20.0
    bin_high: float = 20.0
    # optimizers
    wm_lr: float = 1e-4
    ac_lr: float = 3e-5
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    wm_clip: float = 1000.0
    ac_clip: float = 100.0
    # replay (spec §4.4)
    batch_size: int = 16
    seq_len: int = 64
    min_death_seqs: int = 2               # stratified: min sequences containing a terminal
    replay_capacity: int = 200_000        # transitions, FIFO by episode
    # imagination (spec §4.2)
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_eta: float = 3e-4
    actor_grad: str = "reinforce"          # "reinforce" | "dynamics"
    retnorm_decay: float = 0.99
    slow_critic_decay: float = 0.98
    slow_critic_scale: float = 1.0
    replay_critic_scale: float = 0.3
    # cadence (spec §4.5/§8)
    train_every: int = 8                   # K
    young_ratio_cap: int = 4               # max extra imagination passes for young actors
    young_age_ticks: int = 200             # "young" = age < this
    wm_warmup_updates: int = 100           # imagination gated until this many WM updates
    prefill_transitions: int = 4096        # random-action prefill before first WM update
    own_start_frac_max: float = 0.75       # per-agent imagination seeding blend (spec §4.2)
    # slab (spec §8)
    max_slots: int = 128
    prototype_decay: float = 0.995
    # devices
    train_device: str = "cuda"             # WM-train + imagination; act path is ALWAYS cpu
```

Note: `wm_warmup_updates` (update-count gate) replaces the spec's "loss threshold" for the imagination warm-up — deterministic and tuning-free; record this as a reviewed refinement in the results note.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_config.py -q`
Expected: 3 passed

- [ ] **Step 5: Full-suite sanity + commit**

Run: `venv/bin/python -m pytest -q` → all green (no existing test imports rssm).

```bash
git add artificial_society/agents/rssm/ tests/agents/rssm/
git commit -m "feat(rssm): scaffold, RSSMConfig, isolated RNG generators"
```

---

### Task 2: util.py — symlog, fixed-support twohot, MLP factory

**Files:**
- Create: `artificial_society/agents/rssm/util.py`
- Test: `tests/agents/rssm/test_util.py`

**Interfaces:**
- Produces: `symlog(x)`, `symexp(x)`; `make_bins(cfg) -> (num_bins,) tensor` (symlog-spaced support, i.e. `symexp(linspace(bin_low, bin_high))`... NO — bins live IN symlog space: `linspace(bin_low, bin_high, num_bins)`, targets are `symlog(value)`); `twohot(y_symlog, bins) -> (…, num_bins)`; `twohot_mean(logits, bins) -> (…,)` returning the **symexp'd** scalar expectation; `mlp(in_dim, hidden, out_dim, layers=2) -> nn.Sequential` (Linear→LayerNorm→SiLU blocks, plain Linear out); `zero_init_(linear)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rssm/test_util.py
from __future__ import annotations

import torch

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.agents.rssm.util import (
    make_bins, mlp, symexp, symlog, twohot, twohot_mean, zero_init_,
)

CFG = RSSMConfig()


def test_symlog_roundtrip():
    x = torch.tensor([-1e4, -3.7, 0.0, 0.5, 2e5])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4, atol=1e-5)


def test_twohot_roundtrip():
    bins = make_bins(CFG)
    assert bins.shape == (255,)
    vals = torch.tensor([-70.0, -3.0, 0.0, 0.25, 1.5, 400.0])
    probs = twohot(symlog(vals), bins)
    assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-6)
    # decode via log-probs path used at train time: twohot_mean expects logits
    logits = (probs + 1e-8).log()
    assert torch.allclose(twohot_mean(logits, bins), vals, rtol=1e-2, atol=1e-2)


def test_twohot_clamps_outside_support():
    bins = make_bins(CFG)
    probs = twohot(torch.tensor([999.0]), bins)  # symlog-space input beyond bin_high
    assert probs[0, -1] == 1.0


def test_mlp_shapes_and_zero_init():
    net = mlp(10, 32, 5)
    y = net(torch.randn(4, 10))
    assert y.shape == (4, 5)
    zero_init_(net[-1])
    assert torch.all(net[-1].weight == 0) and torch.all(net[-1].bias == 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_util.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement**

```python
# artificial_society/agents/rssm/util.py
"""Shared numeric primitives: symlog, fixed twohot support, MLP factory (spec §4.1)."""
from __future__ import annotations

import torch
from torch import nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def make_bins(cfg) -> torch.Tensor:
    """Fixed twohot support in symlog space (Dreamer-v3: 255 bins over [-20, 20])."""
    return torch.linspace(cfg.bin_low, cfg.bin_high, cfg.num_bins)


def twohot(y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """y: symlog-space targets (...,). Returns (..., num_bins) twohot distribution."""
    y = y.clamp(bins[0], bins[-1])
    idx = torch.searchsorted(bins, y.detach(), right=True).clamp(1, len(bins) - 1)
    lo, hi = bins[idx - 1], bins[idx]
    w_hi = ((y - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)
    out = torch.zeros(*y.shape, len(bins), dtype=torch.float32, device=y.device)
    out.scatter_(-1, (idx - 1).unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
    out.scatter_add_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


def twohot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value under softmax(logits) over the symlog support, mapped back via symexp."""
    return symexp((torch.softmax(logits, -1) * bins).sum(-1))


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """Linear→LayerNorm→SiLU ×layers, then plain Linear head."""
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def zero_init_(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_util.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add artificial_society/agents/rssm/util.py tests/agents/rssm/test_util.py
git commit -m "feat(rssm): symlog, fixed-support twohot, MLP factory"
```

---

### Task 3: RSSMWorldModel

**Files:**
- Modify: `artificial_society/agents/rssm/config.py` (add `device` param to `make_generator`; fill `irreducible_dims`)
- Create: `artificial_society/agents/rssm/world_model.py`
- Test: `tests/agents/rssm/test_world_model.py`

**Interfaces:**
- Consumes: `RSSMConfig`, `util` (Task 1/2).
- Produces: `RSSMWorldModel(cfg)` with:
  - `initial_state(batch: int, device) -> (h(B,256), z(B,32,32))` (zeros)
  - `obs_step(h, z, a_prev, obs, gen) -> (h', z_post, post_logits(B,32,32), prior_logits)`
  - `img_step(h, z, a, gen) -> (h', z_prior, prior_logits)`
  - `head_input(h, z, genes4) -> s_g(B,1284)`; `decode(s_g) -> (B,57)`; `reward_logits(s_g) -> (B,255)`; `cont_prob(s_g) -> (B,)`
  - `train_batch(batch: dict, gen) -> dict` — batch keys `obs(B,L,57), act(B,L,7), reward(B,L), cont(B,L), mask(B,L)`; runs one optimizer step; returns float metrics `{"loss","recon","reward","cont","kl_dyn","kl_rep"}`
  - `observe(obs, act, gen) -> (h_seq(B,L,256), z_seq(B,L,32,32), post_logits, prior_logits)` (used by train_batch and by the learner's burn-in re-encode)
  - Attribute `wm_updates: int` (incremented per `train_batch`; the learner's warm-up gate reads it).

- [ ] **Step 1: Extend `make_generator` with a device parameter**

In `config.py`, change the two lines:

```python
def make_generator(global_seed: int, key: object, device: str = "cpu") -> torch.Generator:
    ...
    gen = torch.Generator(device=device)
```

Add to `tests/agents/rssm/test_config.py`:

```python
def test_generator_device_param_default_cpu():
    assert make_generator(1, "x").device.type == "cpu"
```

Run: `venv/bin/python -m pytest tests/agents/rssm/test_config.py -q` → 4 passed.

- [ ] **Step 2: Derive the decoder partition + verify the gene slice (spec §4.1)**

Read the obs layout comment at `artificial_society/agents/brain.py:24-41` (read-only — brain.py is never edited):

Run: `sed -n '20,45p' artificial_society/agents/brain.py`

From the labeled dim ranges, set in `RSSMConfig`:
- `gene_slice`: the range labeled genes (spec says @17-20 → `(17, 21)`; correct it here if the comment says otherwise).
- `irreducible_dims`: the tuple of all dims labeled social-nearby / episodic-retrieval (the groups the spec calls irreducible-from-egocentric-view). Write the actual integers as the new default.

Then add this staticness test (belongs in `tests/agents/rssm/test_world_model.py`):

```python
def test_gene_slice_static_over_life():
    from artificial_society.simulation import Simulation
    from artificial_society.agents.rssm.config import RSSMConfig
    sim = Simulation(headless=True, load_checkpoint=False, seed=42,
                     grid_w=20, grid_h=15, initial_population=8)
    ag = sim.agents[0]
    lo, hi = RSSMConfig().gene_slice
    f0 = ag.local_features(sim.world, sim.agents)[lo:hi]
    for _ in range(5):
        sim.step()
    if ag.alive:
        f1 = ag.local_features(sim.world, sim.agents)[lo:hi]
        assert f0 == f1  # genes never change within a life
```

- [ ] **Step 3: Write the failing model tests**

```python
# tests/agents/rssm/test_world_model.py  (add to the file from Step 2)
from __future__ import annotations

import torch

from artificial_society.agents.rssm.config import RSSMConfig, make_generator
from artificial_society.agents.rssm.world_model import RSSMWorldModel

CFG = RSSMConfig()


def _batch(B=3, L=5, gen=None):
    g = gen or make_generator(0, "data")
    return {
        "obs": torch.rand(B, L, CFG.obs_dim, generator=g),
        "act": torch.rand(B, L, CFG.action_dim, generator=g) * 2 - 1,
        "reward": torch.rand(B, L, generator=g),
        "cont": torch.ones(B, L),
        "mask": torch.ones(B, L),
    }


def test_shapes_and_determinism():
    wm = RSSMWorldModel(CFG)
    b = _batch()
    h, z, post, prior = wm.observe(b["obs"], b["act"], make_generator(7, "s"))
    assert h.shape == (3, 5, 256) and z.shape == (3, 5, 32, 32)
    assert post.shape == prior.shape == (3, 5, 32, 32)
    assert torch.all(z.sum(-1) == 1.0)  # one-hot per categorical
    h2, z2, _, _ = wm.observe(b["obs"], b["act"], make_generator(7, "s"))
    assert torch.equal(z, z2)  # same generator seed → identical samples


def test_kl_free_bits_floor():
    wm = RSSMWorldModel(CFG)
    logits = torch.zeros(4, 6, 32, 32)
    kl_dyn, kl_rep = wm.kl_losses(logits, logits.clone(), torch.ones(4, 6))
    assert abs(kl_dyn.item() - 1.0) < 1e-5  # identical dists → clamped to 1 nat
    assert abs(kl_rep.item() - 1.0) < 1e-5


def test_train_batch_steps_and_counts():
    wm = RSSMWorldModel(CFG)
    before = [p.clone() for p in wm.parameters()]
    m = wm.train_batch(_batch(), make_generator(1, "t"))
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values())
    assert wm.wm_updates == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, wm.parameters()))


def test_decoder_weights_downweight_irreducible():
    wm = RSSMWorldModel(CFG)
    w = wm.recon_weight
    for d in CFG.irreducible_dims:
        assert w[d] == CFG.irreducible_weight
    keep = [i for i in range(CFG.obs_dim) if i not in CFG.irreducible_dims]
    assert torch.all(w[keep] == 1.0)


def test_reward_head_zero_init():
    wm = RSSMWorldModel(CFG)
    s_g = torch.randn(2, 256 + 32 * 32 + 4)
    assert torch.all(wm.reward_logits(s_g) == 0)  # zero-init final layer
```

Run: `venv/bin/python -m pytest tests/agents/rssm/test_world_model.py -q`
Expected: FAIL (`ImportError: RSSMWorldModel`)

- [ ] **Step 4: Implement**

```python
# artificial_society/agents/rssm/world_model.py
"""Shared Dreamer-v3-style RSSM world model (spec §4.1). One instance, sim-owned."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .util import make_bins, mlp, symlog, twohot, zero_init_


def st_sample(logits: torch.Tensor, unimix: float, gen: torch.Generator) -> torch.Tensor:
    """Straight-through one-hot sample from (B, stoch, classes) logits with unimix."""
    probs = torch.softmax(logits, -1)
    probs = (1.0 - unimix) * probs + unimix / probs.shape[-1]
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, 1, generator=gen).reshape(probs.shape[:-1])
    onehot = torch.nn.functional.one_hot(idx, probs.shape[-1]).float()
    return onehot + probs - probs.detach()  # straight-through gradients


def _unimix_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    p = torch.softmax(logits, -1)
    return (1.0 - unimix) * p + unimix / p.shape[-1]


class RSSMWorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        z_dim = cfg.stoch * cfg.classes
        gene_dim = cfg.gene_slice[1] - cfg.gene_slice[0]
        head_in = cfg.deter + z_dim + gene_dim
        self.embed = mlp(cfg.obs_dim, cfg.mlp_hidden, cfg.mlp_hidden)
        self.post_net = nn.Linear(cfg.deter + cfg.mlp_hidden, z_dim)
        self.prior_net = mlp(cfg.deter, cfg.mlp_hidden, z_dim)
        self.pre_gru = mlp(z_dim + cfg.action_dim, cfg.mlp_hidden, cfg.mlp_hidden, layers=1)
        self.gru = nn.GRUCell(cfg.mlp_hidden, cfg.deter)
        self.decoder = mlp(head_in, cfg.mlp_hidden, cfg.obs_dim)
        self.reward_head = mlp(head_in, cfg.mlp_hidden, cfg.num_bins)
        zero_init_(self.reward_head[-1])
        self.cont_head = mlp(head_in, cfg.mlp_hidden, 1)
        self.register_buffer("bins", make_bins(cfg))
        w = torch.ones(cfg.obs_dim)
        for d in cfg.irreducible_dims:
            w[d] = cfg.irreducible_weight
        self.register_buffer("recon_weight", w)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.wm_lr, eps=cfg.adam_eps,
                                    betas=(cfg.adam_beta1, cfg.adam_beta2))
        self.wm_updates = 0

    # --- state machinery -------------------------------------------------
    def initial_state(self, batch: int, device=None):
        c = self.cfg
        dev = device or next(self.parameters()).device
        return (torch.zeros(batch, c.deter, device=dev),
                torch.zeros(batch, c.stoch, c.classes, device=dev))

    def _advance(self, h, z, a_prev):
        x = self.pre_gru(torch.cat([z.flatten(1), a_prev], -1))
        return self.gru(x, h)

    def obs_step(self, h, z, a_prev, obs, gen):
        h = self._advance(h, z, a_prev)
        prior_logits = self.prior_net(h).reshape(-1, self.cfg.stoch, self.cfg.classes)
        e = self.embed(symlog(obs))
        post_logits = self.post_net(torch.cat([h, e], -1)).reshape(-1, self.cfg.stoch, self.cfg.classes)
        z_post = st_sample(post_logits, self.cfg.unimix, gen)
        return h, z_post, post_logits, prior_logits

    def img_step(self, h, z, a, gen):
        h = self._advance(h, z, a)
        prior_logits = self.prior_net(h).reshape(-1, self.cfg.stoch, self.cfg.classes)
        z_prior = st_sample(prior_logits, self.cfg.unimix, gen)
        return h, z_prior, prior_logits

    def observe(self, obs, act, gen):
        B, L, _ = obs.shape
        h, z = self.initial_state(B, obs.device)
        hs, zs, posts, priors = [], [], [], []
        a_prev = torch.zeros(B, self.cfg.action_dim, device=obs.device)
        for t in range(L):
            h, z, post_l, prior_l = self.obs_step(h, z, a_prev, obs[:, t], gen)
            a_prev = act[:, t]
            hs.append(h); zs.append(z); posts.append(post_l); priors.append(prior_l)
        stack = lambda xs: torch.stack(xs, 1)
        return stack(hs), stack(zs), stack(posts), stack(priors)

    # --- heads ------------------------------------------------------------
    def head_input(self, h, z, genes):
        return torch.cat([h, z.flatten(-2), genes], -1)

    def decode(self, s_g):
        return self.decoder(s_g)

    def reward_logits(self, s_g):
        return self.reward_head(s_g)

    def cont_prob(self, s_g):
        return torch.sigmoid(self.cont_head(s_g)).squeeze(-1)

    # --- losses -----------------------------------------------------------
    def kl_losses(self, post_logits, prior_logits, mask):
        c = self.cfg
        q, qs = _unimix_probs(post_logits, c.unimix), _unimix_probs(post_logits.detach(), c.unimix)
        p, ps = _unimix_probs(prior_logits, c.unimix), _unimix_probs(prior_logits.detach(), c.unimix)
        kl = lambda a, b: (a * (a.clamp_min(1e-8).log() - b.clamp_min(1e-8).log())).sum(-1).sum(-1)
        # per-timestep clamp (free bits) BEFORE averaging; mask out padding
        dyn = (torch.clamp(kl(qs, p), min=1.0) * mask).sum() / mask.sum()
        rep = (torch.clamp(kl(q, ps), min=1.0) * mask).sum() / mask.sum()
        return dyn, rep

    def train_batch(self, batch, gen):
        c = self.cfg
        obs, act = batch["obs"], batch["act"]
        mask = batch["mask"]
        h, z, post_l, prior_l = self.observe(obs, act, gen)
        genes = obs[..., c.gene_slice[0]:c.gene_slice[1]]
        s_g = self.head_input(h, z, genes)
        recon = ((self.decode(s_g) - symlog(obs)) ** 2 * self.recon_weight).mean(-1)
        recon = (recon * mask).sum() / mask.sum()
        target = twohot(symlog(batch["reward"]), self.bins)
        logp = torch.log_softmax(self.reward_logits(s_g), -1)
        rew_loss = (-(target * logp).sum(-1) * mask).sum() / mask.sum()
        cont_logit = self.cont_head(s_g).squeeze(-1)
        cont_loss = nn.functional.binary_cross_entropy_with_logits(
            cont_logit, batch["cont"], reduction="none")
        cont_loss = (cont_loss * mask).sum() / mask.sum()
        kl_dyn, kl_rep = self.kl_losses(post_l, prior_l, mask)
        loss = (c.beta_pred * (recon + rew_loss + cont_loss)
                + c.beta_dyn * kl_dyn + c.beta_rep * kl_rep)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.parameters(), c.wm_clip)
        self.opt.step()
        self.wm_updates += 1
        return {"loss": loss.item(), "recon": recon.item(), "reward": rew_loss.item(),
                "cont": cont_loss.item(), "kl_dyn": kl_dyn.item(), "kl_rep": kl_rep.item()}
```

Note for the implementer: `train_batch` is called with tensors already on `cfg.train_device` and a generator created with that device (Task 7); unit tests run everything on CPU.

- [ ] **Step 5: Run tests**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_world_model.py tests/agents/rssm/test_config.py -q`
Expected: all passed (the staticness test constructs a sim — a few seconds).

- [ ] **Step 6: Commit**

```bash
git add artificial_society/agents/rssm/ tests/agents/rssm/
git commit -m "feat(rssm): RSSM world model — grouped decoder, twohot reward, KL balancing + free bits"
```

---

### Task 4: SharedReplay

**Files:**
- Modify: `artificial_society/agents/rssm/config.py` (add `burn_in: int = 8`)
- Create: `artificial_society/agents/rssm/replay.py`
- Test: `tests/agents/rssm/test_replay.py`

**Interfaces:**
- Produces: `SharedReplay(cfg)` with:
  - `start_episode(agent_id: int, origin: str)`
  - `add(agent_id, obs: list[float] | Tensor(57,), action: Tensor(7,), reward: float, done: bool)` (auto-closes the episode when `done=True`)
  - `end_episode(agent_id)` (idempotent; for despawn-without-death)
  - `__len__() -> int` total stored transitions
  - `num_death_episodes: int`
  - `sample_sequences(gen) -> dict` — keys `obs(B,L,57), act(B,L,7), reward(B,L), cont(B,L), mask(B,L)`; per-life windows only; ≥`min_death_seqs` windows containing a terminal when available; FIFO capacity by whole episodes
  - `sample_starts(agent_id, n, own_frac, gen) -> (obs(n,burn,57), act(n,burn,7)) | None` — burn-in windows for imagination seeding, `own_frac` of them from `agent_id`'s episodes, rest pooled; `None` if replay has no full burn-in window yet.

- [ ] **Step 1: Add `burn_in: int = 8` to `RSSMConfig`** (in the "imagination" block) and this line to `test_defaults_match_spec`: `assert cfg.burn_in == 8`. Run config test → green.

- [ ] **Step 2: Write the failing tests**

```python
# tests/agents/rssm/test_replay.py
from __future__ import annotations

import torch

from artificial_society.agents.rssm.config import RSSMConfig, make_generator
from artificial_society.agents.rssm.replay import SharedReplay

CFG = RSSMConfig()


def _fill(rp, agent_id, n, die=False, mark=0.0):
    rp.start_episode(agent_id, "birth")
    for i in range(n):
        done = die and i == n - 1
        rp.add(agent_id, torch.full((57,), mark), torch.zeros(7), float(i), done)
    if not die:
        rp.end_episode(agent_id)


def test_windows_never_cross_life_seams():
    rp = SharedReplay(CFG)
    _fill(rp, 1, 40, die=True, mark=1.0)   # life A, obs==1
    _fill(rp, 1, 40, die=True, mark=2.0)   # life B (same agent reborn), obs==2
    batch = rp.sample_sequences(make_generator(0, "s"))
    valid = batch["mask"].bool()
    per_seq_vals = [batch["obs"][b][valid[b]][:, 0].unique() for b in range(batch["obs"].shape[0])]
    assert all(len(v) == 1 for v in per_seq_vals)  # one life per window


def test_padding_masked_and_cont_semantics():
    rp = SharedReplay(CFG)
    _fill(rp, 3, 10, die=True)             # short life → padded window
    batch = rp.sample_sequences(make_generator(1, "s"))
    m = batch["mask"]
    assert m.shape == (CFG.batch_size, CFG.seq_len)
    assert (m.sum(1) <= CFG.seq_len).all() and (m.sum(1) >= 1).all()
    # cont==0 exactly at real deaths, never at window truncation/padding
    deaths = (batch["cont"] == 0) & m.bool()
    assert deaths.sum() >= CFG.min_death_seqs
    assert ((batch["cont"] == 0) & ~m.bool()).sum() == 0


def test_stratified_death_sampling():
    rp = SharedReplay(CFG)
    for a in range(20):
        _fill(rp, a, 80, die=False)        # many survivor episodes
    _fill(rp, 99, 30, die=True)            # one death episode
    for seed in range(5):
        batch = rp.sample_sequences(make_generator(seed, "s"))
        has_death = ((batch["cont"] == 0) & batch["mask"].bool()).any(1)
        assert has_death.sum() >= min(CFG.min_death_seqs, 1)


def test_sample_starts_blend_and_none_when_empty():
    rp = SharedReplay(CFG)
    assert rp.sample_starts(5, 4, 0.5, make_generator(0, "s")) is None
    _fill(rp, 5, 30, mark=5.0)
    _fill(rp, 6, 30, mark=6.0)
    obs, act = rp.sample_starts(5, 8, own_frac=1.0, gen=make_generator(2, "s"))
    assert obs.shape == (8, CFG.burn_in, 57) and act.shape == (8, CFG.burn_in, 7)
    assert (obs[:, :, 0] == 5.0).all()     # own_frac=1 → all windows from agent 5


def test_fifo_eviction_whole_episodes():
    import dataclasses
    cfg = dataclasses.replace(CFG, replay_capacity=100)
    rp = SharedReplay(cfg)
    _fill(rp, 1, 60, mark=1.0)
    _fill(rp, 2, 60, mark=2.0)             # pushes over 100 → episode 1 evicted whole
    assert len(rp) == 60
```

Run: `venv/bin/python -m pytest tests/agents/rssm/test_replay.py -q` → FAIL (ImportError)

- [ ] **Step 3: Implement**

```python
# artificial_society/agents/rssm/replay.py
"""Per-agent-life segmented sequence replay (spec §4.4). Stores raw transitions only."""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

import torch


@dataclass
class _Episode:
    agent_id: int
    origin: str
    obs: list = field(default_factory=list)      # (57,) float32 tensors
    act: list = field(default_factory=list)      # (7,) float32 tensors
    reward: list = field(default_factory=list)   # floats
    done: bool = False                            # True iff ended by real death
    open: bool = True

    def __len__(self):
        return len(self.obs)


class SharedReplay:
    def __init__(self, cfg):
        self.cfg = cfg
        self._episodes: deque[_Episode] = deque()        # closed episodes, FIFO
        self._open: OrderedDict[int, _Episode] = OrderedDict()
        self._size = 0

    # --- writing ----------------------------------------------------------
    def start_episode(self, agent_id: int, origin: str) -> None:
        self._open[agent_id] = _Episode(agent_id, origin)

    def add(self, agent_id, obs, action, reward, done) -> None:
        ep = self._open.get(agent_id)
        if ep is None:                                    # tolerate missed start
            ep = _Episode(agent_id, "unknown")
            self._open[agent_id] = ep
        ep.obs.append(torch.as_tensor(obs, dtype=torch.float32))
        ep.act.append(torch.as_tensor(action, dtype=torch.float32))
        ep.reward.append(float(reward))
        self._size += 1
        if done:
            ep.done = True
            self.end_episode(agent_id)

    def end_episode(self, agent_id: int) -> None:
        ep = self._open.pop(agent_id, None)
        if ep is None or len(ep) == 0:
            return
        ep.open = False
        self._episodes.append(ep)
        while self._size > self.cfg.replay_capacity and len(self._episodes) > 1:
            evicted = self._episodes.popleft()
            self._size -= len(evicted)

    def __len__(self) -> int:
        return self._size

    @property
    def num_death_episodes(self) -> int:
        return sum(1 for e in self._episodes if e.done)

    # --- sampling ---------------------------------------------------------
    def _all(self):
        return list(self._episodes) + [e for e in self._open.values() if len(e) > 0]

    def _window(self, ep: _Episode, gen) -> tuple:
        L = self.cfg.seq_len
        n = len(ep)
        start = 0 if n <= L else int(torch.randint(0, n - L + 1, (1,), generator=gen))
        end = min(start + L, n)
        k = end - start
        obs = torch.zeros(L, self.cfg.obs_dim)
        act = torch.zeros(L, self.cfg.action_dim)
        rew = torch.zeros(L)
        cont = torch.ones(L)
        mask = torch.zeros(L)
        obs[:k] = torch.stack(ep.obs[start:end])
        act[:k] = torch.stack(ep.act[start:end])
        rew[:k] = torch.tensor(ep.reward[start:end])
        mask[:k] = 1.0
        if ep.done and end == n:
            cont[k - 1] = 0.0                             # real death only
        return obs, act, rew, cont, mask

    def sample_sequences(self, gen) -> dict:
        eps = self._all()
        if not eps:
            raise ValueError("empty replay")
        B = self.cfg.batch_size
        death_eps = [e for e in eps if e.done]
        picks = []
        for _ in range(min(self.cfg.min_death_seqs, len(death_eps))):
            picks.append(death_eps[int(torch.randint(0, len(death_eps), (1,), generator=gen))])
        while len(picks) < B:
            picks.append(eps[int(torch.randint(0, len(eps), (1,), generator=gen))])
        # deaths must land inside the window: bias those windows to the episode tail
        cols = [self._window(e, gen) for e in picks[self.cfg.min_death_seqs:]]
        for e in picks[:min(self.cfg.min_death_seqs, len(death_eps))]:
            L = self.cfg.seq_len
            start = max(0, len(e) - L)
            tail = _Episode(e.agent_id, e.origin, e.obs[start:], e.act[start:], e.reward[start:], e.done, False)
            cols.insert(0, self._window(tail, gen))
        obs, act, rew, cont, mask = (torch.stack(x) for x in zip(*cols))
        return {"obs": obs, "act": act, "reward": rew, "cont": cont, "mask": mask}

    def sample_starts(self, agent_id, n, own_frac, gen):
        b = self.cfg.burn_in
        pool = [e for e in self._all() if len(e) >= b]
        if not pool:
            return None
        own = [e for e in pool if e.agent_id == agent_id] or pool
        n_own = int(round(n * own_frac))
        srcs = [own[int(torch.randint(0, len(own), (1,), generator=gen))] for _ in range(n_own)]
        srcs += [pool[int(torch.randint(0, len(pool), (1,), generator=gen))] for _ in range(n - n_own)]
        obs_w, act_w = [], []
        for e in srcs:
            s = int(torch.randint(0, len(e) - b + 1, (1,), generator=gen))
            obs_w.append(torch.stack(e.obs[s:s + b]))
            act_w.append(torch.stack(e.act[s:s + b]))
        return torch.stack(obs_w), torch.stack(act_w)
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_replay.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add artificial_society/agents/rssm/ tests/agents/rssm/
git commit -m "feat(rssm): SharedReplay — per-life windows, stratified deaths, burn-in starts"
```

---

### Task 5: ActorCriticSlab — batched actors, slab-Adam, imagination update

**Files:**
- Create: `artificial_society/agents/rssm/actor_critic.py`
- Test: `tests/agents/rssm/test_actor_critic.py`

**Interfaces:**
- Consumes: `RSSMWorldModel.img_step/head_input/reward_logits/cont_prob`, `util.twohot/twohot_mean/symlog`, `make_generator`.
- Produces: `ActorCriticSlab(cfg, gen)` with:
  - `params`: dict of stacked tensors, per layer `(max_slots, out, in)` weights + `(max_slots, out)` biases — actor: `a_w1,a_b1,a_w2,a_b2,a_mu_w,a_mu_b,a_std_w,a_std_b`; critic: `c_w1,c_b1,c_w2,c_b2,c_out_w,c_out_b` (critic `c_out` zero-initialized; `slow_` copies of all critic tensors for the EMA critic)
  - `alive: torch.BoolTensor(max_slots)`; `step_count: torch.LongTensor(max_slots)`
  - `acquire_slot(init_params: dict | None) -> int` (zeroes/initializes params + both Adam moments + step count; init from prototype dict when given)
  - `release_slot(slot: int)`
  - `act_single(slot, s_g(1,1284), gen) -> (action(7,), None)` — CPU, no_grad, tanh-Normal sample
  - `imagination_update(wm, h0(N,256), z0(N,32,32), genes(N,4), slots: LongTensor(N), gen, replay_batch: dict | None) -> dict` — full Dreamer update (rollout H, λ-returns, reinforce or dynamics loss, entropy, critic twohot + slow-critic reg + optional replay-critic term, slab-Adam step masked to `slots`)
  - `snapshot_slot(slot) -> dict` / `load_state(dict)` / `state_dict_all() -> dict` (for prototype + checkpoint)

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/rssm/test_actor_critic.py
from __future__ import annotations

import dataclasses

import torch

from artificial_society.agents.rssm.actor_critic import ActorCriticSlab, lambda_returns
from artificial_society.agents.rssm.config import RSSMConfig, make_generator
from artificial_society.agents.rssm.world_model import RSSMWorldModel

CFG = dataclasses.replace(RSSMConfig(), max_slots=8, horizon=5)


def _slab(cfg=CFG):
    return ActorCriticSlab(cfg, make_generator(0, "slab"))


def _starts(n=4):
    g = make_generator(1, "st")
    return (torch.randn(n, 256, generator=g), torch.zeros(n, 32, 32),
            torch.rand(n, 4, generator=g))


def test_lambda_returns_recursion():
    # H=3, gamma=1, lam=1 → pure discounted sum with continues folded in
    rew = torch.tensor([[1.0, 1.0, 1.0]])
    cont = torch.tensor([[1.0, 0.0, 1.0]])
    val = torch.zeros(1, 4)
    R = lambda_returns(rew, cont, val, gamma=1.0, lam=1.0)
    assert torch.allclose(R[0], torch.tensor([2.0, 1.0, 1.0]))  # death at t=1 cuts the tail


def test_act_single_deterministic_and_slot_isolated():
    slab = _slab()
    s0, s1 = slab.acquire_slot(None), slab.acquire_slot(None)
    s_g = torch.randn(1, 256 + 1024 + 4, generator=make_generator(2, "x"))
    a1, _ = slab.act_single(s0, s_g, make_generator(3, ("a", 1)))
    a2, _ = slab.act_single(s0, s_g, make_generator(3, ("a", 1)))
    assert torch.equal(a1, a2) and a1.shape == (7,)
    assert torch.all(a1 <= 1.0) and torch.all(a1 >= -1.0)


def test_imagination_update_moves_only_given_slots():
    slab = _slab()
    slots = torch.tensor([slab.acquire_slot(None), slab.acquire_slot(None)])
    frozen = slab.acquire_slot(None)
    before_frozen = slab.params["a_w1"][frozen].clone()
    before_active = slab.params["a_w1"][slots[0]].clone()
    wm = RSSMWorldModel(CFG)
    h0, z0, genes = _starts(2)
    m = slab.imagination_update(wm, h0, z0, genes, slots, make_generator(4, "img"), None)
    assert all(map(lambda v: torch.isfinite(torch.tensor(float(v))), m.values()))
    assert torch.equal(slab.params["a_w1"][frozen], before_frozen)      # masked update
    assert not torch.equal(slab.params["a_w1"][slots[0]], before_active)


def test_reinforce_gradient_flow_contract():
    """Reinforce mode: WM must receive ZERO grad from the actor loss; actor params must receive grad."""
    slab = _slab()
    slots = torch.tensor([slab.acquire_slot(None)])
    wm = RSSMWorldModel(CFG)
    h0, z0, genes = _starts(1)
    wm_before = [p.clone() for p in wm.parameters()]
    slab.imagination_update(wm, h0, z0, genes, slots, make_generator(5, "img"), None)
    assert all(torch.equal(a, b) for a, b in zip(wm_before, wm.parameters()))  # WM untouched
    assert all(p.grad is None for p in wm.parameters())                        # and no grads left


def test_dynamics_mode_end_to_end_gradient():
    cfg = dataclasses.replace(CFG, actor_grad="dynamics")
    slab = ActorCriticSlab(cfg, make_generator(0, "slab"))
    slots = torch.tensor([slab.acquire_slot(None)])
    wm = RSSMWorldModel(cfg)
    h0, z0, genes = _starts(1)
    before = slab.params["a_w1"][slots[0]].clone()
    m = slab.imagination_update(wm, h0, z0, genes, slots, make_generator(6, "img"), None)
    assert not torch.equal(slab.params["a_w1"][slots[0]], before)  # end-to-end grads arrived
    assert all(p.grad is None for p in wm.parameters())            # WM frozen during actor update


def test_slot_hygiene_on_release_and_reacquire():
    slab = _slab()
    s = slab.acquire_slot(None)
    wm = RSSMWorldModel(CFG)
    h0, z0, genes = _starts(1)
    slab.imagination_update(wm, h0, z0, genes, torch.tensor([s]), make_generator(7, "i"), None)
    assert slab.step_count[s] > 0
    slab.release_slot(s)
    assert not slab.alive[s]
    s2 = slab.acquire_slot(None)
    assert s2 == s                                      # freed slot reused
    assert slab.step_count[s2] == 0                     # step count reset
    assert torch.all(slab.adam_m["a_w1"][s2] == 0)      # both moments zeroed
    assert torch.all(slab.adam_v["a_w1"][s2] == 0)


def test_prototype_snapshot_roundtrip():
    slab = _slab()
    s = slab.acquire_slot(None)
    snap = slab.snapshot_slot(s)
    s2 = slab.acquire_slot(snap)
    assert torch.equal(slab.params["a_w1"][s], slab.params["a_w1"][s2])
```

Run: `venv/bin/python -m pytest tests/agents/rssm/test_actor_critic.py -q` → FAIL (ImportError)

- [ ] **Step 2: Implement**

```python
# artificial_society/agents/rssm/actor_critic.py
"""Per-agent actor-critics in a pre-allocated slab (spec §4.2/§8).

Stacked tensors (max_slots, out, in); batched linears via einsum; hand-rolled
elementwise Adam on the slab with per-slot step counts and alive-masked updates.
"""
from __future__ import annotations

import math

import torch.nn.functional as F

import torch

from .util import make_bins, symlog, twohot, twohot_mean

LOGSTD_MIN, LOGSTD_MAX = -5.0, 1.0


def lambda_returns(reward, cont, value, gamma, lam):
    """reward/cont: (N,H); value: (N,H+1). Returns (N,H) λ-returns with γ·cont discounts."""
    H = reward.shape[1]
    R = torch.zeros_like(reward)
    nxt = value[:, -1]
    for t in range(H - 1, -1, -1):
        disc = gamma * cont[:, t]
        blended = (1 - lam) * value[:, t + 1] + lam * nxt
        R[:, t] = reward[:, t] + disc * blended
        nxt = R[:, t]
    return R


def _blin(w, b, x):
    """Batched linear: w (N,out,in), b (N,out), x (N,in) -> (N,out)."""
    return torch.einsum("noi,ni->no", w, x) + b


class ActorCriticSlab:
    ACTOR_KEYS = ("a_w1", "a_b1", "a_w2", "a_b2", "a_mu_w", "a_mu_b", "a_std_w", "a_std_b")
    CRITIC_KEYS = ("c_w1", "c_b1", "c_w2", "c_b2", "c_out_w", "c_out_b")

    def __init__(self, cfg, gen):
        self.cfg = cfg
        self.gen = gen
        d_in = cfg.deter + cfg.stoch * cfg.classes + (cfg.gene_slice[1] - cfg.gene_slice[0])
        h, N = cfg.ac_hidden, cfg.max_slots
        shapes = {
            "a_w1": (N, h, d_in), "a_b1": (N, h), "a_w2": (N, h, h), "a_b2": (N, h),
            "a_mu_w": (N, cfg.action_dim, h), "a_mu_b": (N, cfg.action_dim),
            "a_std_w": (N, cfg.action_dim, h), "a_std_b": (N, cfg.action_dim),
            "c_w1": (N, h, d_in), "c_b1": (N, h), "c_w2": (N, h, h), "c_b2": (N, h),
            "c_out_w": (N, cfg.num_bins, h), "c_out_b": (N, cfg.num_bins),
        }
        self.params = {k: torch.zeros(*s) for k, s in shapes.items()}
        self.slow = {k: torch.zeros_like(self.params[k]) for k in self.CRITIC_KEYS}
        self.adam_m = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.adam_v = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.step_count = torch.zeros(N, dtype=torch.long)
        self.alive = torch.zeros(N, dtype=torch.bool)
        self.ret_scale = 1.0  # EMA(p95-p5) of λ-returns
        self.bins = make_bins(cfg)

    # --- slots --------------------------------------------------------------
    def _fresh_init(self, slot):
        for k, p in self.params.items():
            if p.dim() == 3:  # weights: truncated-normal-ish fan-in init
                fan_in = p.shape[2]
                p[slot] = torch.randn(p.shape[1], p.shape[2], generator=self.gen) / math.sqrt(fan_in)
            else:
                p[slot] = 0.0
        self.params["c_out_w"][slot] = 0.0        # zero-init critic head (spec §4.2)
        self.params["a_std_b"][slot] = -1.0       # start with modest stddev
        for k in self.CRITIC_KEYS:
            self.slow[k][slot] = self.params[k][slot]

    def acquire_slot(self, init_params):
        free = (~self.alive).nonzero(as_tuple=True)[0]
        if len(free) == 0:
            raise RuntimeError("actor slab full — raise RSSMConfig.max_slots")
        slot = int(free[0])
        for k in self.params:
            self.adam_m[k][slot] = 0.0
            self.adam_v[k][slot] = 0.0
        self.step_count[slot] = 0
        if init_params is None:
            self._fresh_init(slot)
        else:
            for k, v in init_params.items():
                (self.params if k in self.params else self.slow)[k][slot] = v
            for k in self.CRITIC_KEYS:
                self.slow[k][slot] = self.params[k][slot]
        self.alive[slot] = True
        return slot

    def release_slot(self, slot):
        self.alive[slot] = False

    def snapshot_slot(self, slot):
        return {k: self.params[k][slot].clone() for k in self.params}

    # --- forward ------------------------------------------------------------
    def _actor_dist(self, P, idx, s_g):
        x = F.silu(_blin(P["a_w1"][idx], P["a_b1"][idx], s_g))
        x = F.silu(_blin(P["a_w2"][idx], P["a_b2"][idx], x))
        mu = _blin(P["a_mu_w"][idx], P["a_mu_b"][idx], x)
        logstd = _blin(P["a_std_w"][idx], P["a_std_b"][idx], x).clamp(LOGSTD_MIN, LOGSTD_MAX)
        return mu, logstd

    def _critic_logits(self, P, idx, s_g):
        x = F.silu(_blin(P["c_w1"][idx], P["c_b1"][idx], s_g))
        x = F.silu(_blin(P["c_w2"][idx], P["c_b2"][idx], x))
        return _blin(P["c_out_w"][idx], P["c_out_b"][idx], x)

    @staticmethod
    def _tanh_normal_sample(mu, logstd, gen):
        eps = torch.randn(mu.shape, generator=gen, device=mu.device)
        pre = mu + eps * logstd.exp()
        return torch.tanh(pre), pre

    @staticmethod
    def _log_prob(mu, logstd, pre, act):
        base = -0.5 * (((pre - mu) / logstd.exp()) ** 2 + 2 * logstd + math.log(2 * math.pi))
        # tanh log-det-Jacobian (stable form)
        jac = 2.0 * (math.log(2.0) - pre - torch.nn.functional.softplus(-2.0 * pre))
        return (base + jac).sum(-1)

    def act_single(self, slot, s_g, gen):
        with torch.no_grad():
            idx = torch.tensor([slot])
            mu, logstd = self._actor_dist(self.params, idx, s_g)
            a, _ = self._tanh_normal_sample(mu, logstd, gen)
        return a.squeeze(0), None

    # --- training -----------------------------------------------------------
    def imagination_update(self, wm, h0, z0, genes, slots, gen, replay_batch):
        cfg = self.cfg
        N, H = h0.shape[0], cfg.horizon
        reinforce = cfg.actor_grad == "reinforce"
        for p in self.params.values():
            p.requires_grad_(True)
        wm_req = [p.requires_grad for p in wm.parameters()]
        for p in wm.parameters():
            p.requires_grad_(False)               # WM frozen during AC update (both modes)

        h, z = h0, z0
        s_list, logp_list, ent_list, rew_list, cont_list = [], [], [], [], []
        for _ in range(H):
            s_g = wm.head_input(h, z, genes)
            mu, logstd = self._actor_dist(self.params, slots, s_g)
            a, pre = self._tanh_normal_sample(mu, logstd, gen)
            logp = self._log_prob(mu, logstd, pre, a)
            s_list.append(s_g)
            logp_list.append(logp)
            ent_list.append(-logp)                # sample-based entropy estimate
            a_step = a if not reinforce else a.detach()
            h, z, _ = wm.img_step(h, z, a_step, gen)
            s_next = wm.head_input(h, z, genes)
            rew_list.append(twohot_mean(wm.reward_logits(s_next), wm.bins))
            cont_list.append(wm.cont_prob(s_next))
        S = torch.stack(s_list, 1)                                    # (N,H,1284)
        logp = torch.stack(logp_list, 1)
        entropy = torch.stack(ent_list, 1)
        reward = torch.stack(rew_list, 1)
        cont = torch.stack(cont_list, 1)

        flat = S.reshape(N * H, -1)
        idx_rep = slots.repeat_interleave(H)
        v_logits = self._critic_logits(self.params, idx_rep, flat).reshape(N, H, -1)
        values = twohot_mean(v_logits, self.bins)                     # (N,H)
        with torch.no_grad():
            s_gH = wm.head_input(h, z, genes)
            v_last = twohot_mean(self._critic_logits(self.params, slots, s_gH), self.bins)
        val_ext = torch.cat([values.detach() if reinforce else values, v_last.unsqueeze(1)], 1)

        R = lambda_returns(reward, cont, val_ext, cfg.gamma, cfg.lam)  # (N,H)
        w = torch.cumprod(torch.cat([torch.ones(N, 1), (cfg.gamma * cont)[:, :-1]], 1), 1).detach()

        # return normalization S = EMA(p95-p5), divide by max(1,S)
        with torch.no_grad():
            spread = (torch.quantile(R, 0.95) - torch.quantile(R, 0.05)).item()
            self.ret_scale = cfg.retnorm_decay * self.ret_scale + (1 - cfg.retnorm_decay) * spread
        scale = max(1.0, self.ret_scale)

        if reinforce:
            adv = ((R - values.detach()) / scale).detach()
            actor_loss = -(w * adv * logp).mean() - cfg.entropy_eta * (w * entropy).mean()
        else:  # dynamics backprop: stop-grad baseline, differentiable R
            adv = (R - values.detach()) / scale
            actor_loss = -(w * adv).mean() - cfg.entropy_eta * (w * entropy).mean()

        # critic loss: twohot CE to λ-returns + slow-critic regularizer (+ replay term)
        tgt = twohot(symlog(R.detach()), self.bins)
        logp_v = torch.log_softmax(v_logits, -1)
        critic_loss = -(w * (tgt * logp_v).sum(-1)).mean()
        with torch.no_grad():
            slow_logits = self._critic_logits(self.slow, idx_rep, flat.detach()).reshape(N, H, -1)
        critic_loss = critic_loss + cfg.slow_critic_scale * (
            -(w * (torch.softmax(slow_logits, -1) * logp_v).sum(-1)).mean())
        if replay_batch is not None:
            ro, ra = replay_batch["s_g"], replay_batch["returns"]     # built by learner
            rl = self._critic_logits(self.params, replay_batch["slots"], ro)
            critic_loss = critic_loss + cfg.replay_critic_scale * (
                -(twohot(symlog(ra), self.bins) * torch.log_softmax(rl, -1)).sum(-1).mean())

        loss = actor_loss + critic_loss
        grads = torch.autograd.grad(loss, list(self.params.values()), allow_unused=True)
        self._adam_step(dict(zip(self.params.keys(), grads)), slots)
        for p, r in zip(wm.parameters(), wm_req):
            p.requires_grad_(r)
        for p in self.params.values():
            p.requires_grad_(False)
        self._slow_critic_ema(slots)
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "ret_scale": scale, "entropy": entropy.mean().item()}

    def _adam_step(self, grads, slots):
        cfg = self.cfg
        mask = torch.zeros(cfg.max_slots)
        mask[slots] = 1.0
        self.step_count[slots] += 1
        with torch.no_grad():
            # global-norm clip over the touched slots' grads
            total = torch.sqrt(sum((g[slots] ** 2).sum() for g in grads.values() if g is not None))
            clip = min(1.0, cfg.ac_clip / (float(total) + 1e-8))
            for k, g in grads.items():
                if g is None:
                    continue
                g = g * clip
                m, v = self.adam_m[k], self.adam_v[k]
                m[slots] = cfg.adam_beta1 * m[slots] + (1 - cfg.adam_beta1) * g[slots]
                v[slots] = cfg.adam_beta2 * v[slots] + (1 - cfg.adam_beta2) * g[slots] ** 2
                t = self.step_count[slots].float().clamp(min=1)
                shape = [-1] + [1] * (m.dim() - 1)
                mhat = m[slots] / (1 - cfg.adam_beta1 ** t).reshape(*shape)
                vhat = v[slots] / (1 - cfg.adam_beta2 ** t).reshape(*shape)
                self.params[k].data[slots] -= cfg.ac_lr * mhat / (vhat.sqrt() + cfg.adam_eps)

    def _slow_critic_ema(self, slots):
        d = self.cfg.slow_critic_decay
        with torch.no_grad():
            for k in self.CRITIC_KEYS:
                self.slow[k][slots] = d * self.slow[k][slots] + (1 - d) * self.params[k][slots]

    # --- persistence ----------------------------------------------------------
    def state_dict_all(self):
        return {"params": self.params, "slow": self.slow, "adam_m": self.adam_m,
                "adam_v": self.adam_v, "step_count": self.step_count, "alive": self.alive,
                "ret_scale": self.ret_scale}

    def load_state(self, sd):
        for name in ("params", "slow", "adam_m", "adam_v"):
            for k, v in sd[name].items():
                getattr(self, name)[k].copy_(v)
        self.step_count.copy_(sd["step_count"])
        self.alive.copy_(sd["alive"])
        self.ret_scale = sd["ret_scale"]
```

Implementation notes (read before coding): (1) delete the stray `m.mul_(0).add_(m * 0)` guard line — it is a plan artifact, write the masked Adam directly; (2) `torch.silu` is `torch.nn.functional.silu` on torch 2.8 — import accordingly; (3) in reinforce mode `val_ext` uses detached values — λ-returns must not backprop into the critic from the actor loss; the critic learns only from its own CE term.

- [ ] **Step 3: Run tests**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_actor_critic.py -q`
Expected: 8 passed

- [ ] **Step 4: Full suite + commit**

Run: `venv/bin/python -m pytest -q` → all green.

```bash
git add artificial_society/agents/rssm/actor_critic.py tests/agents/rssm/test_actor_critic.py
git commit -m "feat(rssm): actor-critic slab — reinforce-in-imagination, slab-Adam, slot hygiene"
```

---

### Task 6: PrototypeActor (cultural warm-start)

**Files:**
- Create: `artificial_society/agents/rssm/prototype.py`
- Test: `tests/agents/rssm/test_prototype.py`

**Interfaces:**
- Consumes: `ActorCriticSlab.snapshot_slot/params/alive`.
- Produces: `PrototypeActor(cfg)` with `template() -> dict | None` (None until first update → slab does fresh init), `update(slab)` (EMA over **alive** slots' params, decay `cfg.prototype_decay`; first call = plain mean), `state_dict()/load_state_dict(sd)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/rssm/test_prototype.py
from __future__ import annotations

import dataclasses

import torch

from artificial_society.agents.rssm.actor_critic import ActorCriticSlab
from artificial_society.agents.rssm.config import RSSMConfig, make_generator
from artificial_society.agents.rssm.prototype import PrototypeActor

CFG = dataclasses.replace(RSSMConfig(), max_slots=4)


def test_newborn_equals_prototype_then_diverges():
    slab = ActorCriticSlab(CFG, make_generator(0, "s"))
    proto = PrototypeActor(CFG)
    assert proto.template() is None
    s0 = slab.acquire_slot(proto.template())          # fresh init (no template yet)
    proto.update(slab)
    s1 = slab.acquire_slot(proto.template())          # warm-start from prototype
    assert torch.equal(slab.params["a_w1"][s1], proto.state_dict()["a_w1"])
    slab.params["a_w1"][s1] += 1.0                    # fine-tuning diverges
    assert not torch.equal(slab.params["a_w1"][s1], proto.state_dict()["a_w1"])


def test_ema_moves_toward_population():
    slab = ActorCriticSlab(CFG, make_generator(0, "s"))
    proto = PrototypeActor(CFG)
    a = slab.acquire_slot(None)
    proto.update(slab)
    before = proto.state_dict()["a_w1"].clone()
    slab.params["a_w1"][a] += 10.0
    proto.update(slab)
    after = proto.state_dict()["a_w1"]
    assert not torch.equal(before, after)
    assert torch.all((after - before).abs() <= 10.0 * (1 - CFG.prototype_decay) + 1e-6)
```

Run: `venv/bin/python -m pytest tests/agents/rssm/test_prototype.py -q` → FAIL

- [ ] **Step 2: Implement**

```python
# artificial_society/agents/rssm/prototype.py
"""Slowly-updated population template actor for cultural warm-start (spec §4.3)."""
from __future__ import annotations

import torch


class PrototypeActor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._ema: dict | None = None

    def template(self):
        return None if self._ema is None else {k: v.clone() for k, v in self._ema.items()}

    def update(self, slab) -> None:
        alive = slab.alive.nonzero(as_tuple=True)[0]
        if len(alive) == 0:
            return
        mean = {k: slab.params[k][alive].mean(0) for k in slab.params}
        if self._ema is None:
            self._ema = mean
            return
        d = self.cfg.prototype_decay
        with torch.no_grad():
            for k in self._ema:
                self._ema[k] = d * self._ema[k] + (1 - d) * mean[k]

    def state_dict(self):
        return {} if self._ema is None else self._ema

    def load_state_dict(self, sd):
        self._ema = sd or None
```

- [ ] **Step 3: Run + commit**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_prototype.py -q` → 2 passed

```bash
git add artificial_society/agents/rssm/prototype.py tests/agents/rssm/test_prototype.py
git commit -m "feat(rssm): prototype actor — EMA cultural warm-start"
```

---

### Task 7: SharedLearner — act path, cadence, spawn/death hooks, checkpoint payload

**Files:**
- Create: `artificial_society/agents/rssm/learner.py`
- Test: `tests/agents/rssm/test_learner.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6; `Agent` attributes `id`, `alive`, `hidden_state`, `birth_tick`.
- Produces: `SharedLearner(cfg, global_seed: int)` with:
  - `act(agent, features: list[float]) -> dict` — the v1-compatible contract: `{"action_list": [7 floats], "next_hidden": (h(1,256), z(1,32,32)), "obs_tensor": Tensor(57,), "action_tensor": Tensor(7,)}`. Runs posterior inference on **CPU** under `no_grad`, per-agent generator. `agent.hidden_state` may be `None` (fresh) — then `initial_state(1)`.
  - `store_transition(agent, brain_step, reward: float, done: bool)` → replay `add`.
  - `on_spawn(agent, origin: str, tick: int)` → `agent.rssm_slot = slab.acquire_slot(prototype.template())`, `agent.brain_arch = "rssm"`, `agent.spawn_origin = origin`, `replay.start_episode`.
  - `on_death(agent)` → `slab.release_slot`, `replay.end_episode` (the terminal transition itself arrives via `store_transition(..., done=True)`).
  - `maybe_train(tick: int, agents: list) -> dict | None` — no-op unless `tick % cfg.train_every == 0` and replay ≥ `cfg.prefill_transitions`; then: 1 WM `train_batch`; if `wm.wm_updates > cfg.wm_warmup_updates`: imagination update for all alive slots (start-states = burn-in **re-encode with current WM** from `sample_starts`, own-frac blended by age; young agents get up to `young_ratio_cap` extra passes); prototype EMA update. Returns metrics.
  - `checkpoint_payload() -> dict` / `load_checkpoint_payload(sd, agents)` — WM (`.cpu()` state_dict + optimizer state), prototype, slab `state_dict_all()`, slot↔agent_id map rebuilt from `agent.rssm_slot`.
  - Device: WM lives on `cfg.train_device` if available **else CPU with a one-line warning** (unit tests run CPU); a **separate CPU copy is not kept** — act path calls the WM on CPU tensors only when the WM itself is on CPU, otherwise it uses a cached CPU **clone synced after each train hook** (`self._act_wm`). Keep it simple: `_act_wm = copy.deepcopy(wm).cpu().eval()` refreshed at the end of every `maybe_train` that ran.

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/rssm/test_learner.py
from __future__ import annotations

import dataclasses
import types

import torch

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.agents.rssm.learner import SharedLearner

CFG = dataclasses.replace(RSSMConfig(), max_slots=8, horizon=4, batch_size=4, seq_len=16,
                          prefill_transitions=64, wm_warmup_updates=1, train_device="cpu")


def _agent(aid):
    return types.SimpleNamespace(id=aid, alive=True, hidden_state=None, birth_tick=0)


def _feats(v=0.5):
    return [v] * 57


def test_act_contract_and_determinism():
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    step = ln.act(ag, _feats())
    assert set(step) >= {"action_list", "next_hidden", "obs_tensor", "action_tensor"}
    assert len(step["action_list"]) == 7
    assert all(-1.0 <= a <= 1.0 for a in step["action_list"])
    h, z = step["next_hidden"]
    assert h.shape == (1, 256) and z.shape == (1, 32, 32)
    # same seed twice → identical trajectory of draws
    ln2 = SharedLearner(CFG, 42)
    ag2 = _agent(1)
    ln2.on_spawn(ag2, "initial", 0)
    step2 = ln2.act(ag2, _feats())
    assert step["action_list"] == step2["action_list"]


def test_global_rng_untouched_by_full_cycle():
    torch.manual_seed(123)
    before = torch.get_rng_state()
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    for t in range(80):
        step = ln.act(ag, _feats())
        ag.hidden_state = step["next_hidden"]
        ln.store_transition(ag, step, 0.1, done=(t == 79))
    ln.on_death(ag)
    assert torch.equal(before, torch.get_rng_state())


def test_train_cadence_and_warmup_gate():
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    for t in range(70):
        step = ln.act(ag, _feats())
        ag.hidden_state = step["next_hidden"]
        ln.store_transition(ag, step, 0.0, done=False)
    assert ln.maybe_train(3, [ag]) is None          # 3 % train_every != 0
    m1 = ln.maybe_train(8, [ag])                    # replay ≥ prefill → WM trains
    assert m1 is not None and ln.wm.wm_updates == 1
    assert "actor_loss" not in m1                   # warm-up gate: no imagination yet
    m2 = ln.maybe_train(16, [ag])
    assert "actor_loss" in m2                       # gate open after warmup updates


def test_spawn_death_slot_lifecycle():
    ln = SharedLearner(CFG, 42)
    a, b = _agent(1), _agent(2)
    ln.on_spawn(a, "initial", 0)
    ln.on_spawn(b, "birth", 5)
    assert a.rssm_slot != b.rssm_slot and a.spawn_origin == "initial"
    ln.on_death(a)
    c = _agent(3)
    ln.on_spawn(c, "respawn", 9)
    assert c.rssm_slot == a.rssm_slot               # freed slot reused


def test_checkpoint_roundtrip():
    ln = SharedLearner(CFG, 42)
    ag = _agent(1)
    ln.on_spawn(ag, "initial", 0)
    step = ln.act(ag, _feats())
    payload = ln.checkpoint_payload()
    ln2 = SharedLearner(CFG, 42)
    ag2 = _agent(1)
    ag2.rssm_slot = ag.rssm_slot
    ln2.load_checkpoint_payload(payload, [ag2])
    step2 = ln2.act(ag2, _feats())
    assert step["action_list"] == step2["action_list"]
```

Run: `venv/bin/python -m pytest tests/agents/rssm/test_learner.py -q` → FAIL

- [ ] **Step 2: Implement**

```python
# artificial_society/agents/rssm/learner.py
"""SharedLearner: owns WM + replay + slab + prototype; drives acting and training (spec §4.5)."""
from __future__ import annotations

import copy

import torch

from .actor_critic import ActorCriticSlab, lambda_returns  # noqa: F401 (returns used in replay-critic)
from .config import make_generator
from .prototype import PrototypeActor
from .replay import SharedReplay
from .world_model import RSSMWorldModel


class SharedLearner:
    def __init__(self, cfg, global_seed: int):
        self.cfg = cfg
        self.global_seed = int(global_seed if global_seed is not None else 0)
        want = cfg.train_device
        if want.startswith("cuda") and not torch.cuda.is_available():
            print("[rssm] train_device=cuda unavailable — falling back to CPU")
            want = "cpu"
        self.train_device = want
        self.wm = RSSMWorldModel(cfg).to(self.train_device)
        self._act_wm = self.wm if self.train_device == "cpu" else copy.deepcopy(self.wm).cpu().eval()
        self.slab = ActorCriticSlab(cfg, make_generator(self.global_seed, "slab-init"))
        self.prototype = PrototypeActor(cfg)
        self.replay = SharedReplay(cfg)
        self._gen_learner = make_generator(self.global_seed, "learner")
        self._gen_train = make_generator(self.global_seed, "train", device=self.train_device)
        self._agent_gens: dict[int, torch.Generator] = {}

    # --- per-agent RNG ------------------------------------------------------
    def _gen(self, agent_id: int) -> torch.Generator:
        g = self._agent_gens.get(agent_id)
        if g is None:
            g = make_generator(self.global_seed, ("agent", agent_id))
            self._agent_gens[agent_id] = g
        return g

    # --- lifecycle ------------------------------------------------------------
    def on_spawn(self, agent, origin: str, tick: int) -> None:
        agent.rssm_slot = self.slab.acquire_slot(self.prototype.template())
        agent.brain_arch = "rssm"
        agent.spawn_origin = origin
        agent.hidden_state = None
        self.replay.start_episode(agent.id, origin)

    def on_death(self, agent) -> None:
        self.replay.end_episode(agent.id)
        self.slab.release_slot(agent.rssm_slot)
        self._agent_gens.pop(agent.id, None)

    # --- acting (batch-1, CPU, no_grad) ----------------------------------------
    def act(self, agent, features) -> dict:
        cfg = self.cfg
        gen = self._gen(agent.id)
        obs = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if agent.hidden_state is None:
                h, z = self._act_wm.initial_state(1, torch.device("cpu"))
                a_prev = torch.zeros(1, cfg.action_dim)
            else:
                h, z = agent.hidden_state
                a_prev = getattr(agent, "_rssm_last_action", torch.zeros(1, cfg.action_dim))
            h, z, _, _ = self._act_wm.obs_step(h, z, a_prev, obs, gen)
            genes = obs[:, cfg.gene_slice[0]:cfg.gene_slice[1]]
            s_g = self._act_wm.head_input(h, z, genes)
            action, _ = self.slab.act_single(agent.rssm_slot, s_g, gen)
        agent._rssm_last_action = action.unsqueeze(0)
        return {"action_list": [float(x) for x in action],
                "next_hidden": (h, z),
                "obs_tensor": obs.squeeze(0),
                "action_tensor": action}

    def store_transition(self, agent, brain_step, reward: float, done: bool) -> None:
        self.replay.add(agent.id, brain_step["obs_tensor"], brain_step["action_tensor"],
                        reward, done)

    # --- training cadence -------------------------------------------------------
    def maybe_train(self, tick: int, agents) -> dict | None:
        cfg = self.cfg
        if tick % cfg.train_every != 0 or len(self.replay) < cfg.prefill_transitions:
            return None
        batch = self.replay.sample_sequences(self._gen_learner)
        batch = {k: v.to(self.train_device) for k, v in batch.items()}
        metrics = self.wm.train_batch(batch, self._gen_train)
        if self.wm.wm_updates > cfg.wm_warmup_updates:
            img = self._imagination_pass(agents, tick)
            if img:
                metrics.update(img)
            self.prototype.update(self.slab)
        if self._act_wm is not self.wm:
            self._act_wm = copy.deepcopy(self.wm).cpu().eval()
        return metrics

    def _imagination_pass(self, agents, tick: int) -> dict | None:
        cfg = self.cfg
        rssm_agents = [a for a in agents if a.alive and getattr(a, "rssm_slot", None) is not None]
        if not rssm_agents:
            return None
        h0s, z0s, genes_l, slots = [], [], [], []
        for a in rssm_agents:
            age = tick - getattr(a, "birth_tick", 0)
            own = min(cfg.own_start_frac_max, age / max(1, cfg.young_age_ticks))
            reps = cfg.young_ratio_cap if age < cfg.young_age_ticks else 1
            starts = self.replay.sample_starts(a.id, reps, own, self._gen_learner)
            if starts is None:
                continue
            obs_w, act_w = (t.to(self.train_device) for t in starts)
            with torch.no_grad():   # burn-in RE-ENCODE with the current WM (spec §4.2/§4.5)
                h, z, _, _ = self.wm.observe(obs_w, act_w, self._gen_train)
            h0s.append(h[:, -1]); z0s.append(z[:, -1])
            genes_l.append(obs_w[:, -1, cfg.gene_slice[0]:cfg.gene_slice[1]])
            slots.append(torch.full((obs_w.shape[0],), a.rssm_slot, dtype=torch.long))
        if not h0s:
            return None
        return self.slab.imagination_update(
            self.wm, torch.cat(h0s), torch.cat(z0s), torch.cat(genes_l),
            torch.cat(slots), self._gen_train, None)

    # --- checkpoint ---------------------------------------------------------------
    def checkpoint_payload(self) -> dict:
        return {"wm": {k: v.cpu() for k, v in self.wm.state_dict().items()},
                "wm_opt": self.wm.opt.state_dict(),
                "wm_updates": self.wm.wm_updates,
                "prototype": self.prototype.state_dict(),
                "slab": self.slab.state_dict_all()}

    def load_checkpoint_payload(self, sd, agents) -> None:
        self.wm.load_state_dict({k: v.to(self.train_device) for k, v in sd["wm"].items()})
        self.wm.opt.load_state_dict(sd["wm_opt"])
        self.wm.wm_updates = sd["wm_updates"]
        self.prototype.load_state_dict(sd["prototype"])
        self.slab.load_state(sd["slab"])
        if self._act_wm is not self.wm:
            self._act_wm = copy.deepcopy(self.wm).cpu().eval()
        for a in agents:   # re-open episodes for living agents
            if getattr(a, "rssm_slot", None) is not None:
                self.replay.start_episode(a.id, getattr(a, "spawn_origin", "unknown"))
```

Notes: (1) the replay-critic term is wired in Task 9's tuning pass only if pilot WM metrics look unstable — `imagination_update(..., replay_batch=None)` until then (the slab API already accepts it); (2) `_rssm_last_action` intentionally lives on the agent (pickles as a small tensor; harmless).

- [ ] **Step 3: Run tests**

Run: `venv/bin/python -m pytest tests/agents/rssm/test_learner.py -q`
Expected: 5 passed

- [ ] **Step 4: Full suite + commit**

Run: `venv/bin/python -m pytest -q` → all green.

```bash
git add artificial_society/agents/rssm/learner.py tests/agents/rssm/test_learner.py
git commit -m "feat(rssm): SharedLearner — CPU act path, K-cadence training, warm-up gate, checkpoint payload"
```

---

### Task 8: Sim integration — flag, dispatch, spawn wiring, registered training system, checkpoint, serve pin

⚠️ Hot files: `simulation.py`, `agents/agent.py`. Minimal diffs only; every edit below is flag-gated so v1/v2 behavior is byte-identical. `agents/brain.py` is NOT touched.

**Files:**
- Modify: `artificial_society/simulation.py` (init flag+construction; spawn wiring ×3; remove_dead hook; checkpoint; update-call kwarg)
- Modify: `artificial_society/agents/agent.py` (`ensure_fields` branch; `update()` kwarg + dispatch + store)
- Create: `artificial_society/systems/rssm_learning.py`
- Modify: `artificial_society/serve/runner.py` (pin `brain_arch`)
- Test: `tests/test_rssm_integration.py`

**Interfaces:**
- Consumes: `SharedLearner` (Task 7).
- Produces: `Simulation(..., brain_arch: str | None = None, rssm_config=None)`; `sim.brain_arch: str` (resolution order: explicit arg → `AS_BRAIN_ARCH` env → `"v1"`); `sim.rssm_learner: SharedLearner | None`; `Agent.update(..., learner=None)`.

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/test_rssm_integration.py
from __future__ import annotations

import dataclasses

import pytest
import torch

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.simulation import CheckpointIncompatibleError, Simulation

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)
_FAST = dataclasses.replace(RSSMConfig(), prefill_transitions=64, wm_warmup_updates=1,
                            batch_size=4, seq_len=16, horizon=4, max_slots=32,
                            train_device="cpu")


def _digest(sim):
    return [(a.id, round(a.energy, 4)) for a in sim.agents]


def test_off_arm_leaves_torch_rng_untouched():
    sim = Simulation(seed=42, **_PARAMS)
    state_after_init = torch.get_rng_state()
    sim2 = Simulation(seed=42, **_PARAMS)
    assert torch.equal(state_after_init, torch.get_rng_state())  # identical construction draws
    assert sim.rssm_learner is None and sim.brain_arch == "v1"
    assert _digest(sim) == _digest(sim2)


def test_rssm_run_deterministic_across_births_and_deaths():
    def run():
        sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
        for _ in range(40):
            sim.step()
        return _digest(sim)
    assert run() == run()


def test_rssm_agents_skip_v1_brain():
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    sim.step()
    for a in sim.agents:
        assert getattr(a, "brain_arch", None) == "rssm"
        assert getattr(a, "rssm_slot", None) is not None
        assert getattr(a, "brain", None) is None      # no v1 Brain built


def test_training_hook_fires_via_registry():
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    for _ in range(24):                                # 8 agents × 24 ticks > 64 prefill
        sim.step()
    assert sim.rssm_learner.wm.wm_updates >= 1


def test_checkpoint_roundtrip_and_mismatch_guard(tmp_path, monkeypatch):
    import artificial_society.simulation as sim_mod
    monkeypatch.setattr(sim_mod, "CHECKPOINT_PATH", str(tmp_path / "cp.pkl"))
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    for _ in range(10):
        sim.step()
    sim._save_checkpoint()
    with pytest.raises(CheckpointIncompatibleError):
        Simulation(seed=42, brain_arch="v1",
                   **{**_PARAMS, "load_checkpoint": True})
    sim2 = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST,
                      **{**_PARAMS, "load_checkpoint": True})
    assert sim2.tick == sim.tick
    assert sim2.rssm_learner.wm.wm_updates == sim.rssm_learner.wm.wm_updates
```

Run: `venv/bin/python -m pytest tests/test_rssm_integration.py -q` → FAIL (unexpected kwarg `brain_arch`)

- [ ] **Step 2: `simulation.py` — flag + gated construction**

In `__init__` signature (`simulation.py:127`) append parameters `brain_arch=None, rssm_config=None`. Next to `self.physics_v2 = bool(physics_v2)` (`:143`) add:

```python
        import os as _os
        self.brain_arch = brain_arch or _os.environ.get("AS_BRAIN_ARCH", "v1")
        self.rssm_learner = None
        if self.brain_arch == "rssm":
            from artificial_society.agents.rssm.config import RSSMConfig
            from artificial_society.agents.rssm.learner import SharedLearner
            self.rssm_learner = SharedLearner(rssm_config or RSSMConfig(), seed if seed is not None else 0)
```

The lazy import + `is None` default guarantees an off-arm run performs zero rssm work and zero extra torch draws (module import happens only on the rssm arm). Placement: BEFORE the initial-population spawn so slots exist for founders.

- [ ] **Step 3: `simulation.py` — spawn wiring (3 sites) + remove_dead + update-call kwarg**

1. `spawn_initial_population` (`:190-196`): after each agent is created/appended:
```python
            if self.rssm_learner is not None:
                self.rssm_learner.on_spawn(agent, "initial", self.tick)
```
2. `spawn_child_from_parent` (`:198+`): the existing body branches v1 (`child.brain.inherit_weights_from(...)`) vs v2. Add an rssm branch FIRST that skips both inheritance paths (no v1 weight copy — cultural warm-start instead, spec §4.3):
```python
        if self.rssm_learner is not None:
            self.rssm_learner.on_spawn(child, "birth", self.tick)
        elif self.physics_v2:
            ...existing v2 branch unchanged...
        else:
            ...existing v1 inherit_weights_from branch unchanged...
```
   (`child.hidden_state = child.brain.initial_hidden()` in the existing code must also be inside the non-rssm branches — under rssm, `on_spawn` sets `hidden_state = None` and there is no `child.brain`.)
3. `emergency_respawn` (`:305-312`): same pattern, origin `"respawn"`.
4. `remove_dead` (`:314-334`): where the v2 branch calls `brain.finalize_terminal()`, add:
```python
            if self.rssm_learner is not None and getattr(agent, "rssm_slot", None) is not None:
                self.rssm_learner.on_death(agent)
```
   Known approximation (documented, accepted): if an agent dies outside its own `update()` (e.g. attacked later in the tick), its last stored transition has `done=False` and the episode is closed terminal-less; `cont=0` fires only for deaths visible at store time. The continue head still learns deaths from the (majority) in-update deaths + stratified sampling.
5. The `agent.update(...)` call site inside `step()` (find it: `grep -n "\.update(self.world" artificial_society/simulation.py` — the loop over `self.agents`): append `learner=self.rssm_learner` to the call's kwargs.

- [ ] **Step 4: `agents/agent.py` — `ensure_fields` branch + `update()` seam**

1. `ensure_fields` (`agent.py:142-159`): wrap the ENTIRE brain block (both the `not hasattr` construction AND the `input_size` rebuild AND `_brain_device`) in:
```python
    if getattr(agent, "brain_arch", "v1") != "rssm":
        ...existing brain block verbatim...
    else:
        if not hasattr(agent, "brain"):
            agent.brain = None
        if not hasattr(agent, "hidden_state"):
            agent.hidden_state = None
```
2. `Agent.update` signature (`:1272`): append `learner=None`.
3. Dispatch (`:1346-1385`): put the rssm branch before the existing two:
```python
        if learner is not None:
            brain_step = learner.act(self, features)
        elif self.physics_v2:
            ...unchanged...
        else:
            ...unchanged...
```
4. Store seam (`:1543-1564`): rssm branch first, reusing the v1 scalar reward (spec §2.6 — "existing scalar reward is kept as the RSSM's reward target"):
```python
        if learner is not None:
            learner.store_transition(self, brain_step, effective_reward, not self.alive)
        elif self.physics_v2:
            ...unchanged...
        else:
            ...unchanged (v1 store_transition)...
```
5. `maybe_train` call (`:1573`): `if learner is None: self.brain.maybe_train()` (centralized training under rssm, spec §4.5).
6. Guard sweep: run `grep -n "self\.brain\." artificial_society/agents/agent.py` and `grep -rn "\.brain\." artificial_society/agents/endocrine.py artificial_society/systems/social_learning.py` — any line reachable in an rssm run (smoke test in Step 7 will hit them) gets a `if self.brain is not None` guard or an early-return for rssm agents. Do NOT touch lines only reachable under v1/v2. (`imitate_from` weight-copying and dopamine/episodic couplings are v1-brain features — under rssm they no-op via the guard; noted as scoped-out, spec §3.)

- [ ] **Step 5: Registered training system (no `step()` edit)**

```python
# artificial_society/systems/rssm_learning.py
"""Every-K-ticks centralized RSSM training hook (spec §4.5), via the system registry."""
from __future__ import annotations

from .registry import register


@register("rssm_learning", order=950)   # after all builtins; adapt decorator args to registry.py
class RssmLearningSystem:
    def __init__(self, sim):
        pass

    def tick(self, sim, tick):
        learner = getattr(sim, "rssm_learner", None)
        if learner is not None:
            learner.maybe_train(tick, sim.agents)
```

Match the exact decorator/`__init__`/`tick` signatures to `systems/registry.py` conventions (mirror the newest system module under `systems/`; the `/new-system` scaffold shows the canonical shape). The `getattr` guard makes the system a no-op on v1/v2 arms — zero draws, golden safe.

- [ ] **Step 6: Checkpoint + serve pin**

1. `simulation.py:47`: `CHECKPOINT_FORMAT_VERSION = 3`.
2. `_save_checkpoint` payload (`:442-451`): add `"brain_arch": self.brain_arch` and `"rssm": (self.rssm_learner.checkpoint_payload() if self.rssm_learner else None)`.
3. `_load_checkpoint`: next to the physics_v2 guard (`:470-474`) add the mirror guard:
```python
            saved_arch = payload.get("brain_arch", "v1")
            if saved_arch != self.brain_arch:
                raise CheckpointIncompatibleError(
                    f"checkpoint brain_arch={saved_arch} != Simulation brain_arch={self.brain_arch}")
            if self.rssm_learner is not None and payload.get("rssm") is not None:
                self.rssm_learner.load_checkpoint_payload(payload["rssm"], self.agents)
```
   (Place the rssm load AFTER agents are restored — slot map validity comes from `agent.rssm_slot` pickled with each agent.)
4. `serve/runner.py` (`:131-140`): add `brain_arch=params.get("brain_arch", "v1")` to the `Simulation(...)` construction — dashboard sims can never be flipped silently by the `AS_BRAIN_ARCH` env var.

- [ ] **Step 7: Run all gates**

```
venv/bin/python -m pytest tests/test_rssm_integration.py -q          # new tests green
venv/bin/python -m pytest -q                                          # FULL suite incl. golden — must be green untouched
venv/bin/python -c "
from artificial_society.simulation import Simulation
import dataclasses
from artificial_society.agents.rssm.config import RSSMConfig
cfg = dataclasses.replace(RSSMConfig(), prefill_transitions=512, wm_warmup_updates=5, train_device='cpu')
sim = Simulation(headless=True, load_checkpoint=False, seed=42, grid_w=30, grid_h=20,
                 initial_population=16, brain_arch='rssm', rssm_config=cfg)
[sim.step() for _ in range(500)]
print('alive:', len(sim.agents), 'wm_updates:', sim.rssm_learner.wm.wm_updates)
"
```
Expected: full suite green; the 500-tick rssm smoke does not collapse (`alive > 0`) and `wm_updates > 0`. Delete any root `checkpoint.pkl` afterwards (known gotcha).

If the golden moved: STOP — an off-arm draw leaked. Find it (`git stash` halves of the diff), never regenerate the golden.

- [ ] **Step 8: Commit**

```bash
git add artificial_society/simulation.py artificial_society/agents/agent.py \
        artificial_society/systems/rssm_learning.py artificial_society/serve/runner.py \
        tests/test_rssm_integration.py
git commit -m "feat(rssm): sim integration — brain_arch flag, registry training hook, checkpoint v3, serve pin (hot-files: core-lead at merge)"
```

---

### Task 9: Arm C (MPC mode) + A/B harness with RMST analysis

**Files:**
- Modify: `artificial_society/agents/rssm/config.py` (add `policy_mode: str = "actor"`, `mpc_candidates: int = 12`, `mpc_horizon: int = 2`)
- Modify: `artificial_society/agents/rssm/learner.py` (MPC act path + mode guards)
- Create: `scripts/rssm_ab.py`
- Test: `tests/agents/rssm/test_mpc_and_rmst.py`

**Interfaces:**
- Produces: arm mapping — **A** = `brain_arch="v1"`; **B** = `brain_arch="rssm"`, `policy_mode="actor"`; **C** = `brain_arch="rssm"`, `policy_mode="mpc"` (WM trains, NO slab/imagination/prototype). `scripts/rssm_ab.py run --arm {A,B,C} --seed S --ticks T --out DIR [--cpu]` writes `DIR/<arm>_s<S>.events.jsonl` + `.summary.json`; `scripts/rssm_ab.py analyze DIR --tau 2000 --transient 1500` prints per-arm RMST + paired deltas + exact Wilcoxon signed-rank p (enumeration, N≤12). Exposes `rmst(lifespans, censored, tau)` for unit tests.

- [ ] **Step 1: Failing tests**

```python
# tests/agents/rssm/test_mpc_and_rmst.py
from __future__ import annotations

import dataclasses

import torch

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.simulation import Simulation

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("rssm_ab", "scripts/rssm_ab.py")
rssm_ab = _ilu.module_from_spec(_spec); _spec.loader.exec_module(rssm_ab)

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)
_FAST = dataclasses.replace(RSSMConfig(), prefill_transitions=64, wm_warmup_updates=1,
                            batch_size=4, seq_len=16, horizon=4, max_slots=32,
                            train_device="cpu", policy_mode="mpc")


def test_mpc_mode_runs_without_slab():
    sim = Simulation(seed=42, brain_arch="rssm", rssm_config=_FAST, **_PARAMS)
    for _ in range(20):
        sim.step()
    assert all(getattr(a, "rssm_slot", None) is None for a in sim.agents)
    assert len(sim.agents) > 0


def test_rmst_uncensored_and_censored():
    # all die before tau → plain mean of min(life, tau)
    assert abs(rssm_ab.rmst([10, 20, 30], [False] * 3, tau=25) - (10 + 20 + 25) / 3) < 1e-9
    # censoring: KM keeps survival at 1.0 past a censored obs with no later deaths
    v = rssm_ab.rmst([10, 15], [False, True], tau=20)
    assert v > rssm_ab.rmst([10, 15], [False, False], tau=20)  # censored ≥ death-observed


def test_wilcoxon_exact_signs():
    p = rssm_ab.wilcoxon_exact([1.0, 2.0, 3.0, 4.0, 5.0])       # all positive deltas
    assert p == min(1.0, 2 * (1 / 2 ** 5) * 1)                   # most extreme rank sum
```

Run: `venv/bin/python -m pytest tests/agents/rssm/test_mpc_and_rmst.py -q` → FAIL

- [ ] **Step 2: MPC mode in the learner**

Add config fields (+ `assert cfg.policy_mode == "actor"` etc. to the config test). In `SharedLearner`:
- `on_spawn`: acquire a slab slot ONLY when `cfg.policy_mode == "actor"`; in mpc mode set `agent.rssm_slot = None` (still set `brain_arch`/`spawn_origin`, still `start_episode`).
- `on_death`: release only if `agent.rssm_slot is not None`.
- `maybe_train`: skip `_imagination_pass` + `prototype.update` when `policy_mode == "mpc"` (WM training + `_act_wm` refresh stay).
- `act`: branch to `_act_mpc` when mpc:

```python
    def _act_mpc(self, agent, obs, h, z, a_prev, gen):
        cfg = self.cfg
        with torch.no_grad():
            h, z, _, _ = self._act_wm.obs_step(h, z, a_prev, obs, gen)
            K = cfg.mpc_candidates
            cand = torch.rand(K, cfg.action_dim, generator=gen) * 2 - 1
            hh = h.expand(K, -1).contiguous()
            zz = z.expand(K, -1, -1).contiguous()
            genes = obs[:, cfg.gene_slice[0]:cfg.gene_slice[1]].expand(K, -1)
            score = torch.zeros(K)
            discount = torch.ones(K)
            a = cand
            for _ in range(cfg.mpc_horizon):
                hh, zz, _ = self._act_wm.img_step(hh, zz, a, gen)
                s_g = self._act_wm.head_input(hh, zz, genes)
                from .util import twohot_mean
                score += discount * twohot_mean(self._act_wm.reward_logits(s_g), self._act_wm.bins)
                discount *= cfg.gamma * self._act_wm.cont_prob(s_g)
                a = torch.rand(K, cfg.action_dim, generator=gen) * 2 - 1
            best = cand[int(score.argmax())]
        return best, h, z
```

`act()` then packages `best` exactly like the actor path (same contract dict). Re-run Task 7/8 test files → still green.

- [ ] **Step 3: Implement `scripts/rssm_ab.py`**

```python
#!/usr/bin/env python
"""3-arm RSSM A/B runner + RMST analysis (spec §10.2). Run on the GPU-PC.

  run:     venv/bin/python scripts/rssm_ab.py run --arm B --seed 1 --ticks 20000 --out runs/ab1
  analyze: venv/bin/python scripts/rssm_ab.py analyze runs/ab1 --tau 2000 --transient 1500
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def rmst(lifespans, censored, tau):
    """Kaplan-Meier restricted mean survival time to horizon tau."""
    events = sorted(zip(lifespans, censored))
    n, surv, prev_t, area = len(events), 1.0, 0.0, 0.0
    at_risk = n
    for t, cens in events:
        t_c = min(t, tau)
        area += surv * (t_c - prev_t)
        prev_t = t_c
        if t >= tau:
            break
        if not cens:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
    area += surv * max(0.0, tau - prev_t)
    return area


def wilcoxon_exact(deltas):
    """Exact two-sided signed-rank p for N<=12 (enumeration). Zero deltas dropped."""
    import itertools
    d = [x for x in deltas if x != 0]
    n = len(d)
    ranks = {i: r + 1 for r, (i, _) in enumerate(sorted(enumerate(map(abs, d)), key=lambda kv: kv[1]))}
    w_obs = sum(ranks[i] for i, x in enumerate(d) if x > 0)
    total = list(ranks.values())
    ws = [sum(c) for k in range(n + 1) for c in itertools.combinations(total, k)]
    mean_w = sum(total) / 2
    extreme = sum(1 for w in ws if abs(w - mean_w) >= abs(w_obs - mean_w))
    return min(1.0, extreme / len(ws))


def _arm_sim(arm, seed, cpu):
    from artificial_society.agents.rssm.config import RSSMConfig
    from artificial_society.simulation import Simulation
    kw = dict(headless=True, load_checkpoint=False, seed=seed, grid_w=60, grid_h=40,
              initial_population=36)
    if arm == "A":
        return Simulation(**kw)
    dev = "cpu" if cpu else "cuda"
    if not cpu:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") != "-1", \
            "unset CUDA_VISIBLE_DEVICES=-1 for arms B/C (spec §8)"
        import torch
        assert torch.cuda.is_available(), "arms B/C need CUDA (or pass --cpu for smoke only)"
    cfg = dataclasses.replace(RSSMConfig(), train_device=dev,
                              policy_mode=("actor" if arm == "B" else "mpc"))
    return Simulation(brain_arch="rssm", rssm_config=cfg, **kw)


def cmd_run(a):
    sim = _arm_sim(a.arm, a.seed, a.cpu)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev = open(out / f"{a.arm}_s{a.seed}.events.jsonl", "w")
    known = {}   # id -> (birth_tick, origin)
    def scan(tick):
        ids = set()
        for ag in sim.agents:
            ids.add(ag.id)
            if ag.id not in known:
                known[ag.id] = (getattr(ag, "birth_tick", tick), getattr(ag, "spawn_origin", "initial"))
                ev.write(json.dumps({"e": "birth", "id": ag.id, "t": known[ag.id][0],
                                     "origin": known[ag.id][1]}) + "\n")
        for aid in [k for k in known if k not in ids and known[k] is not None]:
            ev.write(json.dumps({"e": "death", "id": aid, "t": tick}) + "\n")
            known[aid] = None
    scan(0)
    for t in range(1, a.ticks + 1):
        sim.step()
        scan(t)
        if t % 500 == 0:
            wm_u = sim.rssm_learner.wm.wm_updates if sim.rssm_learner else 0
            ev.write(json.dumps({"e": "log", "t": t, "alive": len(sim.agents),
                                 "transitions": len(sim.rssm_learner.replay) if sim.rssm_learner else t * len(sim.agents),
                                 "wm_updates": wm_u}) + "\n")
    ev.close()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import torch
    (out / f"{a.arm}_s{a.seed}.summary.json").write_text(json.dumps({
        "arm": a.arm, "seed": a.seed, "ticks": a.ticks, "git": sha,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cfg_hash": hashlib.sha1(repr(sorted(vars(a).items())).encode()).hexdigest()[:12]}))


def cmd_analyze(a):
    runs = {}
    for f in pathlib.Path(a.dir).glob("*_s*.events.jsonl"):
        arm, seed = f.stem.split(".")[0].split("_s")
        births, deaths, last_t = {}, {}, 0
        for line in f.open():
            r = json.loads(line)
            last_t = max(last_t, r.get("t", 0))
            if r["e"] == "birth":
                births[r["id"]] = r
            elif r["e"] == "death":
                deaths[r["id"]] = r["t"]
        cohort = [(i, b) for i, b in births.items()
                  if b["origin"] == "birth" and b["t"] >= a.transient]
        lifespans = [(deaths.get(i, last_t) - b["t"]) for i, b in cohort]
        censored = [i not in deaths for i, _ in cohort]
        if lifespans:
            runs.setdefault(arm, {})[int(seed)] = rmst(lifespans, censored, a.tau)
    for arm in sorted(runs):
        vals = list(runs[arm].values())
        print(f"arm {arm}: n={len(vals)} RMST median={sorted(vals)[len(vals)//2]:.1f} values={['%.1f' % v for v in vals]}")
    for x, y in (("B", "A"), ("B", "C")):
        if x in runs and y in runs:
            seeds = sorted(set(runs[x]) & set(runs[y]))
            deltas = [runs[x][s] - runs[y][s] for s in seeds]
            if deltas:
                print(f"{x}-vs-{y}: paired deltas={['%.1f' % d for d in deltas]} "
                      f"wilcoxon_p={wilcoxon_exact(deltas):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["A", "B", "C"], required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--ticks", type=int, default=20000)
    r.add_argument("--out", required=True)
    r.add_argument("--cpu", action="store_true")
    an = sub.add_parser("analyze")
    an.add_argument("dir")
    an.add_argument("--tau", type=int, default=2000)
    an.add_argument("--transient", type=int, default=1500)
    args = p.parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)
```

- [ ] **Step 4: Run tests + local smoke**

```
venv/bin/python -m pytest tests/agents/rssm/test_mpc_and_rmst.py -q          # green
venv/bin/python -m pytest -q                                                  # full suite green
venv/bin/python scripts/rssm_ab.py run --arm B --seed 1 --ticks 300 --out /tmp/abtest --cpu
venv/bin/python scripts/rssm_ab.py analyze /tmp/abtest --tau 100 --transient 50
```
Expected: smoke run writes events + summary; analyze prints an RMST line. (300 CPU ticks: minutes, not hours.)

- [ ] **Step 5: Commit**

```bash
git add artificial_society/agents/rssm/ scripts/rssm_ab.py tests/agents/rssm/test_mpc_and_rmst.py
git commit -m "feat(rssm): arm-C MPC mode + A/B runner with KM-RMST and exact Wilcoxon"
```

---

### Task 10: Pre-registration file + GPU-PC pilot

**Files:**
- Create: `docs/experiments/rssm-ab-prereg.md`
- No source changes.

- [ ] **Step 1: Write and commit the pre-registration file**

```markdown
# Pre-registration: RSSM sample-efficiency A/B (spec §10.2)

Frozen BEFORE confirmatory seeds. Pilot (3 seeds) may amend §Numbers once, before confirmatory runs.

## Design (frozen)
- Arms: A = v1 baseline; B = RSSM imagination actor; C = RSSM-WM-as-MPC ablation.
- Scale: grid 60×40, initial_population 36 (dashboard defaults). Run length: past founder
  transient to baseline plateau (pilot determines ticks; default 20000).
- Primary endpoint: KM-RMST(τ) of natural-born agents (spawn_origin=="birth", birth ≥ transient
  cutoff), right-censored at run end. Implemented in scripts/rssm_ab.py::rmst (unit-tested).
- Primary regime: transitions-matched. Compute axis reported as secondary.
- Confirmatory tests: {B-vs-A, B-vs-C} on RMST, Wilcoxon signed-rank on within-seed deltas,
  Holm-corrected (2 comparisons). All other metrics exploratory.
- Collapse := population < 8 (MIN_POPULATION) for ≥ 200 consecutive ticks. Collapsed runs
  retained; worst-rank sensitivity analysis reported.
- Minimum effect of interest: +25% RMST (B vs A). No interim peeking at RMST.
- Tuning parity: equal tuning-run budget per arm, all tuning runs logged in this directory.

## Numbers (filled from the 3-seed pilot, then frozen)
- transient cutoff: ___ ticks (founder-overshoot end per pilot population curves)
- τ: ___ ticks; run length: ___ ticks
- SD of within-seed deltas: ___ ; seed-pair correlation r: ___ (if r < 0.3 → unpaired analysis)
- collapse fraction per arm: ___
- N (paired seeds) for 80% power at +25% RMST: ___ ; if N > 12 → downgrade to estimation-with-CI.
```

```bash
git add docs/experiments/rssm-ab-prereg.md
git commit -m "docs(rssm): pre-register A/B design before pilot"
```

- [ ] **Step 2: Sync branch to the GPU-PC and run the 3-seed pilot**

Standing rule: compute-heavy runs happen on the GPU-PC (`ssh mickg@192.168.178.76`, repo `C:\Projects\artificial-society`). The repo convention `CUDA_VISIBLE_DEVICES=-1` must be CLEARED in the A/B shell (arms B/C need CUDA; the runner asserts this).

```bash
git push origin experiment/rssm-dreamer-brain
ssh mickg@192.168.178.76 "cd C:\Projects\artificial-society && git fetch && git checkout experiment/rssm-dreamer-brain && git pull"
# per arm×seed (subprocess isolation = one process per invocation); cmd.exe syntax:
ssh mickg@192.168.178.76 "cd C:\Projects\artificial-society && set CUDA_VISIBLE_DEVICES= && python scripts\rssm_ab.py run --arm A --seed 1 --ticks 20000 --out runs\pilot"
#   ... repeat for arms A,B,C × seeds 1,2,3 (9 runs; long-running: launch via WMI
#   Win32_Process.Create per the remote-host notes so SSH disconnects don't kill them)
ssh mickg@192.168.178.76 "cd C:\Projects\artificial-society && python scripts\rssm_ab.py analyze runs\pilot --tau 2000 --transient 1500"
```

- [ ] **Step 3: Fill prereg §Numbers from the pilot, commit, then hand off**

Compute the power numbers (SD of deltas, r, N) from the pilot output, fill the blanks, commit:
```bash
git add docs/experiments/rssm-ab-prereg.md && git commit -m "docs(rssm): freeze prereg numbers after 3-seed pilot"
```
**STOP for user decision:** confirmatory N seeds are compute-days on the GPU-PC — present the pilot numbers (incl. collapse fractions and whether B shows any signal) and let the user decide to commit N seeds, iterate hyperparameters (equal tuning budget for A!), or halt the lane. This is the spec's §11 make-or-break checkpoint (actor-competence-vs-age).

---

## Coverage map (spec → tasks)

| Spec | Task |
|---|---|
| §4.1 WM (grouped decoder, twohot, KL, zero-init) | 2, 3 |
| §4.2 actor-critic (reinforce default, stop-grads, cumprod weights, λ-returns, return-norm, slow critic, replay-critic API) | 5 |
| §4.3 cultural warm-start | 6, 7 |
| §4.4 replay (per-life, masking, stratified deaths) | 4 |
| §4.5 cadence, prefill/warm-up, re-encode, young train-ratio | 7 |
| §5 seams (flag, kwarg, ensure_fields, spawn+spawn_origin, checkpoint, serve pin, registry hook) | 8 |
| §6 RNG isolation, zero-draw gating, rng-state test | 1, 7, 8 |
| §7 slab-owner checkpointing | 5, 7, 8 |
| §8 device policy, K default, slab + slot hygiene, CUDA gotcha | 5, 7, 9 |
| §9 config table | 1 (+4, 9 additions) |
| §10.1 unit tests | in every task |
| §10.2 A/B (arms, RMST, regime, Wilcoxon, prereg, axes) | 9, 10 |
| §10.3 correctness floor (full pytest + 500-tick no-collapse) | 8 Step 7 |
| §12 phasing | Tasks 1–10 ≙ phases 1–8 |

Known deliberate deviations from spec text (documented in the results note): update-count warm-up gate instead of loss threshold (Task 1); replay-critic term wired but default-off until pilot signal (Task 7); out-of-update deaths close episodes terminal-less (Task 8 Step 3.4).

