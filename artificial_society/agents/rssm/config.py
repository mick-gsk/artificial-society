"""All RSSM hyperparameters (spec §9) + the RNG-isolation helper (spec §6)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch


def make_generator(global_seed: int, key: object, device: str = "cpu") -> torch.Generator:
    """torch.Generator seeded from (global_seed, key), independent of global RNG state.

    key is any repr-stable object, e.g. ("agent", agent_id) or "learner".
    """
    mix = zlib.crc32(repr(key).encode("utf-8"))
    gen = torch.Generator(device=device)
    gen.manual_seed((int(global_seed) * 0x9E3779B1 + mix) % (2**63))
    return gen


@dataclass(frozen=True)
class RSSMConfig:
    # substrate
    obs_dim: int = 57
    action_dim: int = 7
    gene_slice: tuple = (17, 21)  # obs dims carrying genes (verified Task 3)
    # latent sizes
    deter: int = 256
    stoch: int = 32
    classes: int = 32
    mlp_hidden: int = 256  # WM MLP width
    ac_hidden: int = 128  # actor/critic MLP width
    # WM loss (spec §4.1)
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    unimix: float = 0.01
    # decoder grouping (dims down-weighted in recon target; filled in Task 3)
    # brain.py:24-41 layout: 15..16 = social (nearby count, friends count),
    # 37..48 = episodic memory retrieval (12) — neither is a deterministic
    # function of the agent's own egocentric state, so both groups are
    # down-weighted in the reconstruction loss (spec §4.1).
    irreducible_dims: tuple = (15, 16, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48)
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
    min_death_seqs: int = 2  # stratified: min sequences containing a terminal
    replay_capacity: int = 200_000  # transitions, FIFO by episode
    # imagination (spec §4.2)
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_eta: float = 3e-4
    actor_grad: str = "reinforce"  # "reinforce" | "dynamics"
    retnorm_decay: float = 0.99
    slow_critic_decay: float = 0.98
    slow_critic_scale: float = 1.0
    # deferred: replay-critic term wired but off until pilot signal (plan deviation)
    replay_critic_scale: float = 0.0
    # cadence (spec §4.5/§8)
    train_every: int = 8  # K
    young_ratio_cap: int = 4  # max extra imagination passes for young actors
    young_age_ticks: int = 200  # "young" = age < this
    wm_warmup_updates: int = 100  # imagination gated until this many WM updates
    prefill_transitions: int = 4096  # random-action prefill before first WM update
    own_start_frac_max: float = 0.75  # per-agent imagination seeding blend (spec §4.2)
    burn_in: int = 8  # burn-in window length for imagination seeding
    # slab (spec §8). 256 gives ample headroom over the ~50-population A/B
    # equilibrium (review fix, Important 7: acquire_slot still raises on
    # exhaustion — a graceful degrade path isn't worth the complexity at this
    # headroom).
    max_slots: int = 256
    prototype_decay: float = 0.995
    # devices
    train_device: str = "cuda"  # WM-train + imagination; act path is ALWAYS cpu
    # policy mode (spec §10.2, arm C): "actor" = slab/imagination/prototype (arm B);
    # "mpc" = WM-only, no slab — act picks the best of K random action candidates by
    # imagined return (arm C). WM training is identical in both modes.
    policy_mode: str = "actor"
    mpc_candidates: int = 12
    mpc_horizon: int = 2
