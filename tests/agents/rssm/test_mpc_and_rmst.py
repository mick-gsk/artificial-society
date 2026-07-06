from __future__ import annotations

import dataclasses
import importlib.util as _ilu
import pathlib as _pathlib

from artificial_society.agents.rssm.config import RSSMConfig
from artificial_society.simulation import Simulation

_REPO_ROOT = _pathlib.Path(__file__).resolve().parents[3]
_spec = _ilu.spec_from_file_location("rssm_ab", _REPO_ROOT / "scripts" / "rssm_ab.py")
rssm_ab = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rssm_ab)

_PARAMS = dict(headless=True, load_checkpoint=False, grid_w=20, grid_h=15, initial_population=8)
_FAST = dataclasses.replace(
    RSSMConfig(),
    prefill_transitions=64,
    wm_warmup_updates=1,
    batch_size=4,
    seq_len=16,
    horizon=4,
    max_slots=32,
    train_device="cpu",
    policy_mode="mpc",
)


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
    p = rssm_ab.wilcoxon_exact([1.0, 2.0, 3.0, 4.0, 5.0])  # all positive deltas
    assert p == min(1.0, 2 * (1 / 2**5) * 1)  # most extreme rank sum
