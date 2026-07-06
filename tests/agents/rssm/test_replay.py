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
    _fill(rp, 1, 40, die=True, mark=1.0)  # life A, obs==1
    _fill(rp, 1, 40, die=True, mark=2.0)  # life B (same agent reborn), obs==2
    batch = rp.sample_sequences(make_generator(0, "s"))
    valid = batch["mask"].bool()
    per_seq_vals = [batch["obs"][b][valid[b]][:, 0].unique() for b in range(batch["obs"].shape[0])]
    assert all(len(v) == 1 for v in per_seq_vals)  # one life per window


def test_padding_masked_and_cont_semantics():
    rp = SharedReplay(CFG)
    _fill(rp, 3, 10, die=True)  # short life → padded window
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
        _fill(rp, a, 80, die=False)  # many survivor episodes
    _fill(rp, 99, 30, die=True)  # one death episode
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
    assert (obs[:, :, 0] == 5.0).all()  # own_frac=1 → all windows from agent 5


def test_fifo_eviction_whole_episodes():
    import dataclasses

    cfg = dataclasses.replace(CFG, replay_capacity=100)
    rp = SharedReplay(cfg)
    _fill(rp, 1, 60, mark=1.0)
    _fill(rp, 2, 60, mark=2.0)  # pushes over 100 → episode 1 evicted whole
    assert len(rp) == 60
