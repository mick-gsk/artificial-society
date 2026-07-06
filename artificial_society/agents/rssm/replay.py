"""Per-agent-life segmented sequence replay (spec §4.4). Stores raw transitions only."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import RSSMConfig


@dataclass
class _Episode:
    agent_id: int
    origin: str
    obs: list = field(default_factory=list)  # (57,) float32 tensors
    act: list = field(default_factory=list)  # (7,) float32 tensors
    reward: list = field(default_factory=list)  # floats
    done: bool = False  # True iff ended by real death
    open: bool = True

    def __len__(self):
        return len(self.obs)


class SharedReplay:
    def __init__(self, cfg: RSSMConfig) -> None:
        self.cfg = cfg
        self._episodes: deque[_Episode] = deque()  # closed episodes, FIFO
        self._open: OrderedDict[int, _Episode] = OrderedDict()
        self._size = 0

    # --- writing ----------------------------------------------------------
    def start_episode(self, agent_id: int, origin: str) -> None:
        self._open[agent_id] = _Episode(agent_id, origin)

    def add(self, agent_id: int, obs: Any, action: Any, reward: float, done: bool) -> None:
        ep = self._open.get(agent_id)
        if ep is None:  # tolerate missed start
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

    def _window(
        self, ep: _Episode, gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            cont[k - 1] = 0.0  # real death only
        return obs, act, rew, cont, mask

    def sample_sequences(self, gen: torch.Generator) -> dict[str, torch.Tensor]:
        eps = self._all()
        if not eps:
            raise ValueError("empty replay")
        B = self.cfg.batch_size
        death_eps = [e for e in eps if e.done]
        # Sample with replacement from death episodes to guarantee min_death_seqs when any death exists
        n_deaths = min(self.cfg.min_death_seqs, B) if death_eps else 0
        picks = []
        for _ in range(n_deaths):
            picks.append(death_eps[int(torch.randint(0, len(death_eps), (1,), generator=gen))])
        while len(picks) < B:
            picks.append(eps[int(torch.randint(0, len(eps), (1,), generator=gen))])
        # deaths must land inside the window: bias those windows to the episode tail
        cols = [self._window(e, gen) for e in picks[n_deaths:]]
        for e in picks[:n_deaths]:
            L = self.cfg.seq_len
            start = max(0, len(e) - L)
            tail = _Episode(
                e.agent_id, e.origin, e.obs[start:], e.act[start:], e.reward[start:], e.done, False
            )
            cols.insert(0, self._window(tail, gen))
        obs, act, rew, cont, mask = (torch.stack(x) for x in zip(*cols))
        return {"obs": obs, "act": act, "reward": rew, "cont": cont, "mask": mask}

    def sample_starts(
        self, agent_id: int, n: int, own_frac: float, gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        b = self.cfg.burn_in
        pool = [e for e in self._all() if len(e) >= b]
        if not pool:
            return None
        own = [e for e in pool if e.agent_id == agent_id] or pool
        n_own = int(round(n * own_frac))
        srcs = [own[int(torch.randint(0, len(own), (1,), generator=gen))] for _ in range(n_own)]
        srcs += [
            pool[int(torch.randint(0, len(pool), (1,), generator=gen))] for _ in range(n - n_own)
        ]
        obs_w, act_w = [], []
        for e in srcs:
            s = int(torch.randint(0, len(e) - b + 1, (1,), generator=gen))
            obs_w.append(torch.stack(e.obs[s : s + b]))
            act_w.append(torch.stack(e.act[s : s + b]))
        return torch.stack(obs_w), torch.stack(act_w)
