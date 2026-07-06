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
