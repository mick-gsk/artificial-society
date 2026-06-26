"""Central deterministic seeding for reproducible runs.

Phase 0 introduces a single seeding entry point so a run can be reproduced
from a seed. It seeds every stochastic source the simulation currently relies
on (the global ``random`` module, NumPy, and Torch). A later phase may thread
an explicit ``random.Random`` instance through the core; until then this keeps
the global generators as the one seeded source of truth.
"""
from __future__ import annotations

import random


def seed_all(seed: int) -> None:
    """Seed every RNG the simulation uses. Safe to call before constructing the world."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
