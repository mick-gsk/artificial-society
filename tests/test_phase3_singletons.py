"""Phase 3: all accumulating global singletons must reset on construction.

Reproducibility requires that a freshly-constructed Simulation is independent of
any prior simulation in the same process. `DISCOVERY_REGISTRY` and `TOKEN_WORLD`
already reset; these tests lock in the same guarantee for the two registries that
previously leaked across in-process runs: `SEQUENCE_LIBRARY` and `RECIPE_DISCOVERY`.
"""

from __future__ import annotations

from artificial_society.simulation import Simulation
from artificial_society.systems.goal_stack_ext import (
    RECIPE_DISCOVERY,
    SEQUENCE_LIBRARY,
    SequenceLibrary,
)

SMALL = dict(headless=True, load_checkpoint=False, grid_w=12, grid_h=8, initial_population=4)


def test_recipe_discovery_reset_clears_outcomes():
    RECIPE_DISCOVERY.outcomes["seq_x"] = [("mat_0001", 1.0)]
    RECIPE_DISCOVERY.reset()
    assert RECIPE_DISCOVERY.outcomes == {}


def test_sequence_library_reset_clears_learned_keeps_defaults():
    default_ids = set(SequenceLibrary().sequences)
    SEQUENCE_LIBRARY.sequences["__leak_test__"] = object()
    SEQUENCE_LIBRARY.reset()
    assert "__leak_test__" not in SEQUENCE_LIBRARY.sequences
    assert set(SEQUENCE_LIBRARY.sequences) == default_ids


def test_construction_resets_accumulating_singletons():
    # Pollute the singletons as if a prior in-process run left state behind.
    RECIPE_DISCOVERY.outcomes["seq_leak"] = [("mat_9999", 2.0)]
    SEQUENCE_LIBRARY.sequences["__leak_test__"] = object()

    Simulation(**SMALL)

    assert RECIPE_DISCOVERY.outcomes == {}
    assert "__leak_test__" not in SEQUENCE_LIBRARY.sequences
