"""Lane test: token creation is open to any usable marker, not gated on pigment props.

Phase 5 de-scripting (Task 2). ``agent_mark`` previously required a pigment-like material
(``solubility > 0.3 AND (conductivity > 0.2 OR scent > 0.3)``). After de-scripting,
signalling can emerge from ANY usable marker the agent holds, provided a markable surface
is present — so symbols aren't capped at the handful of "pigment" materials.
"""

import numpy as np

from artificial_society.environment.materials import N_PROPS
from artificial_society.systems.language import TOKEN_WORLD, agent_mark


class _MarkAgent:
    def __init__(self, inventory):
        self.id = 1
        self.pos = (3, 3)
        self.x = 3
        self.y = 3
        self.material_inventory = dict(inventory)
        self.trust = {}


def test_non_pigment_material_can_seed_a_token():
    TOKEN_WORLD.reset()
    # 'stone' fails the old pigment gate (solubility 0.0) but is a usable marker.
    agent = _MarkAgent({"stone": 1.0})
    cell = {"materials": {"stone": 1.0}}  # markable surface present
    context = np.ones(N_PROPS, dtype=np.float32)

    token_id = agent_mark(agent, cell, context, tick=5)

    assert token_id is not None, "a usable non-pigment marker failed to seed a token"
    assert token_id in TOKEN_WORLD.tokens


def test_no_marker_means_no_token():
    """With nothing usable in inventory, no token forms (guard against over-loosening)."""
    TOKEN_WORLD.reset()
    agent = _MarkAgent({})
    cell = {"materials": {"stone": 1.0}}
    context = np.ones(N_PROPS, dtype=np.float32)

    assert agent_mark(agent, cell, context, tick=5) is None


def test_marker_requires_a_markable_surface():
    """A held marker still needs a surface — loosening the marker doesn't drop the surface."""
    TOKEN_WORLD.reset()
    agent = _MarkAgent({"stone": 1.0})
    cell = {"materials": {"dry_grass": 1.0}}  # no stone/clay/flat surface
    context = np.ones(N_PROPS, dtype=np.float32)

    assert agent_mark(agent, cell, context, tick=5) is None
