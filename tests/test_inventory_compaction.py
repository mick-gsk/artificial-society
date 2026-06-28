"""Behaviour test: inventory compaction is value-ranked, not privileging scripted items.

Phase 5 de-scripting (Task 6). ``_compact_material_inventory`` previously protected every
non-``mat_`` (scripted) item over discovered ``mat_*`` materials and, on overflow, fell back
to a hardcoded "essentials" list. After de-scripting, every item competes purely on its
emergent value to the agent (``material_reward`` over its property vector); a high-value
DISCOVERED material is retained over low-value scripted names.
"""

from artificial_society.agents.agent import _compact_material_inventory
from artificial_society.environment.materials import DISCOVERY_REGISTRY, get_vector


class _Agent:
    def __init__(self, energy, inventory):
        self.energy = energy
        self.material_inventory = dict(inventory)


def _register_high_value_material() -> str:
    # A nutritious discovered material (high edibility) -> high material_reward.
    vec = get_vector("raw_meat").copy()  # edibility 0.5, scent 0.4
    return DISCOVERY_REGISTRY.register(
        vec, discoverer_id=1, tick=0, recipe=("eat", "raw_meat", None)
    )


def test_high_value_discovered_material_kept_over_low_value_scripted():
    mat_id = _register_high_value_material()
    assert mat_id.startswith("mat_")
    # Hungry agent -> nutrition is highly valued; wood/stone are ~worthless here.
    agent = _Agent(energy=20.0, inventory={"wood": 5.0, "stone": 5.0, mat_id: 1.0})

    _compact_material_inventory(agent, max_entries=2)

    # The old code protected wood/stone (scripted) and dropped the discovered material.
    assert mat_id in agent.material_inventory, "high-value discovered material was dropped"
    assert len(agent.material_inventory) == 2


def test_no_overflow_keeps_everything():
    agent = _Agent(energy=120.0, inventory={"wood": 1.0, "fire": 1.0})
    _compact_material_inventory(agent, max_entries=24)
    assert set(agent.material_inventory) == {"wood", "fire"}


def test_compaction_respects_the_cap():
    mat_id = _register_high_value_material()
    inv = {f"junk_{i}": 0.5 for i in range(40)}
    inv[mat_id] = 1.0
    agent = _Agent(energy=20.0, inventory=inv)
    _compact_material_inventory(agent, max_entries=5)
    assert len(agent.material_inventory) == 5
    assert mat_id in agent.material_inventory
