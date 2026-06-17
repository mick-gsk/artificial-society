"""
Emergent Invention Engine
--------------------------
Agents perform primitive physical actions on materials in their environment.
There are NO predefined recipes. Outcomes are governed entirely by the
vector-based physical simulation in materials.py.

Two parallel code paths run simultaneously:
  1. Legacy string path  -- keeps fire/cooked_meat/sharp_stone working
  2. Emergent vector path -- produces genuinely new mat_XXXX materials

An agent's brain learns WHAT combinations are valuable purely through
homeostatic reward. No label lookup, no hardcoded recipe tree.
"""

import random
import numpy as np

from artificial_society.environment.materials import (
    apply_interaction,
    combine_vectors,
    material_heat,
    material_light,
    material_danger,
    material_reward,
    decay_materials,
    get_vector,
    MATERIALS,
    DISCOVERY_REGISTRY,
    IDX,
    N_PROPS,
)

PRIMITIVE_ACTIONS = ['rub', 'strike', 'place_on_heat', 'bundle', 'blow', 'carry', 'eat', 'bind']

# Legacy reward constants kept for named-material fast paths
WARMTH_REWARD  = 0.08
LIGHT_REWARD   = 0.06
COOK_REWARD    = 0.35
SHARP_REWARD   = 0.45
DANGER_PENALTY = -0.12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def agent_try_invention(agent, cell: dict, env: dict) -> float:
    """
    Agent picks two materials from the cell (or one from cell + one from
    inventory) and applies a primitive action.
    Returns the reward signal.
    """
    slot = cell.get('materials', {})
    if not slot:
        return 0.0
    available = [m for m, q in slot.items() if q > 0.05]
    if not available:
        return 0.0

    mat_a = random.choice(available)
    mat_b = None
    if len(available) > 1:
        candidates = [m for m in available if m != mat_a]
        mat_b = random.choice(candidates)
    if mat_b is None:
        inv = getattr(agent, 'material_inventory', {})
        if inv:
            mat_b = random.choice(list(inv.keys()))

    causal_mem = getattr(agent, 'causal_memory', None)
    action = _choose_action(agent, causal_mem, mat_a, mat_b)

    # --- Legacy string path ---
    legacy_outcomes = apply_interaction(action, mat_a, mat_b, env)
    legacy_reward   = _evaluate_legacy_outcomes(agent, cell, slot, legacy_outcomes, env)

    # --- Emergent vector path ---
    vec_a    = get_vector(mat_a)
    vec_b    = get_vector(mat_b) if mat_b else None
    new_vec  = combine_vectors(vec_a, vec_b, action, env)
    emergent_reward = 0.0
    if new_vec is not None and float(new_vec.sum()) > 0.1:
        agent_state = _agent_homeostatic_state(agent, cell)
        emergent_reward = material_reward(new_vec, agent_state)
        mat_id = DISCOVERY_REGISTRY.register(
            new_vec,
            discoverer_id=agent.id,
            tick=getattr(agent, 'age', 0),
            recipe=(action, mat_a, mat_b),
        )
        # Place in agent inventory
        inv = getattr(agent, 'material_inventory', {})
        inv[mat_id] = inv.get(mat_id, 0.0) + 0.5
        # Upgrade tool if newly invented material is sharper than current
        _maybe_upgrade_tool(agent, mat_id, new_vec)
        # Endocrine discovery burst
        if emergent_reward > 0.3 and hasattr(agent, 'endocrine'):
            agent.endocrine.apply_discovery(min(1.0, emergent_reward))

    total_reward = legacy_reward + emergent_reward * 0.6

    if causal_mem is not None:
        causal_mem.record(action, mat_a, mat_b, legacy_outcomes, total_reward)

    return total_reward


def agent_try_cook(agent, cell: dict) -> float:
    """
    Explicit cooking attempt (called from agent.update).
    Kept for backward compatibility.
    """
    slot = cell.get('materials', {})
    heat_sources = [m for m in slot if slot[m] > 0.1 and m in ('fire', 'ember')]
    if not heat_sources:
        return 0.0

    inv = getattr(agent, 'material_inventory', {})
    cookable = [m for m in inv if m in ('raw_meat', 'raw_root') and inv[m] > 0.1]
    if cookable:
        source = inv
    else:
        cookable = [m for m in slot if m in ('raw_meat', 'raw_root') and slot[m] > 0.1]
        source = slot
    if not cookable:
        return 0.0

    heat_mat = heat_sources[0]
    food_mat = cookable[0]
    env = {
        'wind':        cell.get('disturbance', 0) / 100.0,
        'moisture':    cell.get('moisture', 50)   / 100.0,
        'temperature': cell.get('temperature', 20),
    }

    # Legacy path
    outcomes     = apply_interaction('place_on_heat', food_mat, heat_mat, env)
    causal_mem   = getattr(agent, 'causal_memory', None)
    legacy_r     = _evaluate_legacy_outcomes(agent, cell, slot, outcomes, env)
    result_mats  = [o for o in outcomes if not o.startswith('_')]
    if result_mats:
        source[food_mat] = max(0.0, source.get(food_mat, 0) - 0.5)

    # Emergent path
    vec_food = get_vector(food_mat)
    vec_heat = get_vector(heat_mat)
    new_vec  = combine_vectors(vec_food, vec_heat, 'place_on_heat', env)
    emergent_r = 0.0
    if new_vec is not None and float(new_vec.sum()) > 0.1:
        agent_state = _agent_homeostatic_state(agent, cell)
        emergent_r  = material_reward(new_vec, agent_state)
        mat_id = DISCOVERY_REGISTRY.register(
            new_vec,
            discoverer_id=agent.id,
            tick=getattr(agent, 'age', 0),
            recipe=('place_on_heat', food_mat, heat_mat),
        )
        inv[mat_id] = inv.get(mat_id, 0.0) + 0.6

    total = legacy_r + emergent_r * 0.6
    if causal_mem is not None:
        causal_mem.record('place_on_heat', food_mat, heat_mat, outcomes, total)
    return total


def share_discovery(teacher, student, mat_id: str) -> bool:
    """
    Social knowledge transfer: teacher shows student a discovered material.
    Student gains the mat_id in inventory if trust is sufficient.
    Returns True if transfer happened.
    """
    trust = teacher.trust.get(student.id, 0.0)
    if trust < 0.25:
        return False
    if mat_id not in DISCOVERY_REGISTRY.known_ids():
        return False
    inv = getattr(student, 'material_inventory', {})
    if inv.get(mat_id, 0.0) < 0.1:
        inv[mat_id] = 0.3
        if hasattr(student, 'endocrine'):
            student.endocrine.apply_discovery(0.3)
        return True
    return False


# ---------------------------------------------------------------------------
# Tick-level world update
# ---------------------------------------------------------------------------

def tick_materials(world):
    for y in range(world.height):
        for x in range(world.width):
            cell = world.cells[y][x]
            if 'materials' in cell and cell['materials']:
                env = {
                    'wind':        cell.get('disturbance', 0) / 100.0,
                    'moisture':    cell.get('moisture', 50)   / 100.0,
                    'temperature': cell.get('temperature', 20),
                }
                decay_materials(cell['materials'], env)


def seed_world_materials(world):
    import random
    biome_mats = {
        'forest':    [('dry_wood', 0.4), ('wet_wood', 0.3), ('dry_grass', 0.2), ('fiber', 0.3)],
        'grassland': [('dry_grass', 0.5), ('fiber', 0.4), ('raw_root', 0.2)],
        'mountain':  [('stone', 0.5), ('flint', 0.25), ('bone', 0.1)],
        'desert':    [('dry_grass', 0.3), ('flint', 0.2), ('stone', 0.3), ('dry_wood', 0.1)],
        'swamp':     [('wet_wood', 0.4), ('fiber', 0.4), ('raw_root', 0.3)],
        'tundra':    [('dry_grass', 0.2), ('bone', 0.15), ('stone', 0.3)],
    }
    for y in range(world.height):
        for x in range(world.width):
            biome = world.biomes[y][x]
            if biome == 'water':
                continue
            cell = world.cells[y][x]
            slot = {}
            for mat, base_qty in biome_mats.get(biome, []):
                if random.random() < 0.35:
                    slot[mat] = round(base_qty * random.uniform(0.5, 1.5), 2)
            cell['materials'] = slot
            cell['warmth'] = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _choose_action(agent, causal_mem, mat_a: str, mat_b: str | None) -> str:
    if causal_mem is None:
        return random.choice(PRIMITIVE_ACTIONS)
    good = causal_mem.best_known(min_successes=1)
    if good and random.random() < 0.55:
        for (act, ma, mb), _ in good:
            if ma == mat_a or mb == (mat_b or ''):
                return act
    return random.choice(PRIMITIVE_ACTIONS)


def _agent_homeostatic_state(agent, cell: dict) -> dict:
    from artificial_society.agents.agent import MAX_ENERGY
    heat = material_heat(cell.get('materials', {}))
    return {
        'energy':    agent.energy    / MAX_ENERGY,
        'hydration': agent.hydration / 100.0,
        'health':    agent.health    / 100.0,
        'cold':      cell.get('temperature', 20) < 8,
        'dark':      cell.get('light', 1.0) < 0.3,
    }


def _maybe_upgrade_tool(agent, mat_id: str, vec: np.ndarray):
    """Equip invented material as tool if it is sharper than current tool."""
    current_sharpness = 0.0
    if agent.tool == 'sharp_stone':
        current_sharpness = float(get_vector('sharp_stone')[IDX['sharpness']])
    elif agent.tool and agent.tool.startswith('mat_'):
        current_sharpness = float(get_vector(agent.tool)[IDX['sharpness']])
    if vec[IDX['sharpness']] > current_sharpness + 0.05:
        agent.tool = mat_id


def _evaluate_legacy_outcomes(
    agent, cell: dict, slot: dict, outcomes: list[str], env: dict
) -> float:
    """Handle named-material outcomes from the legacy apply_interaction path."""
    reward = 0.0
    for result in outcomes:
        if result == '_spark':
            pass
        elif result == '_heat_trace':
            pass
        elif result == '_tinder_bundle':
            inv = getattr(agent, 'material_inventory', {})
            inv['_tinder_bundle'] = inv.get('_tinder_bundle', 0) + 1.0
        elif result.startswith('_nutrition:'):
            val = float(result.split(':')[1])
            agent.energy = min(getattr(agent, 'energy', 100) + val * 18.0, 240.0)
            reward += val * 0.4
        elif result == 'ember':
            slot['ember'] = slot.get('ember', 0.0) + 0.6
            cell['materials'] = slot
            reward += 0.12
        elif result == 'fire':
            slot['fire'] = slot.get('fire', 0.0) + 1.0
            cell['materials'] = slot
            reward += 0.55
        elif result == 'ash':
            slot['ash'] = slot.get('ash', 0.0) + 0.5
        elif result == 'cooked_meat':
            inv = getattr(agent, 'material_inventory', {})
            inv['cooked_meat'] = inv.get('cooked_meat', 0.0) + 0.8
            reward += COOK_REWARD
        elif result == 'cooked_root':
            inv = getattr(agent, 'material_inventory', {})
            inv['cooked_root'] = inv.get('cooked_root', 0.0) + 0.8
            reward += COOK_REWARD * 0.7
        elif result == 'sharp_stone':
            inv = getattr(agent, 'material_inventory', {})
            inv['sharp_stone'] = inv.get('sharp_stone', 0.0) + 1.0
            reward += SHARP_REWARD
            agent.tool = 'sharp_stone'
        elif result.startswith('mat_'):
            # Emergent material produced via legacy path (shouldn't normally happen
            # but handled gracefully)
            inv = getattr(agent, 'material_inventory', {})
            inv[result] = inv.get(result, 0.0) + 0.5

    heat   = material_heat(slot)
    light  = material_light(slot)
    danger = material_danger(slot)
    reward += heat * WARMTH_REWARD + light * LIGHT_REWARD + danger * DANGER_PENALTY
    cell['warmth'] = heat
    return reward
