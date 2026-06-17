"""
Emergent Invention Engine
--------------------------
Agents perform primitive physical actions on materials in their environment.
There are NO predefined recipes. Outcomes are governed entirely by
artificial_society.environment.materials.apply_interaction().

An agent's brain decides WHAT to try based on local features.
The effects of trying (energy gain, warmth, damage, social attraction)
are fed back as reward — the brain learns purely from consequences.
"""

import random
from artificial_society.environment.materials import (
    apply_interaction,
    material_heat,
    material_light,
    material_danger,
    decay_materials,
    MATERIALS,
)

PRIMITIVE_ACTIONS = ['rub', 'strike', 'place_on_heat', 'bundle', 'blow', 'carry', 'eat']

# How much warmth contributes to comfort (reward)
WARMTH_REWARD = 0.08
# Light at night contributes to safety
LIGHT_REWARD = 0.06
# Cooking reward bonus per successful cooked item created
COOK_REWARD = 0.35
# Sharp tool creation bonus
SHARP_REWARD = 0.45
# Danger penalty from fire/heat exposure
DANGER_PENALTY = -0.12


def agent_try_invention(agent, cell: dict, env: dict) -> float:
    """
    Agent picks up to 2 materials from the cell and tries a random
    primitive action on them.  Outcomes are fed back as reward and
    stored in the agent's CausalMemory.

    Returns the immediate reward signal.
    """
    slot = cell.get('materials', {})
    if not slot:
        return 0.0

    # Pick materials available in cell
    available = [m for m, q in slot.items() if q > 0.05]
    if not available:
        return 0.0

    mat_a = random.choice(available)
    mat_b = random.choice([m for m in available if m != mat_a]) if len(available) > 1 else None

    # Agent also uses materials in their own inventory
    inv = getattr(agent, 'material_inventory', {})
    if not mat_b and inv:
        mat_b = random.choice(list(inv.keys()))

    # Choose action — biased by agent's existing causal memory
    causal_mem = getattr(agent, 'causal_memory', None)
    action = _choose_action(agent, causal_mem, mat_a, mat_b)

    outcomes = apply_interaction(action, mat_a, mat_b, env)
    reward = _evaluate_outcomes(agent, cell, slot, outcomes, env)

    # Store in causal memory
    if causal_mem is not None:
        causal_mem.record(action, mat_a, mat_b, outcomes, reward)

    return reward


def _choose_action(agent, causal_mem, mat_a: str, mat_b: str | None) -> str:
    """Bias action choice toward known successful sequences."""
    if causal_mem is None:
        return random.choice(PRIMITIVE_ACTIONS)
    good = causal_mem.best_known(min_successes=1)
    if good and random.random() < 0.55:  # 55% exploit known good sequences
        for (act, ma, mb), _ in good:
            if ma == mat_a or mb == (mat_b or ''):
                return act
    return random.choice(PRIMITIVE_ACTIONS)


def _evaluate_outcomes(agent, cell: dict, slot: dict, outcomes: list[str], env: dict) -> float:
    reward = 0.0
    for result in outcomes:
        if result == '_spark':
            pass  # no direct reward, but if fire emerges later, that will reward
        elif result == '_heat_trace':
            pass
        elif result == '_tinder_bundle':
            # Store virtual bundle in inventory
            inv = getattr(agent, 'material_inventory', {})
            inv['_tinder_bundle'] = inv.get('_tinder_bundle', 0) + 1.0
        elif result.startswith('_nutrition:'):
            val = float(result.split(':')[1])
            agent.energy = min(getattr(agent, 'energy', 100) + val * 18.0, 240.0)
            reward += val * 0.4
        elif result == 'ember':
            slot['ember'] = slot.get('ember', 0.0) + 0.6
            cell['materials'] = slot
            reward += 0.12  # something warm appeared — interesting
        elif result == 'fire':
            slot['fire'] = slot.get('fire', 0.0) + 1.0
            cell['materials'] = slot
            reward += 0.55  # very interesting — warm, bright, social
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
            if not getattr(agent, 'tool', None):
                agent.tool = 'sharp_stone'

    # Environmental effects of heat/fire in cell
    heat = material_heat(slot)
    light = material_light(slot)
    danger = material_danger(slot)
    reward += heat * WARMTH_REWARD
    reward += light * LIGHT_REWARD
    reward += danger * DANGER_PENALTY
    # Standing near fire increases social attraction (warmth draws others)
    cell['warmth'] = heat

    return reward


def tick_materials(world):
    """Decay all material slots in the world every tick."""
    for y in range(world.height):
        for x in range(world.width):
            cell = world.cells[y][x]
            if 'materials' in cell and cell['materials']:
                env = {
                    'wind': cell.get('disturbance', 0) / 100.0,
                    'moisture': cell.get('moisture', 50) / 100.0,
                    'temperature': cell.get('temperature', 20),
                }
                decay_materials(cell['materials'], env)


def seed_world_materials(world):
    """
    Place raw materials in the world at spawn.
    Agents never know what they are by name — they discover properties
    through interaction and remember the causal sequences that work.
    """
    from artificial_society.environment.biomes import BIOME_BASE
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
