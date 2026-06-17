"""
Emergent Invention Engine
--------------------------
Agents perform primitive physical actions on materials in their environment.
There are NO predefined recipes. Outcomes are governed entirely by
artificial_society.environment.materials.apply_interaction().

An agent's brain decides WHAT to try based on local features.
The effects of trying (energy gain, warmth, damage, social attraction)
are fed back as reward -- the brain learns purely from consequences.
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

WARMTH_REWARD  = 0.08
LIGHT_REWARD   = 0.06
COOK_REWARD    = 0.35
SHARP_REWARD   = 0.45
DANGER_PENALTY = -0.12


def agent_try_invention(agent, cell: dict, env: dict) -> float:
    slot = cell.get('materials', {})
    if not slot:
        return 0.0
    available = [m for m, q in slot.items() if q > 0.05]
    if not available:
        return 0.0

    mat_a = random.choice(available)
    mat_b = random.choice([m for m in available if m != mat_a]) if len(available) > 1 else None

    inv = getattr(agent, 'material_inventory', {})
    if not mat_b and inv:
        mat_b = random.choice(list(inv.keys()))

    causal_mem = getattr(agent, 'causal_memory', None)
    action = _choose_action(agent, causal_mem, mat_a, mat_b)
    outcomes = apply_interaction(action, mat_a, mat_b, env)
    reward = _evaluate_outcomes(agent, cell, slot, outcomes, env)

    if causal_mem is not None:
        causal_mem.record(action, mat_a, mat_b, outcomes, reward)

    return reward


def agent_try_cook(agent, cell: dict) -> float:
    """
    Explizites Kochversuch-Interface fuer maybe_craft().
    Agent versucht Rohfleisch oder Rohwurzel auf einem Feuer/Ember
    in der aktuellen Zelle zu garen.
    Kein vordefiniertes Rezept -- der Agent muss 'place_on_heat' mit
    dem richtigen Material und einer Hitzequelle kombinieren.
    Gibt 0.0 zurueck wenn keine Hitzequelle oder kein rohes Essen vorhanden.
    Reward kommt ausschliesslich durch den Energiegewinn beim spaetere Essen.
    """
    slot = cell.get('materials', {})
    heat_sources = [m for m in slot if slot[m] > 0.1 and m in ('fire', 'ember')]
    if not heat_sources:
        return 0.0

    inv = getattr(agent, 'material_inventory', {})
    cookable = [m for m in inv if m in ('raw_meat', 'raw_root') and inv[m] > 0.1]
    if not cookable:
        # Auch in der Zelle nach kochbarem suchen
        cookable = [m for m in slot if m in ('raw_meat', 'raw_root') and slot[m] > 0.1]
        source = slot
    else:
        source = inv

    if not cookable:
        return 0.0

    heat_mat = heat_sources[0]
    food_mat = cookable[0]
    env = {
        'wind':        cell.get('disturbance', 0) / 100.0,
        'moisture':    cell.get('moisture', 50)   / 100.0,
        'temperature': cell.get('temperature', 20),
    }
    outcomes = apply_interaction('place_on_heat', food_mat, heat_mat, env)
    if not outcomes:
        return 0.0

    causal_mem = getattr(agent, 'causal_memory', None)
    reward = _evaluate_outcomes(agent, cell, slot, outcomes, env)
    if causal_mem is not None:
        causal_mem.record('place_on_heat', food_mat, heat_mat, outcomes, reward)
    # Rohes Material verbrauchen wenn Ergebnis produziert wurde
    result_mats = [o for o in outcomes if not o.startswith('_')]
    if result_mats:
        source[food_mat] = max(0.0, source.get(food_mat, 0) - 0.5)
    return reward


def _choose_action(agent, causal_mem, mat_a: str, mat_b: str | None) -> str:
    if causal_mem is None:
        return random.choice(PRIMITIVE_ACTIONS)
    good = causal_mem.best_known(min_successes=1)
    if good and random.random() < 0.55:
        for (act, ma, mb), _ in good:
            if ma == mat_a or mb == (mat_b or ''):
                return act
    return random.choice(PRIMITIVE_ACTIONS)


def _evaluate_outcomes(agent, cell: dict, slot: dict, outcomes: list[str], env: dict) -> float:
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
            # Tool setzen -- auch wenn bereits eines vorhanden ist
            # (Agent traegt jetzt explizit das scharfe Werkzeug)
            agent.tool = 'sharp_stone'

    heat   = material_heat(slot)
    light  = material_light(slot)
    danger = material_danger(slot)
    reward += heat * WARMTH_REWARD
    reward += light * LIGHT_REWARD
    reward += danger * DANGER_PENALTY
    cell['warmth'] = heat
    return reward


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
