"""
Materials & Affordance System
-----------------------------
Objects and cells carry physical properties (affordances).
Fire, cooking, sharpening etc. are NOT hardcoded recipes.
They emerge from physical state transitions when agents interact
with objects that have compatible properties.
"""

import random

# ---------------------------------------------------------------------------
# Material property definitions
# ---------------------------------------------------------------------------

# Each material is a dict of physical properties (floats 0-1 or bool).
# Agents cannot read these labels directly — they only observe the EFFECTS
# (warmth, energy gain, damage, light, smell) via their feature vector.

MATERIALS = {
    'dry_grass':  {'flammable': 1.0, 'dryness': 1.0, 'hardness': 0.05, 'edible_raw': 0.2, 'mass': 0.1},
    'dry_wood':   {'flammable': 0.85,'dryness': 0.9,  'hardness': 0.5,  'edible_raw': 0.0, 'mass': 0.6},
    'wet_wood':   {'flammable': 0.15,'dryness': 0.1,  'hardness': 0.5,  'edible_raw': 0.0, 'mass': 0.8},
    'stone':      {'flammable': 0.0, 'dryness': 0.5,  'hardness': 1.0,  'edible_raw': 0.0, 'mass': 1.0, 'spark_on_strike': 0.35},
    'flint':      {'flammable': 0.0, 'dryness': 0.9,  'hardness': 1.0,  'edible_raw': 0.0, 'mass': 0.8, 'spark_on_strike': 0.75, 'sharp_when_knapped': 0.9},
    'bone':       {'flammable': 0.2, 'dryness': 0.6,  'hardness': 0.6,  'edible_raw': 0.1, 'mass': 0.3},
    'raw_meat':   {'flammable': 0.1, 'dryness': 0.3,  'hardness': 0.1,  'edible_raw': 0.5, 'edible_cooked': 0.95, 'mass': 0.4, 'perishable': True},
    'raw_root':   {'flammable': 0.05,'dryness': 0.3,  'hardness': 0.2,  'edible_raw': 0.4, 'edible_cooked': 0.9,  'mass': 0.2, 'perishable': True},
    'ember':      {'flammable': 0.0, 'dryness': 1.0,  'hardness': 0.0,  'edible_raw': 0.0, 'heat': 0.6, 'light': 0.4, 'mass': 0.05, 'decays': True},
    'fire':       {'flammable': 0.0, 'dryness': 1.0,  'hardness': 0.0,  'edible_raw': 0.0, 'heat': 1.0, 'light': 1.0, 'mass': 0.0,  'decays': True, 'dangerous': True},
    'ash':        {'flammable': 0.0, 'dryness': 1.0,  'hardness': 0.0,  'edible_raw': 0.0, 'heat': 0.05,'mass': 0.05},
    'cooked_meat':{'flammable': 0.0, 'dryness': 0.7,  'hardness': 0.2,  'edible_raw': 0.0, 'edible_cooked': 1.0, 'mass': 0.35, 'energy_bonus': 0.4},
    'cooked_root':{'flammable': 0.0, 'dryness': 0.5,  'hardness': 0.1,  'edible_raw': 0.0, 'edible_cooked': 1.0, 'mass': 0.15, 'energy_bonus': 0.25},
    'sharp_stone':{'flammable': 0.0, 'dryness': 0.8,  'hardness': 0.95, 'edible_raw': 0.0, 'mass': 0.5, 'cutting': 0.8},
    'fiber':      {'flammable': 0.5, 'dryness': 0.5,  'hardness': 0.05, 'edible_raw': 0.0, 'mass': 0.1},
}


def get_material(name: str) -> dict:
    base = MATERIALS.get(name, {})
    return dict(base)


# ---------------------------------------------------------------------------
# State transition engine — the core of emergent invention
# ---------------------------------------------------------------------------

def apply_interaction(action: str, mat_a: str, mat_b: str | None, env: dict) -> list[str]:
    """
    Given a primitive action and two materials (or one + env context),
    return a list of resulting material names produced.
    NO material is labelled 'fire' upfront — it can only appear as a
    consequence of physically plausible state sequences.

    env keys: 'wind' (0-1), 'moisture' (0-1), 'temperature' (-20..50)
    """
    results = []
    props_a = get_material(mat_a)
    props_b = get_material(mat_b) if mat_b else {}

    if action == 'strike':
        spark_chance = props_a.get('spark_on_strike', 0.0) + props_b.get('spark_on_strike', 0.0)
        spark_chance *= (1.0 - env.get('moisture', 0.5))
        if random.random() < spark_chance:
            # Spark produced — if flammable tinder nearby, ember can form
            results.append('_spark')  # internal signal, not a storable object
        if props_a.get('sharp_when_knapped', 0) > 0.5 and random.random() < props_a['sharp_when_knapped']:
            results.append('sharp_stone')

    elif action == 'rub':
        friction_heat = props_a.get('hardness', 0) * props_b.get('hardness', 0)
        dryness = min(props_a.get('dryness', 0), props_b.get('dryness', 0))
        ignite_chance = friction_heat * dryness * props_a.get('flammable', 0) * 0.4
        ignite_chance *= (1.0 - env.get('moisture', 0.5))
        ignite_chance *= max(0.5, env.get('wind', 0.3))
        if random.random() < ignite_chance:
            results.append('ember')
        elif random.random() < friction_heat * 0.3:
            results.append('_heat_trace')

    elif action == 'place_on_heat':
        heat = props_b.get('heat', 0.0)
        if heat > 0.5:
            if mat_a in ('raw_meat',) and random.random() < heat:
                results.append('cooked_meat')
            elif mat_a in ('raw_root',) and random.random() < heat * 0.9:
                results.append('cooked_root')
            elif props_a.get('flammable', 0) > 0.5 and random.random() < heat * props_a['flammable']:
                if heat > 0.8:
                    results.append('fire')
                else:
                    results.append('ember')

    elif action == 'bundle':
        combined_flamm = max(props_a.get('flammable', 0), props_b.get('flammable', 0))
        combined_dry = (props_a.get('dryness', 0) + props_b.get('dryness', 0)) / 2
        # Bundling flammables increases ignition potential stored in the bundle
        if combined_flamm > 0.5 and combined_dry > 0.6:
            results.append('_tinder_bundle')  # virtual: better flammable score

    elif action == 'carry':
        # carrying ember keeps it alive with some probability
        if mat_a == 'ember':
            results.append('ember' if random.random() < 0.55 else 'ash')

    elif action == 'blow':
        if mat_a == 'ember':
            boost = env.get('wind', 0.3) + 0.3
            if random.random() < boost:
                results.append('fire')
            else:
                results.append('ember')
        elif mat_a == 'fire':
            results.append('fire')  # stays alive

    elif action == 'eat':
        edibility = props_a.get('edible_cooked', 0) if mat_a.startswith('cooked') else props_a.get('edible_raw', 0)
        if edibility > 0.3:
            results.append('_nutrition:' + str(round(edibility + props_a.get('energy_bonus', 0.0), 2)))

    return results


# ---------------------------------------------------------------------------
# World-cell material slots
# ---------------------------------------------------------------------------

def empty_material_slot():
    """Each cell can hold up to 4 material stacks."""
    return {}  # {material_name: quantity}


def decay_materials(slot: dict, env: dict) -> dict:
    """Tick-wise decay for perishables and burning materials."""
    remove = []
    for mat, qty in slot.items():
        props = get_material(mat)
        if props.get('decays'):
            # fire/ember diminish over time unless fed
            slot[mat] = max(0.0, qty - 0.05 - 0.03 * env.get('wind', 0.3))
            if slot[mat] < 0.05:
                remove.append(mat)
                if mat == 'fire':
                    slot['ash'] = slot.get('ash', 0.0) + qty * 0.4
                elif mat == 'ember':
                    slot['ash'] = slot.get('ash', 0.0) + qty * 0.8
        elif props.get('perishable') and env.get('moisture', 0.5) > 0.6:
            slot[mat] = max(0.0, qty - 0.01)
            if slot[mat] < 0.01:
                remove.append(mat)
    for m in remove:
        del slot[m]
    return slot


def material_heat(slot: dict) -> float:
    """Total heat emitted by materials in a cell slot."""
    return sum(get_material(m).get('heat', 0.0) * q for m, q in slot.items())


def material_light(slot: dict) -> float:
    return sum(get_material(m).get('light', 0.0) * q for m, q in slot.items())


def material_danger(slot: dict) -> float:
    return sum(1.0 * q for m, q in slot.items() if get_material(m).get('dangerous'))
