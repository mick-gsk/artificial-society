"""
Materials & Emergent Invention System
--------------------------------------
Every material is a 12-dimensional physical property vector.
NO string-based recipe lookup. Outcomes are synthesised by
physical interaction rules that can produce genuinely new vectors
nobody has ever defined upfront.

Property dimensions (PROP_DIMS):
  0  flammable       -- ignition susceptibility   [0..1]
  1  hardness        -- structural rigidity        [0..1]
  2  edibility       -- caloric/nutritional value  [0..1]
  3  toxicity        -- harm when ingested         [0..1]
  4  heat_emission   -- radiated warmth            [0..1]
  5  light_emission  -- luminance                  [0..1]
  6  mass            -- physical weight            [0..1]
  7  dryness         -- absence of moisture        [0..1]
  8  sharpness       -- cutting ability            [0..1]
  9  solubility      -- dissolves in liquid        [0..1]
  10 conductivity    -- heat/energy transfer       [0..1]
  11 scent           -- volatile aroma strength    [0..1]
"""

import random
import math
import numpy as np

# ---------------------------------------------------------------------------
# Property index map
# ---------------------------------------------------------------------------
PROP_DIMS = [
    'flammable', 'hardness', 'edibility', 'toxicity',
    'heat_emission', 'light_emission', 'mass',
    'dryness', 'sharpness', 'solubility', 'conductivity', 'scent',
]
IDX = {name: i for i, name in enumerate(PROP_DIMS)}
N_PROPS = len(PROP_DIMS)  # 12


def _v(**kwargs) -> np.ndarray:
    """Build a property vector from keyword args; missing dims default to 0."""
    v = np.zeros(N_PROPS, dtype=np.float32)
    for k, val in kwargs.items():
        v[IDX[k]] = float(val)
    return v


# ---------------------------------------------------------------------------
# Seed materials as property vectors
# Legacy string names are preserved so existing world/agent code still works.
# ---------------------------------------------------------------------------
MATERIALS: dict[str, np.ndarray] = {
    'dry_grass':   _v(flammable=1.0, hardness=0.05, edibility=0.2,  mass=0.1,  dryness=1.0),
    'dry_wood':    _v(flammable=0.85,hardness=0.5,  edibility=0.0,  mass=0.6,  dryness=0.9),
    'wet_wood':    _v(flammable=0.15,hardness=0.5,  edibility=0.0,  mass=0.8,  dryness=0.1),
    'stone':       _v(flammable=0.0, hardness=1.0,  edibility=0.0,  mass=1.0,  dryness=0.5,  conductivity=0.4),
    'flint':       _v(flammable=0.0, hardness=1.0,  edibility=0.0,  mass=0.8,  dryness=0.9,  sharpness=0.9,  conductivity=0.3),
    'bone':        _v(flammable=0.2, hardness=0.6,  edibility=0.1,  mass=0.3,  dryness=0.6,  sharpness=0.3),
    'raw_meat':    _v(flammable=0.1, hardness=0.1,  edibility=0.5,  mass=0.4,  dryness=0.3,  scent=0.4),
    'raw_root':    _v(flammable=0.05,hardness=0.2,  edibility=0.4,  mass=0.2,  dryness=0.3,  solubility=0.2),
    'ember':       _v(flammable=0.0, hardness=0.0,  edibility=0.0,  mass=0.05, dryness=1.0,  heat_emission=0.6, light_emission=0.4),
    'fire':        _v(flammable=0.0, hardness=0.0,  edibility=0.0,  mass=0.0,  dryness=1.0,  heat_emission=1.0, light_emission=1.0, toxicity=0.1),
    'ash':         _v(flammable=0.0, hardness=0.0,  edibility=0.0,  mass=0.05, dryness=1.0,  heat_emission=0.05,solubility=0.3),
    'cooked_meat': _v(flammable=0.0, hardness=0.2,  edibility=0.95, mass=0.35, dryness=0.7,  scent=0.55),
    'cooked_root': _v(flammable=0.0, hardness=0.1,  edibility=0.85, mass=0.15, dryness=0.5,  scent=0.2),
    'sharp_stone': _v(flammable=0.0, hardness=0.95, edibility=0.0,  mass=0.5,  dryness=0.8,  sharpness=0.8),
    'fiber':       _v(flammable=0.5, hardness=0.05, edibility=0.0,  mass=0.1,  dryness=0.5),
}

# Legacy compatibility: boolean/special flags per material
_LEGACY_FLAGS: dict[str, dict] = {
    'raw_meat':    {'perishable': True},
    'raw_root':    {'perishable': True},
    'ember':       {'decays': True},
    'fire':        {'decays': True, 'dangerous': True},
    'stone':       {'spark_on_strike': 0.35},
    'flint':       {'spark_on_strike': 0.75, 'sharp_when_knapped': 0.9},
    'cooked_meat': {'energy_bonus': 0.4},
    'cooked_root': {'energy_bonus': 0.25},
}


def get_material(name: str) -> dict:
    """Legacy-compatible dict accessor."""
    vec = MATERIALS.get(name, np.zeros(N_PROPS, dtype=np.float32))
    d = {dim: float(vec[i]) for i, dim in enumerate(PROP_DIMS)}
    d.update(_LEGACY_FLAGS.get(name, {}))
    return d


def get_vector(name_or_id: str) -> np.ndarray:
    """Return property vector for a named seed material or a discovered mat_XXXX."""
    if name_or_id in MATERIALS:
        return MATERIALS[name_or_id].copy()
    return DISCOVERY_REGISTRY.get_vector(name_or_id)


# ---------------------------------------------------------------------------
# Discovery Registry — runtime-invented materials
# ---------------------------------------------------------------------------
class DiscoveryRegistry:
    """
    Stores all materials invented at runtime.
    Uses Euclidean distance to deduplicate near-identical vectors.
    """
    def __init__(self, similarity_threshold: float = 0.08):
        self.entries: list[dict] = []
        self.threshold = similarity_threshold

    def register(
        self,
        vector: np.ndarray,
        discoverer_id: int = -1,
        tick: int = 0,
        recipe: tuple | None = None,
    ) -> str:
        for entry in self.entries:
            if np.linalg.norm(vector - entry['vector']) < self.threshold:
                return entry['id']
        new_id = f'mat_{len(self.entries):04d}'
        self.entries.append({
            'id':           new_id,
            'vector':       vector.copy(),
            'discovered_by': discoverer_id,
            'tick':         tick,
            'recipe':       recipe,  # (action, mat_a, mat_b)
            'uses':         0,
        })
        print(f'[DISCOVERY] tick={tick} agent={discoverer_id} invented {new_id} '
              f'edibility={vector[IDX["edibility"]]:.2f} '
              f'heat={vector[IDX["heat_emission"]]:.2f} '
              f'sharpness={vector[IDX["sharpness"]]:.2f} '
              f'scent={vector[IDX["scent"]]:.2f}')
        return new_id

    def get_vector(self, mat_id: str) -> np.ndarray:
        for entry in self.entries:
            if entry['id'] == mat_id:
                entry['uses'] += 1
                return entry['vector'].copy()
        return np.zeros(N_PROPS, dtype=np.float32)

    def known_ids(self) -> list[str]:
        return [e['id'] for e in self.entries]

    def summary(self) -> list[dict]:
        return [
            {
                'id':    e['id'],
                'tick':  e['tick'],
                'agent': e['discovered_by'],
                'uses':  e['uses'],
                'props': {PROP_DIMS[i]: round(float(e['vector'][i]), 3)
                          for i in range(N_PROPS) if e['vector'][i] > 0.01},
            }
            for e in self.entries
        ]


# Global singleton — imported by invention.py and agent.py
DISCOVERY_REGISTRY = DiscoveryRegistry()


# ---------------------------------------------------------------------------
# Physical combination engine
# Produces a NEW property vector from two inputs + action + environment.
# Returns None when no stable reaction occurs.
# ---------------------------------------------------------------------------
def _thermal_reaction(
    vec_a: np.ndarray, vec_b: np.ndarray, env: dict
) -> np.ndarray | None:
    """Heat from b transforms a (cooking, burning, drying)."""
    heat = vec_b[IDX['heat_emission']]
    if heat < 0.35:
        return None
    moisture_damp = 1.0 - env.get('moisture', 0.5) * 0.6
    effective_heat = heat * moisture_damp
    result = vec_a.copy()
    # Edibility increases (Maillard / cooking)
    if vec_a[IDX['edibility']] > 0.1:
        result[IDX['edibility']] = min(1.0, vec_a[IDX['edibility']] * (1.0 + 0.8 * effective_heat))
        result[IDX['toxicity']]  = max(0.0, vec_a[IDX['toxicity']]  * (1.0 - 0.7 * effective_heat))
        result[IDX['scent']]     = min(1.0, vec_a[IDX['scent']] + 0.25 * effective_heat)
        result[IDX['mass']]     *= max(0.7, 1.0 - 0.15 * effective_heat)
    # Flammable materials partially combust
    if vec_a[IDX['flammable']] > 0.5 and effective_heat > 0.6:
        result[IDX['flammable']]     = max(0.0, vec_a[IDX['flammable']] - 0.5)
        result[IDX['heat_emission']] = min(1.0, vec_a[IDX['heat_emission']] + 0.4 * effective_heat)
        result[IDX['light_emission']]= min(1.0, vec_a[IDX['light_emission']] + 0.3)
        result[IDX['mass']]         *= 0.6
        result[IDX['dryness']]       = 1.0
    # Drying
    result[IDX['dryness']] = min(1.0, vec_a[IDX['dryness']] + 0.3 * effective_heat)
    if np.linalg.norm(result - vec_a) < 0.05:
        return None
    return result


def _mechanical_reaction(
    vec_a: np.ndarray, vec_b: np.ndarray, action: str
) -> np.ndarray | None:
    """Striking/binding/bundling — structural combination."""
    if action == 'strike':
        # Hard on hard: knapping → sharpness emerges
        if vec_a[IDX['hardness']] > 0.6 and vec_b[IDX['hardness']] > 0.6:
            result = vec_a.copy()
            result[IDX['sharpness']] = min(1.0, (vec_a[IDX['hardness']] + vec_b[IDX['hardness']]) * 0.55)
            result[IDX['mass']]     *= 0.75  # fragments shed
            return result
        # Hard on soft: pulverising → solubility increases
        if vec_a[IDX['hardness']] < 0.3 and vec_b[IDX['hardness']] > 0.7:
            result = vec_a.copy()
            result[IDX['solubility']] = min(1.0, vec_a[IDX['solubility']] + 0.4)
            result[IDX['mass']]      *= 0.8
            result[IDX['hardness']]  *= 0.3
            return result

    elif action == 'bind':
        # Flexible binds rigid: composite tool/material
        flex = vec_a[IDX['hardness']] < 0.25
        rigid = vec_b[IDX['hardness']] > 0.5
        if flex and rigid:
            result = np.zeros(N_PROPS, dtype=np.float32)
            result[IDX['hardness']]  = vec_b[IDX['hardness']] * 0.8
            result[IDX['mass']]      = vec_a[IDX['mass']] + vec_b[IDX['mass']]
            result[IDX['sharpness']] = vec_b[IDX['sharpness']]  # stone edge preserved
            result[IDX['flammable']] = vec_a[IDX['flammable']] * 0.5  # fiber portion
            return result

    elif action == 'bundle':
        # Two materials bundled together → additive properties, new composite
        combined_flamm = (vec_a[IDX['flammable']] + vec_b[IDX['flammable']]) * 0.6
        combined_dry   = (vec_a[IDX['dryness']]   + vec_b[IDX['dryness']])   * 0.5
        if combined_flamm > 0.4 or combined_dry > 0.55:
            result = (vec_a + vec_b) * 0.5
            result[IDX['flammable']] = combined_flamm
            result[IDX['dryness']]   = combined_dry
            result[IDX['mass']]      = vec_a[IDX['mass']] + vec_b[IDX['mass']]
            return result

    return None


def _solvent_reaction(
    vec_a: np.ndarray, vec_b: np.ndarray, env: dict
) -> np.ndarray | None:
    """Dissolving/mixing: solubility drives extraction of volatile compounds."""
    moisture = env.get('moisture', 0.0)
    solubility_a = vec_a[IDX['solubility']]
    if solubility_a < 0.25 or moisture < 0.4:
        return None
    # Volatile scent compounds extracted (perfume/dye analog)
    result = vec_b.copy()  # carrier (liquid base)
    result[IDX['scent']]      = min(1.0, vec_b[IDX['scent']] + vec_a[IDX['scent']] * solubility_a * 1.4)
    result[IDX['edibility']]  = min(1.0, (vec_a[IDX['edibility']] + vec_b[IDX['edibility']]) * 0.55)
    result[IDX['toxicity']]   = max(vec_a[IDX['toxicity']], vec_b[IDX['toxicity']])
    result[IDX['solubility']] = (vec_a[IDX['solubility']] + vec_b[IDX['solubility']]) * 0.5
    result[IDX['mass']]       = (vec_a[IDX['mass']] + vec_b[IDX['mass']]) * 0.7
    if np.linalg.norm(result - vec_b) < 0.05:
        return None
    return result


def combine_vectors(
    vec_a: np.ndarray,
    vec_b: np.ndarray | None,
    action: str,
    env: dict,
) -> np.ndarray | None:
    """
    Core emergent synthesis engine.
    Returns a new property vector or None if no stable reaction.
    The result is clamped to [0,1] in all dimensions.
    """
    if vec_b is None:
        vec_b = np.zeros(N_PROPS, dtype=np.float32)

    result = None

    if action in ('place_on_heat', 'blow', 'carry'):
        result = _thermal_reaction(vec_a, vec_b, env)
        if result is None and vec_b[IDX['heat_emission']] < 0.1:
            result = _thermal_reaction(vec_b, vec_a, env)

    elif action in ('strike', 'bind', 'bundle'):
        result = _mechanical_reaction(vec_a, vec_b, action)

    elif action == 'rub':
        # Friction: thermal if dry enough, otherwise polishing (sharpness up)
        friction = vec_a[IDX['hardness']] * vec_b[IDX['hardness']]
        dryness  = min(vec_a[IDX['dryness']], vec_b[IDX['dryness']])
        ignite_p = friction * dryness * vec_a[IDX['flammable']] * 0.4
        ignite_p *= (1.0 - env.get('moisture', 0.5))
        if ignite_p > 0.15 and random.random() < ignite_p:
            # Friction fire: produce ember-like vector
            result = np.zeros(N_PROPS, dtype=np.float32)
            result[IDX['heat_emission']]  = 0.5 + random.uniform(0, 0.3)
            result[IDX['light_emission']] = 0.3 + random.uniform(0, 0.2)
            result[IDX['dryness']]        = 1.0
            result[IDX['mass']]           = 0.05
        else:
            result = _mechanical_reaction(vec_a, vec_b, 'bundle')

    elif action == 'eat':
        # Metabolic mixing: ingestion of two materials → blend
        if vec_a[IDX['edibility']] > 0.1 and (vec_b is not None and vec_b[IDX['edibility']] > 0.1):
            result = (vec_a + vec_b) * 0.5
            result[IDX['mass']] = (vec_a[IDX['mass']] + vec_b[IDX['mass']]) * 0.3

    if result is None:
        # Fallback: solvent extraction attempt regardless of action
        result = _solvent_reaction(vec_a, vec_b, env)

    if result is not None:
        result = np.clip(result, 0.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Legacy apply_interaction() — kept for backward compatibility
# Now also tries the vector engine; legacy string results still returned
# for known outcomes so old code paths keep working.
# ---------------------------------------------------------------------------
def apply_interaction(action: str, mat_a: str, mat_b: str | None, env: dict) -> list[str]:
    """
    Legacy interface. Returns list of result material name strings.
    For newly invented materials the result is a registered mat_XXXX id.
    """
    results = []
    vec_a = get_vector(mat_a)
    vec_b = get_vector(mat_b) if mat_b else None

    # ---- Legacy deterministic special cases (kept for game-feel) ----
    props_a = get_material(mat_a)
    props_b = get_material(mat_b) if mat_b else {}

    if action == 'strike':
        spark_chance = props_a.get('spark_on_strike', 0.0) + props_b.get('spark_on_strike', 0.0)
        spark_chance *= (1.0 - env.get('moisture', 0.5))
        if random.random() < spark_chance:
            results.append('_spark')
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
        heat = props_b.get('heat_emission', 0.0)
        if heat > 0.5:
            if mat_a == 'raw_meat' and random.random() < heat:
                results.append('cooked_meat')
            elif mat_a == 'raw_root' and random.random() < heat * 0.9:
                results.append('cooked_root')
            elif props_a.get('flammable', 0) > 0.5 and random.random() < heat * props_a['flammable']:
                results.append('fire' if heat > 0.8 else 'ember')

    elif action == 'bundle':
        combined_flamm = max(props_a.get('flammable', 0), props_b.get('flammable', 0))
        combined_dry   = (props_a.get('dryness', 0) + props_b.get('dryness', 0)) / 2
        if combined_flamm > 0.5 and combined_dry > 0.6:
            results.append('_tinder_bundle')

    elif action == 'carry':
        if mat_a == 'ember':
            results.append('ember' if random.random() < 0.55 else 'ash')

    elif action == 'blow':
        if mat_a == 'ember':
            boost = env.get('wind', 0.3) + 0.3
            results.append('fire' if random.random() < boost else 'ember')
        elif mat_a == 'fire':
            results.append('fire')

    elif action == 'eat':
        edibility = props_a.get('edible_cooked', 0) if mat_a.startswith('cooked') else props_a.get('edible_raw', 0)
        if not edibility:
            edibility = props_a.get('edibility', 0)
        if edibility > 0.3:
            results.append('_nutrition:' + str(round(edibility + props_a.get('energy_bonus', 0.0), 2)))

    # ---- Emergent vector engine: runs in ADDITION to legacy cases ----
    new_vec = combine_vectors(vec_a, vec_b, action, env)
    if new_vec is not None:
        # Only register if the vector is meaningfully non-zero
        if float(new_vec.sum()) > 0.1:
            mat_id = DISCOVERY_REGISTRY.register(new_vec, discoverer_id=-1, tick=0,
                                                  recipe=(action, mat_a, mat_b))
            # Avoid duplicate if legacy already produced the same named mat
            named_equiv = _find_named_equivalent(new_vec)
            if named_equiv is None:
                results.append(mat_id)

    return results


def _find_named_equivalent(vec: np.ndarray, threshold: float = 0.10) -> str | None:
    """Return seed material name if vec is very close to it."""
    for name, seed_vec in MATERIALS.items():
        if np.linalg.norm(vec - seed_vec) < threshold:
            return name
    return None


# ---------------------------------------------------------------------------
# Reward from physical consequences (no label lookup)
# ---------------------------------------------------------------------------
def material_reward(vector: np.ndarray, agent_state: dict) -> float:
    """
    Derives reward purely from the physical properties of a newly created
    material and the agent's current homeostatic state.
    No string matching. No hardcoded recipes.
    """
    reward = 0.0
    energy_deficit   = max(0.0, 1.0 - agent_state.get('energy',  1.0))
    hydration_deficit= max(0.0, 1.0 - agent_state.get('hydration', 1.0))
    health_deficit   = max(0.0, 1.0 - agent_state.get('health',  1.0))
    cold             = agent_state.get('cold', False)
    dark             = agent_state.get('dark', False)

    # Hunger → edibility matters
    reward += vector[IDX['edibility']] * energy_deficit * 2.5
    # Toxicity always penalises
    reward -= vector[IDX['toxicity']] * 3.0
    # Warmth in cold conditions
    if cold:
        reward += vector[IDX['heat_emission']] * 1.8
    # Light in darkness
    if dark:
        reward += vector[IDX['light_emission']] * 1.2
    # Sharp tools are intrinsically useful
    reward += vector[IDX['sharpness']] * 0.6
    # Novel scent is mildly rewarding (curiosity / social signal)
    reward += vector[IDX['scent']] * 0.25
    # Very heavy objects without utility are a mild burden
    utility = (vector[IDX['edibility']] + vector[IDX['sharpness']]
               + vector[IDX['heat_emission']] + vector[IDX['light_emission']])
    if utility < 0.1:
        reward -= vector[IDX['mass']] * 0.05

    return float(reward)


# ---------------------------------------------------------------------------
# World-cell accessors (unchanged interface)
# ---------------------------------------------------------------------------
def empty_material_slot():
    return {}


def decay_materials(slot: dict, env: dict) -> dict:
    remove = []
    ash_add = 0.0
    for mat, qty in list(slot.items()):
        props = get_material(mat)
        if props.get('decays'):
            slot[mat] = max(0.0, qty - 0.05 - 0.03 * env.get('wind', 0.3))
            if slot[mat] < 0.05:
                remove.append(mat)
                if mat == 'fire':
                    ash_add += qty * 0.4
                elif mat == 'ember':
                    ash_add += qty * 0.8
        elif props.get('perishable') and env.get('moisture', 0.5) > 0.6:
            slot[mat] = max(0.0, qty - 0.01)
            if slot[mat] < 0.01:
                remove.append(mat)
    for m in remove:
        del slot[m]
    if ash_add > 0.0:
        slot['ash'] = slot.get('ash', 0.0) + ash_add
    return slot


def material_heat(slot: dict) -> float:
    total = 0.0
    for m, q in slot.items():
        vec = get_vector(m)
        total += float(vec[IDX['heat_emission']]) * q
    return total


def material_light(slot: dict) -> float:
    total = 0.0
    for m, q in slot.items():
        vec = get_vector(m)
        total += float(vec[IDX['light_emission']]) * q
    return total


def material_danger(slot: dict) -> float:
    total = 0.0
    for m, q in slot.items():
        vec = get_vector(m)
        total += float(vec[IDX['toxicity']]) * q
    return total
