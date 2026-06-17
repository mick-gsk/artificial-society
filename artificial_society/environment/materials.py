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
#
# ORIGINAL materials: preserved 1:1, legacy string names kept.
# NEW materials (scent/solubility group): enable solvent/perfume emergence.
# ---------------------------------------------------------------------------
MATERIALS: dict[str, np.ndarray] = {
    # ── Originals ──────────────────────────────────────────────────────────
    'dry_grass':    _v(flammable=1.0,  hardness=0.05, edibility=0.2,  mass=0.1,  dryness=1.0),
    'dry_wood':     _v(flammable=0.85, hardness=0.5,  edibility=0.0,  mass=0.6,  dryness=0.9),
    'wet_wood':     _v(flammable=0.15, hardness=0.5,  edibility=0.0,  mass=0.8,  dryness=0.1),
    'stone':        _v(flammable=0.0,  hardness=1.0,  edibility=0.0,  mass=1.0,  dryness=0.5,  conductivity=0.4),
    'flint':        _v(flammable=0.0,  hardness=1.0,  edibility=0.0,  mass=0.8,  dryness=0.9,  sharpness=0.9,   conductivity=0.3),
    'bone':         _v(flammable=0.2,  hardness=0.6,  edibility=0.1,  mass=0.3,  dryness=0.6,  sharpness=0.3),
    'raw_meat':     _v(flammable=0.1,  hardness=0.1,  edibility=0.5,  mass=0.4,  dryness=0.3,  scent=0.4),
    'raw_root':     _v(flammable=0.05, hardness=0.2,  edibility=0.4,  mass=0.2,  dryness=0.3,  solubility=0.2),
    'ember':        _v(flammable=0.0,  hardness=0.0,  edibility=0.0,  mass=0.05, dryness=1.0,  heat_emission=0.6,  light_emission=0.4),
    'fire':         _v(flammable=0.0,  hardness=0.0,  edibility=0.0,  mass=0.0,  dryness=1.0,  heat_emission=1.0,  light_emission=1.0, toxicity=0.1),
    'ash':          _v(flammable=0.0,  hardness=0.0,  edibility=0.0,  mass=0.05, dryness=1.0,  heat_emission=0.05, solubility=0.3),
    'cooked_meat':  _v(flammable=0.0,  hardness=0.2,  edibility=0.95, mass=0.35, dryness=0.7,  scent=0.55),
    'cooked_root':  _v(flammable=0.0,  hardness=0.1,  edibility=0.85, mass=0.15, dryness=0.5,  scent=0.2),
    'sharp_stone':  _v(flammable=0.0,  hardness=0.95, edibility=0.0,  mass=0.5,  dryness=0.8,  sharpness=0.8),
    'fiber':        _v(flammable=0.5,  hardness=0.05, edibility=0.0,  mass=0.1,  dryness=0.5),

    # ── Scent / solubility group (enables perfume / extract emergence) ──────
    # Blütenblätter: hohes scent + solubility → bei Feuchtigkeit extrahierbar
    'flower_petals': _v(flammable=0.2,  hardness=0.02, edibility=0.15, mass=0.05,
                        dryness=0.4,  solubility=0.7,  scent=0.9),

    # Baumharz: klebrig, brennbar, duftend → Klebstoff, Fackel, Parfüm-Basis
    'tree_resin':    _v(flammable=0.6,  hardness=0.15, edibility=0.0,  mass=0.3,
                        dryness=0.6,  solubility=0.45, scent=0.7,  conductivity=0.2),

    # Zerriebene Kräuter: löslich, duftend, leicht medizinisch
    'crushed_herb':  _v(flammable=0.1,  hardness=0.05, edibility=0.3,  toxicity=0.1,
                        mass=0.08, dryness=0.4,  solubility=0.55, scent=0.6),

    # ── Keramik / Werkzeug-Gruppe ────────────────────────────────────────────
    # Ton: formbar, löslich, wird durch Hitze zu hartem Material (Keramik)
    'clay':          _v(flammable=0.0,  hardness=0.35, edibility=0.0,  mass=0.9,
                        dryness=0.1,  solubility=0.6,  conductivity=0.3),

    # Holzkohle: entsteht aus Holz+Feuer; leitfähig, löslich → Tinte, Farbe
    'charcoal':      _v(flammable=0.3,  hardness=0.2,  edibility=0.0,  mass=0.15,
                        dryness=1.0,  solubility=0.5,  conductivity=0.6,  light_emission=0.05),

    # Tierfett: brennbar, essbar, duftend → Kerzen, Seife, Schmierung
    'animal_fat':    _v(flammable=0.7,  hardness=0.02, edibility=0.3,  mass=0.25,
                        dryness=0.5,  solubility=0.4,  scent=0.3),
}

# Legacy compatibility: boolean/special flags per material
_LEGACY_FLAGS: dict[str, dict] = {
    'raw_meat':      {'perishable': True},
    'raw_root':      {'perishable': True},
    'ember':         {'decays': True},
    'fire':          {'decays': True, 'dangerous': True},
    'stone':         {'spark_on_strike': 0.35},
    'flint':         {'spark_on_strike': 0.75, 'sharp_when_knapped': 0.9},
    'cooked_meat':   {'energy_bonus': 0.4},
    'cooked_root':   {'energy_bonus': 0.25},
    'flower_petals': {'perishable': True},
    'crushed_herb':  {'perishable': True},
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
            'id':            new_id,
            'vector':        vector.copy(),
            'discovered_by': discoverer_id,
            'tick':          tick,
            'recipe':        recipe,
            'uses':          0,
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
# ---------------------------------------------------------------------------
def _thermal_reaction(
    vec_a: np.ndarray, vec_b: np.ndarray, env: dict
) -> np.ndarray | None:
    """Heat from b transforms a (cooking, burning, drying, distillation)."""
    heat = vec_b[IDX['heat_emission']]
    if heat < 0.35:
        return None
    moisture_damp  = 1.0 - env.get('moisture', 0.5) * 0.6
    effective_heat = heat * moisture_damp
    result = vec_a.copy()
    # Cooking / Maillard
    if vec_a[IDX['edibility']] > 0.1:
        result[IDX['edibility']] = min(1.0, vec_a[IDX['edibility']] * (1.0 + 0.8 * effective_heat))
        result[IDX['toxicity']]  = max(0.0, vec_a[IDX['toxicity']]  * (1.0 - 0.7 * effective_heat))
        result[IDX['scent']]     = min(1.0, vec_a[IDX['scent']] + 0.25 * effective_heat)
        result[IDX['mass']]     *= max(0.7, 1.0 - 0.15 * effective_heat)
    # Combustion
    if vec_a[IDX['flammable']] > 0.5 and effective_heat > 0.6:
        result[IDX['flammable']]      = max(0.0, vec_a[IDX['flammable']] - 0.5)
        result[IDX['heat_emission']]  = min(1.0, vec_a[IDX['heat_emission']] + 0.4 * effective_heat)
        result[IDX['light_emission']] = min(1.0, vec_a[IDX['light_emission']] + 0.3)
        result[IDX['mass']]          *= 0.6
        result[IDX['dryness']]        = 1.0
    # Clay → ceramic: hardness spikes under heat
    if vec_a[IDX['solubility']] > 0.5 and vec_a[IDX['hardness']] < 0.5 and effective_heat > 0.7:
        result[IDX['hardness']]   = min(1.0, vec_a[IDX['hardness']] + 0.55 * effective_heat)
        result[IDX['solubility']] = max(0.0, vec_a[IDX['solubility']] - 0.5)
        result[IDX['dryness']]    = 1.0
    # Scent distillation: volatile scent concentrates under mild heat
    if vec_a[IDX['scent']] > 0.3 and effective_heat < 0.7:
        result[IDX['scent']] = min(1.0, vec_a[IDX['scent']] * (1.0 + 0.3 * effective_heat))
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
            result[IDX['mass']]     *= 0.75
            return result
        # Hard on soft: pulverising → solubility increases (crushed herb / powder)
        if vec_a[IDX['hardness']] < 0.3 and vec_b[IDX['hardness']] > 0.7:
            result = vec_a.copy()
            result[IDX['solubility']] = min(1.0, vec_a[IDX['solubility']] + 0.4)
            result[IDX['scent']]      = min(1.0, vec_a[IDX['scent']] + 0.2)  # crushing releases aroma
            result[IDX['mass']]      *= 0.8
            result[IDX['hardness']]  *= 0.3
            return result

    elif action == 'bind':
        flex  = vec_a[IDX['hardness']] < 0.25
        rigid = vec_b[IDX['hardness']] > 0.5
        if flex and rigid:
            result = np.zeros(N_PROPS, dtype=np.float32)
            result[IDX['hardness']]  = vec_b[IDX['hardness']] * 0.8
            result[IDX['mass']]      = vec_a[IDX['mass']] + vec_b[IDX['mass']]
            result[IDX['sharpness']] = vec_b[IDX['sharpness']]
            result[IDX['flammable']] = vec_a[IDX['flammable']] * 0.5
            # Resin as binder: if vec_a is sticky (high solubility + scent), bond is stronger
            if vec_a[IDX['scent']] > 0.4 and vec_a[IDX['solubility']] > 0.3:
                result[IDX['hardness']] = min(1.0, result[IDX['hardness']] + 0.1)
                result[IDX['scent']]    = vec_a[IDX['scent']] * 0.4  # residual aroma
            return result

    elif action == 'bundle':
        combined_flamm = (vec_a[IDX['flammable']] + vec_b[IDX['flammable']]) * 0.6
        combined_dry   = (vec_a[IDX['dryness']]   + vec_b[IDX['dryness']])   * 0.5
        combined_scent = (vec_a[IDX['scent']]      + vec_b[IDX['scent']])     * 0.7
        if combined_flamm > 0.4 or combined_dry > 0.55 or combined_scent > 0.5:
            result = (vec_a + vec_b) * 0.5
            result[IDX['flammable']] = combined_flamm
            result[IDX['dryness']]   = combined_dry
            result[IDX['scent']]     = min(1.0, combined_scent)  # scent-bundle (incense analog)
            result[IDX['mass']]      = vec_a[IDX['mass']] + vec_b[IDX['mass']]
            return result

    return None


def _solvent_reaction(
    vec_a: np.ndarray, vec_b: np.ndarray, env: dict
) -> np.ndarray | None:
    """
    Dissolving/mixing: solubility drives extraction of volatile compounds.
    vec_a = soluble substance (flower_petals, crushed_herb, resin, charcoal ...)
    vec_b = carrier / solvent (water-rich material or zero-vector = ambient water)
    env['moisture'] must be >= 0.4 OR vec_b already has mass (liquid carrier).
    """
    moisture     = env.get('moisture', 0.0)
    solubility_a = vec_a[IDX['solubility']]
    # Accept if moisture in env is high OR carrier has solubility itself
    if solubility_a < 0.25:
        return None
    if moisture < 0.4 and vec_b[IDX['solubility']] < 0.3:
        return None

    effective_moisture = max(moisture, vec_b[IDX['solubility']])
    result = vec_b.copy()
    # Volatile scent compounds concentrate in the carrier
    result[IDX['scent']]      = min(1.0, vec_b[IDX['scent']]
                                    + vec_a[IDX['scent']] * solubility_a * 1.6 * effective_moisture)
    result[IDX['edibility']]  = min(1.0, (vec_a[IDX['edibility']] + vec_b[IDX['edibility']]) * 0.55)
    result[IDX['toxicity']]   = max(vec_a[IDX['toxicity']], vec_b[IDX['toxicity']])
    result[IDX['solubility']] = (vec_a[IDX['solubility']] + vec_b[IDX['solubility']]) * 0.5
    result[IDX['mass']]       = (vec_a[IDX['mass']] + vec_b[IDX['mass']]) * 0.7
    # Charcoal as carrier: conductivity of result increases (ink / dye)
    if vec_b[IDX['conductivity']] > 0.4 or vec_a[IDX['conductivity']] > 0.4:
        result[IDX['conductivity']] = min(1.0, max(vec_a[IDX['conductivity']],
                                                    vec_b[IDX['conductivity']]) + 0.15)
        result[IDX['light_emission']] = min(0.3, result[IDX['scent']] * 0.2)  # faint glow (dye)
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
        # Also try mechanical with swapped order for bind
        if result is None and action == 'bind':
            result = _mechanical_reaction(vec_b, vec_a, action)

    elif action == 'rub':
        friction = vec_a[IDX['hardness']] * vec_b[IDX['hardness']]
        dryness  = min(vec_a[IDX['dryness']], vec_b[IDX['dryness']])
        ignite_p = friction * dryness * vec_a[IDX['flammable']] * 0.4
        ignite_p *= (1.0 - env.get('moisture', 0.5))
        if ignite_p > 0.15 and random.random() < ignite_p:
            result = np.zeros(N_PROPS, dtype=np.float32)
            result[IDX['heat_emission']]  = 0.5 + random.uniform(0, 0.3)
            result[IDX['light_emission']] = 0.3 + random.uniform(0, 0.2)
            result[IDX['dryness']]        = 1.0
            result[IDX['mass']]           = 0.05
        else:
            result = _mechanical_reaction(vec_a, vec_b, 'bundle')
            # Rubbing releases scent even without fire (spice grinding)
            if result is None and (vec_a[IDX['scent']] > 0.3 or vec_b[IDX['scent']] > 0.3):
                result = vec_a.copy()
                result[IDX['scent']]      = min(1.0, vec_a[IDX['scent']] + 0.15)
                result[IDX['solubility']] = min(1.0, vec_a[IDX['solubility']] + 0.1)
                result[IDX['hardness']]  *= 0.5
                result[IDX['mass']]      *= 0.85

    elif action == 'eat':
        if vec_a[IDX['edibility']] > 0.1 and vec_b[IDX['edibility']] > 0.1:
            result = (vec_a + vec_b) * 0.5
            result[IDX['mass']] = (vec_a[IDX['mass']] + vec_b[IDX['mass']]) * 0.3

    # Fallback: solvent extraction
    if result is None:
        result = _solvent_reaction(vec_a, vec_b, env)
    # Also try with swapped args (vec_b might be the soluble one)
    if result is None:
        result = _solvent_reaction(vec_b, vec_a, env)

    if result is not None:
        result = np.clip(result, 0.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Legacy apply_interaction() — backward compatible
# ---------------------------------------------------------------------------
def apply_interaction(action: str, mat_a: str, mat_b: str | None, env: dict) -> list[str]:
    results = []
    vec_a = get_vector(mat_a)
    vec_b = get_vector(mat_b) if mat_b else None

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

    # Emergent vector engine
    new_vec = combine_vectors(vec_a, vec_b, action, env)
    if new_vec is not None and float(new_vec.sum()) > 0.1:
        mat_id = DISCOVERY_REGISTRY.register(new_vec, discoverer_id=-1, tick=0,
                                              recipe=(action, mat_a, mat_b))
        named_equiv = _find_named_equivalent(new_vec)
        if named_equiv is None:
            results.append(mat_id)

    return results


def _find_named_equivalent(vec: np.ndarray, threshold: float = 0.10) -> str | None:
    for name, seed_vec in MATERIALS.items():
        if np.linalg.norm(vec - seed_vec) < threshold:
            return name
    return None


# ---------------------------------------------------------------------------
# Reward from physical consequences
# ---------------------------------------------------------------------------
def material_reward(vector: np.ndarray, agent_state: dict) -> float:
    reward = 0.0
    energy_deficit    = max(0.0, 1.0 - agent_state.get('energy',    1.0))
    hydration_deficit = max(0.0, 1.0 - agent_state.get('hydration', 1.0))
    health_deficit    = max(0.0, 1.0 - agent_state.get('health',    1.0))
    cold = agent_state.get('cold', False)
    dark = agent_state.get('dark', False)

    reward += vector[IDX['edibility']] * energy_deficit * 2.5
    reward -= vector[IDX['toxicity']] * 3.0
    if cold:
        reward += vector[IDX['heat_emission']] * 1.8
    if dark:
        reward += vector[IDX['light_emission']] * 1.2
    reward += vector[IDX['sharpness']] * 0.6
    reward += vector[IDX['scent']] * 0.25
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
    remove  = []
    ash_add = 0.0
    for mat, qty in list(slot.items()):
        props = get_material(mat)
        if props.get('decays'):
            slot[mat] = max(0.0, qty - 0.05 - 0.03 * env.get('wind', 0.3))
            if slot[mat] < 0.05:
                remove.append(mat)
                if mat == 'fire':  ash_add += qty * 0.4
                elif mat == 'ember': ash_add += qty * 0.8
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
    return sum(float(get_vector(m)[IDX['heat_emission']]) * q for m, q in slot.items())


def material_light(slot: dict) -> float:
    return sum(float(get_vector(m)[IDX['light_emission']]) * q for m, q in slot.items())


def material_danger(slot: dict) -> float:
    return sum(float(get_vector(m)[IDX['toxicity']]) * q for m, q in slot.items())
