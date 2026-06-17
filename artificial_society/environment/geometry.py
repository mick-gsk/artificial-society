"""
Geometry & Form System
-----------------------
Materialien haben keine Form — bis ein Agent ihnen eine gibt.

Eine Form ist eine räumliche Konfiguration von Material-Vektoren.
Sie erzeugt emergente Eigenschaften, die kein Einzelmaterial hat:
  - hollow  → kann andere Materialien halten (Behälter)
  - flat    → bietet Oberfläche (Schlafunterlage, Tisch, Dach)
  - sealed  → schützt Inhalt vor Umwelt (Feuchtigkeit, Kälte)
  - layered → Isolation (Wärme, Schall)
  - pointed → Angriffs/Jagd-Bonus
  - woven   → Flexibilität + Struktur (Korb, Netz, Kleidung)

Kein Hardcode der Endprodukte. Die Form-Engine kombiniert:
  Materialvektor + Shape-Action + Werkzeug → CompositeObject
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from artificial_society.environment.materials import (
    IDX, N_PROPS, PROP_DIMS, get_vector, DISCOVERY_REGISTRY
)

# ---------------------------------------------------------------------------
# Shape vocabulary
# ---------------------------------------------------------------------------
SHAPES = {
    # name         → (hollow, flat, sealed, layered, pointed, woven)
    'amorphous':   (False, False, False, False, False, False),  # Rohzustand
    'flat':        (False, True,  False, False, False, False),  # Schicht/Brett
    'hollow':      (True,  False, False, False, False, False),  # Schale/Schüssel
    'sealed':      (True,  False, True,  False, False, False),  # Behälter/Krug
    'layered':     (False, True,  False, True,  False, False),  # Stapel/Matte
    'pointed':     (False, False, False, False, True,  False),  # Spitze/Pfeil
    'woven':       (False, True,  False, True,  False, True),   # Korb/Netz/Kleid
    'dome':        (True,  False, True,  True,  False, False),  # Hütte/Ofen
}

SHAPE_FLAGS = ('hollow', 'flat', 'sealed', 'layered', 'pointed', 'woven')

# Welche Actions erzeugen welche Shape-Änderung
SHAPE_ACTIONS = {
    'form':    'hollow',    # Ton/Holz mit Händen formen → Schüssel
    'flatten': 'flat',      # Schlagen/Rollen → Brett/Schicht
    'coil':    'sealed',    # Spiralförmig aufbauen (Töpferei) → Krug
    'weave':   'woven',     # Fasern verflechten → Korb/Netz
    'stack':   'layered',   # Übereinanderlegen → Isolation/Matratze
    'point':   'pointed',   # Zuspitzen → Speerspitze
    'arch':    'dome',      # Wölben → Kuppel/Ofen
}

# Mindest-Materialanforderungen für Shape-Actions
# (prop_name, min_value)
SHAPE_REQUIREMENTS: dict[str, list[tuple]] = {
    'form':    [('hardness', 0.1), ('hardness', 0.6, 'MAX')],  # nicht zu hart
    'flatten': [],                                              # immer möglich
    'coil':    [('hardness', 0.1), ('hardness', 0.5, 'MAX')],
    'weave':   [('hardness', 0.0), ('hardness', 0.2, 'MAX')],  # braucht Flexibilität
    'stack':   [],
    'point':   [('sharpness', 0.1)],                           # Werkzeug nötig
    'arch':    [('hardness', 0.3)],
}


# ---------------------------------------------------------------------------
# CompositeObject — das Ergebnis einer Form-Action
# ---------------------------------------------------------------------------
@dataclass
class CompositeObject:
    """
    Ein geformtes Objekt aus einem oder mehreren Materialvektoren.
    Besitzt emergente Eigenschaften basierend auf Form + Materialien.
    """
    obj_id:     str
    shape:      str
    components: list[str]          # Material-Namen oder mat_XXXX IDs
    vectors:    list[np.ndarray]   # korrespondierende Vektoren
    creator_id: int = -1
    tick_made:  int = 0

    # Emergente Properties (berechnet)
    capacity:     float = 0.0   # Liter-Analogon: wie viel kann es halten
    insulation:   float = 0.0   # Wärme-/Kälteschutz
    comfort:      float = 0.0   # Schlaf/Ruhe-Bonus
    attack_bonus: float = 0.0   # Jagd/Kampf-Bonus
    shelter:      float = 0.0   # Schutz vor Wetter
    portability:  float = 0.0   # Wie leicht tragbar
    durability:   float = 0.0   # Haltbarkeit

    # Inventar wenn hollow/sealed
    contents: dict = field(default_factory=dict)

    def flags(self) -> dict:
        shape_flags = SHAPES.get(self.shape, SHAPES['amorphous'])
        return dict(zip(SHAPE_FLAGS, shape_flags))

    def can_hold(self) -> bool:
        return self.flags().get('hollow', False) or self.flags().get('sealed', False)

    def is_shelter(self) -> bool:
        return self.shelter > 0.3

    def summary(self) -> dict:
        return {
            'id':         self.obj_id,
            'shape':      self.shape,
            'components': self.components,
            'capacity':   round(self.capacity, 2),
            'insulation': round(self.insulation, 2),
            'comfort':    round(self.comfort, 2),
            'attack':     round(self.attack_bonus, 2),
            'shelter':    round(self.shelter, 2),
            'durability': round(self.durability, 2),
        }


# ---------------------------------------------------------------------------
# Object Registry
# ---------------------------------------------------------------------------
class ObjectRegistry:
    """Speichert alle geformten Objekte der Welt."""
    def __init__(self):
        self.objects: dict[str, CompositeObject] = {}
        self._counter = 0

    def register(self, obj: CompositeObject) -> str:
        obj_id = f'obj_{self._counter:04d}'
        obj.obj_id = obj_id
        self.objects[obj_id] = obj
        self._counter += 1
        print(f'[OBJECT] {obj_id} shape={obj.shape} '
              f'cap={obj.capacity:.2f} ins={obj.insulation:.2f} '
              f'comfort={obj.comfort:.2f} shelter={obj.shelter:.2f} '
              f'by=agent_{obj.creator_id} tick={obj.tick_made}')
        return obj_id

    def get(self, obj_id: str) -> Optional[CompositeObject]:
        return self.objects.get(obj_id)

    def all_shelters(self) -> list[CompositeObject]:
        return [o for o in self.objects.values() if o.is_shelter()]


OBJECT_REGISTRY = ObjectRegistry()


# ---------------------------------------------------------------------------
# Emergente Eigenschaftsberechnung
# ---------------------------------------------------------------------------
def _compute_emergent_properties(shape: str, vectors: list[np.ndarray]) -> dict:
    """
    Berechnet emergente Eigenschaften rein aus Shape + Materialvektoren.
    Keine Hardcodes für 'Bett' oder 'Schüssel' — nur physikalische Logik.
    """
    flags = dict(zip(SHAPE_FLAGS, SHAPES.get(shape, SHAPES['amorphous'])))

    # Aggregate material properties
    avg_vec  = np.mean(vectors, axis=0) if vectors else np.zeros(N_PROPS)
    sum_mass = float(sum(v[IDX['mass']] for v in vectors))
    n        = max(1, len(vectors))

    hardness     = float(avg_vec[IDX['hardness']])
    softness     = 1.0 - hardness
    mass         = float(avg_vec[IDX['mass']])
    flammable    = float(avg_vec[IDX['flammable']])
    conductivity = float(avg_vec[IDX['conductivity']])
    dryness      = float(avg_vec[IDX['dryness']])

    props = {}

    # CAPACITY: hollow/sealed + niedrige hardness → größerer Hohlraum
    if flags['hollow'] or flags['sealed']:
        props['capacity'] = softness * 0.4 + hardness * 0.6  # hart = stabiler Behälter
        props['capacity'] *= (1.0 + 0.3 * n)  # mehr Material = größer

    # INSULATION: layered/woven + viele Schichten + niedrige Leitfähigkeit
    if flags['layered'] or flags['woven']:
        props['insulation'] = (1.0 - conductivity) * 0.7 + softness * 0.3
        props['insulation'] *= min(1.0, 0.3 * n)  # mehr Schichten = besser

    # COMFORT: flat/layered + weich + nicht zu schwer
    if flags['flat'] or flags['layered']:
        props['comfort'] = softness * 0.6 + (1.0 - min(1.0, mass)) * 0.4
        if flags['woven']:
            props['comfort'] = min(1.0, props.get('comfort', 0) + 0.2)  # gewebte Matte

    # ATTACK: pointed + hart + scharf
    if flags['pointed']:
        sharpness = float(avg_vec[IDX['sharpness']])
        props['attack_bonus'] = sharpness * 0.5 + hardness * 0.5

    # SHELTER: dome + schwer + hart = strukturelle Integrität
    if shape == 'dome':
        props['shelter'] = hardness * 0.5 + (1.0 - flammable) * 0.3 + min(1.0, mass) * 0.2

    # PORTABILITY: umgekehrt proportional zur Masse
    props['portability'] = max(0.0, 1.0 - sum_mass)

    # DURABILITY: Härte + Trockenheit (nasses Material fault)
    props['durability'] = hardness * 0.6 + dryness * 0.4

    return props


# ---------------------------------------------------------------------------
# Shape-Action Engine
# ---------------------------------------------------------------------------
def can_perform_shape_action(action: str, vec: np.ndarray, tool_vec: Optional[np.ndarray]) -> bool:
    """Prüft ob Material für diese Form-Action geeignet ist."""
    reqs = SHAPE_REQUIREMENTS.get(action, [])
    for req in reqs:
        prop, min_val = req[0], req[1]
        is_max = len(req) > 2 and req[2] == 'MAX'
        val = float(vec[IDX[prop]])
        if is_max and val > min_val:
            return False
        if not is_max and val < min_val:
            return False
    if action == 'point' and tool_vec is not None:
        if float(tool_vec[IDX['sharpness']]) < 0.3:
            return False
    return True


def apply_shape_action(
    action: str,
    mat_name: str,
    vec: np.ndarray,
    tool_vec: Optional[np.ndarray],
    additional_mats: list[tuple[str, np.ndarray]],
    agent_id: int,
    tick: int,
) -> Optional[CompositeObject]:
    """
    Kernfunktion: Transformiert Material + Action → CompositeObject.
    Gibt None zurück wenn Action nicht möglich.
    """
    if not can_perform_shape_action(action, vec, tool_vec):
        return None

    target_shape = SHAPE_ACTIONS.get(action)
    if target_shape is None:
        return None

    all_components = [(mat_name, vec)] + additional_mats
    all_vecs       = [v for _, v in all_components]
    all_names      = [n for n, _ in all_components]

    emergent = _compute_emergent_properties(target_shape, all_vecs)

    obj = CompositeObject(
        obj_id     = '',  # assigned by registry
        shape      = target_shape,
        components = all_names,
        vectors    = all_vecs,
        creator_id = agent_id,
        tick_made  = tick,
        capacity     = emergent.get('capacity',     0.0),
        insulation   = emergent.get('insulation',   0.0),
        comfort      = emergent.get('comfort',       0.0),
        attack_bonus = emergent.get('attack_bonus',  0.0),
        shelter      = emergent.get('shelter',       0.0),
        portability  = emergent.get('portability',   0.0),
        durability   = emergent.get('durability',    0.0),
    )

    return obj


def object_reward(obj: CompositeObject, agent_state: dict) -> float:
    """
    Reward für ein geformtes Objekt — rein aus emergenten Eigenschaften.
    """
    reward = 0.0
    cold  = agent_state.get('cold',  False)
    dark  = agent_state.get('dark',  False)
    tired = agent_state.get('tired', False)
    hungry_for_storage = agent_state.get('energy', 1.0) > 0.7  # satt → Lager wertvoll

    if cold:
        reward += obj.insulation * 1.5 + obj.shelter * 1.2
    if tired:
        reward += obj.comfort * 2.0
    if hungry_for_storage:
        reward += obj.capacity * 0.8  # Behälter für später
    reward += obj.attack_bonus * 0.7
    reward -= (1.0 - obj.durability) * 0.1  # brüchige Dinge weniger wertvoll
    return float(reward)
