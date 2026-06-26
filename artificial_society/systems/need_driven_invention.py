"""
Need-Driven Emergent Invention
================================
Agenten erfinden nicht zufaellig beim Explorieren, sondern weil sie
ein konkretes Problem loesen muessen.

Das System funktioniert vollstaendig ohne hardcodierte Rezepte:

  1. NEED-VECTOR
     Aus dem aktuellen Zustand des Agenten (Energie, Gesundheit,
     Temperatur, Krankheit, etc.) wird ein 12-dimensionaler Vektor
     berechnet, der beschreibt welche physikalischen Eigenschaften
     ein Material haben muesste um zu helfen.
     z.B. bei Hunger:  edibility=1.0, heat=0.0
          bei Kaelte:  heat_emission=1.0, edibility=0.0
          bei Krankheit: toxicity=-1.0, edibility=0.3

  2. INVENTORY-SEARCH
     Der Agent durchsucht sein Inventar und die Zelle nach Materialien
     deren Vektoren dem Need-Vector am naechsten liegen.
     Er kombiniert sie mit der Action die am ehesten den Need erfuellt.

  3. TRIAL-AND-ERROR-LERNEN
     Der Reward nach der Kombination wird in CausalMemory gespeichert.
     Ueber Ticks lernt der Agent welche Kombinationen welche Needs loesen.
     Dieses Wissen wird sozial weitergegeben und vererbt.

  4. KEINE HARDCODIERTEN REGELN
     Es gibt keine if-cold-then-make-fire Logik.
     Der Agent lernt das selbst durch Reward-Signal.
"""

import random
from typing import Optional

import numpy as np

from artificial_society.environment.materials import (
    DISCOVERY_REGISTRY,
    IDX,
    N_PROPS,
    PROP_DIMS,
    apply_interaction,
    combine_vectors,
    get_vector,
    material_reward,
)
from artificial_society.systems.invention import (
    NEW_DISCOVERY_BONUS,
    REDISCOVERY_REWARD,
    _agent_homeostatic_state,
    _evaluate_legacy_outcomes,
    _maybe_upgrade_tool,
)

# Wie stark Need-Vektor die Materialauswahl beeinflusst vs. Zufall
NEED_SEARCH_WEIGHT  = 0.70
# Wie viele Kandidaten aus Inventar+Zelle verglichen werden
CANDIDATE_POOL_SIZE = 8
# Minimaler Need-Magnitude damit need-driven Erfindung ausgeloest wird
NEED_THRESHOLD      = 0.25
# Bonus-Reward wenn neu entdecktes Material den Need stark erfuellt
NEED_FULFILLMENT_BONUS = 1.2
# Gewichtung des emergenten Pfads im Gesamtreward.
# Vorher 0.7 — damit wurde echte Emergenz gegenüber dem ungewichteten
# Legacy-Pfad systematisch benachteiligt. Auf 1.0 angehoben (gleichberechtigt),
# passend zum Rebalance in invention.py.
EMERGENT_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Need-Vector Berechnung
# ---------------------------------------------------------------------------

def compute_need_vector(agent, cell: dict) -> np.ndarray:
    """
    Berechnet einen 12-dimensionalen Need-Vektor der beschreibt,
    welche physikalischen Eigenschaften ein Material haben muesste
    um dem Agenten jetzt am meisten zu helfen.

    Der Vektor hat KEINE hardcodierten Rezepte.
    Er beschreibt nur physikalische Eigenschaften, nicht Materialien.

    Positiver Wert = diese Eigenschaft wird benoetigt
    Negativer Wert = diese Eigenschaft ist schaedlich (wird gemieden)
    """
    from artificial_society.agents.agent import MAX_ENERGY

    need = np.zeros(N_PROPS, dtype=np.float32)

    energy_ratio    = agent.energy    / MAX_ENERGY
    health_ratio    = agent.health    / 100.0
    temperature     = cell.get('temperature', 20)
    light           = cell.get('light', 1.0)
    is_sick         = getattr(agent, 'disease_id', None) is not None
    has_tool        = getattr(agent, 'tool', None) is not None

    # Hunger -> edibility benoetigt
    hunger_drive = max(0.0, 1.0 - energy_ratio)
    need[IDX['edibility']] += hunger_drive * 1.5

    # Immer: Toxisches meiden
    need[IDX['toxicity']] -= 2.0

    # Kaelte -> Waerme benoetigt
    cold_drive = max(0.0, (15 - temperature) / 15.0)
    need[IDX['heat_emission']] += cold_drive * 1.8
    # Bei Kaelte: entzuendbare Materialien sind wertvoll
    need[IDX['flammable']] += cold_drive * 0.6

    # Dunkelheit -> Licht benoetigt
    dark_drive = max(0.0, 0.5 - light)
    need[IDX['light_emission']] += dark_drive * 1.2

    # Kein Werkzeug -> Schaerfe benoetigt
    if not has_tool:
        need[IDX['sharpness']] += 0.9
        need[IDX['hardness']]  += 0.5

    # Krank -> Heilende Eigenschaften (niedrige Toxizitaet + hohe Edibility)
    if is_sick:
        need[IDX['edibility']] += 0.6
        need[IDX['toxicity']]  -= 1.0  # extra toxicity-aversion
        need[IDX['scent']]     += 0.3  # aromatische Heilpflanzen

    # Niedrige Gesundheit (verletzt, nicht unbedingt krank)
    health_drive = max(0.0, 0.6 - health_ratio)
    need[IDX['edibility']]  += health_drive * 0.8
    need[IDX['conductivity']] += health_drive * 0.4  # Waermeleitend = lindernd

    # Neugier-Bonus: Agents mit hoher Neugier suchen breitere Eigenschaften
    curiosity = agent.genes.get('curiosity', 0.5)
    if curiosity > 0.7 and hunger_drive < 0.3 and cold_drive < 0.3:
        # Kein unmittelbarer Druck -> erkunde interessante Eigenschaften
        need[IDX['scent']]        += curiosity * 0.4
        need[IDX['conductivity']] += curiosity * 0.3
        need[IDX['light_emission']] += curiosity * 0.2

    # Endocrine-Modulation: Cortisol verstaerkt alle Beduerfte
    if hasattr(agent, 'endocrine'):
        cortisol = getattr(agent.endocrine, 'h', [0]*8)
        stress   = cortisol[1] if len(cortisol) > 1 else 0.0
        need    *= (1.0 + stress * 0.4)

    return need


# ---------------------------------------------------------------------------
# Material-Kandidaten nach Need suchen
# ---------------------------------------------------------------------------

def _need_score(material_vec: np.ndarray, need: np.ndarray) -> float:
    """
    Wie gut erfuellt ein Materialvektor den Need-Vektor?
    Positiver Need + passende Eigenschaft = hoher Score.
    Negativer Need + vorhandene Eigenschaft = Penalty.
    """
    # Nur positive Needs: dot product der positiven Dimensionen
    positive_need = np.maximum(need, 0.0)
    negative_need = np.maximum(-need, 0.0)
    benefit  = float(np.dot(material_vec, positive_need))
    penalty  = float(np.dot(material_vec, negative_need))
    return benefit - penalty


def _select_materials_by_need(
    agent,
    cell: dict,
    need: np.ndarray,
) -> tuple[Optional[str], Optional[str]]:
    """
    Waehlt die zwei Materialien aus die den Need am besten erfuellen.
    Kein Hardcoding welche Materialien gewaehlt werden --
    die Entscheidung basiert rein auf Vektoraehnlichkeit zum Need.
    """
    inv  = getattr(agent, 'material_inventory', {})
    slot = cell.get('materials', {})

    # Alle verfuegbaren Materialien sammeln
    candidates: list[tuple[str, float]] = []

    for mat, qty in slot.items():
        if qty > 0.05:
            vec   = get_vector(mat)
            score = _need_score(vec, need)
            candidates.append((mat, score))

    for mat, qty in inv.items():
        if qty > 0.1 and not mat.startswith('_'):
            vec   = get_vector(mat)
            score = _need_score(vec, need)
            candidates.append((mat, score))

    if not candidates:
        return None, None

    # Sortiere nach Need-Score, waehle mit Temperatur-Sampling
    candidates.sort(key=lambda x: -x[1])
    pool = candidates[:CANDIDATE_POOL_SIZE]

    # Softmax-Sampling: bessere Kandidaten wahrscheinlicher, aber nicht deterministisch
    scores  = np.array([max(0.001, s + 2.0) for _, s in pool], dtype=np.float32)
    probs   = scores / scores.sum()

    if random.random() < NEED_SEARCH_WEIGHT:
        # Need-getriebene Auswahl
        idx_a = np.random.choice(len(pool), p=probs)
        mat_a = pool[idx_a][0]
    else:
        # Zufaellige Exploration
        mat_a = random.choice(pool)[0]

    # mat_b: komplementaer zu mat_a waehlen
    remaining = [m for m, _ in pool if m != mat_a]
    mat_b = random.choice(remaining) if remaining else None

    return mat_a, mat_b


# ---------------------------------------------------------------------------
# Action-Auswahl nach Need
# ---------------------------------------------------------------------------

def _select_action_by_need(
    need: np.ndarray,
    mat_a_vec: np.ndarray,
    mat_b_vec: Optional[np.ndarray],
    causal_mem,
) -> str:
    """
    Waehlt die Action die am wahrscheinlichsten den Need erfuellt.
    Basiert auf physikalischer Logik der Materialvektoren, nicht auf
    hardcodierten Regeln.
    """
    # Action-Affinitaeten: welche Action passt zu welcher Materialkombination?
    # Das ist KEINE Rezepttabelle -- es sind physikalische Heuristiken
    # (Reibung erzeugt Waerme, Binden erzeugt Werkzeuge, etc.)
    action_scores: dict[str, float] = {}

    heat_need   = float(max(need[IDX['heat_emission']], need[IDX['flammable']]))
    sharp_need  = float(need[IDX['sharpness']])
    food_need   = float(need[IDX['edibility']])
    scent_need  = float(need[IDX['scent']])
    light_need  = float(need[IDX['light_emission']])

    mat_b_vec_safe = mat_b_vec if mat_b_vec is not None else np.zeros(N_PROPS, dtype=np.float32)

    # rub: gut wenn beide Materialien hart+trocken sind (Feuer) oder weich (Mischen)
    rub_affinity = (
        float(mat_a_vec[IDX['hardness']])  * float(mat_b_vec_safe[IDX['hardness']]) * heat_need
        + float(mat_a_vec[IDX['scent']])   * scent_need * 0.4
    )
    action_scores['rub'] = rub_affinity

    # strike: gut wenn harte Materialien -> Schaerfe
    strike_affinity = (
        (float(mat_a_vec[IDX['hardness']]) + float(mat_b_vec_safe[IDX['hardness']])) * 0.5
        * sharp_need
    )
    action_scores['strike'] = strike_affinity

    # place_on_heat: gut wenn eines der Materialien essbar + Waermequelle vorhanden
    heat_src = max(float(mat_a_vec[IDX['heat_emission']]), float(mat_b_vec_safe[IDX['heat_emission']]))
    place_affinity = (
        float(mat_a_vec[IDX['edibility']]) * food_need * heat_src
    )
    action_scores['place_on_heat'] = place_affinity

    # bundle: gut wenn beide leicht+brennbar oder beide duftend sind
    bundle_affinity = (
        (float(mat_a_vec[IDX['flammable']]) + float(mat_b_vec_safe[IDX['flammable']])) * heat_need * 0.5
        + (float(mat_a_vec[IDX['scent']]) + float(mat_b_vec_safe[IDX['scent']])) * scent_need * 0.6
    )
    action_scores['bundle'] = bundle_affinity

    # blow: gut wenn Glut/Feuer vorhanden
    ember_a = float(mat_a_vec[IDX['heat_emission']]) > 0.4
    blow_affinity = float(ember_a) * (heat_need + light_need)
    action_scores['blow'] = blow_affinity

    # bind: gut wenn eines flexibel + eines rigid (Werkzeug)
    flex  = float(mat_a_vec[IDX['hardness']]) < 0.25
    rigid = float(mat_b_vec_safe[IDX['hardness']]) > 0.5
    bind_affinity = float(flex and rigid) * sharp_need * 1.2
    action_scores['bind'] = bind_affinity

    # eat: gut wenn essbar
    eat_affinity = float(mat_a_vec[IDX['edibility']]) * food_need
    action_scores['eat'] = eat_affinity

    # carry: fallback
    action_scores['carry'] = 0.1

    # CausalMemory-Boost: bekannte erfolgreiche Sequenzen bevorteilen
    if causal_mem is not None:
        for (act, _ma, _mb), stats in causal_mem.sequences.items():
            if stats.get('successes', 0) > 0 and act in action_scores:
                action_scores[act] += stats['reward'] * 0.3

    # Softmax-Sampling
    actions = list(action_scores.keys())
    raw     = np.array([max(0.001, action_scores[a]) for a in actions], dtype=np.float32)
    probs   = raw / raw.sum()

    return np.random.choice(actions, p=probs)


# ---------------------------------------------------------------------------
# Haupt-API: Need-Driven Invention
# ---------------------------------------------------------------------------

def agent_invent_from_need(
    agent,
    cell: dict,
    env: dict,
    tick: int = 0,
) -> float:
    """
    Kernfunktion des Need-Driven Invention Systems.

    Ablauf:
      1. Berechne Need-Vektor aus aktuellem Zustand
      2. Pruefe ob Need stark genug ist um Erfindung auszuloesen
      3. Suche Materialien die den Need am besten erfuellen koennen
      4. Waehle Action basierend auf Materialvektoren + Need
      5. Kombiniere Materialien physikalisch (kein Rezept-Lookup)
      6. Berechne Reward als: wie gut erfuellt das Ergebnis den Need?
      7. Speichere in CausalMemory fuer spaeteres Lernen

    Returns:
        float: Reward fuer diesen Erfindungsversuch
    """
    # Step 1: Need berechnen
    need = compute_need_vector(agent, cell)
    need_magnitude = float(np.linalg.norm(np.maximum(need, 0.0)))

    # Step 2: Schwellenwert - nur bei echtem Bedarf erfinden
    # (verhindert blindes Zufalls-Experimentieren wenn alles gut ist)
    # Neugierde-Gen kann den Schwellenwert senken
    curiosity    = agent.genes.get('curiosity', 0.5)
    eff_threshold = NEED_THRESHOLD * (1.2 - curiosity * 0.4)
    if need_magnitude < eff_threshold:
        return 0.0

    # Step 3: Materialien nach Need auswaehlen
    mat_a, mat_b = _select_materials_by_need(agent, cell, need)
    if mat_a is None:
        return 0.0

    causal_mem = getattr(agent, 'causal_memory', None)
    vec_a      = get_vector(mat_a)
    vec_b      = get_vector(mat_b) if mat_b else None

    # Step 4: Action basierend auf Materialvektoren + Need
    action = _select_action_by_need(need, vec_a, vec_b, causal_mem)

    # Step 5a: Legacy-Pfad (fuer bekannte Interaktionen wie Feuer schlagen)
    legacy_outcomes = apply_interaction(action, mat_a, mat_b, env)
    legacy_reward   = _evaluate_legacy_outcomes(agent, cell, cell.get('materials', {}), legacy_outcomes, env)

    # Step 5b: Emergent-Pfad (Vektorkombination -> neues Material)
    new_vec        = combine_vectors(vec_a, vec_b, action, env)
    emergent_reward = 0.0

    if new_vec is not None and float(new_vec.sum()) > 0.1:
        agent_state = _agent_homeostatic_state(agent, cell)
        base_reward = material_reward(new_vec, agent_state)

        # Step 6: Need-Fulfillment Score
        # Wie gut erfuellt das neue Material den Need?
        fulfillment = _need_score(new_vec, need)
        fulfillment_normalized = min(1.0, max(0.0, fulfillment / max(0.1, need_magnitude)))

        # Emergent Reward = base reward + Need-Fulfillment Bonus
        emergent_reward = base_reward + fulfillment_normalized * NEED_FULFILLMENT_BONUS

        # Prüfen ob die Kombination wirklich neu ist, BEVOR sie registriert wird.
        existing_ids = (
            set(DISCOVERY_REGISTRY.known_ids())
            if hasattr(DISCOVERY_REGISTRY, 'known_ids') else set()
        )
        mat_id = DISCOVERY_REGISTRY.register(
            new_vec,
            discoverer_id=agent.id,
            tick=tick,
            recipe=(action, mat_a, mat_b),
        )

        # Rebalance: genuine Neu-Entdeckung erhält den vollen Discovery-Bonus
        # (wie im klassischen invention.py-Pfad), Re-Entdeckung mindestens den
        # Legacy-konkurrenzfähigen Re-Discovery-Reward. So ist need-getriebene
        # Emergenz nie schwächer als ein Skript-Rezept.
        if mat_id not in existing_ids:
            emergent_reward += NEW_DISCOVERY_BONUS
        else:
            emergent_reward = max(emergent_reward, REDISCOVERY_REWARD)

        inv = getattr(agent, 'material_inventory', {})
        inv[mat_id] = inv.get(mat_id, 0.0) + 0.5
        agent.material_inventory = inv

        _maybe_upgrade_tool(agent, mat_id, new_vec)

        # Starker Dopamin-Kick wenn Erfindung echten Need loest
        if emergent_reward > 0.5 and hasattr(agent, 'endocrine'):
            agent.endocrine.apply_discovery(min(1.0, emergent_reward * 0.7))

    total_reward = legacy_reward + emergent_reward * EMERGENT_WEIGHT

    # Step 7: In CausalMemory speichern
    if causal_mem is not None:
        causal_mem.record(action, mat_a, mat_b, legacy_outcomes, total_reward)

    return total_reward


# ---------------------------------------------------------------------------
# Need-State fuer HUD (optional)
# ---------------------------------------------------------------------------

def agent_need_summary(agent, cell: dict) -> dict:
    """
    Gibt einen lesbaren Need-Summary fuer das HUD zurueck.
    """
    need = compute_need_vector(agent, cell)
    top_needs = sorted(
        [(PROP_DIMS[i], float(need[i])) for i in range(N_PROPS) if need[i] > 0.2],
        key=lambda x: -x[1]
    )
    return {
        'magnitude': float(np.linalg.norm(np.maximum(need, 0.0))),
        'top_needs': top_needs[:3],
    }
