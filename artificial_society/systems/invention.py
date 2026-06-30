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

EMERGENZ-ERWEITERUNGEN (v2):
  - Rekursive Kombinierbarkeit: Entdeckte mat_XXXX aus dem Inventory werden
    als gleichberechtigte Inputs für neue Kombinationen genutzt. Agenten
    können also aus bereits verarbeiteten Materialien weitere neue Dinge
    erfinden (Parfüm-Prinzip: erst Öl, dann Öl+Blüten → Parfüm).
  - Bedürfnisgetriebene Aktionswahl: Je nach aktuellem Zustand des Agenten
    (Hunger, Kälte, Krankheit, Neugier) werden passende Actions priorisiert.
    Erfindungen entstehen aus Not, nicht aus Zufall.

FIX (v3):
  - share_discovery: Trust-Gate von 0.25 auf 0.05 gesenkt; Stammesmitglieder
    können immer teilen (tribe_id-Check). Verhindert, dass Wissenstransfer
    in den ersten hunderten Ticks komplett blockiert wird.
  - agent_try_invention: Echte Neu-Entdeckungen erhalten +3.0 Discovery-Bonus.
    Re-Entdeckungen erhalten kleinen Reinforcement-Reward (+0.1) statt 0,
    damit Exploration stabil verstärkt wird.
  - emergent_reward-Gewichtung von 0.6 auf 1.0 erhöht (war zu schwach gegen
    Homeostasis-Rauschen).
"""

import random

import numpy as np

from artificial_society.environment.materials import (
    DISCOVERY_REGISTRY,
    IDX,
    apply_interaction,
    combine_vectors,
    decay_materials,
    get_vector,
    material_danger,
    material_heat,
    material_light,
    material_reward,
)

PRIMITIVE_ACTIONS = ["rub", "strike", "place_on_heat", "bundle", "blow", "carry", "eat", "bind"]

# Bedürfnis-basierte Action-Cluster
ACTIONS_FOR_HUNGER = ["eat", "place_on_heat", "bundle"]
ACTIONS_FOR_COLD = ["rub", "strike", "blow", "carry"]
ACTIONS_FOR_SICK = ["bundle", "bind", "eat"]
ACTIONS_FOR_CURIOUS = PRIMITIVE_ACTIONS  # Kein Druck → maximale Exploration

# ---------------------------------------------------------------------------
# REWARD REBALANCE (Emergenz > Skript)
# ---------------------------------------------------------------------------
# Projektziel: Emergenz aus lokalen Regeln, NICHT geskriptete Outcomes.
# Die 5 Legacy-Rezepte (sharp_stone, cooked_meat, fire/warmth, light, tinder)
# bleiben erhalten — sie bootstrappen das frühe Überleben — werden aber so
# gedämpft, dass sie echte Vektor-Entdeckungen nicht mehr übertönen.
#
# Vorher gaben Legacy-Outcomes pro Versuch verlässlich 0.35–0.55, während eine
# Re-Entdeckung nur 0.1 brachte. Dadurch lernten Agenten, ausschließlich die
# Skript-Rezepte zu wiederholen. Jetzt gilt: Neu-Entdeckung >> Legacy, und eine
# Re-Entdeckung ist mindestens so wertvoll wie das stärkste Legacy-Outcome.
WARMTH_REWARD = 0.06  # war 0.08 — Wärme weiter nützlich, aber kein Selbstläufer
LIGHT_REWARD = 0.05  # war 0.06
COOK_REWARD = 0.22  # war 0.35 — kochen lohnt sich, dominiert aber nicht
SHARP_REWARD = 0.28  # war 0.45 — Werkzeug schärfen bleibt attraktiv
DANGER_PENALTY = -0.12

# Obergrenze für das stärkste verlässliche Legacy-Outcome (fire ≈ 0.55 vorher).
# Dient als Referenz, gegen die emergente Rewards kalibriert werden.
LEGACY_MAX_REWARD = max(COOK_REWARD, SHARP_REWARD, 0.40)  # ~0.40 (fire gedämpft, s.u.)

# Bonus für echte Neu-Entdeckung (noch nie registriert).
# Deutlich über jedem Legacy-Outcome: eine genuin neue Erfindung ist das
# wertvollste Ereignis im System.
NEW_DISCOVERY_BONUS = 4.5  # war 3.0
# Reward für Re-Entdeckung (Wiederholen einer bekannten emergenten Kombination).
# Mindestens so hoch wie das stärkste Legacy-Outcome, damit das Wiederholen einer
# selbst-entdeckten Vektor-Kombination nie schlechter ist als ein Skript-Rezept.
REDISCOVERY_REWARD = LEGACY_MAX_REWARD * 1.2  # war 0.1  → ~0.54

# Maximale Anzahl entdeckter Materialien die als rekursive Inputs berücksichtigt werden
MAX_RECURSIVE_INPUTS = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def agent_try_invention(agent, world, x, y) -> float:
    cell = world.get_cell(x, y)
    env = cell
    slot = cell.get("materials", {})
    inv = getattr(agent, "material_inventory", {})

    # --- SCHICHT 1: Rekursive Kombinierbarkeit ---
    # Rohstoffe aus der Zelle
    cell_available = [m for m, q in slot.items() if q > 0.05]
    # Bereits entdeckte mat_XXXX aus dem eigenen Inventory als gleichberechtigte Inputs
    discovered_inv = [m for m, q in inv.items() if q > 0.1 and m.startswith("mat_")]
    # Bekannte Nicht-mat Materialien aus Inventory (z.B. sharp_stone, cooked_meat)
    known_inv = [m for m, q in inv.items() if q > 0.1 and not m.startswith("_")]

    # Kombiniere alle Quellen; entdeckte Materialien werden priorisiert
    available = list(set(cell_available + known_inv))
    if discovered_inv:
        # Mindestens ein entdecktes Material als potenzieller Input einmischen
        recursive_sample = random.sample(
            discovered_inv, min(MAX_RECURSIVE_INPUTS, len(discovered_inv))
        )
        available = list(set(available + recursive_sample))

    if not available:
        return 0.0

    mat_a = random.choice(available)
    mat_b = None
    if len(available) > 1:
        mat_b = random.choice([m for m in available if m != mat_a])

    causal_mem = getattr(agent, "causal_memory", None)

    # --- SCHICHT 2: Bedürfnisgetriebene Aktionswahl ---
    action = _choose_action_by_need(agent, causal_mem, mat_a, mat_b, cell)

    legacy_outcomes = apply_interaction(action, mat_a, mat_b, env, discoverer_id=agent.id, tick=getattr(agent, "age", 0))
    legacy_reward = _evaluate_legacy_outcomes(agent, world, x, y, slot, legacy_outcomes, env)

    vec_a = get_vector(mat_a)
    vec_b = get_vector(mat_b) if mat_b else None
    new_vec = combine_vectors(vec_a, vec_b, action, env)
    emergent_reward = 0.0
    if new_vec is not None and float(new_vec.sum()) > 0.1:
        agent_state = _agent_homeostatic_state(agent, cell)
        emergent_reward = material_reward(new_vec, agent_state)

        # Prüfe ob die resultierende Kombination bereits bekannt ist (vor der Registrierung)
        existing_ids = (
            set(DISCOVERY_REGISTRY.known_ids())
            if hasattr(DISCOVERY_REGISTRY, "known_ids")
            else set()
        )

        mat_id = DISCOVERY_REGISTRY.register(
            new_vec,
            discoverer_id=agent.id,
            tick=getattr(agent, "age", 0),
            recipe=(action, mat_a, mat_b),
        )
        inv = getattr(agent, "material_inventory", {})
        # FIX: Echter Neu-Entdeckungs-Bonus vs. kleiner Re-Entdeckungs-Reward
        is_new_discovery = mat_id not in existing_ids
        if is_new_discovery:
            emergent_reward += NEW_DISCOVERY_BONUS
        else:
            emergent_reward = max(emergent_reward, REDISCOVERY_REWARD)

        inv[mat_id] = inv.get(mat_id, 0.0) + 0.5
        agent.material_inventory = inv
        _maybe_upgrade_tool(agent, mat_id, new_vec)
        if emergent_reward > 0.3 and hasattr(agent, "endocrine"):
            agent.endocrine.apply_discovery(min(1.0, emergent_reward))

    # FIX: emergent_reward-Gewichtung von 0.6 auf 1.0 erhöht
    total_reward = legacy_reward + emergent_reward * 1.0
    if causal_mem is not None:
        causal_mem.record(action, mat_a, mat_b, legacy_outcomes, total_reward)
    return total_reward


def agent_try_cook(agent, world, x, y) -> float:
    cell = world.get_cell(x, y)
    slot = cell.get("materials", {})
    heat_sources = [m for m in slot if slot[m] > 0.1 and m in ("fire", "ember")]
    if not heat_sources:
        return 0.0

    inv = getattr(agent, "material_inventory", {})
    cookable = [m for m in inv if m in ("raw_meat", "raw_root") and inv[m] > 0.1]
    if cookable:
        source = inv
    else:
        cookable = [m for m in slot if m in ("raw_meat", "raw_root") and slot[m] > 0.1]
        source = slot
    if not cookable:
        return 0.0

    heat_mat = heat_sources[0]
    food_mat = cookable[0]
    env = {
        "wind": cell.get("disturbance", 0) / 100.0,
        "moisture": cell.get("moisture", 50) / 100.0,
        "temperature": cell.get("temperature", 20),
    }

    outcomes = apply_interaction("place_on_heat", food_mat, heat_mat, env, discoverer_id=agent.id, tick=getattr(agent, "age", 0))
    causal_mem = getattr(agent, "causal_memory", None)
    legacy_r = _evaluate_legacy_outcomes(agent, world, x, y, slot, outcomes, env)
    result_mats = [o for o in outcomes if not o.startswith("_")]
    if result_mats:
        source[food_mat] = max(0.0, source.get(food_mat, 0) - 0.5)

    vec_food = get_vector(food_mat)
    vec_heat = get_vector(heat_mat)
    new_vec = combine_vectors(vec_food, vec_heat, "place_on_heat", env)
    emergent_r = 0.0
    if new_vec is not None and float(new_vec.sum()) > 0.1:
        agent_state = _agent_homeostatic_state(agent, cell)
        emergent_r = material_reward(new_vec, agent_state)

        existing_ids = (
            set(DISCOVERY_REGISTRY.known_ids())
            if hasattr(DISCOVERY_REGISTRY, "known_ids")
            else set()
        )
        mat_id = DISCOVERY_REGISTRY.register(
            new_vec,
            discoverer_id=agent.id,
            tick=getattr(agent, "age", 0),
            recipe=("place_on_heat", food_mat, heat_mat),
        )
        is_new = mat_id not in existing_ids
        if is_new:
            emergent_r += NEW_DISCOVERY_BONUS
            print(
                f"[cook] NEW mat_id={mat_id} by agent={agent.id} recipe=(place_on_heat,{food_mat},{heat_mat})"
            )
        else:
            emergent_r = max(emergent_r, REDISCOVERY_REWARD)

        inv[mat_id] = inv.get(mat_id, 0.0) + 0.6

    total = legacy_r + emergent_r * 1.0
    if causal_mem is not None:
        causal_mem.record("place_on_heat", food_mat, heat_mat, outcomes, total)
    return total


def share_discovery(teacher, student, mat_id: str) -> bool:
    """
    Social knowledge transfer: teacher shows student a discovered material.
    Student gains the mat_id in inventory if trust is sufficient.

    FIX v3: Trust-Gate von 0.25 auf 0.05 gesenkt. Stammesmitglieder (gleiche
    tribe_id) können immer teilen — verhindert, dass Wissenstransfer in der
    kritischen Bootstrap-Phase (erste ~200 Ticks) komplett blockiert wird.
    """
    same_tribe = getattr(teacher, "tribe_id", None) is not None and teacher.tribe_id == getattr(
        student, "tribe_id", None
    )
    trust = teacher.trust.get(student.id, 0.0)
    # Stammesmitglieder teilen immer; alle anderen brauchen minimales Vertrauen
    if not same_tribe and trust < 0.05:
        return False
    if mat_id not in DISCOVERY_REGISTRY.known_ids():
        return False
    inv = getattr(student, "material_inventory", {})
    if inv.get(mat_id, 0.0) < 0.1:
        inv[mat_id] = 0.3
        student.material_inventory = inv
        if hasattr(student, "endocrine"):
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
            if "materials" in cell and cell["materials"]:
                env = {
                    "wind": cell.get("disturbance", 0) / 100.0,
                    "moisture": cell.get("moisture", 50) / 100.0,
                    "temperature": cell.get("temperature", 20),
                }
                decay_materials(cell["materials"], env)


def seed_world_materials(world):
    """
    Platziert Seed-Materialien in der Welt nach Biom.
    Neue scent/solubility-Materialien werden in passenden Biomen geseeded:
      flower_petals  → grassland, forest, swamp
      tree_resin     → forest
      crushed_herb   → grassland, swamp  (als rohe herb-Pflanze)
      clay           → swamp, mountain
      charcoal       → erzeugt sich durch fire-decay, aber kleiner Startvorrat
      animal_fat     → tundra, mountain (von Tieren)
    """
    biome_mats = {
        "forest": [
            ("dry_wood", 0.4),
            ("wet_wood", 0.3),
            ("dry_grass", 0.2),
            ("fiber", 0.3),
            ("tree_resin", 0.2),
            ("flower_petals", 0.1),
        ],
        "grassland": [
            ("dry_grass", 0.5),
            ("fiber", 0.4),
            ("raw_root", 0.2),
            ("flower_petals", 0.25),
            ("crushed_herb", 0.15),
        ],
        "mountain": [
            ("stone", 0.5),
            ("flint", 0.25),
            ("bone", 0.1),
            ("clay", 0.2),
            ("animal_fat", 0.1),
        ],
        "desert": [
            ("dry_grass", 0.3),
            ("flint", 0.2),
            ("stone", 0.3),
            ("dry_wood", 0.1),
        ],
        "swamp": [
            ("wet_wood", 0.4),
            ("fiber", 0.4),
            ("raw_root", 0.3),
            ("flower_petals", 0.2),
            ("crushed_herb", 0.2),
            ("clay", 0.35),
        ],
        "tundra": [
            ("dry_grass", 0.2),
            ("bone", 0.15),
            ("stone", 0.3),
            ("animal_fat", 0.2),
        ],
    }
    for y in range(world.height):
        for x in range(world.width):
            biome = world.biomes[y][x]
            if biome == "water":
                continue
            slot = {}
            for mat, base_qty in biome_mats.get(biome, []):
                if random.random() < 0.35:
                    slot[mat] = round(base_qty * random.uniform(0.5, 1.5), 2)
            world.set_cell(x, y, "materials", slot)
            world.set_cell(x, y, "warmth", 0.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _choose_action_by_need(agent, causal_mem, mat_a: str, mat_b, cell: dict) -> str:
    """
    SCHICHT 2: Bedürfnisgetriebene Aktionswahl.
    Agenten erfinden aus Not, nicht aus Zufall. Je nach aktuellem Zustand
    werden bestimmte Aktions-Cluster bevorzugt — aber Exploration bleibt möglich.
    """
    from artificial_society.agents.agent import MAX_ENERGY

    max_energy = MAX_ENERGY if MAX_ENERGY else 240.0
    energy_ratio = agent.energy / max_energy
    health_ratio = agent.health / 100.0
    temperature = cell.get("temperature", 20)
    is_sick = getattr(agent, "disease_id", None) is not None

    # Bestimme dominantes Bedürfnis
    if energy_ratio < 0.25:
        preferred = ACTIONS_FOR_HUNGER  # Hunger ist stärkstes Motiv
    elif temperature < 5 or (
        hasattr(agent, "endocrine") and getattr(agent.endocrine, "cortisol", 0) > 0.7
    ):
        preferred = ACTIONS_FOR_COLD  # Kälte/Stress → Feuer/Wärme
    elif is_sick or health_ratio < 0.4:
        preferred = ACTIONS_FOR_SICK  # Krank → Heilmittel
    else:
        preferred = ACTIONS_FOR_CURIOUS  # Keine Not → maximale Exploration

    # CausalMemory-Bias: Erfolgreiche bekannte Sequenzen bevorzugen (55%)
    if causal_mem is not None:
        good = causal_mem.best_known(min_successes=1)
        if good and random.random() < 0.55:
            for (act, ma, mb), _ in good:
                if (ma == mat_a or mb == (mat_b or "")) and act in preferred:
                    return act
            # Fallback: irgendeine bekannte gute Action
            for (act, ma, mb), _ in good:
                if ma == mat_a or mb == (mat_b or ""):
                    return act

    # Bedürfnis-basierte Zufallswahl (mit 20% Chance auf völlig freie Exploration)
    if random.random() < 0.20:
        return random.choice(PRIMITIVE_ACTIONS)
    return random.choice(preferred)


# Legacy-Wrapper für alten Code der _choose_action nutzt
def _choose_action(agent, causal_mem, mat_a: str, mat_b) -> str:
    return _choose_action_by_need(agent, causal_mem, mat_a, mat_b, {})


def _agent_homeostatic_state(agent, cell: dict) -> dict:
    from artificial_society.agents.agent import MAX_ENERGY

    return {
        "energy": agent.energy / MAX_ENERGY,
        "hydration": agent.hydration / 100.0,
        "health": agent.health / 100.0,
        "cold": cell.get("temperature", 20) < 8,
        "dark": cell.get("light", 1.0) < 0.3,
    }


def _maybe_upgrade_tool(agent, mat_id: str, vec: np.ndarray):
    current_sharpness = 0.0
    if agent.tool == "sharp_stone":
        current_sharpness = float(get_vector("sharp_stone")[IDX["sharpness"]])
    elif agent.tool and agent.tool.startswith("mat_"):
        current_sharpness = float(get_vector(agent.tool)[IDX["sharpness"]])
    if vec[IDX["sharpness"]] > current_sharpness + 0.05:
        agent.tool = mat_id


def _evaluate_legacy_outcomes(
    agent, world, x, y, slot: dict, outcomes: list[str], env: dict
) -> float:
    reward = 0.0
    for result in outcomes:
        if result in ("_spark", "_heat_trace"):
            pass
        elif result == "_tinder_bundle":
            inv = getattr(agent, "material_inventory", {})
            inv["_tinder_bundle"] = inv.get("_tinder_bundle", 0) + 1.0
        elif result.startswith("_nutrition:"):
            val = float(result.split(":")[1])
            agent.energy = min(getattr(agent, "energy", 100) + val * 18.0, 240.0)
            reward += val * 0.4
        elif result == "ember":
            slot["ember"] = slot.get("ember", 0.0) + 0.6
            world.set_cell(x, y, "materials", slot)
            reward += 0.10  # war 0.12
        elif result == "fire":
            slot["fire"] = slot.get("fire", 0.0) + 1.0
            world.set_cell(x, y, "materials", slot)
            # war 0.55 — gedämpft auf LEGACY_MAX_REWARD-Niveau, damit Feuer-Schlagen
            # nicht das dominante Skript-Outcome bleibt und Emergenz erstickt.
            reward += 0.40
        elif result == "ash":
            slot["ash"] = slot.get("ash", 0.0) + 0.5
        elif result == "cooked_meat":
            inv = getattr(agent, "material_inventory", {})
            inv["cooked_meat"] = inv.get("cooked_meat", 0.0) + 0.8
            reward += COOK_REWARD
        elif result == "cooked_root":
            inv = getattr(agent, "material_inventory", {})
            inv["cooked_root"] = inv.get("cooked_root", 0.0) + 0.8
            reward += COOK_REWARD * 0.7
        elif result == "sharp_stone":
            inv = getattr(agent, "material_inventory", {})
            inv["sharp_stone"] = inv.get("sharp_stone", 0.0) + 1.0
            reward += SHARP_REWARD
            agent.tool = "sharp_stone"
        elif result.startswith("mat_"):
            inv = getattr(agent, "material_inventory", {})
            inv[result] = inv.get(result, 0.0) + 0.5

    heat = material_heat(slot)
    light = material_light(slot)
    danger = material_danger(slot)
    reward += heat * WARMTH_REWARD + light * LIGHT_REWARD + danger * DANGER_PENALTY
    world.set_cell(x, y, "warmth", heat)
    return reward
