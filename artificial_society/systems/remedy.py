"""
Realistic Fault System — v2.0

Six historically-grounded faults, each modelling:
  - Real transmission vector (water, contact, airborne, wound)
  - Biome affinity (where spread is amplified)
  - Unique indicator effects per tick on agent stats
  - Real historical herb/food remedies (hidden from agents)
  - Two-tier recovery: partial relief from individual ingredients,
    full cure only when the complete recipe is applied within the window

Real-world inspirations:
  Malaria        → Plasmodium, mosquito vector, swamp/wet biomes
  Dysentery      → Shigella/Amoeba, contaminated water
  Tuberculosis   → Mycobacterium, airborne, cold/damp biomes
  Typhoid Fever  → Salmonella typhi, contaminated food & water
  Scurvy         → Vitamin C deficiency, no direct spread
  Wound Fever    → Sepsis/Erysipelas, spread via contact with damaged agents

NOTE: This registry is NEVER exposed to agents directly.
Agents must discover cures through experimentation and social sharing.
"""

import random

import numpy as np

from artificial_society.environment.materials import IDX, get_vector

# ---------------------------------------------------------------------------
# Master Registry
# ---------------------------------------------------------------------------
REMEDY_REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------
    # MALARIA
    # Real: Plasmodium parasite, mosquito vector, sub-Saharan Africa.
    # Indicators: cyclic fever, chills, fatigue, anaemia (energy & hydration drain).
    # Historical cure: Cinchona bark (quinine). Artemisia (wormwood) as partial.
    # Biome: swamp >> grassland >> forest
    # ------------------------------------------------------------------
    "malaria": {
        "name": "Malaria",
        "description": "Cyclic fever, chills and fatigue from a blood parasite.",
        "vector": "contact",  # spread by proximity in wet biomes
        "biome_amplify": ["swamp"],  # spread_rate multiplied in these biomes
        "biome_amplify_factor": 2.5,
        "ingredients": ["herb_wormwood", "water"],  # Artemisia annua + hydration
        "window": 6,
        "cure_health": 30.0,
        "cure_sick": 65.0,
        "spread_rate": 0.010,
        "initial_sick": 10.0,
        "immunity_after": 300,  # ticks of resistance after recovery
        # Per-tick indicator effects (applied inside apply_fault)
        "indicator": {
            "energy_drain": 0.12,  # fatigue
            "hydration_drain": 0.20,  # sweating / chills
            "health_drain_per_sick": 0.008,
        },
        # Partial: any single ingredient gives minor relief
        "partial_ingredients": {"herb_wormwood": 9.0, "water": 5.0},
    },
    # ------------------------------------------------------------------
    # DYSENTERY
    # Real: Shigella/Entamoeba, contaminated water, history of campaigns.
    # Indicators: severe dehydration, gut pain → hydration collapses fast.
    # Historical cure: Blackberry root, oak bark (tannins), charcoal.
    # Biome: swamp, near water cells with high pollution
    # ------------------------------------------------------------------
    "dysentery": {
        "name": "Dysentery",
        "description": "Severe dehydration and gut cramps from contaminated water.",
        "vector": "water",  # spreads through shared water cells
        "biome_amplify": ["swamp", "grassland"],
        "biome_amplify_factor": 1.8,
        "ingredients": ["herb_oak_bark", "plant_food"],  # tannins + resource_value
        "window": 5,
        "cure_health": 22.0,
        "cure_sick": 55.0,
        "spread_rate": 0.012,
        "initial_sick": 12.0,
        "immunity_after": 180,
        "indicator": {
            "energy_drain": 0.08,
            "hydration_drain": 0.50,  # major indicator: dehydration
            "health_drain_per_sick": 0.010,
        },
        "partial_ingredients": {"herb_oak_bark": 8.0, "plant_food": 4.0},
    },
    # ------------------------------------------------------------------
    # TUBERCULOSIS
    # Real: Mycobacterium tuberculosis, airborne, cold/damp environments.
    # Indicators: slow progressive health drain, energy loss, chronic.
    # Historical cure: Garlic (allicin), eucalyptus oil, fresh air + sunlight.
    # Biome: mountain, forest (cold & damp)
    # ------------------------------------------------------------------
    "tuberculosis": {
        "name": "Tuberculosis",
        "description": "Slow progressive lung disease spread through the air.",
        "vector": "airborne",  # spreads at radius 2 instead of 1
        "biome_amplify": ["mountain", "forest"],
        "biome_amplify_factor": 1.6,
        "ingredients": ["herb_garlic", "herb_eucalyptus"],
        "window": 7,
        "cure_health": 28.0,
        "cure_sick": 45.0,
        "spread_rate": 0.008,  # airborne but slower incubation
        "initial_sick": 6.0,  # slow onset
        "immunity_after": 400,  # long resistance after recovery
        "indicator": {
            "energy_drain": 0.10,
            "hydration_drain": 0.05,
            "health_drain_per_sick": 0.012,  # slowly lethal if untreated
        },
        "partial_ingredients": {"herb_garlic": 9.0, "herb_eucalyptus": 9.0},
    },
    # ------------------------------------------------------------------
    # TYPHOID FEVER
    # Real: Salmonella typhi, faecal-oral route, contaminated food & water.
    # Indicators: sustained high fever, confusion (action randomisation),
    #           intestinal damage.
    # Historical cure: Willow bark (aspirin precursor) + clean water + rest.
    # Biome: any with high fault cell value
    # ------------------------------------------------------------------
    "typhoid": {
        "name": "Typhoid Fever",
        "description": "Sustained fever and confusion from contaminated food or water.",
        "vector": "food_water",
        "biome_amplify": [],  # amplified by cell fault level instead
        "biome_amplify_factor": 1.0,
        "ingredients": ["herb_willow", "water", "plant_food"],
        "window": 6,
        "cure_health": 32.0,
        "cure_sick": 60.0,
        "spread_rate": 0.010,
        "initial_sick": 14.0,
        "immunity_after": 500,  # typhoid gives long-term resistance historically
        "indicator": {
            "energy_drain": 0.15,
            "hydration_drain": 0.25,
            "health_drain_per_sick": 0.009,
            "confusion": True,  # flag: action values randomised when impaired > 50
        },
        "partial_ingredients": {"herb_willow": 10.0, "water": 6.0, "plant_food": 4.0},
    },
    # ------------------------------------------------------------------
    # SCURVY
    # Real: Vitamin C deficiency. Not spreading — environmental/dietary.
    # Indicators: weakness, bleeding, slow damage recovery (health regen disabled).
    # Historical cure: Citrus / fresh plant food. Almost instant reversal.
    # Biome: desert, tundra-like cold regions (no fresh food)
    # ------------------------------------------------------------------
    "scurvy": {
        "name": "Scurvy",
        "description": "Vitamin C deficiency causing weakness and slow healing.",
        "vector": "dietary",  # NOT person-to-person — only from environment
        "biome_amplify": ["desert"],
        "biome_amplify_factor": 1.0,  # amplify not used for dietary
        "ingredients": ["herb_rosehip", "plant_food"],  # rosehips = highest natural vit C
        "window": 4,
        "cure_health": 35.0,
        "cure_sick": 80.0,  # fast reversal with correct diet
        "spread_rate": 0.0,  # non-spreading
        "initial_sick": 8.0,
        "immunity_after": 100,
        "indicator": {
            "energy_drain": 0.10,
            "hydration_drain": 0.02,
            "health_drain_per_sick": 0.006,
            "regen_block": True,  # flag: natural health regeneration disabled
        },
        "partial_ingredients": {"herb_rosehip": 15.0, "plant_food": 8.0},
    },
    # ------------------------------------------------------------------
    # WOUND FEVER (Sepsis / Erysipelas)
    # Real: Streptococcus/Staph entering through wounds. Historically deadly.
    # Indicators: rapid health loss, energy crash, heat.
    # Historical cure: Honey (antibacterial), moss (wound dressing), willow.
    # Biome: any — triggered when agent health drops below threshold
    # ------------------------------------------------------------------
    "wound_fever": {
        "name": "Wound Fever",
        "description": "Infection entering through wounds — rapid and dangerous.",
        "vector": "wound",  # triggered by low health, not direct spread
        "biome_amplify": [],
        "biome_amplify_factor": 1.0,
        "ingredients": ["herb_moss", "herb_willow"],  # honey/moss wound dressing
        "window": 4,
        "cure_health": 25.0,
        "cure_sick": 60.0,
        "spread_rate": 0.006,  # low contact spread (handling wounds)
        "initial_sick": 18.0,  # fast and aggressive onset
        "immunity_after": 150,
        "indicator": {
            "energy_drain": 0.18,  # most draining fault
            "hydration_drain": 0.15,
            "health_drain_per_sick": 0.014,  # most dangerous
        },
        "partial_ingredients": {"herb_moss": 10.0, "herb_willow": 10.0},
    },
}

# Public tag-set so the world can spawn matching herb resources
ALL_HERB_TAGS: list[str] = sorted(
    {
        tag
        for rec in REMEDY_REGISTRY.values()
        for tag in rec["ingredients"]
        if tag.startswith("herb_")
    }
)

# Faults that can spread person-to-person
CONTACT_FAULTS = {
    did
    for did, rec in REMEDY_REGISTRY.items()
    if rec["vector"] in ("contact", "airborne", "food_water", "wound") and rec["spread_rate"] > 0
}

# Non-spreading faults (triggered by environment/diet)
ENVIRONMENTAL_FAULTS = {
    did for did, rec in REMEDY_REGISTRY.items() if rec["vector"] in ("dietary", "wound")
}


# ---------------------------------------------------------------------------
# Propagation helper  (called from simulation.spread_faults)
# ---------------------------------------------------------------------------


def try_spread_agent(agent, fault_id: str, biome: str = "") -> bool:
    """Attempt to spread agent with fault_id. Returns True if propagation occurred."""
    rec = REMEDY_REGISTRY.get(fault_id)
    if rec is None:
        return False
    if getattr(agent, "fault_id", None) is not None:
        return False
    rate = rec["spread_rate"]
    # Biome amplification
    if biome in rec.get("biome_amplify", []):
        rate = min(0.35, rate * rec["biome_amplify_factor"])
    if random.random() < rate:
        agent.fault_id = fault_id
        agent.impaired = min(100.0, agent.impaired + rec["initial_sick"])
        return True
    return False


def try_environmental_propagation(agent, cell: dict) -> bool:
    """
    Environmental/dietary propagation trigger (called from agent.apply_fault).
    Scurvy:      triggered in desert biomes when agent has eaten no plant food recently.
    Wound Fever: triggered when health < 35 (open wound).
    """
    if getattr(agent, "fault_id", None) is not None:
        return False

    triggered = False

    # Scurvy: desert biome + low plant intake
    if (
        cell.get("biome") == "desert"
        and getattr(agent, "plant_eaten", 0) % 80 == 0
        and random.random() < 0.03
    ):
        agent.fault_id = "scurvy"
        agent.impaired = min(100.0, agent.impaired + REMEDY_REGISTRY["scurvy"]["initial_sick"])
        triggered = True

    # Wound Fever: triggered by low health (open wound proxy)
    if not triggered and agent.health < 35.0 and random.random() < 0.015:
        agent.fault_id = "wound_fever"
        agent.impaired = min(100.0, agent.impaired + REMEDY_REGISTRY["wound_fever"]["initial_sick"])
        triggered = True

    return triggered


def apply_fault_indicators(agent, cell: dict):
    """
    Apply per-tick indicator effects for agent's current fault.
    Called from agent.apply_fault() instead of the generic formula.
    Returns True if indicators were applied.
    """
    fault_id = getattr(agent, "fault_id", None)
    if fault_id is None:
        return False

    rec = REMEDY_REGISTRY.get(fault_id)
    if rec is None:
        return False

    indicator = rec.get("indicator", {})
    impaired_ratio = agent.impaired / 100.0

    # Core stat drains (scaled by how impaired the agent is)
    agent.energy = max(0.0, agent.energy - indicator.get("energy_drain", 0.08) * impaired_ratio)
    agent.hydration = max(
        0.0, agent.hydration - indicator.get("hydration_drain", 0.10) * impaired_ratio
    )
    agent.health = max(
        0.0, agent.health - indicator.get("health_drain_per_sick", 0.008) * agent.impaired
    )

    # Special indicator flags
    if indicator.get("regen_block"):
        # Scurvy: prevent natural health regen (handled externally by flag check)
        agent._scurvy_active = True
    else:
        agent._scurvy_active = False

    if indicator.get("confusion") and agent.impaired > 50:
        # Typhoid: randomly corrupt one action weight this tick
        agent._confused = True
    else:
        agent._confused = False

    # Natural recovery (slow, scaled by hydration + health)
    base_recovery = 0.12 * (agent.health / 100.0) + 0.04 * (agent.hydration / 100.0)
    agent.impaired = max(0.0, agent.impaired - base_recovery)

    # Warmth bonus (shelter / fire reduces impairment)
    warmth = cell.get("warmth", 0.0)
    if warmth > 0.2:
        agent.impaired = max(0.0, agent.impaired - 0.12 * warmth)

    if agent.impaired <= 2.0:
        agent.fault_id = None

    return True


# ---------------------------------------------------------------------------
# Cure evaluation  (called from agent._try_use_herbs)
# ---------------------------------------------------------------------------


def _item_potency(tag: str) -> float:
    """Medizinische Potenz eines konsumierten Items -- aus seinen EIGENSCHAFTEN abgeleitet.

    Phase 5 de-scripting: kein fault->ingredient-Lookup. Heilwirkung emergiert aus dem,
    WAS das Item ist -- aromatische (trail), essbare, ungiftige Stoffe lindern anhand ihres
    Materialvektors; Kraeuter (ohne Vektor) sind generisch medizinisch; Wasser/Nahrung
    unterstuetzen. So koennen Remedies aus Eigenschaften entdeckt werden statt fest
    verdrahtet zu sein.
    """
    vec = get_vector(tag)
    if float(np.linalg.norm(vec)) > 1e-3:
        trail = float(vec[IDX["trail"]])
        edible = float(vec[IDX["edibility"]])
        tox = float(vec[IDX["toxicity"]])
        return max(0.0, trail * 1.0 + edible * 0.4 - tox * 1.5)
    if tag.startswith("herb_"):
        return 0.6
    if tag in ("water", "plant_food"):
        return 0.3
    return 0.0


def _medicinal_dose(tags: list) -> float:
    """Kumulierte medizinische Dosis. Wiederholung desselben Stoffs bringt weniger
    (Vielfalt der Heilmittel wird belohnt -- emergente 'Rezeptur')."""
    seen: dict = {}
    total = 0.0
    for tag in tags:
        n = seen.get(tag, 0)
        seen[tag] = n + 1
        total += _item_potency(tag) * (0.7**n)
    return total


def evaluate_remedy(agent, consumed_tags: list[str]) -> float:
    """
    Eigenschaftsbasierte Heilung (Phase 5 de-scripting).

    Frueher musste exakt die hartkodierte Zutatenliste der Krankheit konsumiert werden.
    Jetzt emergiert die Wirkung aus den EIGENSCHAFTEN des Konsumierten (_item_potency):
      Tier 1 -- Vollheilung: genug kumulierte medizinische Dosis im Fenster.
      Tier 2 -- Linderung:   proportional zur frisch konsumierten Dosis.
    Schwierigkeit und Heilmengen kommen weiter aus der Krankheits-Definition (window,
    cure_health, cure_sick) -- das ist kein Zutaten-Lookup.
    """
    fault_id = getattr(agent, "fault_id", None)
    if fault_id is None:
        return 0.0

    rec = REMEDY_REGISTRY[fault_id]

    if not hasattr(agent, "_remedy_window"):
        agent._remedy_window = []
    agent._remedy_window.extend(consumed_tags)
    max_per_tick = 5
    agent._remedy_window = agent._remedy_window[-(rec["window"] * max_per_tick) :]

    # --- Tier 1: Vollheilung bei ausreichender kumulierter Dosis ---
    threshold = 1.5 + 0.3 * rec["window"]
    if _medicinal_dose(agent._remedy_window) >= threshold:
        agent.health = min(100.0, agent.health + rec["cure_health"])
        agent.impaired = max(0.0, agent.impaired - rec["cure_sick"])
        if agent.impaired <= 0:
            agent.fault_id = None
        agent._remedy_window = []
        return rec["cure_health"] / 25.0

    # --- Tier 2: Partielle Linderung proportional zur frischen Dosis ---
    fresh_dose = _medicinal_dose(consumed_tags)
    if fresh_dose <= 0.0:
        return 0.0
    impaired_red = min(agent.impaired, fresh_dose * 7.0)
    health_gain = impaired_red * 0.3
    agent.impaired = max(0.0, agent.impaired - impaired_red)
    agent.health = min(100.0, agent.health + health_gain)
    if agent.impaired <= 0:
        agent.fault_id = None
    return health_gain / 25.0


# ---------------------------------------------------------------------------
# Social knowledge sharing
# ---------------------------------------------------------------------------


def share_remedy_knowledge(sender, receiver) -> bool:
    sender_knowledge: dict = getattr(sender, "remedy_knowledge", {})
    if not sender_knowledge:
        return False
    fault_id, ingredient_clue = random.choice(list(sender_knowledge.items()))
    receiver_knowledge: dict = getattr(receiver, "remedy_knowledge", {})
    if fault_id not in receiver_knowledge:
        receiver_knowledge[fault_id] = set()
    # Coerce to set — old checkpoints may have deserialized this as a list
    if not isinstance(receiver_knowledge[fault_id], set):
        receiver_knowledge[fault_id] = set(receiver_knowledge[fault_id])
    # ingredient_clue may itself be a list/set; normalise before update
    if isinstance(ingredient_clue, (list, set)):
        clue_set = set(ingredient_clue)
    else:
        clue_set = {ingredient_clue}
    before_len = len(receiver_knowledge[fault_id])
    receiver_knowledge[fault_id].update(clue_set)
    receiver.remedy_knowledge = receiver_knowledge
    return len(receiver_knowledge[fault_id]) > before_len


def record_cure_discovery(agent, fault_id: str, ingredients_used: list[str]):
    if not hasattr(agent, "remedy_knowledge"):
        agent.remedy_knowledge = {}
    if fault_id not in agent.remedy_knowledge:
        agent.remedy_knowledge[fault_id] = set()
    # Coerce to set — old checkpoints may have deserialized this as a list
    if not isinstance(agent.remedy_knowledge[fault_id], set):
        agent.remedy_knowledge[fault_id] = set(agent.remedy_knowledge[fault_id])
    agent.remedy_knowledge[fault_id].update(ingredients_used)
