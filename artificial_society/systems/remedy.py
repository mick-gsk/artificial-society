"""
Realistic Disease System — v2.0

Six historically-grounded diseases, each modelling:
  - Real transmission vector (water, contact, airborne, wound)
  - Biome affinity (where spread is amplified)
  - Unique symptom effects per tick on agent stats
  - Real historical herb/food remedies (hidden from agents)
  - Two-tier healing: partial relief from individual ingredients,
    full cure only when the complete recipe is applied within the window

Real-world inspirations:
  Malaria        → Plasmodium, mosquito vector, swamp/wet biomes
  Dysentery      → Shigella/Amoeba, contaminated water
  Tuberculosis   → Mycobacterium, airborne, cold/damp biomes
  Typhoid Fever  → Salmonella typhi, contaminated food & water
  Scurvy         → Vitamin C deficiency, no direct spread
  Wound Fever    → Sepsis/Erysipelas, spread via contact with injured agents

NOTE: This registry is NEVER exposed to agents directly.
Agents must discover cures through experimentation and social sharing.
"""

import random

# ---------------------------------------------------------------------------
# Master Registry
# ---------------------------------------------------------------------------
REMEDY_REGISTRY: dict[str, dict] = {

    # ------------------------------------------------------------------
    # MALARIA
    # Real: Plasmodium parasite, mosquito vector, sub-Saharan Africa.
    # Symptoms: cyclic fever, chills, fatigue, anaemia (energy & hydration drain).
    # Historical cure: Cinchona bark (quinine). Artemisia (wormwood) as partial.
    # Biome: swamp >> grassland >> forest
    # ------------------------------------------------------------------
    'malaria': {
        'name': 'Malaria',
        'description': 'Cyclic fever, chills and fatigue from a blood parasite.',
        'vector': 'contact',          # spread by proximity in wet biomes
        'biome_amplify': ['swamp'],   # spread_rate multiplied in these biomes
        'biome_amplify_factor': 2.5,
        'ingredients': ['herb_wormwood', 'water'],   # Artemisia annua + hydration
        'window': 6,
        'cure_health': 30.0,
        'cure_sick': 65.0,
        'spread_rate': 0.010,
        'initial_sick': 10.0,
        'immunity_after': 300,        # ticks of immunity after recovery
        # Per-tick symptom effects (applied inside apply_disease)
        'symptom': {
            'energy_drain': 0.12,     # fatigue
            'hydration_drain': 0.20,  # sweating / chills
            'health_drain_per_sick': 0.008,
        },
        # Partial: any single ingredient gives minor relief
        'partial_ingredients': {'herb_wormwood': 9.0, 'water': 5.0},
    },

    # ------------------------------------------------------------------
    # DYSENTERY
    # Real: Shigella/Entamoeba, contaminated water, history of campaigns.
    # Symptoms: severe dehydration, gut pain → hydration collapses fast.
    # Historical cure: Blackberry root, oak bark (tannins), charcoal.
    # Biome: swamp, near water cells with high pollution
    # ------------------------------------------------------------------
    'dysentery': {
        'name': 'Dysentery',
        'description': 'Severe dehydration and gut cramps from contaminated water.',
        'vector': 'water',            # spreads through shared water cells
        'biome_amplify': ['swamp', 'grassland'],
        'biome_amplify_factor': 1.8,
        'ingredients': ['herb_oak_bark', 'plant_food'],   # tannins + nutrition
        'window': 5,
        'cure_health': 22.0,
        'cure_sick': 55.0,
        'spread_rate': 0.012,
        'initial_sick': 12.0,
        'immunity_after': 180,
        'symptom': {
            'energy_drain': 0.08,
            'hydration_drain': 0.50,  # major symptom: dehydration
            'health_drain_per_sick': 0.010,
        },
        'partial_ingredients': {'herb_oak_bark': 8.0, 'plant_food': 4.0},
    },

    # ------------------------------------------------------------------
    # TUBERCULOSIS
    # Real: Mycobacterium tuberculosis, airborne, cold/damp environments.
    # Symptoms: slow progressive health drain, energy loss, chronic.
    # Historical cure: Garlic (allicin), eucalyptus oil, fresh air + sunlight.
    # Biome: mountain, forest (cold & damp)
    # ------------------------------------------------------------------
    'tuberculosis': {
        'name': 'Tuberculosis',
        'description': 'Slow progressive lung disease spread through the air.',
        'vector': 'airborne',         # spreads at radius 2 instead of 1
        'biome_amplify': ['mountain', 'forest'],
        'biome_amplify_factor': 1.6,
        'ingredients': ['herb_garlic', 'herb_eucalyptus'],
        'window': 7,
        'cure_health': 28.0,
        'cure_sick': 45.0,
        'spread_rate': 0.008,         # airborne but slower incubation
        'initial_sick': 6.0,          # slow onset
        'immunity_after': 400,        # long immunity after recovery
        'symptom': {
            'energy_drain': 0.10,
            'hydration_drain': 0.05,
            'health_drain_per_sick': 0.012,   # slowly lethal if untreated
        },
        'partial_ingredients': {'herb_garlic': 9.0, 'herb_eucalyptus': 9.0},
    },

    # ------------------------------------------------------------------
    # TYPHOID FEVER
    # Real: Salmonella typhi, faecal-oral route, contaminated food & water.
    # Symptoms: sustained high fever, confusion (action randomisation),
    #           intestinal damage.
    # Historical cure: Willow bark (aspirin precursor) + clean water + rest.
    # Biome: any with high disease cell value
    # ------------------------------------------------------------------
    'typhoid': {
        'name': 'Typhoid Fever',
        'description': 'Sustained fever and confusion from contaminated food or water.',
        'vector': 'food_water',
        'biome_amplify': [],          # amplified by cell disease level instead
        'biome_amplify_factor': 1.0,
        'ingredients': ['herb_willow', 'water', 'plant_food'],
        'window': 6,
        'cure_health': 32.0,
        'cure_sick': 60.0,
        'spread_rate': 0.010,
        'initial_sick': 14.0,
        'immunity_after': 500,        # typhoid gives long-term immunity historically
        'symptom': {
            'energy_drain': 0.15,
            'hydration_drain': 0.25,
            'health_drain_per_sick': 0.009,
            'confusion': True,        # flag: action values randomised when sick > 50
        },
        'partial_ingredients': {'herb_willow': 10.0, 'water': 6.0, 'plant_food': 4.0},
    },

    # ------------------------------------------------------------------
    # SCURVY
    # Real: Vitamin C deficiency. Not contagious — environmental/dietary.
    # Symptoms: weakness, bleeding, slow wound healing (health regen disabled).
    # Historical cure: Citrus / fresh plant food. Almost instant reversal.
    # Biome: desert, tundra-like cold regions (no fresh food)
    # ------------------------------------------------------------------
    'scurvy': {
        'name': 'Scurvy',
        'description': 'Vitamin C deficiency causing weakness and slow healing.',
        'vector': 'dietary',          # NOT person-to-person — only from environment
        'biome_amplify': ['desert'],
        'biome_amplify_factor': 1.0,  # amplify not used for dietary
        'ingredients': ['herb_rosehip', 'plant_food'],   # rosehips = highest natural vit C
        'window': 4,
        'cure_health': 35.0,
        'cure_sick': 80.0,            # fast reversal with correct diet
        'spread_rate': 0.0,           # non-contagious
        'initial_sick': 8.0,
        'immunity_after': 100,
        'symptom': {
            'energy_drain': 0.10,
            'hydration_drain': 0.02,
            'health_drain_per_sick': 0.006,
            'regen_block': True,      # flag: natural health regeneration disabled
        },
        'partial_ingredients': {'herb_rosehip': 15.0, 'plant_food': 8.0},
    },

    # ------------------------------------------------------------------
    # WOUND FEVER (Sepsis / Erysipelas)
    # Real: Streptococcus/Staph entering through wounds. Historically deadly.
    # Symptoms: rapid health loss, energy crash, heat.
    # Historical cure: Honey (antibacterial), moss (wound dressing), willow.
    # Biome: any — triggered when agent health drops below threshold
    # ------------------------------------------------------------------
    'wound_fever': {
        'name': 'Wound Fever',
        'description': 'Infection entering through wounds — rapid and dangerous.',
        'vector': 'wound',            # triggered by low health, not direct spread
        'biome_amplify': [],
        'biome_amplify_factor': 1.0,
        'ingredients': ['herb_moss', 'herb_willow'],   # honey/moss wound dressing
        'window': 4,
        'cure_health': 25.0,
        'cure_sick': 60.0,
        'spread_rate': 0.006,         # low contact spread (handling wounds)
        'initial_sick': 18.0,         # fast and aggressive onset
        'immunity_after': 150,
        'symptom': {
            'energy_drain': 0.18,     # most draining disease
            'hydration_drain': 0.15,
            'health_drain_per_sick': 0.014,   # most dangerous
        },
        'partial_ingredients': {'herb_moss': 10.0, 'herb_willow': 10.0},
    },
}

# Public tag-set so the world can spawn matching herb resources
ALL_HERB_TAGS: list[str] = sorted({
    tag
    for rec in REMEDY_REGISTRY.values()
    for tag in rec['ingredients']
    if tag.startswith('herb_')
})

# Diseases that can spread person-to-person
CONTACT_DISEASES = {
    did for did, rec in REMEDY_REGISTRY.items()
    if rec['vector'] in ('contact', 'airborne', 'food_water', 'wound')
    and rec['spread_rate'] > 0
}

# Non-contagious diseases (triggered by environment/diet)
ENVIRONMENTAL_DISEASES = {
    did for did, rec in REMEDY_REGISTRY.items()
    if rec['vector'] in ('dietary', 'wound')
}


# ---------------------------------------------------------------------------
# Infection helper  (called from simulation.spread_diseases)
# ---------------------------------------------------------------------------

def try_infect_agent(agent, disease_id: str, biome: str = '') -> bool:
    """Attempt to infect agent with disease_id. Returns True if infection occurred."""
    rec = REMEDY_REGISTRY.get(disease_id)
    if rec is None:
        return False
    if getattr(agent, 'disease_id', None) is not None:
        return False
    rate = rec['spread_rate']
    # Biome amplification
    if biome in rec.get('biome_amplify', []):
        rate = min(0.35, rate * rec['biome_amplify_factor'])
    if random.random() < rate:
        agent.disease_id = disease_id
        agent.sick = min(100.0, agent.sick + rec['initial_sick'])
        return True
    return False


def try_environmental_infection(agent, cell: dict) -> bool:
    """
    Environmental/dietary infection trigger (called from agent.apply_disease).
    Scurvy:      triggered in desert biomes when agent has eaten no plant food recently.
    Wound Fever: triggered when health < 35 (open wounds).
    """
    if getattr(agent, 'disease_id', None) is not None:
        return False

    triggered = False

    # Scurvy: desert biome + low plant intake
    if cell.get('biome') == 'desert' and getattr(agent, 'plant_eaten', 0) % 80 == 0:
        if random.random() < 0.03:
            agent.disease_id = 'scurvy'
            agent.sick = min(100.0, agent.sick + REMEDY_REGISTRY['scurvy']['initial_sick'])
            triggered = True

    # Wound Fever: triggered by low health (open wound proxy)
    if not triggered and agent.health < 35.0 and random.random() < 0.015:
        agent.disease_id = 'wound_fever'
        agent.sick = min(100.0, agent.sick + REMEDY_REGISTRY['wound_fever']['initial_sick'])
        triggered = True

    return triggered


def apply_disease_symptoms(agent, cell: dict):
    """
    Apply per-tick symptom effects for agent's current disease.
    Called from agent.apply_disease() instead of the generic formula.
    Returns True if symptoms were applied.
    """
    disease_id = getattr(agent, 'disease_id', None)
    if disease_id is None:
        return False

    rec = REMEDY_REGISTRY.get(disease_id)
    if rec is None:
        return False

    sym = rec.get('symptom', {})
    sick_ratio = agent.sick / 100.0

    # Core stat drains (scaled by how sick the agent is)
    agent.energy = max(0.0, agent.energy - sym.get('energy_drain', 0.08) * sick_ratio)
    agent.hydration = max(0.0, agent.hydration - sym.get('hydration_drain', 0.10) * sick_ratio)
    agent.health = max(0.0, agent.health - sym.get('health_drain_per_sick', 0.008) * agent.sick)

    # Special symptom flags
    if sym.get('regen_block'):
        # Scurvy: prevent natural health regen (handled externally by flag check)
        agent._scurvy_active = True
    else:
        agent._scurvy_active = False

    if sym.get('confusion') and agent.sick > 50:
        # Typhoid: randomly corrupt one action weight this tick
        agent._confused = True
    else:
        agent._confused = False

    # Natural recovery (slow, scaled by hydration + health)
    base_recovery = 0.12 * (agent.health / 100.0) + 0.04 * (agent.hydration / 100.0)
    agent.sick = max(0.0, agent.sick - base_recovery)

    # Warmth bonus (shelter / fire reduces sickness)
    warmth = cell.get('warmth', 0.0)
    if warmth > 0.2:
        agent.sick = max(0.0, agent.sick - 0.12 * warmth)

    if agent.sick <= 2.0:
        agent.disease_id = None

    return True


# ---------------------------------------------------------------------------
# Cure evaluation  (called from agent._try_use_herbs)
# ---------------------------------------------------------------------------

def evaluate_remedy(agent, consumed_tags: list[str]) -> float:
    """
    Two-tier healing:
    Tier 1 — Full cure:    all ingredients in rolling window  → major heal.
    Tier 2 — Partial cure: single known ingredient matched    → minor relief.
    """
    disease_id = getattr(agent, 'disease_id', None)
    if disease_id is None:
        return 0.0

    rec = REMEDY_REGISTRY[disease_id]
    required = set(rec['ingredients'])
    partial_map: dict = rec.get('partial_ingredients', {})

    if not hasattr(agent, '_remedy_window'):
        agent._remedy_window = []
    agent._remedy_window.extend(consumed_tags)
    max_per_tick = 5
    agent._remedy_window = agent._remedy_window[-(rec['window'] * max_per_tick):]

    # --- Tier 1: Full cure ---
    if required.issubset(set(agent._remedy_window)):
        agent.health = min(100.0, agent.health + rec['cure_health'])
        agent.sick = max(0.0, agent.sick - rec['cure_sick'])
        if agent.sick <= 0:
            agent.disease_id = None
        agent._remedy_window = []
        return rec['cure_health'] / 25.0

    # --- Tier 2: Partial cure ---
    partial_bonus = 0.0
    for tag in consumed_tags:
        if tag in partial_map:
            sick_red = partial_map[tag] * 0.22
            health_gain = sick_red * 0.3
            agent.sick = max(0.0, agent.sick - sick_red)
            agent.health = min(100.0, agent.health + health_gain)
            partial_bonus += health_gain / 25.0
            if agent.sick <= 0:
                agent.disease_id = None
                break
    return partial_bonus


# ---------------------------------------------------------------------------
# Social knowledge sharing
# ---------------------------------------------------------------------------

def share_remedy_knowledge(sender, receiver) -> bool:
    sender_knowledge: dict = getattr(sender, 'remedy_knowledge', {})
    if not sender_knowledge:
        return False
    disease_id, ingredient_clue = random.choice(list(sender_knowledge.items()))
    receiver_knowledge: dict = getattr(receiver, 'remedy_knowledge', {})
    if disease_id not in receiver_knowledge:
        receiver_knowledge[disease_id] = set()
    before_len = len(receiver_knowledge[disease_id])
    receiver_knowledge[disease_id].update(ingredient_clue)
    receiver.remedy_knowledge = receiver_knowledge
    return len(receiver_knowledge[disease_id]) > before_len


def record_cure_discovery(agent, disease_id: str, ingredients_used: list[str]):
    if not hasattr(agent, 'remedy_knowledge'):
        agent.remedy_knowledge = {}
    if disease_id not in agent.remedy_knowledge:
        agent.remedy_knowledge[disease_id] = set()
    agent.remedy_knowledge[disease_id].update(ingredients_used)
