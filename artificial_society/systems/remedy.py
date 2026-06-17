"""
Hidden remedy system.

Only this module knows the actual cure recipes.
Agents only observe: they are sick, and what resources/plants they just consumed.
Through trial-and-error and social sharing they must discover what heals them.
"""

import random

# ---------------------------------------------------------------------------
# Master registry  –  NEVER exposed to agents directly
# ---------------------------------------------------------------------------
REMEDY_REGISTRY: dict[str, dict] = {
    'fever': {
        'name': 'Fever',
        'ingredients': ['herb_willow', 'water'],
        'window': 4,
        'cure_health': 25.0,
        'cure_sick': 50.0,
        'spread_rate': 0.012,   # further reduced from 0.018
        # Partial cure: any single ingredient gives minor relief
        'partial_ingredients': {'herb_willow': 8.0, 'water': 5.0},
    },
    'plague': {
        'name': 'Plague',
        'ingredients': ['herb_garlic', 'herb_elderberry', 'meat'],
        'window': 5,
        'cure_health': 35.0,
        'cure_sick': 70.0,
        'spread_rate': 0.015,   # reduced from 0.025
        'partial_ingredients': {'herb_garlic': 10.0, 'herb_elderberry': 10.0, 'meat': 5.0},
    },
    'blight_sickness': {
        'name': 'Blight Sickness',
        'ingredients': ['herb_mushroom', 'plant_food'],
        'window': 5,
        'cure_health': 20.0,
        'cure_sick': 40.0,
        'spread_rate': 0.008,   # reduced from 0.012
        'partial_ingredients': {'herb_mushroom': 8.0, 'plant_food': 4.0},
    },
    'swamp_rot': {
        'name': 'Swamp Rot',
        'ingredients': ['herb_moss', 'herb_willow', 'water'],
        'window': 5,
        'cure_health': 30.0,
        'cure_sick': 55.0,
        'spread_rate': 0.012,   # reduced from 0.020
        'partial_ingredients': {'herb_moss': 8.0, 'herb_willow': 8.0, 'water': 4.0},
    },
}

# Public tag-set so the world can spawn matching herb resources
ALL_HERB_TAGS: list[str] = sorted({
    tag
    for rec in REMEDY_REGISTRY.values()
    for tag in rec['ingredients']
    if tag.startswith('herb_')
})


# ---------------------------------------------------------------------------
# Infection helper  (called from world / simulation)
# ---------------------------------------------------------------------------

def try_infect_agent(agent, disease_id: str) -> bool:
    """Attempt to infect agent with disease_id if it is not already infected."""
    rec = REMEDY_REGISTRY.get(disease_id)
    if rec is None:
        return False
    if getattr(agent, 'disease_id', None) is not None:
        return False          # already sick with something
    if random.random() < rec['spread_rate']:
        agent.disease_id = disease_id
        agent.sick = min(100.0, agent.sick + 12.0)  # softer initial hit (was 15)
        return True
    return False


# ---------------------------------------------------------------------------
# Cure evaluation  (called from agent.forage / agent.use_herb)
# ---------------------------------------------------------------------------

def evaluate_remedy(agent, consumed_tags: list[str]) -> float:
    """
    Two-tier healing system:
    1. FULL CURE: all ingredients consumed within window → big heal.
    2. PARTIAL CURE: any single known ingredient → minor sick reduction each use.
       This gives agents a fighting chance even before discovering the full recipe.
    Returns a health-bonus float (0 if nothing matched).
    """
    disease_id = getattr(agent, 'disease_id', None)
    if disease_id is None:
        return 0.0

    rec = REMEDY_REGISTRY[disease_id]
    required = set(rec['ingredients'])
    partial_map: dict = rec.get('partial_ingredients', {})

    # Build rolling window of consumed tags
    if not hasattr(agent, '_remedy_window'):
        agent._remedy_window = []
    agent._remedy_window.extend(consumed_tags)
    if not hasattr(agent, '_remedy_window_ticks'):
        agent._remedy_window_ticks = 0
    agent._remedy_window_ticks += 1
    window_limit = rec['window']
    max_per_tick = 4
    agent._remedy_window = agent._remedy_window[-(window_limit * max_per_tick):]

    # --- Tier 1: Full cure ---
    if required.issubset(set(agent._remedy_window)):
        agent.health = min(100.0, agent.health + rec['cure_health'])
        agent.sick = max(0.0, agent.sick - rec['cure_sick'])
        if agent.sick <= 0:
            agent.disease_id = None
        agent._remedy_window = []
        agent._remedy_window_ticks = 0
        return rec['cure_health'] / 25.0

    # --- Tier 2: Partial cure (any matching ingredient gives minor relief) ---
    partial_bonus = 0.0
    for tag in consumed_tags:
        if tag in partial_map:
            sick_reduction = partial_map[tag] * 0.25   # 25% of full partial value per use
            health_gain = sick_reduction * 0.3
            agent.sick = max(0.0, agent.sick - sick_reduction)
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
