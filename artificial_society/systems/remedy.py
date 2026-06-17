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
# Structure:
#   disease_id -> {
#       'name'        : human-readable label
#       'ingredients' : list of resource tags that MUST all be consumed in one
#                       tick (or within 'window' ticks) to trigger a cure
#       'window'      : how many consecutive ticks the ingredients may spread
#       'cure_health' : health restored on cure
#       'cure_sick'   : sick-level removed on cure
#       'spread_rate' : base probability per tick of infecting a neighbour
#   }
# ---------------------------------------------------------------------------
REMEDY_REGISTRY: dict[str, dict] = {
    'fever': {
        'name': 'Fever',
        'ingredients': ['herb_willow', 'water'],
        'window': 3,
        'cure_health': 25.0,
        'cure_sick': 40.0,
        'spread_rate': 0.04,
    },
    'plague': {
        'name': 'Plague',
        'ingredients': ['herb_garlic', 'herb_elderberry', 'meat'],
        'window': 2,
        'cure_health': 35.0,
        'cure_sick': 60.0,
        'spread_rate': 0.07,
    },
    'blight_sickness': {
        'name': 'Blight Sickness',
        'ingredients': ['herb_mushroom', 'plant_food'],
        'window': 4,
        'cure_health': 20.0,
        'cure_sick': 30.0,
        'spread_rate': 0.03,
    },
    'swamp_rot': {
        'name': 'Swamp Rot',
        'ingredients': ['herb_moss', 'herb_willow', 'water'],
        'window': 3,
        'cure_health': 30.0,
        'cure_sick': 50.0,
        'spread_rate': 0.05,
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
        agent.sick = min(100.0, agent.sick + 20.0)
        return True
    return False


# ---------------------------------------------------------------------------
# Cure evaluation  (called from agent.forage / agent.use_herb)
# ---------------------------------------------------------------------------

def evaluate_remedy(agent, consumed_tags: list[str]) -> float:
    """
    Check whether the recently consumed ingredient set triggers a cure for the
    agent's current disease.  Returns a health-bonus float (0 if no cure).

    Agents do NOT call this directly with knowledge of what is required –
    they simply pass what they happened to consume, and the system resolves it.
    """
    disease_id = getattr(agent, 'disease_id', None)
    if disease_id is None:
        return 0.0

    rec = REMEDY_REGISTRY[disease_id]
    required = set(rec['ingredients'])

    # Build rolling window of consumed tags
    if not hasattr(agent, '_remedy_window'):
        agent._remedy_window = []
    agent._remedy_window.extend(consumed_tags)
    # Keep only the last `window` ticks worth (we store per-tick lists)
    if not hasattr(agent, '_remedy_window_ticks'):
        agent._remedy_window_ticks = 0
    agent._remedy_window_ticks += 1
    window_limit = rec['window']
    # Trim old entries – crude: keep only last window_limit * max_tags_per_tick
    max_per_tick = 4
    agent._remedy_window = agent._remedy_window[-(window_limit * max_per_tick):]

    if required.issubset(set(agent._remedy_window)):
        # Cure triggered!
        agent.health = min(100.0, agent.health + rec['cure_health'])
        agent.sick = max(0.0, agent.sick - rec['cure_sick'])
        if agent.sick <= 0:
            agent.disease_id = None
        agent._remedy_window = []           # reset window after cure
        agent._remedy_window_ticks = 0
        # Return a positive reward signal so the brain can learn the association
        return rec['cure_health'] / 25.0    # normalised ~0-1.4 range

    return 0.0


# ---------------------------------------------------------------------------
# Social knowledge sharing  (called from communication system)
# ---------------------------------------------------------------------------

def share_remedy_knowledge(sender, receiver) -> bool:
    """
    An agent that has discovered a cure (has entries in remedy_knowledge) can
    share one piece of that knowledge with a nearby agent.
    Returns True if something was shared.
    """
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
    """
    Called internally when a cure fires so the agent can remember which
    ingredients were in the window (partial knowledge – only what was consumed).
    """
    if not hasattr(agent, 'remedy_knowledge'):
        agent.remedy_knowledge = {}
    if disease_id not in agent.remedy_knowledge:
        agent.remedy_knowledge[disease_id] = set()
    agent.remedy_knowledge[disease_id].update(ingredients_used)
