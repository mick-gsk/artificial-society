"""
Herb resource layer.

Defines which biomes can spawn which herbs, their rarity, and
how agents consume them (tagging the tag into consumed_tags).

Herb tags must match those listed in remedy.REMEDY_REGISTRY ingredients.
"""

import random

# herb_tag -> {biomes where it can appear, base_spawn_chance per tick per cell,
#              max_amount, regrow_rate}
HERB_DEFINITIONS: dict[str, dict] = {
    'herb_willow': {
        'biomes': ('forest', 'grassland', 'swamp'),
        'spawn_chance': 0.012,
        'max_amount': 6.0,
        'regrow_rate': 0.08,
        'description': 'Willow bark – bitter, found near water and trees',
    },
    'herb_garlic': {
        'biomes': ('grassland', 'forest'),
        'spawn_chance': 0.010,
        'max_amount': 5.0,
        'regrow_rate': 0.07,
        'description': 'Wild garlic – pungent bulb of open meadows',
    },
    'herb_elderberry': {
        'biomes': ('forest', 'grassland'),
        'spawn_chance': 0.008,
        'max_amount': 4.0,
        'regrow_rate': 0.06,
        'description': 'Elderberry – dark clusters on forest edges',
    },
    'herb_mushroom': {
        'biomes': ('forest', 'swamp'),
        'spawn_chance': 0.014,
        'max_amount': 7.0,
        'regrow_rate': 0.10,
        'description': 'Forest mushroom – grows on dead wood and damp soil',
    },
    'herb_moss': {
        'biomes': ('swamp', 'forest'),
        'spawn_chance': 0.016,
        'max_amount': 8.0,
        'regrow_rate': 0.12,
        'description': 'Bog moss – dense, water-soaked ground cover',
    },
}


def regrow_herbs(cell: dict, biome: str):
    """Regenerate herb amounts each tick.  Cell must have a 'herbs' sub-dict."""
    herbs = cell.setdefault('herbs', {})
    for tag, defn in HERB_DEFINITIONS.items():
        if biome not in defn['biomes']:
            continue
        current = herbs.get(tag, 0.0)
        # Spontaneous spawning if absent
        if current == 0.0 and random.random() < defn['spawn_chance']:
            herbs[tag] = random.uniform(0.5, 2.0)
        elif current > 0.0:
            herbs[tag] = min(defn['max_amount'], current + defn['regrow_rate'])


def collect_herb(cell: dict, tag: str, amount: float = 1.0) -> float:
    """Remove up to `amount` of herb from cell; returns how much was taken."""
    herbs = cell.get('herbs', {})
    available = herbs.get(tag, 0.0)
    taken = min(available, amount)
    herbs[tag] = available - taken
    cell['herbs'] = herbs
    return taken


def available_herbs(cell: dict) -> list[str]:
    """Return list of herb tags that have a non-zero amount in this cell."""
    return [tag for tag, amt in cell.get('herbs', {}).items() if amt > 0.0]
