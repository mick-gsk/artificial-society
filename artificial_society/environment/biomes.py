import math
import random

BIOMES = ['forest', 'grassland', 'desert', 'mountain', 'swamp', 'water']

BIOME_BASE = {
    'forest':    {'food': 70, 'water': 40, 'temperature': 18, 'danger': 25, 'soil_fertility': 78, 'carrying_capacity': 95, 'move_cost': 1.05},
    'grassland': {'food': 55, 'water': 45, 'temperature': 21, 'danger': 18, 'soil_fertility': 66, 'carrying_capacity': 82, 'move_cost': 0.92},
    'desert':    {'food': 18, 'water': 12, 'temperature': 33, 'danger': 42, 'soil_fertility': 18, 'carrying_capacity': 28, 'move_cost': 1.35},
    'mountain':  {'food': 22, 'water': 34, 'temperature':  8, 'danger': 34, 'soil_fertility': 28, 'carrying_capacity': 36, 'move_cost': 1.25},
    'swamp':     {'food': 48, 'water': 68, 'temperature': 24, 'danger': 38, 'soil_fertility': 58, 'carrying_capacity': 70, 'move_cost': 1.30},
    'water':     {'food':  0, 'water':100, 'temperature': 15, 'danger': 52, 'soil_fertility':  0, 'carrying_capacity':  0, 'move_cost': 2.0},
}

# Realistic base colors per biome (satellite-map inspired)
BIOME_BASE_COLOR = {
    'forest':    (34,  88,  42),   # deep conifer green
    'grassland': (88, 148,  58),   # warm meadow green
    'desert':    (196,168,  96),   # sandy tan
    'mountain':  (118,110, 102),   # grey rock
    'swamp':     (52,  84,  60),   # dark murky green
    'water':     (36,  82, 148),   # deep ocean blue
}

# Accent tones for texture details
BIOME_DETAIL_COLOR = {
    'forest':    (22,  60,  28),   # dark shadow under canopy
    'grassland': (110,170,  68),   # bright grass highlight
    'desert':    (214,190, 120),   # light sand crest
    'mountain':  (160,155, 148),   # lighter rock face
    'swamp':     (70, 105,  72),   # lighter murk
    'water':     (55, 110, 175),   # lighter wave
}


def biome_color(biome, cell):
    """Return the base terrain color for a cell, modulated by its resource levels."""
    base = list(BIOME_BASE_COLOR.get(biome, (100, 100, 100)))
    pollution = cell.get('pollution', 0)

    if biome == 'forest':
        # Greener when more plant food; browner under stress
        green_bonus = int(cell['plant_food'] * 0.25)
        base[1] = min(130, base[1] + green_bonus)
        base[0] = max(15, base[0] - int(pollution * 0.3))

    elif biome == 'grassland':
        # Yellow-green gradient based on soil fertility
        fertility = cell.get('soil_fertility', 66)
        base[0] = min(140, 75 + int(fertility * 0.35))
        base[1] = min(175, 130 + int(cell['plant_food'] * 0.25))
        base[2] = max(30, 50 - int(pollution * 0.4))

    elif biome == 'desert':
        # Warmer orange when hotter; slightly green near water
        base[0] = min(220, 185 + int(cell['temperature'] * 0.5))
        base[1] = max(130, 155 - int(pollution * 0.3))
        if cell['water'] > 20:
            base[1] = min(175, base[1] + 15)

    elif biome == 'mountain':
        # Snow-white tint at cold temperatures
        snow = max(0, int((10 - cell['temperature']) * 5))
        base = [min(255, b + snow) for b in base]

    elif biome == 'swamp':
        # Murkier green; more blue when water-logged
        base[1] = min(115, 80 + int(cell['water'] * 0.28))
        base[2] = min(80, 55 + int(cell['water'] * 0.18))

    elif biome == 'water':
        # Deeper blue in open water; greener near shore (shallow)
        depth_tone = max(28, 36 - int(cell.get('moisture', 50) * 0.05))
        base[0] = depth_tone
        base[1] = min(100, 82 + int(cell.get('moisture', 50) * 0.1))

    # Pollution darkens everything
    if pollution > 5:
        grey_shift = int(pollution * 0.25)
        base = [max(0, c - grey_shift) for c in base]

    return tuple(base)


def noise(x, y):
    return math.sin(x * 0.17) + math.cos(y * 0.21) + 0.45 * math.sin((x + y) * 0.09)


def generate_biome_grid(width, height):
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            nx = x / max(1, width - 1)
            ny = y / max(1, height - 1)
            v = noise(x, y) + 0.7 * (1.0 - math.hypot(nx - 0.5, ny - 0.5))
            if v < -1.0:
                row.append('water')
            elif v < -0.15:
                row.append('mountain')
            elif v < 0.35:
                row.append('grassland')
            elif v < 0.95:
                row.append('forest')
            elif v < 1.45:
                row.append('swamp')
            else:
                row.append('desert')
        grid.append(row)
    return grid
