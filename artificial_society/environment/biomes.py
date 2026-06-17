import math

BIOMES = ['forest', 'grassland', 'desert', 'mountain', 'swamp', 'water']

BIOME_BASE = {
    'forest': {'food': 70, 'water': 40, 'temperature': 18, 'danger': 25, 'soil_fertility': 78, 'carrying_capacity': 95, 'move_cost': 1.05},
    'grassland': {'food': 55, 'water': 45, 'temperature': 21, 'danger': 18, 'soil_fertility': 66, 'carrying_capacity': 82, 'move_cost': 0.92},
    'desert': {'food': 18, 'water': 12, 'temperature': 33, 'danger': 42, 'soil_fertility': 18, 'carrying_capacity': 28, 'move_cost': 1.35},
    'mountain': {'food': 22, 'water': 34, 'temperature': 8, 'danger': 34, 'soil_fertility': 28, 'carrying_capacity': 36, 'move_cost': 1.25},
    'swamp': {'food': 48, 'water': 68, 'temperature': 24, 'danger': 38, 'soil_fertility': 58, 'carrying_capacity': 70, 'move_cost': 1.30},
    'water': {'food': 0, 'water': 100, 'temperature': 15, 'danger': 52, 'soil_fertility': 0, 'carrying_capacity': 0, 'move_cost': 2.0},
}


def biome_color(biome, cell):
    pollution = int(cell.get('pollution', 0) * 0.8)
    pressure = int(cell.get('usage_pressure', 0) * 0.4)
    if biome == 'forest':
        return (max(12, 25 - pollution // 5), max(35, 90 + int(cell['plant_food'] * 0.55) - pressure), max(18, 35 - pollution // 8))
    if biome == 'grassland':
        return (90 + int(cell['soil_fertility'] * 0.2), max(80, 125 + int(cell['plant_food'] * 0.35) - pressure), 60)
    if biome == 'desert':
        return (180 + int(cell['temperature'] * 0.4), 160 - pollution // 8, 90)
    if biome == 'mountain':
        tone = 95 + int(cell['meat_food'] * 0.15)
        return (tone, tone, 110 - pollution // 10)
    if biome == 'swamp':
        return (85 - pollution // 10, max(55, 95 + int(cell['water'] * 0.35) - pressure), 60)
    return (28, 76, 150)


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
