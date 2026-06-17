from artificial_society.environment.biomes import BIOME_BASE


DISEASE_POLLUTION_THRESHOLD = 36.0
MEAT_SPOIL_RATE = 0.04
CARCASS_DECAY = 0.14


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def initial_cell_state(biome):
    base = BIOME_BASE[biome]
    plant_food = float(base['food'] * (1.15 if biome in ('forest', 'grassland', 'swamp') else 0.65))
    meat_food = float(base['food'] * (0.35 if biome in ('mountain', 'forest', 'swamp') else 0.12))
    return {
        'food': float(base['food']),
        'plant_food': plant_food,
        'meat_food': meat_food,
        'water': float(base['water']),
        'temperature': float(base['temperature']),
        'danger': float(base['danger']),
        'soil_fertility': float(base['soil_fertility']),
        'pollution': 0.0,
        'usage_pressure': 0.0,
        'carrying_capacity': float(base['carrying_capacity']),
        'spoilage': 0.0,
        'carcasses': 0.0,
        'disease': 0.0,
        'moisture': float(base['water'] * 0.7),
        'ash': 0.0,
        'disturbance': 0.0,
        'structures': {'camp': 0.0, 'farm': 0.0, 'well': 0.0},
        'biome': biome,
        'tick': 0,
    }


def apply_consumption(cell, plant=0.0, meat=0.0, water=0.0):
    cell['plant_food'] = clamp(cell['plant_food'] - plant, 0.0, 160.0)
    cell['meat_food'] = clamp(cell['meat_food'] - meat, 0.0, 110.0)
    cell['water'] = clamp(cell['water'] - water, 0.0, 100.0)
    cell['food'] = clamp(cell['plant_food'] + cell['meat_food'] * 0.7, 0.0, 180.0)
    pressure = plant + meat * 1.3 + water * 0.7
    cell['usage_pressure'] = clamp(cell['usage_pressure'] + pressure * 0.85, 0.0, 100.0)
    cell['pollution'] = clamp(cell['pollution'] + meat * 0.14 + pressure * 0.05, 0.0, 100.0)
    cell['soil_fertility'] = clamp(cell['soil_fertility'] - pressure * 0.04, 0.0, 100.0)
    cell['disturbance'] = clamp(cell['disturbance'] + pressure * 0.25, 0.0, 100.0)


def add_carcass(cell, energy_value):
    cell['carcasses'] = clamp(cell['carcasses'] + energy_value, 0.0, 140.0)
    cell['meat_food'] = clamp(cell['meat_food'] + energy_value * 0.45, 0.0, 110.0)
    cell['spoilage'] = clamp(cell['spoilage'] + energy_value * 0.16, 0.0, 100.0)
    cell['pollution'] = clamp(cell['pollution'] + energy_value * 0.12, 0.0, 100.0)
    cell['danger'] = clamp(cell['danger'] + energy_value * 0.05, 0.0, 100.0)
    cell['disturbance'] = clamp(cell['disturbance'] + energy_value * 0.08, 0.0, 100.0)
    cell['food'] = clamp(cell['plant_food'] + cell['meat_food'] * 0.7, 0.0, 180.0)


def maybe_build_structure(cell, resources):
    built = None
    if resources['wood'] >= 2 and resources['fiber'] >= 1 and cell['structures']['camp'] < 1.0:
        resources['wood'] -= 2
        resources['fiber'] -= 1
        cell['structures']['camp'] = 1.0
        built = 'camp'
    elif resources['wood'] >= 2 and resources['stone'] >= 1 and cell['structures']['well'] < 1.0:
        resources['wood'] -= 2
        resources['stone'] -= 1
        cell['structures']['well'] = 1.0
        built = 'well'
    elif resources['wood'] >= 1 and resources['fiber'] >= 2 and cell['structures']['farm'] < 1.0:
        resources['wood'] -= 1
        resources['fiber'] -= 2
        cell['structures']['farm'] = 1.0
        built = 'farm'
    return built


def apply_event(cell, event_strength):
    drought = event_strength.get('drought', 0.0)
    storm = event_strength.get('storm', 0.0)
    fire = event_strength.get('fire', 0.0)
    blight = event_strength.get('blight', 0.0)
    disturbance = event_strength.get('disturbance', 0.0)
    cell['moisture'] = clamp(cell['moisture'] - 12.0 * drought + 8.0 * storm - 18.0 * fire, 0.0, 100.0)
    cell['water'] = clamp(cell['water'] - 0.35 * drought + 0.45 * storm - 0.1 * fire, 0.0, 100.0)
    cell['plant_food'] = clamp(cell['plant_food'] - 2.4 * drought - 5.8 * fire - 3.6 * blight, 0.0, 180.0)
    cell['meat_food'] = clamp(cell['meat_food'] - 1.8 * fire, 0.0, 110.0)
    cell['ash'] = clamp(cell['ash'] + 7.0 * fire - 0.3 * storm, 0.0, 100.0)
    cell['pollution'] = clamp(cell['pollution'] + 4.0 * fire + 1.6 * blight, 0.0, 100.0)
    cell['disease'] = clamp(cell['disease'] + 4.0 * blight + 1.2 * storm, 0.0, 100.0)
    cell['disturbance'] = clamp(cell['disturbance'] + 9.0 * disturbance, 0.0, 100.0)
    cell['food'] = clamp(cell['plant_food'] + cell['meat_food'] * 0.7, 0.0, 180.0)


def diffuse_step(cell, neighbor_avgs):
    cell['pollution'] = clamp(0.92 * cell['pollution'] + 0.08 * neighbor_avgs['pollution'], 0.0, 100.0)
    cell['disease'] = clamp(0.9 * cell['disease'] + 0.1 * neighbor_avgs['disease'], 0.0, 100.0)
    cell['moisture'] = clamp(0.88 * cell['moisture'] + 0.12 * neighbor_avgs['moisture'], 0.0, 100.0)
    cell['ash'] = clamp(0.9 * cell['ash'] + 0.1 * neighbor_avgs['ash'], 0.0, 100.0)
    cell['disturbance'] = clamp(0.9 * cell['disturbance'] + 0.1 * neighbor_avgs['disturbance'], 0.0, 100.0)


def regrow_cell(cell, biome, season_state, weather_state, tick, event_strength):
    season_food = season_state.get('food_factor', 1.0)
    rain = weather_state.get('rain_map', 0.0)
    temp_shift = weather_state.get('temperature_shift', 0.0)
    wind = weather_state.get('wind', 0.0)
    base = BIOME_BASE[biome]
    cell['tick'] = tick
    cell['temperature'] = clamp(base['temperature'] + temp_shift + season_state.get('temperature_shift', 0.0), -12, 52)

    usage = cell['usage_pressure']
    pollution = cell['pollution']
    fertility = cell['soil_fertility']
    capacity = cell['carrying_capacity']
    spoilage = cell['spoilage']
    moisture = cell['moisture']
    disturbance = cell['disturbance']
    camp_bonus = 8.0 if cell['structures']['camp'] else 0.0
    farm_bonus = 10.0 if cell['structures']['farm'] else 0.0
    well_bonus = 10.0 if cell['structures']['well'] else 0.0

    water_gain = 0.08 * rain + (0.12 if biome == 'swamp' else 0.03) + 0.03 * cell['structures']['well']
    fert_factor = 0.4 + fertility / 100.0
    moisture_factor = 0.35 + moisture / 100.0
    cap_factor = 0.35 + (capacity + farm_bonus) / 120.0
    stress_factor = max(0.12, 1.0 - pollution / 120.0 - usage / 180.0 - disturbance / 180.0)
    plant_gain = 0.05 * season_food * fert_factor * moisture_factor * cap_factor * stress_factor * (1.15 if biome == 'forest' else 1.0)
    plant_gain += 0.03 * cell['structures']['farm']
    meat_gain = 0.016 * season_food * cap_factor * max(0.2, 1.0 - pollution / 150.0) * (1.1 if biome in ('mountain', 'forest', 'swamp') else 0.7)

    if biome == 'desert':
        plant_gain *= 0.3
        meat_gain *= 0.22
        water_gain *= 0.25
    if biome == 'mountain':
        plant_gain *= 0.55
        meat_gain *= 1.18
    if biome == 'water':
        water_gain = 0.0
        plant_gain = 0.0
        meat_gain = 0.0

    decayed_meat = min(cell['meat_food'], spoilage * MEAT_SPOIL_RATE)
    cell['meat_food'] = clamp(cell['meat_food'] - decayed_meat, 0.0, max(15.0, capacity))
    cell['carcasses'] = clamp(cell['carcasses'] - CARCASS_DECAY, 0.0, 140.0)
    cell['spoilage'] = clamp(cell['spoilage'] + decayed_meat * 0.85 - 0.05 * rain - 0.04 * cell['structures']['camp'], 0.0, 100.0)
    cell['disease'] = clamp(cell['disease'] + max(0.0, pollution - DISEASE_POLLUTION_THRESHOLD) * 0.025 + cell['spoilage'] * 0.03 + cell['carcasses'] * 0.012 - 0.05 * rain - 0.03 * cell['structures']['well'], 0.0, 100.0)
    cell['plant_food'] = clamp(cell['plant_food'] + plant_gain - 0.012 * wind - 0.02 * cell['ash'], 0.0, max(20.0, capacity * 1.45 + farm_bonus))
    cell['meat_food'] = clamp(cell['meat_food'] + meat_gain, 0.0, max(15.0, capacity))
    cell['water'] = clamp(cell['water'] + water_gain + 0.05 * cell['structures']['well'], 0.0, 100.0)
    cell['moisture'] = clamp(cell['moisture'] + 0.2 * cell['water'] / 100.0 + 0.35 * rain + 0.1 * well_bonus - 0.03 * abs(cell['temperature'] - 20), 0.0, 100.0)
    cell['soil_fertility'] = clamp(cell['soil_fertility'] + 0.05 * rain - 0.03 * pollution + 0.015 * max(0.0, 60 - usage) + 0.03 * cell['carcasses'] + 0.04 * cell['ash'], 0.0, 100.0)
    cell['pollution'] = clamp(cell['pollution'] - 0.03 * rain - 0.015 * max(0.0, fertility - 25) + 0.02 * cell['spoilage'] - 0.02 * cell['structures']['camp'], 0.0, 100.0)
    cell['ash'] = clamp(cell['ash'] - 0.06 * rain - 0.03 * moisture / 100.0, 0.0, 100.0)
    cell['usage_pressure'] = clamp(cell['usage_pressure'] * 0.91, 0.0, 100.0)
    cell['disturbance'] = clamp(cell['disturbance'] * 0.94, 0.0, 100.0)
    apply_event(cell, event_strength)
    cell['food'] = clamp(cell['plant_food'] + cell['meat_food'] * 0.7, 0.0, 180.0)
    cell['danger'] = clamp(base['danger'] + weather_state.get('storm_risk', 0.0) * 20 + abs(cell['temperature'] - 20) * 0.5 + cell['pollution'] * 0.16 + cell['disease'] * 0.12 + cell['disturbance'] * 0.18, 0.0, 100.0)
