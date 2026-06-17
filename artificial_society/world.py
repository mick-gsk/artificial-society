import math
import random
from artificial_society.environment.biomes import BIOME_BASE, biome_color, generate_biome_grid
from artificial_society.environment.resources import clamp, diffuse_step, initial_cell_state, regrow_cell
from artificial_society.environment.herbs import regrow_herbs


class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.biomes = generate_biome_grid(width, height)
        self.cells = [[initial_cell_state(self.biomes[y][x]) for x in range(width)] for y in range(height)]
        self.land_positions = [(x, y) for y in range(height) for x in range(width) if self.biomes[y][x] != 'water']
        self.active_events = []

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x, y):
        x = clamp(x, 0, self.width - 1)
        y = clamp(y, 0, self.height - 1)
        return self.cells[y][x]

    def get_biome(self, x, y):
        x = clamp(x, 0, self.width - 1)
        y = clamp(y, 0, self.height - 1)
        return self.biomes[y][x]

    def biome_move_cost(self, x, y):
        biome = self.get_biome(x, y)
        return BIOME_BASE[biome]['move_cost']

    def random_land_position(self):
        return random.choice(self.land_positions)

    def find_free_neighbor(self, pos):
        px, py = pos
        candidates = [(px + dx, py + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        random.shuffle(candidates)
        for x, y in candidates:
            if self.in_bounds(x, y) and self.get_biome(x, y) != 'water':
                return x, y
        return None, None

    def neighbors(self, x, y, radius=1):
        out = []
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if self.in_bounds(xx, yy):
                    out.append((xx, yy, self.cells[yy][xx], self.biomes[yy][xx]))
        return out

    def spawn_disturbance(self, tick, season_state, weather_state):
        season_name = season_state.get('name', '').lower()
        weights = [
            ('drought', 1.4 if season_name == 'summer' else 1.0),
            ('storm', 1.3 + weather_state.get('storm_risk', 0.0)),
            ('fire', 1.2 if season_name == 'summer' else 0.7),
            ('blight', 1.0),
        ]
        kinds = [k for k, _ in weights]
        probs = [w for _, w in weights]
        total = sum(probs)
        r = random.random() * total
        acc = 0.0
        kind = kinds[0]
        for k, p in zip(kinds, probs):
            acc += p
            if r <= acc:
                kind = k
                break
        x, y = self.random_land_position()
        self.active_events.append({
            'kind': kind,
            'x': x,
            'y': y,
            'radius': random.randint(4, 9),
            'intensity': random.uniform(0.55, 1.1),
            'ttl': random.randint(40, 100),
        })

    def update_events(self, tick, season_state, weather_state):
        if tick % 55 == 0 or (random.random() < 0.015 and len(self.active_events) < 5):
            self.spawn_disturbance(tick, season_state, weather_state)
        kept = []
        for event in self.active_events:
            event['ttl'] -= 1
            if event['kind'] == 'storm':
                event['radius'] = min(max(3, event['radius'] + random.choice([-1, 0, 1])), 10)
            elif random.random() < 0.12:
                event['x'] = int(clamp(event['x'] + random.choice([-1, 0, 1]), 0, self.width - 1))
                event['y'] = int(clamp(event['y'] + random.choice([-1, 0, 1]), 0, self.height - 1))
            event['intensity'] *= 0.992
            if event['ttl'] > 0 and event['intensity'] > 0.15:
                kept.append(event)
        self.active_events = kept

    def event_field(self, x, y):
        out = {'drought': 0.0, 'storm': 0.0, 'fire': 0.0, 'blight': 0.0, 'disturbance': 0.0}
        for event in self.active_events:
            d = math.hypot(x - event['x'], y - event['y'])
            if d > event['radius']:
                continue
            strength = max(0.0, 1.0 - d / max(1.0, event['radius'])) * event['intensity']
            out[event['kind']] += strength
            out['disturbance'] += strength
        return out

    def diffuse_fields(self):
        avgs = [[None for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                cells = [c for _, _, c, _ in self.neighbors(x, y, 1)]
                n = len(cells)
                avgs[y][x] = {
                    'pollution': sum(c['pollution'] for c in cells) / n,
                    'disease': sum(c['disease'] for c in cells) / n,
                    'moisture': sum(c['moisture'] for c in cells) / n,
                    'ash': sum(c['ash'] for c in cells) / n,
                    'disturbance': sum(c['disturbance'] for c in cells) / n,
                }
        for y in range(self.height):
            for x in range(self.width):
                diffuse_step(self.cells[y][x], avgs[y][x])

    def regional_means(self):
        total_food = total_water = total_pollution = total_fertility = total_capacity = total_disease = total_disturbance = 0.0
        n = self.width * self.height
        for row in self.cells:
            for cell in row:
                total_food += cell['food']
                total_water += cell['water']
                total_pollution += cell['pollution']
                total_fertility += cell['soil_fertility']
                total_capacity += cell['carrying_capacity']
                total_disease += cell['disease']
                total_disturbance += cell['disturbance']
        return {
            'food': total_food / n,
            'water': total_water / n,
            'pollution': total_pollution / n,
            'soil_fertility': total_fertility / n,
            'carrying_capacity': total_capacity / n,
            'disease': total_disease / n,
            'disturbance': total_disturbance / n,
            'events': len(self.active_events),
        }

    def hotspots(self, min_pollution=35, min_disease=30):
        out = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                if cell['pollution'] >= min_pollution or cell['disease'] >= min_disease or cell['disturbance'] >= 30 or any(cell['structures'].values()):
                    out.append((x, y, cell))
        return out

    def update_environment(self, season_state, weather_state, tick):
        self.update_events(tick, season_state, weather_state)
        for y in range(self.height):
            for x in range(self.width):
                biome = self.biomes[y][x]
                cell = self.cells[y][x]
                regrow_cell(cell, biome, season_state, weather_state, tick, self.event_field(x, y))
                # FIX 2: regrow herbs on every land cell every tick
                if biome != 'water':
                    regrow_herbs(cell, biome)
        self.diffuse_fields()

    def color_at(self, x, y):
        cell = self.cells[y][x]
        return biome_color(self.biomes[y][x], cell)
