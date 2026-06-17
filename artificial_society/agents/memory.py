# EpisodicMemory – erweitert um:
# - Territorium-Tracking und Reward-Bridge
# - Jahreszeiten-Kontext in Erinnerungen (saisonales Heimrevierverhalten)
# - Korrektur: position_reward prueft ob Ressource noch vorhanden ist
# - Korrektur: memory_capacity skaliert korrekt pro Speichertyp (nicht geteilt)

MEMORY_TERRITORY_RADIUS   = 3
MEMORY_REVISIT_REWARD     = 0.12
MEMORY_FRESH_TICKS        = 300
MEMORY_DANGER_PENALTY     = 0.18
# Saisonale Toleranz: Erinnerungen aus anderer Jahreszeit werden abgediscontet
SEASON_DISCOUNT           = 0.45


class EpisodicMemory:
    def __init__(self, capacity=10):
        # Jeder Speichertyp bekommt seine eigene Kapazitaet -- memory_capacity-Gen
        # skaliert JEDEN Typ separat statt alle zu teilen.
        # Vorher: capacity=10 -> 10 fuer alle drei = effektiv 30 total
        # Jetzt:  capacity=10 -> 10 pro Typ = konsistent mit dem Gen-Intent
        self.capacity        = capacity
        self.resource_memory = []  # max: capacity Eintraege
        self.danger_memory   = []  # max: capacity Eintraege
        self.social_memory   = []  # max: capacity Eintraege

    def _append(self, store, item):
        store.append(item)
        if len(store) > self.capacity:
            del store[0]

    def remember_resource(self, location, food, water, tick, season_id=None):
        self._append(self.resource_memory, {
            'location':  tuple(location),
            'food':      round(food, 2),
            'water':     round(water, 2),
            'timestamp': tick,
            'season_id': season_id,  # z.B. 'summer', 'winter' -- NEU
        })

    def remember_danger(self, location, danger, tick, season_id=None):
        self._append(self.danger_memory, {
            'location':  tuple(location),
            'danger':    round(danger, 2),
            'timestamp': tick,
            'season_id': season_id,
        })

    def remember_social(self, other_id, trust, helpful, tick):
        self._append(self.social_memory, {
            'agent_id':  other_id,
            'trust':     round(trust, 2),
            'helpful':   helpful,
            'timestamp': tick,
        })

    def best_known_resource(self):
        if not self.resource_memory:
            return None
        return max(self.resource_memory, key=lambda m: m['food'] + 0.8 * m['water'])

    def best_known_danger(self):
        if not self.danger_memory:
            return None
        return max(self.danger_memory, key=lambda m: m['danger'])

    def best_social_partner(self):
        if not self.social_memory:
            return None
        return max(self.social_memory, key=lambda m: m['trust'])

    def position_reward(self, pos: tuple, tick: int, world=None, current_season: str = None) -> float:
        """
        Reward-Bridge: Bonus fuer Revisit bekannter Ressourcen, Penalty fuer Gefahrenzonen.

        NEU:
        - Prueft ob die Ressource in der Zelle noch wirklich vorhanden ist (world-Parameter).
          Wenn leer -> kein Bonus (verhindert leere Revisits nach Drought/Ueberfischung).
        - Saisonaler Discount: Erinnerungen aus anderer Jahreszeit werden abgediscontet
          (SEASON_DISCOUNT). Agent lernt dass Sommerressourcen im Winter nicht da sind.
        """
        reward = 0.0
        px, py = pos

        # Ressourcen-Bonus
        for mem in self.resource_memory:
            if mem['food'] < 20.0 and mem['water'] < 20.0:
                continue
            age = tick - mem['timestamp']
            if age > MEMORY_FRESH_TICKS:
                continue
            lx, ly = mem['location']
            dist = abs(px - lx) + abs(py - ly)
            if dist <= MEMORY_TERRITORY_RADIUS:
                # Pruefe ob Ressource noch da ist
                if world is not None:
                    cell = world.get_cell(lx, ly)
                    if cell['food'] < 5.0 and cell['water'] < 5.0:
                        continue  # Leer -- kein Bonus fuer leere Erinnerung
                freshness = max(0.0, 1.0 - age / MEMORY_FRESH_TICKS)
                quality   = min(1.0, (mem['food'] + mem['water']) / 120.0)
                # Saisonaler Discount wenn Erinnerung aus anderer Saison
                season_match = 1.0
                if current_season and mem.get('season_id') and mem['season_id'] != current_season:
                    season_match = SEASON_DISCOUNT
                reward += MEMORY_REVISIT_REWARD * freshness * quality * season_match

        # Gefahren-Penalty
        for mem in self.danger_memory:
            if mem['danger'] < 40.0:
                continue
            age = tick - mem['timestamp']
            if age > MEMORY_FRESH_TICKS:
                continue
            lx, ly = mem['location']
            dist = abs(px - lx) + abs(py - ly)
            if dist <= MEMORY_TERRITORY_RADIUS:
                freshness = max(0.0, 1.0 - age / MEMORY_FRESH_TICKS)
                season_match = 1.0
                if current_season and mem.get('season_id') and mem['season_id'] != current_season:
                    season_match = SEASON_DISCOUNT
                reward -= MEMORY_DANGER_PENALTY * freshness * season_match

        return reward

    def retrieval_features(self, x, y, tick, world_w=1, world_h=1):
        res = self.best_known_resource()
        dan = self.best_known_danger()
        soc = self.best_social_partner()
        if res:
            res_dx    = (res['location'][0] - x) / max(1, world_w)
            res_dy    = (res['location'][1] - y) / max(1, world_h)
            res_food  = res['food']  / 100.0
            res_water = res['water'] / 100.0
            res_age   = min(1.0, (tick - res['timestamp']) / 500.0)
        else:
            res_dx = res_dy = res_food = res_water = res_age = 0.0
        if dan:
            dan_dx  = (dan['location'][0] - x) / max(1, world_w)
            dan_dy  = (dan['location'][1] - y) / max(1, world_h)
            dan_val = dan['danger'] / 100.0
            dan_age = min(1.0, (tick - dan['timestamp']) / 500.0)
        else:
            dan_dx = dan_dy = dan_val = dan_age = 0.0
        if soc:
            soc_trust   = (soc['trust'] + 1.0) * 0.5
            soc_helpful = 1.0 if soc['helpful'] else 0.0
            soc_age     = min(1.0, (tick - soc['timestamp']) / 500.0)
        else:
            soc_trust = soc_helpful = soc_age = 0.0
        return [
            res_dx, res_dy, res_food, res_water, res_age,
            dan_dx, dan_dy, dan_val, dan_age,
            soc_trust, soc_helpful, soc_age,
        ]
