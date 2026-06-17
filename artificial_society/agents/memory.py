class EpisodicMemory:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.resource_memory = []
        self.danger_memory = []
        self.social_memory = []

    def _append(self, store, item):
        store.append(item)
        if len(store) > self.capacity:
            del store[0]

    def remember_resource(self, location, food, water, tick):
        self._append(self.resource_memory, {
            'location': tuple(location),
            'food': round(food, 2),
            'water': round(water, 2),
            'timestamp': tick,
        })

    def remember_danger(self, location, danger, tick):
        self._append(self.danger_memory, {
            'location': tuple(location),
            'danger': round(danger, 2),
            'timestamp': tick,
        })

    def remember_social(self, other_id, trust, helpful, tick):
        self._append(self.social_memory, {
            'agent_id': other_id,
            'trust': round(trust, 2),
            'helpful': helpful,
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

    def retrieval_features(self, x, y, tick, world_w=1, world_h=1):
        res = self.best_known_resource()
        dan = self.best_known_danger()
        soc = self.best_social_partner()
        if res:
            res_dx = (res['location'][0] - x) / max(1, world_w)
            res_dy = (res['location'][1] - y) / max(1, world_h)
            res_food = res['food'] / 100.0
            res_water = res['water'] / 100.0
            res_age = min(1.0, (tick - res['timestamp']) / 500.0)
        else:
            res_dx = res_dy = res_food = res_water = res_age = 0.0
        if dan:
            dan_dx = (dan['location'][0] - x) / max(1, world_w)
            dan_dy = (dan['location'][1] - y) / max(1, world_h)
            dan_val = dan['danger'] / 100.0
            dan_age = min(1.0, (tick - dan['timestamp']) / 500.0)
        else:
            dan_dx = dan_dy = dan_val = dan_age = 0.0
        if soc:
            soc_trust = (soc['trust'] + 1.0) * 0.5
            soc_helpful = 1.0 if soc['helpful'] else 0.0
            soc_age = min(1.0, (tick - soc['timestamp']) / 500.0)
        else:
            soc_trust = soc_helpful = soc_age = 0.0
        return [
            res_dx, res_dy, res_food, res_water, res_age,
            dan_dx, dan_dy, dan_val, dan_age,
            soc_trust, soc_helpful, soc_age,
        ]
