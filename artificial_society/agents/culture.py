class CultureProfile:
    def __init__(self):
        self.shared_resource_sites = []
        self.norm_cooperation = 0.0
        self.norm_aggression = 0.0

    def learn_from_members(self, agents):
        if not agents:
            return
        self.norm_cooperation = sum(a.genes['cooperation'] for a in agents) / len(agents)
        self.norm_aggression = sum(a.genes['aggression'] for a in agents) / len(agents)
        sites = []
        for a in agents:
            best = a.memory.best_known_resource()
            if best:
                sites.append(best['location'])
        self.shared_resource_sites = sites[-12:]
