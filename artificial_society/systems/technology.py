class TechnologySystem:
    def __init__(self):
        self.discoveries = {}

    def discover(self, tribe_id, tech_name):
        tribe_id = tribe_id or 0
        self.discoveries.setdefault(tribe_id, set()).add(tech_name)

    def tribe_technologies(self, tribe_id):
        return sorted(self.discoveries.get(tribe_id or 0, set()))

    def update(self, agents, tribes):
        for agent in agents:
            if agent.tool == 'axe':
                self.discover(agent.tribe_id, 'axe')
