TECH_TREE = {
    'axe':     {'requires': [],          'description': 'Basic tool; improves foraging and collection'},
    'shelter': {'requires': ['axe'],      'description': 'Reduces environmental health loss by 40%'},
    'pottery': {'requires': ['axe'],      'description': 'Reduces water-loss rate by 30%'},
    'medicine':{'requires': ['shelter'],  'description': 'Passively reduces sick level by +0.05/tick'},
}

# Bonus constants applied in agent code via has_tech()
TECH_EFFECTS = {
    'shelter': {'health_loss_mult': 0.60},
    'pottery': {'hydration_drain_mult': 0.70},
    'medicine': {'sick_reduction_bonus': 0.05},
}


class TechnologySystem:
    def __init__(self):
        self.discoveries = {}   # tribe_id -> set of tech names

    def discover(self, tribe_id, tech_name):
        tribe_id = tribe_id or 0
        existing = self.discoveries.setdefault(tribe_id, set())
        if tech_name in existing:
            return False
        req = TECH_TREE.get(tech_name, {}).get('requires', [])
        if all(r in existing for r in req):
            existing.add(tech_name)
            return True
        return False

    def tribe_technologies(self, tribe_id):
        return sorted(self.discoveries.get(tribe_id or 0, set()))

    def has_tech(self, tribe_id, tech_name) -> bool:
        return tech_name in self.discoveries.get(tribe_id or 0, set())

    def update(self, agents, tribes):
        for agent in agents:
            if not agent.alive:
                continue
            if agent.tool == 'axe':
                self.discover(agent.tribe_id, 'axe')
            # Unlock shelter when camp is built on agent's cell
            # (world cell access not available here; agent carries last_action_mode)
            if agent.last_action_mode == 'build:camp' and self.has_tech(agent.tribe_id, 'axe'):
                self.discover(agent.tribe_id, 'shelter')
            if agent.last_action_mode == 'build:well' and self.has_tech(agent.tribe_id, 'axe'):
                self.discover(agent.tribe_id, 'pottery')
            if self.has_tech(agent.tribe_id, 'shelter'):
                self.discover(agent.tribe_id, 'medicine')
