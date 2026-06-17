import random


class EconomySystem:
    def __init__(self):
        self.trade_count = 0

    def maybe_trade(self, agent, agents):
        neighbors = [a for a in agents if a is not agent and a.alive and a.pos == agent.pos]
        for other in neighbors:
            if agent.trust.get(other.id, 0.0) < 0.1:
                continue
            if agent.resources['wood'] > 1 and other.resources['stone'] > 1:
                agent.resources['wood'] -= 1
                other.resources['wood'] += 1
                other.resources['stone'] -= 1
                agent.resources['stone'] += 1
                self.trade_count += 1
                agent.trust[other.id] = min(1.0, agent.trust.get(other.id, 0.0) + 0.05)
                other.trust[agent.id] = min(1.0, other.trust.get(agent.id, 0.0) + 0.05)
                break

    def update(self, agents, tribes):
        return None
