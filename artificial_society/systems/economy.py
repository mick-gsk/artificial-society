import random


class EconomySystem:
    def __init__(self):
        self.trade_count = 0
        self.prices = {'wood': 1.0, 'stone': 1.0, 'fiber': 1.0}

    def maybe_trade(self, agent, agents):
        # Radius 2 statt identischer Position -- auf einem 60x40-Grid mit ~36 Agenten
        # war exakt gleiche Zelle so selten dass Handel faktisch nie vorkam.
        # Radius 2 entspricht realistischem Sichtkontakt.
        neighbors = [
            a for a in agents
            if a is not agent and a.alive
            and abs(a.pos[0] - agent.pos[0]) <= 2
            and abs(a.pos[1] - agent.pos[1]) <= 2
        ]
        for other in neighbors:
            if agent.trust.get(other.id, 0.0) < 0.1:
                continue
            for give_res, want_res in [('wood', 'stone'), ('stone', 'fiber'), ('fiber', 'wood')]:
                if agent.resources[give_res] > 1 and other.resources[want_res] > 1:
                    agent.resources[give_res] -= 1
                    other.resources[give_res] += 1
                    other.resources[want_res] -= 1
                    agent.resources[want_res] += 1
                    self.trade_count += 1
                    # Vertrauen steigt proportional zur Ressourcenknappheit des erhaltenen Guts
                    # (seltene Waren staerken Vertrauen staerker -- emergent, kein explizites Label)
                    trust_gain = 0.03 + 0.02 * self.prices[give_res]
                    agent.trust[other.id] = min(1.0, agent.trust.get(other.id, 0.0) + trust_gain)
                    other.trust[agent.id] = min(1.0, other.trust.get(agent.id, 0.0) + trust_gain)
                    break

    def update(self, agents, tribes):
        totals = {'wood': 0, 'stone': 0, 'fiber': 0}
        for a in agents:
            if a.alive:
                for r in totals:
                    totals[r] += a.resources[r]
        total_all = sum(totals.values()) or 1
        for r in self.prices:
            share = totals[r] / total_all
            self.prices[r] = max(0.5, min(3.0, 1.0 / (share + 0.1)))
