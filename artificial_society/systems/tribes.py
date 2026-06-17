import random
from artificial_society.agents.culture import CultureProfile


class TribeSystem:
    def __init__(self):
        self.tribes = {}
        self.next_id = 1
        self.palette = [(240, 120, 90), (90, 180, 255), (200, 220, 80), (180, 100, 220), (110, 220, 180)]

    def create_tribe(self, agent_ids):
        tid = self.next_id
        self.next_id += 1
        self.tribes[tid] = {'members': set(agent_ids), 'culture': CultureProfile()}
        return tid

    def color_for(self, tribe_id):
        if tribe_id is None:
            return (220, 220, 220)
        return self.palette[(tribe_id - 1) % len(self.palette)]

    def consider_join(self, agent, nearby):
        if agent.tribe_id is not None:
            return
        trusted = [a for a in nearby if agent.trust.get(a.id, 0.0) > 0.25]
        same = [a for a in trusted if a.tribe_id is not None]
        if same:
            agent.tribe_id = same[0].tribe_id
            self.tribes[agent.tribe_id]['members'].add(agent.id)
        elif len(trusted) >= 2 and agent.genes['sociality'] > 0.45:
            ids = [agent.id] + [a.id for a in trusted[:2]]
            tid = self.create_tribe(ids)
            agent.tribe_id = tid
            for other in trusted[:2]:
                other.tribe_id = tid

    def update_membership(self, agents):
        by_id = {a.id: a for a in agents}
        for tid, data in list(self.tribes.items()):
            data['members'] = {aid for aid in data['members'] if aid in by_id and by_id[aid].alive}
            members = [by_id[aid] for aid in data['members'] if aid in by_id]
            data['culture'].learn_from_members(members)

    def cleanup(self, agents):
        alive_ids = {a.id for a in agents if a.alive}
        for tid in list(self.tribes.keys()):
            self.tribes[tid]['members'] &= alive_ids
            if not self.tribes[tid]['members']:
                del self.tribes[tid]

    def count(self):
        return len(self.tribes)
