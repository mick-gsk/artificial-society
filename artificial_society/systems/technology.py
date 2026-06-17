"""
TechnologySystem — Emergent Only
---------------------------------
The old hardcoded tech tree (axe, shelter, pottery, medicine) is REMOVED.
Technology now emerges from what agents have actually discovered via
causal experimentation and cultural transmission.

This module tracks WHAT capabilities have spread through the population
without assigning any predefined recipes or names.
"""

from artificial_society.systems.culture import CultureTracker


class TechnologySystem:
    def __init__(self):
        self.culture_tracker = CultureTracker()
        # capability_map: sequence -> set of tribe_ids that have agents knowing it
        self.capability_map: dict[tuple, set] = {}

    def discover(self, tribe_id, tech_name):
        """No-op stub kept for backward compatibility. Real discovery is emergent."""
        pass

    def tribe_technologies(self, tribe_id):
        """Return list of sequences known by at least one member of the tribe."""
        known = []
        for seq, tribes in self.capability_map.items():
            if (tribe_id or 0) in tribes:
                known.append(seq)
        return known

    def has_tech(self, tribe_id, tech_name) -> bool:
        """Always False — there are no named techs anymore."""
        return False

    def cultural_diversity(self) -> float:
        return self.culture_tracker.cultural_diversity()

    def culture_summary(self) -> dict:
        return self.culture_tracker.summary()

    def update(self, agents, tribes):
        self.culture_tracker.update(agents, self._current_tick(agents))
        self.capability_map.clear()
        for agent in agents:
            if not agent.alive:
                continue
            causal_mem = getattr(agent, 'causal_memory', None)
            if causal_mem is None:
                continue
            for seq in causal_mem.sequences:
                if seq not in self.capability_map:
                    self.capability_map[seq] = set()
                self.capability_map[seq].add(agent.tribe_id or 0)

    def _current_tick(self, agents):
        for a in agents:
            if hasattr(a, 'birth_tick'):
                return getattr(a, 'age', 0)
        return 0
