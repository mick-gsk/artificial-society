"""
Cultural Memory & Transmission System
--------------------------------------
Agents do NOT receive tech labels ("fire", "axe").
They store causal sequences: (action, mat_a, mat_b) -> observed_effects.
Successful sequences can be socially observed and imperfectly transmitted.
This is the foundation of cumulative culture without pre-programmed recipes.
"""

import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Sequence memory per agent
# ---------------------------------------------------------------------------

class CausalMemory:
    """
    Stores (action, mat_a, mat_b) -> list of observed outcome tags.
    success_count tracks how often this sequence produced beneficial outcomes.
    """

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.sequences: dict[tuple, dict] = {}  # key -> {outcomes, successes, attempts}

    def record(self, action: str, mat_a: str, mat_b: str | None, outcomes: list[str], reward: float):
        key = (action, mat_a, mat_b or '')
        if key not in self.sequences:
            if len(self.sequences) >= self.capacity:
                # Evict least successful
                worst = min(self.sequences, key=lambda k: self.sequences[k]['successes'])
                del self.sequences[worst]
            self.sequences[key] = {'outcomes': outcomes, 'successes': 0, 'attempts': 0, 'reward': 0.0}
        entry = self.sequences[key]
        entry['attempts'] += 1
        entry['reward'] = entry['reward'] * 0.8 + reward * 0.2  # EMA
        if reward > 0.1:
            entry['successes'] += 1
        entry['outcomes'] = outcomes  # update with latest

    def best_known(self, min_successes: int = 2) -> list[tuple]:
        """Return sequences with at least min_successes, sorted by reward."""
        return sorted(
            [(k, v) for k, v in self.sequences.items() if v['successes'] >= min_successes],
            key=lambda x: -x[1]['reward']
        )

    def sample_for_transmission(self) -> tuple | None:
        """Pick a sequence to share with another agent (preferably successful ones)."""
        good = self.best_known(min_successes=1)
        if not good:
            return None
        # Weighted by success count
        weights = [v['successes'] for _, v in good]
        total = sum(weights)
        r = random.random() * total
        acc = 0
        for (k, v), w in zip(good, weights):
            acc += w
            if r <= acc:
                return k  # (action, mat_a, mat_b)
        return good[0][0]

    def receive_transmitted(self, sequence: tuple, fidelity: float = 0.8):
        """
        Accept a transmitted sequence from another agent.
        fidelity < 1.0 means imperfect copying (a material or action may mutate).
        This is the key mechanism of cultural evolution with variation.
        """
        action, mat_a, mat_b = sequence
        if random.random() > fidelity:
            # Imperfect transmission: mutate one element
            from artificial_society.environment.materials import MATERIALS
            all_mats = list(MATERIALS.keys())
            choice = random.random()
            if choice < 0.33:
                action = random.choice(['rub', 'strike', 'place_on_heat', 'bundle', 'blow', 'carry', 'eat'])
            elif choice < 0.66:
                mat_a = random.choice(all_mats)
            else:
                mat_b = random.choice(all_mats) if mat_b else mat_b
        key = (action, mat_a, mat_b)
        if key not in self.sequences:
            if len(self.sequences) >= self.capacity:
                worst = min(self.sequences, key=lambda k: self.sequences[k]['successes'])
                del self.sequences[worst]
            self.sequences[key] = {'outcomes': [], 'successes': 0, 'attempts': 0, 'reward': 0.0}

    def feature_vector(self) -> list[float]:
        """Compact summary for agent feature input."""
        n = len(self.sequences)
        successes = sum(v['successes'] for v in self.sequences.values())
        best_r = max((v['reward'] for v in self.sequences.values()), default=0.0)
        return [
            min(1.0, n / self.capacity),
            min(1.0, successes / max(1, n * 3)),
            min(1.0, best_r / 5.0),
        ]


# ---------------------------------------------------------------------------
# Population-level cultural tracking (for statistics & visualisation)
# ---------------------------------------------------------------------------

class CultureTracker:
    """
    Tracks which sequences exist in the population and how many
    agents know them. Gives an indicator of cultural diversity and
    the spread of discovered causal patterns (without naming them).
    """

    def __init__(self):
        self.population_sequences: dict[tuple, int] = defaultdict(int)  # seq -> agent count
        self.discovery_tick: dict[tuple, int] = {}

    def update(self, agents, tick: int):
        counts = defaultdict(int)
        for a in agents:
            if not a.alive:
                continue
            for seq in getattr(a, 'causal_memory', CausalMemory()).sequences:
                counts[seq] += 1
                if seq not in self.discovery_tick:
                    self.discovery_tick[seq] = tick
        self.population_sequences = counts

    def cultural_diversity(self) -> float:
        """Number of distinct sequences known in population."""
        return float(len(self.population_sequences))

    def most_widespread(self, top_n: int = 5) -> list[tuple]:
        return sorted(self.population_sequences, key=lambda k: -self.population_sequences[k])[:top_n]

    def summary(self) -> dict:
        return {
            'distinct_sequences': len(self.population_sequences),
            'total_knowledge_units': sum(self.population_sequences.values()),
            'most_widespread': [str(s) for s in self.most_widespread(3)],
        }
