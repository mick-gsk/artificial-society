"""
Social Learning & Cultural Transmission
-----------------------------------------
Agents learn from observing neighbours who appear to be succeeding.
Transmission is imperfect (see CausalMemory.receive_transmitted).
There is NO direct teaching label — agents copy action sequences,
not named technologies.
"""

import random


TRANSMISSION_RADIUS = 2   # cells
OBSERVE_PROB = 0.18       # chance an agent looks at a successful neighbour
TEACH_PROB = 0.12         # chance a successful agent actively shows a sequence
FIDELITY_BASE = 0.72      # base copying fidelity
FIDELITY_TRUST_BONUS = 0.18  # extra fidelity for trusted partners


def social_learning_step(agent, agents: list, tick: int) -> float:
    """
    Agent may observe a nearby successful agent and copy one of their
    known causal sequences.  Returns a small reward for successful transmission.
    """
    causal_mem = getattr(agent, 'causal_memory', None)
    if causal_mem is None:
        return 0.0

    nearby = [
        a for a in agents
        if a is not agent and a.alive
        and abs(a.pos[0] - agent.pos[0]) <= TRANSMISSION_RADIUS
        and abs(a.pos[1] - agent.pos[1]) <= TRANSMISSION_RADIUS
    ]
    if not nearby:
        return 0.0

    reward = 0.0

    for other in nearby:
        other_mem = getattr(other, 'causal_memory', None)
        if other_mem is None:
            continue

        # Observe: does other agent look successful?
        other_reward = getattr(other, 'last_reward', 0.0)
        curiosity = agent.genes.get('curiosity', 0.5)
        observe_chance = OBSERVE_PROB + 0.15 * curiosity
        if other_reward > 0.5 and random.random() < observe_chance:
            seq = other_mem.sample_for_transmission()
            if seq:
                trust = agent.trust.get(other.id, 0.0)
                fidelity = FIDELITY_BASE + FIDELITY_TRUST_BONUS * max(0.0, trust)
                causal_mem.receive_transmitted(seq, fidelity=fidelity)
                reward += 0.05

        # Active teaching: high-trust agents may proactively share
        trust = agent.trust.get(other.id, 0.0)
        if trust > 0.5 and random.random() < TEACH_PROB:
            seq = causal_mem.sample_for_transmission()
            other_causal = getattr(other, 'causal_memory', None)
            if seq and other_causal:
                other_causal.receive_transmitted(seq, fidelity=FIDELITY_BASE + FIDELITY_TRUST_BONUS * trust)
                reward += 0.06

    return reward
