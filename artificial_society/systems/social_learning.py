"""
Social Learning & Cultural Transmission
-----------------------------------------
Agents learn from observing neighbours who appear to be succeeding.
Transmission is imperfect (see CausalMemory.receive_transmitted).
There is NO direct teaching label -- agents copy action sequences,
not named technologies.

Neu (Spiegelneuronen-Analogie / Bandura's Social Learning Theory):
Zusaetzlich zur kausalen Sequenz-Uebertragung werden bei hohem Vertrauen
und deutlichem Erfolgsunterschied die Gehirn-Gewichte leicht angeglichen.
Das bildet nach wie beobachtetes Verhalten unbewusst imitiert wird,
ohne dass explizites Lehren stattfindet.
"""

import random


TRANSMISSION_RADIUS = 2
OBSERVE_PROB = 0.18
TEACH_PROB = 0.12
FIDELITY_BASE = 0.72
FIDELITY_TRUST_BONUS = 0.18

# Schwellenwert: Nachbar muss X-mal erfolgreicher sein damit Imitation ausgeloest wird
IMITATION_SUCCESS_RATIO = 1.5  # Nachbar muss mind. 50% mehr Reward haben
IMITATION_MIN_TRUST     = 0.4  # Mindest-Vertrauen fuer Gewichtsangleichung


def social_learning_step(agent, agents: list, tick: int) -> float:
    """
    Agent may observe a nearby successful agent and:
    1. Copy one of their known causal sequences (wie bisher)
    2. Leicht die eigenen Gehirngewichte in Richtung des Erfolgsagenten anpassen
       (Spiegelneuronen-Analogie: unbewusste Verhaltensimitation bei Beobachtung)
    Returns a small reward for successful transmission.
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
    agent_reward = getattr(agent, 'last_reward', 0.0)

    for other in nearby:
        other_mem = getattr(other, 'causal_memory', None)
        if other_mem is None:
            continue

        other_reward = getattr(other, 'last_reward', 0.0)
        curiosity = agent.genes.get('curiosity', 0.5)
        trust = agent.trust.get(other.id, 0.0)

        # --- 1. Kausal-Sequenz beobachten (unveraendert) ---
        observe_chance = OBSERVE_PROB + 0.15 * curiosity
        if other_reward > 0.5 and random.random() < observe_chance:
            seq = other_mem.sample_for_transmission()
            if seq:
                fidelity = FIDELITY_BASE + FIDELITY_TRUST_BONUS * max(0.0, trust)
                causal_mem.receive_transmitted(seq, fidelity=fidelity)
                reward += 0.05

        # --- 2. Aktives Lehren bei hohem Vertrauen (unveraendert) ---
        if trust > 0.5 and random.random() < TEACH_PROB:
            seq = causal_mem.sample_for_transmission()
            other_causal = getattr(other, 'causal_memory', None)
            if seq and other_causal:
                other_causal.receive_transmitted(seq, fidelity=FIDELITY_BASE + FIDELITY_TRUST_BONUS * trust)
                reward += 0.06

        # --- 3. Neuronale Gewichtsangleichung (Spiegelneuronen-Analogie) ---
        # Wird nur ausgeloest wenn:
        #   a) Nachbar deutlich erfolgreicher ist (IMITATION_SUCCESS_RATIO)
        #   b) Vertrauen hoch genug ist (IMITATION_MIN_TRUST)
        #   c) Plastizitaets-Gen erlaubt Imitation (curiosity als Proxy)
        # Biologisch: Menschen imitieren Verhalten das sie als erfolgreich beobachten,
        # vor allem wenn sie der Person vertrauen (Bandura, 1977).
        if (
            other_reward > agent_reward * IMITATION_SUCCESS_RATIO
            and other_reward > 0.3        # Nachbar muss absolut positiv sein
            and trust >= IMITATION_MIN_TRUST
            and random.random() < 0.15 + 0.20 * curiosity
        ):
            agent_brain = getattr(agent, 'brain', None)
            other_brain = getattr(other, 'brain', None)
            if agent_brain is not None and other_brain is not None:
                agent_brain.imitate_from(other_brain)
                reward += 0.08

    return reward
