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

Emergenz v3 (Lagerfeuer-Kollektivwissen):
campfire_knowledge_sharing() ermoeglicht gruppenweiten Wissensaustausch
an Feuerquellen. Alle Agenten in der Naehe eines Feuers tragen ihr Wissen
in einen gemeinsamen Pool und lernen daraus -- kein bilateraler Transfer mehr.
"""

import random

TRANSMISSION_RADIUS = 2
OBSERVE_PROB = 0.18
TEACH_PROB = 0.12
FIDELITY_BASE = 0.72
FIDELITY_TRUST_BONUS = 0.18

# Schwellenwert: Nachbar muss X-mal erfolgreicher sein damit Imitation ausgeloest wird
IMITATION_SUCCESS_RATIO = 1.5
IMITATION_MIN_TRUST = 0.4

# Emergenz v3: Lagerfeuer-Kollektivwissen
CAMPFIRE_RADIUS = 3  # Wie nah muessen Agenten am Feuer sein?
CAMPFIRE_MIN_AGENTS = 2  # Mindestanzahl Agenten fuer Gruppenlernen
CAMPFIRE_CONF_THRESHOLD = 0.3  # Mindest-Confidence fuer Weitergabe


def social_learning_step(agent, agents: list, tick: int) -> float:
    """
    Agent may observe a nearby successful agent and:
    1. Copy one of their known causal sequences (wie bisher)
    2. Leicht die eigenen Gehirngewichte in Richtung des Erfolgsagenten anpassen
       (Spiegelneuronen-Analogie: unbewusste Verhaltensimitation bei Beobachtung)
    Returns a small reward for successful transmission.
    """
    causal_mem = getattr(agent, "causal_memory", None)
    if causal_mem is None:
        return 0.0

    # Nachbar-Cache wird einmal pro Tick auf dem Agenten gehalten (Agent._nearby_cached);
    # Radius 2 == TRANSMISSION_RADIUS.
    nearby = agent._nearby_cached(agents, 2)
    if not nearby:
        return 0.0

    reward = 0.0
    agent_reward = getattr(agent, "last_reward", 0.0)

    for other in nearby:
        other_mem = getattr(other, "causal_memory", None)
        if other_mem is None:
            continue

        other_reward = getattr(other, "last_reward", 0.0)
        curiosity = agent.genes.get("curiosity", 0.5)
        trust = agent.trust.get(other.id, 0.0)

        # --- 1. Kausal-Sequenz beobachten ---
        observe_chance = OBSERVE_PROB + 0.15 * curiosity
        if other_reward > 0.5 and random.random() < observe_chance:
            seq = other_mem.sample_for_transmission()
            if seq:
                fidelity = FIDELITY_BASE + FIDELITY_TRUST_BONUS * max(0.0, trust)
                causal_mem.receive_transmitted(seq, fidelity=fidelity)
                reward += 0.05

        # --- 2. Aktives Lehren bei hohem Vertrauen ---
        if trust > 0.5 and random.random() < TEACH_PROB:
            seq = causal_mem.sample_for_transmission()
            other_causal = getattr(other, "causal_memory", None)
            if seq and other_causal:
                other_causal.receive_transmitted(
                    seq, fidelity=FIDELITY_BASE + FIDELITY_TRUST_BONUS * trust
                )
                reward += 0.06

        # --- 3. Neuronale Gewichtsangleichung (Spiegelneuronen-Analogie) ---
        if (
            other_reward > agent_reward * IMITATION_SUCCESS_RATIO
            and other_reward > 0.3
            and trust >= IMITATION_MIN_TRUST
            and random.random() < 0.15 + 0.20 * curiosity
        ):
            agent_brain = getattr(agent, "brain", None)
            other_brain = getattr(other, "brain", None)
            if agent_brain is not None and other_brain is not None:
                agent_brain.imitate_from(other_brain)
                reward += 0.08

    return reward


def campfire_knowledge_sharing(agents: list, world, tick: int) -> int:
    """
    Emergenz v3: Lagerfeuer-Kollektivwissen.

    Scannt alle Weltzellen auf Feuerquellen. Agenten in der Naehe eines Feuers
    bilden eine temporaere Lerngruppe und teilen ihren KnowledgeGraph kollektiv.

    Biologisches Vorbild: Das Lagerfeuer als erste Institution der Menschheit --
    gemeinsames Wissen ueber Jagdwege, Pflanzen, Werkzeugbau wurde abends
    am Feuer geteilt. Dadurch entstanden gemeinsame Kulturen und kumulative
    Technologien, die kein Einzelner alleine haette entwickeln koennen.

    Rückgabewert: Anzahl der Wissenstransfers die stattfanden.
    """
    transfers = 0
    alive_agents = [a for a in agents if a.alive]
    if not alive_agents:
        return 0

    # Sammle alle Zellen mit aktiven Feuerquellen
    fire_positions = []
    for y in range(world.height):
        for x in range(world.width):
            cell = world.cells[y][x]
            materials = cell.get("materials", {})
            if materials.get("fire", 0) > 0.3 or materials.get("ember", 0) > 0.5:
                fire_positions.append((x, y))

    for fx, fy in fire_positions:
        # Agenten in der Naehe des Feuers
        near_fire = [
            a
            for a in alive_agents
            if abs(a.pos[0] - fx) <= CAMPFIRE_RADIUS and abs(a.pos[1] - fy) <= CAMPFIRE_RADIUS
        ]
        if len(near_fire) < CAMPFIRE_MIN_AGENTS:
            continue

        # Kollektiven Wissenspool aufbauen:
        # Alle confident facts und macros der Gruppe zusammenfuehren
        pooled_facts = {}  # key -> best CausalFact
        pooled_macros = {}  # key -> best CompositeAction

        for a in near_fire:
            kg = getattr(a, "knowledge", None)
            if kg is None:
                continue
            for key, fact in kg.facts.items():
                if fact.confidence >= CAMPFIRE_CONF_THRESHOLD:
                    existing = pooled_facts.get(key)
                    if existing is None or fact.confidence > existing.confidence:
                        pooled_facts[key] = fact
            for key, macro in kg.macro_actions.items():
                if macro.confidence >= CAMPFIRE_CONF_THRESHOLD:
                    existing = pooled_macros.get(key)
                    if existing is None or macro.confidence > existing.confidence:
                        pooled_macros[key] = macro

        # Alle Agenten am Feuer lernen aus dem Pool
        for a in near_fire:
            kg = getattr(a, "knowledge", None)
            if kg is None:
                continue
            for key, fact in pooled_facts.items():
                if key not in kg.facts or kg.facts[key].confidence < fact.confidence * 0.5:
                    kg.record(key, fact.outcome_ids, True)
                    transfers += 1
            for key, macro in pooled_macros.items():
                if key not in kg.macro_actions:
                    kg.record_macro(
                        steps=list(macro.steps),
                        reward=macro.avg_reward * 0.7,  # Etwas abgeschwaecht -- Lernlaerm
                        materials=list(macro.context_materials),
                    )
                    transfers += 1

    return transfers
