"""
Territorialverhalten & raeumliche Pfadabhaengigkeit
----------------------------------------------------
Biologisches Vorbild: Territoriale Tiere (Primaten, Hunde, Voegel) markieren
Reviere durch Geruchsstoffe, Gesang oder physische Verteidigung. Das Revier
gibt Futtersicherheit und Reproduktionsvorteil.

In unserem Modell hinterlassen Agenten keine Geruchsmarken (zu komplex),
stattdessen wird Territorialitaet durch PERSISTENTE ZELLATTRIBUTE abgebildet:
- `territory_claim`: dict {tribe_id -> strength (0..1)}
- Anwesenheit eines Stammes staerkt den Claim, Abwesenheit schwaecht ihn
- Fremde Agenten auf einem beanspruchten Territorium erhalten einen Malus
  (erhoehte Gefahrenwahrnehmung -> Adrenalin -> Stresskosten)
- Der claimende Stamm erhaelt einen kleinen Ressourcenbonus (Heimvorteil)

Das erzeugt PFADABHAENGIGKEIT: Entscheidungen wo man wohnt haben
langfristige Konsequenzen die ueber den 64-Step-Horizont wirken.
"""

CLAIM_STRENGTHEN_RATE = 0.04   # Pro Tick in dem ein Stammesmitglied auf der Zelle ist
CLAIM_DECAY_RATE      = 0.008  # Zerfallsrate wenn niemand da ist
CLAIM_MAX             = 1.0
CLAIM_MIN             = 0.0
HOME_RESOURCE_BONUS   = 0.18   # Zusaetzliche Sammeleffizienz auf eigenem Territorium
INTRUDER_DANGER_BONUS = 12.0   # Gefahrenwahrnehmung fuer Eindringlinge
CLAIM_THRESHOLD       = 0.35   # Ab wann gilt eine Zelle als beansprucht


def update_territory_claims(world, agents):
    """
    Aktualisiert alle Zell-Territory-Claims basierend auf aktueller Agentenpraesenz.
    Wird einmal pro Tick von simulation.py aufgerufen.
    """
    # Zuerst alle Claims etwas zerfallen lassen
    for y in range(world.height):
        for x in range(world.width):
            cell = world.cells[y][x]
            claims = cell.get('territory_claim', {})
            for tid in list(claims):
                claims[tid] = max(CLAIM_MIN, claims[tid] - CLAIM_DECAY_RATE)
                if claims[tid] < 0.01:
                    del claims[tid]
            cell['territory_claim'] = claims

    # Dann Praesenz-Verstaerkung durch lebende Agenten
    for agent in agents:
        if not agent.alive or agent.tribe_id is None:
            continue
        x, y = agent.pos
        cell  = world.cells[y][x]
        claims = cell.get('territory_claim', {})
        tid    = agent.tribe_id
        claims[tid] = min(CLAIM_MAX, claims.get(tid, 0.0) + CLAIM_STRENGTHEN_RATE)
        cell['territory_claim'] = claims


def territory_reward_for_agent(agent, world) -> float:
    """
    Gibt einen Reward-Bonus oder Malus zurueck basierend auf dem
    Territorialstatus der aktuellen Agenten-Zelle.

    Heimvorteil: eigener Stamm hat starken Claim -> kleiner Ressourcenbonus
    Eindringling: fremder Stamm hat starken Claim -> Gefahrenmalus (Adrenalin)
    """
    x, y  = agent.pos
    cell  = world.cells[y][x]
    claims = cell.get('territory_claim', {})
    if not claims:
        return 0.0

    reward = 0.0
    own_tid = agent.tribe_id
    strongest_foreign = 0.0
    own_strength = claims.get(own_tid, 0.0) if own_tid is not None else 0.0

    for tid, strength in claims.items():
        if tid != own_tid:
            strongest_foreign = max(strongest_foreign, strength)

    # Heimvorteil: eigener Stamm dominiert diese Zelle
    if own_strength > CLAIM_THRESHOLD:
        reward += HOME_RESOURCE_BONUS * own_strength * 0.05  # klein aber konsistent

    # Eindringling-Malus: fremder Stamm hat starken Claim hier
    if strongest_foreign > CLAIM_THRESHOLD and own_strength < strongest_foreign:
        # Erhoeht Gefahrenwahrnehmung -> Adrenalin -> Flucht oder Kampf
        if hasattr(agent, 'endocrine'):
            agent.endocrine.h[1] = min(1.0, agent.endocrine.h[1] + 0.08 * strongest_foreign)
        reward -= 0.06 * strongest_foreign

    return reward


def get_home_forage_bonus(agent, world) -> float:
    """
    Gibt den Sammel-Effizienz-Bonus auf eigenem Territorium zurueck.
    Wird in agent.forage() als Multiplikator addiert.
    Biologisch: Tiere kennen ihr Revier besser und sammeln effizienter.
    """
    if agent.tribe_id is None:
        return 0.0
    x, y   = agent.pos
    cell   = world.cells[y][x]
    claims = cell.get('territory_claim', {})
    own    = claims.get(agent.tribe_id, 0.0)
    if own < CLAIM_THRESHOLD:
        return 0.0
    return HOME_RESOURCE_BONUS * own
