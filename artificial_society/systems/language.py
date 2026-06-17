"""
Proto-Language & Symbol System
--------------------------------
Sprache entsteht hier NICHT durch ein Woerterbuch.
Sie entsteht durch wiederholte Kopplung zwischen einem physischen
Token-Objekt und einem Kontext (Materialvektor / Situation).

Mechanismus:
  1. Agent fuehrt 'mark'-Action aus: charcoal + clay/stone/flat_obj
     → erzeugt ein Token-Objekt mit eindeutiger visueller ID (Form+Muster)
  2. Der Agent speichert in seiner TokenMemory:
     token_id → (context_vector, use_count, reinforcements)
  3. Andere Agenten sehen das Token (im Sichtfeld).
     Wenn ein Agent es in einem aehnlichen Kontext sieht und
     der Ersteller einen positiven Reward hatte: Imitation
  4. Wenn >50% der Population dasselbe Token fuer denselben
     Kontext-Cluster verwenden: Shared Symbol emergiert
     → [LANGUAGE_EVENT] geloggt

Was daraus emergieren kann:
  - Warnsignale (Token = 'Gefahr hier')
  - Eigentumsmarkierungen (Token = 'mein Territorium')
  - Rezept-Symbole (Token = 'hier Feuer machen')
  - Schliesslich: kombinierte Token = Proto-Saetze
"""

import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from artificial_society.environment.materials import (
    IDX, N_PROPS, MATERIALS, get_vector, DISCOVERY_REGISTRY
)


# ---------------------------------------------------------------------------
# Token: das physische Symbol-Objekt
# ---------------------------------------------------------------------------
@dataclass
class Token:
    """
    Ein physisches Zeichen in der Welt.
    Entsteht aus charcoal/pigment auf einer flachen Oberflaeche.
    token_shape ist ein 8-bit Hash des Schoepfer-IDs + Tick -- kein
    Alphabet, nur eine eindeutige visuelle Signatur.
    """
    token_id:    str
    creator_id:  int
    tick_made:   int
    x: int
    y: int
    # Visuelle Signatur (deterministisch aus creator + tick)
    shape_bits:  int   = 0
    # Wie stark verblasst das Token (Haltbarkeit)
    durability:  float = 1.0
    # Wie viele Agenten haben es gesehen/genutzt
    observers:   list  = field(default_factory=list)


def make_token(creator_id: int, tick: int, x: int, y: int,
               pigment_vec: np.ndarray) -> Token:
    """
    Erstellt ein Token. Die shape_bits sind deterministisch aber
    'beliebig' -- kein Alphabet, nur Identitaet.
    Die Haltbarkeit haengt von der Pigment-Qualitaet ab
    (conductivity als Proxy fuer Bindekraft).
    """
    raw = f'{creator_id}:{tick}:{x}:{y}'
    shape_bits = int(hashlib.md5(raw.encode()).hexdigest()[:4], 16)
    durability = 0.3 + float(pigment_vec[IDX['conductivity']]) * 0.7
    token_id   = f'tok_{shape_bits:05d}'
    return Token(
        token_id   = token_id,
        creator_id = creator_id,
        tick_made  = tick,
        x = x, y = y,
        shape_bits = shape_bits,
        durability = durability,
    )


# ---------------------------------------------------------------------------
# TokenMemory: pro-Agent Assoziations-Speicher
# ---------------------------------------------------------------------------
@dataclass
class TokenAssociation:
    token_id:        str
    context_vectors: list          # Liste von np.ndarray (Kontexte)
    use_count:       int   = 0
    reward_sum:      float = 0.0
    is_shared:       bool  = False  # True wenn von Gruppe konvergiert


class TokenMemory:
    """
    Pro-Agent Gedaechtnis fuer Token-Bedeutungen.
    Kein Dictionary von 'Wort -> Bedeutung'.
    Stattdessen: Cluster von Kontextvektoren pro Token.
    Bedeutung emergiert aus statistischer Haeufung.
    """
    def __init__(self):
        # token_id → TokenAssociation
        self.associations: dict[str, TokenAssociation] = {}
        # Kontext-Aehnlichkeits-Schwellwert
        self.similarity_threshold = 0.25

    def record_use(self, token_id: str, context_vec: np.ndarray,
                   reward: float):
        """Agent hat Token in einem bestimmten Kontext benutzt/gesehen."""
        if token_id not in self.associations:
            self.associations[token_id] = TokenAssociation(
                token_id        = token_id,
                context_vectors = [context_vec.copy()],
                use_count       = 1,
                reward_sum      = reward,
            )
        else:
            assoc = self.associations[token_id]
            assoc.context_vectors.append(context_vec.copy())
            assoc.use_count  += 1
            assoc.reward_sum += reward
            # Kap auf letzte 20 Kontexte (Gedaechtnis-Limit)
            if len(assoc.context_vectors) > 20:
                assoc.context_vectors = assoc.context_vectors[-20:]

    def best_token_for_context(self, context_vec: np.ndarray
                               ) -> Optional[str]:
        """
        Findet das Token, das am besten zu diesem Kontext passt
        (basierend auf gespeicherten Assoziationen + positivem Reward).
        """
        best_id, best_score = None, -999.0
        for token_id, assoc in self.associations.items():
            if not assoc.context_vectors:
                continue
            # Durchschnittlicher Kontext-Vektor dieser Assoziation
            avg_ctx = np.mean(assoc.context_vectors, axis=0)
            sim     = float(np.dot(context_vec, avg_ctx) /
                            (np.linalg.norm(context_vec) *
                             np.linalg.norm(avg_ctx) + 1e-8))
            # Score: Aehnlichkeit * positiver Reward-Durchschnitt
            avg_reward = assoc.reward_sum / max(1, assoc.use_count)
            score = sim * max(0.0, avg_reward)
            if score > best_score:
                best_score = score
                best_id    = token_id
        return best_id if best_score > 0.05 else None

    def mean_context(self, token_id: str) -> Optional[np.ndarray]:
        assoc = self.associations.get(token_id)
        if not assoc or not assoc.context_vectors:
            return None
        return np.mean(assoc.context_vectors, axis=0)


# ---------------------------------------------------------------------------
# Globale Token-Welt: alle platzierten Token
# ---------------------------------------------------------------------------
class TokenWorld:
    """
    Verwaltet alle physischen Token in der Welt.
    Prueft auf Konvergenz (Sprach-Emergenz).
    """
    def __init__(self):
        self.tokens: dict[str, Token] = {}      # token_id → Token
        self.world_log: list[dict]    = []      # Sprach-Events

    def place_token(self, token: Token):
        self.tokens[token.token_id] = token

    def tokens_at(self, x: int, y: int, radius: int = 1) -> list[Token]:
        result = []
        for tok in self.tokens.values():
            if abs(tok.x - x) <= radius and abs(tok.y - y) <= radius:
                result.append(tok)
        return result

    def tick_decay(self):
        """Token verblassen. Charcoal auf Stein haelt lange, auf Holz kurz."""
        to_remove = []
        for tid, tok in self.tokens.items():
            tok.durability -= 0.005  # sehr langsam
            if tok.durability <= 0:
                to_remove.append(tid)
        for tid in to_remove:
            del self.tokens[tid]

    def check_convergence(
        self,
        agent_memories: list[TokenMemory],
        tick: int,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Prueft ob ein Token von genug Agenten fuer denselben Kontext
        verwendet wird. Wenn ja: Shared Symbol = emergierte Sprache.
        """
        events = []
        # Alle bekannten Token-IDs
        all_token_ids: set[str] = set()
        for mem in agent_memories:
            all_token_ids.update(mem.associations.keys())

        for token_id in all_token_ids:
            users = [
                mem for mem in agent_memories
                if token_id in mem.associations
                and mem.associations[token_id].use_count >= 3
            ]
            if len(users) < 2:
                continue

            # Vergleiche mittlere Kontextvektoren aller User
            mean_ctxs = [
                mem.mean_context(token_id)
                for mem in users
                if mem.mean_context(token_id) is not None
            ]
            if len(mean_ctxs) < 2:
                continue

            # Paarweise Cosinus-Aehnlichkeit
            similarities = []
            for i in range(len(mean_ctxs)):
                for j in range(i+1, len(mean_ctxs)):
                    a, b = mean_ctxs[i], mean_ctxs[j]
                    sim = float(np.dot(a, b) /
                                (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                    similarities.append(sim)

            avg_sim = float(np.mean(similarities)) if similarities else 0.0
            share   = len(users) / max(1, len(agent_memories))

            if avg_sim > 0.7 and share >= threshold:
                # Markiere als geteiltes Symbol
                for mem in users:
                    if token_id in mem.associations:
                        mem.associations[token_id].is_shared = True

                event = {
                    'type':       'LANGUAGE_EVENT',
                    'tick':       tick,
                    'token_id':   token_id,
                    'users':      len(users),
                    'similarity': round(avg_sim, 3),
                    'share':      round(share, 3),
                }
                self.world_log.append(event)
                events.append(event)
                print(f'[LANGUAGE] tick={tick} shared_symbol={token_id} '
                      f'users={len(users)} ctx_similarity={avg_sim:.3f}')

        return events


TOKEN_WORLD = TokenWorld()


# ---------------------------------------------------------------------------
# Mark-Action: Agent erzeugt Token
# ---------------------------------------------------------------------------
def agent_mark(
    agent,
    cell: dict,
    context_vec: np.ndarray,
    tick: int,
) -> Optional[str]:
    """
    Agent fuehrt 'mark'-Action aus.
    Benoetigt charcoal oder anderes Pigment (conductivity > 0.3)
    und eine flache Oberflaeche (flat CompositeObject oder stone/clay).

    Gibt token_id zurueck oder None wenn nicht moeglich.
    """
    inv = getattr(agent, 'material_inventory', {})

    # Pigment suchen (charcoal, flower_petals als Farbe, etc.)
    pigment = None
    pigment_vec = None
    for mat_id, qty in inv.items():
        if qty < 0.05:
            continue
        vec = get_vector(mat_id)
        # Pigment: hohe Loeslichkeit + (conductivity oder scent) = Farbe/Tinte
        if (float(vec[IDX['solubility']]) > 0.3 and
                (float(vec[IDX['conductivity']]) > 0.2 or
                 float(vec[IDX['scent']]) > 0.3)):
            pigment     = mat_id
            pigment_vec = vec
            break

    if pigment is None or pigment_vec is None:
        return None  # Kein Pigment verfuegbar

    # Oberflaeche pruefen: Stein/Lehm in Zelle oder flat obj
    surface_ok = (
        'stone' in cell.get('materials', {})
        or 'clay' in cell.get('materials', {})
        or any(getattr(obj, 'flags', lambda: {})().get('flat', False)
               for obj in cell.get('objects', []))
    )
    if not surface_ok:
        return None

    # Token erstellen
    token = make_token(
        creator_id  = agent.id,
        tick        = tick,
        x           = agent.x,
        y           = agent.y,
        pigment_vec = pigment_vec,
    )

    # In Welt platzieren
    TOKEN_WORLD.place_token(token)

    # In Agent-Gedaechtnis speichern
    if not hasattr(agent, 'token_memory'):
        agent.token_memory = TokenMemory()
    agent.token_memory.record_use(token.token_id, context_vec, reward=0.1)

    # Pigment verbrauchen
    inv[pigment] = max(0.0, inv[pigment] - 0.05)

    return token.token_id


def agent_observe_token(
    agent,
    token: Token,
    context_vec: np.ndarray,
    reward_signal: float,
):
    """
    Agent sieht ein Token von jemand anderem.
    Wenn Reward-Signal positiv: Assoziation lernen (Imitation).
    """
    if not hasattr(agent, 'token_memory'):
        agent.token_memory = TokenMemory()
    # Nur lernen wenn Token-Ersteller vertraut wird
    trust = getattr(agent, 'trust', {}).get(token.creator_id, 0.0)
    effective_reward = reward_signal * (0.3 + 0.7 * trust)
    agent.token_memory.record_use(token.token_id, context_vec, effective_reward)
