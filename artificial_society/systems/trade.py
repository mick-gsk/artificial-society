"""
Inventar-Tausch & emergente Wirtschaft
----------------------------------------
Kein zentraler Markt. Kein Geld (noch).
Tausch entsteht bilateral zwischen zwei Agenten die sich nahe sind.

Wert ist nicht hardcodiert. Er emergiert aus:
  - Angebot: wie viel hat Agent A von Material X?
  - Nachfrage: wie dringend braucht Agent B Material X?
    (= wie hoch ist material_reward(X, agent_B.state)?)
  - Vertrauen: vorherige Tausche zwischen A und B
  - Reputation: wie oft hat A gute Tausche gemacht?

Aus diesem System emergieren:
  - Spezialisierung: Agent A produziert immer Keramik,
    Agent B immer Nahrung -- weil es sich lohnt
  - Preisfindung: Verhältnis der getauschten Mengen konvergiert
    zu einem stabilen Wert (emergenter 'Preis')
  - Wertaufbewahrung: Materialien mit hoher Nachfrage werden
    gehortet und nicht konsumiert
  - Soziale Hierarchie: Agenten mit seltenen Gutern haben Macht
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from artificial_society.environment.materials import (
    IDX, N_PROPS, get_vector, material_reward, DISCOVERY_REGISTRY
)


MAX_TRADE_RADIUS = 2   # Zellen Abstand fuer Tausch
MIN_TRADE_QTY    = 0.1  # Mindestmenge fuer Tausch


# ---------------------------------------------------------------------------
# Wert-Funktion
# ---------------------------------------------------------------------------
def compute_value(mat_id: str, qty: float, agent_state: dict) -> float:
    """
    Subjektiver Wert eines Materials fuer einen Agenten.
    Hohe Nachfrage (Hunger, Kaelte) = hoher Wert.
    Viel davon im Inventar = niedrigerer Wert (Grenznutzen).
    """
    vec       = get_vector(mat_id)
    base_val  = material_reward(vec, agent_state)
    own_qty   = agent_state.get('inv_' + mat_id, 0.0)
    # Grenznutzen: je mehr man hat, desto weniger wert
    margin    = max(0.1, 1.0 - own_qty * 0.4)
    return float(base_val * qty * margin)


# ---------------------------------------------------------------------------
# Trade Proposal
# ---------------------------------------------------------------------------
@dataclass
class TradeProposal:
    proposer_id:  int
    receiver_id:  int
    offer_mat:    str    # Was Proposer anbietet
    offer_qty:    float
    request_mat:  str    # Was Proposer haben moechte
    request_qty:  float
    tick:         int
    accepted:     bool   = False

    def proposer_surplus(self, agent_states: dict) -> float:
        """Netto-Gewinn fuer Proposer wenn Tausch angenommen."""
        give = compute_value(self.offer_mat,   self.offer_qty,   agent_states.get(self.proposer_id, {}))
        get  = compute_value(self.request_mat, self.request_qty, agent_states.get(self.proposer_id, {}))
        return get - give

    def receiver_surplus(self, agent_states: dict) -> float:
        """Netto-Gewinn fuer Receiver wenn Tausch angenommen."""
        give = compute_value(self.request_mat, self.request_qty, agent_states.get(self.receiver_id, {}))
        get  = compute_value(self.offer_mat,   self.offer_qty,   agent_states.get(self.receiver_id, {}))
        return get - give


# ---------------------------------------------------------------------------
# TradeMemory pro Agent
# ---------------------------------------------------------------------------
@dataclass
class TradeRecord:
    partner_id:  int
    gave_mat:    str
    gave_qty:    float
    got_mat:     str
    got_qty:     float
    net_reward:  float  # ex-post Reward nach dem Tausch
    tick:        int


class TradeMemory:
    """
    Agenten erinnern sich an Tausche und ihre Ergebnisse.
    Basis fuer Spezialisierung und Reputations-System.
    """
    def __init__(self):
        self.records:    list[TradeRecord]      = []
        # partner_id -> avg_net_reward
        self.partner_score: dict[int, float]    = {}
        # mat_id -> wie oft angeboten
        self.offer_freq:   dict[str, int]       = {}
        # mat_id -> wie oft nachgefragt
        self.request_freq: dict[str, int]       = {}

    def record(self, trade: TradeRecord):
        self.records.append(trade)
        # Kap auf letzte 50
        if len(self.records) > 50:
            self.records = self.records[-50:]
        # Partner-Score aktualisieren
        prev = self.partner_score.get(trade.partner_id, 0.0)
        self.partner_score[trade.partner_id] = prev * 0.8 + trade.net_reward * 0.2
        # Frequenzen
        self.offer_freq[trade.gave_mat]     = self.offer_freq.get(trade.gave_mat, 0) + 1
        self.request_freq[trade.got_mat]    = self.request_freq.get(trade.got_mat, 0) + 1

    def best_partner(self) -> Optional[int]:
        """Wen soll ich als naechstes ansprechen?"""
        if not self.partner_score:
            return None
        return max(self.partner_score, key=self.partner_score.get)

    def specialization_index(self) -> float:
        """
        0 = kein Spezialist, 1 = perfekter Spezialist.
        Hoch wenn Agent immer dasselbe Material anbietet.
        """
        if not self.offer_freq:
            return 0.0
        total  = sum(self.offer_freq.values())
        top    = max(self.offer_freq.values())
        return top / max(1, total)

    def most_offered(self) -> Optional[str]:
        if not self.offer_freq:
            return None
        return max(self.offer_freq, key=self.offer_freq.get)


# ---------------------------------------------------------------------------
# Trade Engine
# ---------------------------------------------------------------------------
class TradeEngine:
    """
    Verwaltet alle aktiven Tausch-Vorschlaege und deren Ausfuehrung.
    Wird pro Tick vom World-Step aufgerufen.
    """
    def __init__(self):
        self.pending:  list[TradeProposal] = []
        self.history:  list[TradeProposal] = []
        self.tick_log: list[dict]          = []

        # Emergenter Preisspiegel: mat_pair -> ratio-Liste
        # Aus Wiederholungen entsteht ein stabiler 'Preis'
        self.price_memory: dict[tuple, list[float]] = {}

    def propose(
        self,
        proposer,
        receiver,
        offer_mat: str,
        offer_qty: float,
        request_mat: str,
        request_qty: float,
        tick: int,
    ) -> Optional[TradeProposal]:
        """Agent schlaegt Tausch vor. Prueft Inventar-Verfuegbarkeit."""
        prop_inv = getattr(proposer, 'material_inventory', {})
        if prop_inv.get(offer_mat, 0.0) < offer_qty:
            return None  # Nicht genug zum Anbieten

        proposal = TradeProposal(
            proposer_id = proposer.id,
            receiver_id = receiver.id,
            offer_mat   = offer_mat,
            offer_qty   = offer_qty,
            request_mat = request_mat,
            request_qty = request_qty,
            tick        = tick,
        )
        self.pending.append(proposal)
        return proposal

    def evaluate(
        self,
        proposal: TradeProposal,
        receiver,
        agent_states: dict,
    ) -> bool:
        """
        Receiver entscheidet ob er annimmt.
        Entscheidung: Receiver-Surplus > 0 und Inventar ausreichend.
        Vertrauen moduliert den Schwellwert.
        """
        recv_inv = getattr(receiver, 'material_inventory', {})
        if recv_inv.get(proposal.request_mat, 0.0) < proposal.request_qty:
            return False  # Kann nicht liefern

        surplus = proposal.receiver_surplus(agent_states)

        # Vertrauens-Modulation
        trust = getattr(receiver, 'trust', {}).get(proposal.proposer_id, 0.3)
        threshold = -0.1 + (1.0 - trust) * 0.3  # Misstrauen = hoehere Huerde

        return surplus > threshold

    def execute(
        self,
        proposal: TradeProposal,
        proposer,
        receiver,
        tick: int,
    ) -> dict:
        """Fuehrt den Tausch aus. Transferiert Materialien."""
        prop_inv = getattr(proposer, 'material_inventory', {})
        recv_inv = getattr(receiver, 'material_inventory', {})

        # Transfer
        prop_inv[proposal.offer_mat]    = max(0.0, prop_inv.get(proposal.offer_mat,   0.0) - proposal.offer_qty)
        recv_inv[proposal.offer_mat]    = recv_inv.get(proposal.offer_mat, 0.0)   + proposal.offer_qty
        recv_inv[proposal.request_mat]  = max(0.0, recv_inv.get(proposal.request_mat, 0.0) - proposal.request_qty)
        prop_inv[proposal.request_mat]  = prop_inv.get(proposal.request_mat, 0.0) + proposal.request_qty

        proposal.accepted = True
        self.history.append(proposal)

        # Preis-Memory aktualisieren
        pair  = (proposal.offer_mat, proposal.request_mat)
        ratio = proposal.offer_qty / max(0.01, proposal.request_qty)
        if pair not in self.price_memory:
            self.price_memory[pair] = []
        self.price_memory[pair].append(ratio)
        if len(self.price_memory[pair]) > 30:
            self.price_memory[pair] = self.price_memory[pair][-30:]

        event = {
            'type':          'TRADE',
            'tick':          tick,
            'proposer':      proposal.proposer_id,
            'receiver':      proposal.receiver_id,
            'gave':          (proposal.offer_mat, proposal.offer_qty),
            'got':           (proposal.request_mat, proposal.request_qty),
        }
        self.tick_log.append(event)

        # Specialization logging
        if not hasattr(proposer, 'trade_memory'):
            proposer.trade_memory = TradeMemory()
        if not hasattr(receiver, 'trade_memory'):
            receiver.trade_memory = TradeMemory()

        print(f'[TRADE] tick={tick} '
              f'agent_{proposal.proposer_id} gave {proposal.offer_qty:.2f}x{proposal.offer_mat} '
              f'for {proposal.request_qty:.2f}x{proposal.request_mat} '
              f'from agent_{proposal.receiver_id}')

        return event

    def get_market_price(
        self, mat_a: str, mat_b: str
    ) -> Optional[float]:
        """
        Gibt den emergenten 'Preis' zurueck (Tauschverhaeltnis).
        None wenn noch kein Tausch stattgefunden hat.
        """
        pair = (mat_a, mat_b)
        if pair in self.price_memory and self.price_memory[pair]:
            return float(np.mean(self.price_memory[pair][-10:]))
        return None

    def tick_trade_opportunities(
        self,
        agents: list,
        agent_states: dict,
        current_tick: int,
    ) -> list[dict]:
        """
        Findet alle nahen Agenten-Paare und ermoeglicht Tausch.
        Wird vom World-Tick aufgerufen.
        """
        events = []
        paired = set()

        for i, agent_a in enumerate(agents):
            if agent_a.id in paired:
                continue
            inv_a = getattr(agent_a, 'material_inventory', {})
            if not inv_a:
                continue

            for agent_b in agents[i+1:]:
                if agent_b.id in paired:
                    continue
                # Distanz-Check
                dist = abs(agent_a.x - agent_b.x) + abs(agent_a.y - agent_b.y)
                if dist > MAX_TRADE_RADIUS:
                    continue

                inv_b = getattr(agent_b, 'material_inventory', {})
                if not inv_b:
                    continue

                # Bestes Tauschangebot finden
                proposal = self._best_proposal(agent_a, agent_b, agent_states, current_tick)
                if proposal is None:
                    continue

                # Receiver entscheidet
                if self.evaluate(proposal, agent_b, agent_states):
                    event = self.execute(proposal, agent_a, agent_b, current_tick)
                    events.append(event)
                    paired.add(agent_a.id)
                    paired.add(agent_b.id)
                    break

        return events

    def _best_proposal(
        self,
        proposer,
        receiver,
        agent_states: dict,
        tick: int,
    ) -> Optional[TradeProposal]:
        """
        Findet das beste Tauschangebot fuer dieses Paar.
        Proposer bietet das an was Receiver am meisten braucht
        und bittet um das was er selbst am meisten braucht.
        """
        prop_inv  = getattr(proposer, 'material_inventory', {})
        recv_inv  = getattr(receiver, 'material_inventory', {})
        prop_state = agent_states.get(proposer.id, {})
        recv_state = agent_states.get(receiver.id, {})

        # Was braucht Receiver am meisten? (hoechster Wert fuer Receiver)
        best_offer_mat, best_offer_val = None, -999.0
        for mat, qty in prop_inv.items():
            if qty < MIN_TRADE_QTY:
                continue
            val = compute_value(mat, qty, recv_state)
            if val > best_offer_val:
                best_offer_val = val
                best_offer_mat = mat

        if best_offer_mat is None or best_offer_val <= 0:
            return None

        # Was braucht Proposer am meisten? (hoechster Wert fuer Proposer)
        best_req_mat, best_req_val = None, -999.0
        for mat, qty in recv_inv.items():
            if qty < MIN_TRADE_QTY:
                continue
            val = compute_value(mat, qty, prop_state)
            if val > best_req_val:
                best_req_val = val
                best_req_mat = mat

        if best_req_mat is None or best_req_val <= 0:
            return None

        # Mengen proportional zum relativen Wert
        offer_qty   = min(prop_inv[best_offer_mat] * 0.4, 1.0)
        request_qty = min(recv_inv[best_req_mat]   * 0.4, 1.0)

        return self.propose(
            proposer    = proposer,
            receiver    = receiver,
            offer_mat   = best_offer_mat,
            offer_qty   = offer_qty,
            request_mat = best_req_mat,
            request_qty = request_qty,
            tick        = tick,
        )


TRADE_ENGINE = TradeEngine()
