"""Bilateraler Inventar-Tausch (emergente Wirtschaft).

Phase 5 de-scripting (Task 1): Handel ist nicht mehr auf die hartkodierte
``("wood", "stone", "fiber")``-Triade beschränkt. Zwei nahe, sich vertrauende
Agenten tauschen komplementäre Überschüsse *beliebiger* gehaltener Materialien
(Ressourcen **und** entdeckte/benannte ``material_inventory``-Güter). Der Wert ist
nicht hartkodiert — er emergiert aus der Nachfrage (``material_reward``) plus
Knappheit; so kann sich Spezialisierung über das ganze Materialspektrum bilden.
"""

from artificial_society.environment.materials import get_vector, material_reward

# Die drei Basis-Ressourcen leben in ``agent.resources``; alles andere
# (entdeckte ``mat_*`` und benannte Materialien) in ``agent.material_inventory``.
RESOURCE_KEYS = ("wood", "stone", "fiber")
# Nur ein echter Überschuss (>1 Einheit) wird angeboten — wie im Legacy-Gate.
TRADE_SURPLUS = 1.0


class EconomySystem:
    def __init__(self):
        self.trade_count = 0
        self.prices = {"wood": 1.0, "stone": 1.0, "fiber": 1.0}

    def maybe_trade(self, agent, agents):
        # Radius 2 statt identischer Position -- realistischer Sichtkontakt. Der
        # Nachbar-Cache wird einmal pro Tick auf dem Agenten gehalten.
        neighbors = agent._nearby_cached(agents, 2)
        if not neighbors:
            return
        for other in neighbors:
            if agent.trust.get(other.id, 0.0) < 0.1:
                continue
            if self._barter(agent, other):
                break

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _value_state(agent) -> dict:
        """Minimaler homeostatischer Zustand, den ``material_reward`` auswertet."""
        return {
            "energy": getattr(agent, "energy", 120.0) / 240.0,
            "cold": False,
            "dark": False,
        }

    @staticmethod
    def _demand(mat: str, state: dict) -> float:
        """Subjektive Nachfrage eines Agenten nach einem Material (>= 0)."""
        try:
            return max(0.0, material_reward(get_vector(mat), state))
        except Exception:
            return 0.0

    @staticmethod
    def _holdings(agent) -> dict:
        """Alle handelbaren Güter eines Agenten: Ressourcen + Inventar."""
        pool: dict = {}
        resources = getattr(agent, "resources", {}) or {}
        for k in RESOURCE_KEYS:
            q = float(resources.get(k, 0.0))
            if q > 0.0:
                pool[k] = q
        for mat, q in (getattr(agent, "material_inventory", {}) or {}).items():
            if mat.startswith("_"):
                continue  # interne Scratch-Items (_tinder_bundle etc.)
            q = float(q)
            if q > 0.0:
                pool[mat] = pool.get(mat, 0.0) + q
        return pool

    @staticmethod
    def _add(agent, mat: str, delta: float) -> None:
        if mat in RESOURCE_KEYS:
            agent.resources[mat] = agent.resources.get(mat, 0.0) + delta
        else:
            inv = getattr(agent, "material_inventory", None)
            if inv is None:
                inv = {}
                agent.material_inventory = inv
            inv[mat] = inv.get(mat, 0.0) + delta

    def _barter(self, agent, other) -> bool:
        """Findet und vollzieht den besten Ein-Einheiten-Tausch für ein Paar."""
        gives = [m for m, q in self._holdings(agent).items() if q > TRADE_SURPLUS]
        wants = [m for m, q in self._holdings(other).items() if q > TRADE_SURPLUS]
        if not gives or not wants:
            return False

        a_state = self._value_state(agent)
        o_state = self._value_state(other)
        # Gib das Gut, das der Partner am stärksten nachfragt; nimm das, das du
        # selbst am stärksten nachfragst. Stabiler Name-Tiebreak bei Gleichstand.
        give_mat = max(gives, key=lambda m: (self._demand(m, o_state), m))
        want_mat = max(wants, key=lambda m: (self._demand(m, a_state), m))
        if give_mat == want_mat:
            alt = [m for m in wants if m != give_mat]
            if not alt:
                return False
            want_mat = max(alt, key=lambda m: (self._demand(m, a_state), m))

        self._add(agent, give_mat, -1.0)
        self._add(other, give_mat, +1.0)
        self._add(other, want_mat, -1.0)
        self._add(agent, want_mat, +1.0)
        self.trade_count += 1
        # Vertrauen steigt proportional zur Knappheit des abgegebenen Guts.
        trust_gain = 0.03 + 0.02 * self.prices.get(give_mat, 1.0)
        agent.trust[other.id] = min(1.0, agent.trust.get(other.id, 0.0) + trust_gain)
        other.trust[agent.id] = min(1.0, other.trust.get(agent.id, 0.0) + trust_gain)
        return True

    def update(self, agents, tribes):
        totals = {"wood": 0, "stone": 0, "fiber": 0}
        for a in agents:
            if a.alive:
                for r in totals:
                    totals[r] += a.resources[r]
        total_all = sum(totals.values()) or 1
        for r in self.prices:
            share = totals[r] / total_all
            self.prices[r] = max(0.5, min(3.0, 1.0 / (share + 0.1)))
