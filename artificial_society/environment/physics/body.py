"""Körper v1 — die physische Hülle des Agenten (Spec §4, Embodiment).

Der Körper gehört zur Weltphysik: reale Grenzen (Tragkraft, Kraft, Ermüdung)
erzeugen den Erfindungsdruck — ein Werkzeug lohnt sich genau deshalb, weil der
nackte Körper an diesen Grenzen scheitert. Vollständig deterministisch, kein
Zufall. Sim-Integration (Gene→strength, Energie-Kopplung) kommt im
Lern-Kopplungs-Plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import cal

# Körper-Physik (reale Anker: siehe cal()-Einträge unten)
CARRY_FRACTION_SUSTAINED = 0.30
STRIKE_ENERGY_MIN_J = 5.0
STRIKE_ENERGY_MAX_J = 50.0
FATIGUE_PER_JOULE = 0.0001  # ~200 kräftige Schläge (45 J) bis fatigue ~0.9
FATIGUE_RECOVERY_PER_TICK = 0.02  # volle Erholung nach ~50 Ruhe-Ticks
CARRY_FATIGUE_PER_TICK_AT_CAPACITY = (
    0.002  # Dauerlast an der Traggrenze: ~500 Ticks bis Erschöpfung
)

# Vom Realitäts-Gate geprüfte Körper-Parameter (wächst mit Tasks 2/3).
CALIBRATED_BODY_PARAMS = ("carry_capacity", "strike_energy", "fatigue")


@dataclass
class Body:
    body_mass: float  # kg
    strength: float  # [0,1] relative Kraft (1 ≈ kräftiger Erwachsener)
    fatigue: float = 0.0  # [0,1] (0 = ausgeruht)

    def __post_init__(self) -> None:
        if self.body_mass <= 0:
            raise ValueError("body_mass muss > 0 sein")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength muss in [0,1] liegen")
        if not 0.0 <= self.fatigue <= 1.0:
            raise ValueError("fatigue muss in [0,1] liegen")

    def carry_capacity_kg(self) -> float:
        """Dauerhaft tragbare Masse: ~30 % des Körpergewichts, skaliert mit
        Kraft, gedämpft durch Ermüdung."""
        return (
            CARRY_FRACTION_SUSTAINED
            * self.body_mass
            * (0.6 + 0.4 * self.strength)
            * (1.0 - 0.5 * self.fatigue)
        )

    def strike_energy_j(self, effort: float) -> float:
        """Aufprallenergie eines Handschlags mit Werkzeugstein (5–50 J),
        skaliert mit Kraft, gedämpft durch Ermüdung."""
        effort = min(max(effort, 0.0), 1.0)
        base = STRIKE_ENERGY_MIN_J + (STRIKE_ENERGY_MAX_J - STRIKE_ENERGY_MIN_J) * effort
        return base * (0.5 + 0.5 * self.strength) * (1.0 - 0.6 * self.fatigue)

    def exert_strike(self, energy_j: float) -> None:
        """Ein ausgeführter Schlag ermüdet proportional zur aufgewandten Energie."""
        self.fatigue = min(1.0, self.fatigue + max(energy_j, 0.0) * FATIGUE_PER_JOULE)

    def rest_tick(self) -> None:
        """Ein Tick Ruhe baut Ermüdung ab."""
        self.fatigue = max(0.0, self.fatigue - FATIGUE_RECOVERY_PER_TICK)

    def carry_tick(self, carried_mass_kg: float) -> None:
        """Ein Tick Tragen ermüdet proportional zur Auslastung der Tragkapazität."""
        if carried_mass_kg <= 0.0:
            return
        load = carried_mass_kg / self.carry_capacity_kg()
        self.fatigue = min(1.0, self.fatigue + CARRY_FATIGUE_PER_TICK_AT_CAPACITY * load)


cal(
    "body",
    "carry_capacity",
    "Dauer-Tragfähigkeit ≈ 30 % des Körpergewichts (Trekking-Richtwert 20–25 %, "
    "militärisches Marschgepäck 30–45 % mit Ermüdungsfolgen); skaliert mit Kraft, "
    "gedämpft durch Ermüdung",
    "Ergonomie-/Militär-Richtwerte zum Lastentragen",
)
cal(
    "body",
    "strike_energy",
    "Schlagenergie eines Handschlags mit Werkzeugstein 5–50 J (deckungsgleich mit dem "
    "Anker des Prozesses strike: kräftiger Handschlag 10–50 J); skaliert mit Kraft, "
    "gedämpft durch Ermüdung",
    "Biomechanik des Hammerschlags; experimentelle Archäologie",
)
cal(
    "body",
    "fatigue",
    "Ermüdung/Erholung, Größenordnungen: ~200 kräftige Schläge bis deutliche Erschöpfung "
    "(geübte Steinschläger arbeiten stundenlang); Dauerlast an der Traggrenze über "
    "Hunderte Ticks tragbar; Erholung in Ruhe über Dutzende Ticks. [Zeitskala "
    "Sim-Tick↔Realzeit bewusst qualitativ, bis die Sim-Integration sie fixiert]",
    "Arbeitsphysiologie (Ermüdung/Erholung beim Lastentragen und repetitiver Arbeit)",
)
