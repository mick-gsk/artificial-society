"""Verkörperte Aktions-Mechanik der Physik v2 (Plan 3a, Spec B4/B5).

Dieses Modul bindet Body/Hands/Prozesse/ObjectLayer — es kennt weder Gehirn
noch Belohnung. Alle Konstanten sind real geankert (cal-Einträge unten,
Gate-pflichtig über CALIBRATED_ACTION_PARAMS). Energie-Konvention: nutrition
ist kcal/100 g ÷ 400 (raw_meat 0.35 ≙ 140 kcal/100 g), gekoppelt an die
v1-Energieskala über SIM_ENERGY_PER_KCAL (1-kg-Fleischmahlzeit ≙ MEAT_ENERGY 45).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from artificial_society.environment.daynight import TICKS_PER_DAY

from .body import Body, Hands
from .calibration import cal
from .objects import PhysObject
from .processes import cut, strike
from .props import IDX2

# --- Energie-Kopplung kcal ↔ Sim-Energie (B5) --------------------------------
SIM_ENERGY_PER_KCAL = 0.032
KCAL_PER_KG_PER_NUTRITION = 4000.0  # nutrition 1.0 ≙ 400 kcal/100 g = 4000 kcal/kg
MUSCLE_EFFICIENCY = 0.25
JOULE_PER_KCAL = 4184.0
# Spec B5 (Rev. 4): energy -= joules / MUSCLE_EFFICIENCY / 4184 * SIM_ENERGY_PER_KCAL
# — 200 Schläge à 45 J ≈ 0.28 Sim-Energie (Anker-Test in test_actions.py hält das fest).
WORK_SIM_ENERGY_PER_JOULE = SIM_ENERGY_PER_KCAL / (MUSCLE_EFFICIENCY * JOULE_PER_KCAL)

# --- Schlag (B4) --------------------------------------------------------------
V_MAX_STRIKE = 14.0  # m/s — ½·m·v² kappt die gelieferte Energie leichter Schläger

# --- Schneiden (B4) -----------------------------------------------------------
BLADE_MASS_REF = 0.15  # kg — brauchbare Schlacht-Abschläge ≥ ~150 g
BLADE_HANDLE_MAX = 1.0  # kg — einhändig führbares Schneidwerkzeug ≤ ~1 kg
CUT_WORK_J_BASE = 15.0
CUT_WORK_J_PER_EFFORT = 35.0  # CUT_WORK_J = 15 + 35·effort

# --- Essen (B4/B5) --------------------------------------------------------------
BITE_MASS_KG = 0.3
MIN_NUTRITION_EDIBLE = 0.02  # darunter wirkungslos (Steinbeißen = No-op)
TOX_DAMAGE_PER_KG = 20.0

# --- Verwesung (B3.4; Vollzug in phys_objects.tick_decay) ----------------------
DECAY_HALF_LIFE_DAYS = 10.0
DECAY_RATE = math.log(2.0) / (DECAY_HALF_LIFE_DAYS * TICKS_PER_DAY)  # ≈ 2.888e-4
TOX_SPOILAGE_PER_TICK = 5e-4
TOX_SPOILAGE_CAP = 0.6
DECAY_MOISTURE_MIN = 0.5  # nur feuchte, nahrhafte Stoffe verwesen (Weichgewebe)

# Vom Realitäts-Gate geprüfte Aktions-/Kopplungs-Parameter (kind "action").
CALIBRATED_ACTION_PARAMS = (
    "sim_energy_per_kcal",
    "muscle_efficiency",
    "v_max_strike",
    "blade_mass_factor",
    "cut_work",
    "bite",
    "toxin_damage",
    "decay",
    "spoilage",
)

cal(
    "action",
    "sim_energy_per_kcal",
    "Kopplung kcal↔Sim-Energie: 0.032 Sim-Energie/kcal — eine 1-kg-Fleischmahlzeit "
    "(0.35·4000 = 1400 kcal) ergibt ≈ 45 ≙ v1 MEAT_ENERGY; Gate prüft das PRODUKT "
    "nutrition-Konvention × Kopplung (± 1)",
    "USDA-Nährwerttabellen + v1-Energieökonomie (MEAT_ENERGY 45, MAX_ENERGY 240)",
)
cal(
    "action",
    "muscle_efficiency",
    "Brutto-Wirkungsgrad Skelettmuskel 0.25 (20–25 %); mechanische Arbeit kostet "
    "joules/0.25/4184 kcal ≙ ×0.032 Sim-Energie (200 Schläge à 45 J ≈ 0.3 Sim-Energie — "
    "bewusst klein, der reale Begrenzer ist die Ermüdung)",
    "Arbeitsphysiologie: Wirkungsgrad Muskelarbeit",
)
cal(
    "action",
    "v_max_strike",
    "Maximale Schlag-Endgeschwindigkeit 14 m/s; gelieferte Energie ≤ ½·m·v² — ein "
    "0.05-kg-Kiesel liefert damit max ~4.9 J (kein Flint-Knacken über Kiesel-Exploit), "
    "ein 0.5–2-kg-Schlagstein die vollen 49–50 J",
    "Biomechanik Hammerschlag/Knapping: Endgeschwindigkeit 10–15 m/s",
)
cal(
    "action",
    "blade_mass_factor",
    "Klingen-Massen-Faktor beidseitig begrenzt: clamp(sqrt(m/0.15), 0.2, 1) · "
    "clamp((2−m)/1, 0.2, 1) — brauchbare Abschläge ≥ ~150 g (BLADE_MASS_REF), einhändig "
    "führbar ≤ ~1 kg (BLADE_HANDLE_MAX); 20-g-Splitter und 4-kg-Brocken schneiden nur "
    "mit Faktor ≤ 0.4 bzw. 0.2 (kein 8-kg-Skalpell)",
    "Ethnographie/experimentelle Archäologie der Lithik: Handhabbarkeit von Schneidwerkzeug",
)
cal(
    "action",
    "cut_work",
    "Schneidearbeit CUT_WORK_J = 15 + 35·effort Joule pro Schneidevorgang, über den "
    "Ermüdungspfad (FATIGUE_PER_JOULE) und die Arbeits-Metabolik verbucht; der "
    "Schnitt-Ertrag skaliert mit Effort: yield × (0.5 + 0.5·effort)",
    "Arbeitsphysiologie: repetitives Schneiden/Zerwirken — Zerlegegeschwindigkeit "
    "skaliert mit aufgebrachter Kraft",
)
cal(
    "action",
    "bite",
    "Ein Biss pro Tick: bite = min(0.3 kg, Restmasse); Objekte mit nutrition ≤ 0.02 sind "
    "wirkungslos (Steinbeißen = No-op)",
    "Ernährungsphysiologie: Bolus-/Bissgrößen, Größenordnung Mahlzeit ~0.3–1 kg",
)
cal(
    "action",
    "toxin_damage",
    "Toxin-Schaden 20 Health/kg·toxicity: 0.3 kg stark toxischen Materials (0.8) ≈ 5 Health "
    "≙ spürbar, wiederholt tödlich; verdorbenes Fleisch (0.6) ≈ 3.6 Health/Biss",
    "Lebensmittelhygiene/Toxikologie: Dosis-Wirkung roher, verdorbener Tierprodukte",
)
cal(
    "action",
    "decay",
    "Verwesung: mass·(1−λ) und nutrition·(1−λ) pro Tick mit λ = ln(2)/(10 Tage · 240 Ticks) "
    "≈ 2.89e-4 (Weichgewebe-Halbwertszeit temperiert ~7–14 Tage); wirkt nur auf feuchte, "
    "nahrhafte Stoffe (moisture ≥ 0.5, nutrition > 0 — Stein/Holz verwesen nicht)",
    "Forensische Taphonomie: Weichgewebe-Dekomposition",
)
cal(
    "action",
    "spoilage",
    "Verderb: toxicity += 5e-4 pro Tick, gekappt bei 0.6 — rohes Fleisch wird bei "
    "Umgebungstemperatur in ~3–5 Tagen gefährlich (Kappe nach ~5 Tagen ≙ 1200 Ticks)",
    "Lebensmittelhygiene: Verderb roher Tierprodukte ungekühlt",
)


# ---------------------------------------------------------------------------
# Aktions-Mechanik (B4): eine Manipulation pro Agent und Tick; Ergebnis-Objekt
# für Logging/Neugier/Tests — kein Zugriff auf Gehirn oder Belohnung.
# ---------------------------------------------------------------------------
@dataclass
class ActionResult:
    ok: bool
    verb: str
    reason: str = ""
    energy_delta_sim: float = 0.0  # Sim-Energie-Delta des Agenten (Essen +, Arbeit −)
    health_delta: float = 0.0  # Toxin-Schaden (≤ 0)
    fragments: list = field(default_factory=list)
    extracted: PhysObject | None = None
    remainder: PhysObject | None = (
        None  # cut: physische Fortsetzung des Ziels (3b, Causal-Target C4)
    )
    bite_kg: float = 0.0
    work_j: float = 0.0


def do_grasp(body: Body, hands: Hands, layer, pos, target: PhysObject) -> ActionResult:
    """Boden-Objekt im Chebyshev-Radius 1 greifen. Scheitert als No-op bei
    vollen Händen (MAX_HELD) oder Massen-Budget-Überschreitung."""
    pos = (int(pos[0]), int(pos[1]))
    tpos = layer.position_of(target)
    if tpos is None:
        return ActionResult(ok=False, verb="grasp", reason="not_on_ground")
    if max(abs(tpos[0] - pos[0]), abs(tpos[1] - pos[1])) > 1:
        return ActionResult(ok=False, verb="grasp", reason="out_of_reach")
    if not hands.can_grasp(target, body):
        return ActionResult(ok=False, verb="grasp", reason="hands_full_or_too_heavy")
    layer.remove(target)
    hands.grasp(target, body)
    return ActionResult(ok=True, verb="grasp")


def do_release(body: Body, hands: Hands, layer, pos, held: PhysObject) -> ActionResult:
    """Gehaltenes Objekt an der eigenen Position ablegen (ledger-neutral)."""
    pos = (int(pos[0]), int(pos[1]))
    if held not in hands.held:
        return ActionResult(ok=False, verb="release", reason="not_held")
    hands.release(held)
    layer.add(held, pos)
    return ActionResult(ok=True, verb="release")


def enforce_carry_budget(body: Body, hands: Hands, layer, pos) -> list:
    """Überlast-Drop (B4): übersteigt die gehaltene Masse die AKTUELLE
    Tragkapazität (z. B. weil Ermüdung sie senkt), fällt das jeweils schwerste
    Objekt, bis das Budget wieder eingehalten ist. Zu Tick-Beginn aufrufen."""
    pos = (int(pos[0]), int(pos[1]))
    dropped: list = []
    while hands.held and hands.carried_mass_kg() > body.carry_capacity_kg():
        schwerstes = max(hands.held, key=lambda o: o.mass)
        hands.release(schwerstes)
        layer.add(schwerstes, pos)
        dropped.append(schwerstes)
    return dropped


def do_strike(
    body: Body,
    hands: Hands,
    layer,
    pos,
    striker_held: PhysObject,
    target: PhysObject,
    effort: float,
    rng: random.Random,
) -> ActionResult:
    """Schlag mit gehaltenem Schläger auf ein Boden-Objekt an eigener Position
    oder auf das andere gehaltene Objekt.

    Gelieferte Energie E = min(body.strike_energy_j(effort), ½·m_striker·V_MAX²)
    — die Kappung schließt den Kiesel-Exploit (leichter Schläger kann nicht
    mehr kinetische Energie tragen als ½mv²). body.exert_strike bindet
    verpflichtend: JEDER ausgeführte Schlag ermüdet, auch ohne Bruch.
    Ungültiges Ziel (nicht erreichbar / Schläger nicht in der Hand) ist ein
    No-op OHNE Exert — es wurde nicht geschlagen.
    """
    pos = (int(pos[0]), int(pos[1]))
    if striker_held not in hands.held:
        return ActionResult(ok=False, verb="strike", reason="striker_not_held")
    if target is striker_held:
        return ActionResult(ok=False, verb="strike", reason="target_is_striker")
    target_held = target in hands.held
    if not target_held and layer.position_of(target) != pos:
        return ActionResult(ok=False, verb="strike", reason="target_out_of_reach")

    effort = min(max(float(effort), 0.0), 1.0)
    impact_j = min(body.strike_energy_j(effort), 0.5 * striker_held.mass * V_MAX_STRIKE**2)
    body.exert_strike(impact_j)  # verpflichtende exert↔strike-Bindung (B4)
    energy_delta = -impact_j * WORK_SIM_ENERGY_PER_JOULE  # Arbeits-Metabolik (B5)

    result = strike(target, striker_held, impact_j, rng)
    if not result.fractured:
        return ActionResult(
            ok=True,
            verb="strike",
            reason="no_fracture",
            energy_delta_sim=energy_delta,
            work_j=impact_j,
        )
    # Ziel durch Fragmente ersetzen (Masse exakt erhalten, ledger-neutral);
    # Fragmente fallen zu Boden — 2–3 Bruchstücke passen in keine Hand.
    if target_held:
        hands.release(target)
    else:
        layer.remove(target)
    for frag in result.fragments:
        layer.add(frag, pos)
        layer.discovery.register(frag.props)
    layer.metrics["fragments_total"] += len(result.fragments)
    return ActionResult(
        ok=True,
        verb="strike",
        energy_delta_sim=energy_delta,
        fragments=result.fragments,
        work_j=impact_j,
    )


def blade_factor(blade: PhysObject | None) -> float:
    """Klingen-Massen-Faktor (B4), beidseitig begrenzt: zu leichte Splitter
    (< ~150 g) und unhandliche Brocken (> ~1 kg einhändig) schneiden schlecht.
    blade_factor = clamp(sqrt(m/BLADE_MASS_REF), 0.2, 1) · clamp((2·HANDLE_MAX − m)/HANDLE_MAX, 0.2, 1)
    Bloße Hand: 1.0 (kein Werkzeug, kein Massen-Faktor — der Hand-Malus steckt
    in BARE_HAND_SHARPNESS der Prozess-Physik)."""
    if blade is None:
        return 1.0
    leicht = min(max(math.sqrt(blade.mass / BLADE_MASS_REF), 0.2), 1.0)
    handlich = min(max((BLADE_HANDLE_MAX * 2.0 - blade.mass) / BLADE_HANDLE_MAX, 0.2), 1.0)
    return leicht * handlich


def do_cut(
    body: Body,
    hands: Hands,
    layer,
    pos,
    blade_held: PhysObject | None,
    target: PhysObject,
    effort: float,
) -> ActionResult:
    """Schneiden mit Klinge aus der Hand oder bloßer Hand (blade_held=None).

    Der Ertrag der Prozess-Physik (processes.cut) wird mit dem beidseitigen
    Klingen-Massen-Faktor UND dem Effort-Faktor (0.5 + 0.5·effort) skaliert
    (Spec B4 Rev. 4: Zerlegegeschwindigkeit skaliert mit aufgebrachter Kraft);
    Masse bleibt exakt erhalten (der nicht abgetrennte Anteil bleibt im
    remainder). Schneiden kostet Arbeit CUT_WORK_J = 15 + 35·effort über
    denselben Ermüdungspfad wie der Schlag. Platzierung: extracted fällt zu
    Boden; remainder ersetzt das Ziel an dessen Ort (Hand bleibt Hand, Boden
    bleibt Boden)."""
    pos = (int(pos[0]), int(pos[1]))
    if blade_held is not None and blade_held not in hands.held:
        return ActionResult(ok=False, verb="cut", reason="blade_not_held")
    if target is blade_held:
        return ActionResult(ok=False, verb="cut", reason="target_is_blade")
    target_held = target in hands.held
    if not target_held and layer.position_of(target) != pos:
        return ActionResult(ok=False, verb="cut", reason="target_out_of_reach")

    effort = min(max(float(effort), 0.0), 1.0)
    work_j = CUT_WORK_J_BASE + CUT_WORK_J_PER_EFFORT * effort
    body.exert_strike(work_j)  # derselbe Ermüdungspfad (FATIGUE_PER_JOULE)
    energy_delta = -work_j * WORK_SIM_ENERGY_PER_JOULE

    result = cut(target, blade_held)
    if result.extracted is None:
        return ActionResult(
            ok=True, verb="cut", reason="no_yield", energy_delta_sim=energy_delta, work_j=work_j
        )
    extracted, remainder = result.extracted, result.remainder
    # Ertrag = Prozess-Yield × Klingen-Massen-Faktor × Effort-Faktor (B4 Rev. 4);
    # die Kosten (CUT_WORK_J, Ermüdung) skalieren bereits oben mit Effort.
    faktor = blade_factor(blade_held) * (0.5 + 0.5 * effort)
    if faktor < 1.0:
        skaliert = extracted.mass * faktor
        if skaliert <= 1e-9:
            return ActionResult(
                ok=True,
                verb="cut",
                reason="no_yield",
                energy_delta_sim=energy_delta,
                work_j=work_j,
            )
        extracted.mass = skaliert
    remainder.mass = target.mass - extracted.mass  # Masse exakt erhalten

    if target_held:
        hands.held[hands.held.index(target)] = remainder  # Masse sinkt strikt → kein Budget-Check
    else:
        layer.remove(target)
        layer.add(remainder, pos)
    layer.add(extracted, pos)
    layer.discovery.register(extracted.props)
    layer.metrics["cuts_with_tool" if blade_held is not None else "cuts_bare_hand"] += 1
    return ActionResult(
        ok=True,
        verb="cut",
        energy_delta_sim=energy_delta,
        extracted=extracted,
        remainder=remainder,
        work_j=work_j,
    )


def do_eat(body: Body, hands: Hands, layer, pos, target: PhysObject) -> ActionResult:
    """Ein Biss pro Tick: bite = min(BITE_MASS_KG, Restmasse). Energie-Kopplung
    (B5): energy += nutrition·4000·bite·SIM_ENERGY_PER_KCAL; Toxin-Schaden:
    health −= toxicity·bite·TOX_DAMAGE_PER_KG. Objekte mit nutrition ≤ 0.02
    sind wirkungslos (Steinbeißen = No-op). Gegessene Masse fließt bilanziert
    in ledger['eaten']; vollständig verzehrte Objekte verschwinden."""
    pos = (int(pos[0]), int(pos[1]))
    target_held = target in hands.held
    if not target_held and layer.position_of(target) != pos:
        return ActionResult(ok=False, verb="eat", reason="target_out_of_reach")
    nutrition = float(target.props[IDX2["nutrition"]])
    if nutrition <= MIN_NUTRITION_EDIBLE:
        return ActionResult(ok=False, verb="eat", reason="not_edible")

    bite = min(BITE_MASS_KG, target.mass)
    if target.mass - bite <= 1e-9:
        bite = target.mass  # Krümel-Reste mitessen statt Masse zu verlieren
    energy_gain = nutrition * KCAL_PER_KG_PER_NUTRITION * bite * SIM_ENERGY_PER_KCAL
    toxicity = float(target.props[IDX2["toxicity"]])
    health_delta = -(toxicity * bite * TOX_DAMAGE_PER_KG)

    if bite >= target.mass:  # vollständig verzehrt
        if target_held:
            hands.release(target)
        else:
            layer.remove(target)
    else:
        target.mass -= bite
    layer.ledger["eaten"] += bite
    kcal_by_kind = layer.metrics["kcal_eaten_by_kind"]
    kcal_by_kind[target.kind] = (
        kcal_by_kind.get(target.kind, 0.0) + nutrition * KCAL_PER_KG_PER_NUTRITION * bite
    )
    return ActionResult(
        ok=True, verb="eat", energy_delta_sim=energy_gain, health_delta=health_delta, bite_kg=bite
    )
