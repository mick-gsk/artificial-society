"""Verkörperte Aktions-Mechanik der Physik v2 (Plan 3a, Spec B4/B5).

Dieses Modul bindet Body/Hands/Prozesse/ObjectLayer — es kennt weder Gehirn
noch Belohnung. Alle Konstanten sind real geankert (cal-Einträge unten,
Gate-pflichtig über CALIBRATED_ACTION_PARAMS). Energie-Konvention: nutrition
ist kcal/100 g ÷ 400 (raw_meat 0.35 ≙ 140 kcal/100 g), gekoppelt an die
v1-Energieskala über SIM_ENERGY_PER_KCAL (1-kg-Fleischmahlzeit ≙ MEAT_ENERGY 45).
"""

from __future__ import annotations

import math

from artificial_society.environment.daynight import TICKS_PER_DAY

from .calibration import cal

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
