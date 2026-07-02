"""Startmaterialien der Physik v2 — reale Stoffe, real kalibrierte Werte.

Kein Startmaterial ist ein Rezept-Endprodukt, keines startet mit sharpness > 0
(Schärfe entsteht nur durch Prozesse, z.B. Feuerstein schlagen). Werte sind
[0,1]-normiert nach den Ankern in calibration.py.
"""

from __future__ import annotations

from .calibration import cal
from .props import pv

MATERIALS_V2: dict = {
    "granite": pv(
        hardness=0.65,
        density=0.54,
        brittleness=0.45,
        tensile_strength=0.01,
        ignition_temp=1.0,
        melting_point=0.62,
        thermal_conductivity=0.28,
        moisture=0.02,
        grain_fineness=0.15,
    ),
    "flint": pv(
        hardness=0.70,
        density=0.52,
        brittleness=0.90,
        tensile_strength=0.01,
        ignition_temp=1.0,
        melting_point=0.84,
        thermal_conductivity=0.14,
        moisture=0.01,
        grain_fineness=0.95,
    ),
    "dry_wood": pv(
        hardness=0.25,
        density=0.13,
        brittleness=0.20,
        tensile_strength=0.10,
        flammability=0.80,
        ignition_temp=0.30,
        melting_point=1.0,
        thermal_conductivity=0.02,
        moisture=0.12,
        grain_fineness=0.30,
    ),
    "plant_fiber": pv(
        hardness=0.05,
        density=0.06,
        brittleness=0.05,
        tensile_strength=0.60,
        flammability=0.85,
        ignition_temp=0.25,
        melting_point=1.0,
        thermal_conductivity=0.01,
        nutrition=0.01,
        moisture=0.15,
        grain_fineness=0.50,
    ),
    "clay_moist": pv(
        hardness=0.10,
        density=0.40,
        brittleness=0.15,
        ignition_temp=1.0,
        melting_point=0.50,
        thermal_conductivity=0.10,
        toxicity=0.02,
        moisture=0.35,
        grain_fineness=0.80,
    ),
    "water": pv(
        density=0.20,
        ignition_temp=1.0,
        thermal_conductivity=0.06,
        moisture=1.0,
    ),
    "berries": pv(
        hardness=0.02,
        density=0.21,
        brittleness=0.10,
        flammability=0.05,
        ignition_temp=0.40,
        melting_point=1.0,
        thermal_conductivity=0.05,
        nutrition=0.13,
        toxicity=0.05,
        moisture=0.85,
        grain_fineness=0.60,
    ),
    "raw_meat": pv(
        hardness=0.05,
        density=0.21,
        brittleness=0.15,
        tensile_strength=0.02,
        flammability=0.05,
        ignition_temp=0.60,
        melting_point=1.0,
        thermal_conductivity=0.05,
        nutrition=0.35,
        toxicity=0.15,
        moisture=0.70,
        grain_fineness=0.40,
    ),
    "carcass": pv(
        hardness=0.10,
        density=0.20,
        brittleness=0.10,
        tensile_strength=0.10,
        flammability=0.05,
        ignition_temp=0.60,
        melting_point=1.0,
        thermal_conductivity=0.05,
        nutrition=0.30,
        toxicity=0.10,
        moisture=0.65,
        grain_fineness=0.30,
    ),
}

cal(
    "material",
    "granite",
    "Granit: Mohs ≈ 6.5, ρ ≈ 2700 kg/m³, grobkristallin, mäßig spröde — splittert unter "
    "starkem Schlag in stumpfe Bruchstücke",
    "Gesteinskunde-Standardwerte (Granit)",
)
cal(
    "material",
    "flint",
    "Feuerstein/Silex: Mohs ≈ 7, ρ ≈ 2600, kryptokristallin → muscheliger Bruch; klassisches "
    "Ausgangsmaterial für Klingen; im Rohzustand NICHT scharf",
    "Petrologie Silex; experimentelle Archäologie Feuersteinschlagen",
)
cal(
    "material",
    "dry_wood",
    "Lufttrockenes Holz: ρ ≈ 650, Zündtemp ≈ 300 °C, Zugfestigkeit längs ≈ 100 MPa, zäh "
    "(splittert nicht muschelig)",
    "Holztechnik-Tabellenwerte",
)
cal(
    "material",
    "plant_fiber",
    "Bastfaser (Hanf/Lein): Zugfestigkeit ≈ 600 MPa, sehr leicht, gut brennbar",
    "Werkstoffkunde Naturfasern",
)
cal(
    "material",
    "clay_moist",
    "Feuchter Ton: weich/plastisch, sintert ab ≈ 1000 °C, sehr feines Gefüge",
    "Keramik-Grundlagen",
)
cal(
    "material",
    "water",
    "Wasser: ρ = 1000 kg/m³ — Referenzstoff der Dichte- und Feuchteskala",
    "CRC Handbook",
)
cal(
    "material",
    "berries",
    "Beeren: ≈ 50 kcal/100 g, wasserreich, weich; leichte Rest-Toxizität (Wildsammlung)",
    "USDA Nährwerttabelle (Beeren)",
)
cal(
    "material",
    "raw_meat",
    "Rohes Muskelfleisch: ≈ 140 kcal/100 g, ≈ 70 % Wasser, zäh (niedrige Sprödigkeit), roh "
    "leicht keimbelastet",
    "USDA Nährwerttabellen; Lebensmittelhygiene",
)
cal(
    "material",
    "carcass",
    "Tierkadaver: Verbund aus Haut/Sehnen/Fleisch/Knochen — sehr zäh; Nährwert praktisch nur "
    "durch Zerteilen (Schneiden) erschließbar",
    "Zoologie/Jagdpraxis: Zerwirken",
)
