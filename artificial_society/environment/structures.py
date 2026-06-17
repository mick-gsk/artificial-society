"""
Struktur-Effekte auf Zell- und Agenten-Ebene
---------------------------------------------
Strukturen (Camp, Farm, Brunnen) wurden bisher gebaut aber hatten
ausser einem kleinen Ressourcenbonus in resources.py keine
wahrnehmbaren Auswirkungen fuer den Agenten.

Jetzt:
- Camp:   reduziert Kaelteschaden, reduziert Krankheitsrisiko durch Wetterschutz
- Farm:   erhoeht plant_food-Wachstum der Zelle (bereits in resources.py)
          NEU: Agenten auf der Zelle sammeln schneller (Heimfeld-Bonus)
- Brunnen: reduziert Dehydration, leichte Krankheits-Pufferung

Alle Effekte wirken als passiv-physikalische Konsequenzen auf energy/health/hydration.
Kein expliziter Reward -- das Gehirn lernt Strukturen zu bauen weil
die Vitalkurven danach besser laufen.
"""

CAMP_COLD_REDUCTION     = 0.65   # Kaelteschaden-Faktor mit Camp (35% weniger)
CAMP_DISEASE_REDUCTION  = 0.50   # Krankheitsexpositions-Faktor mit Camp
CAMP_RAIN_PROTECTION    = 0.40   # Feuchtigkeitsverlust-Reduktion
WELL_HYDRATION_BONUS    = 0.25   # Zusaetzliche Hydration pro Tick auf Brunnen-Zelle
WELL_DISEASE_REDUCTION  = 0.30   # Sauberes Wasser reduziert Krankheitsrisiko
FARM_FORAGE_BONUS       = 0.22   # Sammel-Effizienz-Bonus auf Farm-Zelle

# Reward fuer erfolgreichen Bau -- NICHT als "Bauen ist gut"-Label,
# sondern als unmittelbare physische Befriedigung:
# Camp bauen = sofort warmer/sicherer Ort (wie eine Houeste finden)
# Brunnen = Durst unmittelbar loeschbar
# Farm = erste Ernte gibt Energie
BUILD_ENERGY_COST = {
    'camp':   12.0,   # Bau kostet Energie (Arbeit)
    'well':   10.0,
    'farm':    8.0,
}


def apply_structure_effects(agent, cell: dict):
    """
    Passt physikalische Kosten des Agenten basierend auf vorhandenen
    Strukturen der aktuellen Zelle an. Wird in apply_environmental_effects
    aufgerufen BEVOR Schaden berechnet wird.

    Gibt dict mit Modifikatoren zurueck die der Aufrufer anwendet.
    """
    structs = cell.get('structures', {})
    mods = {
        'cold_factor':     1.0,
        'disease_factor':  1.0,
        'hydration_bonus': 0.0,
        'forage_bonus':    0.0,
    }

    camp_level  = structs.get('camp',  0.0)
    well_level  = structs.get('well',  0.0)
    farm_level  = structs.get('farm',  0.0)

    if camp_level > 0:
        mods['cold_factor']    *= max(CAMP_COLD_REDUCTION, 1.0 - 0.35 * camp_level)
        mods['disease_factor'] *= max(CAMP_DISEASE_REDUCTION, 1.0 - 0.50 * camp_level)

    if well_level > 0:
        mods['hydration_bonus'] += WELL_HYDRATION_BONUS * well_level
        mods['disease_factor']  *= max(WELL_DISEASE_REDUCTION, 1.0 - 0.30 * well_level)

    if farm_level > 0:
        mods['forage_bonus'] += FARM_FORAGE_BONUS * farm_level

    return mods


def structure_feature_vector(cell: dict) -> list:
    """
    Gibt einen 3-dimensionalen Feature-Vektor fuer local_features() zurueck:
    [camp_level, well_level, farm_level] (alle 0..1)
    """
    structs = cell.get('structures', {})
    return [
        float(structs.get('camp',  0.0)),
        float(structs.get('well',  0.0)),
        float(structs.get('farm',  0.0)),
    ]
