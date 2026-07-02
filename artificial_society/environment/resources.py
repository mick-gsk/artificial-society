import numpy as np

from artificial_society.environment.biomes import BIOME_BASE

DISEASE_POLLUTION_THRESHOLD = 36.0
MEAT_SPOIL_RATE = 0.04
CARCASS_DECAY = 0.14

# ------------------------------------------------------------------
# Ressourcenknappheit
# Biologisches Vorbild: In der Natur ist Nahrung der wichtigste
# Selektionsdruck. Zu viel Nahrung eliminiert jeden evolutionaeren
# Druck hin zu Kooperation, Werkzeugbau oder Territorialverhalten.
# Wir reduzieren plant_gain und senken die Initialmengen.
# Die Welt wird jetzt brutal: Einzelgaenger sterben in
# Duerre/Winter, Gruppen ueberleben.
#
# WICHTIG (Knappheit): Frueher lief der Nachwuchs ungebremst, bis
# eine UNBERUEHRTE Zelle ihren Standbestand auf ~1.0-1.2x der
# carrying_capacity hochgewachsen hatte (siehe regrow_cell-Clamp
# 1.45*capacity). Da die meisten Zellen nie befressen werden, sass
# die Welt damit dauerhaft an der Saettigung -> keine Knappheit.
# Loesung: logistische Decke. Der Nachwuchs wird gegen NULL gefahren,
# sobald plant_food sich dem scarce Zielbestand
# (SCARCITY_CEILING_FACTOR * capacity) naehert. Standbestand
# plateaut so bei einem Bruchteil der Kapazitaet; jeder Verbrauch
# drueckt die Zelle real darunter, Erholung ist langsam.
# ------------------------------------------------------------------
FOOD_SCARCITY_FACTOR = 0.50  # Pflanzenwachstum auf 50% reduziert
MEAT_SCARCITY_FACTOR = 0.55  # Fleischnachwuchs auf 55% reduziert
INITIAL_FOOD_FACTOR = 0.30  # Startwert stark gesenkt -> sofortiger Hungerdruck

# Logistische Decke: unberuehrte Zellen pendeln sich auf diesen
# Bruchteil der carrying_capacity ein (statt frueher ~1.0-1.2x).
# 0.35 => mittlerer Zell-food bleibt deutlich unter der Haelfte der
# Kapazitaet, auch wenn niemand befrisst.
SCARCITY_CEILING_FACTOR = 0.35
# Analog fuer Fleischnachwuchs (Beutetiere sind ohnehin knapper).
MEAT_CEILING_FACTOR = 0.25

# Biome-specific scarcity (Phase 4): harsh biomes plateau at a lower standing
# stock than fertile ones, so *where* an agent lives is a real selection
# pressure. These scale the logistic plant ceiling per biome;
# SCARCITY_CEILING_FACTOR is the fallback for any unlisted biome.
BIOME_SCARCITY_CEILING = {
    "forest": 0.35,
    "grassland": 0.33,
    "swamp": 0.32,
    "mountain": 0.22,
    "desert": 0.14,
    "water": 0.0,
}

# Plant growth peaks in a temperate band and falls off when a cell runs cold
# (winter / mountains) or scorching (desert / summer). Couples temperature ->
# regrowth so seasons and biomes both bite. Deliberately gentle: the floor keeps
# a temperate spring world (the equilibrium regime) almost untouched.
PLANT_TEMP_OPTIMUM = 18.0
PLANT_TEMP_TOLERANCE = 60.0
PLANT_TEMP_FLOOR = 0.5

# Cold cells read as more dangerous (exposure + scarce forage), so agents learn
# to avoid and migrate out of winter zones. This is perceptual -- danger is not
# directly lethal -- but it drives the migration that turns winter into a
# selection event.
COLD_DANGER_THRESHOLD = 6.0
COLD_DANGER_COEFF = 0.9


def biome_scarcity_ceiling(biome):
    return BIOME_SCARCITY_CEILING.get(biome, SCARCITY_CEILING_FACTOR)


def plant_temperature_factor(temperature):
    """Multiplier in [PLANT_TEMP_FLOOR, 1.0] for how well plants grow at this
    temperature; 1.0 at PLANT_TEMP_OPTIMUM, falling off linearly either side."""
    off = abs(temperature - PLANT_TEMP_OPTIMUM) / PLANT_TEMP_TOLERANCE
    return clamp(1.0 - off, PLANT_TEMP_FLOOR, 1.0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def initial_cell_state(biome):
    base = BIOME_BASE[biome]
    # Halbierte Startwerte erzeugen sofortigen Hungerdruck
    plant_food = float(
        base["food"]
        * INITIAL_FOOD_FACTOR
        * (1.3 if biome in ("forest", "grassland", "swamp") else 0.6)
    )
    meat_food = float(
        base["food"]
        * INITIAL_FOOD_FACTOR
        * (0.35 if biome in ("mountain", "forest", "swamp") else 0.12)
    )
    return {
        "food": float(base["food"] * INITIAL_FOOD_FACTOR),
        "plant_food": plant_food,
        "meat_food": meat_food,
        "water": float(base["water"]),
        "temperature": float(base["temperature"]),
        "danger": float(base["danger"]),
        "soil_fertility": float(base["soil_fertility"]),
        "pollution": 0.0,
        "usage_pressure": 0.0,
        "carrying_capacity": float(base["carrying_capacity"]),
        "spoilage": 0.0,
        "carcasses": 0.0,
        "disease": 0.0,
        "moisture": float(base["water"] * 0.7),
        "ash": 0.0,
        "disturbance": 0.0,
        "structures": {"camp": 0.0, "farm": 0.0, "well": 0.0},
        "biome": biome,
        "tick": 0,
    }


def apply_consumption(world, x, y, plant=0.0, meat=0.0, water=0.0):
    cell = world.get_cell(x, y)
    world.set_cell(x, y, "plant_food", clamp(cell["plant_food"] - plant, 0.0, 160.0))
    world.set_cell(x, y, "meat_food", clamp(cell["meat_food"] - meat, 0.0, 110.0))
    world.set_cell(x, y, "water", clamp(cell["water"] - water, 0.0, 100.0))
    world.set_cell(x, y, "food", clamp(cell["plant_food"] + cell["meat_food"] * 0.7, 0.0, 180.0))
    pressure = plant + meat * 1.3 + water * 0.7
    world.set_cell(
        x, y, "usage_pressure", clamp(cell["usage_pressure"] + pressure * 0.85, 0.0, 100.0)
    )
    world.set_cell(
        x, y, "pollution", clamp(cell["pollution"] + meat * 0.14 + pressure * 0.05, 0.0, 100.0)
    )
    world.set_cell(
        x, y, "soil_fertility", clamp(cell["soil_fertility"] - pressure * 0.04, 0.0, 100.0)
    )
    world.set_cell(x, y, "disturbance", clamp(cell["disturbance"] + pressure * 0.25, 0.0, 100.0))


def add_carcass(world, x, y, energy_value):
    cell = world.get_cell(x, y)
    # Energy conservation (Phase 4): a corpse is worth exactly `energy_value` of
    # harvestable food, split across the two consumable pools a forager can reach
    # (carcasses + meat_food). They must SUM to energy_value, not each receive it
    # -- otherwise a death mints ~1.45x energy now that the carcass-field bug is
    # fixed and the carcasses pool is actually edible. The meat_food share keeps a
    # fresh carcass visible through the derived `food` aggregate.
    world.set_cell(x, y, "carcasses", clamp(cell["carcasses"] + energy_value * 0.55, 0.0, 140.0))
    world.set_cell(x, y, "meat_food", clamp(cell["meat_food"] + energy_value * 0.45, 0.0, 110.0))
    world.set_cell(x, y, "spoilage", clamp(cell["spoilage"] + energy_value * 0.16, 0.0, 100.0))
    world.set_cell(x, y, "pollution", clamp(cell["pollution"] + energy_value * 0.12, 0.0, 100.0))
    world.set_cell(x, y, "danger", clamp(cell["danger"] + energy_value * 0.05, 0.0, 100.0))
    world.set_cell(
        x, y, "disturbance", clamp(cell["disturbance"] + energy_value * 0.08, 0.0, 100.0)
    )
    world.set_cell(x, y, "food", clamp(cell["plant_food"] + cell["meat_food"] * 0.7, 0.0, 180.0))


def maybe_build_structure(cell, resources):
    built = None
    if resources["wood"] >= 2 and resources["fiber"] >= 1 and cell["structures"]["camp"] < 1.0:
        resources["wood"] -= 2
        resources["fiber"] -= 1
        cell["structures"]["camp"] = 1.0
        built = "camp"
    elif resources["wood"] >= 2 and resources["stone"] >= 1 and cell["structures"]["well"] < 1.0:
        resources["wood"] -= 2
        resources["stone"] -= 1
        cell["structures"]["well"] = 1.0
        built = "well"
    elif resources["wood"] >= 1 and resources["fiber"] >= 2 and cell["structures"]["farm"] < 1.0:
        resources["wood"] -= 1
        resources["fiber"] -= 2
        cell["structures"]["farm"] = 1.0
        built = "farm"
    return built


def apply_event(world, x, y, event_strength):
    cell = world.get_cell(x, y)
    drought = event_strength.get("drought", 0.0)
    storm = event_strength.get("storm", 0.0)
    fire = event_strength.get("fire", 0.0)
    blight = event_strength.get("blight", 0.0)
    disturbance = event_strength.get("disturbance", 0.0)
    world.set_cell(
        x,
        y,
        "moisture",
        clamp(cell["moisture"] - 12.0 * drought + 8.0 * storm - 18.0 * fire, 0.0, 100.0),
    )
    world.set_cell(
        x, y, "water", clamp(cell["water"] - 0.35 * drought + 0.45 * storm - 0.1 * fire, 0.0, 100.0)
    )
    world.set_cell(
        x,
        y,
        "plant_food",
        clamp(cell["plant_food"] - 2.4 * drought - 5.8 * fire - 3.6 * blight, 0.0, 180.0),
    )
    world.set_cell(x, y, "meat_food", clamp(cell["meat_food"] - 1.8 * fire, 0.0, 110.0))
    world.set_cell(x, y, "ash", clamp(cell["ash"] + 7.0 * fire - 0.3 * storm, 0.0, 100.0))
    world.set_cell(
        x, y, "pollution", clamp(cell["pollution"] + 4.0 * fire + 1.6 * blight, 0.0, 100.0)
    )
    world.set_cell(x, y, "disease", clamp(cell["disease"] + 4.0 * blight + 1.2 * storm, 0.0, 100.0))
    world.set_cell(x, y, "disturbance", clamp(cell["disturbance"] + 9.0 * disturbance, 0.0, 100.0))
    world.set_cell(x, y, "food", clamp(cell["plant_food"] + cell["meat_food"] * 0.7, 0.0, 180.0))


def diffuse_step(world, x, y, neighbor_avgs):
    cell = world.get_cell(x, y)
    world.set_cell(
        x,
        y,
        "pollution",
        clamp(0.92 * cell["pollution"] + 0.08 * neighbor_avgs["pollution"], 0.0, 100.0),
    )
    world.set_cell(
        x, y, "disease", clamp(0.9 * cell["disease"] + 0.1 * neighbor_avgs["disease"], 0.0, 100.0)
    )
    world.set_cell(
        x,
        y,
        "moisture",
        clamp(0.88 * cell["moisture"] + 0.12 * neighbor_avgs["moisture"], 0.0, 100.0),
    )
    world.set_cell(x, y, "ash", clamp(0.9 * cell["ash"] + 0.1 * neighbor_avgs["ash"], 0.0, 100.0))
    world.set_cell(
        x,
        y,
        "disturbance",
        clamp(0.9 * cell["disturbance"] + 0.1 * neighbor_avgs["disturbance"], 0.0, 100.0),
    )


def regrow_cell(world, x, y, biome, season_state, weather_state, tick, event_strength):
    cell = world.get_cell(x, y)
    season_food = season_state.get("food_factor", 1.0)
    rain = weather_state.get("rain_map", 0.0)
    temp_shift = weather_state.get("temperature_shift", 0.0)
    wind = weather_state.get("wind", 0.0)
    base = BIOME_BASE[biome]
    world.set_cell(x, y, "tick", tick)
    world.set_cell(
        x,
        y,
        "temperature",
        clamp(
            base["temperature"] + temp_shift + season_state.get("temperature_shift", 0.0), -12, 52
        ),
    )

    usage = cell["usage_pressure"]
    pollution = cell["pollution"]
    fertility = cell["soil_fertility"]
    capacity = cell["carrying_capacity"]
    spoilage = cell["spoilage"]
    moisture = cell["moisture"]
    disturbance = cell["disturbance"]
    farm_bonus = 10.0 if cell["structures"]["farm"] else 0.0
    well_bonus = 10.0 if cell["structures"]["well"] else 0.0

    water_gain = (
        0.08 * rain + (0.12 if biome == "swamp" else 0.03) + 0.03 * cell["structures"]["well"]
    )
    fert_factor = 0.4 + fertility / 100.0
    moisture_factor = 0.35 + moisture / 100.0
    cap_factor = 0.35 + (capacity + farm_bonus) / 120.0
    # Stress-Floor auf 0.12 gesenkt (war 0.30): Zellen koennen jetzt
    # wirklich ausgelaugt werden – das erzwingt Migration und Kooperation
    stress_factor = max(0.12, 1.0 - pollution / 120.0 - usage / 180.0 - disturbance / 180.0)

    # KNAPPHEIT: plant_gain (Basisrate)
    plant_gain = (
        0.042
        * FOOD_SCARCITY_FACTOR
        * season_food
        * fert_factor
        * moisture_factor
        * cap_factor
        * stress_factor
        * (1.15 if biome == "forest" else 1.0)
    )
    plant_gain += (
        0.012 * cell["structures"]["farm"]
    )  # Farm hilft, aber nicht genug um Knappheit aufzuheben
    # Temperatur-Kopplung: kalte (Winter/Gebirge) oder gluehende (Wueste/Sommer)
    # Zellen wachsen schlechter nach. Sanft -> temperiertes Fruehlingsgleichgewicht
    # bleibt fast unberuehrt.
    plant_gain *= plant_temperature_factor(cell["temperature"])

    # KNAPPHEIT: meat_gain (Basisrate)
    meat_gain = (
        0.015
        * MEAT_SCARCITY_FACTOR
        * season_food
        * cap_factor
        * max(0.2, 1.0 - pollution / 150.0)
        * (1.1 if biome in ("mountain", "forest", "swamp") else 0.7)
    )

    # Logistische Decke (carrying-capacity-Kopplung):
    # Nachwuchs faellt linear gegen 0, je naeher der aktuelle
    # Standbestand am scarce Zielbestand liegt. Farm hebt das Ziel
    # leicht an. Ergebnis: eine unberuehrte Zelle waechst nur bis
    # ~SCARCITY_CEILING_FACTOR*capacity und bleibt dort knapp;
    # befressene Zellen erholen sich langsam von unten.
    plant_ceiling = biome_scarcity_ceiling(biome)
    plant_target = plant_ceiling * capacity + farm_bonus
    meat_target = MEAT_CEILING_FACTOR * capacity
    plant_headroom = max(0.0, 1.0 - cell["plant_food"] / max(1.0, plant_target))
    meat_headroom = max(0.0, 1.0 - cell["meat_food"] / max(1.0, meat_target))
    plant_gain *= plant_headroom
    meat_gain *= meat_headroom

    if biome == "desert":
        plant_gain *= 0.3
        meat_gain *= 0.22
        water_gain *= 0.25
    if biome == "mountain":
        plant_gain *= 0.55
        meat_gain *= 1.18
    if biome == "water":
        water_gain = 0.0
        plant_gain = 0.0
        meat_gain = 0.0

    decayed_meat = min(cell["meat_food"], spoilage * MEAT_SPOIL_RATE)
    world.set_cell(
        x, y, "meat_food", clamp(cell["meat_food"] - decayed_meat, 0.0, max(15.0, capacity))
    )
    world.set_cell(x, y, "carcasses", clamp(cell["carcasses"] - CARCASS_DECAY, 0.0, 140.0))
    world.set_cell(
        x,
        y,
        "spoilage",
        clamp(
            cell["spoilage"]
            + decayed_meat * 0.85
            - 0.05 * rain
            - 0.04 * cell["structures"]["camp"],
            0.0,
            100.0,
        ),
    )
    world.set_cell(
        x,
        y,
        "disease",
        clamp(
            cell["disease"]
            + max(0.0, pollution - DISEASE_POLLUTION_THRESHOLD) * 0.025
            + cell["spoilage"] * 0.03
            + cell["carcasses"] * 0.012
            - 0.05 * rain
            - 0.03 * cell["structures"]["well"],
            0.0,
            100.0,
        ),
    )
    # Harte Decke jetzt knapp ueber dem scarce Zielbestand (war 1.45*capacity).
    # So kann selbst ein temporaerer Ueberschuss (Carcass-Schub, Regen)
    # die Zelle nicht in Ueberfluss kippen lassen.
    plant_hard_cap = max(20.0, plant_ceiling * capacity * 1.25 + farm_bonus)
    world.set_cell(
        x,
        y,
        "plant_food",
        clamp(
            cell["plant_food"] + plant_gain - 0.012 * wind - 0.02 * cell["ash"], 0.0, plant_hard_cap
        ),
    )
    world.set_cell(
        x, y, "meat_food", clamp(cell["meat_food"] + meat_gain, 0.0, max(15.0, capacity))
    )
    world.set_cell(
        x,
        y,
        "water",
        clamp(cell["water"] + water_gain + 0.05 * cell["structures"]["well"], 0.0, 100.0),
    )
    world.set_cell(
        x,
        y,
        "moisture",
        clamp(
            cell["moisture"]
            + 0.2 * cell["water"] / 100.0
            + 0.35 * rain
            + 0.1 * well_bonus
            - 0.03 * abs(cell["temperature"] - 20),
            0.0,
            100.0,
        ),
    )
    world.set_cell(
        x,
        y,
        "soil_fertility",
        clamp(
            cell["soil_fertility"]
            + 0.05 * rain
            - 0.03 * pollution
            + 0.015 * max(0.0, 60 - usage)
            + 0.03 * cell["carcasses"]
            + 0.04 * cell["ash"],
            0.0,
            100.0,
        ),
    )
    world.set_cell(
        x,
        y,
        "pollution",
        clamp(
            cell["pollution"]
            - 0.03 * rain
            - 0.015 * max(0.0, fertility - 25)
            + 0.02 * cell["spoilage"]
            - 0.02 * cell["structures"]["camp"],
            0.0,
            100.0,
        ),
    )
    world.set_cell(
        x, y, "ash", clamp(cell["ash"] - 0.06 * rain - 0.03 * moisture / 100.0, 0.0, 100.0)
    )
    world.set_cell(x, y, "usage_pressure", clamp(cell["usage_pressure"] * 0.91, 0.0, 100.0))
    world.set_cell(x, y, "disturbance", clamp(cell["disturbance"] * 0.94, 0.0, 100.0))
    apply_event(world, x, y, event_strength)
    world.set_cell(x, y, "food", clamp(cell["plant_food"] + cell["meat_food"] * 0.7, 0.0, 180.0))
    world.set_cell(
        x,
        y,
        "danger",
        clamp(
            base["danger"]
            + weather_state.get("storm_risk", 0.0) * 20
            + abs(cell["temperature"] - 20) * 0.5
            + max(0.0, COLD_DANGER_THRESHOLD - cell["temperature"]) * COLD_DANGER_COEFF
            + cell["pollution"] * 0.16
            + cell["disease"] * 0.12
            + cell["disturbance"] * 0.18,
            0.0,
            100.0,
        ),
    )


# ---------------------------------------------------------------------------
# Vectorized grid regrowth (perf Tier 1)
#
# regrow_grid is a statement-for-statement transliteration of the per-cell
# regrow_cell + apply_event pair into whole-grid numpy expressions. The
# per-cell semantics (which reads see original vs freshly written values) are
# preserved exactly, and since neither function has cross-cell reads or RNG,
# the result is bit-identical to running the scalar pair over every cell.
# The scalar functions above stay as the reference oracle (see
# tests/environment/test_vectorized_equivalence.py) and for unit-test fakes.
# ---------------------------------------------------------------------------


def regrow_grid(world, season_state, weather_state, tick, event_fields):
    F = world.F
    S = world.S
    bio = world._bio

    season_food = season_state.get("food_factor", 1.0)
    rain = weather_state.get("rain_map", 0.0)
    temp_shift = weather_state.get("temperature_shift", 0.0)
    wind = weather_state.get("wind", 0.0)

    # Originals captured up front, mirroring regrow_cell's local snapshot and
    # its reads of not-yet-updated fields.
    usage = F["usage_pressure"].copy()
    pollution = F["pollution"].copy()
    fertility = F["soil_fertility"].copy()
    capacity = F["carrying_capacity"]  # never written here
    spoilage = F["spoilage"].copy()
    moisture = F["moisture"].copy()
    disturbance = F["disturbance"].copy()
    plant_food0 = F["plant_food"].copy()
    meat_food0 = F["meat_food"].copy()
    ash0 = F["ash"].copy()
    water0 = F["water"].copy()
    disease0 = F["disease"].copy()
    carcasses0 = F["carcasses"].copy()

    farm = S["farm"]
    camp = S["camp"]
    well = S["well"]
    farm_bonus = np.where(farm != 0.0, 10.0, 0.0)
    well_bonus = np.where(well != 0.0, 10.0, 0.0)

    world.tick_grid[:] = tick
    F["temperature"] = np.clip(
        bio["base_temperature"] + temp_shift + season_state.get("temperature_shift", 0.0), -12, 52
    )
    temperature = F["temperature"]

    water_gain = 0.08 * rain + np.where(bio["is_swamp"], 0.12, 0.03) + 0.03 * well
    fert_factor = 0.4 + fertility / 100.0
    moisture_factor = 0.35 + moisture / 100.0
    cap_factor = 0.35 + (capacity + farm_bonus) / 120.0
    stress_factor = np.maximum(0.12, 1.0 - pollution / 120.0 - usage / 180.0 - disturbance / 180.0)

    plant_gain = (
        0.042
        * FOOD_SCARCITY_FACTOR
        * season_food
        * fert_factor
        * moisture_factor
        * cap_factor
        * stress_factor
        * np.where(bio["is_forest"], 1.15, 1.0)
    )
    plant_gain = plant_gain + 0.012 * farm
    # plant_temperature_factor, vectorized (clamp == clip for finite values)
    plant_gain = plant_gain * np.clip(
        1.0 - np.abs(temperature - PLANT_TEMP_OPTIMUM) / PLANT_TEMP_TOLERANCE,
        PLANT_TEMP_FLOOR,
        1.0,
    )

    meat_gain = (
        0.015
        * MEAT_SCARCITY_FACTOR
        * season_food
        * cap_factor
        * np.maximum(0.2, 1.0 - pollution / 150.0)
        * np.where(bio["is_meat_biome"], 1.1, 0.7)
    )

    plant_ceiling = bio["plant_ceiling"]
    plant_target = plant_ceiling * capacity + farm_bonus
    meat_target = MEAT_CEILING_FACTOR * capacity
    plant_headroom = np.maximum(0.0, 1.0 - plant_food0 / np.maximum(1.0, plant_target))
    meat_headroom = np.maximum(0.0, 1.0 - meat_food0 / np.maximum(1.0, meat_target))
    plant_gain = plant_gain * plant_headroom
    meat_gain = meat_gain * meat_headroom

    plant_gain = np.where(bio["is_desert"], plant_gain * 0.3, plant_gain)
    meat_gain = np.where(bio["is_desert"], meat_gain * 0.22, meat_gain)
    water_gain = np.where(bio["is_desert"], water_gain * 0.25, water_gain)
    plant_gain = np.where(bio["is_mountain"], plant_gain * 0.55, plant_gain)
    meat_gain = np.where(bio["is_mountain"], meat_gain * 1.18, meat_gain)
    water_gain = np.where(bio["is_water"], 0.0, water_gain)
    plant_gain = np.where(bio["is_water"], 0.0, plant_gain)
    meat_gain = np.where(bio["is_water"], 0.0, meat_gain)

    meat_cap = np.maximum(15.0, capacity)
    decayed_meat = np.minimum(meat_food0, spoilage * MEAT_SPOIL_RATE)
    F["meat_food"] = np.clip(meat_food0 - decayed_meat, 0.0, meat_cap)
    F["carcasses"] = np.clip(carcasses0 - CARCASS_DECAY, 0.0, 140.0)
    F["spoilage"] = np.clip(
        spoilage + decayed_meat * 0.85 - 0.05 * rain - 0.04 * camp, 0.0, 100.0
    )
    F["disease"] = np.clip(
        disease0
        + np.maximum(0.0, pollution - DISEASE_POLLUTION_THRESHOLD) * 0.025
        + F["spoilage"] * 0.03
        + F["carcasses"] * 0.012
        - 0.05 * rain
        - 0.03 * well,
        0.0,
        100.0,
    )
    plant_hard_cap = np.maximum(20.0, plant_ceiling * capacity * 1.25 + farm_bonus)
    F["plant_food"] = np.clip(
        plant_food0 + plant_gain - 0.012 * wind - 0.02 * ash0, 0.0, plant_hard_cap
    )
    F["meat_food"] = np.clip(F["meat_food"] + meat_gain, 0.0, meat_cap)
    F["water"] = np.clip(water0 + water_gain + 0.05 * well, 0.0, 100.0)
    F["moisture"] = np.clip(
        moisture
        + 0.2 * F["water"] / 100.0
        + 0.35 * rain
        + 0.1 * well_bonus
        - 0.03 * np.abs(temperature - 20),
        0.0,
        100.0,
    )
    F["soil_fertility"] = np.clip(
        fertility
        + 0.05 * rain
        - 0.03 * pollution
        + 0.015 * np.maximum(0.0, 60 - usage)
        + 0.03 * F["carcasses"]
        + 0.04 * ash0,
        0.0,
        100.0,
    )
    F["pollution"] = np.clip(
        pollution
        - 0.03 * rain
        - 0.015 * np.maximum(0.0, fertility - 25)
        + 0.02 * F["spoilage"]
        - 0.02 * camp,
        0.0,
        100.0,
    )
    F["ash"] = np.clip(ash0 - 0.06 * rain - 0.03 * moisture / 100.0, 0.0, 100.0)
    F["usage_pressure"] = np.clip(usage * 0.91, 0.0, 100.0)
    F["disturbance"] = np.clip(disturbance * 0.94, 0.0, 100.0)

    # apply_event, vectorized (reads the freshly written values, like the
    # scalar call placed after the regrowth writes)
    drought = event_fields["drought"]
    storm = event_fields["storm"]
    fire = event_fields["fire"]
    blight = event_fields["blight"]
    dist_ev = event_fields["disturbance"]
    F["moisture"] = np.clip(
        F["moisture"] - 12.0 * drought + 8.0 * storm - 18.0 * fire, 0.0, 100.0
    )
    F["water"] = np.clip(F["water"] - 0.35 * drought + 0.45 * storm - 0.1 * fire, 0.0, 100.0)
    F["plant_food"] = np.clip(
        F["plant_food"] - 2.4 * drought - 5.8 * fire - 3.6 * blight, 0.0, 180.0
    )
    F["meat_food"] = np.clip(F["meat_food"] - 1.8 * fire, 0.0, 110.0)
    F["ash"] = np.clip(F["ash"] + 7.0 * fire - 0.3 * storm, 0.0, 100.0)
    F["pollution"] = np.clip(F["pollution"] + 4.0 * fire + 1.6 * blight, 0.0, 100.0)
    F["disease"] = np.clip(F["disease"] + 4.0 * blight + 1.2 * storm, 0.0, 100.0)
    F["disturbance"] = np.clip(F["disturbance"] + 9.0 * dist_ev, 0.0, 100.0)
    F["food"] = np.clip(F["plant_food"] + F["meat_food"] * 0.7, 0.0, 180.0)

    # trailing food + danger updates from regrow_cell
    F["food"] = np.clip(F["plant_food"] + F["meat_food"] * 0.7, 0.0, 180.0)
    F["danger"] = np.clip(
        bio["base_danger"]
        + weather_state.get("storm_risk", 0.0) * 20
        + np.abs(F["temperature"] - 20) * 0.5
        + np.maximum(0.0, COLD_DANGER_THRESHOLD - F["temperature"]) * COLD_DANGER_COEFF
        + F["pollution"] * 0.16
        + F["disease"] * 0.12
        + F["disturbance"] * 0.18,
        0.0,
        100.0,
    )
