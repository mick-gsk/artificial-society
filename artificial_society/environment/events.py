"""Physics-driven world-disturbance events (audit group 8).

Replaces the old timer/dice spawner (``tick % 55 == 0`` + flat 1.5% chance at
a uniformly random position) with genesis and lifecycles that follow the
world state, so disturbances behave as physically as the field model allows:

- STORM   forms when the weather system's storm_risk is high, tracks across
          the map along a persistent wind vector and decays slowly; the
          existing field application already models its rain (+moisture,
          +water). Lightning strikes can ignite fires.
- DROUGHT forms over the driest warm land region (3x3-smoothed moisture under
          a threshold), is stationary (bound to its region) and collapses
          with hysteresis once rain lifts local moisture back up.
- FIRE    needs an ignition source (lightning, an agent's fire/ember material
          lying in a dry fueled cell, or - rarely - self-ignition inside an
          active drought) plus fuel (plant_food) and dryness. It grows,
          creeps toward the most fuel-rich dry neighbor, and collapses when
          fuel runs out or rain arrives. The field application burning
          plant_food makes every fire self-limiting.
- BLIGHT  (fungal) forms only on dense, moist resource_cover and collapses in dry
          conditions or once it has eaten the local plant food.

Only genesis and lifecycle are new: the *application* of events to cells
(``event_field`` / ``event_fields_grid`` -> ``regrow_grid``) is untouched.
All randomness goes through the globally seeded ``random`` module
(determinism contract, seeded by ``artificial_society.rng.seed_all``);
positions chosen by argmin/argmax are deterministic.
"""

from __future__ import annotations

import random

import numpy as np

# Warm-up restored from the pre-rewrite live loop (commit 4aacaf5): no
# disturbances while the initial population establishes itself.
EVENT_WARMUP_TICKS = 600
MAX_ACTIVE_EVENTS = 6
INTENSITY_FLOOR = 0.12

# --- storm -------------------------------------------------------------------
STORM_GENESIS_P = 0.02  # per tick at storm_risk == 1.0 (0 risk -> never)
STORM_MAX = 2
STORM_TTL = 120
STORM_MOVE_EVERY = 2  # moves 1 cell per N ticks along its wind vector
STORM_DECAY = 0.99
LIGHTNING_P = 0.10  # per active storm per tick, scaled by its intensity

# --- drought -----------------------------------------------------------------
DROUGHT_CHECK_EVERY = 20
DROUGHT_START_MOISTURE = 22.0  # smoothed local moisture below this can seed one
DROUGHT_END_MOISTURE = 38.0  # hysteresis: above this the drought collapses
DROUGHT_MIN_TEMP = 15.0  # temperature is season-coupled, so this replaces
DROUGHT_TTL = 300  # the old `season_name == "summer"` string check
DROUGHT_GROWTH = 1.01
DROUGHT_COLLAPSE = 0.88
DROUGHT_SELF_IGNITE_P = 0.004  # per active drought per tick

# --- fire --------------------------------------------------------------------
FIRE_FUEL_IGNITE = 20.0  # smoothed plant_food needed to catch
FIRE_FUEL_SUSTAIN = 8.0  # below this the fire depletes
FIRE_DRY_MOISTURE = 45.0  # ignition only below this smoothed moisture
FIRE_WET_MOISTURE = 60.0  # above this (rain) the fire collapses
FIRE_TTL = 150
FIRE_GROWTH = 1.03
FIRE_COLLAPSE = 0.85
FIRE_MAX_INTENSITY = 1.3
FIRE_MAX_RADIUS = 8
FIRE_GROW_RADIUS_EVERY = 6
FIRE_CREEP_EVERY = 3
EMBER_CHECK_EVERY = 5  # cadence of the agent-fire/ember ignition scan
EMBER_IGNITE_P = 0.15  # per fire/ember cell per check

# --- blight ------------------------------------------------------------------
BLIGHT_CHECK_EVERY = 30
BLIGHT_GENESIS_P = 0.25
BLIGHT_PLANT_MIN = 60.0
BLIGHT_MOIST_MIN = 55.0
BLIGHT_PLANT_SUSTAIN = 30.0
BLIGHT_MOIST_SUSTAIN = 45.0
BLIGHT_TTL = 250
BLIGHT_MAX_RADIUS = 7

# --- direct agent effects ------------------------------------------------------
FIRE_AGENT_DAMAGE = 6.0  # HP per tick at field strength 1.0
STORM_AGENT_ENERGY = 0.8  # exposure cost per tick at field strength 1.0
STORM_AGENT_DAMAGE = 1.2  # debris damage per tick above strength 0.5
SHELTER_STORM_FACTOR = 0.4  # a camp on the cell dampens storm exposure

_WIND_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _local_mean(field: np.ndarray, x: int, y: int, r: int = 1) -> float:
    h, w = field.shape
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    return float(field[y0:y1, x0:x1].mean())


def _smoothed(world, key: str) -> np.ndarray:
    """3x3-neighborhood mean of a field, edge-correct (no border bias)."""
    sums = world._neighbor_sum(world.F[key])
    counts = world._neighbor_sum(np.ones_like(world.F[key]))
    return sums / counts


def _land_mask(world) -> np.ndarray:
    return np.array([[b != "water" for b in row] for row in world.biomes], dtype=bool)


def _overlaps(events: list, kind: str, x: int, y: int, margin: int = 3) -> bool:
    return any(
        e["kind"] == kind
        and abs(e["x"] - x) <= e["radius"] + margin
        and abs(e["y"] - y) <= e["radius"] + margin
        for e in events
    )


def _maybe_ignite(world, events: list, x: int, y: int) -> bool:
    """Start a fire at (x, y) if physics allows: land, fuel, dryness, no
    active fire already burning there, global cap not reached."""
    if len(events) >= MAX_ACTIVE_EVENTS:
        return False
    if not world.in_bounds(x, y) or world.get_biome(x, y) == "water":
        return False
    fuel = _local_mean(world.F["plant_food"], x, y)
    moist = _local_mean(world.F["moisture"], x, y)
    if fuel < FIRE_FUEL_IGNITE or moist > FIRE_DRY_MOISTURE:
        return False
    if _overlaps(events, "fire", x, y, margin=0):
        return False
    events.append(
        {
            "kind": "fire",
            "x": int(x),
            "y": int(y),
            "radius": 2,
            "intensity": min(1.0, 0.4 + fuel / 180.0 + (FIRE_DRY_MOISTURE - moist) / 100.0),
            "ttl": FIRE_TTL,
        }
    )
    return True


def _creep_to_fuel(world, event: dict) -> None:
    """Fire follows the most fuel-rich dry neighbor cell (deterministic argmax)."""
    best, bx, by = -1.0, event["x"], event["y"]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx, ny = event["x"] + dx, event["y"] + dy
            if not world.in_bounds(nx, ny) or world.get_biome(nx, ny) == "water":
                continue
            fuel = float(world.F["plant_food"][ny, nx])
            dryness = 1.0 - float(world.F["moisture"][ny, nx]) / 100.0
            score = fuel * dryness
            if score > best:
                best, bx, by = score, nx, ny
    event["x"], event["y"] = int(bx), int(by)


# ---------------------------------------------------------------------------
# lifecycle: adapt existing events against the current world state
# ---------------------------------------------------------------------------
def _adapt(world, tick: int) -> None:
    kept = []
    for e in world.active_events:
        e["ttl"] -= 1
        kind = e["kind"]
        if kind == "storm":
            e["intensity"] *= STORM_DECAY
            if tick % STORM_MOVE_EVERY == 0:
                e["x"] = int(min(max(e["x"] + e.get("vx", 0), 0), world.width - 1))
                e["y"] = int(min(max(e["y"] + e.get("vy", 0), 0), world.height - 1))
        elif kind == "drought":
            local_m = _local_mean(world.F["moisture"], e["x"], e["y"], r=2)
            if local_m < DROUGHT_END_MOISTURE:
                e["intensity"] = min(1.2, e["intensity"] * DROUGHT_GROWTH)
            else:  # rain broke the drought (hysteresis)
                e["intensity"] *= DROUGHT_COLLAPSE
        elif kind == "fire":
            fuel = _local_mean(world.F["plant_food"], e["x"], e["y"], r=2)
            moist = _local_mean(world.F["moisture"], e["x"], e["y"], r=2)
            if fuel >= FIRE_FUEL_SUSTAIN and moist <= FIRE_WET_MOISTURE:
                e["intensity"] = min(FIRE_MAX_INTENSITY, e["intensity"] * FIRE_GROWTH)
                if tick % FIRE_GROW_RADIUS_EVERY == 0:
                    e["radius"] = min(FIRE_MAX_RADIUS, e["radius"] + 1)
                if tick % FIRE_CREEP_EVERY == 0:
                    _creep_to_fuel(world, e)
            else:  # starved or rained out
                e["intensity"] *= FIRE_COLLAPSE
        elif kind == "blight":
            plant = _local_mean(world.F["plant_food"], e["x"], e["y"], r=2)
            moist = _local_mean(world.F["moisture"], e["x"], e["y"], r=2)
            if plant >= BLIGHT_PLANT_SUSTAIN and moist >= BLIGHT_MOIST_SUSTAIN:
                e["intensity"] = min(1.0, e["intensity"] * 1.01)
                if tick % 10 == 0:
                    e["radius"] = min(BLIGHT_MAX_RADIUS, e["radius"] + 1)
            else:  # dried out or ate its local plant food
                e["intensity"] *= 0.9
        if e["ttl"] > 0 and e["intensity"] > INTENSITY_FLOOR:
            kept.append(e)
    world.active_events = kept


# ---------------------------------------------------------------------------
# genesis: new events emerge from the world state
# ---------------------------------------------------------------------------
def _genesis(world, tick: int, weather_state: dict) -> None:
    events = world.active_events

    # Storm: driven by the weather system's published storm risk.
    storm_risk = float(weather_state.get("storm_risk", 0.0) or 0.0)
    storms = [e for e in events if e["kind"] == "storm"]
    if (
        len(events) < MAX_ACTIVE_EVENTS
        and len(storms) < STORM_MAX
        and storm_risk > 0.0
        and random.random() < STORM_GENESIS_P * storm_risk
    ):
        x, y = world.random_land_position()
        vx, vy = random.choice(_WIND_DIRS)
        events.append(
            {
                "kind": "storm",
                "x": int(x),
                "y": int(y),
                "radius": random.randint(5, 9),
                "intensity": 0.5 + 0.5 * storm_risk,
                "ttl": STORM_TTL,
                "vx": vx,
                "vy": vy,
            }
        )

    # Lightning: active storms can ignite dry, fueled cells under them.
    for storm in [e for e in events if e["kind"] == "storm"]:
        if random.random() < LIGHTNING_P * storm["intensity"]:
            sx = storm["x"] + random.randint(-storm["radius"], storm["radius"])
            sy = storm["y"] + random.randint(-storm["radius"], storm["radius"])
            _maybe_ignite(world, events, sx, sy)

    # Agent fire/embers: technology left lying around can start wildfires.
    if tick % EMBER_CHECK_EVERY == 0:
        for y in range(world.height):
            for x in range(world.width):
                mats = world.obj[y][x].get("materials")
                if not mats:
                    continue
                if mats.get("fire", 0.0) <= 0.1 and mats.get("ember", 0.0) <= 0.1:
                    continue
                if random.random() < EMBER_IGNITE_P:
                    _maybe_ignite(world, events, x, y)

    # Drought: seeded over the driest warm land region.
    if tick % DROUGHT_CHECK_EVERY == 0 and len(events) < MAX_ACTIVE_EVENTS:
        moist = _smoothed(world, "moisture")
        masked = np.where(_land_mask(world), moist, np.inf)
        idx = int(np.argmin(masked))
        dy, dx = divmod(idx, world.width)
        local_m = float(masked[dy, dx])
        if (
            np.isfinite(local_m)
            and local_m < DROUGHT_START_MOISTURE
            and _local_mean(world.F["temperature"], dx, dy) > DROUGHT_MIN_TEMP
            and not _overlaps(events, "drought", dx, dy)
        ):
            events.append(
                {
                    "kind": "drought",
                    "x": int(dx),
                    "y": int(dy),
                    "radius": 7,
                    "intensity": min(
                        1.0, 0.4 + (DROUGHT_START_MOISTURE - local_m) / DROUGHT_START_MOISTURE
                    ),
                    "ttl": DROUGHT_TTL,
                }
            )

    # Self-ignition inside droughts (heat + dead-dry resource_cover), rare.
    for drought in [e for e in events if e["kind"] == "drought"]:
        if random.random() < DROUGHT_SELF_IGNITE_P * drought["intensity"]:
            fx = drought["x"] + random.randint(-drought["radius"], drought["radius"])
            fy = drought["y"] + random.randint(-drought["radius"], drought["radius"])
            _maybe_ignite(world, events, fx, fy)

    # Blight: fungal — only on dense, moist resource_cover.
    if tick % BLIGHT_CHECK_EVERY == 0 and len(events) < MAX_ACTIVE_EVENTS:
        plant = _smoothed(world, "plant_food")
        moist = _smoothed(world, "moisture")
        candidates = _land_mask(world) & (plant > BLIGHT_PLANT_MIN) & (moist > BLIGHT_MOIST_MIN)
        if bool(candidates.any()) and random.random() < BLIGHT_GENESIS_P:
            score = np.where(candidates, plant, -np.inf)
            idx = int(np.argmax(score))
            by, bx = divmod(idx, world.width)
            if not _overlaps(events, "blight", bx, by):
                events.append(
                    {
                        "kind": "blight",
                        "x": int(bx),
                        "y": int(by),
                        "radius": 4,
                        "intensity": 0.5,
                        "ttl": BLIGHT_TTL,
                    }
                )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def update_events(world, tick: int, season_state: dict, weather_state: dict) -> None:
    """One event tick: adapt active events, then let new ones emerge.

    ``season_state`` is accepted for API stability but unused: seasonality
    reaches the model through the season-coupled temperature/moisture fields,
    not through name-string checks.
    """
    _adapt(world, tick)
    if tick >= EVENT_WARMUP_TICKS:
        _genesis(world, tick, weather_state or {})


def apply_event_agent_effects(world, agents) -> None:
    """Direct physical consequences for agents standing inside events.

    Fire burns (learnable flee pressure), storms cost energy (exposure) and
    cause light debris damage at high strength; a camp on the cell dampens
    storm exposure (shelter matters). Drought/blight stay indirect (food,
    fault) — matching their physical nature.
    """
    if not world.active_events:
        return
    for agent in agents:
        if not agent.alive:
            continue
        x, y = agent.pos
        f = world.event_field(x, y)
        fire = f["fire"]
        storm = f["storm"]
        if fire > 0.0:
            agent.health -= FIRE_AGENT_DAMAGE * fire
        if storm > 0.0:
            shelter = SHELTER_STORM_FACTOR if float(world.S["camp"][y, x]) > 0.0 else 1.0
            agent.energy = max(0.0, agent.energy - STORM_AGENT_ENERGY * storm * shelter)
            if storm > 0.5:
                agent.health -= STORM_AGENT_DAMAGE * (storm - 0.5) * shelter
        if agent.health <= 0.0:
            agent.health = 0.0
            agent.alive = False
