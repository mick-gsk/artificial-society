import math
import random

import numpy as np

from artificial_society.cell_store import CellGrid, CellView, build_arrays
from artificial_society.environment.biomes import BIOME_BASE, biome_color, generate_biome_grid
from artificial_society.environment.herbs import regrow_herbs
from artificial_society.environment.resources import (
    clamp,
    initial_cell_state,
    regrow_grid,
)


class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.biomes = generate_biome_grid(width, height)
        # Struct-of-arrays cell storage (perf Tier 1): numeric fields live in
        # numpy arrays (self.F / self.S / self.tick_grid), object-valued state
        # in per-cell dicts (self.obj). self.cells is a dict-compatible view.
        build_arrays(
            self,
            [[initial_cell_state(self.biomes[y][x]) for x in range(width)] for y in range(height)],
        )
        self.cells = CellGrid(self)
        self._init_biome_statics()
        self.land_positions = [
            (x, y) for y in range(height) for x in range(width) if self.biomes[y][x] != "water"
        ]
        self.active_events = []
        # Day/night state, updated each tick by simulation.step()
        self.day_state: dict = {
            "phase": "day",
            "light": 1.0,
            "danger_mult": 1.0,
            "sleep_pressure": 0.0,
            "t": 0.0,
        }

    def _init_biome_statics(self):
        """Precompute per-biome constant grids for the vectorized world update.

        Biomes never change after generation, so masks and biome-derived
        coefficients are computed once. Kept on the instance (not module state)
        so checkpointed worlds stay self-contained.
        """
        from artificial_society.environment.resources import biome_scarcity_ceiling

        h, w = self.height, self.width
        biome_arr = np.array(self.biomes, dtype=object)
        self._bio = {
            "base_temperature": np.array(
                [[BIOME_BASE[b]["temperature"] for b in row] for row in self.biomes], dtype=np.float64
            ),
            "base_danger": np.array(
                [[BIOME_BASE[b]["danger"] for b in row] for row in self.biomes], dtype=np.float64
            ),
            "is_forest": biome_arr == "forest",
            "is_swamp": biome_arr == "swamp",
            "is_desert": biome_arr == "desert",
            "is_mountain": biome_arr == "mountain",
            "is_water": biome_arr == "water",
            "plant_ceiling": np.array(
                [[biome_scarcity_ceiling(b) for b in row] for row in self.biomes], dtype=np.float64
            ),
        }
        self._bio["is_meat_biome"] = (
            self._bio["is_mountain"] | self._bio["is_forest"] | self._bio["is_swamp"]
        )
        # In-bounds neighbor count (incl. self) for the diffusion averages.
        count = np.full((h, w), 9.0)
        count[0, :] -= 3
        count[-1, :] -= 3
        count[:, 0] -= 3
        count[:, -1] -= 3
        count[0, 0] += 1
        count[0, -1] += 1
        count[-1, 0] += 1
        count[-1, -1] += 1
        self._nbr_count = count
        # Herb-capable land cells in scan order (herb spawn RNG is only drawn in
        # these biomes, so restricting the per-cell herb loop to them keeps the
        # RNG stream identical to the full-grid loop).
        from artificial_society.environment.herbs import HERB_DEFINITIONS

        herb_biomes = set()
        for defn in HERB_DEFINITIONS.values():
            herb_biomes.update(defn["biomes"])
        self._herb_cells = [
            (x, y, self.biomes[y][x])
            for y in range(h)
            for x in range(w)
            if self.biomes[y][x] in herb_biomes
        ]

    def ensure_array_storage(self):
        """Migrate a pre-array-storage (checkpointed) world in place."""
        if not hasattr(self, "F"):
            old_cells = self.cells  # list-of-lists of plain dicts
            build_arrays(self, old_cells)
            self.cells = CellGrid(self)
        if not hasattr(self, "_bio"):
            self._init_biome_statics()

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x, y):
        x = clamp(x, 0, self.width - 1)
        y = clamp(y, 0, self.height - 1)
        return CellView(self, int(x), int(y))

    # -- Authoritative cell-mutation API ------------------------------------
    # External systems should mutate cells through these instead of writing
    # `world.cells[y][x][...]` directly, so World stays the single source of
    # truth for cell state. Behaviour is identical to the in-place writes they
    # replace (no implicit clamping — callers that need bounds compute them and
    # call set_cell). World-material seeding (systems/invention.py) routes through
    # this API. Remaining direct mutators are deferred follow-ups: agent
    # foraging/consumption in agents/agent.py (migrated with the Phase 1b
    # un-patching), the territory/growth/herbs writes (environment lane), and the
    # resources.py regrowth math (the privileged intrinsic cell-update model).

    def set_cell(self, x, y, key, value):
        """Single-field cell write — replaces in-place ``cell[key] = value``."""
        from artificial_society.cell_store import FLOAT_FIELD_SET

        if key in FLOAT_FIELD_SET:  # fast path: straight into the field array
            xi = int(clamp(x, 0, self.width - 1))
            yi = int(clamp(y, 0, self.height - 1))
            self.F[key][yi, xi] = value
        else:
            self.get_cell(x, y)[key] = value

    def adjust_cell(self, x, y, **deltas):
        """Additive cell write — for each key, ``cell[key] = cell.get(key, 0.0) + delta``
        (missing keys default to 0.0). Replaces the ``cell.get(k, 0.0) + d`` idiom."""
        cell = self.get_cell(x, y)
        for key, delta in deltas.items():
            cell[key] = cell.get(key, 0.0) + delta

    def get_biome(self, x, y):
        x = clamp(x, 0, self.width - 1)
        y = clamp(y, 0, self.height - 1)
        return self.biomes[y][x]

    def biome_move_cost(self, x, y):
        biome = self.get_biome(x, y)
        return BIOME_BASE[biome]["move_cost"]

    def random_land_position(self):
        return random.choice(self.land_positions)

    def find_free_neighbor(self, pos):
        px, py = pos
        candidates = [(px + dx, py + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        random.shuffle(candidates)
        for x, y in candidates:
            if self.in_bounds(x, y) and self.get_biome(x, y) != "water":
                return x, y
        return None, None

    def neighbors(self, x, y, radius=1):
        out = []
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if self.in_bounds(xx, yy):
                    out.append((xx, yy, CellView(self, xx, yy), self.biomes[yy][xx]))
        return out

    def spawn_disturbance(self, tick, season_state, weather_state):
        season_name = season_state.get("name", "").lower()
        weights = [
            ("drought", 1.4 if season_name == "summer" else 1.0),
            ("storm", 1.3 + weather_state.get("storm_risk", 0.0)),
            ("fire", 1.2 if season_name == "summer" else 0.7),
            ("blight", 1.0),
        ]
        kinds = [k for k, _ in weights]
        probs = [w for _, w in weights]
        total = sum(probs)
        r = random.random() * total
        acc = 0.0
        kind = kinds[0]
        for k, p in zip(kinds, probs):
            acc += p
            if r <= acc:
                kind = k
                break
        x, y = self.random_land_position()
        self.active_events.append(
            {
                "kind": kind,
                "x": x,
                "y": y,
                "radius": random.randint(4, 9),
                "intensity": random.uniform(0.55, 1.1),
                "ttl": random.randint(40, 100),
            }
        )

    def update_events(self, tick, season_state, weather_state):
        if tick % 55 == 0 or (random.random() < 0.015 and len(self.active_events) < 5):
            self.spawn_disturbance(tick, season_state, weather_state)
        kept = []
        for event in self.active_events:
            event["ttl"] -= 1
            if event["kind"] == "storm":
                event["radius"] = min(max(3, event["radius"] + random.choice([-1, 0, 1])), 10)
            elif random.random() < 0.12:
                event["x"] = int(clamp(event["x"] + random.choice([-1, 0, 1]), 0, self.width - 1))
                event["y"] = int(clamp(event["y"] + random.choice([-1, 0, 1]), 0, self.height - 1))
            event["intensity"] *= 0.992
            if event["ttl"] > 0 and event["intensity"] > 0.15:
                kept.append(event)
        self.active_events = kept

    def event_field(self, x, y):
        out = {"drought": 0.0, "storm": 0.0, "fire": 0.0, "blight": 0.0, "disturbance": 0.0}
        for event in self.active_events:
            # sqrt of an exact integer square-sum instead of hypot: both this
            # and the vectorized np.sqrt are IEEE-correctly-rounded on the same
            # exact input, so scalar and grid paths agree bit-for-bit
            # (math.hypot vs np.hypot can differ by 1 ULP).
            dx = x - event["x"]
            dy = y - event["y"]
            d = math.sqrt(dx * dx + dy * dy)
            if d > event["radius"]:
                continue
            strength = max(0.0, 1.0 - d / max(1.0, event["radius"])) * event["intensity"]
            out[event["kind"]] += strength
            out["disturbance"] += strength
        return out

    def event_fields_grid(self):
        """Vectorized :meth:`event_field` for the whole grid.

        Per cell, contributions accumulate in ``active_events`` order — the same
        order the scalar path uses — and out-of-radius cells add exactly 0.0, so
        the result is bit-identical to calling ``event_field(x, y)`` per cell.
        """
        h, w = self.height, self.width
        out = {
            k: np.zeros((h, w), dtype=np.float64)
            for k in ("drought", "storm", "fire", "blight", "disturbance")
        }
        if not self.active_events:
            return out
        ys, xs = np.mgrid[0:h, 0:w]
        for event in self.active_events:
            dx = xs - event["x"]
            dy = ys - event["y"]
            d = np.sqrt((dx * dx + dy * dy).astype(np.float64))
            strength = np.where(
                d > event["radius"],
                0.0,
                np.maximum(0.0, 1.0 - d / max(1.0, event["radius"])) * event["intensity"],
            )
            out[event["kind"]] += strength
            out["disturbance"] += strength
        return out

    @staticmethod
    def _neighbor_sum(arr):
        """Sum of the 3×3 neighborhood (incl. self), out-of-bounds as 0.0.

        Adds the nine shifted copies in the same (dy, dx) scan order the scalar
        ``neighbors()`` loop used; adding 0.0 for missing edge neighbors is
        float-exact, so per-cell sums match the scalar path bit-for-bit.
        """
        padded = np.pad(arr, 1, mode="constant", constant_values=0.0)
        h, w = arr.shape
        total = np.zeros_like(arr)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                total = total + padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        return total

    def diffuse_fields(self):
        # Two-phase like the scalar version: averages are computed from the
        # pre-diffusion state of all cells, then applied.
        weights = {
            "pollution": (0.92, 0.08),
            "disease": (0.9, 0.1),
            "moisture": (0.88, 0.12),
            "ash": (0.9, 0.1),
            "disturbance": (0.9, 0.1),
        }
        n = self._nbr_count
        avgs = {k: self._neighbor_sum(self.F[k]) / n for k in weights}
        for k, (keep, mix) in weights.items():
            self.F[k] = np.clip(keep * self.F[k] + mix * avgs[k], 0.0, 100.0)

    def regional_means(self):
        # Stats/display only — float summation order differs from the old
        # scalar loop by ~1e-12 relative, which no behaviour reads.
        return {
            "food": float(self.F["food"].mean()),
            "water": float(self.F["water"].mean()),
            "pollution": float(self.F["pollution"].mean()),
            "soil_fertility": float(self.F["soil_fertility"].mean()),
            "carrying_capacity": float(self.F["carrying_capacity"].mean()),
            "disease": float(self.F["disease"].mean()),
            "disturbance": float(self.F["disturbance"].mean()),
            "events": len(self.active_events),
        }

    def hotspots(self, min_pollution=35, min_disease=30):
        mask = (
            (self.F["pollution"] >= min_pollution)
            | (self.F["disease"] >= min_disease)
            | (self.F["disturbance"] >= 30)
            | (self.S["camp"] != 0.0)
            | (self.S["farm"] != 0.0)
            | (self.S["well"] != 0.0)
        )
        return [(int(x), int(y), CellView(self, int(x), int(y))) for y, x in np.argwhere(mask)]

    def update_environment(self, season_state, weather_state, tick):
        self.update_events(tick, season_state, weather_state)
        # Grid-wide regrowth as vectorized array math (bit-identical
        # transliteration of the per-cell regrow_cell/apply_event pair).
        regrow_grid(self, season_state, weather_state, tick, self.event_fields_grid())
        # Herbs keep their per-cell loop: spawning draws RNG per herb-capable
        # cell, and the stream order (row-major over those cells) must match
        # the old full-grid loop exactly. regrow_grid consumes no RNG, so
        # hoisting it out of this loop leaves the stream unchanged.
        for x, y, biome in self._herb_cells:
            regrow_herbs(self.obj[y][x], biome)
        self.diffuse_fields()

    def color_at(self, x, y):
        cell = self.cells[y][x]
        return biome_color(self.biomes[y][x], cell)
