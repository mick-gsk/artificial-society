"""Serialize the spatial simulation state into a compact, JSON-friendly frame.

The dashboard's WebSocket layer (see :mod:`artificial_society.serve.app`) pushes one
of these per render tick so the browser can draw the live world — agent positions, a
biome/resource grid and active disturbances — rather than only aggregate numbers.

``build_frame`` is **read-only**: it never draws RNG, mutates state, or touches the
hot-file class sources beyond reading public attributes. That keeps the determinism
contract intact and lets the runner build a frame off its own thread safely (the same
thread that owns ``sim.step``).

Scalars are quantized to small ints so a 60x40 grid (~2400 cells) stays in the
single-digit-KB range per frame at ~20 Hz.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from typing import Any

import numpy as np

from artificial_society.environment.biomes import BIOME_BASE_COLOR
from artificial_society.environment.materials import IDX, PROP_DIMS, get_vector

# Biomes are fixed after world generation — cache the flattened index list per
# world instead of rebuilding a w*h Python loop for every ~20 Hz frame.
_BIOME_IDX_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Stable index order so the client can cache the legend sent in the hello frame.
_BIOME_NAMES: list[str] = sorted(BIOME_BASE_COLOR)
_BIOME_INDEX: dict[str, int] = {name: i for i, name in enumerate(_BIOME_NAMES)}

# Agent life-stage -> small int (mirrors Agent.life_stage()).
_STAGE_IDX: dict[str, int] = {"child": 0, "adult": 1, "elder": 2}

# Agent action mode -> small int (mirrors Agent.last_action_mode; 5 = sleeping,
# which overrides the mode because a sleeping agent executes no actions).
_ACT_IDX: dict[str, int] = {"idle": 0, "forage": 1, "cooperate": 2, "attack": 3, "build": 4}
ACT_SLEEP = 5

# --- ground materials (Physik v2 / Embodiment) --------------------------------
#
# Cells carry a sparse ``materials`` dict (id -> mass). For the picture world we
# reduce each id to a small visual class; the client owns one sprite per class.
# Classes double as a priority order (highest first) when a cell holds several
# materials or the frame item cap forces a cut — fire, fresh flakes and
# discovered materials must always make it onto the wire.
ITEM_FIRE = 11
ITEM_SHARD = 10  # sharpness above _SHARP_MIN — knapped flakes / blades
ITEM_WONDER = 9  # discovered (registry) material without an edge
ITEM_FLINT = 8
ITEM_BONE = 7
ITEM_MEAT = 6
ITEM_CLAY = 5
ITEM_STONE = 4
ITEM_WOOD = 3
ITEM_FIBER = 2
ITEM_BERRY = 1

_SHARP_MIN = 0.45
_ITEM_QTY_MIN = 0.15  # ignore trace amounts (growth seeds, mined-out slots)
_ITEM_CAP = 800  # per frame, priority-ranked
_ITEM_REFRESH_TICKS = 10  # ground materials change slowly; rescan every N ticks

# Raw materials blanket most of the map; drawn 1:1 they bury the landscape and
# the special finds. Common classes (flint deposits included — they cover whole
# mountainsides) only ship from richer slots and only for a deterministic ~1/3
# of cells — enough to say "here lies wood", calm enough to keep the rare
# classes (flakes, discoveries, fire) visible at a glance.
_THINNED_CLASSES = frozenset(
    (ITEM_BERRY, ITEM_FIBER, ITEM_WOOD, ITEM_STONE, ITEM_CLAY, ITEM_FLINT)
)
_COMMON_QTY_MIN = 0.5
_COMMON_KEEP_MOD = 3  # keep cells whose spatial hash % mod == 0

_NAMED_ITEM_CLASS: dict[str, int] = {
    "fire": ITEM_FIRE,
    "ember": ITEM_FIRE,
    "sharp_stone": ITEM_SHARD,
    "flint": ITEM_FLINT,
    "carcass": ITEM_BONE,
    "bone": ITEM_BONE,
    "raw_meat": ITEM_MEAT,
    "cooked_meat": ITEM_MEAT,
    "clay_moist": ITEM_CLAY,
    "clay": ITEM_CLAY,
    "stone": ITEM_STONE,
    "granite": ITEM_STONE,
    "dry_wood": ITEM_WOOD,
    "wet_wood": ITEM_WOOD,
    "wood": ITEM_WOOD,
    "plant_fiber": ITEM_FIBER,
    "fiber": ITEM_FIBER,
    "dry_grass": ITEM_FIBER,
    "berries": ITEM_BERRY,
}

# id -> visual class, resolved once per material id (registry lookups allocate).
_MAT_CLASS_CACHE: dict[str, int] = {}

# world -> (tick_bucket, items) — reuse the scan between refreshes.
_ITEMS_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _item_class(mat_id: str) -> int:
    """Visual class for a material id; 0 = not worth drawing."""
    cls = _MAT_CLASS_CACHE.get(mat_id)
    if cls is None:
        cls = _NAMED_ITEM_CLASS.get(mat_id, 0)
        if cls == 0:
            try:
                vec = get_vector(mat_id)
                sharp = float(vec[IDX["sharpness"]]) if vec is not None else 0.0
            except Exception:
                sharp = 0.0
            if sharp >= _SHARP_MIN:
                cls = ITEM_SHARD
            elif mat_id.startswith(("mat_", "pmat_")):
                cls = ITEM_WONDER
        _MAT_CLASS_CACHE[mat_id] = cls
    return cls


def _ground_items(world, tick: int) -> list[int]:
    """Flat ``[k0, x0, y0, k1, x1, y1, …]`` of the most visible material per cell.

    Priority-ranked before the cap so fire/flakes/discoveries always ship; ties
    within a class break on a spatial hash so a cap cut thins the whole map
    uniformly instead of chopping off the bottom rows.
    """
    bucket = tick // _ITEM_REFRESH_TICKS
    cached = _ITEMS_CACHE.get(world)
    if cached is not None and cached[0] == bucket:
        return cached[1]

    found: list[tuple[int, int, int, int]] = []  # (-class, hash, x, y)
    for y, row in enumerate(world.obj):
        for x, cell in enumerate(row):
            slot = cell.get("materials")
            if not slot:
                continue
            cell_hash = (x * 73856093 ^ y * 19349663) & 0xFFFF
            best = 0
            for mat_id, qty in slot.items():
                if qty < _ITEM_QTY_MIN:
                    continue
                cls = _item_class(mat_id)
                if cls <= best:
                    continue
                if cls in _THINNED_CLASSES and (
                    qty < _COMMON_QTY_MIN or cell_hash % _COMMON_KEEP_MOD
                ):
                    continue
                best = cls
            if best:
                found.append((-best, cell_hash, x, y))

    found.sort()
    items: list[int] = []
    for negcls, _, x, y in found[:_ITEM_CAP]:
        items.extend((-negcls, x, y))
    _ITEMS_CACHE[world] = (bucket, items)
    return items


def _tool_class(agent) -> int:
    """0 = empty hands, 1 = blunt tool, 2 = sharp blade."""
    tool = getattr(agent, "tool", None)
    if not tool:
        return 0
    return 2 if _item_class(tool) == ITEM_SHARD else 1


# --- behaviour fields (Frame-Schema v2) ---------------------------------------
#
# The compact per-agent behaviour fields let the viz show *why* an agent acts,
# not just where it stands. Everything here is a pure read over sim state — no
# RNG, no mutation, no side-effecting method calls — so two build_frame calls
# without an intervening sim.step() return identical dicts (determinism gate).

# Dominant-need codes for the ``nd`` field. Index into ``_NEED_NAMES``.
NEED_NONE = 0
NEED_HUNGER = 1
NEED_THIRST = 2
NEED_COLD = 3
NEED_SICK = 4
NEED_TOOL = 5
NEED_RESEARCH = 6
NEED_TIRED = 7

_NEED_NAMES: list[str] = [
    "none",
    "hunger",
    "durst",
    "kaelte",
    "krank",
    "werkzeug",
    "forschung",
    "muede",
]

# Below this magnitude every drive counts as "nothing pressing" -> code 0.
_NEED_MIN = 0.25

# 1-based goal codes come straight from PROP_DIMS order (goal labels are
# ``need_<prop>``); the client caches this legend from the hello frame.
_GOAL_CODE: dict[str, int] = {prop: i + 1 for i, prop in enumerate(PROP_DIMS)}

# ``_MAX_ENERGY`` / body-scale constants mirror the sim so drive formulas match
# their source (need_driven_invention.compute_need_vector). Imported lazily in
# ``_dominant_need`` to avoid a heavy import at module load.


def _dominant_need(agent, world) -> int:
    """Dominant survival drive for ``agent`` as a small int code (0 = none).

    Read-only mirror of the drive formulas in
    ``systems/need_driven_invention.compute_need_vector`` (hunger ~L100,
    cold ~L107, tool ~L117, sick ~L96/122, curiosity ~L132-138), extended with
    thirst and tiredness per the frame-v2 spec. We do *not* call
    ``compute_need_vector`` directly: it returns a 12-dim material-property
    vector (edibility/heat_emission/…) that conflates several drives into the
    same axis, so it can't be argmax'd back into these need categories. The raw
    drive scalars below are the clean, un-conflated source.

    Cell temperature/light come straight off the World field arrays (same style
    as ``build_frame``'s ``world.F[...]`` reads) — no ``get_cell``/CellView churn
    at ~20 Hz.
    """
    from artificial_society.agents.agent import MAX_ENERGY

    x, y = agent.pos

    energy = float(getattr(agent, "energy", MAX_ENERGY))
    hunger = max(0.0, 1.0 - energy / MAX_ENERGY)  # cf. compute_need_vector L100

    hydration = float(getattr(agent, "hydration", 100.0))
    thirst = max(0.0, 1.0 - hydration / 100.0)  # spec: durst

    temp_field = world.F.get("temperature") if hasattr(world, "F") else None
    if temp_field is not None:
        temperature = float(temp_field[y, x])
    else:  # defensive: pre-array checkpoint world
        temperature = float(world.get_cell(x, y).get("temperature", 20.0))
    cold = max(0.0, (15.0 - temperature) / 15.0)  # cf. compute_need_vector L107

    sick = getattr(agent, "disease_id", None) is not None  # cf. L96/122

    tool_need = 0.9 if getattr(agent, "tool", None) is None else 0.0  # cf. L117

    endo = getattr(agent, "endocrine", None)
    h = getattr(endo, "h", None)
    tired = float(h[2]) if h is not None and len(h) > 2 else 0.0  # melatonin

    # Research: only meaningful once nothing physical presses (mirrors the
    # curiosity block's gate in compute_need_vector L132-138).
    genes = getattr(agent, "genes", {})
    curiosity = float(genes.get("curiosity", 0.5)) if hasattr(genes, "get") else 0.5
    research = curiosity if (curiosity > 0.7 and hunger < 0.3 and cold < 0.3) else 0.0

    drives = (
        (NEED_HUNGER, hunger),
        (NEED_THIRST, thirst),
        (NEED_COLD, cold),
        (NEED_SICK, 1.0 if sick else 0.0),
        (NEED_TOOL, tool_need),
        (NEED_RESEARCH, research),
        (NEED_TIRED, tired),
    )
    # argmax; deterministic tie-break on the code order above (stable sort).
    code, val = max(drives, key=lambda d: d[1])
    return code if val >= _NEED_MIN else NEED_NONE


def _goal_code(label: str) -> int:
    """1-based goal code from a ``need_<prop>`` SubGoal label; 0 if unknown."""
    if label and label.startswith("need_"):
        return _GOAL_CODE.get(label[len("need_") :], 0)
    return 0


def _agent_flags(agent) -> int:
    """Bitfield of coarse status flags (see build_frame docstring)."""
    flags = 0
    if getattr(agent, "disease_id", None) is not None:
        flags |= 1  # bit0 sick
    if getattr(agent, "pregnant", False):
        flags |= 2  # bit1 pregnant
    msg = getattr(agent, "message_vector", None)
    if msg and max(abs(float(v)) for v in msg) > 0.15:
        flags |= 4  # bit2 communicating
    gs = getattr(agent, "goal_stack", None)
    if gs is not None and getattr(gs, "depth", lambda: 0)() > 1:
        flags |= 8  # bit3 goal stack deeper than one
    return flags


def biome_legend() -> list[dict[str, Any]]:
    """Stable ``name -> {idx, rgb}`` legend; sent once in the WS hello frame."""
    return [
        {"name": name, "idx": i, "rgb": list(BIOME_BASE_COLOR[name])}
        for i, name in enumerate(_BIOME_NAMES)
    ]


def behavior_legend() -> dict[str, list[str]]:
    """Stable legends for the frame-v2 behaviour codes; sent in the WS hello.

    ``needs``      : index i -> name for the ``nd`` field (0..7).
    ``goal_props`` : index i -> property; the ``gl`` goal code is ``i + 1``.
    ``acts``       : index i -> name for the ``act`` field (matches ``_ACT_IDX``
                     plus the sleeping override at 5).
    """
    acts = [""] * (max(_ACT_IDX.values()) + 1)
    for name, idx in _ACT_IDX.items():
        acts[idx] = name
    acts.append("sleep")  # ACT_SLEEP == len(_ACT_IDX)
    return {
        "needs": list(_NEED_NAMES),
        "goal_props": list(PROP_DIMS),
        "acts": acts,
    }


def _agent_detail(agent, world) -> dict[str, Any]:
    """Deep, on-demand read-only snapshot for one inspected agent.

    Pure: only ``getattr(..., default)`` reads and local computation. Every
    value is JSON-native (Python int/float/str/list/dict) — numpy scalars are
    cast to ``float``/``int`` so ``json.dumps`` over the whole frame never
    trips. Missing/absent sub-systems drop their key rather than crashing.
    """
    from artificial_society.agents.agent import MAX_ENERGY

    d: dict[str, Any] = {
        "id": int(agent.id),
        "age": int(getattr(agent, "age", 0)),
        "gen": int(getattr(agent, "generation", 0)),
        "kids": int(getattr(agent, "children", 0)),
        "hyd": round(float(getattr(agent, "hydration", 0.0)), 1),
        "rew": round(float(getattr(agent, "last_reward", 0.0)), 3),
        "learn": round(float(getattr(agent, "learning_score", 0.0)), 3),
    }

    sex = getattr(agent, "sex", None)
    if sex:
        d["sex"] = str(sex)
    par = getattr(agent, "parent_id", None)
    if par is not None:
        d["par"] = int(par)

    # --- raw need drives (rohe Floats; separate from the argmax code) --------
    energy = float(getattr(agent, "energy", MAX_ENERGY))
    hydration = float(getattr(agent, "hydration", 100.0))
    x, y = agent.pos
    temp_field = world.F.get("temperature") if hasattr(world, "F") else None
    temperature = (
        float(temp_field[y, x])
        if temp_field is not None
        else float(world.get_cell(x, y).get("temperature", 20.0))
    )
    endo = getattr(agent, "endocrine", None)
    h = getattr(endo, "h", None)
    d["needs"] = {
        "hunger": round(max(0.0, 1.0 - energy / MAX_ENERGY), 3),
        "durst": round(max(0.0, 1.0 - hydration / 100.0), 3),
        "kaelte": round(max(0.0, (15.0 - temperature) / 15.0), 3),
        "krank": 1.0 if getattr(agent, "disease_id", None) is not None else 0.0,
        "werkzeug": 1.0 if getattr(agent, "tool", None) is None else 0.0,
        "forschung": round(
            float(getattr(agent, "genes", {}).get("curiosity", 0.5))
            if hasattr(getattr(agent, "genes", {}), "get")
            else 0.5,
            3,
        ),
        "muede": round(float(h[2]), 3) if h is not None and len(h) > 2 else 0.0,
    }

    # --- goal stack (bottom -> top, full stack) -----------------------------
    gs = getattr(agent, "goal_stack", None)
    stack = getattr(gs, "stack", None) if gs is not None else None
    if stack:
        d["goals"] = [
            {
                "lbl": str(getattr(g, "label", "")),
                "act": str(getattr(g, "action", "")),
                "mat": getattr(g, "target_mat", None),
                "x": getattr(g, "target_x", None),
                "y": getattr(g, "target_y", None),
                "gp": int(getattr(g, "ticks_spent", 0)),
                "gm": int(getattr(g, "max_ticks", 0)),
            }
            for g in stack
        ]

    # --- hormones (8 floats) -------------------------------------------------
    if h is not None:
        d["horm"] = [round(float(v), 3) for v in h]

    # --- inventory (top 12 by qty, then id for stable ties) -----------------
    inv = getattr(agent, "material_inventory", None)
    if inv:
        top = sorted(inv.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[:12]
        d["inv"] = {str(m): round(float(q), 1) for m, q in top}

    # --- trust (top 5 by |value|, then id) ----------------------------------
    trust = getattr(agent, "trust", None)
    if trust:
        top_t = sorted(
            trust.items(), key=lambda kv: (-abs(float(kv[1])), int(kv[0]))
        )[:5]
        d["trust"] = [[int(oid), round(float(v), 2)] for oid, v in top_t]

    # --- theory of mind (agent.tom.models: dict[id -> AgentModel]) ----------
    tom = getattr(agent, "tom", None)
    models = getattr(tom, "models", None) if tom is not None else None
    if models:
        top_m = sorted(
            models.values(),
            key=lambda m: (-abs(float(getattr(m, "trust_estimate", 0.0))), int(getattr(m, "agent_id", 0))),
        )[:5]
        d["tom"] = [
            [
                int(getattr(m, "agent_id", 0)),
                str(getattr(m, "role", "unknown")),
                round(float(getattr(m, "trust_estimate", 0.0)), 2),
            ]
            for m in top_m
        ]

    # --- mood (emotional_memory: persistent float + label) ------------------
    em = getattr(agent, "emotional_memory", None)
    if em is not None:
        mood_val = getattr(em, "mood", None)
        if mood_val is not None:
            mood: dict[str, Any] = {"mood": round(float(mood_val), 3)}
            label = getattr(em, "mood_label", None)
            if label:
                mood["label"] = str(label)
            d["mood"] = mood

    # --- message vector (raw comms output) ----------------------------------
    msg = getattr(agent, "message_vector", None)
    if msg:
        d["msg"] = [round(float(v), 3) for v in msg]

    return d


def _hex(rgb: Iterable[float]) -> str:
    """``(r, g, b)`` 0..255 -> ``"#rrggbb"`` (clamped, rounded)."""
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_frame(sim, inspect_ids=frozenset()) -> dict[str, Any]:
    """Read-only snapshot of the spatial state for one render tick.

    ``inspect_ids`` (a set of agent ids): for each *living* inspected agent a
    deep ``detail`` blob is attached; when empty the ``detail`` key is omitted.

    Layout (all JSON-native):
        ``tick``      : int
        ``grid``      : {w, h}
        ``daylight``  : float 0..1 (world.day_state light level)
        ``cells``     : {food[w*h], water[w*h], biome[w*h], moist[w*h],
                        ash[w*h]?} — row-major, quantized ints. ``moist`` is
                        ``world.F["moisture"]`` (0..100), sent every frame (it
                        shifts with weather). ``ash`` is ``world.F["ash"]``
                        (0..100), sent only every 5th tick AND only while any
                        cell still holds visible ash (``max() > 0.5``); the key
                        is omitted otherwise so steady-state frames stay
                        byte-identical — the client caches the last array and
                        keeps painting it until a fresh one arrives. Both keys
                        are omitted entirely on worlds that predate these
                        fields (defensive ``getattr``/``in`` checks).
        ``structures``: [{x, y, k}] sparse; k ∈ {camp, farm, well}
        ``items``     : flat [k, x, y, …] sparse ground materials; k = ITEM_* class
        ``agents``    : [{id, x, y, e, hp, st, act, tl, cg, tribe, col}] plus, per
                        agent, the compact behaviour fields below (falsy values
                        are omitted, so the key is simply absent):
            ``nd`` dominant need code 0..7 (see ``behavior_legend()['needs']``)
            ``gl`` top-goal code 1..12 (index+1 into ``goal_props``)
            ``gp``, ``gm`` top-goal ticks_spent / max_ticks
            ``gx``, ``gy`` top-goal target x / y
            ``tg`` last_action_target (partner/victim id; None -> omitted)
            ``fl`` flag bitfield: bit0 sick, bit1 pregnant, bit2 communicating
                   (max|message_vector| > 0.15), bit3 goal-stack depth > 1
            ``pa`` parent_id (None -> omitted)
        ``events``    : [{kind, x, y, r, i}]
        ``stats``     : the aggregate dict the cards already use (``sim.stats.last``)
        ``detail``    : {str(id): _agent_detail(...)} — only for inspected agents;
                        omitted when ``inspect_ids`` is empty
    """
    world = sim.world
    w, h = world.width, world.height
    biomes = world.biomes

    # Straight off the struct-of-arrays field storage (perf Tier 1): two
    # vectorized ops instead of 2*w*h per-cell view reads at ~20 Hz.
    food = np.rint(world.F["food"]).astype(int).ravel().tolist()
    water = np.rint(world.F["water"]).astype(int).ravel().tolist()

    # Moisture/ash (Physik v2 ground fields): defensive against older worlds
    # that predate these keys — a missing field just drops from the frame
    # rather than raising, mirroring the getattr-defensive style used above.
    moisture_arr = world.F.get("moisture") if hasattr(world, "F") else None
    moist = (
        np.rint(moisture_arr).astype(int).ravel().tolist()
        if moisture_arr is not None
        else None
    )

    ash_arr = world.F.get("ash") if hasattr(world, "F") else None
    ash = None
    if ash_arr is not None and sim.tick % 5 == 0 and float(ash_arr.max()) > 0.5:
        ash = np.rint(ash_arr).astype(int).ravel().tolist()

    biome_idx = _BIOME_IDX_CACHE.get(world)
    if biome_idx is None:
        biome_idx = [
            _BIOME_INDEX.get(biomes[y][x], 0) for y in range(h) for x in range(w)
        ]
        _BIOME_IDX_CACHE[world] = biome_idx

    # Structures are rare — ship them sparse straight off the S arrays.
    structures = []
    for kind in ("camp", "farm", "well"):
        arr = world.S.get(kind)
        if arr is not None:
            for sy, sx in np.argwhere(arr != 0.0):
                structures.append({"x": int(sx), "y": int(sy), "k": kind})

    agents = []
    for a in sim.agents:
        d = {
            "id": a.id,
            "x": a.x,
            "y": a.y,
            "e": round(a.energy),
            "hp": round(a.health),
            "st": _STAGE_IDX.get(a.life_stage(), 1),
            "act": ACT_SLEEP
            if getattr(a, "is_sleeping", False)
            else _ACT_IDX.get(getattr(a, "last_action_mode", "idle"), 0),
            "tl": _tool_class(a),
            "cg": min(9, int(sum(getattr(a, "material_inventory", {}).values()))),
            "tribe": a.tribe_id,
            "col": _hex(a.display_color()),
        }

        # --- compact behaviour fields (v2); falsy -> omit -------------------
        nd = _dominant_need(a, world)
        if nd:
            d["nd"] = nd

        gs = getattr(a, "goal_stack", None)
        goal = gs.peek() if gs is not None else None
        if goal is not None:
            gl = _goal_code(getattr(goal, "label", ""))
            if gl:
                d["gl"] = gl
            gp = int(getattr(goal, "ticks_spent", 0))
            if gp:
                d["gp"] = gp
            gm = int(getattr(goal, "max_ticks", 0))
            if gm:
                d["gm"] = gm
            gx = getattr(goal, "target_x", None)
            if gx is not None:
                d["gx"] = int(gx)
            gy = getattr(goal, "target_y", None)
            if gy is not None:
                d["gy"] = int(gy)

        tg = getattr(a, "last_action_target", None)
        if tg is not None:
            d["tg"] = int(tg)

        fl = _agent_flags(a)
        if fl:
            d["fl"] = fl

        pa = getattr(a, "parent_id", None)
        if pa is not None:
            d["pa"] = int(pa)

        agents.append(d)

    events = [
        {
            "kind": e.get("kind", "?"),
            "x": e.get("x", 0),
            "y": e.get("y", 0),
            "r": e.get("radius", 0),
            "i": round(float(e.get("intensity", 0.0)), 2),
        }
        for e in getattr(world, "active_events", [])
    ]

    frame = {
        "type": "frame",
        "tick": sim.tick,
        "grid": {"w": w, "h": h},
        "daylight": round(float(getattr(world, "day_state", {}).get("light", 1.0)), 3),
        "cells": {"food": food, "water": water, "biome": biome_idx},
        "structures": structures,
        "items": _ground_items(world, sim.tick),
        "agents": agents,
        "events": events,
        "stats": dict(getattr(sim.stats, "last", {})),
    }
    if moist is not None:
        frame["cells"]["moist"] = moist
    if ash is not None:
        frame["cells"]["ash"] = ash

    if inspect_ids:
        detail = {
            str(a.id): _agent_detail(a, world)
            for a in sim.agents
            if a.id in inspect_ids
        }
        if detail:
            frame["detail"] = detail

    return frame
