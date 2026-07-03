// PixiJS (WebGL) renderer for the live simulation world.
//
// Design language: a living landscape under an observation instrument. Terrain
// is a single subpixel-painted texture (3x3 texels per cell) with dithered
// biome transitions, shimmering water, shore foam and a warm dawn/dusk tint.
// Life on top is pictorial: outlined pixel figures with a two-frame walk,
// shadows, tools in hands, carry bundles, emote bubbles; materials, decor and
// real building sprites populate the ground. Weather is visible weather —
// rain falls inside storm radii, fires glow and flicker.
//
// The whole world lives in `worldRoot`, which the user can zoom (wheel) and
// pan (drag); a click picks the nearest agent for the inspector panel.
//
// Layers, back to front:
//   terrain → decor → grid → items → structures → shadows → trails → fx →
//   glow → figures → emotes → events
//
// Frames arrive ~20 Hz; the ticker runs at display rate and eases agent motion
// between frames so movement stays fluid.

import { Application, Container, Graphics, Sprite, Text, Texture } from "pixi.js";

import {
  makeDecorTextures,
  makeEmoteTextures,
  makeItemTextures,
  makeStructureTextures,
  makeToolTextures,
} from "./sprites.js";

import { ACT_KEY, buildFigureAtlas } from "./figures.js";

import {
  buildTerrainMesh,
  makeAshTexture,
  makeDataTexture,
  makeLutTexture,
} from "./terrain-shader.js";

import { createParticleSystem } from "./particles.js";

const STAGE_SCALE = [0.62, 1.0, 1.06]; // child, adult, elder (figure size)
const STAGE_KEY = ["child", "adult", "elder"]; // st index → atlas stage key
// Agents never shrink below this many on-screen px tall — figures stay legible
// as points on huge grids / when zoomed out. Applied AFTER zoom in _tick.
const MIN_AGENT_PX = 14;
const ACT_EMOTE = { 1: "forage", 2: "cooperate", 3: "attack", 4: "build", 5: "sleep" };
const TRAIL_LEN = 8;
const ZOOM_MIN = 1;
// ZOOM_MAX is dynamic (R4) — recomputed in _rebuildGrid from the current
// cellPx so a huge grid (small cellPx) can still zoom in to real screen-size
// cells. ZOOM_MAX_FLOOR is the absolute minimum ceiling (matches the old
// static constant) so small grids don't lose max-zoom headroom.
const ZOOM_MAX_FLOOR = 8;
// Zoom-LOD (R4): three detail tiers driven by on-screen cell size
// s = cellPx * zoom, with ±10% hysteresis at both boundaries so a slow zoom
// across a threshold doesn't flicker. Base thresholds per the spec:
//   FAR  s < 8      MID  8 <= s < 24      NEAR s >= 24
const LOD_FAR_MID = 8;
const LOD_MID_NEAR = 24;
const LOD_HYST = 0.1;

const EVENT_COLORS = {
  drought: 0xf0b030,
  storm: 0x4cc6ff,
  fire: 0xff7a3c,
  blight: 0xc774f0,
};

// Event kind → shader code (packed as kind*10 + intensity in uEvents[].w).
// Must match the branch order in terrain-shader.js.
const EVENT_KIND_CODE = { drought: 1, fire: 2, blight: 3, storm: 4 };

// Action codes from serve/frame.py: 0 idle/move, 1 forage, 2 cooperate,
// 3 attack, 4 build, 5 sleep.
export const ACT = { IDLE: 0, FORAGE: 1, COOPERATE: 2, ATTACK: 3, BUILD: 4, SLEEP: 5 };

// Action-line marker kinds + their palette. ttl (seconds) is the fade-out
// window; a re-trigger for the same actor→target pair is throttled until it
// lapses (see _spawnActionMarkers). Attack/coop links draw at every LOD (the
// most important read); pulse/build/comms are MID/NEAR only (anti-clutter).
const MARKER_ATTACK = 0; // red line actor→victim
const MARKER_COOP = 1; // cyan arc actor↔partner
const MARKER_FORAGE = 2; // green expanding pulse on the actor cell
const MARKER_BUILD = 3; // amber ring on the actor cell
const MARKER_COMMS = 4; // expanding comms ring around the agent
const MARKER_TTL = 0.5; // seconds
const MARKER_COLORS = {
  [MARKER_ATTACK]: 0xff5d6c,
  [MARKER_COOP]: 0x3fc5f0,
  [MARKER_FORAGE]: 0x49d17c,
  [MARKER_BUILD]: 0xffb54d,
  [MARKER_COMMS]: 0x9fd0ff,
};
export const ACT_COLORS = {
  [ACT.FORAGE]: 0x49d17c,
  [ACT.COOPERATE]: 0x3fc5f0,
  [ACT.ATTACK]: 0xff5d6c,
  [ACT.BUILD]: 0xffb54d,
  [ACT.SLEEP]: 0x9b8cff,
};

// Curated landscape palette by biome name (fallback: the legend rgb, quieted).
// Two tones per biome: parched (low food) → lush (high food).
const BIOME_PALETTE = {
  water: { dry: 0x14324f, lush: 0x1d4e79 },
  ocean: { dry: 0x102a44, lush: 0x1a4468 },
  forest: { dry: 0x24402c, lush: 0x2f6b40 },
  grassland: { dry: 0x4a5a33, lush: 0x6d8f44 },
  swamp: { dry: 0x2e4034, lush: 0x3d5c46 },
  mountain: { dry: 0x4e5560, lush: 0x5d6874 },
  desert: { dry: 0x8a7148, lush: 0xa08a58 },
  tundra: { dry: 0x5d6b72, lush: 0x74868d },
};

const COLORS = {
  gridLine: 0x0e1620,
  gridTick: 0x3a536f,
  eventDefault: 0xf0b030,
};

export class WorldScene {
  constructor() {
    this.app = new Application();
    this.biomeTones = {}; // biome idx -> {dry:{r,g,b}, lush:{r,g,b}, isWater}
    this.grid = null; // {w, h}
    this.cellPx = 12;
    this.offX = 0;
    this.offY = 0;
    this.daylight = 1.0;
    this._jitter = null; // per-cell brightness noise so terrain isn't plastic-flat

    // everything world-positioned goes under worldRoot: zoom/pan = one transform
    this.worldRoot = new Container();
    this.terrainLayer = new Container();
    this.decorLayer = new Container();
    this.gridLayer = new Container();
    this.itemLayer = new Container();
    this.structLayer = new Container();
    this.shadowLayer = new Graphics();
    this.trailLayer = new Graphics();
    this.fxLayer = new Graphics();
    // Action lines/rings (attack, cooperate, forage pulse, build, comms) live in
    // their own Graphics layer between fx and glow so they read under the figure
    // glow but over the fx/trail scribbles. Redrawn each tick from a pool of
    // marker records with a ttl — no per-frame allocation in the steady state.
    this.actionLayer = new Graphics();
    // Pooled particle system (R5): rain, fire, smoke, dust, spores, birth/death,
    // discovery sparks. Its ParticleContainer sits between the action lines and
    // the figure glow so weather reads under the figures' luminous halos but over
    // the fx/action scribbles. Fire's additive light lives in glowLayer, not here.
    // Created in init() once the Application (GL context) exists.
    this.particles = null;
    this.glowLayer = new Container();
    this.figureLayer = new Container();
    this.emoteLayer = new Container();
    this.eventLayer = new Container();

    this.agents = new Map(); // id -> {glow, figure, emote, tool, bundle, fromX..toY, t, trail, act, actAge, tint}
    this._structKey = "";
    this._decorKey = "";
    this._itemKey = "";
    this._specialSet = null; // "k@x,y" of shard/wonder/fire cells — burst on new ones
    this._fireSprites = [];
    this._bursts = []; // {cx, cy, age, color} — knapping sparks, discoveries, fire
    this._biomeArr = null;
    this._events = [];

    // -- particle-system bookkeeping (R5) -------------------------------------
    // Fire sources this frame in WORLD px ([x,y,...]) — filled by _paintItems
    // (FIRE items) and update() (fire events), consumed by _tick to emit flames,
    // smoke and to place the additive night-glow sprites in glowLayer.
    this._fireSources = [];
    // Chimney sources (camp structures) in WORLD px — smoke only.
    this._chimneys = [];
    // Reused additive glow sprites for fire light sources (grow the pool as
    // needed, hide the surplus). alpha ∝ (1 - daylight) → fires light the night.
    this._fireGlows = [];
    // Global client wind vector {x,y} in world px/s, rotates slowly off time.
    this._wind = { x: 0, y: 0 };
    // Birth/death lifecycle: which agent ids we saw last frame (for genuinely-new
    // detection), a first-frame guard so a fresh connect doesn't burst every
    // agent as "born", and the death-fade list (records kept ~600 ms after their
    // id vanishes: figure fades out, a wisp rises, THEN destroy).
    this._agentConnected = false; // set true after the first _syncAgents
    this._dying = []; // {rec, age} — mid-death-fade records (out of this.agents)
    this._deathGlowTex = null; // radial texture for fire glows (built in init)

    // Debug-only cell grid. Off by default (the shader terrain reads as an
    // organic landscape, not a game board). Flip at runtime via
    // window.__scene.showGrid = true to get a faint alignment grid.
    this.showGrid = false;

    // Environment events are drawn INTO the terrain shader (drought/fire/blight/
    // storm as ground scars). Up to 8 slots packed strongest-first each frame,
    // one vec4 per slot: (cx, cy, radius, kind*10 + intensity). Persistent buffer
    // — no per-frame allocation. Mirrors the mesh's uEvents uniform value.
    this._eventPack = new Float32Array(8 * 4);
    this._eventCount = 0;

    // terrain texture state (subpixel canvas) — the Canvas2D fallback painter
    this._terCanvas = null;
    this._terCtx = null;
    this._terImg = null;
    this._terTex = null;
    this._terSprite = null;
    this._baseRGB = null; // per-cell day-lit base color, for neighbour blending
    this._isWater = null;
    this._frameNo = 0;
    this._phase = 0; // water shimmer phase

    // GPU-shader terrain (default path; falls back to the canvas painter on a
    // shader-compile error). Data texture = one RGBA8 texel per cell; a 16×2
    // palette LUT drives per-biome colour. See terrain-shader.js.
    this.useShaderTerrain = true;
    this._terMesh = null;
    this._terUniforms = null;
    this._terShader = null;
    this._terData = null; // Uint8Array w*h*4, packed each frame (R/G/B/A = biome/food/water/moisture)
    this._terDataTex = null;
    this._lutData = new Uint8Array(16 * 2 * 4); // dry row + lush row
    this._lutTex = null;
    this._terTime = 0; // seconds fed to uTime
    this._terFilled = false; // has the data buffer been filled since rebuild?

    // Ash overlay (R6): a separate w*h RGBA8 buffer/texture (R channel only
    // read by the shader), refreshed ONLY on frames that ship a fresh
    // `cells.ash` array (server gates it to every 5th tick + only while ash
    // is actually present). On every other frame the buffer/texture are left
    // untouched, so the last-known burn scar keeps painting — the client-side
    // cache the brief calls for.
    this._ashData = null; // Uint8Array w*h*4, refilled only when cells.ash arrives
    this._ashTex = null;

    // view state
    this._zoom = 1;
    this._zoomMax = ZOOM_MAX_FLOOR; // recomputed per grid in _rebuildGrid
    this._drag = null;
    this.onPick = null; // (agentId | null) => void
    this.onHover = null; // (agentId | null) => void — nearest agent under cursor
    this.selectedId = null;

    // Follow camera (B5). When set to a living agent id the ticker eases
    // worldRoot so that agent stays centred; a manual drag clears it. panToCell /
    // panToAgent implement the inspector's "jump" without engaging follow.
    // onFollowChange(id|null) lets the owner keep its follow button in sync when
    // the scene clears follow itself (manual drag / followed agent gone).
    this.followId = null;
    this.onFollowChange = null;

    // Action-line markers: a pool of {kind, from, to, cx, cy, ttl, life, color}.
    // `from`/`to` are agent-record refs (not fixed coords) so a line follows the
    // eased render positions of both endpoints across its whole ttl. Re-triggers
    // for the same actor→target pair are throttled until the previous marker
    // expires (keyed in _actionKeys) to avoid a redraw storm.
    this._markers = [];
    this._markerPool = [];
    this._actionKeys = new Map(); // "kind:actorId:targetId" -> remaining ttl

    // Floating "why" label over the selected agent — one reused Pixi Text set
    // from World.svelte via setSelectedLabel(). Created lazily in init().
    this._selLabel = null;
    this._selLabelText = "";

    // Zoom-LOD (R4): 'far' | 'mid' | 'near', driven by on-screen cell size with
    // hysteresis. Recomputed on zoom change and grid rebuild; a change triggers
    // one pass over agent records to set visibility flags (see _applyLod).
    this._lod = "mid";

    this._glowTex = null;
    this._pulse = 0;
    this._fps = 60;
    this._hudAccum = 0;
    this.onHud = null;
  }

  async init(host) {
    await this.app.init({ background: 0x05070b, antialias: true, resizeTo: host });
    host.appendChild(this.app.canvas);
    this.glowLayer.blendMode = "add";
    this.worldRoot.addChild(
      this.terrainLayer,
      this.decorLayer,
      this.gridLayer,
      this.itemLayer,
      this.structLayer,
      this.shadowLayer,
      this.trailLayer,
      this.fxLayer,
      this.actionLayer,
      this.glowLayer,
      this.figureLayer,
      this.emoteLayer,
      this.eventLayer,
    );
    this.app.stage.addChild(this.worldRoot);
    this._glowTex = makeRadialTexture(64, [
      [0, "rgba(255,255,255,1)"],
      [0.4, "rgba(255,255,255,0.45)"],
      [1, "rgba(255,255,255,0)"],
    ]);
    // 20×28 animated figure atlas: 17 poses × 3 stages in one texture source,
    // sliced into per-frame Textures so every agent batches in one draw call.
    // frames[stage][actKey] = [Texture,...]; hand[...] = matching hand anchors
    // (figure-local px); anim[actKey] = {durMs, n}. See figures.js.
    this._atlas = buildFigureAtlas();
    this._emoteTex = makeEmoteTextures();
    this._decorTex = makeDecorTextures();
    this._itemTex = makeItemTextures();
    this._toolTex = makeToolTextures();
    this._structTex = makeStructureTextures();

    // Pooled particle system (R5). Its ParticleContainer is inserted into
    // worldRoot right before the glow layer (over action lines, under figure
    // glow). One shared atlas over TWO adjacent containers — normal blend for
    // rain/smoke/dust/wisps, additive for flames/spores/sparks (two draw calls);
    // one fixed pool with an 800 cap behind both.
    this.particles = createParticleSystem({ container: this.worldRoot });
    this.worldRoot.setChildIndex(this.particles.container, this.worldRoot.getChildIndex(this.glowLayer));
    this.worldRoot.setChildIndex(this.particles.containerAdd, this.worldRoot.getChildIndex(this.glowLayer));
    // Radial texture for the additive fire glows (a soft warm falloff).
    this._deathGlowTex = makeRadialTexture(64, [
      [0, "rgba(255,196,120,1)"],
      [0.5, "rgba(255,150,70,0.5)"],
      [1, "rgba(255,120,50,0)"],
    ]);

    // Floating "why" label over the selected agent. Lives on the emote layer so
    // it renders above figures; positioned in the ticker, text set from World via
    // setSelectedLabel(). One reused object — never re-created per frame.
    this._selLabel = new Text({
      text: "",
      style: {
        fill: 0xf2f6ff,
        fontFamily: "ui-monospace, Menlo, monospace",
        fontSize: 13,
        stroke: { color: 0x05070b, width: 4 },
        align: "center",
      },
    });
    this._selLabel.anchor.set(0.5, 1);
    this._selLabel.visible = false;
    this._selLabel.resolution = 2;
    this.emoteLayer.addChild(this._selLabel);

    this._bindViewControls();
    this._bindHover();
    this.app.ticker.add((t) => this._tick(t.deltaMS));
    this.app.renderer.on("resize", () => this._rebuildGrid());
  }

  // A WS hello implies the stream may now describe a different run (reconnect
  // or restart). Arm the next _syncAgents to treat its population as a reset:
  // no birth sparkles, vanished records leave without death theatre.
  notifyReset() {
    this._agentConnected = false;
  }

  // Text of the floating label over the selected agent. World.svelte owns the
  // warum.js string and pushes it here; world-render.js never touches warum.js.
  setSelectedLabel(str) {
    this._selLabelText = str ?? "";
    if (this._selLabel && this._selLabel.text !== this._selLabelText) {
      this._selLabel.text = this._selLabelText;
    }
  }

  // -- zoom / pan / pick -------------------------------------------------------

  _bindViewControls() {
    const canvas = this.app.canvas;
    canvas.style.cursor = "grab";
    canvas.style.touchAction = "none";
    // offsetX/offsetY is 0 on synthetic events and flaky across browsers —
    // derive canvas-local coordinates from clientX/Y instead.
    const local = (e) => {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const z = clampNum(this._zoom * Math.exp(-e.deltaY * 0.0018), ZOOM_MIN, this._zoomMax);
        const p = local(e);
        this._applyZoom(z, p.x, p.y);
      },
      { passive: false },
    );
    canvas.addEventListener("pointerdown", (e) => {
      this._drag = { x: e.clientX, y: e.clientY, moved: 0 };
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        /* synthetic events have no active pointer */
      }
      canvas.style.cursor = "grabbing";
    });
    // A manual pan breaks follow-mode — once the user grabs the world we stop
    // chasing the agent. Fires on the first move past the click threshold so a
    // plain click (which selects) doesn't cancel a follow the user just started.
    // Notify the owner so its follow button can drop out of the "on" state.
    canvas.addEventListener("pointermove", () => {
      if (this._drag && this.followId != null && this._drag.moved >= 5) {
        this.followId = null;
        this.onFollowChange?.(null);
      }
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this._drag) return;
      const dx = e.clientX - this._drag.x;
      const dy = e.clientY - this._drag.y;
      this._drag.x = e.clientX;
      this._drag.y = e.clientY;
      this._drag.moved += Math.abs(dx) + Math.abs(dy);
      this.worldRoot.x += dx;
      this.worldRoot.y += dy;
      this._clampPan();
    });
    const endDrag = (e) => {
      if (this._drag && this._drag.moved < 5) {
        const p = local(e);
        this._pick(p.x, p.y);
      }
      this._drag = null;
      canvas.style.cursor = "grab";
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", () => (this._drag = null));
    canvas.addEventListener("dblclick", () => {
      this._zoom = 1;
      this.worldRoot.scale.set(1);
      this.worldRoot.position.set(0, 0);
      this._updateLod();
    });
  }

  _applyZoom(z, cx, cy) {
    const k = z / this._zoom;
    this.worldRoot.x = cx - (cx - this.worldRoot.x) * k;
    this.worldRoot.y = cy - (cy - this.worldRoot.y) * k;
    this._zoom = z;
    this.worldRoot.scale.set(z);
    this._clampPan();
    this._updateLod();
  }

  _clampPan() {
    const W = this.app.renderer.width;
    const H = this.app.renderer.height;
    const z = this._zoom;
    this.worldRoot.x = clampNum(this.worldRoot.x, W - W * z, 0);
    this.worldRoot.y = clampNum(this.worldRoot.y, H - H * z, 0);
  }

  // Nearest agent to a canvas-space point whose hit-disc (radius from rec.hpx,
  // the zoom-aware figure size honouring the MIN_AGENT_PX floor) the point falls
  // inside; null if none. Shared by click-pick and hover so both stay in sync.
  _nearestAgent(sx, sy) {
    const wx = (sx - this.worldRoot.x) / this._zoom;
    const wy = (sy - this.worldRoot.y) / this._zoom;
    let best = null;
    let bestD = Infinity;
    for (const [id, rec] of this.agents) {
      if (rec._px == null) continue; // not yet placed by the ticker
      const dx = rec._px - wx;
      const dy = rec._py - wy;
      const d = dx * dx + dy * dy;
      const r = (rec.hpx ?? this.cellPx) * 0.7;
      if (d < r * r && d < bestD) {
        bestD = d;
        best = id;
      }
    }
    return best;
  }

  _pick(sx, sy) {
    const best = this._nearestAgent(sx, sy);
    this.selectedId = best;
    this.onPick?.(best);
  }

  // -- hover (B5) --------------------------------------------------------------
  //
  // pointermove on the canvas, throttled to ~10/s: resolve the nearest agent and
  // fire onHover only when it changes. Cheap (one linear pass over agents) and
  // idle-friendly (skipped when nobody listens or a drag is in progress).
  _bindHover() {
    const canvas = this.app.canvas;
    let last = 0;
    let lastId = undefined;
    const emit = (id) => {
      if (id !== lastId) {
        lastId = id;
        this.onHover?.(id);
      }
    };
    canvas.addEventListener("pointermove", (e) => {
      if (!this.onHover || this._drag) return;
      const now = performance.now();
      if (now - last < 100) return; // ~10 Hz
      last = now;
      const r = canvas.getBoundingClientRect();
      emit(this._nearestAgent(e.clientX - r.left, e.clientY - r.top));
    });
    canvas.addEventListener("pointerleave", () => emit(null));
  }

  // -- camera jumps (B5) — used by the inspector's clickable targets -----------
  //
  // Centre a world cell (or an agent) in the viewport without engaging follow.
  // Respects the same pan clamp as manual drag so the world can't slide off.
  panToCell(cx, cy) {
    if (!this.grid) return;
    const W = this.app.renderer.width;
    const H = this.app.renderer.height;
    const wx = this.offX + (cx + 0.5) * this.cellPx;
    const wy = this.offY + (cy + 0.5) * this.cellPx;
    this.worldRoot.x = W / 2 - wx * this._zoom;
    this.worldRoot.y = H / 2 - wy * this._zoom;
    this._clampPan();
  }

  panToAgent(id) {
    const rec = this.agents.get(id);
    if (rec) this.panToCell(rec.toX, rec.toY);
  }

  // -- zoom-LOD (R4) -----------------------------------------------------------
  //
  // s = on-screen cell size. Three tiers with ±10% hysteresis at both
  // boundaries: once in a tier, s has to cross the boundary by 10% before the
  // tier changes, so a slow zoom back and forth across a threshold doesn't
  // flicker. Called on zoom change and grid rebuild (cellPx change) — NOT
  // every tick.
  _lodForScale(s, current) {
    const midUp = LOD_FAR_MID * (1 + LOD_HYST); // 8.8
    const midDown = LOD_FAR_MID * (1 - LOD_HYST); // 7.2
    const nearUp = LOD_MID_NEAR * (1 + LOD_HYST); // 26.4
    const nearDown = LOD_MID_NEAR * (1 - LOD_HYST); // 21.6
    if (current === "far") {
      if (s >= midUp) return s >= nearUp ? "near" : "mid";
      return "far";
    }
    if (current === "near") {
      if (s < nearDown) return s < midDown ? "far" : "mid";
      return "near";
    }
    // current === "mid"
    if (s < midDown) return "far";
    if (s >= nearUp) return "near";
    return "mid";
  }

  _updateLod() {
    const s = this.cellPx * this._zoom;
    const next = this._lodForScale(s, this._lod);
    if (next !== this._lod) {
      this._lod = next;
      this._applyLod();
    }
  }

  // One pass over agent records setting visible flags per the new LOD tier —
  // not per-tick branching. FAR hides tool/bundle/emote/per-agent-shadow (and
  // switches to the pawn silhouette + no breathing, handled in _tick's texture
  // lookup); trails + coop-links are gated by a single top-level check in
  // _tick (they're per-frame Graphics redraws, not persistent sprites, so
  // there is nothing to toggle here). Glow stays visible at every tier.
  _applyLod() {
    const detail = this._lod !== "far";
    for (const rec of this.agents.values()) {
      rec.tool.visible = detail && !!rec.toolOn;
      rec.bundle.visible = detail && !!rec.bundleOn;
      rec.emote.visible = detail && !!rec.emoteOn;
    }
  }

  setLegend(biomes) {
    this.biomeTones = {};
    this._biomeNameByIdx = {};
    for (const b of biomes) {
      const pal = BIOME_PALETTE[b.name];
      const dry = pal ? splitRGB(pal.dry) : quietFallback(b.rgb, 0.85);
      const lush = pal ? splitRGB(pal.lush) : quietFallback(b.rgb, 1.15);
      this.biomeTones[b.idx] = { dry, lush, isWater: b.name === "water" || b.name === "ocean" };
      this._biomeNameByIdx[b.idx] = b.name;
    }
    this._decorKey = ""; // legend can arrive after the first frames
    this._packLut(); // refresh the shader palette LUT from the same tones
  }

  // Pack the biome tones into the 16×2 LUT: row 0 = dry colour, row 1 = lush.
  // The dry row's alpha carries the isWater flag (1 = water biome) so the shader
  // knows to interpolate dry→lush by water level instead of food. Indices with
  // no legend entry stay a quiet slate so an unknown biome still reads as ground.
  _packLut() {
    const d = this._lutData;
    for (let i = 0; i < 16; i++) {
      const tone = this.biomeTones[i];
      const dry = tone ? tone.dry : { r: 20, g: 26, b: 34 };
      const lush = tone ? tone.lush : { r: 26, g: 32, b: 40 };
      const isW = tone && tone.isWater ? 255 : 0;
      const c0 = i * 4; // dry row (row 0)
      const c1 = (16 + i) * 4; // lush row (row 1)
      d[c0] = dry.r;
      d[c0 + 1] = dry.g;
      d[c0 + 2] = dry.b;
      d[c0 + 3] = isW;
      d[c1] = lush.r;
      d[c1 + 1] = lush.g;
      d[c1 + 2] = lush.b;
      d[c1 + 3] = 255;
    }
    if (this._lutTex) this._lutTex.source.update();
  }

  update(frame) {
    const g = frame.grid;
    const resized = !this.grid || this.grid.w !== g.w || this.grid.h !== g.h;
    this.grid = g;
    this.daylight = frame.daylight ?? 1.0;
    this._biomeArr = frame.cells.biome;
    this._events = frame.events ?? [];
    if (resized) this._rebuildGrid();
    this._frameNo++;
    this._syncTerrainMode();
    if (this.useShaderTerrain) this._fillTerrainData(frame.cells);
    else this._paintTerrain(frame.cells);
    this._buildDecor();
    this._paintItems(frame.items ?? []);
    this._paintStructures(frame.structures ?? []);
    this._syncAgents(frame.agents);
    this._packEvents(this._events); // event → shader uniforms (20 Hz, no realloc)
    this._paintEvents(this._events);
  }

  // -- terrain: one subpixel-painted texture ------------------------------------

  _ensureTerrain() {
    const { w, h } = this.grid;
    if (this._terCanvas && this._terCanvas.width === w * 3) return;
    this._terCanvas = document.createElement("canvas");
    this._terCanvas.width = w * 3;
    this._terCanvas.height = h * 3;
    this._terCtx = this._terCanvas.getContext("2d");
    this._terImg = this._terCtx.createImageData(w * 3, h * 3);
    if (this._terTex) this._terTex.destroy(true);
    this._terTex = Texture.from(this._terCanvas);
    this._terTex.source.scaleMode = "nearest";
    if (!this._terSprite) {
      this._terSprite = new Sprite(this._terTex);
      this.terrainLayer.addChild(this._terSprite);
    } else {
      this._terSprite.texture = this._terTex;
    }
    this._baseRGB = new Uint8ClampedArray(w * h * 3);
    this._isWater = new Uint8Array(w * h);
  }

  _paintTerrain(cells) {
    if (!this.grid) return;
    const { w, h } = this.grid;
    // big worlds repaint every 3rd frame — terrain moves slowly anyway
    if (w * h > 20000 && this._frameNo % 3) return;
    this._ensureTerrain();

    const { food, water, biome } = cells;
    const dl = this.daylight;
    const day = 0.45 + 0.55 * dl;
    // dawn/dusk: a warm band while the light passes ~0.35
    const dusk = Math.max(0, 1 - Math.abs(dl - 0.35) / 0.25);
    const warmR = 1 + dusk * 0.22;
    const warmB = 1 - dusk * 0.18;
    const nightBlue = (1 - dl) * 10;

    // pass 1: per-cell day-lit base color
    const base = this._baseRGB;
    const isW = this._isWater;
    for (let i = 0; i < biome.length; i++) {
      const tone = this.biomeTones[biome[i]];
      let r, g, b;
      if (!tone) {
        r = 20;
        g = 26;
        b = 34;
        isW[i] = 0;
      } else if (tone.isWater) {
        const wn = Math.min(1, water[i] / 100);
        r = lerp(tone.dry.r, tone.lush.r, wn);
        g = lerp(tone.dry.g, tone.lush.g, wn);
        b = lerp(tone.dry.b, tone.lush.b, wn);
        isW[i] = 1;
      } else {
        const fn = Math.min(1, food[i] / 90);
        r = lerp(tone.dry.r, tone.lush.r, fn);
        g = lerp(tone.dry.g, tone.lush.g, fn);
        b = lerp(tone.dry.b, tone.lush.b, fn);
        isW[i] = 0;
      }
      const j = this._jitter ? this._jitter[i] : 1;
      base[i * 3] = r * j * day * warmR;
      base[i * 3 + 1] = g * j * day;
      base[i * 3 + 2] = b * j * day * warmB + nightBlue;
    }

    // pass 2: 3x3 subpixels — dithered biome edges, water shimmer + shore foam
    this._phase = (this._phase + 1) & 1023;
    const ph = this._phase;
    const d = this._terImg.data;
    const W3 = w * 3;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const bi = biome[i];
        const r0 = base[i * 3];
        const g0 = base[i * 3 + 1];
        const b0 = base[i * 3 + 2];
        const wtr = isW[i];
        for (let sy = 0; sy < 3; sy++) {
          for (let sx = 0; sx < 3; sx++) {
            let r = r0;
            let g = g0;
            let b = b0;
            // edge dithering toward a differing neighbour (checker pattern)
            let n = -1;
            if (sx === 0 && x > 0 && biome[i - 1] !== bi) n = i - 1;
            else if (sx === 2 && x < w - 1 && biome[i + 1] !== bi) n = i + 1;
            else if (sy === 0 && y > 0 && biome[i - w] !== bi) n = i - w;
            else if (sy === 2 && y < h - 1 && biome[i + w] !== bi) n = i + w;
            if (n >= 0 && (x * 3 + sx + y * 3 + sy) & 1) {
              if (wtr && !this._isWater[n]) {
                // shoreline: land meets water as pale foam, not land-colored dither
                r = r0 + 55;
                g = g0 + 60;
                b = b0 + 65;
              } else {
                r = (r + base[n * 3]) >> 1;
                g = (g + base[n * 3 + 1]) >> 1;
                b = (b + base[n * 3 + 2]) >> 1;
              }
            } else if (wtr) {
              // moving sparse highlights make water read as liquid
              const t = x * 3 + sx + (y * 3 + sy) * 2 + (ph >> 2);
              if (t % 11 === 0) {
                r += 14;
                g += 20;
                b += 26;
              }
            }
            const p = ((y * 3 + sy) * W3 + x * 3 + sx) * 4;
            d[p] = r;
            d[p + 1] = g;
            d[p + 2] = b;
            d[p + 3] = 255;
          }
        }
      }
    }
    this._terCtx.putImageData(this._terImg, 0, 0);
    this._terTex.source.update();
  }

  // -- terrain: GPU data-texture + fragment shader (default path) --------------

  // Reconcile which terrain path is visible with the `useShaderTerrain` flag.
  // Lets the World.svelte debug hook flip the mode at runtime: the next frame's
  // paint fills the now-visible path and the other one is hidden (not destroyed,
  // so toggling back is instant).
  _syncTerrainMode() {
    if (this.useShaderTerrain) {
      if (this._terSprite) this._terSprite.visible = false;
      if (this._terMesh) this._terMesh.visible = true;
    } else {
      if (this._terSprite) this._terSprite.visible = true;
      if (this._terMesh) this._terMesh.visible = false;
    }
  }

  // Build the shader mesh, data texture and palette LUT for the current grid.
  // On any shader-compile failure this flips `useShaderTerrain` off, tears down
  // partial state, and returns false so the caller keeps the canvas painter.
  _ensureShaderTerrain() {
    if (!this.grid) return false;
    const { w, h } = this.grid;
    if (this._terMesh && this._terData && this._terData.length === w * h * 4) return true;

    // fresh grid size: (re)build persistent buffers + textures + mesh
    this._destroyShaderTerrain();
    try {
      this._terData = new Uint8Array(w * h * 4);
      this._terFilled = false; // fresh buffer — force a fill on the next frame
      this._terDataTex = makeDataTexture(w, h, this._terData);
      // Ash starts at zero (no scars on a fresh world/rebuild) — the buffer is
      // only touched again once a real `cells.ash` array arrives.
      this._ashData = new Uint8Array(w * h * 4);
      this._ashTex = makeAshTexture(w, h, this._ashData);
      this._lutTex = makeLutTexture(this._lutData);
      this._packLut(); // ensure LUT reflects any legend already received
      const built = buildTerrainMesh({
        w,
        h,
        cellPx: this.cellPx,
        offX: this.offX,
        offY: this.offY,
        dataTex: this._terDataTex,
        ashTex: this._ashTex,
        lutTex: this._lutTex,
      });
      this._terMesh = built.mesh;
      this._terUniforms = built.uniforms;
      this._terShader = built.shader;
      this.terrainLayer.addChild(this._terMesh);
      this._syncTerrainMode(); // shader owns the terrain: hide the canvas sprite
      return true;
    } catch (err) {
      console.warn("[terrain] shader unavailable, falling back to canvas:", err);
      this.useShaderTerrain = false;
      this._destroyShaderTerrain();
      return false;
    }
  }

  _destroyShaderTerrain() {
    if (this._terMesh) {
      this._terMesh.destroy();
      this._terMesh = null;
    }
    if (this._terDataTex) {
      this._terDataTex.destroy(true);
      this._terDataTex = null;
    }
    if (this._ashTex) {
      this._ashTex.destroy(true);
      this._ashTex = null;
    }
    if (this._lutTex) {
      this._lutTex.destroy(true);
      this._lutTex = null;
    }
    this._terUniforms = null;
    this._terShader = null;
    this._terData = null;
    this._ashData = null;
  }

  // Pack one RGBA8 texel per cell into the persistent data buffer, then upload.
  // R = biome index, G = food (÷90→255), B = water (÷100→255), A = moisture
  // (÷100→255). Cheap: 160 KB at 200×200. The big-world every-3rd-frame
  // throttle from the canvas painter carries over — the GPU animates
  // water/light from uTime each frame regardless of buffer refresh.
  _fillTerrainData(cells) {
    if (!this._ensureShaderTerrain()) {
      this._paintTerrain(cells); // fallback flipped on during ensure
      return;
    }
    const { w, h } = this.grid;
    // big worlds refresh the data buffer every 3rd frame (as the canvas painter
    // did) — but always fill once right after a rebuild so no blank frame shows.
    if (this._terFilled && w * h > 20000 && this._frameNo % 3) return;
    this._terFilled = true;

    const { food, water, biome, moist } = cells;
    const d = this._terData;
    const n = w * h;
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      d[p] = biome[i] & 255;
      const f = food[i] / 90;
      d[p + 1] = f > 1 ? 255 : (f * 255) | 0;
      const wt = water[i] / 100;
      d[p + 2] = wt > 1 ? 255 : (wt * 255) | 0;
      // moisture ships every frame (server side), but stay defensive against
      // an old/replayed frame that predates the field — fall back to a
      // neutral mid-value rather than leaving the alpha channel stale.
      const m = moist ? moist[i] / 100 : 0.5;
      d[p + 3] = m > 1 ? 255 : m < 0 ? 0 : (m * 255) | 0;
    }
    this._terDataTex.source.update();

    this._fillAshData(cells);
  }

  // Refresh the ash overlay ONLY on frames that ship a fresh `cells.ash`
  // array (server gates this to every 5th tick, and only while ash is
  // actually present anywhere — see frame.py). On every other frame this is
  // a no-op: the existing texture keeps painting the last-known scar, which
  // is exactly the "client caches the last array" behaviour the brief wants.
  // R channel only (G/B/A unused; a full RGBA buffer is the simplest way to
  // reuse the same BufferImageSource path as the main data texture).
  _fillAshData(cells) {
    const ash = cells.ash;
    if (!ash || !this._ashData || !this._ashTex) return;
    const d = this._ashData;
    const n = this.grid.w * this.grid.h;
    for (let i = 0; i < n; i++) {
      const a = ash[i] / 100;
      d[i * 4] = a > 1 ? 255 : a < 0 ? 0 : (a * 255) | 0;
    }
    this._ashTex.source.update();
  }

  // Pack the (up to 8) strongest environment events into the shader's uEvents
  // uniform so the terrain shader can paint them as ground scars. Runs at frame
  // rate (~20 Hz) from update(), not per display tick — no per-frame allocation:
  // the sort works on a small array and the packed floats go into the persistent
  // `_eventPack` buffer, which is the mesh uniform's backing store.
  //
  // Slot layout (vec4): x,y = event centre in cells; z = radius in cells;
  // w = kind*10 + intensity (kind 1=drought 2=fire 3=blight 4=storm).
  _packEvents(events) {
    const pack = this._eventPack;
    if (!this.useShaderTerrain || !this._terUniforms || !events || !events.length) {
      this._eventCount = 0;
      if (this._terUniforms) this._terUniforms.uniforms.uEventCount = 0;
      return;
    }
    // strongest first by intensity × radius; take the top 8.
    const top =
      events.length > 8
        ? [...events].sort((a, b) => b.i * b.r - a.i * a.r).slice(0, 8)
        : events;
    let n = 0;
    for (let k = 0; k < top.length && n < 8; k++) {
      const e = top[k];
      const kind = EVENT_KIND_CODE[e.kind];
      if (!kind) continue; // unknown kind → the shader has no branch for it
      const p = n * 4;
      pack[p] = e.x;
      pack[p + 1] = e.y;
      pack[p + 2] = Math.max(1, e.r);
      pack[p + 3] = kind * 10 + Math.min(0.999, Math.max(0, e.i ?? 0));
      n++;
    }
    this._eventCount = n;
    // uEvents value is the same Float32Array (mutated in place); reassigning is
    // cheap and keeps Pixi's dirty tracking honest.
    this._terUniforms.uniforms.uEvents = pack;
    this._terUniforms.uniforms.uEventCount = n;
  }

  // -- terrain decorations (trees, rocks, tufts) — static per world ------------

  _buildDecor() {
    if (!this.grid || !this._biomeArr || !Object.keys(this.biomeTones).length) return;
    const { w, h } = this.grid;
    const key = `${w}x${h}@${this.cellPx}`;
    if (key === this._decorKey) return;
    this._decorKey = key;
    this.decorLayer.removeChildren().forEach((c) => c.destroy());

    const nameByIdx = this._biomeNameByIdx ?? {};
    const cp = this.cellPx;
    // Keep sprite counts sane on big grids.
    const densityScale = Math.min(1, 9000 / (w * h));
    const rnd = (i, salt) => {
      const n = Math.sin(i * 127.1 + salt * 269.5) * 43758.5453;
      return n - Math.floor(n);
    };
    const DECOR = {
      forest: [
        ["tree", 0.42],
        ["pine", 0.22],
        ["tuft", 0.1],
      ],
      grassland: [["tuft", 0.3]],
      mountain: [["rock", 0.26]],
      desert: [
        ["cactus", 0.07],
        ["rock", 0.05],
      ],
      swamp: [
        ["reed", 0.3],
        ["tree", 0.08],
      ],
      tundra: [["rock", 0.1]],
    };
    for (let i = 0; i < this._biomeArr.length; i++) {
      const name = nameByIdx[this._biomeArr[i]];
      const spec = DECOR[name];
      if (!spec) continue;
      const r = rnd(i, 1);
      let acc = 0;
      for (const [kind, p] of spec) {
        acc += p * densityScale;
        if (r < acc) {
          const t = this._decorTex[kind];
          const s = new Sprite(t);
          s.anchor.set(0.5, 1);
          const jx = (rnd(i, 2) - 0.5) * 0.6;
          const jy = rnd(i, 3) * 0.35;
          s.x = this.offX + ((i % w) + 0.5 + jx) * cp;
          s.y = this.offY + (Math.floor(i / w) + 0.92 + jy * 0.2) * cp;
          const hpx = cp * (kind === "tuft" || kind === "reed" ? 0.75 : 1.35);
          s.scale.set(hpx / t.height);
          this.decorLayer.addChild(s);
          break;
        }
      }
    }
    // paint order: lower rows in front
    this.decorLayer.children.sort((a, b) => a.y - b.y);
  }

  // -- ground materials (Physik v2) --------------------------------------------

  // items is the flat [k, x, y, …] list from frame.py; the server refreshes it
  // every few ticks, so most frames repaint nothing.
  _paintItems(items) {
    const key = items.join() + "@" + this.cellPx;
    if (key === this._itemKey) return;
    this._itemKey = key;
    this.itemLayer.removeChildren().forEach((c) => c.destroy());
    this._fireSprites = [];

    const cp = this.cellPx;
    const rnd = (i, salt) => {
      const n = Math.sin(i * 127.1 + salt * 269.5) * 43758.5453;
      return n - Math.floor(n);
    };
    const special = new Set();
    const BURST_COLOR = { 10: 0xffd166, 9: 0xc084fc, 11: 0xff8a3c };
    for (let i = 0; i < items.length; i += 3) {
      const k = items[i];
      const x = items[i + 1];
      const y = items[i + 2];
      const t = this._itemTex[k];
      if (!t) continue;
      const s = new Sprite(t);
      s.anchor.set(0.5, 1);
      const ci = y * (this.grid?.w ?? 1) + x;
      s.x = this.offX + (x + 0.5 + (rnd(ci, 4) - 0.5) * 0.4) * cp;
      s.y = this.offY + (y + 0.88) * cp;
      const hpx = cp * (k === 11 ? 0.95 : k >= 9 ? 0.75 : k >= 6 ? 0.68 : 0.58);
      s.scale.set(hpx / t.height);
      this.itemLayer.addChild(s);
      if (k === 11) {
        s._bs = s.scale.x;
        this._fireSprites.push(s);
      }
      if (k >= 9) special.add(`${k}@${x},${y}`);
    }
    this.itemLayer.children.sort((a, b) => a.y - b.y);

    // a flake, discovery or fire appearing is THE moment worth marking
    if (this._specialSet) {
      for (const sk of special) {
        if (!this._specialSet.has(sk)) {
          const [k, pos] = sk.split("@");
          const [x, y] = pos.split(",").map(Number);
          const color = BURST_COLOR[k] ?? 0xffd166;
          this._bursts.push({ cx: x, cy: y, age: 0, color });
          // Discovery (k=9) also throws glut sparks; MID/NEAR only. Emit in world
          // px — the burst star itself stays in _tick/fxLayer as before.
          if (k === "9" && this.particles && this._lod !== "far") {
            const bx = this.offX + (x + 0.5) * this.cellPx;
            const by = this.offY + (y + 0.5) * this.cellPx;
            this.particles.emitters.sparks(bx, by, this.cellPx, color);
          }
        }
      }
    }
    this._specialSet = special;
  }

  // -- terrain / grid --------------------------------------------------------

  _rebuildGrid() {
    if (!this.grid) return;
    const { w, h } = this.grid;
    const W = this.app.renderer.width;
    const H = this.app.renderer.height;
    this.cellPx = Math.max(2, Math.floor(Math.min(W / w, H / h)));
    this.offX = Math.floor((W - this.cellPx * w) / 2);
    this.offY = Math.floor((H - this.cellPx * h) / 2);

    // dynamic ZOOM_MAX (R4): a big grid has a small cellPx, so a fixed zoom
    // ceiling would never let it reach a readable on-screen cell size. Scale
    // the ceiling up so cellPx * zoomMax reaches ~64 screen-px, never below
    // the old static floor.
    this._zoomMax = Math.max(ZOOM_MAX_FLOOR, 64 / this.cellPx);
    if (this._zoom > this._zoomMax) {
      this._zoom = this._zoomMax;
      this.worldRoot.scale.set(this._zoom);
      this._clampPan();
    }

    // deterministic per-cell brightness jitter (hash of index) — breaks up
    // flat color fields into something that reads as ground
    this._jitter = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const n = Math.sin(i * 127.1 + 311.7) * 43758.5453;
      this._jitter[i] = 0.94 + (n - Math.floor(n)) * 0.12; // 0.94 .. 1.06
    }

    this._ensureTerrain();
    this._terSprite.x = this.offX;
    this._terSprite.y = this.offY;
    this._terSprite.width = w * this.cellPx;
    this._terSprite.height = h * this.cellPx;

    // shader terrain: rebuild mesh + data texture at the new grid/scale. Both a
    // grid-size change and a pure resize (offX/offY/cellPx shift) need a fresh
    // quad, so tear down and rebuild rather than resize in place.
    if (this.useShaderTerrain) {
      this._destroyShaderTerrain();
      this._ensureShaderTerrain();
    }

    this._structKey = ""; // force structure re-layout at the new scale
    this._itemKey = "";
    this._drawGrid();
    this._updateLod();
  }

  // Debug-only alignment grid. Off by default — the default landscape shows no
  // cell lines or frame. window.__scene.showGrid = true draws a faint grid; the
  // change takes effect on the next _rebuildGrid (resize) or immediately if you
  // also call __scene._drawGrid().
  _drawGrid() {
    this.gridLayer.removeChildren().forEach((c) => c.destroy());
    if (!this.showGrid) return;

    const { w, h } = this.grid;
    const cp = this.cellPx;
    const x0 = this.offX;
    const y0 = this.offY;
    const x1 = x0 + w * cp;
    const y1 = y0 + h * cp;
    const step = w > 40 ? 10 : 5;

    const g = new Graphics();
    for (let x = 0; x <= w; x += step) {
      const px = x0 + x * cp;
      g.moveTo(px, y0).lineTo(px, y1);
    }
    for (let y = 0; y <= h; y += step) {
      const py = y0 + y * cp;
      g.moveTo(x0, py).lineTo(x1, py);
    }
    g.stroke({ width: 1, color: COLORS.gridLine, alpha: 0.08 });
    g.rect(x0, y0, w * cp, h * cp).stroke({
      width: 1,
      color: COLORS.gridTick,
      alpha: 0.08,
    });
    this.gridLayer.addChild(g);
  }

  // -- structures --------------------------------------------------------------

  _paintStructures(structures) {
    // few structures; redraw only when the set (or scale) changes
    const key = structures.map((s) => `${s.k}${s.x},${s.y}`).join("|") + `@${this.cellPx}`;
    if (key === this._structKey) return;
    this._structKey = key;
    this.structLayer.removeChildren().forEach((c) => c.destroy());
    const cp = this.cellPx;
    // Camp chimneys emit smoke — recompute the world-px source list on any change.
    this._chimneys = [];
    for (const s of structures) {
      const t = this._structTex[s.k];
      if (!t) continue;
      const spr = new Sprite(t);
      spr.anchor.set(0.5, 1);
      spr.x = this.offX + (s.x + 0.5) * cp;
      spr.y = this.offY + (s.y + 0.98) * cp;
      spr.scale.set((cp * 1.15) / t.height);
      this.structLayer.addChild(spr);
      // a camp tent's smoke rises from its peak (a little above the foot).
      if (s.k === "camp") this._chimneys.push(spr.x, spr.y - cp * 0.9);
    }
    this.structLayer.children.sort((a, b) => a.y - b.y);
  }

  // -- agents ----------------------------------------------------------------

  _syncAgents(agents) {
    // Population-reset detection (R5 fix): a run restart mid-connection (POST
    // /api/run → new sim) floods this sync with brand-new ids — that is a cast
    // change, not a wave of births. Reset when (a) this is the first sync after
    // a (re)connect (`notifyReset()` clears the flag on every WS hello), or (b)
    // more than 3 ids are new AND at least half the incoming population is new
    // in ONE sync (genuine births arrive one or two per 20-Hz frame). During a
    // reset: no birth sparkles, vanished records leave without death theatre,
    // and stale particles from the old run are cleared.
    let newIds = 0;
    for (const a of agents) if (!this.agents.has(a.id)) newIds++;
    const reset = !this._agentConnected || (newIds > 3 && newIds * 2 >= agents.length);
    if (reset && (this.agents.size || this._dying.length)) {
      this.particles?.clear(); // old run's weather/fx are stale coordinates now
      for (const d of this._dying) {
        d.rec.glow.destroy();
        d.rec.figure.destroy({ children: true });
        d.rec.emote.destroy();
      }
      this._dying.length = 0;
    }

    const seen = new Set();
    for (const a of agents) {
      seen.add(a.id);
      let rec = this.agents.get(a.id);
      if (!rec) {
        const glow = new Sprite(this._glowTex);
        const figure = new Sprite(this._atlas.frames.adult.idle[0]);
        const emote = new Sprite(this._emoteTex.forage);
        glow.anchor.set(0.5);
        figure.anchor.set(0.5, this._atlas.anchorY); // feet on the cell
        emote.anchor.set(0.5, 1);
        emote.visible = false;
        // tool in the hand + carry bundle at the hip, in figure-local pixels
        // (children inherit scale, breathing/bob and the facing flip). The tool
        // is re-anchored to the active frame's hand anchor every tick.
        const tool = new Sprite(this._toolTex.blunt);
        tool.anchor.set(0.5, 1);
        tool.visible = false;
        const bundle = new Sprite(this._toolTex.bundle);
        bundle.anchor.set(0.5, 0.5);
        bundle.position.set(-4.5, -6.5);
        bundle.visible = false;
        figure.addChild(bundle);
        figure.addChild(tool);
        this.glowLayer.addChild(glow);
        this.figureLayer.addChild(figure);
        this.emoteLayer.addChild(emote);
        rec = {
          id: a.id,
          glow,
          figure,
          emote,
          tool,
          bundle,
          tl: a.tl ?? 0,
          st: a.st ?? 1,
          fromX: a.x,
          fromY: a.y,
          toX: a.x,
          toY: a.y,
          t: 1,
          trail: [],
          act: 0,
          actAge: 999,
          facing: 1,
          // data-driven "wants to be shown" flags — combined with the current
          // LOD tier (see _applyLod) to get the sprite's actual .visible. A
          // brand-new agent inherits whatever LOD is currently active.
          toolOn: false,
          bundleOn: false,
          emoteOn: false,
        };
        this.agents.set(a.id, rec);
        // Birth burst: a genuinely-new id — gold sparks + ring. NOT during a
        // population reset (first sync after connect, or a run restart flooding
        // the frame with new ids). MID/NEAR only (anti-clutter at far zoom).
        if (!reset && this.particles && this._lod !== "far") {
          const bx = this.offX + (a.x + 0.5) * this.cellPx;
          const by = this.offY + (a.y + 0.5) * this.cellPx;
          this.particles.emitters.birth(bx, by, this.cellPx);
        }
      } else {
        rec.fromX = rec.toX;
        rec.fromY = rec.toY;
        rec.toX = a.x;
        rec.toY = a.y;
        rec.t = 0;
        rec.trail.push([a.x, a.y]);
        if (rec.trail.length > TRAIL_LEN) rec.trail.shift();
        if (a.x > rec.fromX) rec.facing = 1;
        else if (a.x < rec.fromX) rec.facing = -1;
        // the knapping moment: hands were empty or blunt, now hold a sharp blade
        if ((a.tl ?? 0) === 2 && rec.tl !== 2) {
          this._bursts.push({ cx: a.x, cy: a.y, age: 0, color: 0xffd166 });
        }
      }
      rec.tl = a.tl ?? 0;
      rec.st = a.st ?? 1;
      if ((a.act ?? 0) !== rec.act) {
        rec.act = a.act ?? 0;
        rec.actAge = 0; // restart the action animation
      }
      // behaviour fields for action markers + intent line (falsy → absent).
      rec.nd = a.nd ?? 0;
      rec.tg = a.tg ?? null;
      rec.gl = a.gl ?? 0;
      rec.gp = a.gp ?? 0;
      rec.gm = a.gm ?? 0;
      rec.gx = a.gx ?? null;
      rec.gy = a.gy ?? null;
      rec.fl = a.fl ?? 0;

      // Spawn action-line markers on this fresh frame (once per frame, not per
      // tick). Endpoints are record refs so the drawn line follows both agents'
      // eased motion; comms/forage/build markers sit on the actor's own cell.
      this._spawnActionMarkers(rec, a);
      const tint = blip(a.col);
      rec.tint = tint;
      rec.energy = a.e;
      rec.cg = a.cg ?? 0;
      const sleeping = rec.act === ACT.SLEEP;

      // frame-data-driven state only. All SIZES (figure/glow/emote/shadow/ring)
      // depend on the zoom-aware rec.hpx and are set in _tick.
      rec.figure.tint = tint;
      rec.figure.alpha = sleeping ? 0.75 : 1.0;
      rec.glow.tint = tint;
      rec.glow.alpha = sleeping ? 0.08 : 0.16 + 0.12 * (1 - this.daylight);

      // what the body carries: blade/stone in hand, bundle at the hip. Store
      // the data-driven desire in *On, then combine with the LOD tier (FAR
      // hides all three regardless of data) for the sprite's real .visible —
      // this keeps _applyLod's LOD-only pass and this data-only pass from
      // fighting each other.
      const detail = this._lod !== "far";
      rec.toolOn = !sleeping && rec.tl > 0;
      rec.tool.visible = detail && rec.toolOn;
      if (rec.tl > 0) rec.tool.texture = rec.tl === 2 ? this._toolTex.sharp : this._toolTex.blunt;
      rec.bundleOn = !sleeping && rec.cg >= 2;
      rec.bundle.visible = detail && rec.bundleOn;

      // emote bubble above the head while acting (texture + visibility here;
      // its size/offset track rec.hpx in _tick)
      const emoteName = ACT_EMOTE[rec.act];
      if (emoteName) {
        rec.emote.texture = this._emoteTex[emoteName];
        rec.emoteOn = true;
      } else {
        rec.emoteOn = false;
      }
      rec.emote.visible = detail && rec.emoteOn;
    }
    for (const [id, rec] of this.agents) {
      if (!seen.has(id)) {
        this.agents.delete(id);
        if (reset) {
          // Wholesale cast replacement (run restart) — the old records leave
          // instantly and silently: no fade, no wisp; nobody "died", the run
          // changed underneath us.
          rec.glow.destroy();
          rec.figure.destroy({ children: true });
          rec.emote.destroy();
        } else {
          // Death: don't destroy immediately. Move the record OUT of this.agents
          // (so selection/hover/markers stop tracking it — World already
          // deselects when an id vanishes) into _dying, where _tick fades the
          // figure over ~600 ms and then destroys it. A grey wisp rises.
          this._dying.push({ rec, age: 0 });
          rec.emote.visible = false;
          if (this.particles && this._lod !== "far") {
            const dx = rec._px ?? this.offX + (rec.toX + 0.5) * this.cellPx;
            const dy = rec._py ?? this.offY + (rec.toY + 0.5) * this.cellPx;
            this.particles.emitters.deathWisp(dx, dy, this.cellPx);
          }
        }
        if (this.selectedId === id) {
          this.selectedId = null;
          this.onPick?.(null);
        }
        if (this.followId === id) {
          this.followId = null;
          this.onFollowChange?.(null);
        }
      }
    }
    this._agentConnected = true; // subsequent new ids are real births
    // lower agents render in front (simple painter's order)
    this.figureLayer.children.sort((a, b) => a.y - b.y);
  }

  // -- action markers (B5) -----------------------------------------------------
  //
  // Spawn the frame's action lines/rings for one agent record. Called once per
  // FRAME (from _syncAgents), not per tick — the ticker only ages + redraws the
  // pooled markers. Endpoint refs (not coords) mean a line follows both agents'
  // eased motion; the throttle map keeps a held action (attack over many ticks)
  // from stacking a new marker every frame — it re-arms only after ttl lapses.
  _spawnActionMarkers(rec, a) {
    const act = rec.act;
    const near = this._lod !== "far"; // pulse/build/comms are MID/NEAR only

    // Attack line — actor → living victim. Drawn at ALL LODs.
    if (act === ACT.ATTACK && rec.tg != null) {
      const target = this.agents.get(rec.tg);
      if (target) this._arm(MARKER_ATTACK, rec, target);
    }

    // Cooperation arc — actor ↔ living partner. Drawn at ALL LODs. A missing tg
    // (legitimate pooling path) simply skips — no line, no error.
    if (act === ACT.COOPERATE && rec.tg != null) {
      const target = this.agents.get(rec.tg);
      if (target) this._arm(MARKER_COOP, rec, target);
    }

    // Forage pulse / build ring — on the actor cell, MID/NEAR only.
    if (near && act === ACT.FORAGE) this._arm(MARKER_FORAGE, rec, null);
    if (near && act === ACT.BUILD) this._arm(MARKER_BUILD, rec, null);

    // Communication ring — fl bit2. MID/NEAR only.
    if (near && rec.fl & 4) this._arm(MARKER_COMMS, rec, null);
  }

  // Add (or refuse, if still throttled) one marker. Endpoints are record refs;
  // pooled objects are reused so the steady state allocates nothing.
  _arm(kind, from, to) {
    const key = `${kind}:${from.id ?? "?"}:${to ? to.id ?? "?" : "-"}`;
    if (this._actionKeys.has(key)) return; // previous still alive → throttle
    const m = this._markerPool.pop() ?? {};
    m.kind = kind;
    m.from = from;
    m.to = to;
    m.ttl = MARKER_TTL;
    m.life = MARKER_TTL;
    m.color = MARKER_COLORS[kind];
    m.key = key;
    this._markers.push(m);
    this._actionKeys.set(key, MARKER_TTL);
  }

  // Age + draw every live marker into actionLayer (already cleared this tick by
  // _tick). Expired markers return to the pool. Positions read the endpoints'
  // current eased render coords (_px/_py) so lines track motion. `from.id` may
  // have been removed (agent died) — guard and expire cleanly.
  _drawMarkers(dt, lw) {
    if (!this._markers.length) {
      // still age the throttle keys so a re-trigger can re-arm even with no live
      // markers (belt-and-braces; normally the marker's own expiry clears them).
      if (this._actionKeys.size) {
        for (const [k, t] of this._actionKeys) {
          const nt = t - dt;
          if (nt <= 0) this._actionKeys.delete(k);
          else this._actionKeys.set(k, nt);
        }
      }
      return;
    }
    const g = this.actionLayer;
    const keep = [];
    for (const m of this._markers) {
      m.ttl -= dt;
      const from = m.from;
      const aliveFrom = from && from._px != null && this.agents.get(from.id) === from;
      const aliveTo = !m.to || (m.to._px != null && this.agents.get(m.to.id) === m.to);
      if (m.ttl <= 0 || !aliveFrom || !aliveTo) {
        this._actionKeys.delete(m.key);
        this._markerPool.push(m);
        continue;
      }
      keep.push(m);
      const p = 1 - m.ttl / m.life; // 0 → 1 over the ttl
      const alpha = Math.max(0, 1 - p); // linear fade
      const ax = from._px;
      const ay = from._py;

      if (m.kind === MARKER_ATTACK) {
        const bx = m.to._px;
        const by = m.to._py;
        g.moveTo(ax, ay).lineTo(bx, by).stroke({ width: 2.2 * lw, color: m.color, alpha: alpha * 0.9 });
        // arrowhead toward the victim
        const ang = Math.atan2(by - ay, bx - ax);
        const hl = this.cellPx * 0.5;
        g.moveTo(bx, by)
          .lineTo(bx - Math.cos(ang - 0.4) * hl, by - Math.sin(ang - 0.4) * hl)
          .moveTo(bx, by)
          .lineTo(bx - Math.cos(ang + 0.4) * hl, by - Math.sin(ang + 0.4) * hl)
          .stroke({ width: 2 * lw, color: m.color, alpha: alpha * 0.9 });
      } else if (m.kind === MARKER_COOP) {
        const bx = m.to._px;
        const by = m.to._py;
        // quadratic bow above the midpoint so a coop link reads distinct from a
        // straight attack line
        const mx = (ax + bx) / 2;
        const my = (ay + by) / 2 - Math.hypot(bx - ax, by - ay) * 0.18;
        g.moveTo(ax, ay)
          .quadraticCurveTo(mx, my, bx, by)
          .stroke({ width: 1.8 * lw, color: m.color, alpha: alpha * 0.8 });
      } else if (m.kind === MARKER_FORAGE) {
        const r = this.cellPx * (0.3 + p * 0.9);
        g.circle(ax, ay, r).stroke({ width: 1.6 * lw, color: m.color, alpha: alpha * 0.85 });
      } else if (m.kind === MARKER_BUILD) {
        const r = this.cellPx * 0.7;
        g.circle(ax, ay, r).stroke({ width: 2 * lw, color: m.color, alpha: alpha * 0.85 });
        g.circle(ax, ay, r * 0.55).stroke({ width: lw, color: m.color, alpha: alpha * 0.5 });
      } else if (m.kind === MARKER_COMMS) {
        const r = this.cellPx * (0.4 + p * 1.3);
        g.circle(ax, ay, r).stroke({ width: 1.4 * lw, color: m.color, alpha: alpha * 0.7 });
      }
      // keep the throttle key alive as long as its marker is
      this._actionKeys.set(m.key, m.ttl);
    }
    this._markers = keep;
  }

  // -- events ----------------------------------------------------------------

  _paintEvents(events) {
    for (const c of this.eventLayer.removeChildren()) c.destroy();
    const cp = this.cellPx;
    const lw = 1 / this._zoom;
    for (const e of events) {
      const cx = this.offX + (e.x + 0.5) * cp;
      const cy = this.offY + (e.y + 0.5) * cp;
      const col = EVENT_COLORS[e.kind] ?? COLORS.eventDefault;
      const rad = Math.max(1, e.r) * cp;
      const g = new Graphics();
      g.circle(cx, cy, rad).stroke({ width: 1.5 * lw, color: col, alpha: 0.7 });
      g.circle(cx, cy, rad * 0.6).stroke({ width: lw, color: col, alpha: 0.4 });
      const m = rad + 4;
      g.moveTo(cx - m, cy).lineTo(cx - m + 6, cy);
      g.moveTo(cx + m - 6, cy).lineTo(cx + m, cy);
      g.moveTo(cx, cy - m).lineTo(cx, cy - m + 6);
      g.moveTo(cx, cy + m - 6).lineTo(cx, cy + m);
      g.stroke({ width: lw, color: col, alpha: 0.7 });
      this.eventLayer.addChild(g);
    }
  }

  // -- animation -------------------------------------------------------------

  _tick(deltaMS) {
    const step = deltaMS / 90;
    const dt = deltaMS / 1000;
    const cp = this.cellPx;
    const z = this._zoom;
    const lw = 1 / z; // constant on-screen line width while zoomed
    const px = (cx) => this.offX + (cx + 0.5) * cp;
    const py = (cy) => this.offY + (cy + 0.5) * cp;

    // drive the terrain shader's animation from the display-rate ticker: water
    // shimmer runs off uTime, day/dusk warmth off uDaylight — no texture uploads
    this._terTime += dt;
    if (this.useShaderTerrain && this._terUniforms) {
      this._terUniforms.uniforms.uTime = this._terTime;
      this._terUniforms.uniforms.uDaylight = this.daylight;
    }

    // ONE global client wind vector, slowly rotating. Deterministic from the
    // ticker clock (no Math.random). Direction sweeps with a slow sine; strength
    // mirrors the sim's own gust feel (abs(sin·)·8) so rain shear / smoke drift
    // read like the world's weather. Magnitude in world px/s, scaled by cellPx.
    {
      const ang = this._terTime * 0.15; // slow direction sweep
      const gust = Math.abs(Math.sin(this._terTime * 0.11)) * 8; // 0..8, gusty
      const mag = this.cellPx * (0.4 + gust * 0.12);
      this._wind.x = Math.cos(ang) * mag;
      this._wind.y = Math.sin(ang) * mag * 0.25; // mostly horizontal
    }

    this.trailLayer.clear();
    this.fxLayer.clear();
    this.shadowLayer.clear();
    this.actionLayer.clear();

    // Trails + coop-links are NEAR-only (single top-level flag, checked once
    // per frame — not a per-sprite branch). Both are per-frame Graphics
    // redraws, so "hiding" them just means skipping the draw calls this tick.
    const nearOnly = this._lod === "near";

    // cooperation links: connect cooperating agents that are near each other
    const coop = [];
    for (const rec of this.agents.values()) if (rec.act === ACT.COOPERATE) coop.push(rec);

    const atlas = this._atlas;
    for (const rec of this.agents.values()) {
      if (rec.t < 1) rec.t = Math.min(1, rec.t + step);
      rec.actAge += dt;
      const x = rec.fromX + (rec.toX - rec.fromX) * rec.t;
      const y = rec.fromY + (rec.toY - rec.fromY) * rec.t;
      rec._px = px(x);
      rec._py = py(y);

      const sleeping = rec.act === ACT.SLEEP;
      const moving = rec.t < 1 && (rec.fromX !== rec.toX || rec.fromY !== rec.toY);
      const stageKey = STAGE_KEY[rec.st] ?? "adult";
      const st = STAGE_SCALE[rec.st] ?? 1.0;

      // zoom-aware size: never smaller than MIN_AGENT_PX on screen. worldRoot is
      // scaled by _zoom, so hpx world-px renders at hpx*_zoom screen-px → the
      // floor is MIN_AGENT_PX / _zoom in world units. Everything below (shadow,
      // glow, emote, selection ring) derives from hpx so it all grows together.
      const hpx = Math.max(st * cp * (sleeping ? 0.9 : 1.45), MIN_AGENT_PX / this._zoom);
      rec.hpx = hpx;

      // --- animation driver: action loop > walk > idle -------------------------
      // Action animation has priority; walk plays only when there is no action
      // animation AND the agent is moving; otherwise idle (breathing transform).
      // FAR-LOD is a texture-lookup override: one static silhouette frame, no
      // per-action pose loop, no breathing/lunge (a single branch here, not a
      // separate per-tick pass over all sprites).
      const far = this._lod === "far";
      let actKey, frameIdx;
      if (far) {
        actKey = "pawn";
        frameIdx = 0;
      } else if (sleeping) {
        actKey = "sleep";
        frameIdx = 0;
      } else {
        const codeKey = ACT_KEY[rec.act] ?? "idle";
        const A = atlas.anim[codeKey];
        if (A && A.durMs > 0 && A.n > 1) {
          // a real per-action pose loop (forage/cooperate/attack/build)
          actKey = codeKey;
          frameIdx = Math.floor(rec.actAge * 1000 / A.durMs) % A.n;
        } else if (moving) {
          const w = atlas.anim.walk;
          actKey = "walk";
          frameIdx = Math.floor(rec.actAge * 1000 / w.durMs) % w.n;
        } else {
          actKey = "idle";
          frameIdx = 0;
        }
      }
      const tex = atlas.frames[stageKey][actKey][frameIdx];
      if (rec.figure.texture !== tex) rec.figure.texture = tex;

      // base scale from height; idle breathes via a tiny scale.y wobble (no
      // extra textures, skipped at FAR — the pawn silhouette doesn't breathe);
      // facing mirrors via negative scale.x (flips children).
      const base = hpx / tex.height;
      const breathe =
        !far && actKey === "idle" && !sleeping ? 1 + 0.015 * Math.sin(this._pulse * 4 + rec._px) : 1;
      rec.figure.scale.set(base);
      rec.figure.scale.y = base * breathe;
      rec.figure.scale.x = base * rec.facing;

      // walking bob while between cells; standing still otherwise (none at FAR)
      const bob = !far && moving ? Math.abs(Math.sin(this._pulse * 14 + rec._px)) * hpx * 0.055 : 0;

      // attack lunge: shove the body forward on the lunge frame (attack1), eased
      let lunge = 0;
      if (!far && actKey === "attack") {
        // triangular ease centred on frame 1 (the lunge) of the 3-frame loop.
        // Deliberately scaled by hpx (not cellPx): the lunge must stay
        // proportional to the drawn figure even when the MIN_AGENT_PX floor
        // has lifted it above cell size.
        const phase = (rec.actAge * 1000 / atlas.anim.attack.durMs) % atlas.anim.attack.n;
        lunge = rec.facing * hpx * 0.25 * Math.max(0, 1 - Math.abs(phase - 1));
      }

      rec.glow.x = rec._px;
      rec.glow.y = rec._py + hpx * 0.14;
      rec.glow.scale.set((hpx * 1.3) / 64);
      rec.figure.x = rec._px + lunge;
      rec.figure.y = rec._py + hpx * 0.24 - bob;

      // --- tool follows the hand ----------------------------------------------
      // hand anchor is figure-local px (children inherit the figure scale + the
      // facing flip), so set the tool right on it. The parent's negative scale.x
      // already mirrors a child's rotation visually (mirror∘R(θ) = R(−θ)∘mirror),
      // so the raw h.rot is correct for BOTH facings — multiplying by facing
      // would cancel the mirror and tilt the tool away from the swing.
      // Hidden while asleep.
      if (rec.tool.visible) {
        const h = atlas.hand[stageKey][actKey][frameIdx];
        rec.tool.position.set(h.x, h.y);
        rec.tool.rotation = h.rot;
      }

      // grounding shadow under the feet — hidden at FAR (per-agent-shadow)
      if (!far) {
        this.shadowLayer
          .ellipse(rec._px, rec._py + hpx * 0.3, hpx * (sleeping ? 0.34 : 0.22), hpx * 0.08)
          .fill({ color: 0x000000, alpha: 0.17 });
      }

      // emote floats above the head with a gentle bob
      if (rec.emote.visible) {
        rec.emote.scale.set((hpx * 0.5) / 12);
        rec.emote.x = rec._px;
        rec.emote.y = rec.figure.y - hpx * 0.85 - Math.sin(this._pulse * 3) * hpx * 0.05;
      }

      // motion trail (footsteps of the recent path) — NEAR only
      const pts = rec.trail;
      if (nearOnly && pts.length > 1 && rec.act !== ACT.SLEEP) {
        for (let i = 1; i < pts.length; i++) {
          this.trailLayer
            .moveTo(px(pts[i - 1][0]), py(pts[i - 1][1]))
            .lineTo(px(pts[i][0]), py(pts[i][1]))
            .stroke({ width: lw, color: rec.tint, alpha: (i / pts.length) * 0.2 });
        }
      }

      // attack stays dramatic beyond the pose: sharp expanding burst
      if (rec.act === ACT.ATTACK) {
        const t = rec.actAge % 0.6;
        const rr = hpx * (0.28 + t * 1.5);
        const col = ACT_COLORS[ACT.ATTACK];
        this.fxLayer
          .circle(rec._px, rec._py, rr)
          .stroke({ width: 2 * lw, color: col, alpha: Math.max(0, 0.85 - t * 1.4) });
      }
    }

    // cooperate: turn each cooperating agent toward its nearest coop partner so
    // the extended arm reads as reaching for someone (the coop[] array already
    // exists for the link lines below).
    for (let i = 0; i < coop.length; i++) {
      const a = coop[i];
      let bx = null;
      let bestD = Infinity;
      for (let j = 0; j < coop.length; j++) {
        if (j === i) continue;
        const dx = coop[j].toX - a.toX;
        const dy = coop[j].toY - a.toY;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          bx = coop[j].toX;
        }
      }
      if (bx != null && bx !== a.toX) {
        a.facing = bx > a.toX ? 1 : -1;
        a.figure.scale.x = Math.abs(a.figure.scale.x) * a.facing;
      }
    }

    // pooled action-line markers (attack/coop/forage/build/comms) — after all
    // agents have their eased _px/_py this tick so lines land on the drawn
    // figures. Cleared at the top of the tick; endpoints track motion.
    this._drawMarkers(dt, lw);

    // selection ring + intent line + floating label for the inspected agent.
    let selRec = null;
    if (this.selectedId != null) selRec = this.agents.get(this.selectedId) ?? null;
    if (selRec) {
      const rec = selRec;
      this.fxLayer
        .circle(rec._px, rec._py, rec.hpx * 0.58)
        .stroke({ width: 1.6 * lw, color: 0xffffff, alpha: 0.85 });
      this.fxLayer
        .circle(rec._px, rec._py, rec.hpx * 0.72)
        .stroke({ width: lw, color: rec.tint ?? 0xffffff, alpha: 0.5 });

      // Intent line — dotted, dezent — from the selected agent to its goal cell
      // gx/gy. ONLY for the selected agent (anti-clutter). Drawn as short dashes
      // along the segment so it reads as "headed there" without a solid rail.
      if (rec.gx != null && rec.gy != null) {
        const gx = px(rec.gx);
        const gy = py(rec.gy);
        const dx = gx - rec._px;
        const dy = gy - rec._py;
        const len = Math.hypot(dx, dy);
        if (len > 1) {
          const ux = dx / len;
          const uy = dy / len;
          const dash = cp * 0.35;
          const gap = cp * 0.3;
          for (let d = rec.hpx * 0.5; d < len; d += dash + gap) {
            const d2 = Math.min(len, d + dash);
            this.fxLayer
              .moveTo(rec._px + ux * d, rec._py + uy * d)
              .lineTo(rec._px + ux * d2, rec._py + uy * d2)
              .stroke({ width: lw, color: 0xdfe8ff, alpha: 0.4 });
          }
          this.fxLayer
            .circle(gx, gy, cp * 0.28)
            .stroke({ width: lw, color: 0xdfe8ff, alpha: 0.4 });
        }
      }

      // floating "why" label over the head; text pushed from World.svelte.
      if (this._selLabel && this._selLabelText) {
        this._selLabel.visible = true;
        this._selLabel.x = rec._px;
        this._selLabel.y = rec.figure.y - rec.hpx * 1.15;
        // keep label legible regardless of world zoom (counter-scale it)
        this._selLabel.scale.set(1 / this._zoom);
      } else if (this._selLabel) {
        this._selLabel.visible = false;
      }
    } else if (this._selLabel) {
      this._selLabel.visible = false;
    }

    // follow camera — ease worldRoot so the followed agent stays centred.
    if (this.followId != null) {
      const rec = this.agents.get(this.followId);
      if (rec && rec._px != null) {
        const W = this.app.renderer.width;
        const H = this.app.renderer.height;
        const targetX = W / 2 - rec._px * this._zoom;
        const targetY = H / 2 - rec._py * this._zoom;
        this.worldRoot.x += (targetX - this.worldRoot.x) * 0.12;
        this.worldRoot.y += (targetY - this.worldRoot.y) * 0.12;
        this._clampPan();
      } else if (!rec) {
        this.followId = null; // followed agent gone → stop following
        this.onFollowChange?.(null);
      }
    }

    // cooperation: link nearby cooperating agents + halo — NEAR only
    if (nearOnly) {
      for (let i = 0; i < coop.length; i++) {
        const a = coop[i];
        this.fxLayer
          .circle(a._px, a._py, cp * 0.7)
          .stroke({ width: lw, color: ACT_COLORS[ACT.COOPERATE], alpha: 0.5 });
        for (let j = i + 1; j < coop.length; j++) {
          const b = coop[j];
          const dx = a.toX - b.toX;
          const dy = a.toY - b.toY;
          if (dx * dx + dy * dy <= 16) {
            this.fxLayer
              .moveTo(a._px, a._py)
              .lineTo(b._px, b._py)
              .stroke({ width: lw, color: ACT_COLORS[ACT.COOPERATE], alpha: 0.45 });
          }
        }
      }
    }

    // weather is visible weather — now a real, deterministic particle system.
    // Storm rain, drought dust and blight spores are EMITTED here from the live
    // events (the old Math.random() rain scribble is gone). The flat
    // fire/drought/blight tint discs are the OLD event look; when the shader
    // terrain is on it paints those as ground scars, so the discs only draw as a
    // fallback when useShaderTerrain is off.
    const flatEvents = !this.useShaderTerrain;
    for (const e of this._events) {
      const ex = px(e.x);
      const ey = py(e.y);
      const rad = Math.max(1, e.r) * cp;
      if (e.kind === "storm") {
        // rain particles inside the disc, sheared by wind (emitter uses cell
        // coords + cellPx internally, in the same world-px space).
        this.particles?.emitters.rain(ex, ey, e.r, e.i ?? 0.6, cp);
      } else if (e.kind === "drought") {
        this.particles?.emitters.dust(ex, ey, rad, cp);
        if (flatEvents)
          this.fxLayer.circle(ex, ey, rad).fill({ color: 0xf0b030, alpha: 0.05 });
      } else if (e.kind === "blight") {
        this.particles?.emitters.spores(ex, ey, rad, cp);
        if (flatEvents)
          this.fxLayer.circle(ex, ey, rad).fill({ color: 0xc774f0, alpha: 0.05 });
      } else if (flatEvents && e.kind === "fire") {
        const fl = 0.06 + 0.05 * Math.sin(this._pulse * 11 + e.x);
        this.fxLayer.circle(ex, ey, rad).fill({ color: 0xff7a3c, alpha: fl });
      }
    }

    // -- fire & smoke particles + additive night-glow -------------------------
    // Fire sources = FIRE ground items (persistent _fireSprites, at their world
    // px) PLUS fire EVENTS (this frame's _events). Each throws flames + smoke;
    // camp chimneys throw smoke only. A soft additive glow sprite (reused pool)
    // sits on every fire source with alpha ∝ (1 - daylight) → fires light night.
    if (this.particles) {
      let gi = 0; // index into the reused fire-glow sprite pool
      const nightGlow = Math.max(0, 1 - this.daylight);
      const placeGlow = (gx, gy, scale) => {
        let g = this._fireGlows[gi];
        if (!g) {
          g = new Sprite(this._deathGlowTex);
          g.anchor.set(0.5);
          g.blendMode = "add";
          this.glowLayer.addChild(g);
          this._fireGlows[gi] = g;
        }
        g.visible = true;
        g.x = gx;
        g.y = gy;
        g.scale.set(scale);
        // even by day a small ember glow; by night it blooms into a light source
        g.alpha = 0.12 + 0.6 * nightGlow + 0.06 * Math.sin(this._pulse * 7 + gx);
        gi++;
      };
      for (const s of this._fireSprites) {
        this.particles.emitters.fire(s.x, s.y - cp * 0.2, cp, (s.x * 7) | 0);
        this.particles.emitters.smoke(s.x, s.y - cp * 0.6, cp, (s.x * 3) | 0);
        placeGlow(s.x, s.y - cp * 0.4, (cp * 2.4) / 64);
      }
      for (const e of this._events) {
        if (e.kind !== "fire") continue;
        const ex = px(e.x);
        const ey = py(e.y);
        this.particles.emitters.fire(ex, ey, cp, (e.x * 11) | 0);
        this.particles.emitters.smoke(ex, ey - cp * 0.4, cp, (e.x * 5) | 0);
        placeGlow(ex, ey, ((Math.max(1, e.r) * cp) / 64) * 1.5);
      }
      // camp chimney smoke
      for (let c = 0; c < this._chimneys.length; c += 2) {
        this.particles.emitters.smoke(this._chimneys[c], this._chimneys[c + 1], cp, c);
      }
      // hide surplus glow sprites from a previous, larger fire set
      for (; gi < this._fireGlows.length; gi++) this._fireGlows[gi].visible = false;
    }

    // -- death fades ----------------------------------------------------------
    // Records whose id vanished linger ~600 ms: figure/glow fade to 0, then the
    // sprites are destroyed and the record drops from _dying. The rising wisp is
    // already airborne (emitted in _syncAgents). Fade runs at ALL LODs.
    if (this._dying.length) {
      const keep = [];
      for (const d of this._dying) {
        d.age += dt;
        const p = d.age / 0.6;
        if (p >= 1) {
          d.rec.glow.destroy();
          d.rec.figure.destroy({ children: true });
          d.rec.emote.destroy();
          continue;
        }
        keep.push(d);
        const a = 1 - p;
        d.rec.figure.alpha = a;
        d.rec.glow.alpha = a * 0.16;
      }
      this._dying = keep;
    }

    // knapping sparks / discovery / fire-lit moments: a short radiant star
    if (this._bursts.length) {
      const keep = [];
      for (const b of this._bursts) {
        b.age += dt;
        if (b.age >= 0.8) continue;
        keep.push(b);
        const bx = px(b.cx);
        const by = py(b.cy);
        const p = b.age / 0.8;
        const alpha = 0.9 * (1 - p);
        const r0 = cp * (0.2 + p * 0.9);
        const r1 = cp * (0.55 + p * 1.6);
        for (let k = 0; k < 6; k++) {
          const ang = (k / 6) * Math.PI * 2 + 0.35;
          this.fxLayer
            .moveTo(bx + Math.cos(ang) * r0, by + Math.sin(ang) * r0)
            .lineTo(bx + Math.cos(ang) * r1, by + Math.sin(ang) * r1)
            .stroke({ width: 1.5 * lw, color: b.color, alpha });
        }
        this.fxLayer
          .circle(bx, by, r0 * 0.9)
          .stroke({ width: lw, color: 0xffffff, alpha: alpha * 0.7 });
      }
      this._bursts = keep;
    }

    // fire flickers
    for (const s of this._fireSprites) {
      s.alpha = 0.82 + 0.18 * Math.sin(this._pulse * 9 + s.x * 0.7);
      s.scale.set(s._bs * (1 + 0.06 * Math.sin(this._pulse * 13 + s.x)));
    }

    // event pulse
    this._pulse += dt;
    const a = 0.45 + 0.35 * Math.sin(this._pulse * 4);
    for (const g of this.eventLayer.children) g.alpha = a;

    // advance the pooled particle system last (all emitters have fed it this
    // frame). O(active); the wind vector drives shear/drift inside.
    this.particles?.update(deltaMS, this._wind);

    if (deltaMS > 0) this._fps += (1000 / deltaMS - this._fps) * 0.08;
    this._hudAccum += deltaMS;
    if (this._hudAccum >= 300) {
      this._hudAccum = 0;
      this.onHud?.({
        fps: Math.round(this._fps),
        agents: this.agents.size,
        zoom: Math.round(this._zoom * 10) / 10,
      });
    }
  }

  destroy() {
    this._destroyShaderTerrain();
    this.particles?.destroy();
    if (this.app) this.app.destroy(true, { children: true });
  }
}

// -- helpers -----------------------------------------------------------------

function splitRGB(int) {
  return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Fallback for biomes without a curated palette: quiet the legend color.
function quietFallback(rgb, gain) {
  const L = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
  const keep = 0.5;
  const mk = (ch) => clamp((L * 255 * (1 - keep) + ch * keep) * 0.45 * gain + 8);
  return { r: mk(rgb[0]), g: mk(rgb[1]), b: mk(rgb[2]) };
}

// Lift an agent's colour into a luminous signal while keeping its identity.
function blip(hexStr) {
  const n = parseInt(hexStr.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return (clamp(r * 0.7 + 26) << 16) | (clamp(g * 0.7 + 86) << 8) | clamp(b * 0.7 + 104);
}

function clamp(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function clampNum(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function makeRadialTexture(size, stops) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const r = size / 2;
  const grad = ctx.createRadialGradient(r, r, 0, r, r, r);
  for (const [pos, col] of stops) grad.addColorStop(pos, col);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return Texture.from(c);
}
