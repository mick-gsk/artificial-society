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

import { Application, Container, Graphics, Sprite, Texture } from "pixi.js";

import {
  makeAgentTextures,
  makeDecorTextures,
  makeEmoteTextures,
  makeItemTextures,
  makeSleepingTexture,
  makeStructureTextures,
  makeToolTextures,
} from "./sprites.js";

const STAGE_SCALE = [0.62, 1.0, 1.06]; // child, adult, elder (figure size)
const ACT_EMOTE = { 1: "forage", 2: "cooperate", 3: "attack", 4: "build", 5: "sleep" };
const TRAIL_LEN = 8;
const ZOOM_MIN = 1;
const ZOOM_MAX = 8;

const EVENT_COLORS = {
  drought: 0xf0b030,
  storm: 0x4cc6ff,
  fire: 0xff7a3c,
  blight: 0xc774f0,
};

// Action codes from serve/frame.py: 0 idle/move, 1 forage, 2 cooperate,
// 3 attack, 4 build, 5 sleep.
export const ACT = { IDLE: 0, FORAGE: 1, COOPERATE: 2, ATTACK: 3, BUILD: 4, SLEEP: 5 };
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

    // terrain texture state (subpixel canvas)
    this._terCanvas = null;
    this._terCtx = null;
    this._terImg = null;
    this._terTex = null;
    this._terSprite = null;
    this._baseRGB = null; // per-cell day-lit base color, for neighbour blending
    this._isWater = null;
    this._frameNo = 0;
    this._phase = 0; // water shimmer phase

    // view state
    this._zoom = 1;
    this._drag = null;
    this.onPick = null; // (agentId | null) => void
    this.selectedId = null;

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
    this._figSet = makeAgentTextures();
    this._figElderSet = makeAgentTextures({ elder: true });
    this._sleepTex = makeSleepingTexture();
    this._emoteTex = makeEmoteTextures();
    this._decorTex = makeDecorTextures();
    this._itemTex = makeItemTextures();
    this._toolTex = makeToolTextures();
    this._structTex = makeStructureTextures();
    this._bindViewControls();
    this.app.ticker.add((t) => this._tick(t.deltaMS));
    this.app.renderer.on("resize", () => this._rebuildGrid());
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
        const z = clampNum(this._zoom * Math.exp(-e.deltaY * 0.0018), ZOOM_MIN, ZOOM_MAX);
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
    });
  }

  _applyZoom(z, cx, cy) {
    const k = z / this._zoom;
    this.worldRoot.x = cx - (cx - this.worldRoot.x) * k;
    this.worldRoot.y = cy - (cy - this.worldRoot.y) * k;
    this._zoom = z;
    this.worldRoot.scale.set(z);
    this._clampPan();
  }

  _clampPan() {
    const W = this.app.renderer.width;
    const H = this.app.renderer.height;
    const z = this._zoom;
    this.worldRoot.x = clampNum(this.worldRoot.x, W - W * z, 0);
    this.worldRoot.y = clampNum(this.worldRoot.y, H - H * z, 0);
  }

  _pick(sx, sy) {
    const wx = (sx - this.worldRoot.x) / this._zoom;
    const wy = (sy - this.worldRoot.y) / this._zoom;
    let best = null;
    let bestD = this.cellPx * this.cellPx;
    for (const [id, rec] of this.agents) {
      const dx = rec._px - wx;
      const dy = rec._py - wy;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = id;
      }
    }
    this.selectedId = best;
    this.onPick?.(best);
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
    this._paintTerrain(frame.cells);
    this._buildDecor();
    this._paintItems(frame.items ?? []);
    this._paintStructures(frame.structures ?? []);
    this._syncAgents(frame.agents);
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
          this._bursts.push({ cx: x, cy: y, age: 0, color: BURST_COLOR[k] ?? 0xffd166 });
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

    this._structKey = ""; // force structure re-layout at the new scale
    this._itemKey = "";
    this._drawGrid();
  }

  _drawGrid() {
    const { w, h } = this.grid;
    const cp = this.cellPx;
    const x0 = this.offX;
    const y0 = this.offY;
    const x1 = x0 + w * cp;
    const y1 = y0 + h * cp;
    const step = w > 40 ? 10 : 5;

    this.gridLayer.removeChildren().forEach((c) => c.destroy());
    const g = new Graphics();
    for (let x = 0; x <= w; x += step) {
      const px = x0 + x * cp;
      g.moveTo(px, y0).lineTo(px, y1);
    }
    for (let y = 0; y <= h; y += step) {
      const py = y0 + y * cp;
      g.moveTo(x0, py).lineTo(x1, py);
    }
    g.stroke({ width: 1, color: COLORS.gridLine, alpha: 0.22 });
    g.rect(x0, y0, w * cp, h * cp).stroke({
      width: 1,
      color: COLORS.gridTick,
      alpha: 0.5,
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
    for (const s of structures) {
      const t = this._structTex[s.k];
      if (!t) continue;
      const spr = new Sprite(t);
      spr.anchor.set(0.5, 1);
      spr.x = this.offX + (s.x + 0.5) * cp;
      spr.y = this.offY + (s.y + 0.98) * cp;
      spr.scale.set((cp * 1.15) / t.height);
      this.structLayer.addChild(spr);
    }
    this.structLayer.children.sort((a, b) => a.y - b.y);
  }

  // -- agents ----------------------------------------------------------------

  _syncAgents(agents) {
    const seen = new Set();
    for (const a of agents) {
      seen.add(a.id);
      let rec = this.agents.get(a.id);
      if (!rec) {
        const glow = new Sprite(this._glowTex);
        const figure = new Sprite(this._figSet.stand);
        const emote = new Sprite(this._emoteTex.forage);
        glow.anchor.set(0.5);
        figure.anchor.set(0.5, 0.85); // feet on the cell
        emote.anchor.set(0.5, 1);
        emote.visible = false;
        // tool in the hand + carry bundle at the hip, in figure-local pixels
        // (children inherit scale, walking bob and the facing flip)
        const tool = new Sprite(this._toolTex.blunt);
        tool.anchor.set(0.5, 1);
        tool.position.set(5.5, -3.5);
        tool.rotation = 0.3;
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
        };
        this.agents.set(a.id, rec);
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
      const tint = blip(a.col);
      rec.tint = tint;
      rec.energy = a.e;
      const cp = this.cellPx;
      const sleeping = rec.act === ACT.SLEEP;

      rec.figure.tint = tint;
      rec.hpx = (STAGE_SCALE[rec.st] ?? 1.0) * cp * (sleeping ? 0.9 : 1.45);
      rec.figure.alpha = sleeping ? 0.75 : 1.0;

      // what the body carries: blade/stone in hand, bundle at the hip
      rec.tool.visible = !sleeping && rec.tl > 0;
      if (rec.tl > 0) rec.tool.texture = rec.tl === 2 ? this._toolTex.sharp : this._toolTex.blunt;
      rec.bundle.visible = !sleeping && (a.cg ?? 0) >= 2;

      // soft ground glow keeps agents findable at night
      rec.glow.tint = tint;
      rec.glow.scale.set((cp * 1.9) / 64);
      rec.glow.alpha = sleeping ? 0.08 : 0.16 + 0.12 * (1 - this.daylight);

      // emote bubble above the head while acting
      const emoteName = ACT_EMOTE[rec.act];
      if (emoteName) {
        rec.emote.texture = this._emoteTex[emoteName];
        rec.emote.visible = true;
        rec.emote.scale.set((cp * 0.85) / 12);
      } else {
        rec.emote.visible = false;
      }
    }
    for (const [id, rec] of this.agents) {
      if (!seen.has(id)) {
        rec.glow.destroy();
        rec.figure.destroy({ children: true }); // takes tool + bundle with it
        rec.emote.destroy();
        this.agents.delete(id);
        if (this.selectedId === id) {
          this.selectedId = null;
          this.onPick?.(null);
        }
      }
    }
    // lower agents render in front (simple painter's order)
    this.figureLayer.children.sort((a, b) => a.y - b.y);
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

    this.trailLayer.clear();
    this.fxLayer.clear();
    this.shadowLayer.clear();

    // cooperation links: connect cooperating agents that are near each other
    const coop = [];
    for (const rec of this.agents.values()) if (rec.act === ACT.COOPERATE) coop.push(rec);

    for (const rec of this.agents.values()) {
      if (rec.t < 1) rec.t = Math.min(1, rec.t + step);
      rec.actAge += dt;
      const x = rec.fromX + (rec.toX - rec.fromX) * rec.t;
      const y = rec.fromY + (rec.toY - rec.fromY) * rec.t;
      rec._px = px(x);
      rec._py = py(y);

      const sleeping = rec.act === ACT.SLEEP;
      const moving = rec.t < 1 && (rec.fromX !== rec.toX || rec.fromY !== rec.toY);

      // body texture: lying asleep / two-frame walk / stand
      const set = rec.st === 2 ? this._figElderSet : this._figSet;
      let tex;
      if (sleeping) tex = this._sleepTex;
      else if (moving) tex = Math.floor(this._pulse * 7 + rec._px * 0.11) & 1 ? set.walkA : set.walkB;
      else tex = set.stand;
      if (rec.figure.texture !== tex) rec.figure.texture = tex;
      rec.figure.scale.set(rec.hpx / tex.height);
      rec.figure.scale.x *= rec.facing;

      // walking bob while between cells; standing still otherwise
      const bob = moving ? Math.abs(Math.sin(this._pulse * 14 + rec._px)) * cp * 0.08 : 0;
      rec.glow.x = rec._px;
      rec.glow.y = rec._py + cp * 0.2;
      rec.figure.x = rec._px;
      rec.figure.y = rec._py + cp * 0.35 - bob;

      // grounding shadow under the feet
      this.shadowLayer
        .ellipse(rec._px, rec._py + cp * 0.38, cp * (sleeping ? 0.42 : 0.28), cp * 0.1)
        .fill({ color: 0x000000, alpha: 0.17 });

      // emote floats above the head with a gentle bob
      if (rec.emote.visible) {
        rec.emote.x = rec._px;
        rec.emote.y =
          rec.figure.y - rec.figure.height * 0.95 - cp * 0.1 - Math.sin(this._pulse * 3) * cp * 0.06;
      }

      // motion trail (footsteps of the recent path)
      const pts = rec.trail;
      if (pts.length > 1 && rec.act !== ACT.SLEEP) {
        for (let i = 1; i < pts.length; i++) {
          this.trailLayer
            .moveTo(px(pts[i - 1][0]), py(pts[i - 1][1]))
            .lineTo(px(pts[i][0]), py(pts[i][1]))
            .stroke({ width: lw, color: rec.tint, alpha: (i / pts.length) * 0.2 });
        }
      }

      // attack stays dramatic beyond the emote: sharp expanding burst
      if (rec.act === ACT.ATTACK) {
        const t = rec.actAge % 0.6;
        const rr = cp * (0.4 + t * 2.2);
        const col = ACT_COLORS[ACT.ATTACK];
        this.fxLayer
          .circle(rec._px, rec._py, rr)
          .stroke({ width: 2 * lw, color: col, alpha: Math.max(0, 0.85 - t * 1.4) });
      }
    }

    // selection ring for the inspector
    if (this.selectedId != null) {
      const rec = this.agents.get(this.selectedId);
      if (rec) {
        this.fxLayer
          .circle(rec._px, rec._py, cp * 0.85)
          .stroke({ width: 1.6 * lw, color: 0xffffff, alpha: 0.85 });
        this.fxLayer
          .circle(rec._px, rec._py, cp * 1.05)
          .stroke({ width: lw, color: rec.tint ?? 0xffffff, alpha: 0.5 });
      }
    }

    // cooperation: link nearby cooperating agents + halo
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

    // weather is visible weather: rain inside storms, glow + haze for the rest
    for (const e of this._events) {
      const ex = px(e.x);
      const ey = py(e.y);
      const rad = Math.max(1, e.r) * cp;
      if (e.kind === "storm") {
        const drops = Math.min(70, Math.ceil(e.r * e.r * 1.5));
        for (let i = 0; i < drops; i++) {
          const ang = Math.random() * Math.PI * 2;
          const rr = Math.sqrt(Math.random()) * rad;
          const dx = ex + Math.cos(ang) * rr;
          const dy = ey + Math.sin(ang) * rr;
          this.fxLayer
            .moveTo(dx, dy)
            .lineTo(dx - cp * 0.12, dy + cp * 0.4)
            .stroke({ width: lw, color: 0x9fd4ff, alpha: 0.35 });
        }
      } else if (e.kind === "fire") {
        const fl = 0.06 + 0.05 * Math.sin(this._pulse * 11 + e.x);
        this.fxLayer.circle(ex, ey, rad).fill({ color: 0xff7a3c, alpha: fl });
      } else if (e.kind === "drought") {
        this.fxLayer.circle(ex, ey, rad).fill({ color: 0xf0b030, alpha: 0.05 });
      } else if (e.kind === "blight") {
        this.fxLayer.circle(ex, ey, rad).fill({ color: 0xc774f0, alpha: 0.05 });
      }
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
