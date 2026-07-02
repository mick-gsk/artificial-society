// PixiJS (WebGL) renderer for the live simulation world.
//
// Design language: a living landscape under an observation instrument. Terrain
// reads as actual land — forest, grassland, desert, water — modulated by food
// (lushness), moisture and the day/night cycle. Life on top stays luminous:
// agents are glowing blips with motion trails, and every non-idle action has a
// legible visual verb (forage pulse, attack burst, build flash, cooperation
// links, sleep dimming). Structures persist as small icons.
//
// Layers, back to front:
//   terrain → grid → structures → trails → fx (action effects) → glow → cores → events
//
// Frames arrive ~20 Hz; the ticker runs at display rate and eases agent motion
// between frames so movement stays fluid.

import { Application, Container, Graphics, Sprite, Text, Texture } from "pixi.js";

import {
  makeAgentTexture,
  makeDecorTextures,
  makeEmoteTextures,
  makeSleepingTexture,
} from "./sprites.js";

const STAGE_SCALE = [0.62, 1.0, 1.06]; // child, adult, elder (figure size)
const ACT_EMOTE = { 1: "forage", 2: "cooperate", 3: "attack", 4: "build", 5: "sleep" };
const TRAIL_LEN = 8;

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
  tickText: 0x6c8bb0,
  eventDefault: 0xf0b030,
  camp: 0xffb54d,
  farm: 0x74c69d,
  well: 0x64b5f6,
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

    this.terrainLayer = new Container();
    this.decorLayer = new Container();
    this.gridLayer = new Container();
    this.structLayer = new Container();
    this.trailLayer = new Graphics();
    this.fxLayer = new Graphics();
    this.glowLayer = new Container();
    this.figureLayer = new Container();
    this.emoteLayer = new Container();
    this.eventLayer = new Container();

    this.terrainSprites = [];
    this.agents = new Map(); // id -> {glow, figure, emote, fromX..toY, t, trail, act, actAge, tint}
    this._structKey = "";
    this._decorKey = "";
    this._biomeArr = null;

    this._glowTex = null;
    this._dotTex = null;
    this._pulse = 0;
    this._fps = 60;
    this._hudAccum = 0;
    this.onHud = null;
  }

  async init(host) {
    await this.app.init({ background: 0x05070b, antialias: true, resizeTo: host });
    host.appendChild(this.app.canvas);
    this.glowLayer.blendMode = "add";
    this.app.stage.addChild(
      this.terrainLayer,
      this.decorLayer,
      this.gridLayer,
      this.structLayer,
      this.trailLayer,
      this.fxLayer,
      this.glowLayer,
      this.figureLayer,
      this.emoteLayer,
      this.eventLayer,
    );
    this._glowTex = makeRadialTexture(64, [
      [0, "rgba(255,255,255,1)"],
      [0.4, "rgba(255,255,255,0.45)"],
      [1, "rgba(255,255,255,0)"],
    ]);
    this._figTex = makeAgentTexture();
    this._figElderTex = makeAgentTexture({ elder: true });
    this._sleepTex = makeSleepingTexture();
    this._emoteTex = makeEmoteTextures();
    this._decorTex = makeDecorTextures();
    this.app.ticker.add((t) => this._tick(t.deltaMS));
    this.app.renderer.on("resize", () => this._rebuildGrid());
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
    if (resized) this._rebuildGrid();
    this._paintTerrain(frame.cells);
    this._buildDecor();
    this._paintStructures(frame.structures ?? []);
    this._syncAgents(frame.agents);
    this._paintEvents(frame.events);
  }

  // -- terrain decorations (trees, rocks, tufts) — static per world ------------

  _buildDecor() {
    if (!this.grid || !this._biomeArr || !Object.keys(this.biomeTones).length) return;
    const { w, h } = this.grid;
    const key = `${w}x${h}@${this.cellPx}`;
    if (key === this._decorKey) return;
    this._decorKey = key;
    this.decorLayer.removeChildren().forEach((c) => c.destroy());

    // biome idx -> name (via tones' source order isn't kept; rebuild from legend names)
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

    this._ensurePool(this.terrainLayer, this.terrainSprites, w * h);
    for (let i = 0; i < w * h; i++) {
      const s = this.terrainSprites[i];
      s.x = this.offX + (i % w) * this.cellPx;
      s.y = this.offY + Math.floor(i / w) * this.cellPx;
      s.width = this.cellPx;
      s.height = this.cellPx;
    }
    this._structKey = ""; // force structure re-layout at the new scale
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
    g.stroke({ width: 1, color: COLORS.gridLine, alpha: 0.35 });
    g.rect(x0, y0, w * cp, h * cp).stroke({
      width: 1,
      color: COLORS.gridTick,
      alpha: 0.55,
    });
    this.gridLayer.addChild(g);

    const mk = (label, px, py, anchorX, anchorY) => {
      const t = new Text({
        text: label,
        style: {
          fontFamily: "ui-monospace, Menlo, monospace",
          fontSize: 10,
          fill: COLORS.tickText,
        },
      });
      t.anchor.set(anchorX, anchorY);
      t.x = px;
      t.y = py;
      this.gridLayer.addChild(t);
    };
    for (let x = 0; x <= w; x += step) mk(String(x), x0 + x * cp, y0 - 4, 0.5, 1);
    for (let y = 0; y <= h; y += step) mk(String(y), x0 - 6, y0 + y * cp, 1, 0.5);
  }

  _ensurePool(layer, arr, n) {
    while (arr.length < n) {
      const s = new Sprite(Texture.WHITE);
      layer.addChild(s);
      arr.push(s);
    }
    while (arr.length > n) {
      const s = arr.pop();
      layer.removeChild(s);
      s.destroy();
    }
  }

  _paintTerrain(cells) {
    const { food, water, biome } = cells;
    // Night dims and cools the land; water keeps a bit more of its light.
    const day = 0.45 + 0.55 * this.daylight;
    const nightBlue = 1.0 - (1.0 - this.daylight) * 0.25;
    for (let i = 0; i < biome.length; i++) {
      const tone = this.biomeTones[biome[i]];
      let r, g, b;
      if (!tone) {
        r = 20;
        g = 26;
        b = 34;
      } else if (tone.isWater) {
        // moisture/water level makes water breathe between shallow and deep
        const wn = Math.min(1, water[i] / 100);
        r = lerp(tone.dry.r, tone.lush.r, wn);
        g = lerp(tone.dry.g, tone.lush.g, wn);
        b = lerp(tone.dry.b, tone.lush.b, wn);
      } else {
        // food lushness moves land between parched and verdant
        const fn = Math.min(1, food[i] / 90);
        r = lerp(tone.dry.r, tone.lush.r, fn);
        g = lerp(tone.dry.g, tone.lush.g, fn);
        b = lerp(tone.dry.b, tone.lush.b, fn);
      }
      const j = this._jitter ? this._jitter[i] : 1;
      r = clamp(r * j * day);
      g = clamp(g * j * day);
      b = clamp(b * j * (day + (1 - day) * (1 - nightBlue) * 2) * nightBlue + (1 - this.daylight) * 10);
      this.terrainSprites[i].tint = (r << 16) | (g << 8) | b;
    }
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
      const cx = this.offX + (s.x + 0.5) * cp;
      const cy = this.offY + (s.y + 0.5) * cp;
      const r = cp * 0.42;
      const g = new Graphics();
      if (s.k === "camp") {
        // tent: triangle + door slit
        g.moveTo(cx, cy - r)
          .lineTo(cx + r, cy + r * 0.8)
          .lineTo(cx - r, cy + r * 0.8)
          .closePath()
          .stroke({ width: 1.5, color: COLORS.camp, alpha: 0.95 });
        g.moveTo(cx, cy + r * 0.8)
          .lineTo(cx, cy + r * 0.1)
          .stroke({ width: 1, color: COLORS.camp, alpha: 0.7 });
      } else if (s.k === "farm") {
        // tilled plot: square with furrow lines
        g.rect(cx - r, cy - r * 0.8, r * 2, r * 1.6).stroke({
          width: 1.5,
          color: COLORS.farm,
          alpha: 0.95,
        });
        for (let k = -1; k <= 1; k++) {
          g.moveTo(cx - r * 0.7, cy + k * r * 0.45)
            .lineTo(cx + r * 0.7, cy + k * r * 0.45)
            .stroke({ width: 1, color: COLORS.farm, alpha: 0.6 });
        }
      } else if (s.k === "well") {
        // well: ring + water dot
        g.circle(cx, cy, r * 0.8).stroke({ width: 1.5, color: COLORS.well, alpha: 0.95 });
        g.circle(cx, cy, r * 0.25).fill({ color: COLORS.well, alpha: 0.8 });
      }
      this.structLayer.addChild(g);
    }
  }

  // -- agents ----------------------------------------------------------------

  _syncAgents(agents) {
    const seen = new Set();
    for (const a of agents) {
      seen.add(a.id);
      let rec = this.agents.get(a.id);
      if (!rec) {
        const glow = new Sprite(this._glowTex);
        const figure = new Sprite(this._figTex);
        const emote = new Sprite(this._emoteTex.forage);
        glow.anchor.set(0.5);
        figure.anchor.set(0.5, 0.85); // feet on the cell
        emote.anchor.set(0.5, 1);
        emote.visible = false;
        this.glowLayer.addChild(glow);
        this.figureLayer.addChild(figure);
        this.emoteLayer.addChild(emote);
        rec = {
          glow,
          figure,
          emote,
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
      }
      if ((a.act ?? 0) !== rec.act) {
        rec.act = a.act ?? 0;
        rec.actAge = 0; // restart the action animation
      }
      const tint = blip(a.col);
      rec.tint = tint;
      rec.energy = a.e;
      const cp = this.cellPx;
      const sleeping = rec.act === ACT.SLEEP;

      // body: lying when asleep, elder crown for elders, tribe tint, facing flip
      const tex = sleeping ? this._sleepTex : a.st === 2 ? this._figElderTex : this._figTex;
      if (rec.figure.texture !== tex) rec.figure.texture = tex;
      rec.figure.tint = tint;
      const hpx = (STAGE_SCALE[a.st] ?? 1.0) * cp * (sleeping ? 0.9 : 1.45);
      rec.figure.scale.set(hpx / tex.height);
      rec.figure.scale.x *= rec.facing;
      rec.figure.alpha = sleeping ? 0.75 : 1.0;

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
        rec.figure.destroy();
        rec.emote.destroy();
        this.agents.delete(id);
      }
    }
    // lower agents render in front (simple painter's order)
    this.figureLayer.children.sort((a, b) => a.y - b.y);
  }

  // -- events ----------------------------------------------------------------

  _paintEvents(events) {
    for (const c of this.eventLayer.removeChildren()) c.destroy();
    const cp = this.cellPx;
    for (const e of events) {
      const cx = this.offX + (e.x + 0.5) * cp;
      const cy = this.offY + (e.y + 0.5) * cp;
      const col = EVENT_COLORS[e.kind] ?? COLORS.eventDefault;
      const rad = Math.max(1, e.r) * cp;
      const g = new Graphics();
      g.circle(cx, cy, rad).stroke({ width: 1.5, color: col, alpha: 0.7 });
      g.circle(cx, cy, rad * 0.6).stroke({ width: 1, color: col, alpha: 0.4 });
      const m = rad + 4;
      g.moveTo(cx - m, cy).lineTo(cx - m + 6, cy);
      g.moveTo(cx + m - 6, cy).lineTo(cx + m, cy);
      g.moveTo(cx, cy - m).lineTo(cx, cy - m + 6);
      g.moveTo(cx, cy + m - 6).lineTo(cx, cy + m);
      g.stroke({ width: 1, color: col, alpha: 0.7 });
      this.eventLayer.addChild(g);
    }
  }

  // -- animation -------------------------------------------------------------

  _tick(deltaMS) {
    const step = deltaMS / 90;
    const dt = deltaMS / 1000;
    const cp = this.cellPx;
    const px = (cx) => this.offX + (cx + 0.5) * cp;
    const py = (cy) => this.offY + (cy + 0.5) * cp;

    this.trailLayer.clear();
    this.fxLayer.clear();

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

      // walking bob while between cells; standing still otherwise
      const moving = rec.t < 1 && (rec.fromX !== rec.toX || rec.fromY !== rec.toY);
      const bob = moving ? Math.abs(Math.sin(this._pulse * 14 + rec._px)) * cp * 0.08 : 0;
      rec.glow.x = rec._px;
      rec.glow.y = rec._py + cp * 0.2;
      rec.figure.x = rec._px;
      rec.figure.y = rec._py + cp * 0.35 - bob;

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
            .stroke({ width: 1, color: rec.tint, alpha: (i / pts.length) * 0.22 });
        }
      }

      // attack stays dramatic beyond the emote: sharp expanding burst
      if (rec.act === ACT.ATTACK) {
        const t = rec.actAge % 0.6;
        const rr = cp * (0.4 + t * 2.2);
        const col = ACT_COLORS[ACT.ATTACK];
        this.fxLayer
          .circle(rec._px, rec._py, rr)
          .stroke({ width: 2, color: col, alpha: Math.max(0, 0.85 - t * 1.4) });
      }
    }

    // cooperation: link nearby cooperating agents + halo
    for (let i = 0; i < coop.length; i++) {
      const a = coop[i];
      this.fxLayer
        .circle(a._px, a._py, cp * 0.7)
        .stroke({ width: 1, color: ACT_COLORS[ACT.COOPERATE], alpha: 0.5 });
      for (let j = i + 1; j < coop.length; j++) {
        const b = coop[j];
        const dx = a.toX - b.toX;
        const dy = a.toY - b.toY;
        if (dx * dx + dy * dy <= 16) {
          this.fxLayer
            .moveTo(a._px, a._py)
            .lineTo(b._px, b._py)
            .stroke({ width: 1, color: ACT_COLORS[ACT.COOPERATE], alpha: 0.45 });
        }
      }
    }

    // event pulse
    this._pulse += dt;
    const a = 0.45 + 0.35 * Math.sin(this._pulse * 4);
    for (const g of this.eventLayer.children) g.alpha = a;

    if (deltaMS > 0) this._fps += (1000 / deltaMS - this._fps) * 0.08;
    this._hudAccum += deltaMS;
    if (this._hudAccum >= 300) {
      this._hudAccum = 0;
      this.onHud?.({ fps: Math.round(this._fps), agents: this.agents.size });
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

function brighten(int) {
  const r = (int >> 16) & 255;
  const g = (int >> 8) & 255;
  const b = int & 255;
  return (clamp(r * 0.5 + 128) << 16) | (clamp(g * 0.5 + 128) << 8) | clamp(b * 0.5 + 130);
}

function clamp(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
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
