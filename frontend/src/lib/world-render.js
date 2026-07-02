// PixiJS (WebGL) telemetry renderer for the live simulation world.
//
// Design language: an observation instrument. The terrain is a quiet, cool,
// desaturated field so the *life* reads as signal — agents are luminous blips
// with additive glow and a short motion trail; disturbances are amber reticles.
// A faint coordinate grid with axis ticks frames the field like a sensor readout.
//
// Layers, back to front:
//   terrain → grid+ticks → trails → glow → cores → events
//
// Frames arrive ~20 Hz; the ticker runs at display rate and eases agent motion
// between frames so movement stays fluid.

import { Application, Container, Graphics, Sprite, Text, Texture } from "pixi.js";

const STAGE_SCALE = [0.7, 1.0, 1.22]; // child, adult, elder (core dot size)
const TRAIL_LEN = 8;
const EVENT_COLORS = {
  drought: 0xf0b030,
  storm: 0x4cc6ff,
  fire: 0xff7a3c,
  blight: 0xc774f0,
};

const COLORS = {
  gridLine: 0x16202e,
  gridTick: 0x3a536f,
  tickText: 0x6c8bb0,
  eventDefault: 0xf0b030,
};

export class WorldScene {
  constructor() {
    this.app = new Application();
    this.terrainBase = {}; // biome idx -> quieted telemetry rgb int
    this.waterIdx = -1;
    this.grid = null; // {w, h}
    this.cellPx = 12;
    this.offX = 0;
    this.offY = 0;

    this.terrainLayer = new Container();
    this.gridLayer = new Container();
    this.trailLayer = new Graphics();
    this.glowLayer = new Container();
    this.coreLayer = new Container();
    this.eventLayer = new Container();

    this.terrainSprites = [];
    this.agents = new Map(); // id -> {glow, core, fromX, fromY, toX, toY, t, trail:[]}

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
      this.gridLayer,
      this.trailLayer,
      this.glowLayer,
      this.coreLayer,
      this.eventLayer,
    );
    this._glowTex = makeRadialTexture(64, [
      [0, "rgba(255,255,255,1)"],
      [0.4, "rgba(255,255,255,0.45)"],
      [1, "rgba(255,255,255,0)"],
    ]);
    this._dotTex = makeRadialTexture(32, [
      [0, "rgba(255,255,255,1)"],
      [0.55, "rgba(255,255,255,1)"],
      [1, "rgba(255,255,255,0)"],
    ]);
    this.app.ticker.add((t) => this._tick(t.deltaMS));
    this.app.renderer.on("resize", () => this._rebuildGrid());
  }

  setLegend(biomes) {
    this.terrainBase = {};
    for (const b of biomes) {
      this.terrainBase[b.idx] = quietTerrain(b.rgb[0], b.rgb[1], b.rgb[2]);
      if (b.name === "water") this.waterIdx = b.idx;
    }
  }

  update(frame) {
    const g = frame.grid;
    const resized = !this.grid || this.grid.w !== g.w || this.grid.h !== g.h;
    this.grid = g;
    if (resized) this._rebuildGrid();
    this._paintTerrain(frame.cells);
    this._syncAgents(frame.agents);
    this._paintEvents(frame.events);
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

    this._ensurePool(this.terrainLayer, this.terrainSprites, w * h);
    for (let i = 0; i < w * h; i++) {
      const s = this.terrainSprites[i];
      s.x = this.offX + (i % w) * this.cellPx;
      s.y = this.offY + Math.floor(i / w) * this.cellPx;
      s.width = this.cellPx;
      s.height = this.cellPx;
    }
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
    g.stroke({ width: 1, color: COLORS.gridLine, alpha: 0.6 });
    // brighter frame
    g.rect(x0, y0, w * cp, h * cp).stroke({
      width: 1,
      color: COLORS.gridTick,
      alpha: 0.55,
    });
    this.gridLayer.addChild(g);

    // axis ticks (drawn in-canvas so they align exactly with cells)
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
    for (let i = 0; i < biome.length; i++) {
      let c = this.terrainBase[biome[i]] ?? 0x141a22;
      let r = (c >> 16) & 255;
      let g = (c >> 8) & 255;
      let b = c & 255;
      // water (hydration) deepens to ink; food lifts a faint phosphor green.
      const wn = Math.min(1, water[i] / 100);
      const fn = Math.min(1, food[i] / 110);
      b = Math.min(255, b + wn * 26);
      g = Math.min(255, g + fn * 26);
      r = Math.min(255, r + fn * 4);
      this.terrainSprites[i].tint = (r << 16) | (g << 8) | b;
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
        const core = new Sprite(this._dotTex);
        glow.anchor.set(0.5);
        core.anchor.set(0.5);
        this.glowLayer.addChild(glow);
        this.coreLayer.addChild(core);
        rec = {
          glow,
          core,
          fromX: a.x,
          fromY: a.y,
          toX: a.x,
          toY: a.y,
          t: 1,
          trail: [],
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
      }
      const tint = blip(a.col);
      rec.tint = tint;
      rec.glow.tint = tint;
      rec.core.tint = brighten(tint);
      const cp = this.cellPx;
      rec.glow.scale.set((cp * 2.4) / 64);
      rec.glow.alpha = 0.55;
      rec.core.scale.set(((STAGE_SCALE[a.st] ?? 1.0) * (cp * 0.85)) / 32);
    }
    for (const [id, rec] of this.agents) {
      if (!seen.has(id)) {
        rec.glow.destroy();
        rec.core.destroy();
        this.agents.delete(id);
      }
    }
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
      // crosshair ticks
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
    const cp = this.cellPx;
    const px = (cx) => this.offX + (cx + 0.5) * cp;
    const py = (cy) => this.offY + (cy + 0.5) * cp;

    this.trailLayer.clear();
    for (const rec of this.agents.values()) {
      if (rec.t < 1) rec.t = Math.min(1, rec.t + step);
      const x = rec.fromX + (rec.toX - rec.fromX) * rec.t;
      const y = rec.fromY + (rec.toY - rec.fromY) * rec.t;
      rec.glow.x = rec.core.x = px(x);
      rec.glow.y = rec.core.y = py(y);

      // motion trail: fading polyline through recent cells into the live head
      const pts = rec.trail;
      if (pts.length > 1) {
        for (let i = 1; i < pts.length; i++) {
          this.trailLayer
            .moveTo(px(pts[i - 1][0]), py(pts[i - 1][1]))
            .lineTo(px(pts[i][0]), py(pts[i][1]))
            .stroke({ width: 1, color: rec.tint, alpha: (i / pts.length) * 0.35 });
        }
        const last = pts[pts.length - 1];
        this.trailLayer
          .moveTo(px(last[0]), py(last[1]))
          .lineTo(px(x), py(y))
          .stroke({ width: 1, color: rec.tint, alpha: 0.4 });
      }
    }

    // event pulse
    this._pulse += deltaMS / 1000;
    const a = 0.45 + 0.35 * Math.sin(this._pulse * 4);
    for (const g of this.eventLayer.children) g.alpha = a;

    // smoothed fps for the HUD readout
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

// Quiet, cool, deep, desaturated terrain so the life on top reads as signal.
function quietTerrain(r, g, b) {
  const L = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255; // 0..1
  const keep = 0.26; // retain only a little of the original chroma
  const base = (ch) => L * 255 * (1 - keep) + ch * keep;
  const rr = base(r) * 0.3 + 5;
  const gg = base(g) * 0.3 + 9;
  const bb = base(b) * 0.34 + 16; // cool floor → everything reads slate-blue
  return (clamp(rr) << 16) | (clamp(gg) << 8) | clamp(bb);
}

// Lift an agent's tribe colour into a luminous cyan-biased signal while keeping
// its identity (a warm-gened agent stays warmer than a cool one).
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
