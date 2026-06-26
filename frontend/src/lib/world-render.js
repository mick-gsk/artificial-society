// PixiJS (WebGL) scene for the live simulation world.
//
// Layers, back to front:
//   1. biome   — one tinted cell sprite per grid cell (static; rebuilt on resize)
//   2. overlay — per-cell food (green) / water (blue) heat, retinted every frame
//   3. agents  — sprite pool diffed by id; positions eased between frames
//   4. events  — pulsing rings for active disturbances
//
// Frames arrive ~20 Hz; the ticker runs at display rate and interpolates agent
// motion so movement stays fluid between frames.

import {
  Application,
  Container,
  Graphics,
  Sprite,
  Texture,
} from "pixi.js";

const STAGE_SCALE = [0.55, 0.85, 1.05]; // child, adult, elder
const EVENT_COLORS = {
  drought: 0xd9a441,
  storm: 0x5b9dff,
  fire: 0xe0552b,
  blight: 0x9b59b6,
};

export class WorldScene {
  constructor() {
    this.app = new Application();
    this.legend = {}; // biome idx -> packed rgb int
    this.grid = null; // {w, h}
    this.cellPx = 12;
    this.offX = 0;
    this.offY = 0;

    this.biomeLayer = new Container();
    this.overlayLayer = new Container();
    this.agentLayer = new Container();
    this.eventLayer = new Container();

    this.biomeSprites = [];
    this.overlaySprites = [];
    this.agentSprites = new Map(); // id -> {sprite, fromX, fromY, toX, toY, t}

    this._circleTex = null;
    this._pulse = 0;
  }

  async init(host) {
    await this.app.init({
      background: 0x0b0d12,
      antialias: true,
      resizeTo: host,
    });
    host.appendChild(this.app.canvas);
    this.app.stage.addChild(
      this.biomeLayer,
      this.overlayLayer,
      this.agentLayer,
      this.eventLayer,
    );

    const g = new Graphics().circle(0, 0, 16).fill(0xffffff);
    this._circleTex = this.app.renderer.generateTexture(g);
    g.destroy();

    this.app.ticker.add((ticker) => this._tick(ticker.deltaMS));
    this.app.renderer.on("resize", () => this._rebuildGrid());
  }

  setLegend(biomes) {
    this.legend = {};
    for (const b of biomes) {
      this.legend[b.idx] = (b.rgb[0] << 16) | (b.rgb[1] << 8) | b.rgb[2];
    }
  }

  update(frame) {
    const g = frame.grid;
    const resized = !this.grid || this.grid.w !== g.w || this.grid.h !== g.h;
    this.grid = g;
    if (resized) this._rebuildGrid();
    this._paintBiome(frame.cells.biome);
    this._paintOverlay(frame.cells.food, frame.cells.water);
    this._syncAgents(frame.agents);
    this._paintEvents(frame.events);
  }

  // -- layer construction ----------------------------------------------------

  _rebuildGrid() {
    if (!this.grid) return;
    const { w, h } = this.grid;
    const W = this.app.renderer.width;
    const H = this.app.renderer.height;
    this.cellPx = Math.max(2, Math.floor(Math.min(W / w, H / h)));
    this.offX = Math.floor((W - this.cellPx * w) / 2);
    this.offY = Math.floor((H - this.cellPx * h) / 2);

    this._ensurePool(this.biomeLayer, this.biomeSprites, w * h);
    this._ensurePool(this.overlayLayer, this.overlaySprites, w * h);

    for (let i = 0; i < w * h; i++) {
      const cx = this.offX + (i % w) * this.cellPx;
      const cy = this.offY + Math.floor(i / w) * this.cellPx;
      for (const arr of [this.biomeSprites, this.overlaySprites]) {
        const s = arr[i];
        s.x = cx;
        s.y = cy;
        s.width = this.cellPx;
        s.height = this.cellPx;
      }
    }
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

  // -- per-frame painting ----------------------------------------------------

  _paintBiome(biome) {
    for (let i = 0; i < biome.length; i++) {
      this.biomeSprites[i].tint = this.legend[biome[i]] ?? 0x444444;
    }
  }

  _paintOverlay(food, water) {
    for (let i = 0; i < food.length; i++) {
      const fn = Math.min(1, food[i] / 100);
      const wn = Math.min(1, water[i] / 100);
      const s = this.overlaySprites[i];
      const gr = (60 + 160 * fn) | 0;
      const bl = (60 + 160 * wn) | 0;
      s.tint = (20 << 16) | (gr << 8) | bl;
      s.alpha = 0.1 + 0.4 * Math.max(fn, wn);
    }
  }

  _syncAgents(agents) {
    const seen = new Set();
    for (const a of agents) {
      seen.add(a.id);
      let rec = this.agentSprites.get(a.id);
      if (!rec) {
        const sprite = new Sprite(this._circleTex);
        sprite.anchor.set(0.5);
        this.agentLayer.addChild(sprite);
        rec = { sprite, fromX: a.x, fromY: a.y, toX: a.x, toY: a.y, t: 1 };
        this.agentSprites.set(a.id, rec);
      } else {
        rec.fromX = rec.toX;
        rec.fromY = rec.toY;
        rec.toX = a.x;
        rec.toY = a.y;
        rec.t = 0;
      }
      rec.sprite.tint = parseInt(a.col.slice(1), 16);
      rec.sprite.scale.set(((STAGE_SCALE[a.st] ?? 0.85) * this.cellPx) / 32);
    }
    for (const [id, rec] of this.agentSprites) {
      if (!seen.has(id)) {
        rec.sprite.destroy();
        this.agentSprites.delete(id);
      }
    }
  }

  _paintEvents(events) {
    for (const c of this.eventLayer.removeChildren()) c.destroy();
    for (const e of events) {
      const cx = this.offX + (e.x + 0.5) * this.cellPx;
      const cy = this.offY + (e.y + 0.5) * this.cellPx;
      const ring = new Graphics()
        .circle(cx, cy, Math.max(1, e.r) * this.cellPx)
        .stroke({ width: 2, color: EVENT_COLORS[e.kind] ?? 0xffffff });
      this.eventLayer.addChild(ring);
    }
  }

  // -- animation -------------------------------------------------------------

  _tick(deltaMS) {
    const step = deltaMS / 90; // ease a frame's worth of motion over ~90ms
    const cp = this.cellPx;
    for (const rec of this.agentSprites.values()) {
      if (rec.t < 1) rec.t = Math.min(1, rec.t + step);
      const x = rec.fromX + (rec.toX - rec.fromX) * rec.t;
      const y = rec.fromY + (rec.toY - rec.fromY) * rec.t;
      rec.sprite.x = this.offX + (x + 0.5) * cp;
      rec.sprite.y = this.offY + (y + 0.5) * cp;
    }
    this._pulse += deltaMS / 1000;
    const a = 0.35 + 0.3 * Math.sin(this._pulse * 4);
    for (const ring of this.eventLayer.children) ring.alpha = a;
  }

  destroy() {
    if (this.app) this.app.destroy(true, { children: true });
  }
}
