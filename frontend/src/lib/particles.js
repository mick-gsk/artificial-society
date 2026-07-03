// Pooled, deterministic particle system for the live world view.
//
// Weather and dramatic moments become visible matter: rain falls (and shears)
// inside storm radii, fires throw rising flames + drift smoke + a night glow,
// droughts kick up warm dust, blight breathes violet spores, births burst gold
// sparks + a ring, deaths exhale a grey wisp, discoveries spit embers.
//
// Design constraints (see the R5 brief):
//   * ONE ParticleContainer, ONE small programmatic atlas → one draw call.
//   * A fixed pre-allocated pool + freelist. Hard global cap of 800 active
//     particles; every emitter enforces its own sub-budget against that cap.
//   * NO Math.random() in the render path — a seeded mulberry32 PRNG per emitter
//     keeps the look stable for the same input.
//   * No per-frame allocations in the steady state: emit reuses pooled objects,
//     update mutates in place, dead particles go back on the freelist.
//
// Coordinates are WORLD px (the same space world-render feeds via px()/py()),
// so the container lives under worldRoot and pans/zooms with everything else.

import { Particle, ParticleContainer, Rectangle, Texture } from "pixi.js";

// -- tunables ----------------------------------------------------------------

export const HARD_CAP = 800; // absolute ceiling on active particles
const BUDGET = {
  rain: 300, // storm rain, shared across all storm discs
  fire: 200, // flames, shared across all fire sources
  smoke: 120, // smoke, shared across fires + camp chimneys
  dust: 90, // drought motes
  spore: 120, // blight spores
  fx: 60, // births / deaths / discovery sparks (short one-shots)
};

// Particle "kinds" — index into the atlas frame table + drive the update math.
const K_RAIN = 0;
const K_FIRE = 1;
const K_SMOKE = 2;
const K_DUST = 3;
const K_SPORE = 4;
const K_SPARK = 5;
const K_WISP = 6;
const K_RING = 7;

// Which sub-budget a kind counts against.
const KIND_BUDGET = {
  [K_RAIN]: "rain",
  [K_FIRE]: "fire",
  [K_SMOKE]: "smoke",
  [K_DUST]: "dust",
  [K_SPORE]: "spore",
  [K_SPARK]: "fx",
  [K_WISP]: "smoke", // death wisps draw from the smoke budget
  [K_RING]: "fx",
};

// -- deterministic PRNG (mulberry32) -----------------------------------------
//
// Tiny, fast, seedable. We keep one generator and RE-SEED it per emitter call
// from the emitter's spatial/id key, so identical inputs give an identical
// spray every time (no Math.random anywhere in the render path).
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Mix a few numbers into a 32-bit seed (order-independent enough for our use).
function seedOf(a, b = 0, c = 0) {
  let h = 2166136261 >>> 0;
  h = Math.imul(h ^ (a | 0), 16777619);
  h = Math.imul(h ^ (b | 0), 16777619);
  h = Math.imul(h ^ (c | 0), 16777619);
  return h >>> 0;
}

// -- atlas -------------------------------------------------------------------
//
// One canvas holds every particle shape; each is sliced into a sub-Texture that
// shares the single GPU source, so the whole system batches into one draw call.
// Shapes are white/soft so a per-particle tint paints the colour. Built once.
function buildParticleAtlas() {
  const S = 32; // cell size in atlas px
  const cols = 8;
  const c = document.createElement("canvas");
  c.width = cols * S;
  c.height = S;
  const ctx = c.getContext("2d");

  const cx = (i) => i * S + S / 2;
  const soft = (i, r, inner = "rgba(255,255,255,1)", outer = "rgba(255,255,255,0)") => {
    const g = ctx.createRadialGradient(cx(i), S / 2, 0, cx(i), S / 2, r);
    g.addColorStop(0, inner);
    g.addColorStop(1, outer);
    ctx.fillStyle = g;
    ctx.fillRect(i * S, 0, S, S);
  };

  // 0 rain streak — a soft vertical white-blue smear (tint recolours it)
  {
    const g = ctx.createLinearGradient(0, 2, 0, S - 2);
    g.addColorStop(0, "rgba(255,255,255,0)");
    g.addColorStop(0.5, "rgba(230,244,255,0.95)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(K_RAIN * S + S / 2 - 2, 2, 4, S - 4);
  }
  // 1 flame — hot soft blob, slightly taller than wide
  {
    const g = ctx.createRadialGradient(cx(K_FIRE), S * 0.6, 0, cx(K_FIRE), S * 0.6, S * 0.5);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.35, "rgba(255,240,200,0.95)");
    g.addColorStop(1, "rgba(255,200,140,0)");
    ctx.fillStyle = g;
    ctx.fillRect(K_FIRE * S, 0, S, S);
  }
  // 2 smoke puff — soft grey cloud
  soft(K_SMOKE, S * 0.5, "rgba(255,255,255,0.9)", "rgba(255,255,255,0)");
  // 3 dust mote — small soft dot
  soft(K_DUST, S * 0.32, "rgba(255,255,255,0.95)", "rgba(255,255,255,0)");
  // 4 spore — small soft dot (tinted violet at emit)
  soft(K_SPORE, S * 0.3, "rgba(255,255,255,1)", "rgba(255,255,255,0)");
  // 5 spark — tiny bright point with a short glow
  soft(K_SPARK, S * 0.22, "rgba(255,255,255,1)", "rgba(255,255,255,0)");
  // 6 wisp — same soft cloud as smoke (grey death breath)
  soft(K_WISP, S * 0.5, "rgba(255,255,255,0.85)", "rgba(255,255,255,0)");
  // 7 ring — soft thin ring for the birth flash
  {
    const i = K_RING;
    const grd = ctx.createRadialGradient(cx(i), S / 2, S * 0.18, cx(i), S / 2, S * 0.5);
    grd.addColorStop(0, "rgba(255,255,255,0)");
    grd.addColorStop(0.6, "rgba(255,255,255,0.9)");
    grd.addColorStop(0.85, "rgba(255,255,255,0.5)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = grd;
    ctx.fillRect(i * S, 0, S, S);
  }

  const source = Texture.from(c).source;
  source.scaleMode = "linear";
  const frames = [];
  for (let i = 0; i < cols; i++) {
    frames[i] = new Texture({ source, frame: new Rectangle(i * S, 0, S, S) });
  }
  return { source, frames, S };
}

// Precompute BGR tints once (the Particle renderer reads a packed ABGR `color`;
// we rebuild only the alpha byte per frame, never re-parsing the colour). These
// mirror Particle's internal tint→BGR conversion for a handful of fixed hues.
function rgb2bgr(hex) {
  const r = (hex >> 16) & 255;
  const g = (hex >> 8) & 255;
  const b = hex & 255;
  return (b << 16) | (g << 8) | r;
}

// -- system ------------------------------------------------------------------

export function createParticleSystem({ container }) {
  const atlas = buildParticleAtlas();

  // One dynamic ParticleContainer: position, rotation, colour (alpha lives in
  // the packed colour) and uvs (per-particle atlas frame) all change per frame,
  // so all four are dynamic. Still ONE draw call — one shared texture source.
  const pc = new ParticleContainer({
    dynamicProperties: { position: true, rotation: true, color: true, uvs: true },
    roundPixels: false,
  });
  // ParticleContainer bounds aren't auto-computed; give it a huge static area so
  // culling never hides the spray. (Cheap — one rect.)
  pc.boundsArea = new Rectangle(-1e5, -1e5, 2e5, 2e5);
  container.addChild(pc);

  // The pool. Each slot is a Particle plus the sim fields we animate. Particles
  // stay in the container for their whole life; inactive ones are parked
  // off-screen at alpha 0 (cheaper than add/remove churn, and keeps the
  // freelist O(1)). We only ever hold HARD_CAP of them.
  const pool = new Array(HARD_CAP);
  const free = new Array(HARD_CAP); // freelist of pool indices
  const sim = new Array(HARD_CAP); // parallel per-particle sim state
  for (let i = 0; i < HARD_CAP; i++) {
    const p = new Particle({ texture: atlas.frames[K_SPARK], anchorX: 0.5, anchorY: 0.5 });
    p.alpha = 0;
    pool[i] = p;
    pc.addParticle(p);
    free[i] = i;
    sim[i] = {
      _alive: false, // freelist/budget truth — release is idempotent on this
      kind: 0,
      vx: 0,
      vy: 0,
      life: 0,
      ttl: 0, // 0 = parked/free — the integrator skips it (only >0 is alive)
      s0: 1,
      s1: 1,
      a0: 1,
      a1: 0,
      tintBGR: 0xffffff,
      wind: 0, // per-particle wind susceptibility (0 = ignores wind)
      spin: 0, // radians/sec
      phase: 0, // for sinusoidal drift (spores)
      freq: 0,
      amp: 0,
    };
  }
  let freeTop = HARD_CAP; // free[0..freeTop-1] are available indices
  const budgetUsed = { rain: 0, fire: 0, smoke: 0, dust: 0, spore: 0, fx: 0 };

  // Global rolling wind vector (see update()). Emitters read windX at spawn for
  // an initial bias; the per-frame integrator applies the live wind too. Wind is
  // treated as horizontal (the world's gusts are lateral) — only windX is used.
  let windX = 0;

  // Monotonic frame counter, bumped once per update(). Emitters fold it into
  // their emit-gate seed so a throttle ("~40% of ticks") actually varies OVER
  // TIME instead of being frozen by a position-only seed — while staying fully
  // deterministic for a given frame sequence (no Math.random).
  let frameSeed = 0;

  // Acquire a pooled slot for `kind` if its budget allows, else -1. The caller
  // then fills the sim fields + the Particle's spawn transform.
  function acquire(kind) {
    if (freeTop === 0) return -1;
    const bkey = KIND_BUDGET[kind];
    if (budgetUsed[bkey] >= BUDGET[bkey]) return -1;
    const idx = free[--freeTop];
    budgetUsed[bkey]++;
    return idx;
  }

  // Return a slot to the freelist. Idempotent: guarded by `_alive` so a double
  // release (from any path) can never corrupt the freelist / budget counters.
  function release(idx) {
    const s = sim[idx];
    if (!s._alive) return; // already parked — no double-free
    s._alive = false;
    s.ttl = 0;
    budgetUsed[KIND_BUDGET[s.kind]]--;
    const p = pool[idx];
    p.alpha = 0;
    p.x = -1e6; // park far off-screen
    free[freeTop++] = idx;
  }

  // Common spawn: set the Particle's frame + transform + initial colour.
  function spawn(idx, kind, x, y, opts) {
    const p = pool[idx];
    const s = sim[idx];
    p.texture = atlas.frames[kind];
    p.x = x;
    p.y = y;
    p.rotation = opts.rot ?? 0;
    s._alive = true;
    s.kind = kind;
    s.vx = opts.vx ?? 0;
    s.vy = opts.vy ?? 0;
    s.life = 0;
    s.ttl = opts.ttl;
    s.s0 = opts.s0;
    s.s1 = opts.s1;
    s.a0 = opts.a0;
    s.a1 = opts.a1 ?? 0;
    s.tintBGR = opts.tintBGR;
    s.wind = opts.wind ?? 0;
    s.spin = opts.spin ?? 0;
    s.phase = opts.phase ?? 0;
    s.freq = opts.freq ?? 0;
    s.amp = opts.amp ?? 0;
    p.scaleX = opts.s0;
    p.scaleY = opts.sy0 ?? opts.s0;
    s._sy0 = opts.sy0 ?? opts.s0;
    s._sy1 = opts.sy1 ?? opts.s1;
    // pack colour = BGR tint + (alpha byte << 24)
    p.color = s.tintBGR + (((opts.a0 * 255) | 0) << 24);
    return p;
  }

  // -- precomputed tints -----------------------------------------------------
  const T_RAIN = rgb2bgr(0xbfe0ff);
  const T_FLAME_HOT = rgb2bgr(0xffd27a);
  const T_FLAME_COOL = rgb2bgr(0x9a2a12); // dark red at the end of a flame's life
  const T_SMOKE = rgb2bgr(0x8a8f96);
  const T_DUST = rgb2bgr(0xd8b878);
  const T_SPORE = rgb2bgr(0xc774f0);
  const T_GOLD = rgb2bgr(0xffd166);
  const T_WISP = rgb2bgr(0x9aa0a8);
  const T_EMBER = rgb2bgr(0xff8a3c);

  // -- emitters --------------------------------------------------------------
  //
  // Emitters are DATA-DRIVEN: world-render calls them each tick with the current
  // sources (events, fire cells, camps, new/dead agents). Budgets are enforced
  // in acquire(); a re-seeded PRNG makes each call's spray deterministic.

  // Rain — only within a storm disc. Count ∝ r²·intensity, capped by the global
  // rain budget. Drops respawn at the TOP of the disc (see the integrator) and
  // shear horizontally with the wind. `event` = {x,y,r,i} in CELL space; cellPx
  // + off convert to world px. Called once per storm per tick.
  function rain(ex, ey, r, intensity, cellPx) {
    const rand = mulberry32(seedOf((ex * 131.1) | 0, (ey * 57.3) | 0, 909 + frameSeed));
    // how many drops to TRY to keep alive this tick (emit a fraction each tick
    // so the disc fills over ~10 ticks and then holds steady at the budget)
    const target = Math.min(BUDGET.rain, Math.ceil(r * r * intensity * 0.9));
    const perTick = Math.max(1, Math.ceil(target / 8));
    const radPx = r * cellPx;
    for (let k = 0; k < perTick; k++) {
      const idx = acquire(K_RAIN);
      if (idx < 0) return;
      const ang = rand() * Math.PI * 2;
      const rr = Math.sqrt(rand()) * radPx;
      const dx = Math.cos(ang) * rr;
      // spawn somewhere in the upper band of the disc so the fall reads
      const sy = ey - radPx + rand() * radPx * 0.5;
      const speed = cellPx * (7 + rand() * 4);
      spawn(idx, K_RAIN, ex + dx, sy, {
        vx: windX * 3,
        vy: speed,
        ttl: 0.6 + rand() * 0.4,
        s0: cellPx / 20,
        s1: cellPx / 20,
        sy0: (cellPx * 1.1) / 20,
        sy1: (cellPx * 1.1) / 20,
        a0: 0.35 + rand() * 0.25,
        a1: 0.0,
        tintBGR: T_RAIN,
        wind: 1,
        rot: 0, // streak texture is already vertical; wind shear tilts it live
      });
      // stash disc bounds so the integrator can respawn a drop at the top when
      // it falls past the bottom of the disc (steady rainfall, not a one-shot)
      const s = sim[idx];
      s._cx = ex;
      s._top = ey - radPx;
      s._bot = ey + radPx * 0.4;
      s._radPx = radPx;
      s._speed = speed;
    }
  }

  // Fire — 6–10 flames per source, rising, shrinking, orange→dark-red, additive.
  // (x,y) world px of the fire foot. Called once per fire source per tick; the
  // per-tick emission is throttled so a source holds ~a dozen live flames.
  function fire(x, y, cellPx, seed = 0) {
    const rand = mulberry32(seedOf((x * 12.9) | 0, (y * 78.2) | 0, 313 + seed + frameSeed));
    const perTick = 2; // ~2 new flames/tick → a steady 6–10 alive
    for (let k = 0; k < perTick; k++) {
      const idx = acquire(K_FIRE);
      if (idx < 0) return;
      const jx = (rand() - 0.5) * cellPx * 0.5;
      spawn(idx, K_FIRE, x + jx, y - cellPx * 0.1, {
        vx: (rand() - 0.5) * cellPx * 0.6 + windX * 0.4,
        vy: -cellPx * (1.6 + rand() * 1.2), // rise
        ttl: 0.5 + rand() * 0.4,
        s0: (cellPx * (0.5 + rand() * 0.3)) / 32,
        s1: (cellPx * 0.12) / 32, // shrink as it rises
        a0: 0.9,
        a1: 0.0,
        tintBGR: T_FLAME_HOT,
        wind: 0.3,
      });
      sim[idx]._coolTo = T_FLAME_COOL; // flames lerp hue toward dark red
    }
  }

  // Smoke — 3–5 per source, slow rise, expanding, grey fade, wind drift. Used
  // over fires AND camp chimneys. Throttled per tick so a source holds a few.
  function smoke(x, y, cellPx, seed = 0) {
    const rand = mulberry32(seedOf((x * 41.7) | 0, (y * 23.4) | 0, 727 + seed + frameSeed));
    if (rand() > 0.6) return; // emit on ~40% of ticks → gentle stream
    const idx = acquire(K_SMOKE);
    if (idx < 0) return;
    spawn(idx, K_SMOKE, x + (rand() - 0.5) * cellPx * 0.3, y - cellPx * 0.3, {
      vx: windX * 1.4 + (rand() - 0.5) * cellPx * 0.2,
      vy: -cellPx * (0.5 + rand() * 0.4),
      ttl: 1.8 + rand() * 1.2,
      s0: (cellPx * 0.4) / 32,
      s1: (cellPx * 1.3) / 32, // expand
      a0: 0.28 + rand() * 0.12,
      a1: 0.0,
      tintBGR: T_SMOKE,
      wind: 1.2,
    });
  }

  // Drought dust — 4–6 sluggish horizontal motes per drought event, warm tint.
  // event centre + radius in world px. Throttled so a drought holds a light haze.
  function dust(ex, ey, radPx, cellPx) {
    const rand = mulberry32(seedOf((ex * 7.1) | 0, (ey * 9.3) | 0, 505 + frameSeed));
    if (rand() > 0.5) return;
    const n = 2;
    for (let k = 0; k < n; k++) {
      const idx = acquire(K_DUST);
      if (idx < 0) return;
      const ang = rand() * Math.PI * 2;
      const rr = Math.sqrt(rand()) * radPx;
      const dir = windX >= 0 ? 1 : -1;
      spawn(idx, K_DUST, ex + Math.cos(ang) * rr, ey + Math.sin(ang) * rr * 0.6, {
        vx: dir * cellPx * (0.4 + rand() * 0.5) + windX * 0.8,
        vy: (rand() - 0.5) * cellPx * 0.15,
        ttl: 1.6 + rand() * 1.4,
        s0: (cellPx * (0.25 + rand() * 0.25)) / 32,
        s1: (cellPx * 0.1) / 32,
        a0: 0.22 + rand() * 0.12,
        a1: 0.0,
        tintBGR: T_DUST,
        wind: 0.6,
      });
    }
  }

  // Blight spores — 10–20 per blight event, sinusoidal drift, violet, additive.
  // Emits a fraction each tick so a blight disc keeps a slow violet shimmer.
  function spores(ex, ey, radPx, cellPx) {
    const rand = mulberry32(seedOf((ex * 3.7) | 0, (ey * 5.9) | 0, 191 + frameSeed));
    const n = 3;
    for (let k = 0; k < n; k++) {
      const idx = acquire(K_SPORE);
      if (idx < 0) return;
      const ang = rand() * Math.PI * 2;
      const rr = Math.sqrt(rand()) * radPx;
      spawn(idx, K_SPORE, ex + Math.cos(ang) * rr, ey + Math.sin(ang) * rr, {
        vx: windX * 0.4,
        vy: -cellPx * (0.2 + rand() * 0.3), // slow lift
        ttl: 2.0 + rand() * 1.5,
        s0: (cellPx * (0.2 + rand() * 0.2)) / 32,
        s1: (cellPx * 0.05) / 32,
        a0: 0.5 + rand() * 0.3,
        a1: 0.0,
        tintBGR: T_SPORE,
        wind: 0.5,
        phase: rand() * Math.PI * 2,
        freq: 2 + rand() * 2,
        amp: cellPx * (0.4 + rand() * 0.4),
      });
    }
  }

  // Birth — one-shot: 6 gold sparks fanning out + a soft expanding ring. (x,y)
  // world px. Called ONCE when a genuinely-new agent id appears.
  function birth(x, y, cellPx) {
    const rand = mulberry32(seedOf((x * 19.7) | 0, (y * 27.1) | 0, 61));
    // ring
    const ri = acquire(K_RING);
    if (ri >= 0) {
      spawn(ri, K_RING, x, y, {
        vx: 0,
        vy: 0,
        ttl: 0.6,
        s0: (cellPx * 0.3) / 32,
        s1: (cellPx * 2.4) / 32, // expand
        a0: 0.85,
        a1: 0.0,
        tintBGR: T_GOLD,
      });
    }
    for (let k = 0; k < 6; k++) {
      const idx = acquire(K_SPARK);
      if (idx < 0) break;
      const ang = (k / 6) * Math.PI * 2 + rand() * 0.4;
      const speed = cellPx * (2 + rand() * 2);
      spawn(idx, K_SPARK, x, y, {
        vx: Math.cos(ang) * speed,
        vy: Math.sin(ang) * speed - cellPx * 0.5,
        ttl: 0.5 + rand() * 0.3,
        s0: (cellPx * 0.35) / 32,
        s1: (cellPx * 0.05) / 32,
        a0: 1.0,
        a1: 0.0,
        tintBGR: T_GOLD,
        wind: 0.2,
      });
    }
  }

  // Death — a grey wisp rising from (x,y). One-shot, called once when a record
  // begins its death fade (the figure fade itself lives in world-render).
  function deathWisp(x, y, cellPx) {
    const rand = mulberry32(seedOf((x * 5.3) | 0, (y * 8.1) | 0, 977));
    const n = 3;
    for (let k = 0; k < n; k++) {
      const idx = acquire(K_WISP);
      if (idx < 0) break;
      spawn(idx, K_WISP, x + (rand() - 0.5) * cellPx * 0.4, y - k * cellPx * 0.2, {
        vx: windX * 1.2 + (rand() - 0.5) * cellPx * 0.15,
        vy: -cellPx * (0.6 + rand() * 0.4),
        ttl: 1.2 + rand() * 0.8,
        s0: (cellPx * 0.35) / 32,
        s1: (cellPx * 1.0) / 32,
        a0: 0.4 + rand() * 0.15,
        a1: 0.0,
        tintBGR: T_WISP,
        wind: 1.0,
      });
    }
  }

  // Discovery sparks — 8 embers fanning out, additive. Complements the existing
  // world-render `_bursts` star. Called once per discovery burst.
  function sparks(x, y, cellPx, color = 0xff8a3c) {
    const rand = mulberry32(seedOf((x * 23.3) | 0, (y * 14.7) | 0, 443));
    const tintBGR = rgb2bgr(color) || T_EMBER;
    for (let k = 0; k < 8; k++) {
      const idx = acquire(K_SPARK);
      if (idx < 0) break;
      const ang = (k / 8) * Math.PI * 2 + rand() * 0.5;
      const speed = cellPx * (2.5 + rand() * 2.5);
      spawn(idx, K_SPARK, x, y, {
        vx: Math.cos(ang) * speed,
        vy: Math.sin(ang) * speed - cellPx * 0.8,
        ttl: 0.6 + rand() * 0.4,
        s0: (cellPx * 0.3) / 32,
        s1: (cellPx * 0.04) / 32,
        a0: 1.0,
        a1: 0.0,
        tintBGR,
        wind: 0.3,
      });
    }
  }

  // -- per-frame integrator --------------------------------------------------
  //
  // O(active): walk every pool slot; skip inactive ones cheaply (ttl<=0). Only
  // what's alive costs anything. `deltaMS` is the display-tick delta; `wind` is
  // the global client wind vector {x,y} in world px/s (see world-render).
  function update(deltaMS, wind) {
    const dt = Math.min(0.05, deltaMS / 1000); // clamp huge tab-switch deltas
    windX = wind ? wind.x : 0;
    frameSeed = (frameSeed + 1) | 0; // advance the emit-gate clock (see emitters)
    // Walk the ACTIVE set only: everything not on the freelist. We track that as
    // "any slot whose sim.ttl > 0". A tiny linear scan over HARD_CAP is fine but
    // we keep it O(active) by checking ttl first (parked slots have ttl 0).
    for (let i = 0; i < HARD_CAP; i++) {
      const s = sim[i];
      if (s.ttl <= 0) continue; // parked / never-used slot
      const p = pool[i];
      s.life += dt;
      const t = s.life / s.ttl;
      if (t >= 1) {
        release(i); // clears _alive + ttl, returns the slot to the freelist
        continue;
      }
      // wind acts on susceptible particles; gravity/own velocity already in vy
      const wx = s.wind ? windX * s.wind : 0;
      p.x += (s.vx + wx) * dt;
      p.y += s.vy * dt;

      // spore sinusoidal side-drift on top of base velocity
      if (s.kind === K_SPORE && s.amp) {
        p.x += Math.cos(s.phase + s.life * s.freq) * s.amp * dt;
      }

      // rain that falls past the bottom of its disc respawns at the top —
      // steady rainfall instead of a single fall. Keeps the drop within budget.
      if (s.kind === K_RAIN && p.y > s._bot) {
        p.y = s._top;
        s.life = 0; // reset fade so it doesn't blink out
      }

      // eased scale + alpha over life
      const sc = s.s0 + (s.s1 - s.s0) * t;
      const scy = s._sy0 + (s._sy1 - s._sy0) * t;
      p.scaleX = sc;
      p.scaleY = scy;

      // rain streak shear: tilt the vertical streak toward the wind direction
      if (s.kind === K_RAIN) {
        p.rotation = Math.atan2(wx, s.vy || 1) * 0.9;
      } else if (s.spin) {
        p.rotation += s.spin * dt;
      }

      let alpha = s.a0 + (s.a1 - s.a0) * t;
      // flames additionally cool their hue orange→dark-red across life
      let tintBGR = s.tintBGR;
      if (s.kind === K_FIRE && s._coolTo != null) {
        tintBGR = lerpBGR(s.tintBGR, s._coolTo, t);
      }
      if (alpha < 0) alpha = 0;
      p.color = tintBGR + (((alpha * 255) | 0) << 24);
    }
  }

  function clear() {
    for (let i = 0; i < HARD_CAP; i++) {
      if (sim[i]._alive) release(i);
    }
  }

  function activeCount() {
    return HARD_CAP - freeTop;
  }

  // Debug/verification hook: snapshot up to `n` live particles of a given kind
  // (their pool index + world x/y/rotation). Lets a test prove coherent motion
  // (positions advance by velocity across frames, not re-randomised). Off the
  // render path — never called in normal operation.
  function debugSample(kind, n = 12) {
    const out = [];
    for (let i = 0; i < HARD_CAP && out.length < n; i++) {
      const s = sim[i];
      if (s._alive && s.kind === kind) {
        const p = pool[i];
        out.push({ i, x: p.x, y: p.y, rot: p.rotation });
      }
    }
    return out;
  }

  return {
    container: pc,
    update,
    clear,
    activeCount,
    debugSample, // verification hook (see debugSample above)
    budgetUsed, // read-only debug view
    emitters: { rain, fire, smoke, dust, spores, birth, deathWisp, sparks },
    destroy() {
      pc.destroy();
      atlas.source.destroy(true);
    },
  };
}

// Lerp between two BGR-packed colours by channel (used for the flame cool-down).
function lerpBGR(a, b, t) {
  const ar = a & 255;
  const ag = (a >> 8) & 255;
  const ab = (a >> 16) & 255;
  const br = b & 255;
  const bg = (b >> 8) & 255;
  const bb = (b >> 16) & 255;
  const r = (ar + (br - ar) * t) | 0;
  const g = (ag + (bg - ag) * t) | 0;
  const bl = (ab + (bb - ab) * t) | 0;
  return (bl << 16) | (g << 8) | r;
}
