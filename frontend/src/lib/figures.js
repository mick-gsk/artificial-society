// Programmatic figure atlas for the agents — no external assets.
//
// One canvas holds every pose for every life stage (17 poses × 3 stages = 51
// frames), sliced into Pixi Textures that all share ONE GPU source, so every
// agent sprite batches into a single draw call. Figures are drawn white/gray
// with a dark 1px outline; a per-sprite `tint` paints the tribe colour on top
// (head stays brightest, so identity reads even when tinted). This replaces the
// old 12×15 stand/walk textures with a 20×28 body that has a real, readable
// pose loop PER ACTION plus a hand anchor per frame so a held tool tracks the
// hand as the arm moves.
//
// The whole thing is generated once at init in well under 20 ms.

import { Rectangle, Texture } from "pixi.js";

// -- atlas geometry -----------------------------------------------------------

// Each atlas cell is 24×32; the 20×28 figure sits centred with a 2px margin so
// swung arms and the lunge never clip the cell edge. Origin of a figure = the
// cell's (CX, GROUND) — feet on the ground line, matching the sprite anchor
// (0.5, GROUND/CELL_H) used in world-render.
export const CELL_W = 24;
export const CELL_H = 32;
const CX = 12; // horizontal centre of a cell, in cell px
const GROUND = 29; // feet line (0.90625 of CELL_H → sprite anchor y)
export const ANCHOR_Y = GROUND / CELL_H;

// One column per pose, one row per life stage.
const POSE_COLS = [
  "idle", // 0
  "walk0", "walk1", "walk2", "walk3", // 1..4  contact/pass/contact/pass
  "forage0", "forage1", "forage2", // 5..7  stand → bend → reach down
  "coop0", "coop1", // 8..9  arm out / nod
  "attack0", "attack1", "attack2", // 10..12 wind-up → lunge → recover
  "build0", "build1", "build2", // 13..15 raise → mid → strike down
  "sleep", // 16
];
const COL = Object.fromEntries(POSE_COLS.map((k, i) => [k, i]));
const N_COLS = POSE_COLS.length; // 17
const STAGES = ["child", "adult", "elder"]; // rows 0,1,2

// Which frames belong to each action key, plus loop timing. `n` is derived so
// the animation driver stays a clean table lookup.
const FRAME_KEYS = {
  idle: ["idle"],
  walk: ["walk0", "walk1", "walk2", "walk3"],
  forage: ["forage0", "forage1", "forage2"],
  cooperate: ["coop0", "coop1"],
  attack: ["attack0", "attack1", "attack2"],
  build: ["build0", "build1", "build2"],
  sleep: ["sleep"],
};
const DUR_MS = {
  idle: 0,
  walk: 130,
  forage: 220,
  cooperate: 300,
  attack: 120,
  build: 160,
  sleep: 0,
};

// -- palette (white/gray so the tribe tint reads on top) ----------------------

const OUTLINE = "#12161d"; // stays near-black under any tribe tint (multiply)
const SKIN = "rgba(255,255,255,1)"; // brightest → tint reads lightest on the head
const EYE = "#3a3f48";
const HAIR = "rgba(150,150,150,1)";
const HAIR_ELDER = "rgba(200,200,200,1)"; // gray crown
const TUNIC_HI = "rgba(214,214,214,1)"; // 3-tone tunic: light / mid / dark
const TUNIC_MID = "rgba(188,188,188,1)";
const TUNIC_LO = "rgba(162,162,162,1)";
const LIMB = "rgba(230,230,230,1)"; // arms
const LEG = "rgba(150,150,150,1)";

function px(ctx, x, y, w, h, col) {
  ctx.fillStyle = col;
  ctx.fillRect(x, y, w, h);
}

// A 1-px "limb" (arm) segment from (x0,y0) toward (x1,y1) WITH its 1-px dark
// outline: first a 3×3 OUTLINE block under every segment pixel, then the 1-px
// LIMB fill on top — the same outline-under-fill treatment the legs get.
// Plotted as pixels so arms read crisply at NEAREST. NOTE: the outline extends
// 1 px beyond the tip in every direction; pose tips must stay ≥1 px inside the
// atlas cell or the outline bleeds into the neighbouring frame.
function limb(ctx, x0, y0, x1, y1) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = Math.round(x0 + (x1 - x0) * t);
    const y = Math.round(y0 + (y1 - y0) * t);
    px(ctx, x - 1, y - 1, 3, 3, OUTLINE);
  }
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = Math.round(x0 + (x1 - x0) * t);
    const y = Math.round(y0 + (y1 - y0) * t);
    px(ctx, x, y, 1, 1, LIMB);
  }
}

// -- one figure -------------------------------------------------------------
//
// Draws a figure into the atlas cell whose top-left is (ox, oy). `p` are the
// stage proportions; `pose` selects arms/legs/torso lean. Returns the leading
// hand anchor {x,y,rot} in ATLAS px (converted to figure-local later).

function drawFigure(ctx, ox, oy, pose, p) {
  const { headTop, headH, headW, torsoTop, torsoH, bodyW, lean, stoop } = p;
  // Base column centre; a stoop shifts the upper body forward (+x = facing dir).
  const cx = ox + CX;
  const hx = cx + lean; // head/torso x centre after lean
  const headL = hx - Math.floor(headW / 2);
  const headTopY = oy + headTop + stoop;
  const torsoTopY = oy + torsoTop + stoop;
  const torsoBot = torsoTopY + torsoH;
  const bodyL = hx - Math.floor(bodyW / 2);
  const legY = oy + GROUND - p.legH; // legs stand on the ground line

  // hand anchor accumulator (leading hand). Defaults to a resting hand at the
  // side; poses override it.
  let hand = { x: cx + Math.floor(bodyW / 2), y: torsoTopY + 3, rot: 0.2 };

  const isSleep = pose === "sleep";
  if (isSleep) {
    return drawSleeper(ctx, ox, oy, p);
  }

  // --- legs (pose-specific stance) ---
  const legDefs = pose.legs || [
    [-2, 0],
    [2, 0],
  ];
  for (const [dx, ph] of legDefs) {
    const lx = hx + dx;
    // outline then fill
    px(ctx, lx - 1, legY, 3, p.legH + ph + 1, OUTLINE);
  }
  for (const [dx, ph] of legDefs) {
    const lx = hx + dx;
    px(ctx, lx, legY, 1, p.legH + ph, LEG);
  }

  // --- torso outline + 3-tone tunic ---
  px(ctx, bodyL - 1, torsoTopY - 1, bodyW + 2, torsoH + 2, OUTLINE);
  const third = Math.max(1, Math.round(torsoH / 3));
  px(ctx, bodyL, torsoTopY, bodyW, third, TUNIC_HI);
  px(ctx, bodyL, torsoTopY + third, bodyW, third, TUNIC_MID);
  px(ctx, bodyL, torsoTopY + 2 * third, bodyW, torsoH - 2 * third, TUNIC_LO);

  // --- head: outline, skin, hair, eyes ---
  px(ctx, headL - 1, headTopY - 1, headW + 2, headH + 2, OUTLINE);
  px(ctx, headL, headTopY, headW, headH, SKIN);
  // hair cap across the top
  px(ctx, headL, headTopY, headW, Math.max(1, Math.round(headH * 0.28)), p.hair);
  if (p.elder) {
    // receding gray crown: leave a bare band under the cap
    px(ctx, headL + 1, headTopY + 1, headW - 2, 1, SKIN);
  }
  // eyes (two dark pixels), nudged toward facing (+x)
  const eyeY = headTopY + Math.round(headH * 0.5);
  const eyeXL = headL + Math.max(1, Math.round(headW * 0.28)) + (lean > 0 ? 1 : 0);
  px(ctx, eyeXL, eyeY, 1, 1, EYE);
  if (headW >= 5) px(ctx, eyeXL + 2, eyeY, 1, 1, EYE);

  // --- arms (pose-specific); leading arm sets the hand anchor ---
  const shoulderY = torsoTopY + 1;
  const backX = bodyL; // trailing shoulder
  const frontX = bodyL + bodyW - 1; // leading shoulder (facing +x)
  if (pose.arms) {
    // trailing arm (drawn first, behind)
    if (pose.arms.back) {
      const [ex, ey] = pose.arms.back;
      limb(ctx, backX, shoulderY, backX + ex, shoulderY + ey);
    }
    // leading arm → hand anchor at its tip
    const [ex, ey, rot] = pose.arms.front;
    const tipX = frontX + ex;
    const tipY = shoulderY + ey;
    limb(ctx, frontX, shoulderY, tipX, tipY);
    hand = { x: tipX, y: tipY, rot: rot ?? 0.2 };
  } else {
    // default: both arms hang at the sides
    limb(ctx, backX, shoulderY, backX - 1, shoulderY + 4);
    limb(ctx, frontX, shoulderY, frontX + 1, shoulderY + 4);
    hand = { x: frontX + 1, y: shoulderY + 4, rot: 0.15 };
  }

  return hand;
}

// Lying figure — horizontal, closed silhouette, drawn in 20×28 quality.
function drawSleeper(ctx, ox, oy, p) {
  const w = 18;
  const h = 8;
  const x0 = ox + CX - Math.floor(w / 2);
  const y0 = oy + GROUND - h;
  px(ctx, x0 - 1, y0 - 1, w + 2, h + 2, OUTLINE);
  // blanket body (3-tone along its length)
  px(ctx, x0 + 4, y0, w - 4, h, TUNIC_MID);
  px(ctx, x0 + 4, y0, w - 4, 2, TUNIC_HI); // blanket top fold
  px(ctx, x0 + 4, y0 + h - 2, w - 4, 2, TUNIC_LO);
  // head resting at one end
  px(ctx, x0, y0 + 1, 5, 5, SKIN);
  px(ctx, x0, y0 + 1, 5, 2, p.hair);
  px(ctx, x0 + 1, y0 + 3, 1, 1, EYE); // closed-eye hint
  // no hand anchor while asleep (tool is hidden)
  return { x: ox + CX, y: y0, rot: 0 };
}

// -- pose definitions per action ----------------------------------------------
//
// Each pose = { legs?, arms? }. `legs`: array of [dx, extraLen] offsets from the
// body centre. `arms.front`: [dx,dy,rot] tip of the LEADING arm (facing +x) →
// hand anchor. `arms.back`: [dx,dy] tip of the trailing arm. Omit `arms` for the
// default hanging arms. dx/dy are in cell px; +x is the facing direction.

const POSES = {
  // idle: relaxed stance, arms at sides (breathing is a transform, not a frame)
  idle: {},

  // walk: 4-frame cycle — contact / pass / contact-other / pass
  walk0: { legs: [[-3, 0], [3, -1]], arms: { front: [2, 5, 0.3], back: [-2, 4] } },
  walk1: { legs: [[-1, 0], [1, 0]], arms: { front: [1, 5, 0.15], back: [-1, 5] } },
  walk2: { legs: [[3, 0], [-3, -1]], arms: { front: [-2, 5, 0.1], back: [2, 4] } },
  walk3: { legs: [[1, 0], [-1, 0]], arms: { front: [1, 5, 0.15], back: [-1, 5] } },

  // forage: stand → bend → reach down to the ground (hand drops each frame)
  forage0: { legs: [[-2, 0], [2, 0]], arms: { front: [3, 4, 0.6], back: [-2, 4] } },
  forage1: {
    legs: [[-2, 0], [2, 0]],
    stoop: 2,
    arms: { front: [4, 7, 1.1], back: [-1, 5] },
  },
  forage2: {
    legs: [[-1, 0], [3, 0]],
    stoop: 4,
    arms: { front: [5, 11, 1.5], back: [0, 6] },
  },

  // cooperate: arm extended toward the partner / small nod
  coop0: { legs: [[-2, 0], [2, 0]], arms: { front: [6, 1, -0.3], back: [-2, 3] } },
  coop1: { legs: [[-2, 0], [2, 0]], stoop: 1, arms: { front: [5, 2, -0.1], back: [-2, 3] } },

  // attack: wind up behind → lunge forward → recover. attack1's thrust must
  // stay inside the 24-px cell: leading tip x = frontX(ox+15/16+lean) + ex, and
  // the arm outline adds 1 px — so lean+ex ≤ 6 (elder is the binding stage).
  // ex 8 overflowed into the attack2 cell; the forward motion comes from the
  // transform-lunge in world-render, not from arm length.
  attack0: { legs: [[-3, 0], [2, 0]], arms: { front: [-2, -3, -1.4], back: [-3, 1] } },
  attack1: { legs: [[-4, 0], [4, 0]], lean: 2, arms: { front: [4, -1, 0.4], back: [-3, 2] } },
  attack2: { legs: [[-2, 0], [2, 0]], arms: { front: [4, 2, 0.1], back: [-2, 3] } },

  // build: raise tool overhead → mid → strike down (hand arcs top → bottom)
  build0: { legs: [[-2, 0], [2, 0]], arms: { front: [3, -6, -1.9], back: [-2, 2] } },
  build1: { legs: [[-2, 0], [2, 0]], arms: { front: [5, -1, -0.4], back: [-2, 3] } },
  build2: { legs: [[-2, 0], [3, 0]], stoop: 1, arms: { front: [5, 6, 0.9], back: [-1, 4] } },

  sleep: "sleep",
};

// -- stage proportions --------------------------------------------------------

function stageProps(stage) {
  if (stage === "child") {
    // big-head proportions, ~70% overall height
    return {
      headTop: 6, headH: 8, headW: 7, // oversized head
      torsoTop: 14, torsoH: 7, bodyW: 7,
      legH: 6,
      lean: 0, stoop: 0,
      hair: HAIR, elder: false,
    };
  }
  if (stage === "elder") {
    return {
      headTop: 2, headH: 6, headW: 5,
      torsoTop: 9, torsoH: 11, bodyW: 8,
      legH: 8,
      lean: 1, stoop: 1, // slightly stooped
      hair: HAIR_ELDER, elder: true,
    };
  }
  // adult
  return {
    headTop: 1, headH: 6, headW: 5,
    torsoTop: 8, torsoH: 12, bodyW: 8,
    legH: 9,
    lean: 0, stoop: 0,
    hair: HAIR, elder: false,
  };
}

// -- atlas builder ------------------------------------------------------------

export function buildFigureAtlas() {
  const canvas = document.createElement("canvas");
  canvas.width = N_COLS * CELL_W;
  canvas.height = STAGES.length * CELL_H;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;

  // frames[stage][actKey] = [Texture, ...]; hand[stage][actKey] = [{x,y,rot}, ...]
  const frames = {};
  const hand = {};
  // one shared source: slice per-frame Textures out of the single atlas texture.
  const source = Texture.from(canvas).source;
  source.scaleMode = "nearest";

  // per-pose hand anchor in ATLAS px, indexed [row][col]
  const handByCell = [];

  for (let row = 0; row < STAGES.length; row++) {
    const stage = STAGES[row];
    const props = stageProps(stage);
    handByCell[row] = [];
    for (let col = 0; col < N_COLS; col++) {
      const ox = col * CELL_W;
      const oy = row * CELL_H;
      const poseKey = POSE_COLS[col];
      // merge base props with this pose's lean/stoop overrides
      const pose = POSES[poseKey];
      const merged = {
        ...props,
        lean: props.lean + ((pose && pose.lean) || 0),
        stoop: props.stoop + ((pose && pose.stoop) || 0),
      };
      const h = drawFigure(ctx, ox, oy, pose === "sleep" ? "sleep" : pose, merged);
      handByCell[row][col] = h;
    }
  }

  // Now slice textures + convert hand anchors to FIGURE-LOCAL px (origin at the
  // sprite anchor: (CX, GROUND) of each cell).
  for (let row = 0; row < STAGES.length; row++) {
    const stage = STAGES[row];
    frames[stage] = {};
    hand[stage] = {};
    for (const [actKey, keys] of Object.entries(FRAME_KEYS)) {
      frames[stage][actKey] = [];
      hand[stage][actKey] = [];
      for (const k of keys) {
        const col = COL[k];
        // Pixi 8 needs a real Rectangle here — a plain {x,y,w,h} leaves
        // texture.width/height undefined and every scale becomes NaN.
        const frame = new Rectangle(col * CELL_W, row * CELL_H, CELL_W, CELL_H);
        const t = new Texture({ source, frame });
        frames[stage][actKey].push(t);
        const hc = handByCell[row][col];
        // figure-local: subtract the anchor point (CX, GROUND) within the cell
        hand[stage][actKey].push({
          x: hc.x - (col * CELL_W + CX),
          y: hc.y - (row * CELL_H + GROUND),
          rot: hc.rot,
        });
      }
    }
  }

  // anim table: one entry per action key
  const anim = {};
  for (const [actKey, keys] of Object.entries(FRAME_KEYS)) {
    anim[actKey] = { durMs: DUR_MS[actKey], n: keys.length };
  }

  return { frames, hand, anim, cellW: CELL_W, cellH: CELL_H, anchorY: ANCHOR_Y };
}

// Map an action code (frame.py) → atlas action key. Walk is handled separately
// in the tick driver (movement, not an `act` code).
export const ACT_KEY = {
  0: "idle",
  1: "forage",
  2: "cooperate",
  3: "attack",
  4: "build",
  5: "sleep",
};
