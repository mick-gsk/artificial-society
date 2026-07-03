// Programmatic pixel-art textures for the world view — no external assets.
//
// Everything is drawn once into small canvases and scaled with NEAREST so it
// stays crisp at any cell size. Agent bodies are drawn in white/grays so a
// tribe tint keeps identity (head stays brighter than the tunic).

import { Texture } from "pixi.js";

function px(ctx, x, y, w, h, col) {
  ctx.fillStyle = col;
  ctx.fillRect(x, y, w, h);
}

function tex(canvas) {
  const t = Texture.from(canvas);
  t.source.scaleMode = "nearest";
  return t;
}

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return [c, c.getContext("2d")];
}

// -- agents -------------------------------------------------------------------
//
// Agent FIGURE textures moved to figures.js (the 20×28 animated atlas with a
// pose loop per action). Structures/decor/items/emotes/tools below stay here.

// -- structures ----------------------------------------------------------------

export function makeStructureTextures() {
  const out = {};

  // camp: a hide tent
  {
    const [c, ctx] = makeCanvas(14, 12);
    // canvas triangle, built up in pixel rows
    for (let r = 0; r < 9; r++) {
      const half = Math.floor((r * 7) / 9) + 1;
      px(ctx, 7 - half, 2 + r, half * 2, 1, r < 2 ? "#8a5f36" : "#b07c46");
    }
    px(ctx, 6, 0, 2, 3, "#6b4a2b"); // pole tip
    px(ctx, 6, 6, 2, 5, "#4a3018"); // door slit
    px(ctx, 2, 10, 10, 1, "#3a2a16"); // ground line
    out.camp = tex(c);
  }
  // farm: tilled plot with sprouting rows
  {
    const [c, ctx] = makeCanvas(14, 11);
    px(ctx, 0, 1, 14, 10, "#4a3620");
    for (let r = 0; r < 3; r++) {
      px(ctx, 1, 2 + r * 3, 12, 2, "#5e4526");
      px(ctx, 1, 3 + r * 3, 12, 1, "#3d2c18"); // furrow shadow
    }
    // sprouts
    for (const [sx, sy] of [[2, 1], [6, 1], [10, 1], [4, 4], [8, 4], [12, 4], [2, 7], [7, 7], [11, 7]]) {
      px(ctx, sx, sy, 1, 2, "#5fae57");
      px(ctx, sx + 1, sy, 1, 1, "#79c46e");
    }
    out.farm = tex(c);
  }
  // well: stone ring, dark water, wooden crossbar
  {
    const [c, ctx] = makeCanvas(12, 12);
    px(ctx, 2, 5, 8, 6, "#7d8790"); // stone ring body
    px(ctx, 1, 6, 10, 4, "#7d8790");
    px(ctx, 3, 4, 6, 2, "#8f99a2"); // rim highlight
    px(ctx, 4, 6, 4, 3, "#16324a"); // water
    px(ctx, 5, 6, 1, 1, "#5aa9d6"); // glint
    px(ctx, 2, 0, 1, 6, "#6b4a2b"); // posts + crossbar
    px(ctx, 9, 0, 1, 6, "#6b4a2b");
    px(ctx, 2, 0, 8, 1, "#7c5a34");
    out.well = tex(c);
  }
  return out;
}

// -- emotes (action bubbles above the head) ------------------------------------

export function makeEmoteTextures() {
  const out = {};

  // forage: red berry with a green leaf
  {
    const [c, ctx] = makeCanvas(12, 12);
    px(ctx, 4, 4, 5, 5, "#e5484d");
    px(ctx, 5, 3, 3, 1, "#e5484d");
    px(ctx, 7, 1, 3, 2, "#46a758");
    px(ctx, 6, 2, 1, 2, "#3d8b4f");
    out.forage = tex(c);
  }
  // build: hammer
  {
    const [c, ctx] = makeCanvas(12, 12);
    px(ctx, 2, 2, 7, 3, "#9ba1a6"); // head
    px(ctx, 8, 1, 2, 5, "#9ba1a6");
    px(ctx, 5, 5, 2, 6, "#ad7f58"); // handle
    out.build = tex(c);
  }
  // attack: sword (blade + guard + grip)
  {
    const [c, ctx] = makeCanvas(12, 12);
    px(ctx, 7, 0, 2, 7, "#e8e8e8");
    px(ctx, 5, 6, 6, 2, "#c9a227");
    px(ctx, 7, 8, 2, 3, "#8a5a2b");
    out.attack = tex(c);
  }
  // cooperate: speech bubble
  {
    const [c, ctx] = makeCanvas(12, 12);
    px(ctx, 1, 1, 10, 7, "#e8f6ff");
    px(ctx, 3, 8, 2, 2, "#e8f6ff");
    px(ctx, 3, 3, 2, 2, "#3fc5f0");
    px(ctx, 6, 3, 2, 2, "#3fc5f0");
    out.cooperate = tex(c);
  }
  // sleep: Z
  {
    const [c, ctx] = makeCanvas(12, 12);
    px(ctx, 2, 1, 8, 2, "#cdc4ff");
    px(ctx, 6, 4, 3, 2, "#cdc4ff");
    px(ctx, 4, 6, 3, 2, "#cdc4ff");
    px(ctx, 2, 9, 8, 2, "#cdc4ff");
    out.sleep = tex(c);
  }
  return out;
}

// -- ground materials (Physik v2) ------------------------------------------------
//
// One texture per ITEM_* class from serve/frame.py. Common raw materials stay
// small and earthy; the special classes (flint, bone, shard, wonder, fire) are
// the ones that tell the tool story and get more contrast.
export function makeItemTextures() {
  const out = {};

  // berry pile (1)
  {
    const [c, ctx] = makeCanvas(8, 6);
    px(ctx, 1, 2, 3, 3, "#c4384d");
    px(ctx, 4, 3, 3, 3, "#a52e40");
    px(ctx, 3, 1, 2, 2, "#d94a5e");
    out[1] = tex(c);
  }
  // fiber sheaf (2)
  {
    const [c, ctx] = makeCanvas(8, 7);
    px(ctx, 1, 1, 1, 6, "#b7a05c");
    px(ctx, 3, 0, 1, 7, "#cbb26a");
    px(ctx, 5, 1, 1, 6, "#a89252");
    px(ctx, 1, 3, 5, 1, "#8a7440"); // binding
    out[2] = tex(c);
  }
  // wood log (3)
  {
    const [c, ctx] = makeCanvas(10, 6);
    px(ctx, 0, 1, 9, 4, "#7c5a34");
    px(ctx, 0, 2, 9, 1, "#8f6b40");
    px(ctx, 8, 1, 2, 4, "#a98756"); // cut face
    px(ctx, 8, 2, 1, 2, "#c2a06c");
    out[3] = tex(c);
  }
  // stone pebbles (4)
  {
    const [c, ctx] = makeCanvas(8, 6);
    px(ctx, 1, 2, 4, 3, "#8a949c");
    px(ctx, 4, 3, 3, 3, "#727c85");
    px(ctx, 2, 1, 2, 2, "#9ba5ad");
    out[4] = tex(c);
  }
  // clay lump (5)
  {
    const [c, ctx] = makeCanvas(8, 6);
    px(ctx, 1, 2, 6, 4, "#a4693e");
    px(ctx, 2, 1, 4, 2, "#b5794c");
    px(ctx, 3, 3, 2, 1, "#8c5731");
    out[5] = tex(c);
  }
  // meat chunk (6)
  {
    const [c, ctx] = makeCanvas(8, 7);
    px(ctx, 1, 1, 6, 5, "#b8434e");
    px(ctx, 2, 2, 2, 2, "#d16b74"); // marbling
    px(ctx, 5, 4, 1, 1, "#e8d8c8");
    out[6] = tex(c);
  }
  // bones / carcass (7)
  {
    const [c, ctx] = makeCanvas(11, 7);
    px(ctx, 1, 3, 9, 1, "#ddd6c8"); // long bone
    px(ctx, 0, 2, 2, 3, "#e9e2d4");
    px(ctx, 9, 2, 2, 3, "#e9e2d4");
    px(ctx, 4, 1, 1, 5, "#cfc6b4"); // rib
    px(ctx, 6, 1, 1, 5, "#cfc6b4");
    out[7] = tex(c);
  }
  // flint nodule (8) — dark rounded stone with a pale knapping scar
  {
    const [c, ctx] = makeCanvas(9, 7);
    px(ctx, 2, 1, 5, 5, "#3c4652");
    px(ctx, 1, 2, 7, 3, "#3c4652");
    px(ctx, 3, 2, 2, 2, "#546070");
    px(ctx, 5, 3, 2, 1, "#8b9aa8"); // exposed pale scar
    out[8] = tex(c);
  }
  // sharp flake / blade lying on the ground (10)
  {
    const [c, ctx] = makeCanvas(9, 8);
    // triangular flake
    px(ctx, 4, 0, 1, 1, "#eef2f5");
    px(ctx, 3, 1, 3, 1, "#dbe2e8");
    px(ctx, 3, 2, 4, 1, "#c4ced6");
    px(ctx, 2, 3, 5, 1, "#aab6c0");
    px(ctx, 2, 4, 6, 1, "#93a1ad");
    px(ctx, 4, 0, 1, 5, "#ffffff"); // edge glint
    out[10] = tex(c);
  }
  // discovered material (9) — small glowing crystal
  {
    const [c, ctx] = makeCanvas(8, 9);
    px(ctx, 3, 0, 2, 2, "#d9b8ff");
    px(ctx, 2, 2, 4, 4, "#a86fe8");
    px(ctx, 3, 6, 2, 2, "#7b4bc4");
    px(ctx, 3, 2, 1, 3, "#ecdcff"); // inner light
    out[9] = tex(c);
  }
  // fire (11)
  {
    const [c, ctx] = makeCanvas(9, 10);
    px(ctx, 2, 8, 5, 2, "#5d4126"); // embers/wood base
    px(ctx, 2, 3, 5, 5, "#e8642c");
    px(ctx, 3, 1, 3, 4, "#f59d3d");
    px(ctx, 4, 0, 1, 3, "#ffd66e");
    px(ctx, 4, 4, 1, 3, "#fff0b8"); // hot core
    out[11] = tex(c);
  }
  return out;
}

// Tool in hand: a hafted blade (sharp) or a fist stone (blunt), drawn to read
// at ~half figure height.
export function makeToolTextures() {
  const out = {};
  {
    const [c, ctx] = makeCanvas(7, 9); // sharp blade
    px(ctx, 3, 0, 2, 4, "#e8eef2");
    px(ctx, 2, 1, 1, 3, "#c8d2da");
    px(ctx, 3, 4, 2, 5, "#8a5a2b"); // grip
    px(ctx, 4, 0, 1, 4, "#ffffff"); // edge
    out.sharp = tex(c);
  }
  {
    const [c, ctx] = makeCanvas(6, 6); // blunt stone
    px(ctx, 1, 1, 4, 4, "#8a949c");
    px(ctx, 2, 1, 2, 1, "#a3adb5");
    out.blunt = tex(c);
  }
  {
    const [c, ctx] = makeCanvas(8, 8); // carry bundle on the back
    px(ctx, 1, 2, 6, 5, "#7a6238");
    px(ctx, 2, 1, 4, 2, "#8d7344");
    px(ctx, 1, 4, 6, 1, "#5f4c2b"); // strap
    out.bundle = tex(c);
  }
  return out;
}

// -- terrain decorations --------------------------------------------------------
//
// Upgraded (R7) to 24–32-px sprites with the same light-upper-left, 2–3-tone
// shading convention as the figure atlas (figures.js OUTLINE = "#12161d"), plus
// a soft directional ground shadow (drawn low-right of the base, opposite the
// light) so decor reads as sitting IN the world rather than floating pixel art.
// Placement/density/dirty-key logic in world-render.js `_buildDecor` is
// unchanged — only these texture bodies grew.

const DECOR_OUTLINE = "#12161d"; // same near-black outline as figures.js

// Soft directional cast shadow: a squashed dark ellipse at the object's ground
// contact point, offset down-right (the light falls from upper-left). Alpha
// blend (not a flat fill) so it reads as a shadow, not a solid shape; anchor
// (0.5, 1) means this must sit right at the canvas's bottom edge.
function castShadow(ctx, cx, groundY, rx, ry) {
  ctx.save();
  ctx.translate(cx + rx * 0.25, groundY);
  ctx.scale(1, ry / rx);
  ctx.beginPath();
  ctx.arc(0, 0, rx, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(4,8,6,0.32)";
  ctx.fill();
  ctx.restore();
}

export function makeDecorTextures() {
  const out = {};

  // broadleaf tree (24×30) — trunk with a lit/shadow face, layered 3-tone
  // canopy (light upper-left, mid, dark lower-right), cast shadow at the base.
  {
    const [c, ctx] = makeCanvas(24, 30);
    castShadow(ctx, 12, 29, 8, 3);
    // trunk: outline, dark shadow-side, lit face
    px(ctx, 9, 18, 6, 10, DECOR_OUTLINE);
    px(ctx, 10, 19, 4, 8, "#4a3018");
    px(ctx, 10, 19, 2, 8, "#6b4a2b"); // lit left face
    // canopy: outline blob, then dark/mid/light lobes (light from upper-left)
    px(ctx, 3, 5, 18, 15, DECOR_OUTLINE);
    px(ctx, 4, 6, 16, 13, "#1f5c33"); // dark base tone
    px(ctx, 5, 7, 12, 10, "#2f7a44"); // mid tone
    px(ctx, 6, 3, 10, 7, "#2f7a44");
    px(ctx, 6, 8, 8, 6, "#3c9455"); // lit lobe, upper-left biased
    px(ctx, 7, 4, 6, 4, "#3c9455");
    px(ctx, 8, 6, 4, 3, "#57b374"); // brightest highlight, upper-left
    out.tree = tex(c);
  }
  // pine (20×30) — conical layered tiers, lit on the left edge of each tier.
  {
    const [c, ctx] = makeCanvas(20, 30);
    castShadow(ctx, 10, 29, 7, 3);
    px(ctx, 8, 22, 4, 7, DECOR_OUTLINE);
    px(ctx, 9, 23, 2, 6, "#5d3f24");
    px(ctx, 9, 23, 1, 6, "#7a5631"); // lit trunk edge
    // three tiers, each outlined then dark/mid/light banded left→right
    px(ctx, 1, 15, 18, 8, DECOR_OUTLINE);
    px(ctx, 2, 16, 16, 6, "#1c4a2c");
    px(ctx, 2, 16, 8, 6, "#25603a");
    px(ctx, 2, 16, 4, 6, "#316e46"); // lit
    px(ctx, 3, 9, 14, 7, DECOR_OUTLINE);
    px(ctx, 4, 10, 12, 5, "#20502f");
    px(ctx, 4, 10, 6, 5, "#2c7245");
    px(ctx, 4, 10, 3, 5, "#388153"); // lit
    px(ctx, 5, 3, 10, 7, DECOR_OUTLINE);
    px(ctx, 6, 4, 8, 5, "#276036");
    px(ctx, 6, 4, 4, 5, "#358350");
    px(ctx, 6, 4, 2, 4, "#48965f"); // lit tip
    out.pine = tex(c);
  }
  // grass tuft (10×8) — kept small/simple (ground cover, not a shading target)
  // but with a hint of a dark base and a tiny contact shadow.
  {
    const [c, ctx] = makeCanvas(10, 8);
    castShadow(ctx, 5, 7, 3, 1.2);
    px(ctx, 1, 2, 1, 4, "#5d824a"); // shadow-side blade
    px(ctx, 3, 0, 1, 6, "#8ab55e"); // lit blade
    px(ctx, 5, 1, 1, 5, "#6f9c4a");
    px(ctx, 7, 2, 1, 4, "#5d824a");
    out.tuft = tex(c);
  }
  // rock (24×20) — faceted boulder: outline, dark shadow-side facet, mid body,
  // bright upper-left highlight facet, cast shadow.
  {
    const [c, ctx] = makeCanvas(24, 20);
    castShadow(ctx, 12, 19, 9, 3);
    px(ctx, 2, 6, 20, 12, DECOR_OUTLINE);
    px(ctx, 3, 7, 18, 10, "#5c656d"); // dark shadow-side base
    px(ctx, 3, 7, 10, 10, "#7d8790"); // mid body, upper-left biased
    px(ctx, 5, 3, 11, 7, DECOR_OUTLINE);
    px(ctx, 6, 4, 9, 5, "#8f99a2");
    px(ctx, 6, 4, 5, 4, "#a6b0b8"); // highlight facet, upper-left
    px(ctx, 14, 12, 6, 4, "#4d555c"); // deep shadow facet, lower-right
    out.rock = tex(c);
  }
  // cactus (16×26) — saguaro silhouette, 3-tone barrel + arms, light-left rib
  // highlight, cast shadow.
  {
    const [c, ctx] = makeCanvas(16, 26);
    castShadow(ctx, 8, 25, 6, 2.4);
    // main trunk
    px(ctx, 5, 2, 6, 22, DECOR_OUTLINE);
    px(ctx, 6, 3, 4, 20, "#3a6f43"); // dark shadow-side
    px(ctx, 6, 3, 2, 20, "#4c8f57"); // mid
    px(ctx, 6, 3, 1, 20, "#63a86e"); // lit rib, upper-left edge
    // left arm
    px(ctx, 0, 8, 6, 9, DECOR_OUTLINE);
    px(ctx, 1, 9, 4, 7, "#3a6f43");
    px(ctx, 1, 9, 2, 7, "#4c8f57");
    // right arm
    px(ctx, 10, 6, 6, 9, DECOR_OUTLINE);
    px(ctx, 11, 7, 4, 7, "#3a6f43");
    px(ctx, 11, 7, 2, 7, "#4c8f57");
    px(ctx, 11, 7, 1, 7, "#5c9c66"); // faint lit edge even on the shadow-facing arm
    out.cactus = tex(c);
  }
  // reed (swamp, 12×16) — three blades + cattail head, light-left tint,
  // small contact shadow (reeds barely cast one — thin/tall).
  {
    const [c, ctx] = makeCanvas(12, 16);
    castShadow(ctx, 6, 15, 4, 1.4);
    px(ctx, 2, 2, 1, 13, "#425c34"); // shadow-side blade
    px(ctx, 5, 0, 1, 15, "#6b8a54"); // lit centre blade
    px(ctx, 5, 0, 1, 6, "#84a568"); // brighter tip
    px(ctx, 8, 3, 1, 12, "#516b41");
    px(ctx, 4, 0, 3, 3, "#8a6a3c"); // cattail head
    px(ctx, 4, 0, 1, 3, "#a3854f"); // lit side of the head
    out.reed = tex(c);
  }
  return out;
}
