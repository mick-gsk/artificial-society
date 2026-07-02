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

// A small standing figure, drawn in whites/grays for tinting.
// 10x14: head (brighter), tunic, legs. `elder` gets a gray crown.
export function makeAgentTexture({ elder = false } = {}) {
  const [c, ctx] = makeCanvas(10, 14);
  // head — brightest, so tint reads lighter here
  px(ctx, 3, 0, 4, 4, "rgba(255,255,255,1)");
  if (elder) px(ctx, 2, 0, 6, 1, "rgba(190,190,190,1)"); // gray hair crown
  // tunic
  px(ctx, 2, 4, 6, 6, "rgba(205,205,205,1)");
  // arms
  px(ctx, 1, 5, 1, 4, "rgba(225,225,225,1)");
  px(ctx, 8, 5, 1, 4, "rgba(225,225,225,1)");
  // legs
  px(ctx, 3, 10, 2, 4, "rgba(160,160,160,1)");
  px(ctx, 6, 10, 2, 4, "rgba(160,160,160,1)");
  return tex(c);
}

// Lying (sleeping) figure — horizontal, closed silhouette.
export function makeSleepingTexture() {
  const [c, ctx] = makeCanvas(14, 8);
  px(ctx, 0, 2, 4, 4, "rgba(255,255,255,1)"); // head
  px(ctx, 4, 3, 9, 4, "rgba(190,190,190,1)"); // body under blanket
  px(ctx, 4, 2, 9, 1, "rgba(150,150,150,1)"); // blanket edge
  return tex(c);
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

export function makeDecorTextures() {
  const out = {};

  // broadleaf tree
  {
    const [c, ctx] = makeCanvas(12, 14);
    px(ctx, 5, 9, 2, 5, "#6b4a2b"); // trunk
    px(ctx, 2, 3, 8, 6, "#2f7a44"); // canopy
    px(ctx, 3, 1, 6, 3, "#3c9455");
    px(ctx, 1, 5, 2, 3, "#2a6c3c");
    px(ctx, 9, 5, 2, 3, "#2a6c3c");
    px(ctx, 4, 3, 2, 2, "#4fae67"); // light spots
    out.tree = tex(c);
  }
  // pine
  {
    const [c, ctx] = makeCanvas(10, 14);
    px(ctx, 4, 11, 2, 3, "#5d3f24");
    px(ctx, 1, 8, 8, 3, "#25603a");
    px(ctx, 2, 5, 6, 3, "#2c7245");
    px(ctx, 3, 2, 4, 3, "#358350");
    px(ctx, 4, 0, 2, 2, "#3f9159");
    out.pine = tex(c);
  }
  // grass tuft
  {
    const [c, ctx] = makeCanvas(8, 6);
    px(ctx, 1, 2, 1, 4, "#78a651");
    px(ctx, 3, 0, 1, 6, "#8ab55e");
    px(ctx, 5, 1, 1, 5, "#6f9c4a");
    out.tuft = tex(c);
  }
  // rock
  {
    const [c, ctx] = makeCanvas(10, 8);
    px(ctx, 1, 3, 8, 5, "#7d8790");
    px(ctx, 2, 1, 5, 3, "#8f99a2");
    px(ctx, 3, 2, 2, 1, "#a6b0b8");
    px(ctx, 6, 5, 2, 2, "#6a747d");
    out.rock = tex(c);
  }
  // cactus
  {
    const [c, ctx] = makeCanvas(10, 12);
    px(ctx, 4, 1, 2, 11, "#4c8f57");
    px(ctx, 1, 3, 2, 4, "#4c8f57");
    px(ctx, 1, 3, 4, 2, "#4c8f57");
    px(ctx, 7, 5, 2, 3, "#437f4d");
    px(ctx, 5, 5, 4, 2, "#437f4d");
    out.cactus = tex(c);
  }
  // reed (swamp)
  {
    const [c, ctx] = makeCanvas(8, 10);
    px(ctx, 2, 1, 1, 9, "#5d7a4a");
    px(ctx, 4, 0, 1, 10, "#6b8a54");
    px(ctx, 6, 2, 1, 8, "#516b41");
    px(ctx, 4, 0, 2, 2, "#8a6a3c"); // cattail head
    out.reed = tex(c);
  }
  return out;
}
