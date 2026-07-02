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
