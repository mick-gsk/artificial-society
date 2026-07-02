// GPU terrain: a data texture + custom fragment shader on one Mesh quad.
//
// The CPU no longer paints per-cell pixels. Instead the world state is packed
// into a tiny w×h data texture (one texel per cell) and a 16×2 palette LUT, and
// this fragment shader turns them into a naturalistic "living diorama": soft,
// organically wavy biome transitions with no visible cell grid, per-biome detail
// noise, and continuously animated water with shore foam — all evaluated per
// screen pixel on the GPU, so a 200×200 world costs the same as a 60×40 one.
//
// Responsibility of this file: the GLSL sources + a `buildTerrainMesh(...)`
// factory. Everything about *feeding* it (packing the data buffer, building the
// LUT, driving uTime/uDaylight, resize) lives in world-render.js.

import { BufferImageSource, Mesh, MeshGeometry, Shader, Texture, UniformGroup } from "pixi.js";

// Pixi's WebGL renderer compiles custom mesh shaders as GLSL ES 1.00 and injects
// a compatibility header (addProgramDefines): in the vertex stage `in`→attribute
// and `out`→varying; in the fragment stage `in`→varying, `texture`→texture2D, and
// crucially it strips the exact line `out vec4 finalColor;` and aliases
// `finalColor`→gl_FragColor. So we write `in`/`out` and output to `finalColor` —
// declaring any other `out` name (or a differently-named output) fails to compile.
// The MVP uniforms below are supplied automatically by Pixi's mesh pipe (global +
// local transform), which is how the quad follows worldRoot's pan/zoom for free.
const TERRAIN_VERT = /* glsl */ `
in vec2 aPosition;
in vec2 aUV;

uniform mat3 uProjectionMatrix;
uniform mat3 uWorldTransformMatrix;
uniform mat3 uTransformMatrix;

out vec2 vUV;

void main() {
  mat3 mvp = uProjectionMatrix * uWorldTransformMatrix * uTransformMatrix;
  gl_Position = vec4((mvp * vec3(aPosition, 1.0)).xy, 0.0, 1.0);
  vUV = aUV;
}
`;

// uData  : w×h RGBA8, NEAREST. R = biome index, G = food (0..255), B = water
//          (0..255), A = reserved (moisture, later). One texel per world cell.
// uLut    : 16×2 RGBA8, NEAREST. Row 0 = dry biome colour, row 1 = lush.
// uGrid   : (w, h) cell counts — lets us walk the data texture by cell centre.
// uTime   : seconds, ever-increasing — drives water ripples/sparkle.
// uDaylight: 0..1 light level (frame.daylight) — dawn/dusk warmth, night blue.
const TERRAIN_FRAG = /* glsl */ `
in vec2 vUV;
out vec4 finalColor;

uniform sampler2D uData;
uniform sampler2D uLut;

uniform vec2 uGrid;
uniform float uTime;
uniform float uDaylight;

// --- cheap hash value-noise (no noise texture; deterministic, tileable-ish) ---
float hash(vec2 p) {
  p = fract(p * vec2(127.1, 311.7));
  p += dot(p, p + 34.5);
  return fract(p.x * p.y);
}

float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f); // smoothstep interpolation
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm2(vec2 p) {
  return valueNoise(p) * 0.65 + valueNoise(p * 2.3 + 7.1) * 0.35;
}

// Colour of a single cell: sample its data texel, pick dry/lush from the LUT by
// biome index, then lerp dry→lush by food (land) or water level (water). This
// mirrors the old CPU painter (_paintTerrain pass 1), just per-cell in GLSL.
// Returns rgb in .rgb and the water level (0..1) in .a so the caller can blend
// a smooth shoreline mask.
vec4 cellColor(vec2 cell) {
  vec2 uv = (cell + 0.5) / uGrid;
  vec4 data = texture(uData, uv);
  float biomeIdx = floor(data.r * 255.0 + 0.5);
  float food = data.g;   // already 0..1
  float water = data.b;  // already 0..1

  // LUT is 16 wide, 2 tall: col = biome index, row 0 = dry, row 1 = lush.
  float lx = (biomeIdx + 0.5) / 16.0;
  vec3 dry = texture(uLut, vec2(lx, 0.25)).rgb;
  vec3 lush = texture(uLut, vec2(lx, 0.75)).rgb;

  // isWater flag is baked into A of the LUT dry row (1.0 = water biome).
  float isWater = texture(uLut, vec2(lx, 0.25)).a;
  float t = mix(food, water, isWater);
  vec3 rgb = mix(dry, lush, clamp(t, 0.0, 1.0));
  return vec4(rgb, mix(0.0, water, isWater));
}

void main() {
  // Fragment position in cell space (0..uGrid), offset so cell centres land on
  // integers — matches how cellColor samples at (cell + 0.5).
  vec2 cellPos = vUV * uGrid - 0.5;

  // Domain-warp the sample position by a little value noise so the bilinear
  // biome bands stop being straight and start wavering organically.
  vec2 warp = vec2(
    valueNoise(cellPos * 3.7) - 0.5,
    valueNoise(cellPos * 3.7 + 19.3) - 0.5
  ) * 0.35;
  vec2 p = cellPos + warp;

  // Blend the FOUR surrounding cell colours (not indices) with smoothstep
  // weights — blending colours is what dissolves the hard cell edges.
  vec2 base = floor(p);
  vec2 f = fract(p);
  vec2 sw = f * f * (3.0 - 2.0 * f);

  vec4 c00 = cellColor(base + vec2(0.0, 0.0));
  vec4 c10 = cellColor(base + vec2(1.0, 0.0));
  vec4 c01 = cellColor(base + vec2(0.0, 1.0));
  vec4 c11 = cellColor(base + vec2(1.0, 1.0));

  vec4 c0 = mix(c00, c10, sw.x);
  vec4 c1 = mix(c01, c11, sw.x);
  vec4 col = mix(c0, c1, sw.y);

  vec3 rgb = col.rgb;
  float waterMask = col.a; // smooth 0..1 water level across the blend

  // Dominant biome under this fragment: sample the nearest cell's index. Used
  // to steer the detail-noise frequency/anisotropy per terrain type.
  float dom = floor(texture(uData, (floor(cellPos + 0.5) + 0.5) / uGrid).r * 255.0 + 0.5);

  // --- per-biome detail: modulate luminance a few percent so ground reads as
  // ground, not flat plastic. Frequency + anisotropy vary by biome family. ---
  if (waterMask < 0.6) {
    float detail;
    // Rough biome buckets by index parity/value; kept simple and cheap.
    // grassland/forest → fine isotropic grain; desert → low-freq dune bands;
    // mountain → streaky vertical striations.
    if (dom < 0.5) {
      // 0 = desert: broad, slow dune bands
      detail = valueNoise(cellPos * vec2(0.6, 1.8)) - 0.5;
    } else if (dom > 2.5 && dom < 3.5) {
      // 3 = mountain: streaky vertical striations
      detail = fbm2(cellPos * vec2(3.5, 1.1)) - 0.5;
    } else {
      // forest / grassland / swamp / other: fine isotropic grain
      detail = fbm2(cellPos * 4.5) - 0.5;
    }
    rgb *= 1.0 + detail * 0.16; // ±8% luminance
  }

  // --- water: two moving ripple wavefronts + sparse specular sparkle. ---
  if (waterMask > 0.6) {
    vec2 wp = cellPos;
    float r1 = sin(dot(wp, vec2(1.9, 1.3)) * 1.7 + uTime * 1.6);
    float r2 = sin(dot(wp, vec2(-1.1, 2.2)) * 1.3 - uTime * 1.1);
    float ripple = (r1 + r2) * 0.5;
    ripple += (valueNoise(wp * 3.0 + uTime * 0.25) - 0.5) * 0.8;

    // deeper water (higher level) sits darker + bluer
    vec3 deep = rgb * 0.82 + vec3(0.0, 0.01, 0.05);
    rgb = mix(rgb, deep, clamp((waterMask - 0.6) / 0.4, 0.0, 1.0));
    rgb += ripple * vec3(0.015, 0.02, 0.03);

    // sparse glinting highlights: thresholded moving noise
    float glint = valueNoise(wp * 6.0 - uTime * 0.6);
    glint = smoothstep(0.86, 0.98, glint);
    rgb += glint * vec3(0.16, 0.19, 0.22);
  }

  // --- shoreline: animated foam band where the water mask crosses ~0.5. ---
  float shoreEdge = 1.0 - abs(waterMask - 0.5) / 0.15;
  if (shoreEdge > 0.0 && waterMask > 0.35 && waterMask < 0.65) {
    float wobble = valueNoise(cellPos * 5.0 + uTime * 0.5) - 0.5;
    float foam = clamp(shoreEdge + wobble * 0.6, 0.0, 1.0);
    rgb = mix(rgb, vec3(0.72, 0.78, 0.82), foam * 0.55);
  }

  // --- daylight: replicate the CPU painter's dawn/dusk warmth + night blue. ---
  float dl = uDaylight;
  float day = 0.45 + 0.55 * dl;
  float dusk = max(0.0, 1.0 - abs(dl - 0.35) / 0.25);
  float warmR = 1.0 + dusk * 0.22;
  float warmB = 1.0 - dusk * 0.18;
  float nightBlue = (1.0 - dl) * 0.04;
  rgb.r *= day * warmR;
  rgb.g *= day;
  rgb.b = rgb.b * day * warmB + nightBlue;

  // at night, mild desaturation toward luminance + a small blue lift
  float lum = dot(rgb, vec3(0.299, 0.587, 0.114));
  rgb = mix(rgb, vec3(lum), (1.0 - dl) * 0.25);
  rgb.b += (1.0 - dl) * 0.015;

  finalColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
`;

// Build a data-texture Texture backed by a persistent Uint8Array (w*h*4 bytes).
// Caller keeps and refills `buffer`, then calls `texture.source.update()`.
export function makeDataTexture(w, h, buffer) {
  const source = new BufferImageSource({
    resource: buffer,
    width: w,
    height: h,
    format: "rgba8unorm",
    scaleMode: "nearest",
    alphaMode: "no-premultiply-alpha",
    addressMode: "clamp-to-edge",
  });
  return new Texture({ source });
}

// Build the 16×2 palette LUT texture (dry row + lush row) from a Uint8Array of
// 16*2*4 bytes. Caller fills it from the biome legend; rebuilt rarely.
export function makeLutTexture(buffer) {
  const source = new BufferImageSource({
    resource: buffer,
    width: 16,
    height: 2,
    format: "rgba8unorm",
    scaleMode: "nearest",
    alphaMode: "no-premultiply-alpha",
    addressMode: "clamp-to-edge",
  });
  return new Texture({ source });
}

// Factory: a Mesh whose quad spans the world rect [offX,offY .. +w*cellPx,+h*cellPx]
// in worldRoot-local pixels, textured by the terrain shader. Returns the mesh and
// its animatable uniform group so the caller can drive uTime/uDaylight per tick
// and repoint uGrid on resize. Throws if the shader fails to compile — the caller
// catches this and falls back to the CPU painter.
export function buildTerrainMesh({ w, h, cellPx, offX, offY, dataTex, lutTex }) {
  const x0 = offX;
  const y0 = offY;
  const x1 = offX + w * cellPx;
  const y1 = offY + h * cellPx;

  const geometry = new MeshGeometry({
    positions: new Float32Array([x0, y0, x1, y0, x1, y1, x0, y1]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  });

  // Animatable scalar/vector uniforms live in one group we can poke each frame.
  const uniforms = new UniformGroup({
    uGrid: { value: new Float32Array([w, h]), type: "vec2<f32>" },
    uTime: { value: 0, type: "f32" },
    uDaylight: { value: 1, type: "f32" },
  });

  const shader = Shader.from({
    gl: { vertex: TERRAIN_VERT, fragment: TERRAIN_FRAG },
    resources: {
      uData: dataTex.source,
      uLut: lutTex.source,
      terrainUniforms: uniforms,
    },
  });

  const mesh = new Mesh({ geometry, shader });
  return { mesh, uniforms, shader };
}

export { TERRAIN_VERT, TERRAIN_FRAG };
