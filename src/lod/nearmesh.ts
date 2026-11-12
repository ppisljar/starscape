/* near-mesh real-3D LOD (DESIGN.md §5 tier-3 custom layer; §12 "mote → mesh") — the CLOSE look.
 *
 * The field renders every node as a GPU billboard (two draw calls, 10⁶ flyable). That is the FAR
 * look and it never changes. This optional subsystem adds a SECOND, tiny render layer: for the
 * handful of nodes that are LARGE ON SCREEN (their projected glyph radius has grown past a pixel
 * threshold as you fly in), it draws a genuine shaded, optionally textured 3D mesh — a planet you
 * can orbit — and fades those nodes' billboards so the two never double-draw. Fly away and the
 * meshes vanish, the billboards return, and nothing of the 10⁶ path was touched.
 *
 * THE THREE HARD RULES this file keeps:
 *   1. OPT-IN ⇒ ZERO COST WHEN OFF. `createNearMesh` is the only entry; a consumer that never
 *      calls it instantiates nothing — no objects, no buffers, no textures, no per-frame work.
 *      Even after creation, `setEnabled(false)` frees the field of every per-frame cost and the
 *      first buffers allocate lazily on the first ENABLED update, not at construction.
 *   2. DEFAULT SPHERE ⇒ ONE DRAW CALL. With no shapeFacet every promoted node is a sphere, so a
 *      single InstancedMesh (one draw call) covers the whole near pool. Alternate shapes cost a
 *      pool each, created LAZILY only when a node of that shape actually promotes.
 *   3. TEXTURES ARE SELF-CONTAINED AND OPT-IN. When textureFacet is null the material never
 *      samples a texture and NONE is generated (flat tinted shading). When it is set, the ~10
 *      planet textures are generated PROCEDURALLY from the lib's own `hash` (DESIGN §10.1) into one
 *      DataArrayTexture — no external image, nothing to download, deterministic forever.
 *
 * PROMOTION IS BY PROJECTED SCREEN SIZE, never a fixed nearest-K. A node becomes a mesh iff its
 * projected glyph radius (the exact `glyphPx` the billboard and the zoom ladder already use)
 * exceeds `promotePx`; `count` is only a PERFORMANCE CAP (promote the largest-on-screen first).
 * So a 1k-node graph seen whole stays all-billboards (every node tiny → nothing promotes); only
 * flying close to a node — making it and its neighbours big on screen — promotes them. The
 * pick grid finds LOCAL candidates cheaply (never an O(n) scan); the px test decides each one.
 *
 * PER-FRAME COST IS O(local). The grid buckets a snapshot of view positions (rebuilt on settle /
 * throttled under motion, never per frame); each promoted node's LIVE position is recomputed every
 * frame via `frames.transformPoint` + spread + field-spin (O(K)), so a mesh tracks the spinning /
 * morphing field exactly and sits precisely where its faded billboard was — a seamless handoff.
 */

import {
  BoxGeometry,
  CapsuleGeometry,
  ConeGeometry,
  CylinderGeometry,
  DataArrayTexture,
  DataTexture,
  DodecahedronGeometry,
  DynamicDrawUsage,
  Group,
  IcosahedronGeometry,
  InstancedBufferAttribute,
  InstancedMesh,
  LinearFilter,
  Matrix4,
  Mesh,
  MeshBasicNodeMaterial,
  OctahedronGeometry,
  RepeatWrapping,
  SphereGeometry,
  SRGBColorSpace,
  TetrahedronGeometry,
  TorusGeometry,
  Vector3,
} from 'three/webgpu';
import type { BufferGeometry, Node, Object3D, Texture } from 'three/webgpu';
import {
  Fn,
  If,
  abs,
  asin,
  atan,
  attribute,
  clamp,
  dot,
  float,
  int,
  max,
  min,
  mix,
  normalLocal,
  pow,
  smoothstep,
  sqrt,
  step,
  texture,
  uint,
  uv,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { Graph } from '../core/types';
import type { Field } from '../field/field';
import { buildGrid, cellMaxOf, type Grid } from '../pick/grid';
import { glyphPx } from '../field/channels';
import { hash } from '../core/hash';
import { D_REF_FRACTION, FALLOFF, NODE_SCALE, SIZE_MAX_WORLD } from '../field/constants';
import {
  createTerrain,
  earthTerrainSource,
  TERRAIN_ALT_ON,
  type Terrain,
  type TerrainColorNode,
  type TerrainHeightSource,
} from './terrain';

/* ---- registries (v1) ------------------------------------------------------------------- */

/** The ten built-in shapes, index 0 = sphere (the default). `setShapeFacet`-style wrapping maps a
 *  facet value index k to shape (k mod SHAPE_NAMES.length). */
export const NEAR_SHAPE_NAMES = [
  'sphere',
  'box',
  'icosahedron',
  'octahedron',
  'tetrahedron',
  'dodecahedron',
  'cylinder',
  'cone',
  'torus',
  'capsule',
] as const;
export type NearShapeName = (typeof NEAR_SHAPE_NAMES)[number];

/** The ten built-in procedural planet textures, in layer order. A texture facet's value index k
 *  selects texture (k mod NEAR_TEXTURE_NAMES.length). */
export const NEAR_TEXTURE_NAMES = [
  'earth',
  'mars',
  'moon',
  'jupiter',
  'neptune',
  'venus',
  'ice',
  'lava',
  'desert',
  'star',
] as const;
export type NearTextureName = (typeof NEAR_TEXTURE_NAMES)[number];

/** Equirectangular texture resolution (one layer). */
export const TEX_W = 512;
export const TEX_H = 256;

/** Scale a geometry so its bounding sphere has radius 1 — "unit-radius normalized", so the
 *  per-instance scale IS the node's world radius. In place; returns the geometry. */
function normalizeUnit(geo: BufferGeometry): BufferGeometry {
  geo.computeBoundingSphere();
  const r = geo.boundingSphere ? geo.boundingSphere.radius : 1;
  if (r > 0 && Number.isFinite(r) && Math.abs(r - 1) > 1e-6) geo.scale(1 / r, 1 / r, 1 / r);
  geo.computeBoundingSphere();
  return geo;
}

/** Build one of the ten built-in shape geometries (unit-radius normalized). The low-poly
 *  polyhedra are faceted (flat normals) for a cut-gem read; the smooth bodies keep their
 *  vertex normals. */
function builtinShape(index: number): BufferGeometry {
  switch (index % NEAR_SHAPE_NAMES.length) {
    case 1:
      return normalizeUnit(new BoxGeometry(1, 1, 1));
    case 2:
      return faceted(new IcosahedronGeometry(1, 0));
    case 3:
      return faceted(new OctahedronGeometry(1, 0));
    case 4:
      return faceted(new TetrahedronGeometry(1, 0));
    case 5:
      return faceted(new DodecahedronGeometry(1, 0));
    case 6:
      return normalizeUnit(new CylinderGeometry(0.7, 0.7, 1.6, 36, 1));
    case 7:
      return normalizeUnit(new ConeGeometry(0.9, 1.8, 36, 1));
    case 8:
      return normalizeUnit(new TorusGeometry(0.66, 0.3, 20, 44));
    case 9:
      return normalizeUnit(new CapsuleGeometry(0.55, 1.0, 10, 20));
    case 0:
    default:
      return normalizeUnit(new SphereGeometry(1, 48, 32));
  }
}

/** Faceted = flat per-triangle normals: drop the index so no vertex is shared, then recompute. */
function faceted(geo: BufferGeometry): BufferGeometry {
  const flat = geo.index ? geo.toNonIndexed() : geo;
  if (flat !== geo) geo.dispose();
  flat.computeVertexNormals();
  return normalizeUnit(flat);
}

/* ---- procedural planet textures (seeded; DESIGN §10.1, no external image) ---------------- */

/** Seamless-in-longitude value noise: the x lattice wraps modulo `period` so the equirectangular
 *  texture has no visible seam when wrapped around a sphere. `u` ∈ [0,1) is longitude, `y` is a
 *  free (latitude) coordinate already scaled to lattice units. */
export function wnoise(u: number, y: number, period: number, salt: number): number {
  const x = u * period;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const xf = x - x0;
  const yf = y - y0;
  const wrap = (i: number): number => ((i % period) + period) % period;
  const h = (ix: number, iy: number): number => hash(wrap(ix) + (iy & 4095) * 8192, salt);
  const su = xf * xf * (3 - 2 * xf);
  const sv = yf * yf * (3 - 2 * yf);
  const a = h(x0, y0);
  const b = h(x0 + 1, y0);
  const c = h(x0, y0 + 1);
  const d = h(x0 + 1, y0 + 1);
  return a * (1 - su) * (1 - sv) + b * su * (1 - sv) + c * (1 - su) * sv + d * su * sv;
}

/** Fractal (fbm) over the seamless noise: `cells` lattice cells across the longitude at octave 0,
 *  doubling each octave; `yscale` the matching latitude cell scale. */
function wfbm(u: number, vy: number, cells: number, yscale: number, salt: number, oct = 4): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let p = cells;
  let ys = yscale;
  for (let i = 0; i < oct; i++) {
    sum += amp * wnoise(u, vy * ys, p, (salt + i * 0x9e37) | 0);
    norm += amp;
    amp *= 0.5;
    p *= 2;
    ys *= 2;
  }
  return sum / norm;
}

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
const smooth = (a: number, b: number, x: number): number => {
  const t = clamp01((x - a) / (b - a || 1e-6));
  return t * t * (3 - 2 * t);
};
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

type Planet = (u: number, v: number, salt: number) => [number, number, number, number];

/* Each planet writes RGB albedo (0..255, sRGB-ish) + A = emissive mask (0..255). A is 0 for
 * non-emissive bodies; lava's cracks and the star glow through it. v: 0 = south pole, 1 = north. */
const PLANETS: Record<NearTextureName, Planet> = {
  earth(u, v, salt) {
    const lat = Math.abs(v - 0.5) * 2; // 0 equator .. 1 pole
    const land = wfbm(u, v, 5, 3, salt, 5);
    const ice = smooth(0.82, 0.96, lat);
    if (land > 0.52) {
      const h = smooth(0.52, 0.8, land);
      const r = lerp(70, 150, h);
      const g = lerp(120, 150, h);
      const b = lerp(55, 95, h);
      return [lerp(r, 235, ice), lerp(g, 240, ice), lerp(b, 245, ice), 0];
    }
    const deep = smooth(0.2, 0.52, land);
    return [lerp(14, 40, deep), lerp(54, 110, deep), lerp(120, 165, deep), 0];
  },
  mars(u, v, salt) {
    const n = wfbm(u, v, 6, 3, salt, 5);
    const maria = smooth(0.45, 0.62, n);
    const lat = Math.abs(v - 0.5) * 2;
    const ice = smooth(0.9, 0.99, lat);
    const r = lerp(165, 110, maria);
    const g = lerp(80, 52, maria);
    const b = lerp(48, 40, maria);
    return [lerp(r, 240, ice), lerp(g, 240, ice), lerp(b, 245, ice), 0];
  },
  moon(u, v, salt) {
    let g = 120 + (wfbm(u, v, 7, 4, salt, 5) - 0.5) * 70;
    // a few hashed craters: darker floor, a bright rim
    for (let k = 0; k < 26; k++) {
      const cu = hash(k, (salt + 11) | 0);
      const cv = 0.1 + hash(k, (salt + 97) | 0) * 0.8;
      const rad = 0.02 + hash(k, (salt + 131) | 0) * 0.06;
      let du = Math.abs(u - cu);
      du = Math.min(du, 1 - du);
      const d = Math.hypot(du, v - cv) / rad;
      if (d < 1.25) {
        g += -38 * smooth(1.0, 0.2, d) + 42 * smooth(0.78, 1.02, d) * smooth(1.25, 1.0, d);
      }
    }
    g = clamp01(g / 255) * 255;
    return [g, g * 0.99, g * 0.95, 0];
  },
  jupiter(u, v, salt) {
    const bands = Math.sin(v * Math.PI * 11) * 0.5 + 0.5;
    const warp = (wfbm(u, v, 9, 2, salt, 4) - 0.5) * 0.12;
    const b = clamp01(bands + warp);
    let r = lerp(205, 150, b);
    let g = lerp(175, 95, b);
    let bl = lerp(130, 60, b);
    // the great swirl
    let du = Math.abs(u - 0.68);
    du = Math.min(du, 1 - du);
    const d = Math.hypot(du * 1.5, v - 0.4) / 0.1;
    const spot = smooth(1.2, 0.2, d);
    r = lerp(r, 200, spot);
    g = lerp(g, 85, spot);
    bl = lerp(bl, 60, spot);
    return [r, g, bl, 0];
  },
  neptune(u, v, salt) {
    const bands = Math.sin(v * Math.PI * 8) * 0.5 + 0.5;
    const warp = (wfbm(u, v, 8, 2, salt, 4) - 0.5) * 0.18;
    const b = clamp01(bands + warp);
    return [lerp(40, 20, b), lerp(90, 60, b), lerp(200, 150, b), 0];
  },
  venus(u, v, salt) {
    const n = wfbm(u + v * 0.3, v, 5, 2, salt, 5); // domain-warped for a cloudy swirl
    const c = clamp01(0.4 + n * 0.6);
    return [lerp(210, 245, c), lerp(185, 225, c), lerp(120, 170, c), 0];
  },
  ice(u, v, salt) {
    const base = wfbm(u, v, 6, 3, salt, 4);
    const crackN = Math.abs(wfbm(u, v, 10, 5, (salt + 7) | 0, 4) - 0.5) * 2;
    const crack = smooth(0.14, 0.0, crackN); // ridged → thin dark-blue fissures
    const w = lerp(235, 205, base * 0.5);
    return [lerp(w, 150, crack), lerp(w + 6, 185, crack), 255, 0];
  },
  lava(u, v, salt) {
    const crackN = Math.abs(wfbm(u, v, 9, 5, salt, 5) - 0.5) * 2;
    const crack = smooth(0.17, 0.0, crackN); // glowing fissures
    const rock = 18 + wfbm(u, v, 12, 6, (salt + 3) | 0, 4) * 26;
    const r = lerp(rock, 255, crack);
    const g = lerp(rock * 0.6, 120, crack);
    const b = lerp(rock * 0.6, 20, crack);
    return [r, g, b, crack * 220]; // emissive on the cracks
  },
  desert(u, v, salt) {
    const dunes = Math.sin(v * Math.PI * 26 + wfbm(u, v, 7, 3, salt, 3) * 5) * 0.5 + 0.5;
    const n = wfbm(u, v, 6, 3, (salt + 5) | 0, 4);
    const t = clamp01(dunes * 0.6 + n * 0.4);
    return [lerp(200, 232, t), lerp(160, 198, t), lerp(96, 140, t), 0];
  },
  star(u, v, salt) {
    const gran = wfbm(u, v, 14, 7, salt, 5);
    const c = clamp01(0.6 + gran * 0.5);
    const r = lerp(250, 255, c);
    const g = lerp(210, 250, c);
    const b = lerp(90, 170, c);
    return [r, g, b, lerp(150, 230, c)]; // emissive everywhere
  },
};

/** sRGB byte (0..255) → linear [0, 1]. The planet textures are authored/generated in sRGB and the
 *  shader samples them through an SRGBColorSpace DataArrayTexture (three decodes to linear before
 *  lighting), so a per-layer average that is to MATCH the sampled planet must be averaged in the
 *  same linear space — otherwise the 'tint' billboards read brighter than the 'full'/mesh planet. */
function srgbToLinear(b: number): number {
  const c = b / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** The per-layer DOMINANT colour driving the billboards' cheap 'tint' mode: the linear-space mean
 *  RGB of each texture layer, as a flat `depth*3` Float32Array (layer L at [L*3 .. L*3+2], 0..1).
 *  Linear so it matches what the 'full' billboard and the near mesh sample — the three share ONE
 *  texture array, so one average per layer keeps far/near identical. Pure (no three/GPU), so the
 *  demo can precompute it once beside `generateNearTextures` and hand it to `Field.setGlyphTexture`. */
export function nearTextureAverages(raw: Uint8Array, layerCount?: number): Float32Array {
  const texels = TEX_W * TEX_H;
  const depth = layerCount ?? Math.max(1, Math.round(raw.length / (texels * 4)));
  const avg = new Float32Array(depth * 3);
  for (let L = 0; L < depth; L++) {
    let rs = 0;
    let gs = 0;
    let bs = 0;
    const base = L * texels * 4;
    for (let t = 0; t < texels; t++) {
      const o = base + t * 4;
      rs += srgbToLinear(raw[o]!);
      gs += srgbToLinear(raw[o + 1]!);
      bs += srgbToLinear(raw[o + 2]!);
    }
    avg[L * 3] = rs / texels;
    avg[L * 3 + 1] = gs / texels;
    avg[L * 3 + 2] = bs / texels;
  }
  return avg;
}

/** Wrap a raw layer-major RGBA buffer into a CONFIGURED planet DataArrayTexture (sRGB, longitude
 *  wrap, linear filter, no mipmaps) — the one place that config lives, shared by this module's own
 *  lazy `ensureTextures` and by a consumer that wants to hand the SAME array to both the LOD
 *  (`createNearMesh({ textures })`) and the billboards (`Field.setGlyphTexture`). */
export function wrapNearTextures(raw: Uint8Array): DataArrayTexture {
  const depth = Math.max(1, Math.round(raw.length / (TEX_W * TEX_H * 4)));
  const tex = new DataArrayTexture(raw, TEX_W, TEX_H, depth);
  tex.colorSpace = SRGBColorSpace; // RGBA8 is the DataArrayTexture default format
  tex.wrapS = RepeatWrapping; // longitude wraps (seamless)
  tex.minFilter = LinearFilter;
  tex.magFilter = LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/** Generate all ten planet textures into one interleaved RGBA Uint8 buffer (layer-major), seeded
 *  deterministically from `seed`. ~0.5 MB; built once, only when a textureFacet is actually used. */
export function generateNearTextures(seed = 1): Uint8Array {
  const W = TEX_W;
  const H = TEX_H;
  const depth = NEAR_TEXTURE_NAMES.length;
  const data = new Uint8Array(W * H * depth * 4);
  for (let L = 0; L < depth; L++) {
    const planet = PLANETS[NEAR_TEXTURE_NAMES[L]!]!;
    const salt = (seed * 0x1000193 + L * 0x9e3779b1) | 0;
    const layerBase = L * W * H * 4;
    for (let py = 0; py < H; py++) {
      const v = (py + 0.5) / H;
      for (let px = 0; px < W; px++) {
        const u = (px + 0.5) / W;
        const [r, g, b, a] = planet(u, v, salt);
        const o = layerBase + (py * W + px) * 4;
        data[o] = clamp01(r / 255) * 255;
        data[o + 1] = clamp01(g / 255) * 255;
        data[o + 2] = clamp01(b / 255) * 255;
        data[o + 3] = clamp01(a / 255) * 255;
      }
    }
  }
  return data;
}

/* ---- procedural surface detail in the fragment shader (the HARD-EDGE planets) ---------------
 *
 * "Option B" / continuous LOD. The baked textures above run out of pixels when you fly very close
 * (magnification → soft). For a planet that is BIG on screen we instead evaluate its surface
 * PROCEDURALLY per-fragment, from the sphere's surface point — continuous, effectively infinite
 * detail, no texture memory, no generation hitch — and blend baked→procedural over a transition as
 * the node grows (`iDetail` 0→1). THE BLEND IS SEAMLESS BY CONSTRUCTION: the low octaves below are
 * a BIT-EXACT TSL port of the CPU value-noise (same `hash`, same lattice, same frequencies/salts),
 * so at detail=0 the procedural colour equals the baked texel; the extra octaves (freqs the 512×256
 * texture never stored) fade in with `iDetail`, so detail only ever ADDS, never jumps.
 *
 * Rolled to the four HARD-EDGE planets whose sharp features a magnified baked texture blurs — EARTH
 * (land/sea coastline, a `land > 0.52` threshold), ICE (cracked ice sheets), LAVA (glowing cracks,
 * emissive) and MOON (crater rims). Each has its own TSL colour function mirroring its `PLANETS.*`
 * baked texel; a fragment runs ONLY its own planet's branch (routed by texture layer in
 * `buildMaterial`), so the smooth bodies (jupiter/neptune/venus/desert/star/mars) stay pure baked.
 * Opt-in and self-contained like the rest of the file; OFF (the default) compiles the exact material
 * shipped today, references no `iDetail`, uploads nothing — zero added cost. */

/** The texture layers that get per-fragment procedural detail: the HARD-EDGE planets whose sharp
 *  features (earth coastlines, ice fissures, lava cracks, moon crater rims) a magnified baked texture
 *  blurs. The smooth bodies (jupiter/neptune/venus/desert/star/mars) stay baked — their win is
 *  negligible and not worth the shader cost. `buildMaterial` routes a fragment to its own planet's
 *  branch by this membership; the fill loop ramps `iDetail` only for a node on one of these layers. */
export const PROCEDURAL_LAYERS: ReadonlySet<number> = new Set(
  (['earth', 'ice', 'lava', 'moon'] as const).map((name) => NEAR_TEXTURE_NAMES.indexOf(name)),
);
/** On-screen glyph radius (CSS px) where baked→procedural detail ramps 0→1. Below LO it is pure
 *  baked (today's look); above HI it is full procedural. ~LO is roughly where the 256-tall baked
 *  texture starts to magnify (planet screen-diameter ≳ texture height). */
const DETAIL_PX_LO = 90;
const DETAIL_PX_HI = 360;
/** Extra octaves added BEYOND the baked set, and how hard their (ramped-in) contribution hits — the
 *  finer coastline the low-res texture never had. Modest, because near the land/sea threshold a
 *  small perturbation already carves visible new inlets. */
export const DETAIL_OCTAVES = 4;
export const DETAIL_GAIN = 2.2;
/** Pole-pinch guard. The equirect uv from `normalLocal` (asin/atan) degenerates toward the poles —
 *  longitude spins faster and faster, so BOTH the baked texel (its texture columns converge) and the
 *  procedural noise smear into a radial starburst over a real polar cap, not just the exact point. We
 *  can't fade to baked (baked pinches too), so each planet blends its surface toward a clean constant
 *  POLE COLOUR across the cap (|latitude| ramps LO→HI, 1 = pole): earth → an ice cap, the moon → its
 *  mean grey, etc. The constant has no longitude term, so there is nothing left to spin — a clean pole,
 *  smoothstep-blended so no ring forms where the detailed surface meets it. */
const POLE_CAP_LO = 0.62;
const POLE_CAP_HI = 0.84;

const INV_2PI = 1 / (2 * Math.PI);
const INV_PI = 1 / Math.PI;

type FNode = Node<'float'>;
type UNode = Node<'uint'>;

/** Exact TSL port of `core/hash.ts` `hash(i, salt)` in uint32 (WGSL/GLSL uint arithmetic wraps mod
 *  2^32, matching `Math.imul` bit-for-bit — verified bit-exact). `idx` a uint node; `saltU` the
 *  salt as an UNSIGNED JS int (`salt >>> 0`). Returns the noise draw in [0, 1) as a float node. */
function tslHash(idx: UNode, saltU: number): FNode {
  const K = uint(0x9e3779b1); // 2654435761 — the golden-ratio multiplier (same constant twice)
  let x = idx.bitXor(uint(saltU >>> 0).mul(K)).mul(K);
  x = x.bitXor(x.shiftRight(uint(15)));
  x = x.mul(uint(2246822519));
  x = x.bitXor(x.shiftRight(uint(13)));
  return x.toFloat().mul(1 / 4294967296) as unknown as FNode;
}

/** Exact TSL port of `wnoise(u, y, period, salt)`: seamless-in-longitude value noise. `uu` ∈ [0,1)
 *  longitude, `yy` the latitude coordinate already in lattice units. `period`/`saltU` are the
 *  octave's compile-time constants (inlined). Lattice wraps modulo `period` in x exactly as the CPU
 *  does, so the sphere has no longitude seam. (iy < 4096 here, so the CPU's `& 4095` is identity.) */
function wnoiseTSL(uu: FNode, yy: FNode, period: number, saltU: number): FNode {
  const pf = float(period);
  const x = uu.mul(period);
  const x0 = x.floor();
  const y0 = yy.floor();
  const xf = x.sub(x0);
  const yf = yy.sub(y0);
  const su = xf.mul(xf).mul(float(3).sub(xf.mul(2)));
  const sv = yf.mul(yf).mul(float(3).sub(yf.mul(2)));
  const wrapF = (ixF: FNode): FNode => ixF.sub(ixF.div(pf).floor().mul(pf)) as unknown as FNode; // [0, period)
  const corner = (ixF: FNode, iyF: FNode): FNode =>
    tslHash(uint(wrapF(ixF).add(iyF.mul(8192))), saltU);
  const a = corner(x0, y0);
  const b = corner(x0.add(1), y0);
  const c = corner(x0, y0.add(1));
  const d = corner(x0.add(1), y0.add(1));
  return mix(mix(a, b, su), mix(c, d, su), sv) as unknown as FNode;
}

/** Ramped fbm matching the CPU `wfbm(u, vy, cells, yscale, salt, bakedOct)`: the BAKED octaves are
 *  bit-exact; `DETAIL_OCTAVES` finer octaves are added, their contribution scaled by `detail` (0 at
 *  the baked scale → seamless; growing as the node fills the screen). Normalised by the BAKED norm so
 *  the low frequencies keep their exact baked amplitude. Returns the fbm value in ~[0,1]. */
export function fbmField(
  uu: FNode,
  vy: FNode,
  detail: FNode,
  cells: number,
  yscale: number,
  bakedOct: number,
  saltBase: number,
): FNode {
  const octSalt = (i: number): number => ((saltBase + i * 0x9e37) | 0) >>> 0;
  let p = cells;
  let ys = yscale;
  let amp = 0.5;
  let norm = 0;
  let low = wnoiseTSL(uu, vy.mul(ys), p, octSalt(0)).mul(amp);
  norm += amp;
  for (let i = 1; i < bakedOct; i++) {
    p *= 2;
    ys *= 2;
    amp *= 0.5;
    norm += amp;
    low = low.add(wnoiseTSL(uu, vy.mul(ys), p, octSalt(i)).mul(amp)) as unknown as FNode;
  }
  // detail octaves: continue the doubling past the baked resolution, ramped in by `detail`.
  let detailSum: FNode | null = null;
  for (let i = bakedOct; i < bakedOct + DETAIL_OCTAVES; i++) {
    p *= 2;
    ys *= 2;
    amp *= 0.5;
    const term = wnoiseTSL(uu, vy.mul(ys), p, octSalt(i)).mul(amp);
    detailSum = detailSum ? (detailSum.add(term) as unknown as FNode) : (term as unknown as FNode);
  }
  const summed = detailSum
    ? (low.add(detailSum.mul(detail).mul(DETAIL_GAIN)) as unknown as FNode)
    : low;
  return summed.mul(1 / norm) as unknown as FNode;
}

/** sRGB [0,1] → linear, matching three's SRGBColorSpace decode (the baked DataArrayTexture is sRGB,
 *  so its sampled `.rgb` is already linear; the procedural colour must decode identically to blend
 *  without a brightness seam). */
function srgb2lin(c: FNode): FNode {
  return mix(c.mul(1 / 12.92), pow(c.add(0.055).mul(1 / 1.055), 2.4), step(0.04045, c)) as unknown as FNode;
}

/** A sRGB byte-triple (0..255) → a linear-RGB vec3 node, the colour the baked texel decodes to. */
function rgb255(r: number, g: number, b: number): Node<'vec3'> {
  return vec3(srgb2lin(float(r / 255)), srgb2lin(float(g / 255)), srgb2lin(float(b / 255)));
}

/** The CPU `smooth(a, b, x)` (clamped smoothstep) as TSL, with a/b COMPILE-TIME edges so the edge
 *  order may DESCEND (b < a) — three/WGSL `smoothstep` is only well-defined for ascending edges, and
 *  the planets' ridge/crater masks use descending ones. Bit-faithful to the baked texel at the blend
 *  boundary (same cubic as `core` `smooth`). */
function smoothT(a: number, b: number, x: FNode): FNode {
  const t = clamp(x.sub(a).div(b - a), 0, 1);
  return t.mul(t).mul(float(3).sub(t.mul(2))) as unknown as FNode;
}

/** Blend a surface colour toward a clean constant `poleColour` across the polar cap (`lat` 0 equator
 *  .. 1 pole) — the pole-pinch guard. The pole colour carries no longitude term, so the equirect
 *  singularity has nothing to smear; smoothstep keeps the join to the detailed surface seamless. */
function poleCap(colour: Node<'vec3'>, lat: FNode, poleColour: Node<'vec3'>): Node<'vec3'> {
  return mix(colour, poleColour, smoothstep(POLE_CAP_LO, POLE_CAP_HI, lat)) as unknown as Node<'vec3'>;
}

/* The four procedural planets. Each takes the SHARED sphere uv (`uu` longitude, `v` latitude) and
 * `lat` (|v−0.5|·2) computed once in `buildMaterial`, its seeded `saltBase` (= the baked layer's salt,
 * so the low octaves reproduce the texel) and `detail` (ramps the extra octaves in), and returns
 * linear RGB + emissive A as a vec4 — mirroring the matching `PLANETS.*` baked function so detail=0 is
 * seamless. The longitude is aligned to the BAKED sampling coordinate (SphereGeometry uv.x), which
 * sits +0.25 off `atan(N.x,N.z)` — handled by the caller so continents/craters register exactly. Each
 * blends to a clean POLE COLOUR across the cap (`poleCap`) so the equirect pole does not pinch. */

/** Earth — a hard `land > 0.52` coastline (razor-sharp per fragment vs the baked bilinear blur),
 *  green→grey highlands, whitened to an ice cap at the poles; extra octaves carve finer inlets. */
export function earthProceduralRGBA(
  uu: FNode,
  v: FNode,
  lat: FNode,
  detail: FNode,
  saltBase: number,
): Node<'vec4'> {
  const land = fbmField(uu, v, detail, 5, 3, 5, saltBase); // CPU: wfbm(u,v,5,3,salt,5)
  const ice = smoothstep(0.82, 0.96, lat);
  const hgt = smoothstep(0.52, 0.8, land);
  const landCol = mix(rgb255(70, 120, 55), rgb255(150, 150, 95), hgt);
  const landIced = mix(landCol, rgb255(235, 240, 245), ice);
  const deep = smoothstep(0.2, 0.52, land);
  const oceanCol = mix(rgb255(14, 54, 120), rgb255(40, 110, 165), deep);
  const isLand = step(0.52, land);
  const surf = mix(oceanCol, landIced, isLand) as unknown as Node<'vec3'>;
  return vec4(poleCap(surf, lat, rgb255(235, 240, 245)), 0) as unknown as Node<'vec4'>; // ice-cap pole
}

/** Ice — a pale base veined by a ridged crack network (`|fbm − 0.5|` → thin blue fissures). The
 *  cracks are a hard ridge, so they stay crisp per fragment and branch finer as detail→1. */
function iceProceduralRGBA(
  uu: FNode,
  v: FNode,
  lat: FNode,
  detail: FNode,
  saltBase: number,
): Node<'vec4'> {
  const base = fbmField(uu, v, detail, 6, 3, 4, saltBase); // wfbm(u,v,6,3,salt,4)
  const crackN = abs(fbmField(uu, v, detail, 10, 5, 4, (saltBase + 7) | 0).sub(0.5)).mul(2);
  const crack = smoothT(0.14, 0.0, crackN); // ridged → thin dark-blue fissures
  const w = mix(float(235), float(205), base.mul(0.5)); // sRGB byte
  const r = srgb2lin(mix(w, float(150), crack).mul(1 / 255));
  const g = srgb2lin(mix(w.add(6), float(185), crack).mul(1 / 255));
  const surf = vec3(r, g, float(1)) as unknown as Node<'vec3'>; // blue channel = 255 → linear 1
  return vec4(poleCap(surf, lat, rgb255(228, 234, 255)), 0) as unknown as Node<'vec4'>; // pale-ice pole
}

/** Lava — dark rock veined by GLOWING cracks (a ridged fbm). The crack mask drives BOTH the bright
 *  albedo and the emissive alpha, so the glow stays sharp (not a magnified blur) and branches finer
 *  as detail→1. */
function lavaProceduralRGBA(
  uu: FNode,
  v: FNode,
  lat: FNode,
  detail: FNode,
  saltBase: number,
): Node<'vec4'> {
  const crackN = abs(fbmField(uu, v, detail, 9, 5, 5, saltBase).sub(0.5)).mul(2); // wfbm(u,v,9,5,salt,5)
  const crack = smoothT(0.17, 0.0, crackN); // glowing fissures
  const rock = float(18).add(fbmField(uu, v, detail, 12, 6, 4, (saltBase + 3) | 0).mul(26)); // byte
  const r = srgb2lin(mix(rock, float(255), crack).mul(1 / 255));
  const g = srgb2lin(mix(rock.mul(0.6), float(120), crack).mul(1 / 255));
  const b = srgb2lin(mix(rock.mul(0.6), float(20), crack).mul(1 / 255));
  const surf = vec3(r, g, b) as unknown as Node<'vec3'>;
  // cool dark rock at the pole (no glow): fade both the albedo AND the emissive A across the cap.
  const capped = poleCap(surf, lat, rgb255(30, 18, 18));
  const emiss = crack.mul(220 / 255).mul(float(1).sub(smoothstep(POLE_CAP_LO, POLE_CAP_HI, lat)));
  return vec4(capped, emiss) as unknown as Node<'vec4'>;
}

interface Crater {
  cu: number;
  cv: number;
  rad: number;
}

/** Moon — a grey fbm plus hashed craters (dark floor, bright rim). The crater centres/radii are
 *  compile-time constants (hashed from the layer salt in the factory, EXACTLY as `PLANETS.moon`), so
 *  the loop unrolls to a fixed cost; each rim stays crisp per fragment where the baked texture softens
 *  it. (The `if (d < 1.25)` CPU guard is a pure speed cull — every term is already 0 past 1.25 — so
 *  dropping it here is exact.) */
function moonProceduralRGBA(
  uu: FNode,
  v: FNode,
  lat: FNode,
  detail: FNode,
  saltBase: number,
  craters: Crater[],
): Node<'vec4'> {
  const base = fbmField(uu, v, detail, 7, 4, 5, saltBase); // wfbm(u,v,7,4,salt,5)
  const g = float(120).add(base.sub(0.5).mul(70)).toVar();
  for (const c of craters) {
    const du = abs(uu.sub(c.cu));
    const duw = min(du, float(1).sub(du)); // wrapped-longitude distance
    const dy = v.sub(c.cv);
    const d = sqrt(duw.mul(duw).add(dy.mul(dy))).mul(1 / c.rad);
    const floorT = smoothT(1.0, 0.2, d);
    const rim = smoothT(0.78, 1.02, d).mul(smoothT(1.25, 1.0, d));
    g.addAssign(floorT.mul(-38).add(rim.mul(42)));
  }
  const gc = clamp(g.mul(1 / 255), 0, 1); // sRGB [0,1]
  const surf = vec3(srgb2lin(gc), srgb2lin(gc.mul(0.99)), srgb2lin(gc.mul(0.95))) as unknown as Node<'vec3'>;
  return vec4(poleCap(surf, lat, rgb255(120, 119, 114)), 0) as unknown as Node<'vec4'>; // mean-grey pole
}

/* ---- per-node texture-source overrides (tiered image sources + progressive LOD) --------------
 *
 * Most promoted nodes are procedural planets rendered in the shared instanced pools (the millions-
 * scale path, above — unchanged). A consumer may PIN a specific node id to a different source:
 *   - a forced built-in planet (`procedural`): still an instanced instance, just a fixed texture
 *     layer rather than the facet-derived one — zero extra draw calls.
 *   - a real IMAGE (`image`), e.g. a Blue-Marble Earth: each needs its OWN texture, so it cannot
 *     share one instanced material. Image nodes are drawn as INDIVIDUAL meshes (a small managed set,
 *     cap 16 — overrides are rare) sampling that node's CURRENT best-loaded tier. Draw calls grow by
 *     the number of VISIBLE image nodes (bounded, few).
 *
 * Tiers are consumer-supplied async loaders keyed by resolution: as a node grows on screen the layer
 * requests the highest tier whose `px` fits the node's on-screen px, loads it once (async, never
 * blocking the frame), keeps showing the best already-loaded tier meanwhile, and swaps the material's
 * map when the sharper tier resolves. A small LRU cache (keyed by node+tier, ~64 MB budget) disposes
 * the least-recently-shown GPU textures under pressure, so flying away reclaims a node's high tiers.
 *
 * OPT-IN, like the rest of the file: with no `overrides` NONE of this machinery exists — no image
 * meshes, no loaders, no cache, byte-identical to today, zero added cost. */

/** A per-node texture SOURCE: pin a node to a forced built-in planet, or to a real tiered image. */
export type NodeSource =
  | { kind: 'procedural'; planet: NearTextureName | string }
  | { kind: 'image'; tiers: ImageTier[] };

/** One resolution tier of an image source. `px` is this tier's pixel resolution (the on-screen px at
 *  which it becomes worth loading — bigger on screen ⇒ sharper tier); `load` is a consumer-supplied
 *  async loader resolving a three Texture (e.g. `() => new THREE.TextureLoader().loadAsync(url)` or an
 *  in-memory DataTexture). Called at most ONCE per (node, tier); list tiers in ascending `px`. */
export interface ImageTier {
  px: number;
  load: () => Promise<Texture>;
}

/** Telemetry a consumer may observe (opts.onImageEvent): a tier finished loading, the displayed tier
 *  for a node changed (progressive swap), or an LRU entry was evicted and its GPU texture disposed. */
export interface ImageEvent {
  type: 'load' | 'swap' | 'evict';
  node: number;
  tier: number;
  px: number;
  /** Total estimated image-cache size (bytes) after the event. */
  cacheBytes: number;
}

/** Highest tier index whose `px` is ≤ the node's on-screen px (bigger on screen ⇒ sharper), falling
 *  back to tier 0 when the node is smaller than every tier. Pure — tiers assumed ascending in `px`. */
export function pickImageTier(tiers: ImageTier[], nodePx: number): number {
  let d = 0;
  for (let t = 0; t < tiers.length; t++) if (tiers[t]!.px <= nodePx) d = t;
  return d;
}

/** Hard cap on how many image-override nodes render as individual meshes at once (overrides are rare;
 *  each is its own draw call). */
export const IMAGE_MESH_CAP = 16;
/** Default image-tier LRU memory budget in bytes (~64 MB); estimate per tier is px·px·4 (RGBA8). */
export const IMAGE_CACHE_BUDGET = 64 * 1024 * 1024;

/* ---- the factory --------------------------------------------------------------------------- */

export interface NearMeshOpts {
  /** Performance CAP on how many nodes may be meshes at once (the largest-on-screen win). Never a
   *  "nearest N are always 3D" rule — promotion is by `promotePx`. Default 128. */
  count?: number;
  /** A node promotes to a mesh when its projected glyph radius exceeds this many CSS px (the same
   *  `glyphPx` the billboards and ladder use). ~16–24 reads well; default 18. */
  promotePx?: number;
  /** Optional hard distance cap on promotion, in node-spacings — a belt over the px gate so a
   *  giant hub cannot promote from across the field. Default: no cap (px decides). */
  radius?: number;
  /** Facet name → shape index (value index k → shape k mod 10). null = every node the default
   *  sphere (ONE draw call). */
  shapeFacet?: string | null;
  /** Facet name → texture layer (value index k → texture k mod 10). null = flat tinted shading,
   *  no texture sampled and NONE generated. */
  textureFacet?: string | null;
  /** Opt-in: evaluate a `PROCEDURAL_LAYERS` planet's surface procedurally in the fragment shader when
   *  a node is big on screen, blending baked→procedural for continuous close-up detail (no texture
   *  magnification blur). Default false ⇒ the exact material shipped today, zero cost. Needs a
   *  textureFacet (it refines the hard-edge baked layers). Live via `setProceduralDetail`. */
  proceduralDetail?: boolean;
  /** Opt-in: TRUE displaced terrain for the single SOLID planet you descend onto (prototype: Earth).
   *  When on AND you are close to a promoted earth planet, that one node's instanced sphere is replaced
   *  by a higher-detail icosphere displaced by Earth's own height field — real mountains with silhouettes
   *  you fly over and land on, resolving SEAMLESSLY out of the textured sphere. Needs a textureFacet and
   *  proceduralDetail (it reuses the earth colour so the handoff is invisible). Default false ⇒ the
   *  terrain subsystem is never created, zero cost. Live via `setTerrain`. */
  terrain?: boolean;
  /** Peak terrain displacement as a fraction of planet radius at full descent (default 0.05 —
   *  exaggerated so silhouettes read in the prototype; a DEM would carry true scale). */
  terrainHeightScale?: number;
  /** Finest terrain icosphere subdivision near the ground (default 6 = 81,920 tris). */
  terrainMaxSubdiv?: number;
  /** Override the built-in shape registry (unit-radius geometries, index 0 the default). */
  shapes?: BufferGeometry[];
  /** Override the built-in textures: one DataArrayTexture (RGBA, A = emissive), or a raw
   *  layer-major RGBA Uint8 buffer the factory wraps. */
  textures?: DataArrayTexture | Uint8Array;
  /** Texture seed (deterministic). Default 1. */
  textureSeed?: number;
  /** Viewport height in CSS px (one pixel unit). Keep current via setView. Default 800. */
  heightCss?: number;
  /** Camera vertical FOV in degrees — must match the field's camera. Default 55. */
  fov?: number;
  /** Perspective falloff the field uses for glyph sizing (match the field). Default FALLOFF (1). */
  falloff?: number;
  /** Per-node texture-source overrides: node id → source (forced planet, or a tiered real image).
   *  Absent/empty ⇒ no image meshes, no loaders, no cache — byte-identical to today. Live via
   *  `setOverrides`. */
  overrides?: Record<number, NodeSource>;
  /** Image-tier LRU memory budget in bytes (over which least-recently-shown tiers are disposed).
   *  Default IMAGE_CACHE_BUDGET (~64 MB). */
  imageCacheBudget?: number;
  /** Optional telemetry for image-source nodes: a tier loaded, a progressive tier swap, or an LRU
   *  eviction. Never called when there are no image overrides. */
  onImageEvent?: (e: ImageEvent) => void;
}

export interface NearMesh {
  /** Scene-attachable objects (one Group holding the shape pools) — add every one to your scene. */
  objects: Object3D[];
  /** Per frame: find the promoted set by projected size, place/scale/tint their meshes at their
   *  LIVE positions, and fade their billboards. O(local) — cheap when nothing promotes. */
  update(cameraWorldPos: { x: number; y: number; z: number }, dt: number): void;
  /** The nearest promoted planet's LIVE centre + world radius to `cameraWorldPos`, measured by
   *  SPHERE-surface distance — the body a flying camera would collide with — or null when the LOD
   *  is off (or nothing is promoted). Reflects the most recent `update` (O(promoted), cheap). The
   *  returned object is REUSED each call (mutated in place): read it, do not keep it. Feeds a
   *  consumer camera's surface floor + altitude speed-scale (see flycam `surfaceAt`). */
  nearestSurface(cameraWorldPos: { x: number; y: number; z: number }):
    | {
        center: readonly [number, number, number];
        radius: number;
        /** When the nearest body is the TERRAIN-active solid planet: its effective surface radius
         *  along a unit direction from the centre (sphere radius + displaced terrain height). A
         *  consumer camera (flycam `surfaceAt`) uses this to land ON the terrain, not the bounding
         *  sphere. Absent for a plain sphere body (stage-A behaviour, byte-identical). */
        terrain?: (dx: number, dy: number, dz: number) => number;
      }
    | null;
  /** Turn the subsystem on/off. Off restores every faded billboard and does zero per-frame work;
   *  buffers allocate lazily on the first enabled update. Default: enabled. */
  setEnabled(on: boolean): void;
  /** Report a viewport height / fov change (one pixel unit; devicePixelRatio is the renderer's). */
  setView(heightCss: number, fovDeg: number): void;
  /** Live promotion threshold in CSS px — a node promotes when its projected glyph radius exceeds
   *  this. Larger = promote later (closer); smaller = sooner. */
  setPromotePx(px: number): void;
  /** Rebind the shape facet (null = all default sphere). Live. */
  setShapeFacet(facet: string | null): void;
  /** Rebind the texture facet (null = flat, no sampling). Switching from null generates the
   *  textures on demand (once). Live. */
  setTextureFacet(facet: string | null): void;
  /** Toggle procedural surface detail for the `PROCEDURAL_LAYERS` planets. Rebuilds the material
   *  (like a texture-facet switch); off restores the exact shipped shader. Live. */
  setProceduralDetail(on: boolean): void;
  /** Toggle TRUE displaced terrain for the close solid planet (prototype: Earth). Off hides it and
   *  the node returns to its instanced sphere next frame. Needs a textureFacet + proceduralDetail to
   *  activate (it reuses the earth colour). The terrain subsystem is created lazily on first
   *  activation ⇒ off, or never close to earth, costs nothing. Live. */
  setTerrain(on: boolean): void;
  /** Replace the per-node texture-source overrides live. Passing null/`{}` removes every override
   *  (image nodes return to the instanced procedural pools) and disposes the image-tier cache. */
  setOverrides(map: Record<number, NodeSource> | null): void;
  /** How many nodes are drawn as meshes right now (telemetry). */
  readonly promoted: number;
  /** How many draw calls the near pools add this frame (telemetry; 0 when nothing promotes).
   *  Includes the instanced shape pools AND the individual image meshes visible this frame. */
  readonly drawCalls: number;
  /** How many individual image-override meshes are visible this frame (telemetry; 0 with no image
   *  overrides). */
  readonly imageMeshes: number;
  /** True while the displaced terrain mesh is drawing for the close solid planet (telemetry). */
  readonly terrainActive: boolean;
  /** Current terrain LOD triangle count (0 when inactive) and subdivision level (telemetry). */
  readonly terrainTriangles: number;
  readonly terrainSubdiv: number;
  /** Current estimated image-tier LRU cache size in bytes (telemetry; 0 with no image overrides). */
  readonly imageCacheBytes: number;
  /** The visible image nodes this frame with their active tier index (-1 = loading, nothing shown
   *  yet) and on-screen px — telemetry for the progressive LOD. A fresh array each frame. */
  readonly imageInfo: ReadonlyArray<{ node: number; tier: number; px: number }>;
  dispose(): void;
}

interface Pool {
  shape: number;
  geo: BufferGeometry;
  mesh: InstancedMesh;
  iTint: Float32Array;
  iTexLayer: Float32Array;
  iDetail: Float32Array; // per-instance baked→procedural blend (0 unless procedural detail is on)
  tintAttr: InstancedBufferAttribute;
  texAttr: InstancedBufferAttribute;
  detailAttr: InstancedBufferAttribute;
}

/** Create the near-mesh LOD layer for a field. Opt-in: the consumer calls this (and adds
 *  `.objects` to the scene) only when they want the close look. */
export function createNearMesh(graph: Graph, field: Field, opts: NearMeshOpts = {}): NearMesh {
  const n = graph.n;
  const count = Math.max(1, (opts.count ?? 128) | 0);
  let promotePx = Number.isFinite(opts.promotePx) ? (opts.promotePx as number) : 18;
  const spacing = field.extent.radius / Math.cbrt(Math.max(n, 1)) || 1;
  const radiusCap = Number.isFinite(opts.radius) ? (opts.radius as number) * spacing : Infinity;
  const dRef = (field.extent.diagonal || 1) * D_REF_FRACTION;
  const falloff = Number.isFinite(opts.falloff) ? (opts.falloff as number) : FALLOFF;
  let halfH = (opts.heightCss ?? 800) / 2;
  let tanHalf = Math.tan(((opts.fov ?? 55) * Math.PI) / 360);
  const textureSeed = opts.textureSeed ?? 1;

  let shapeFacet: string | null = opts.shapeFacet ?? null;
  let textureFacet: string | null = opts.textureFacet ?? null;
  let proceduralDetail = opts.proceduralDetail ?? false;
  let terrainEnabled = opts.terrain ?? false;
  const terrainHeightScale = Number.isFinite(opts.terrainHeightScale)
    ? (opts.terrainHeightScale as number)
    : undefined;
  const terrainMaxSubdiv = Number.isFinite(opts.terrainMaxSubdiv)
    ? (opts.terrainMaxSubdiv as number)
    : undefined;

  /* ---- per-node texture-source overrides (opt-in; empty ⇒ zero machinery). ------------------- */
  let overrides: Record<number, NodeSource> = opts.overrides ?? {};
  let hasOverrides = false; // any override at all (procedural or image) — gates the per-node lookup
  let hasImageOverride = false; // any IMAGE override — gates the individual-mesh + loader + cache path
  const imageBudget =
    Number.isFinite(opts.imageCacheBudget) && (opts.imageCacheBudget as number) > 0
      ? (opts.imageCacheBudget as number)
      : IMAGE_CACHE_BUDGET;
  const emitImage = opts.onImageEvent;
  const refreshOverrideFlags = (): void => {
    hasOverrides = false;
    hasImageOverride = false;
    for (const k in overrides) {
      hasOverrides = true;
      if (overrides[k]!.kind === 'image') hasImageOverride = true;
    }
  };
  refreshOverrideFlags();
  // each promoted planet's seeded salt — must equal `generateNearTextures`' per-layer salt so the
  // shader's low octaves reproduce that planet's baked texel exactly (seamless blend).
  const layerSalt = (L: number): number => (textureSeed * 0x1000193 + L * 0x9e3779b1) | 0;
  const L_EARTH = NEAR_TEXTURE_NAMES.indexOf('earth');
  const L_ICE = NEAR_TEXTURE_NAMES.indexOf('ice');
  const L_LAVA = NEAR_TEXTURE_NAMES.indexOf('lava');
  const L_MOON = NEAR_TEXTURE_NAMES.indexOf('moon');
  const earthSalt = layerSalt(L_EARTH);
  const iceSalt = layerSalt(L_ICE);
  const lavaSalt = layerSalt(L_LAVA);
  const moonSalt = layerSalt(L_MOON);
  // moon craters: centres/radii hashed from the moon salt EXACTLY as `PLANETS.moon` does, precomputed
  // once as compile-time shader constants (the TSL loop in `moonProceduralRGBA` unrolls over them).
  const moonCraters: Crater[] = Array.from({ length: 26 }, (_, k) => ({
    cu: hash(k, (moonSalt + 11) | 0),
    cv: 0.1 + hash(k, (moonSalt + 97) | 0) * 0.8,
    rad: 0.02 + hash(k, (moonSalt + 131) | 0) * 0.06,
  }));
  const facetData = (key: string | null): ArrayLike<number> | null =>
    key !== null && graph.facets[key] ? graph.facets[key]!.data : null;
  let shapeCodes = facetData(shapeFacet);
  let texCodes = facetData(textureFacet);

  const shapeGeos: (BufferGeometry | undefined)[] = []; // lazily built shape templates
  const shapeCount = opts.shapes ? opts.shapes.length : NEAR_SHAPE_NAMES.length;
  const texCount = NEAR_TEXTURE_NAMES.length;
  const getShapeGeo = (index: number): BufferGeometry => {
    const k = ((index % shapeCount) + shapeCount) % shapeCount;
    let g = shapeGeos[k];
    if (!g) {
      g = opts.shapes ? opts.shapes[k]! : builtinShape(k);
      shapeGeos[k] = g;
    }
    return g;
  };

  /* ---- the shared, self-lit material (no scene lights) ---------------------------------- */
  let texArray: DataArrayTexture | null = null;
  let ownsTexArray = false;
  const ensureTextures = (): DataArrayTexture => {
    if (texArray) return texArray;
    if (opts.textures instanceof DataArrayTexture) {
      texArray = opts.textures;
      ownsTexArray = false;
    } else {
      texArray = wrapNearTextures(opts.textures ?? generateNearTextures(textureSeed));
      ownsTexArray = true;
    }
    return texArray;
  };

  /** Build the self-lit node material. `textured` splices in one sampler2DArray fetch keyed by a
   *  per-instance layer; otherwise the body is a flat tint. Lighting is a fixed world-space key
   *  light + hemisphere ambient baked into the shader, so the layer needs no scene lights (it
   *  stays self-contained like the rest of the library). */
  const buildMaterial = (textured: boolean): MeshBasicNodeMaterial => {
    const mat = new MeshBasicNodeMaterial();
    mat.transparent = false;
    mat.depthWrite = true;
    mat.depthTest = true;
    // The per-instance transform rides on InstancedMesh.instanceMatrix (a uniform scale + a
    // translation, composed per frame in the fill loop). three applies that matrix to BOTH the clip
    // position and the depth it writes, in every pass — so a promoted body occludes the field
    // billboards, the links and its peers natively, with no custom positionNode in the depth path.
    const tint = attribute('iTint', 'vec3');
    mat.colorNode = Fn(() => {
      // world normal — no per-instance rotation and a uniform scale, so the geometry normal IS
      // the world normal (a fixed world-space light keeps the lit face stable as you orbit).
      const N = normalLocal.normalize();
      const L = vec3(0.4120, 0.7726, 0.4827); // normalize([0.42, 0.79, 0.49])
      const ndl = max(dot(N, L), float(0));
      const hemi = N.y.mul(0.5).add(0.5); // 0 below .. 1 above
      const ambient = float(0.26).add(hemi.mul(0.22));
      const lightAmt = ambient.add(ndl.mul(0.85));
      if (textured) {
        const tex = ensureTextures();
        const layer = int(attribute('iTexLayer', 'float'));
        const sampled = texture(tex, uv()).depth(layer);
        const bodyRgb = sampled.rgb.toVar();
        // PROCEDURAL surface detail. iDetail is per-instance (0 for every non-promoted layer, and 0
        // until a promoted node is big on screen), so this outer If is instance-uniform — whole
        // non-detailed instances skip the asin/atan and the noise entirely. For a detailed instance we
        // route BY TEXTURE LAYER so a fragment runs ONLY its own planet's code, blending the baked
        // texel toward the per-fragment procedural colour (its low octaves equal the texel, so no
        // seam); each planet guards its own poles by blending to a clean pole colour (`poleCap`).
        // Emissive (lava's glow) blends through bodyA identically, so the glow stays sharp too.
        if (proceduralDetail) {
          const bodyA = sampled.a.toVar();
          const detail = attribute('iDetail', 'float');
          If(detail.greaterThan(0.001), () => {
            const uu = atan(N.x, N.z).mul(INV_2PI).add(0.25).fract();
            const vLat = asin(clamp(N.y, -1, 1)).mul(INV_PI).add(0.5);
            const lat = abs(vLat.sub(0.5)).mul(2); // 0 equator .. 1 pole
            const blend = (proc: Node<'vec4'>): void => {
              bodyRgb.assign(mix(sampled.rgb, proc.xyz, detail));
              bodyA.assign(mix(sampled.a, proc.w, detail));
            };
            If(layer.equal(int(L_EARTH)), () => {
              blend(earthProceduralRGBA(uu, vLat, lat, detail, earthSalt));
            })
              .ElseIf(layer.equal(int(L_ICE)), () => {
                blend(iceProceduralRGBA(uu, vLat, lat, detail, iceSalt));
              })
              .ElseIf(layer.equal(int(L_LAVA)), () => {
                blend(lavaProceduralRGBA(uu, vLat, lat, detail, lavaSalt));
              })
              .ElseIf(layer.equal(int(L_MOON)), () => {
                blend(moonProceduralRGBA(uu, vLat, lat, detail, moonSalt, moonCraters));
              });
          });
          // let the texture carry the identity; the node tint only hints (so a planet stays a planet)
          const albedo = bodyRgb.mul(mix(vec3(1, 1, 1), tint, float(0.25)));
          const emissive = bodyRgb.mul(bodyA).mul(1.7);
          const outc = albedo.mul(lightAmt).add(emissive);
          return vec4(outc, 1) as unknown as ReturnType<typeof vec4>;
        }
        // DEFAULT (baked): the exact shader shipped today — references no iDetail, uploads nothing.
        const albedo = bodyRgb.mul(mix(vec3(1, 1, 1), tint, float(0.25)));
        const emissive = sampled.rgb.mul(sampled.a).mul(1.7);
        const outc = albedo.mul(lightAmt).add(emissive);
        return vec4(outc, 1) as unknown as ReturnType<typeof vec4>;
      }
      return vec4(tint.mul(lightAmt), 1) as unknown as ReturnType<typeof vec4>;
    })() as unknown as typeof mat.colorNode;
    return mat;
  };

  let textured = textureFacet !== null;
  let material = buildMaterial(textured);

  /* ---- the pools (lazy; sphere-only ⇒ one pool ⇒ one draw call) -------------------------- */
  const group = new Group();
  group.renderOrder = 2;
  const pools = new Map<number, Pool>();
  const instMat = new Matrix4(); // scratch: composed per instance in the fill loop
  const makePool = (shape: number): Pool => {
    // a unit-radius shape (cloned so this pool owns its per-instance attribute buffers); the
    // placement is InstancedMesh.instanceMatrix, so the base geometry stays at the origin.
    const geo = getShapeGeo(shape).clone();
    const iTint = new Float32Array(count * 3);
    const iTexLayer = new Float32Array(count);
    const iDetail = new Float32Array(count); // 0 unless procedural detail is on (then set per frame)
    const tintAttr = new InstancedBufferAttribute(iTint, 3);
    const texAttr = new InstancedBufferAttribute(iTexLayer, 1);
    const detailAttr = new InstancedBufferAttribute(iDetail, 1);
    tintAttr.setUsage(DynamicDrawUsage);
    texAttr.setUsage(DynamicDrawUsage);
    detailAttr.setUsage(DynamicDrawUsage);
    geo.setAttribute('iTint', tintAttr);
    geo.setAttribute('iTexLayer', texAttr);
    // iDetail is referenced by the shader ONLY when proceduralDetail is on; off ⇒ the attribute is
    // never bound/uploaded (the default material does not read it), so the default path stays free.
    geo.setAttribute('iDetail', detailAttr);
    const mesh = new InstancedMesh(geo, material, count);
    mesh.count = 0;
    mesh.instanceMatrix.setUsage(DynamicDrawUsage); // rewritten every active frame
    mesh.frustumCulled = false;
    mesh.renderOrder = 2;
    const pool: Pool = { shape, geo, mesh, iTint, iTexLayer, iDetail, tintAttr, texAttr, detailAttr };
    pools.set(shape, pool);
    group.add(mesh);
    return pool;
  };

  /* ---- the field's per-node colour + billboard-alpha buffers (read/written through the public
   * node mesh — the same arrays the shader samples, so a tint can never disagree with the field
   * and a faded billboard is exactly the node that became a mesh). */
  const nodeMesh = field.objects[1] as unknown as Mesh;
  const iColorAttr = nodeMesh.geometry.getAttribute('iColor');
  const iAlphaAttr = nodeMesh.geometry.getAttribute('iAlpha');
  if (!iColorAttr || !iAlphaAttr) {
    throw new Error('createNearMesh: the field node mesh is missing iColor / iAlpha attributes');
  }
  const colors = iColorAttr.array as Float32Array;
  const alpha = iAlphaAttr.array as Float32Array;
  const alphaAttr = iAlphaAttr as unknown as {
    needsUpdate: boolean;
    addUpdateRange(start: number, count: number): void;
    clearUpdateRanges(): void;
  };

  /* ---- live-position machinery (O(local) per frame) ------------------------------------- */
  let base: Float32Array | null = null; // mix(A,B,t) — base endpoints, refreshed on morph/rebuild
  let viewSnap: Float32Array | null = null; // view positions snapshot the grid buckets
  let grid: Grid | null = null;
  let cellMaxSize: Float32Array | undefined; // per-cell largest world radius (prunes far cells whole)
  let lastMorphT = -1;
  let builtOnce = false;
  let wasSettled = false;
  let sinceBuild = 0;
  const REBUILD_THROTTLE = 8; // frames between grid rebuilds while the field is in motion

  const ensureBuffers = (): void => {
    if (!base) base = new Float32Array(n * 3);
    if (!viewSnap) viewSnap = new Float32Array(n * 3);
  };
  const rebuildGrid = (): void => {
    ensureBuffers();
    field.viewPositions(viewSnap!);
    grid = buildGrid(viewSnap!, n);
    // per-cell largest world radius (field size × NODE_SCALE): a cell whose biggest node cannot
    // project over promotePx at the cell's nearest point holds no candidate — skipped whole.
    cellMaxSize = cellMaxOf(grid, (i) => field.sizeOf(i) * NODE_SCALE, cellMaxSize);
    field.basePositions(base!);
    lastMorphT = field.morphT;
    sinceBuild = 0;
    builtOnce = true;
  };

  /** node i's LIVE world position (base → chain → spread → field-spin), the CPU mirror of the
   *  vertex shader — O(1), the handoff anchor so a mesh sits exactly on its faded billboard. */
  const spreadC = new Vector3();
  const fieldQ = new Float32Array(4);
  let spreadK = 1;
  let fieldDisplaced = false;
  const livePos = (i: number, out: Float32Array, o: number): void => {
    const b = base!;
    const f = field.frames;
    if (f !== null) {
      f.transformPoint(field.chainIds[i]!, b, out, i * 3, o);
    } else {
      out[o] = b[i * 3]!;
      out[o + 1] = b[i * 3 + 1]!;
      out[o + 2] = b[i * 3 + 2]!;
    }
    if (spreadK !== 1) {
      out[o] = spreadC.x + (out[o]! - spreadC.x) * spreadK;
      out[o + 1] = spreadC.y + (out[o + 1]! - spreadC.y) * spreadK;
      out[o + 2] = spreadC.z + (out[o + 2]! - spreadC.z) * spreadK;
    }
    if (fieldDisplaced) {
      const qx = fieldQ[0]!, qy = fieldQ[1]!, qz = fieldQ[2]!, qw = fieldQ[3]!;
      const x = out[o]! - spreadC.x;
      const y = out[o + 1]! - spreadC.y;
      const z = out[o + 2]! - spreadC.z;
      const tx = 2 * (qy * z - qz * y);
      const ty = 2 * (qz * x - qx * z);
      const tz = 2 * (qx * y - qy * x);
      out[o] = spreadC.x + x + qw * tx + (qy * tz - qz * ty);
      out[o + 1] = spreadC.y + y + qw * ty + (qz * tx - qx * tz);
      out[o + 2] = spreadC.z + z + qw * tz + (qx * ty - qy * tx);
    }
  };

  /* ---- faded-billboard bookkeeping: node → the iAlpha value it had when it promoted, so leaving
   * the near set restores exactly what the field intended (focus dim level and all). */
  const faded = new Map<number, number>();
  const restoreAll = (): void => {
    if (faded.size === 0) return;
    for (const [i, a] of faded) {
      alpha[i] = a;
      alphaAttr.addUpdateRange(i, 1);
    }
    faded.clear();
    alphaAttr.needsUpdate = true;
  };

  /* ---- candidate scratch (bounded by `count` after the px cull; the pre-cull list is bounded by
   * the cells within d_max, never the corpus). */
  let candIdx = new Int32Array(0);
  let candPx = new Float32Array(0);
  let candPos = new Float32Array(0);
  let candCap = 0;
  const growCand = (need: number): void => {
    if (need <= candCap) return;
    candCap = Math.max(need, candCap * 2, 256);
    candIdx = new Int32Array(candCap);
    candPx = new Float32Array(candCap);
    candPos = new Float32Array(candCap * 3);
  };

  /* ---- nearest-surface query (for a flying camera; see `nearestSurface`) ---------------------
   * Each ENABLED update records the promoted planets' LIVE centres + world radii this frame (the
   * same values that place their meshes, so they track spin/morph exactly). `nearestSurface` then
   * returns the nearest of them by SURFACE distance — the sphere a flycam would collide with — in
   * O(promoted). Zero cost when nothing promotes (surfCount stays 0) and when the LOD is off. */
  let surfCenters = new Float32Array(0);
  let surfRadii = new Float32Array(0);
  let surfCount = 0;
  const ensureSurf = (cap: number): void => {
    if (surfRadii.length >= cap) return;
    const c = Math.max(cap, count);
    surfCenters = new Float32Array(c * 3);
    surfRadii = new Float32Array(c);
  };
  // reused result object (mutated each call) so the per-frame query allocates nothing.
  const surfOut: {
    center: [number, number, number];
    radius: number;
    terrain?: (dx: number, dy: number, dz: number) => number;
  } = { center: [0, 0, 0], radius: 0 };

  const worldPx = (sizeWorld: number, dist: number): number =>
    glyphPx(sizeWorld, dist, halfH, tanHalf, dRef, falloff);

  /* ---- individual image-override meshes + progressive tier LOD + LRU cache -------------------
   * Everything here is created LAZILY on the first image-override promotion; with no image override
   * NONE of it is allocated (hasImageOverride gates every entry point). An image node cannot share
   * the instanced material (each needs its own texture), so it is drawn as its own unit-sphere Mesh
   * with a self-lit textured material (the SAME fixed key light + hemisphere ambient as the pools)
   * whose sampled texture is swapped to the node's current best-loaded tier. */
  const imgCap = Math.min(IMAGE_MESH_CAP, count);
  let imgIds: Int32Array | null = null; // promoted image nodes this frame (ids / px / world pos)
  let imgPx: Float32Array | null = null;
  let imgPos: Float32Array | null = null;
  const ensureImageScratch = (): void => {
    if (imgIds) return;
    imgIds = new Int32Array(imgCap);
    imgPx = new Float32Array(imgCap);
    imgPos = new Float32Array(imgCap * 3);
  };

  const planetLayer = (name: string): number => NEAR_TEXTURE_NAMES.indexOf(name as NearTextureName);

  interface ImageSlot {
    mesh: Mesh;
    texNode: { value: Texture }; // the TextureNode the material samples — swap .value to change tier
    node: number; // which node id this slot shows (-1 = free)
    shownKey: string; // the (node:tier) currently bound, '' = nothing shown yet
  }
  interface CacheEntry {
    tex: Texture;
    bytes: number;
    px: number;
  }
  let imageGeo: BufferGeometry | null = null;
  let placeholderTex: DataTexture | null = null;
  let imageSlots: ImageSlot[] | null = null;
  const imageMaterials: MeshBasicNodeMaterial[] = [];
  // LRU cache: a Map is insertion-ordered, so the FIRST key is the least-recently-shown (we re-set a
  // key on show to move it to the end). Keyed by `${node}:${tier}`.
  const imgCache = new Map<string, CacheEntry>();
  let cacheBytes = 0;
  const inFlight = new Set<string>();
  const currentShown = new Set<string>(); // keys displayed on the last frame (protected from eviction)
  let imageGen = 0; // bumped on setOverrides/dispose — in-flight loads from an old gen are discarded
  let imageInfoArr: { node: number; tier: number; px: number }[] = [];
  let visibleImages = 0;
  const keyOf = (node: number, tier: number): string => `${node}:${tier}`;

  /** The self-lit textured material for an image slot: same lighting as the pools, but sampling a
   *  per-slot swappable 2D texture (the node's current tier) rather than the shared planet array. */
  const buildImageMaterial = (): ImageSlot['texNode'] => {
    if (!placeholderTex) {
      // a 1×1 opaque placeholder so the TextureNode has a value before the first tier resolves.
      placeholderTex = new DataTexture(new Uint8Array([40, 40, 48, 255]), 1, 1);
      placeholderTex.colorSpace = SRGBColorSpace;
      placeholderTex.needsUpdate = true;
    }
    const mat = new MeshBasicNodeMaterial();
    mat.transparent = false;
    mat.depthWrite = true;
    mat.depthTest = true;
    const texNode = texture(placeholderTex, uv());
    mat.colorNode = Fn(() => {
      const N = normalLocal.normalize();
      const L = vec3(0.412, 0.7726, 0.4827); // same fixed key light as the pools
      const ndl = max(dot(N, L), float(0));
      const hemi = N.y.mul(0.5).add(0.5);
      const ambient = float(0.26).add(hemi.mul(0.22));
      const lightAmt = ambient.add(ndl.mul(0.85));
      return vec4(texNode.rgb.mul(lightAmt), 1) as unknown as ReturnType<typeof vec4>;
    })() as unknown as typeof mat.colorNode;
    imageMaterials.push(mat);
    const slot: ImageSlot = {
      mesh: new Mesh(imageGeo!, mat),
      texNode: texNode as unknown as ImageSlot['texNode'],
      node: -1,
      shownKey: '',
    };
    slot.mesh.visible = false;
    slot.mesh.frustumCulled = false; // positioned every frame; a bounding sphere at origin would cull
    slot.mesh.renderOrder = 2;
    group.add(slot.mesh);
    imageSlots!.push(slot);
    return slot.texNode;
  };

  const ensureImageInfra = (): void => {
    if (imageSlots) return;
    imageGeo = builtinShape(0); // one shared unit sphere (equirect uv) — slots share geometry
    imageSlots = [];
    for (let s = 0; s < imgCap; s++) buildImageMaterial();
  };

  const touch = (key: string): void => {
    const e = imgCache.get(key);
    if (!e) return;
    imgCache.delete(key);
    imgCache.set(key, e); // move to most-recent (end)
  };
  const evictToBudget = (): void => {
    if (cacheBytes <= imageBudget) return;
    for (const [key, e] of imgCache) {
      if (cacheBytes <= imageBudget) break;
      if (currentShown.has(key)) continue; // never evict a texture on screen this frame
      imgCache.delete(key);
      cacheBytes -= e.bytes;
      e.tex.dispose();
      const colon = key.indexOf(':');
      emitImage?.({
        type: 'evict',
        node: Number(key.slice(0, colon)),
        tier: Number(key.slice(colon + 1)),
        px: e.px,
        cacheBytes,
      });
    }
  };
  const kickLoad = (node: number, tierIdx: number, tier: ImageTier): void => {
    const key = keyOf(node, tierIdx);
    if (imgCache.has(key) || inFlight.has(key)) return;
    inFlight.add(key);
    const gen = imageGen;
    tier
      .load()
      .then((tex) => {
        inFlight.delete(key);
        if (gen !== imageGen) {
          tex.dispose(); // overrides changed while loading — this tier is stale
          return;
        }
        tex.colorSpace = SRGBColorSpace;
        tex.wrapS = RepeatWrapping; // longitude wraps (equirect)
        tex.minFilter = LinearFilter;
        tex.magFilter = LinearFilter;
        tex.generateMipmaps = false;
        tex.needsUpdate = true;
        const bytes = Math.max(1, (tier.px | 0) * (tier.px | 0) * 4);
        imgCache.set(key, { tex, bytes, px: tier.px });
        cacheBytes += bytes;
        emitImage?.({ type: 'load', node, tier: tierIdx, px: tier.px, cacheBytes });
        evictToBudget();
      })
      .catch(() => {
        inFlight.delete(key); // allow a later retry
      });
  };
  const bestLoaded = (node: number, tierCount: number, desired: number): number => {
    for (let t = desired; t >= 0; t--) if (imgCache.has(keyOf(node, t))) return t; // best ≤ desired
    for (let t = desired + 1; t < tierCount; t++) if (imgCache.has(keyOf(node, t))) return t; // else any
    return -1;
  };
  const slotOf = (node: number): ImageSlot | null => {
    for (const s of imageSlots!) if (s.node === node) return s;
    return null;
  };
  const acquireSlot = (node: number): ImageSlot | null => {
    for (const s of imageSlots!) {
      if (s.node < 0) {
        s.node = node;
        s.shownKey = '';
        return s;
      }
    }
    return null; // all slots busy (cap reached) — this node stays a billboard
  };
  const hideAllImages = (): void => {
    if (!imageSlots) return;
    for (const s of imageSlots) {
      s.mesh.visible = false;
      s.node = -1;
      s.shownKey = '';
    }
    currentShown.clear();
    visibleImages = 0;
    imageInfoArr = [];
  };

  /** Place/scale/texture the promoted image nodes as individual meshes; decide which fade their
   *  billboard (only once a texture is actually on screen). Returns nothing; mutates nextFaded. */
  const updateImages = (cnt: number, nextFaded: Set<number>): void => {
    ensureImageInfra();
    const slots = imageSlots!;
    const ids = imgIds!;
    const pxs = imgPx!;
    const pos = imgPos!;
    const active = new Set<number>();
    currentShown.clear();
    imageInfoArr = [];
    visibleImages = 0;
    for (let k = 0; k < cnt; k++) {
      const node = ids[k]!;
      const src = overrides[node];
      if (!src || src.kind !== 'image' || src.tiers.length === 0) continue;
      active.add(node);
      let slot = slotOf(node);
      if (!slot) slot = acquireSlot(node);
      if (!slot) continue; // no free slot → stays a billboard
      const tiers = src.tiers;
      const px = pxs[k]!;
      // place at the node's live world position, scaled to its world radius (same as the pools).
      const r = field.sizeOf(node) * NODE_SCALE;
      slot.mesh.position.set(pos[k * 3]!, pos[k * 3 + 1]!, pos[k * 3 + 2]!);
      slot.mesh.scale.setScalar(r);
      // choose the sharpest tier that fits the on-screen px; always ensure the base tier so SOMETHING
      // can show. Clamp down past any tier whose own bytes exceed the whole budget — it could never be
      // retained, so requesting it would thrash (load → evict → load …) forever.
      let desired = pickImageTier(tiers, px);
      while (desired > 0) {
        const tpx = tiers[desired]!.px | 0;
        if (tpx * tpx * 4 <= imageBudget) break;
        desired--;
      }
      kickLoad(node, 0, tiers[0]!);
      if (desired > 0) kickLoad(node, desired, tiers[desired]!);
      const best = bestLoaded(node, tiers.length, desired);
      if (best >= 0) {
        const key = keyOf(node, best);
        const entry = imgCache.get(key)!;
        touch(key);
        currentShown.add(key);
        if (slot.shownKey !== key) {
          slot.texNode.value = entry.tex; // swap the sampled texture (no material rebuild)
          slot.shownKey = key;
          emitImage?.({ type: 'swap', node, tier: best, px: tiers[best]!.px, cacheBytes });
        }
        slot.mesh.visible = true;
        visibleImages++;
        nextFaded.add(node); // hand off: fade the billboard only when the image is actually up
        imageInfoArr.push({ node, tier: best, px });
      } else {
        slot.mesh.visible = false; // nothing loaded yet → keep the billboard (no hole)
        imageInfoArr.push({ node, tier: -1, px });
      }
      // reclaim this node's REDUNDANT tiers — keep only the base (instant fallback), the tier on
      // screen (best) and the tier being loaded toward (desired). This bounds the per-node working set
      // so even a budget smaller than two adjacent tiers cannot pile intermediates up and thrash, and
      // it drops a node's high tiers back toward the base as it shrinks / flies away (desired falls, so
      // the sharp tiers become redundant and are released — the LRU budget is the cross-node backstop).
      for (let t = 1; t < tiers.length; t++) {
        if (t === desired || t === best) continue;
        const rk = keyOf(node, t);
        const e = imgCache.get(rk);
        if (e && !currentShown.has(rk)) {
          imgCache.delete(rk);
          cacheBytes -= e.bytes;
          e.tex.dispose();
          emitImage?.({ type: 'evict', node, tier: t, px: e.px, cacheBytes });
        }
      }
    }
    // release slots whose node is no longer a promoted image node
    for (const s of slots) {
      if (s.node >= 0 && !active.has(s.node)) {
        s.mesh.visible = false;
        s.node = -1;
        s.shownKey = '';
      }
    }
    evictToBudget();
  };

  const disposeImages = (): void => {
    imageGen++;
    for (const e of imgCache.values()) e.tex.dispose();
    imgCache.clear();
    cacheBytes = 0;
    inFlight.clear();
    currentShown.clear();
    hideAllImages();
  };

  /* ---- TRUE displaced terrain for the close solid planet (opt-in; lazy; see terrain.ts) ----------
   * Only the single nearest promoted EARTH planet, when you are close enough (altitude < TERRAIN_ALT_ON),
   * gets real displaced geometry; its instanced sphere is suppressed so the two never double-draw, and
   * the handoff is seamless (amplitude ramps from 0 at the swap altitude). Everything is created lazily
   * on the first activation, so the feature off — or never close to earth — costs nothing. */
  let terrain: Terrain | null = null;
  let heightSrc: TerrainHeightSource | null = null;
  let terrainActive = false; // drawing this frame (gates the collision sampler + telemetry)
  let terrainSurfIdx = -1; // which `surfCenters`/`surfRadii` slot is the terrain planet
  let terrainRadius = 0;
  let terrainAmpFrac = 0; // displacement fraction in play (matches the GPU uniform) — collision height
  let terrainDetail = 0;

  /** The earth colour drape for the terrain mesh: byte-for-byte the instanced earth shading (baked
   *  texel ↔ procedural blend by `detail`, pole cap, node tint, the fixed key light) but lit by the
   *  DISPLACED terrain normal — so at amplitude 0 (the swap) it equals the instanced sphere exactly,
   *  and as the relief resolves in only the lighting bends. UV + baked layer come from the ORIGINAL
   *  (undisplaced) direction so continents register with the coastline. */
  const buildTerrainColor = (): TerrainColorNode => {
    const texArr = ensureTextures();
    return (d, normal, detail, tint) => {
      const N = normal.normalize();
      const Lk = vec3(0.412, 0.7726, 0.4827);
      const ndl = max(dot(N, Lk), float(0));
      const hemi = N.y.mul(0.5).add(0.5);
      const ambient = float(0.26).add(hemi.mul(0.22));
      const lightAmt = ambient.add(ndl.mul(0.85));
      const uu = atan(d.x, d.z).mul(INV_2PI).add(0.25).fract();
      const vLat = asin(clamp(d.y, -1, 1)).mul(INV_PI).add(0.5);
      const lat = abs(vLat.sub(0.5)).mul(2);
      const sampled = texture(texArr, vec2(uu, vLat)).depth(int(L_EARTH));
      const proc = earthProceduralRGBA(uu as FNode, vLat as FNode, lat as FNode, detail, earthSalt);
      const bodyRgb = mix(sampled.rgb, proc.xyz, detail);
      const bodyA = mix(sampled.a, proc.w, detail);
      const albedo = bodyRgb.mul(mix(vec3(1, 1, 1), tint, float(0.25)));
      const emissive = bodyRgb.mul(bodyA).mul(1.7);
      return vec4(albedo.mul(lightAmt).add(emissive), 1) as unknown as ReturnType<typeof vec4>;
    };
  };

  const ensureTerrain = (): Terrain => {
    if (terrain) return terrain;
    heightSrc = earthTerrainSource(earthSalt);
    terrain = createTerrain({
      height: heightSrc,
      colorNode: buildTerrainColor(),
      heightScale: terrainHeightScale,
      maxSubdiv: terrainMaxSubdiv,
    });
    group.add(terrain.object);
    return terrain;
  };

  /** Effective surface radius along a unit direction = sphere radius + displaced terrain height. Fed
   *  to the flycam (via `nearestSurface().terrain`) so it lands on the relief, not the bounding sphere.
   *  Uses the SAME height field + amplitude the GPU displaced with this frame. */
  const terrainSample = (dx: number, dy: number, dz: number): number => {
    if (!heightSrc) return terrainRadius;
    const len = Math.hypot(dx, dy, dz) || 1;
    const h = heightSrc.height(dx / len, dy / len, dz / len, terrainDetail);
    return terrainRadius * (1 + h * terrainAmpFrac);
  };

  let enabled = true;
  let promotedCount = 0;

  // scratch for the promoted set (indices into cand), reused
  let order = new Int32Array(0);

  const clearPools = (): void => {
    for (const p of pools.values()) p.mesh.count = 0;
  };

  const update = (cam: { x: number; y: number; z: number }, _dt: number): void => {
    if (!enabled) return;
    ensureBuffers();
    ensureSurf(count);
    surfCount = 0; // rebuilt below from the promoted set; 0 ⇒ nearestSurface returns null
    terrainSurfIdx = -1; // which surface slot (if any) is the terrain planet this frame

    // keep the grid fresh: build once, on every settle edge, and throttled while in motion.
    const settled = field.settled();
    if (!builtOnce) rebuildGrid();
    else if (settled && !wasSettled) rebuildGrid();
    else if (!settled && ++sinceBuild >= REBUILD_THROTTLE) rebuildGrid();
    wasSettled = settled;

    // base endpoints change only during a morph — refresh them then (and they were refreshed on
    // the last grid build); a steady spin keeps base static and applies the chain live per node.
    if (field.morphT !== lastMorphT) {
      field.basePositions(base!);
      lastMorphT = field.morphT;
    }

    // live transform state for livePos (mirror the shader's spread + field-spin stages).
    const c = field.centroid();
    spreadC.set(c[0], c[1], c[2]);
    spreadK = field.spread;
    field.fieldSpinQuat(fieldQ);
    fieldDisplaced = field.fieldSpinDisplaced();

    const g = grid!;
    // how far the LARGEST possible node can still promote — the grid-walk bound (never the corpus).
    const maxSizeWorld = SIZE_MAX_WORLD * NODE_SCALE;
    const baseBig = (maxSizeWorld * halfH) / (tanHalf * dRef);
    let dMax: number;
    if (baseBig <= promotePx) {
      dMax = dRef; // even at the pivot it is under threshold — only the nearest cells matter
    } else {
      dMax = dRef * Math.pow(baseBig / promotePx, 1 / Math.max(falloff, 1e-3));
    }
    if (Number.isFinite(radiusCap)) dMax = Math.min(dMax, radiusCap);
    if (!Number.isFinite(dMax) || dMax <= 0) dMax = g.cellMax;

    // walk the grid cells within dMax of the camera (snapshot bucketing; live px per node).
    growCand(count * 4);
    let m = 0;
    const px0 = cam.x, py0 = cam.y, pz0 = cam.z;
    const rr = Math.ceil(dMax / g.cellMax) + 1;
    const clampCell = (q: number): number => (q < 0 ? 0 : q >= g.side ? g.side - 1 : q);
    const ix = clampCell(((px0 - g.minX) / g.cellX) | 0);
    const iy = clampCell(((py0 - g.minY) / g.cellY) | 0);
    const iz = clampCell(((pz0 - g.minZ) / g.cellZ) | 0);
    const dMax2 = dMax * dMax;
    const aLo = Math.max(-rr, -iz), aHi = Math.min(rr, g.side - 1 - iz);
    const bLo = Math.max(-rr, -iy), bHi = Math.min(rr, g.side - 1 - iy);
    const cLo = Math.max(-rr, -ix), cHi = Math.min(rr, g.side - 1 - ix);
    const posView = g.positions;
    for (let a = aLo; a <= aHi; a++) {
      const zz = iz + a;
      for (let b = bLo; b <= bHi; b++) {
        const yy = iy + b;
        for (let cc = cLo; cc <= cHi; cc++) {
          const xx = ix + cc;
          const k = (zz * g.side + yy) * g.side + xx;
          const lo = g.start[k]!;
          const hi = g.start[k + 1]!;
          if (hi === lo) continue;
          // whole-cell prune: the largest node here, at the cell's NEAREST point to the camera,
          // projects under promotePx ⇒ nothing in the cell can promote (exact, never over-prunes).
          const cx0 = g.minX + xx * g.cellX;
          const cy0 = g.minY + yy * g.cellY;
          const cz0 = g.minZ + zz * g.cellZ;
          const ddx = px0 < cx0 ? cx0 - px0 : px0 > cx0 + g.cellX ? px0 - (cx0 + g.cellX) : 0;
          const ddy = py0 < cy0 ? cy0 - py0 : py0 > cy0 + g.cellY ? py0 - (cy0 + g.cellY) : 0;
          const ddz = pz0 < cz0 ? cz0 - pz0 : pz0 > cz0 + g.cellZ ? pz0 - (cz0 + g.cellZ) : 0;
          const near = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
          if (worldPx(cellMaxSize![k]!, near) < promotePx) continue;
          for (let q = lo; q < hi; q++) {
            const i = g.order[q]!;
            // snapshot reject (cheap) using the bucketed view position, generously padded…
            const sx = posView[i * 3]! - px0;
            const sy = posView[i * 3 + 1]! - py0;
            const sz = posView[i * 3 + 2]! - pz0;
            if (sx * sx + sy * sy + sz * sz > (dMax + g.cellMax) * (dMax + g.cellMax)) continue;
            // …then the LIVE position decides px (handoff-exact under spin/morph).
            if (m >= candCap) growCand(candCap + 1);
            const o = m * 3;
            livePos(i, candPos, o);
            const dx = candPos[o]! - px0;
            const dy = candPos[o + 1]! - py0;
            const dz = candPos[o + 2]! - pz0;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > dMax2) continue;
            const dist = Math.sqrt(d2);
            const sizeWorld = field.sizeOf(i) * NODE_SCALE;
            const px = worldPx(sizeWorld, dist);
            if (px < promotePx) continue;
            candIdx[m] = i;
            candPx[m] = px;
            m++;
          }
        }
      }
    }

    if (m === 0) {
      clearPools();
      restoreAll();
      if (imageSlots) hideAllImages();
      if (terrain) terrain.setVisible(false);
      terrainActive = false;
      promotedCount = 0;
      return;
    }

    // keep the largest-on-screen `count` (the perf cap). Partial order by px desc.
    if (order.length < m) order = new Int32Array(Math.max(m, order.length * 2, count));
    for (let j = 0; j < m; j++) order[j] = j;
    const take = Math.min(m, count);
    // simple selection of the top `take` by px (m is bounded by local cells; take ≤ count)
    for (let s = 0; s < take; s++) {
      let bestJ = s;
      for (let j = s + 1; j < m; j++) {
        if (candPx[order[j]!]! > candPx[order[bestJ]!]!) bestJ = j;
      }
      const t = order[s]!;
      order[s] = order[bestJ]!;
      order[bestJ] = t;
    }

    /* TERRAIN pre-scan: pick the single nearest promoted EARTH planet the camera is close enough to
     * descend onto (altitude < TERRAIN_ALT_ON). That one node's instanced sphere is suppressed in the
     * fill loop and drawn instead by the displaced terrain mesh. Gated on the master switch AND a
     * textured + procedural-detail material (the drape reuses the exact earth colour, so the handoff is
     * seamless); off / not-close ⇒ terrainNode stays −1 and nothing changes. */
    let terrainNode = -1;
    let terrainJ = -1;
    let terrainAltFrac = 0;
    if (terrainEnabled && textured && proceduralDetail) {
      // pick the earth planet you are DESCENDING ONTO: the smallest altitude RELATIVE to its radius
      // (altitude / radius), not the smallest absolute distance — otherwise a tiny earth that happens
      // to drift close in world units would steal terrain from the big planet filling your view.
      let bestAlt = Infinity;
      for (let s = 0; s < take; s++) {
        const j = order[s]!;
        const i = candIdx[j]!;
        let layer = texCodes ? texCodes[i]! % texCount : 0;
        if (hasOverrides) {
          const src = overrides[i];
          if (src && src.kind === 'image') continue; // image nodes are not terrain
          if (src && src.kind === 'procedural') {
            const pl = planetLayer(src.planet);
            if (pl >= 0) layer = pl;
          }
        }
        if (layer !== L_EARTH) continue; // prototype: solid terrain is Earth only
        const p3 = j * 3;
        const dx = candPos[p3]! - cam.x;
        const dy = candPos[p3 + 1]! - cam.y;
        const dz = candPos[p3 + 2]! - cam.z;
        const r = field.sizeOf(i) * NODE_SCALE;
        const altF = (Math.sqrt(dx * dx + dy * dy + dz * dz) - r) / r; // altitude in radius units
        if (altF < bestAlt) {
          bestAlt = altF;
          terrainNode = i;
          terrainJ = j;
        }
      }
      if (terrainNode >= 0) {
        terrainAltFrac = bestAlt;
        if (!(bestAlt < TERRAIN_ALT_ON)) {
          terrainNode = -1; // promoted but still too high — keep the instanced sphere
          terrainJ = -1;
        }
      }
    }

    // reset pool fills
    const fill = new Map<number, number>();
    for (const p of pools.values()) fill.set(p.shape, 0);

    // the new promoted set (to diff against `faded`)
    const nextFaded = new Set<number>();
    if (hasImageOverride) ensureImageScratch();
    let imgN = 0; // promoted IMAGE-override nodes, collected for the individual-mesh manager
    for (let s = 0; s < take; s++) {
      const j = order[s]!;
      const i = candIdx[j]!;
      const p3 = j * 3;
      // record this promoted body as a solid sphere for nearestSurface (both instanced planets and
      // image-override planets are collidable ground); the live centre is candPos, already computed.
      if (surfCount < surfRadii.length) {
        const s3 = surfCount * 3;
        surfCenters[s3] = candPos[p3]!;
        surfCenters[s3 + 1] = candPos[p3 + 1]!;
        surfCenters[s3 + 2] = candPos[p3 + 2]!;
        surfRadii[surfCount] = field.sizeOf(i) * NODE_SCALE;
        surfCount++;
        // the terrain planet is still a collidable surface — but drawn by the displaced terrain mesh,
        // not the instanced pool. Remember its surface slot (so nearestSurface attaches the terrain
        // sampler) and skip its instanced instance so the two never double-draw.
        if (i === terrainNode) {
          terrainSurfIdx = surfCount - 1;
          nextFaded.add(i); // promoted (drawn as terrain) ⇒ fade its billboard
          continue;
        }
      }
      const src = hasOverrides ? (overrides[i] ?? null) : null;
      if (src && src.kind === 'image') {
        // an image-override node is NOT an instanced instance — it becomes its own textured mesh.
        // Its billboard is faded later by updateImages, only once a tier is actually on screen.
        if (hasImageOverride && imgN < imgCap && src.tiers.length > 0) {
          imgIds![imgN] = i;
          imgPx![imgN] = candPx[j]!;
          imgPos![imgN * 3] = candPos[p3]!;
          imgPos![imgN * 3 + 1] = candPos[p3 + 1]!;
          imgPos![imgN * 3 + 2] = candPos[p3 + 2]!;
          imgN++;
        }
        continue;
      }
      nextFaded.add(i);
      const shape = shapeCodes ? shapeCodes[i]! % shapeCount : 0;
      let pool = pools.get(shape);
      if (!pool) pool = makePool(shape);
      let w = fill.get(shape) ?? 0;
      if (w >= count) continue; // this shape's pool is full (rare — count is the global cap)
      const o3 = w * 3;
      const r = field.sizeOf(i) * NODE_SCALE;
      // instance matrix = uniform scale (the node's world radius) then translate to the live world
      // position; three feeds it to the clip position AND the written depth, so the body is solid.
      instMat.makeScale(r, r, r);
      instMat.setPosition(candPos[p3]!, candPos[p3 + 1]!, candPos[p3 + 2]!);
      pool.mesh.setMatrixAt(w, instMat);
      pool.iTint[o3] = colors[i * 3]!;
      pool.iTint[o3 + 1] = colors[i * 3 + 1]!;
      pool.iTint[o3 + 2] = colors[i * 3 + 2]!;
      let layer = texCodes ? texCodes[i]! % texCount : 0;
      // a `procedural` override forces a specific built-in planet layer (still an instanced instance).
      if (src && src.kind === 'procedural') {
        const pl = planetLayer(src.planet);
        if (pl >= 0) layer = pl;
      }
      pool.iTexLayer[w] = layer;
      // procedural detail blend: ramp 0→1 as the node's projected radius crosses the kick-in band,
      // but ONLY for a promoted hard-edge layer (every other layer, and every far node, stays baked).
      if (proceduralDetail) {
        let det = 0;
        if (PROCEDURAL_LAYERS.has(layer)) {
          const t = clamp01((candPx[j]! - DETAIL_PX_LO) / (DETAIL_PX_HI - DETAIL_PX_LO));
          det = t * t * (3 - 2 * t); // smoothstep
        }
        pool.iDetail[w] = det;
      }
      w++;
      fill.set(shape, w);
    }

    // individual image-override meshes: place/scale/texture them and (once a tier is up) fade their
    // billboards. Gated on hasImageOverride so the default path never touches this.
    if (hasImageOverride) updateImages(imgN, nextFaded);
    else if (imageSlots) hideAllImages();

    // commit pool counts + upload (tiny buffers; a full small upload each active frame)
    for (const p of pools.values()) {
      const w = fill.get(p.shape) ?? 0;
      p.mesh.count = w;
      if (w > 0) {
        p.mesh.instanceMatrix.needsUpdate = true;
        p.tintAttr.needsUpdate = true;
        if (textured) p.texAttr.needsUpdate = true;
        if (proceduralDetail) p.detailAttr.needsUpdate = true;
      }
    }

    // fade the promoted nodes' billboards; restore those that left the set.
    let alphaDirty = false;
    for (const [i, a] of faded) {
      if (!nextFaded.has(i)) {
        alpha[i] = a;
        alphaAttr.addUpdateRange(i, 1);
        faded.delete(i);
        alphaDirty = true;
      }
    }
    for (const i of nextFaded) {
      if (!faded.has(i)) faded.set(i, alpha[i]!);
      if (alpha[i] !== 0) {
        alpha[i] = 0; // force every frame: a focus sweep may have re-lit it (idempotent, O(K))
        alphaAttr.addUpdateRange(i, 1);
        alphaDirty = true;
      }
    }
    if (alphaDirty) alphaAttr.needsUpdate = true;

    /* drive (or hide) the displaced terrain mesh for the suppressed earth node. Its live centre is
     * candPos[terrainJ] (exactly where its instanced sphere / faded billboard sat), so it tracks the
     * spinning / morphing field. detail = the SAME px→detail ramp the instanced planet uses, so the
     * draped earth colour matches at the swap. The terrain owns the amplitude ramp; we read its
     * ampFrac back for the collision floor. */
    if (terrainNode >= 0 && terrainJ >= 0) {
      const t = ensureTerrain();
      const p3 = terrainJ * 3;
      const r = field.sizeOf(terrainNode) * NODE_SCALE;
      const tt = clamp01((candPx[terrainJ]! - DETAIL_PX_LO) / (DETAIL_PX_HI - DETAIL_PX_LO));
      const det = tt * tt * (3 - 2 * tt); // smoothstep, matching the instanced detail ramp
      t.update({
        cx: candPos[p3]!,
        cy: candPos[p3 + 1]!,
        cz: candPos[p3 + 2]!,
        radius: r,
        altFrac: terrainAltFrac,
        detail: det,
        tintR: colors[terrainNode * 3]!,
        tintG: colors[terrainNode * 3 + 1]!,
        tintB: colors[terrainNode * 3 + 2]!,
      });
      terrainActive = true;
      terrainRadius = r;
      terrainAmpFrac = t.ampFrac;
      terrainDetail = det;
    } else {
      if (terrain) terrain.setVisible(false);
      terrainActive = false;
    }

    promotedCount = take;
  };

  return {
    objects: [group],
    update,
    nearestSurface(cam: { x: number; y: number; z: number }):
      | { center: readonly [number, number, number]; radius: number }
      | null {
      if (!enabled || surfCount === 0) return null;
      let best = -1;
      let bestD = Infinity;
      for (let k = 0; k < surfCount; k++) {
        const dx = surfCenters[k * 3]! - cam.x;
        const dy = surfCenters[k * 3 + 1]! - cam.y;
        const dz = surfCenters[k * 3 + 2]! - cam.z;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz) - surfRadii[k]!; // surface distance
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      }
      if (best < 0) return null;
      surfOut.center[0] = surfCenters[best * 3]!;
      surfOut.center[1] = surfCenters[best * 3 + 1]!;
      surfOut.center[2] = surfCenters[best * 3 + 2]!;
      surfOut.radius = surfRadii[best]!;
      // when the nearest body is the terrain planet, hand back the per-direction terrain floor so the
      // camera lands on the relief, not the sphere. Any other body stays a plain sphere (stage A).
      surfOut.terrain =
        terrainActive && best === terrainSurfIdx && heightSrc ? terrainSample : undefined;
      return surfOut;
    },
    setEnabled(on: boolean): void {
      if (on === enabled) return;
      enabled = on;
      if (!on) {
        clearPools();
        restoreAll();
        if (imageSlots) hideAllImages();
        if (terrain) terrain.setVisible(false);
        terrainActive = false;
        promotedCount = 0;
      } else {
        builtOnce = false; // rebuild the grid against the current arrangement on the next update
      }
    },
    setTerrain(on: boolean): void {
      if (on === terrainEnabled) return;
      terrainEnabled = on;
      // off: hide immediately (the node returns to its instanced sphere next update). On: the next
      // update lazily creates the mesh once you are actually close to an earth planet.
      if (!on && terrain) {
        terrain.setVisible(false);
        terrainActive = false;
      }
    },
    setView(heightCss: number, fovDeg: number): void {
      if (Number.isFinite(heightCss) && heightCss > 0) halfH = heightCss / 2;
      if (Number.isFinite(fovDeg) && fovDeg > 0) tanHalf = Math.tan((fovDeg * Math.PI) / 360);
    },
    setPromotePx(px: number): void {
      if (Number.isFinite(px) && px > 0) promotePx = px;
    },
    setShapeFacet(facet: string | null): void {
      shapeFacet = facet;
      shapeCodes = facetData(facet);
    },
    setTextureFacet(facet: string | null): void {
      textureFacet = facet;
      texCodes = facetData(facet);
      const want = facet !== null;
      if (want !== textured) {
        textured = want;
        const old = material;
        material = buildMaterial(textured);
        for (const p of pools.values()) p.mesh.material = material;
        old.dispose();
      }
    },
    setProceduralDetail(on: boolean): void {
      if (on === proceduralDetail) return;
      proceduralDetail = on;
      // rebuild the material so OFF compiles the exact shipped shader (references no iDetail) and ON
      // splices the procedural branch in; the next update fills iDetail for promoted hard-edge nodes.
      const old = material;
      material = buildMaterial(textured);
      for (const p of pools.values()) {
        p.mesh.material = material;
        if (!on) {
          p.iDetail.fill(0); // leave the buffers clean when off (no stale blend if re-enabled)
          p.detailAttr.needsUpdate = true;
        }
      }
      old.dispose();
    },
    setOverrides(map: Record<number, NodeSource> | null): void {
      overrides = map ?? {};
      refreshOverrideFlags();
      // overrides changed: drop every cached tier (the old keys may be stale) and release slots; the
      // next update re-promotes and re-loads against the new map. Cheap — overrides change rarely.
      disposeImages();
    },
    get promoted(): number {
      return promotedCount;
    },
    get drawCalls(): number {
      if (!enabled) return 0;
      let d = 0;
      for (const p of pools.values()) if (p.mesh.count > 0) d++;
      d += visibleImages; // each visible image node is its own draw call
      if (terrainActive) d++; // the displaced terrain mesh is its own draw call
      return d;
    },
    get imageMeshes(): number {
      return enabled ? visibleImages : 0;
    },
    get terrainActive(): boolean {
      return enabled && terrainActive;
    },
    get terrainTriangles(): number {
      return enabled && terrain ? terrain.triangles : 0;
    },
    get terrainSubdiv(): number {
      return terrain ? terrain.subdiv : 0;
    },
    get imageCacheBytes(): number {
      return cacheBytes;
    },
    get imageInfo(): ReadonlyArray<{ node: number; tier: number; px: number }> {
      return imageInfoArr;
    },
    dispose(): void {
      restoreAll();
      alphaAttr.needsUpdate = true;
      for (const p of pools.values()) {
        p.mesh.dispose();
        p.geo.dispose();
        group.remove(p.mesh);
      }
      pools.clear();
      material.dispose();
      if (terrain) {
        group.remove(terrain.object);
        terrain.dispose();
        terrain = null;
      }
      for (const g of shapeGeos) if (g) g.dispose();
      if (texArray && ownsTexArray) texArray.dispose();
      // image infra: dispose every cached tier, the slot meshes/materials, the shared geo + placeholder.
      disposeImages();
      if (imageSlots) {
        for (const s of imageSlots) group.remove(s.mesh);
        imageSlots = null;
      }
      for (const mat of imageMaterials) mat.dispose();
      imageMaterials.length = 0;
      if (imageGeo) imageGeo.dispose(); // one shared sphere across all slots
      if (placeholderTex) placeholderTex.dispose();
    },
  };
}
