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
import type { BufferGeometry, Object3D } from 'three/webgpu';
import {
  Fn,
  attribute,
  dot,
  float,
  int,
  max,
  mix,
  normalLocal,
  texture,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import type { Graph } from '../core/types';
import type { Field } from '../field/field';
import { buildGrid, cellMaxOf, type Grid } from '../pick/grid';
import { glyphPx } from '../field/channels';
import { hash } from '../core/hash';
import { D_REF_FRACTION, FALLOFF, NODE_SCALE, SIZE_MAX_WORLD } from '../field/constants';

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
function wnoise(u: number, y: number, period: number, salt: number): number {
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
}

export interface NearMesh {
  /** Scene-attachable objects (one Group holding the shape pools) — add every one to your scene. */
  objects: Object3D[];
  /** Per frame: find the promoted set by projected size, place/scale/tint their meshes at their
   *  LIVE positions, and fade their billboards. O(local) — cheap when nothing promotes. */
  update(cameraWorldPos: { x: number; y: number; z: number }, dt: number): void;
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
  /** How many nodes are drawn as meshes right now (telemetry). */
  readonly promoted: number;
  /** How many draw calls the near pools add this frame (telemetry; 0 when nothing promotes). */
  readonly drawCalls: number;
  dispose(): void;
}

interface Pool {
  shape: number;
  geo: BufferGeometry;
  mesh: InstancedMesh;
  iTint: Float32Array;
  iTexLayer: Float32Array;
  tintAttr: InstancedBufferAttribute;
  texAttr: InstancedBufferAttribute;
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
        // let the texture carry the identity; the node tint only hints (so a planet stays a planet)
        const albedo = sampled.rgb.mul(mix(vec3(1, 1, 1), tint, float(0.25)));
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
    const tintAttr = new InstancedBufferAttribute(iTint, 3);
    const texAttr = new InstancedBufferAttribute(iTexLayer, 1);
    tintAttr.setUsage(DynamicDrawUsage);
    texAttr.setUsage(DynamicDrawUsage);
    geo.setAttribute('iTint', tintAttr);
    geo.setAttribute('iTexLayer', texAttr);
    const mesh = new InstancedMesh(geo, material, count);
    mesh.count = 0;
    mesh.instanceMatrix.setUsage(DynamicDrawUsage); // rewritten every active frame
    mesh.frustumCulled = false;
    mesh.renderOrder = 2;
    const pool: Pool = { shape, geo, mesh, iTint, iTexLayer, tintAttr, texAttr };
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

  const worldPx = (sizeWorld: number, dist: number): number =>
    glyphPx(sizeWorld, dist, halfH, tanHalf, dRef, falloff);

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

    // reset pool fills
    const fill = new Map<number, number>();
    for (const p of pools.values()) fill.set(p.shape, 0);

    // the new promoted set (to diff against `faded`)
    const nextFaded = new Set<number>();
    for (let s = 0; s < take; s++) {
      const j = order[s]!;
      const i = candIdx[j]!;
      nextFaded.add(i);
      const shape = shapeCodes ? shapeCodes[i]! % shapeCount : 0;
      let pool = pools.get(shape);
      if (!pool) pool = makePool(shape);
      let w = fill.get(shape) ?? 0;
      if (w >= count) continue; // this shape's pool is full (rare — count is the global cap)
      const o3 = w * 3;
      const p3 = j * 3;
      const r = field.sizeOf(i) * NODE_SCALE;
      // instance matrix = uniform scale (the node's world radius) then translate to the live world
      // position; three feeds it to the clip position AND the written depth, so the body is solid.
      instMat.makeScale(r, r, r);
      instMat.setPosition(candPos[p3]!, candPos[p3 + 1]!, candPos[p3 + 2]!);
      pool.mesh.setMatrixAt(w, instMat);
      pool.iTint[o3] = colors[i * 3]!;
      pool.iTint[o3 + 1] = colors[i * 3 + 1]!;
      pool.iTint[o3 + 2] = colors[i * 3 + 2]!;
      pool.iTexLayer[w] = texCodes ? texCodes[i]! % texCount : 0;
      w++;
      fill.set(shape, w);
    }

    // commit pool counts + upload (tiny buffers; a full small upload each active frame)
    for (const p of pools.values()) {
      const w = fill.get(p.shape) ?? 0;
      p.mesh.count = w;
      if (w > 0) {
        p.mesh.instanceMatrix.needsUpdate = true;
        p.tintAttr.needsUpdate = true;
        if (textured) p.texAttr.needsUpdate = true;
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

    promotedCount = take;
  };

  return {
    objects: [group],
    update,
    setEnabled(on: boolean): void {
      if (on === enabled) return;
      enabled = on;
      if (!on) {
        clearPools();
        restoreAll();
        promotedCount = 0;
      } else {
        builtOnce = false; // rebuild the grid against the current arrangement on the next update
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
    get promoted(): number {
      return promotedCount;
    },
    get drawCalls(): number {
      if (!enabled) return 0;
      let d = 0;
      for (const p of pools.values()) if (p.mesh.count > 0) d++;
      return d;
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
      for (const g of shapeGeos) if (g) g.dispose();
      if (texArray && ownsTexArray) texArray.dispose();
    },
  };
}
