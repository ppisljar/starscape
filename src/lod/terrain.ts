/* true-geometry planetary terrain (DESIGN.md §5 tier-3; near-mesh "mote → mesh", terrain stage B) —
 * the DEEPEST close look. Stage A made the camera collide with a promoted planet's bounding SPHERE;
 * this stage replaces that sphere, for the ONE planet you are descending onto, with REAL DISPLACED
 * GEOMETRY — a higher-detail icosphere whose vertices are pushed out along their normal by a height
 * field so land becomes mountains with silhouettes you fly over and finally land on, oceans stay
 * flat, and the relief RESOLVES IN seamlessly out of the textured sphere (a Google-Earth descent).
 *
 * THE HARD RULES (same spirit as nearmesh.ts):
 *   1. OPT-IN ⇒ ZERO COST WHEN OFF. `createTerrain` is the only entry; nearmesh calls it LAZILY, only
 *      when the terrain master switch is on AND you are actually close to a solid (earth) planet. Never
 *      created ⇒ no geometry, no material, no per-frame work. One planet at a time (the orbited one).
 *   2. SEAMLESS HANDOFF. The displacement amplitude ramps from 0 at the swap altitude (where relief is
 *      sub-pixel) to full as you descend; at amp 0 the icosphere IS the textured sphere (same radius,
 *      same draped colour), so there is no pop — the geometry resolves in exactly like the procedural
 *      colour detail already sharpens. nearmesh suppresses that one node's instanced sphere so the two
 *      never double-draw.
 *   3. PLUGGABLE HEIGHT. The surface is a `TerrainHeightSource` — a CPU sampler (for collision) and a
 *      bit-matching TSL node (for the vertex displacement). The prototype uses Earth's own procedural
 *      noise (so mountains register with the coastlines); a real tiered DEM image swaps in behind the
 *      SAME interface without touching the displacement / tessellation / collision.
 *
 * NORMALS. The displacement happens in the VERTEX shader (so the amplitude can ramp per frame with no
 * geometry rebuild), so the lit normal is recomputed there analytically: two finite-difference taps of
 * the height field along a tangent basis give the displaced surface slope, crossed into the per-vertex
 * normal and interpolated — relief that lights correctly and, at amp 0, collapses back to the smooth
 * sphere normal (seamless).
 *
 * LOD. A handful of discrete icosphere subdivisions swapped by altitude (coarse high up, finer near the
 * ground), each built lazily on first use. Not CDLOD/clipmaps — enough to judge the look + feel.
 */

import {
  Group,
  IcosahedronGeometry,
  Mesh,
  MeshBasicNodeMaterial,
  Vector3,
} from 'three/webgpu';
import type { BufferGeometry, Node, Object3D } from 'three/webgpu';
import {
  Fn,
  abs,
  asin as asinTSL,
  atan as atanTSL,
  clamp as clampTSL,
  cross,
  dot,
  float,
  mix,
  normalize,
  pow as powTSL,
  positionGeometry,
  sign,
  step,
  uniform,
  varying,
  vec3,
} from 'three/tsl';
import { fbmField, wnoise, DETAIL_OCTAVES, DETAIL_GAIN } from './nearmesh';

type FNode = Node<'float'>;
type V3Node = Node<'vec3'>;
type V4Node = Node<'vec4'>;

/* ---- altitude thresholds (radius units: altitude / planet radius) --------------------------------
 * Terrain exists only below TERRAIN_ALT_ON (nearmesh gates activation on this). The displacement
 * amplitude ramps 0→1 as altitude falls from TERRAIN_ALT_ON to TERRAIN_ALT_FULL, so at the swap edge
 * the relief is ~0 (sub-pixel) and the handoff is invisible. */
export const TERRAIN_ALT_ON = 1.6;
export const TERRAIN_ALT_FULL = 0.25;
/** Default peak displacement as a fraction of planet radius at full ramp. Exaggerated vs real Earth
 *  (~0.1% of radius, invisible) so silhouettes read in a prototype; a DEM source would carry true
 *  elevations and a smaller scale. */
export const TERRAIN_HEIGHT_SCALE = 0.11;
/** Finite-difference step (radians of arc on the unit sphere) for the analytic normal. Small enough
 *  to catch per-vertex slope, large enough not to alias on the finest height octave. */
const NORMAL_EPS = 0.008;

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
const smooth = (a: number, b: number, x: number): number => {
  const t = clamp01((x - a) / (b - a || 1e-6));
  return t * t * (3 - 2 * t);
};
const INV_2PI = 1 / (2 * Math.PI);
const INV_PI = 1 / Math.PI;

/* ---- the pluggable height source ---------------------------------------------------------------- */

/** A terrain surface: a CPU sampler (the camera collision floor) and a bit-matching TSL node (the GPU
 *  vertex displacement). Both return a NORMALIZED height ≈[0,1] (sea level 0) for a UNIT world
 *  direction; `detail` 0→1 ramps the fine octaves in (keep the two in lock-step so the camera lands
 *  on what it sees). This is the DEM seam: swap in a tiered elevation image behind this interface and
 *  nothing downstream (displacement / tessellation / collision) changes. */
export interface TerrainHeightSource {
  /** CPU height at a unit direction (dx,dy,dz). For the flycam collision floor. */
  height(dx: number, dy: number, dz: number, detail: number): number;
  /** TSL displacement height for the vertex shader — `dir` a unit-direction vec3 node. Matches
   *  `height`. */
  node(dir: V3Node, detail: FNode): FNode;
}

/** CPU fbm mirroring the TSL `fbmField` in nearmesh.ts EXACTLY (same `wnoise`, same octave salts, same
 *  baked-norm division, same ramped detail octaves) so the collision height equals the displaced
 *  geometry. `detail` ramps the `DETAIL_OCTAVES` fine octaves in at `DETAIL_GAIN`. */
function fbmCPU(
  u: number,
  vy: number,
  detail: number,
  cells: number,
  yscale: number,
  bakedOct: number,
  saltBase: number,
): number {
  const octSalt = (i: number): number => (saltBase + i * 0x9e37) | 0;
  let p = cells;
  let ys = yscale;
  let amp = 0.5;
  let norm = 0;
  let low = 0;
  for (let i = 0; i < bakedOct; i++) {
    low += amp * wnoise(u, vy * ys, p, octSalt(i));
    norm += amp;
    p *= 2;
    ys *= 2;
    amp *= 0.5;
  }
  let detailSum = 0;
  for (let i = bakedOct; i < bakedOct + DETAIL_OCTAVES; i++) {
    detailSum += amp * wnoise(u, vy * ys, p, octSalt(i));
    p *= 2;
    ys *= 2;
    amp *= 0.5;
  }
  return (low + detailSum * detail * DETAIL_GAIN) / norm;
}

/** Equirectangular (longitude u, latitude v) from a unit direction, matching the shader convention in
 *  nearmesh (`uu = atan(x,z)/2π + 0.25`, `v = asin(y)/π + 0.5`) so continents register with the baked
 *  texture and the draped colour. */
function dirToUV(dx: number, dy: number, dz: number): [number, number] {
  let u = Math.atan2(dx, dz) * INV_2PI + 0.25;
  u -= Math.floor(u);
  const v = Math.asin(Math.max(-1, Math.min(1, dy))) * INV_PI + 0.5;
  return [u, v];
}

/** Earth's height source: the SAME land-mask noise that paints the coastline (`fbmField(...,5,3,5)`
 *  with the earth salt), so relief sits EXACTLY where the land is and oceans stay flat. A `landMask`
 *  rises sharply at the coast (land > 0.52, matching PLANETS.earth) and a higher-frequency RIDGED fbm
 *  carves mountain ranges + valleys over the interior (ridged = `1 − |2·fbm − 1|`, the classic crest
 *  transform). CPU + TSL are bit-matched so the camera lands on exactly what it sees. */
export function earthTerrainSource(earthSalt: number): TerrainHeightSource {
  const COAST = 0.52; // the PLANETS.earth land threshold — relief ramps up over the next band
  const SHELF = 0.6; // land fully raised by here (a tight coastal shelf)
  const RIDGE_SALT = (earthSalt ^ 0x9e9) | 0; // a distinct band for the mountain ridges
  const RIDGE_BASE = 0.15; // interior floor elevation; ridges add the rest (low ⇒ deep valleys)
  const RIDGE_CELLS = 20; // mountain-range count around the longitude (higher ⇒ narrower, sharper)
  const RIDGE_YS = 10;
  const RIDGE_POW = 1.4; // >1 sharpens crests into peaks (vs rounded humps)
  return {
    height(dx, dy, dz, detail): number {
      const [u, v] = dirToUV(dx, dy, dz);
      const land = fbmCPU(u, v, detail, 5, 3, 5, earthSalt);
      const landMask = smooth(COAST, SHELF, land); // 0 at/below the coast → 1 just inland
      if (landMask <= 0) return 0; // ocean: flat
      const m = fbmCPU(u, v, detail, RIDGE_CELLS, RIDGE_YS, 4, RIDGE_SALT);
      const ridge = Math.pow(1 - Math.abs(2 * m - 1), RIDGE_POW); // crests → 1, valleys → 0
      return landMask * (RIDGE_BASE + (1 - RIDGE_BASE) * ridge);
    },
    node(dir, detail): FNode {
      const uu = atan2Node(dir.x as FNode, dir.z as FNode).mul(INV_2PI).add(0.25).fract();
      const vLat = asinNode(dir.y as FNode).mul(INV_PI).add(0.5);
      const land = fbmField(uu as FNode, vLat as FNode, detail, 5, 3, 5, earthSalt);
      // smoothstep(COAST, SHELF, land), bit-faithful to the CPU `smooth`.
      const t = clampTSL(land.sub(COAST).div(SHELF - COAST), 0, 1) as unknown as FNode;
      const landMask = t.mul(t).mul(float(3).sub(t.mul(2))) as unknown as FNode;
      const m = fbmField(uu as FNode, vLat as FNode, detail, RIDGE_CELLS, RIDGE_YS, 4, RIDGE_SALT);
      const ridge = powTSL(float(1).sub(abs(m.mul(2).sub(1))), RIDGE_POW) as unknown as FNode;
      const inner = float(RIDGE_BASE).add(ridge.mul(1 - RIDGE_BASE)) as unknown as FNode;
      return landMask.mul(inner) as unknown as FNode;
    },
  };
}

/* TSL helpers so this module's UV matches nearmesh's exactly: atan(x,z) and asin(clamp(y)). */
function atan2Node(x: FNode, z: FNode): FNode {
  return atanTSL(x, z) as unknown as FNode;
}
function asinNode(y: FNode): FNode {
  return asinTSL(clampTSL(y, -1, 1)) as unknown as FNode;
}

/* ---- the terrain mesh --------------------------------------------------------------------------- */

/** The draped colour: given the ORIGINAL (undisplaced) unit direction, the displaced surface NORMAL,
 *  the detail ramp and the node tint, return the final lit RGBA — identical to the instanced planet's
 *  shading so the handoff is invisible. nearmesh supplies this (it owns the texture array + the earth
 *  colour), keeping this module planet-agnostic. */
export type TerrainColorNode = (dir: V3Node, normal: V3Node, detail: FNode, tint: V3Node) => V4Node;

export interface TerrainOpts {
  /** The height field (prototype: `earthTerrainSource`). The DEM seam. */
  height: TerrainHeightSource;
  /** The draped colour node (nearmesh's earth shading). */
  colorNode: TerrainColorNode;
  /** Peak displacement as a fraction of planet radius at full ramp. Default TERRAIN_HEIGHT_SCALE. */
  heightScale?: number;
  /** Finest icosphere subdivision (the near-ground LOD). Default 6 (81,920 tris). Coarser levels at
   *  `max-1`, `max-2` fill the higher altitudes. */
  maxSubdiv?: number;
}

export interface TerrainUpdate {
  cx: number;
  cy: number;
  cz: number;
  /** Planet world radius (the instanced-sphere radius: `field.sizeOf·NODE_SCALE`). */
  radius: number;
  /** Camera altitude above the sphere, in radius units (altitude / radius). Drives the amplitude
   *  ramp and the LOD band. */
  altFrac: number;
  /** Colour detail ramp (0→1), the SAME value the instanced planet would use at this on-screen px, so
   *  the draped colour matches exactly at the swap. */
  detail: number;
  /** The node's field tint (linear RGB). */
  tintR: number;
  tintG: number;
  tintB: number;
}

export interface Terrain {
  /** Add to the scene (nearmesh adds it to its own group). */
  object: Object3D;
  /** Place / scale / ramp / LOD the terrain for the orbited planet this frame. */
  update(u: TerrainUpdate): void;
  /** Hide it (far from any solid planet, or the feature toggled off). */
  setVisible(on: boolean): void;
  readonly visible: boolean;
  /** Current displacement amplitude as a fraction of radius (= heightScale × ramp). The collision
   *  floor multiplies the height field by this, so it must match the GPU uniform. */
  readonly ampFrac: number;
  /** Current detail ramp (passthrough) — the collision height samples at this detail. */
  readonly detail: number;
  /** Current LOD subdivision + triangle count (telemetry). */
  readonly subdiv: number;
  readonly triangles: number;
  dispose(): void;
}

/** Create the terrain subsystem for one planet at a time. Opt-in + lazy: nearmesh calls this only when
 *  the terrain switch is on and you are close to a solid planet. */
export function createTerrain(opts: TerrainOpts): Terrain {
  const heightScale = Number.isFinite(opts.heightScale)
    ? (opts.heightScale as number)
    : TERRAIN_HEIGHT_SCALE;
  const maxSubdiv = Math.max(1, (opts.maxSubdiv ?? 6) | 0);
  const subdivLevels = [Math.max(1, maxSubdiv - 2), Math.max(2, maxSubdiv - 1), maxSubdiv];

  // uniforms the fill drives each frame; the material reads them, so no rebuild on ramp / LOD swap.
  const uAmp = uniform(0);
  const uDetail = uniform(0);
  const uTint = uniform(new Vector3(1, 1, 1));

  /* ---- the displacement + normal material (built once; shared across LODs) ---------------------- */
  const mat = new MeshBasicNodeMaterial();
  mat.transparent = false;
  mat.depthWrite = true;
  mat.depthTest = true;

  // VERTEX: unit direction → displaced position + analytic displaced normal (two finite-difference
  // height taps along a tangent basis). At uAmp 0 the normal collapses to the smooth sphere normal.
  const dir = positionGeometry.normalize() as unknown as V3Node;
  const uAmpF = uAmp as unknown as FNode;
  const uDetailF = uDetail as unknown as FNode;
  const dispAt = (d: V3Node): V3Node =>
    d.mul(float(1).add(uAmpF.mul(opts.height.node(d, uDetailF)))) as unknown as V3Node;
  const p0 = dispAt(dir);
  // a stable tangent basis: swap the reference axis near the poles so the cross product never degenerates.
  const ref = mix(vec3(0, 1, 0), vec3(1, 0, 0), step(0.9, abs(dir.y))) as unknown as V3Node;
  const t1 = normalize(cross(ref, dir)) as unknown as V3Node;
  const t2 = cross(dir, t1) as unknown as V3Node;
  const eps = float(NORMAL_EPS);
  const dirA = normalize(dir.add(t1.mul(eps))) as unknown as V3Node;
  const dirB = normalize(dir.add(t2.mul(eps))) as unknown as V3Node;
  const pA = dispAt(dirA);
  const pB = dispAt(dirB);
  let nrm = normalize(cross(pA.sub(p0), pB.sub(p0))) as unknown as V3Node;
  nrm = nrm.mul(sign(dot(nrm, dir))) as unknown as V3Node; // force outward
  mat.positionNode = p0 as unknown as typeof mat.positionNode;

  const vDir = varying(dir) as unknown as V3Node;
  const vNrm = varying(nrm) as unknown as V3Node;
  mat.colorNode = Fn(() =>
    opts.colorNode(vDir, vNrm, uDetailF, uTint as unknown as V3Node),
  )() as unknown as typeof mat.colorNode;

  /* ---- the mesh + lazy LOD geometries ----------------------------------------------------------- */
  const group = new Group();
  group.renderOrder = 2;
  const geos = new Map<number, BufferGeometry>();
  const geoFor = (sd: number): BufferGeometry => {
    let g = geos.get(sd);
    if (!g) {
      g = new IcosahedronGeometry(1, sd); // unit radius ⇒ per-instance scale IS the planet radius
      geos.set(sd, g);
    }
    return g;
  };
  let curSubdiv = subdivLevels[0]!;
  const mesh = new Mesh(geoFor(curSubdiv), mat);
  mesh.frustumCulled = false; // positioned every frame; a bounding sphere at origin would mis-cull
  mesh.renderOrder = 2;
  mesh.visible = false;
  group.add(mesh);

  const triOf = (sd: number): number => 20 * Math.pow(4, sd);

  // LOD band: coarser high up, finer near the ground. Discrete (a tiny pop between bands is fine for
  // the prototype — both sample the SAME height field, so silhouettes barely differ).
  const pickSubdiv = (altFrac: number): number => {
    if (altFrac >= 0.8) return subdivLevels[0]!;
    if (altFrac >= 0.35) return subdivLevels[1]!;
    return subdivLevels[2]!;
  };

  let visible = false;
  let ampFrac = 0;
  let detail = 0;

  return {
    object: group,
    update(u: TerrainUpdate): void {
      // amplitude ramp: 0 at the swap altitude (sub-pixel), → 1 near the ground.
      const ramp = 1 - smooth(TERRAIN_ALT_FULL, TERRAIN_ALT_ON, u.altFrac);
      ampFrac = heightScale * ramp;
      detail = u.detail;
      uAmp.value = ampFrac;
      uDetail.value = u.detail;
      (uTint.value as Vector3).set(u.tintR, u.tintG, u.tintB);
      const sd = pickSubdiv(u.altFrac);
      if (sd !== curSubdiv) {
        curSubdiv = sd;
        mesh.geometry = geoFor(sd);
      }
      mesh.position.set(u.cx, u.cy, u.cz);
      mesh.scale.setScalar(u.radius);
      mesh.visible = true;
      visible = true;
    },
    setVisible(on: boolean): void {
      if (on === visible) return;
      visible = on;
      mesh.visible = on;
    },
    get visible(): boolean {
      return visible;
    },
    get ampFrac(): number {
      return ampFrac;
    },
    get detail(): number {
      return detail;
    },
    get subdiv(): number {
      return curSubdiv;
    },
    get triangles(): number {
      return visible ? triOf(curSubdiv) : 0;
    },
    dispose(): void {
      mesh.visible = false;
      group.remove(mesh);
      for (const g of geos.values()) g.dispose();
      geos.clear();
      mat.dispose();
    },
  };
}
