/* Adaptive terrain tessellation — a clipmap/CDLOD PATCH that follows the camera's ground sub-point
 * (near-mesh terrain stage C; DESIGN.md §15 "adaptive CDLOD tessellation" first cut). The uniform
 * icosphere in terrain.ts spreads its triangles over the WHOLE sphere, so at the surface the local
 * edge is coarse and the near ground reads near-flat — you cannot fly BETWEEN individual hills. This
 * module OVERLAYS a localized high-detail grid on the near region (the uniform icosphere still draws
 * the distant surface + the limb), concentrating metre-scale triangles directly under the camera and
 * coarsening outward until its resolution matches the sphere.
 *
 * HOW (a geometry clipmap with a CDLOD morph):
 *   • LEVELS. A nested set of concentric square grids, all the SAME vertex resolution `res`. Level 0 is
 *     a full grid of half-extent H0; each outer level k is a hollow RING of half-extent 2×, so its cell
 *     is 2× coarser and its inner hole is exactly filled by level k−1. Triangle density therefore
 *     HALVES per level outward — dense under the camera, coarse at the rim — for a bounded vertex count
 *     (≈ res² for level 0 + ¾·res² per ring).
 *   • SPHERICAL MAPPING. Each vertex carries a 2-D lattice offset (cells). The vertex shader turns that
 *     into a tangent-plane offset in world metres (cellSize·cell + the level's snapped centre), maps it
 *     onto the sphere by the azimuthal-equidistant exponential map (offset distance d → great-circle
 *     arc d/R about the anchor), then displaces along that direction by the SAME `TerrainHeightSource`
 *     the icosphere uses, with the SAME draped procedural colour — so the patch is indistinguishable
 *     from the sphere except denser.
 *   • FOLLOW + SNAP. The patch centre tracks the camera's sub-point; each level's centre is SNAPPED to
 *     its own 2×cell grid, so as you fly the vertices sit on world-fixed lattice points and the ground
 *     does not swim/crawl — only the covered window shifts, one cell at a time.
 *   • CRACK-FREE (CDLOD morph). Near each level's outer edge a per-vertex morph blends the lattice
 *     position toward the next-coarser level's lattice (odd cells collapse onto even). At the boundary
 *     the fine edge becomes topologically identical to the coarse ring's inner edge AND samples the
 *     height field at the same points, so there is no T-junction and no crack between rings.
 *   • SEAM TO THE SPHERE. The whole patch floats a sub-pixel bias above the icosphere (so it wins the
 *     depth test over the coarser sphere it overlays, no z-fight), and the outermost rim drops a SKIRT
 *     that hides the thin gap where the fine patch meets the coarse sphere.
 *
 * Collision is UNCHANGED: the flycam still floors on the icosphere's CPU height sampler (terrain.ts),
 * which samples the identical field at the identical amplitude, so the camera lands on what the patch
 * draws. This module is pure overlay — opt-in, lazy, one planet, hidden when the terrain is not active.
 */

import {
  DoubleSide,
  Group,
  Mesh,
  MeshBasicNodeMaterial,
  BufferGeometry,
  Float32BufferAttribute,
  Vector2,
  Vector3,
} from 'three/webgpu';
import type { Node, Object3D } from 'three/webgpu';
import {
  Fn,
  abs,
  attribute,
  cos as cosTSL,
  cross,
  dot,
  float,
  max as maxTSL,
  mix,
  normalize,
  round as roundTSL,
  sign,
  sin as sinTSL,
  smoothstep as smoothstepTSL,
  uniform,
  varying,
} from 'three/tsl';
import { TERRAIN_ALT_FULL, TERRAIN_ALT_ON, TERRAIN_HEIGHT_SCALE } from './terrain';
import type { TerrainColorNode, TerrainHeightSource } from './terrain';

type FNode = Node<'float'>;
type V3Node = Node<'vec3'>;

/* ---- tunables ------------------------------------------------------------------------------------ */
/** Cells across one side of every clipmap level (even, so the centre hole is a clean square). Higher ⇒
 *  more triangles under the camera. 128 ⇒ level 0 = 32,768 tris, each ring ~24,576, six levels
 *  ≈ 155k tris total — an eighth of the uniform sphere, but ALL concentrated in a ~0.22R cap. */
const DEFAULT_RES = 128;
/** Number of nested levels (innermost + rings). Each doubles the extent / halves the density. */
const DEFAULT_LEVELS = 6;
/** The whole patch's outer half-extent as a fraction of planet radius (≈ a 0.22R tangent cap, ~12.6°).
 *  Chosen so the OUTERMOST cell (= outerExtent·2/res) ≈ the uniform sphere's edge at subdiv 8, so the
 *  rim refines the sphere rather than clashing with it. */
const DEFAULT_OUTER_EXTENT_FRAC = 0.22;
/** Outward radial bias FLOOR (fraction of radius) so the patch wins the depth test over the coarser
 *  icosphere it overlays. Sub-pixel at ground range; large enough to clear the sphere's tessellation
 *  undershoot when the displacement amplitude is near zero (the swap altitude). */
const DEFAULT_BIAS_FRAC = 0.0016;
/** Amplitude-scaled bias term. The icosphere and the patch displace the SAME height field, but the
 *  icosphere is coarse and LINEARLY interpolates between its vertices — so in concave areas (valley
 *  floors, the inside of a bend) its interpolated surface sits ABOVE the patch's finer surface. Where
 *  it does, the icosphere wins the depth test and POKES THROUGH the patch as flat, bright, under-lit
 *  facets — the close-range "breakdown" (depth order / renderOrder cannot fix it: the icosphere is
 *  physically higher there). That vertical discrepancy scales with the displacement amplitude, so the
 *  bias does too: `bias = max(FLOOR, AMP_K · ampFrac) · radius`. At full amp (0.11R) this is ≈0.0143R,
 *  well under the skirt depth (0.03R), which curtains the resulting sub-percent rim step. The collision
 *  floor samples the field WITHOUT this bias, so the camera still hovers above the true relief. */
const BIAS_AMP_K = 0.13;
/** Outer-rim skirt depth (fraction of radius): the vertical wall that hides the gap to the sphere. */
const DEFAULT_SKIRT_FRAC = 0.03;
/** CDLOD morph band as a fraction of a level's half-extent: native below LO, fully morphed to the
 *  parent lattice by HI (and at the exact boundary, where cheb = res/2 > HI·res/2, morph = 1). */
const MORPH_LO = 0.62;
const MORPH_HI = 0.95;

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
const smooth = (a: number, b: number, x: number): number => {
  const t = clamp01((x - a) / (b - a || 1e-6));
  return t * t * (3 - 2 * t);
};

export interface TerrainPatchOpts {
  /** The SAME height field the icosphere uses (so the two agree everywhere). */
  height: TerrainHeightSource;
  /** The SAME draped colour node the icosphere uses (so the overlay is invisible). */
  colorNode: TerrainColorNode;
  /** Cells per level side (default 128). */
  res?: number;
  /** Nested clipmap levels (default 6). */
  levels?: number;
  /** Outer half-extent as a fraction of radius (default 0.22). */
  outerExtentFrac?: number;
  /** Peak displacement as a fraction of radius at full ramp (default TERRAIN_HEIGHT_SCALE). Only used
   *  when a caller drives the patch by altitude; when the host passes `ampFrac` directly (nearmesh does,
   *  reading it back from the icosphere so the two match exactly) this is ignored. */
  heightScale?: number;
}

export interface TerrainPatchUpdate {
  /** Planet centre, world. */
  cx: number;
  cy: number;
  cz: number;
  /** Planet world radius. */
  radius: number;
  /** Camera world position (defines the sub-point the patch follows + the altitude). */
  camX: number;
  camY: number;
  camZ: number;
  /** Colour detail ramp (0→1), the SAME value the icosphere/instanced sphere uses. */
  detail: number;
  /** Displacement amplitude as a fraction of radius — pass the icosphere's `ampFrac` so the patch and
   *  the sphere displace identically (seamless). When omitted it is derived from the altitude ramp. */
  ampFrac?: number;
  /** The node's field tint (linear RGB). */
  tintR: number;
  tintG: number;
  tintB: number;
}

export interface TerrainPatch {
  object: Object3D;
  update(u: TerrainPatchUpdate): void;
  setVisible(on: boolean): void;
  readonly visible: boolean;
  /** Total triangle count (all visible levels + skirt). */
  readonly triangles: number;
  /** Debug: render the patch as a wireframe (proves the triangle concentration). */
  setWireframe(on: boolean): void;
  dispose(): void;
}

/* ---- per-level geometry (built once, in unitless CELL coordinates) ------------------------------- */
/** A clipmap level as a static grid of `(res+1)²` vertices carrying `aCell` (integer lattice offset in
 *  [-res/2, res/2]) and `aSkirt` (0, or 1 for the dropped outer-rim wall). For k>0 the inner `res/2`
 *  square is left as a hole (filled by the finer level k−1). The outermost level also gets a skirt ring.
 *  Positions are dummy (the vertex shader computes the real position from `aCell`); a fixed oversized
 *  bounding sphere + frustumCulled=false keeps three from culling it. */
function buildLevelGeometry(res: number, hole: boolean, skirt: boolean): {
  geo: BufferGeometry;
  triangles: number;
} {
  const half = res / 2;
  const stride = res + 1;
  const vid = (i: number, j: number): number => j * stride + i;
  // vertex attributes for the (res+1)² base lattice
  const cell: number[] = [];
  const skirtFlag: number[] = [];
  const pos: number[] = [];
  for (let j = 0; j <= res; j++) {
    for (let i = 0; i <= res; i++) {
      cell.push(i - half, j - half);
      skirtFlag.push(0);
      pos.push(0, 0, 0);
    }
  }
  const idx: number[] = [];
  const innerHole = half / 2; // cells: the finer level covers [-res/4, res/4]
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) {
      if (hole) {
        // skip cells that lie ENTIRELY inside the hole (so the finer ring shows through cleanly)
        const cx0 = i - half,
          cx1 = i + 1 - half,
          cy0 = j - half,
          cy1 = j + 1 - half;
        const cheb = Math.max(Math.abs(cx0), Math.abs(cx1), Math.abs(cy0), Math.abs(cy1));
        if (cheb <= innerHole) continue;
      }
      const a = vid(i, j);
      const b = vid(i + 1, j);
      const c = vid(i + 1, j + 1);
      const d = vid(i, j + 1);
      idx.push(a, b, d, b, c, d);
    }
  }
  let triangles = idx.length / 3;

  if (skirt) {
    // walk the outer boundary loop (cheb == half) clockwise and drop a wall of skirt quads. Each rim
    // vertex gets a duplicate flagged aSkirt=1; the shader pushes those radially inward by uSkirt.
    const loop: number[] = [];
    for (let i = 0; i <= res; i++) loop.push(vid(i, 0)); // bottom edge, +i
    for (let j = 1; j <= res; j++) loop.push(vid(res, j)); // right edge
    for (let i = res - 1; i >= 0; i--) loop.push(vid(i, res)); // top edge, -i
    for (let j = res - 1; j >= 1; j--) loop.push(vid(0, j)); // left edge
    // create a skirt duplicate for each rim vertex
    const skirtOf = new Map<number, number>();
    for (const rv of loop) {
      const ni = cell.length / 2;
      cell.push(cell[rv * 2]!, cell[rv * 2 + 1]!);
      skirtFlag.push(1);
      pos.push(0, 0, 0);
      skirtOf.set(rv, ni);
    }
    for (let n = 0; n < loop.length; n++) {
      const rv = loop[n]!;
      const rn = loop[(n + 1) % loop.length]!;
      const sv = skirtOf.get(rv)!;
      const sn = skirtOf.get(rn)!;
      // two triangles of the vertical wall (wound both ways is irrelevant: material is DoubleSide)
      idx.push(rv, rn, sn, rv, sn, sv);
      triangles += 2;
    }
  }

  const geo = new BufferGeometry();
  geo.setAttribute('position', new Float32BufferAttribute(new Float32Array(pos), 3));
  geo.setAttribute('aCell', new Float32BufferAttribute(new Float32Array(cell), 2));
  geo.setAttribute('aSkirt', new Float32BufferAttribute(new Float32Array(skirtFlag), 1));
  geo.setIndex(idx);
  geo.boundingSphere = null;
  return { geo, triangles };
}

/* ---- the patch subsystem ------------------------------------------------------------------------- */

export function createTerrainPatch(opts: TerrainPatchOpts): TerrainPatch {
  const res = Math.max(4, (opts.res ?? DEFAULT_RES) | 0) & ~1; // force even
  const levels = Math.max(1, (opts.levels ?? DEFAULT_LEVELS) | 0);
  const outerFrac = opts.outerExtentFrac ?? DEFAULT_OUTER_EXTENT_FRAC;
  const heightScale = Number.isFinite(opts.heightScale)
    ? (opts.heightScale as number)
    : TERRAIN_HEIGHT_SCALE;
  const half = res / 2;

  // c0 (level-0 cell) as a fraction of radius so the outermost level matches the sphere:
  //   outerExtent = half·c_{L-1} = half·c0·2^{L-1}  ⇒  c0Frac = outerFrac·2 / (res·2^{L-1}).
  const c0Frac = (outerFrac * 2) / (res * Math.pow(2, levels - 1));

  /* ---- shared uniforms (one set, every level's material reads them) ---- */
  const uAnchor = uniform(new Vector3(0, 1, 0)); // the fixed tangent-frame origin (unit dir)
  const uT = uniform(new Vector3(1, 0, 0)); // tangent basis
  const uB = uniform(new Vector3(0, 0, 1));
  const uRadius = uniform(1);
  const uBaseCell = uniform(0); // level-0 cell size in world metres (= c0Frac·radius)
  const uAmp = uniform(0); // displacement fraction (matches the icosphere)
  const uDetail = uniform(0);
  const uBias = uniform(0); // outward depth bias, metres
  const uSkirt = uniform(0); // skirt depth, metres
  const uTint = uniform(new Vector3(1, 1, 1));
  // per-level snapped centres (metres, in the anchor tangent frame). levels+1 so the outermost level
  // has a (virtual, coarser) parent to morph toward.
  const centerU: ReturnType<typeof uniform>[] = [];
  for (let k = 0; k <= levels; k++) centerU.push(uniform(new Vector2(0, 0)));

  /* ---- the shared displacement+colour material, specialised per level (k is a compile-time const) -- */
  const makeMaterial = (k: number): MeshBasicNodeMaterial => {
    const mat = new MeshBasicNodeMaterial();
    mat.transparent = false;
    mat.depthWrite = true;
    mat.depthTest = true;
    // DoubleSide for the same reason terrain.ts is: when the near plane cuts into a ridge the back
    // faces must still read as solid ground, not a see-through hole.
    mat.side = DoubleSide;

    const aCell = attribute('aCell', 'vec2') as unknown as Node<'vec2'>;
    const aSkirt = attribute('aSkirt', 'float') as unknown as FNode;
    const cK = (uBaseCell as unknown as FNode).mul(float(Math.pow(2, k))); // this level's cell (m)
    const cXY = aCell as unknown as { x: FNode; y: FNode };

    // CDLOD morph weight from the level-local chebyshev lattice distance (exact at the boundary = 1).
    const rCheb = maxTSL(abs(cXY.x), abs(cXY.y)) as unknown as FNode;
    const morphK = smoothstepTSL(
      float(MORPH_LO * half),
      float(MORPH_HI * half),
      rCheb,
    ) as unknown as FNode;

    const ck0 = centerU[k] as unknown as { x: FNode; y: FNode };
    const ck1 = centerU[k + 1] as unknown as { x: FNode; y: FNode };
    // native offset (this level's grid) vs parent offset (even-snapped cell on the next-coarser grid)
    const nx = ck0.x.add(cXY.x.mul(cK)) as unknown as FNode;
    const ny = ck0.y.add(cXY.y.mul(cK)) as unknown as FNode;
    const pcx = roundTSL(cXY.x.mul(0.5)).mul(2) as unknown as FNode;
    const pcy = roundTSL(cXY.y.mul(0.5)).mul(2) as unknown as FNode;
    const px = ck1.x.add(pcx.mul(cK)) as unknown as FNode;
    const py = ck1.y.add(pcy.mul(cK)) as unknown as FNode;
    const ox = mix(nx, px, morphK) as unknown as FNode;
    const oy = mix(ny, py, morphK) as unknown as FNode;

    // tangent offset → sphere direction (azimuthal-equidistant exponential map) → displaced local pos.
    const R = uRadius as unknown as FNode;
    const anchor = uAnchor as unknown as V3Node;
    const dirAt = (offx: FNode, offy: FNode): V3Node => {
      const w = (uT as unknown as V3Node).mul(offx).add((uB as unknown as V3Node).mul(offy));
      const d = w.length() as unknown as FNode;
      const theta = d.div(R) as unknown as FNode;
      const safe = maxTSL(d, float(1e-9)) as unknown as FNode;
      const tang = w.div(safe) as unknown as V3Node;
      return anchor
        .mul(cosTSL(theta))
        .add(tang.mul(sinTSL(theta))) as unknown as V3Node;
    };
    const dispAlong = (dir: V3Node): V3Node => {
      const h = opts.height.node(dir, uDetail as unknown as FNode);
      const r = R.mul(float(1).add((uAmp as unknown as FNode).mul(h)))
        .add(uBias as unknown as FNode) as unknown as FNode;
      return dir.mul(r) as unknown as V3Node;
    };

    const dir0 = dirAt(ox, oy);
    const p0 = dispAlong(dir0);
    // analytic normal: two finite-difference taps one cell apart along the tangent axes
    const eps = cK;
    const pA = dispAlong(dirAt(ox.add(eps), oy));
    const pB = dispAlong(dirAt(ox, oy.add(eps)));
    let nrm = normalize(cross(pA.sub(p0), pB.sub(p0))) as unknown as V3Node;
    nrm = nrm.mul(sign(dot(nrm, dir0))) as unknown as V3Node; // force outward

    // skirt vertices drop radially inward to curtain the rim→sphere gap.
    const finalPos = p0.sub(dir0.mul(aSkirt.mul(uSkirt as unknown as FNode))) as unknown as V3Node;
    mat.positionNode = finalPos as unknown as typeof mat.positionNode;

    const vDir = varying(dir0) as unknown as V3Node;
    const vNrm = varying(nrm) as unknown as V3Node;
    mat.colorNode = Fn(() =>
      opts.colorNode(vDir, vNrm, uDetail as unknown as FNode, uTint as unknown as V3Node),
    )() as unknown as typeof mat.colorNode;
    return mat;
  };

  /* ---- the meshes (one per level) ---- */
  const group = new Group();
  group.renderOrder = 3; // after the icosphere terrain (renderOrder 2) so the overlay wins ties
  const mats: MeshBasicNodeMaterial[] = [];
  const meshes: Mesh[] = [];
  let totalTris = 0;
  for (let k = 0; k < levels; k++) {
    const { geo, triangles } = buildLevelGeometry(res, k > 0, k === levels - 1);
    totalTris += triangles;
    const mat = makeMaterial(k);
    mats.push(mat);
    const mesh = new Mesh(geo, mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = 3;
    mesh.visible = false;
    meshes.push(mesh);
    group.add(mesh);
  }

  /* ---- anchor + per-frame state ---- */
  const anchorDir = new Vector3(0, 1, 0);
  const anchorT = new Vector3(1, 0, 0);
  const anchorB = new Vector3(0, 0, 1);
  let anchored = false;
  const subVec = new Vector3();
  const subDir = new Vector3();
  const tv = new Vector3();
  let visible = false;

  const setAnchor = (dir: Vector3): void => {
    anchorDir.copy(dir);
    const ref =
      Math.abs(dir.y) > 0.9 ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0);
    anchorT.copy(ref).cross(anchorDir).normalize();
    anchorB.copy(anchorDir).cross(anchorT).normalize();
    anchored = true;
  };

  // project a unit direction into the anchor tangent frame (azimuthal-equidistant, metres).
  const projXY = (dir: Vector3, radius: number): [number, number] => {
    const cosang = Math.max(-1, Math.min(1, anchorDir.dot(dir)));
    const ang = Math.acos(cosang);
    tv.copy(dir).addScaledVector(anchorDir, -cosang);
    const tl = tv.length();
    if (tl < 1e-9) return [0, 0];
    tv.multiplyScalar(1 / tl);
    const arc = ang * radius;
    return [tv.dot(anchorT) * arc, tv.dot(anchorB) * arc];
  };

  const hide = (): void => {
    if (!visible) return;
    for (const m of meshes) m.visible = false;
    visible = false;
  };

  return {
    object: group,
    update(u: TerrainPatchUpdate): void {
      const radius = u.radius;
      subVec.set(u.camX - u.cx, u.camY - u.cy, u.camZ - u.cz);
      const len = subVec.length();
      if (len < 1e-6 || radius <= 0) {
        hide();
        return;
      }
      subDir.copy(subVec).multiplyScalar(1 / len);
      const altFrac = (len - radius) / radius;

      // amplitude: prefer the icosphere's exact ampFrac; else derive from the altitude ramp.
      const amp = Number.isFinite(u.ampFrac)
        ? (u.ampFrac as number)
        : heightScale * (1 - smooth(TERRAIN_ALT_FULL, TERRAIN_ALT_ON, altFrac));

      // re-anchor only when the sub-point has drifted near the patch rim (rare: a vertical descent
      // barely moves it). Keeping the anchor fixed over a wide region is what makes the sampling
      // world-stable (no swim); the per-level centre snap below is what follows the camera.
      const cLast = c0Frac * radius * Math.pow(2, levels - 1); // outermost cell
      const reanchor = 0.5 * half * cLast; // half the outer half-extent
      if (!anchored) setAnchor(subDir);
      else {
        const [tpx, tpy] = projXY(subDir, radius);
        if (Math.max(Math.abs(tpx), Math.abs(tpy)) > reanchor) setAnchor(subDir);
      }
      const [pxm, pym] = projXY(subDir, radius);

      // per-level snapped centres (each level snaps to its own 2×cell grid ⇒ no swimming at that scale)
      const c0 = c0Frac * radius;
      for (let k = 0; k <= levels; k++) {
        const ck = c0 * Math.pow(2, k);
        const snap = 2 * ck;
        const sx = Math.round(pxm / snap) * snap;
        const sy = Math.round(pym / snap) * snap;
        (centerU[k]!.value as Vector2).set(sx, sy);
      }

      uAnchor.value.copy(anchorDir);
      uT.value.copy(anchorT);
      uB.value.copy(anchorB);
      uRadius.value = radius;
      uBaseCell.value = c0;
      uAmp.value = amp;
      uDetail.value = u.detail;
      uBias.value = Math.max(DEFAULT_BIAS_FRAC, BIAS_AMP_K * amp) * radius;
      uSkirt.value = DEFAULT_SKIRT_FRAC * radius;
      (uTint.value as Vector3).set(u.tintR, u.tintG, u.tintB);

      for (const m of meshes) {
        m.position.set(u.cx, u.cy, u.cz);
        m.visible = true;
      }
      visible = true;
    },
    setVisible(on: boolean): void {
      if (on === visible) return;
      if (on) {
        for (const m of meshes) m.visible = true;
        visible = true;
      } else hide();
    },
    get visible(): boolean {
      return visible;
    },
    get triangles(): number {
      return visible ? totalTris : 0;
    },
    setWireframe(on: boolean): void {
      for (const m of mats) m.wireframe = on;
    },
    dispose(): void {
      hide();
      for (const m of meshes) {
        group.remove(m);
        m.geometry.dispose();
      }
      for (const m of mats) m.dispose();
    },
  };
}
