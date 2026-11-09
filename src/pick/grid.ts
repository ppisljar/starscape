/* Picking and proximity, from a uniform spatial grid over the CURRENT node positions
 * (DESIGN.md §6). A CPU index is the right answer here rather than a compromise because it is
 * synchronous: a click resolves in the same event with zero frames of latency, which is what
 * "game-like" actually means at the pointer.
 *
 * REBUILD STORY (§6): the grid is rebuilt on morph completion — never per frame. While a
 * morph, spin, or spread is in flight, positions move but the index does not; the caller
 * passes `drift`, an upper bound on how far any node can have moved since the grid was built,
 * and every walk widens by exactly that much. Correct at any staleness, and the cost is paid
 * only while something is actually moving. The predecessor learned three things the hard way,
 * all kept here:
 *
 *   * the index is built from the positions being indexed, never from a frozen extent — a
 *     frozen bounding box does not contain a spread field, and clamping every node into the
 *     boundary cells makes the index worse than useless;
 *   * there is ONE grid per field. It was two for a while, the rebuilds updated one and the
 *     picker read the other, and clicking used boot-time positions for the whole session;
 *   * a hit radius expressed in WORLD units is meaningless — the same world radius is 1 pixel
 *     at one depth and 50 at another. The test is a CONE: the tolerance at view depth t is
 *     exactly the world size that `hitPx` CSS pixels subtend at t, derived from the same
 *     projection the vertex shader uses. The hit area is then the drawn disc, by construction
 *     (one CSS-pixel unit shared by shader, picker and ladder — DESIGN.md §5).
 *
 * And the pick keeps the nearest hit *along the ray*, not the nearest *to* the ray: in 3D
 * those are different questions and only the first one is "which node did I click".
 *
 * House rules honoured: typed arrays only — the grid is a counting sort into CSR-style cells
 * (count, prefix-sum, scatter), no arrays-of-arrays, no per-node objects or closures (§10.2);
 * reductions over node-length arrays are loops (§10.3); nothing here draws randomness at all.
 */

import { FALLOFF, NODE_MAX_PX, NODE_MIN_PX, NODE_SCALE } from '../field/constants';

/** A point or unit direction in world space. */
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** Uniform spatial grid over `n` positions: counting-sorted CSR cells. Holds the positions
 *  array BY REFERENCE — walks read live values, which is what makes `drift` sufficient for a
 *  stale index (the bucketing is stale, the coordinates tested are not). */
export interface Grid {
  /** The indexed positions (3n floats), by reference — reads see current values. */
  positions: Float32Array;
  n: number;
  /** Cells per axis; the grid is side³ cells. */
  side: number;
  nCells: number;
  minX: number;
  minY: number;
  minZ: number;
  spanX: number;
  spanY: number;
  spanZ: number;
  /** Per-axis cell edge length (span / side). */
  cellX: number;
  cellY: number;
  cellZ: number;
  /** max(cellX, cellY, cellZ) — the walk's step and ring unit. */
  cellMax: number;
  /** nCells + 1 CSR offsets into `order`; cell k's members are order[start[k]..start[k+1]). */
  start: Uint32Array;
  /** Node indices, counting-sorted by cell key. */
  order: Uint32Array;
  /** Visited-cell stamps; bumping `epoch` begins a new walk with no clearing pass. */
  stamp: Uint32Array;
  epoch: number;
}

/** Sizing defaults shared with the field material — the picker's hit disc must be the drawn
 *  disc, so these ARE the field's constants, imported rather than restated (DESIGN.md §5: one
 *  CSS-pixel unit, constants in one place; three copies of a constant is three chances to
 *  disagree). A renderer with different sizing passes its own via `PickOptions`. `dRef` has no
 *  default on purpose: the field derives it from the live corpus diagonal (× D_REF_FRACTION),
 *  so any fixed number here would be a second, disagreeing pixel unit. */
export const PICK_DEFAULTS = {
  hitPx: 9,
  sizeScale: NODE_SCALE,
  minPx: NODE_MIN_PX,
  maxPx: NODE_MAX_PX,
  falloff: FALLOFF,
} as const;

/** Average nodes per cell the automatic cell sizing aims for. */
export const TARGET_PER_CELL = 4;

const unitSize = (_i: number): number => 1;

function clampCell(v: number, side: number): number {
  const c = v | 0;
  return c < 0 ? 0 : c >= side ? side - 1 : c;
}

/**
 * Build the grid over `positions` (3n floats). The extent is measured from the positions
 * themselves in one O(n) pass — cheaper than being wrong (see the header).
 *
 * @param cell  optional cell edge length in world units; when omitted, cells are sized so
 *              they average ~{@link TARGET_PER_CELL} nodes (the predecessor's heuristic).
 */
export function buildGrid(positions: Float32Array, n: number, cell?: number): Grid {
  const pad = 1e-3;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3]!;
    const y = positions[i * 3 + 1]!;
    const z = positions[i * 3 + 2]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) {
    minX = minY = minZ = 0;
    maxX = maxY = maxZ = 1;
  }
  minX -= pad; minY -= pad; minZ -= pad;
  maxX += pad; maxY += pad; maxZ += pad;
  const spanX = Math.max(maxX - minX, 2 * pad);
  const spanY = Math.max(maxY - minY, 2 * pad);
  const spanZ = Math.max(maxZ - minZ, 2 * pad);

  let side: number;
  if (cell !== undefined && cell > 0) {
    side = Math.max(1, Math.round(Math.max(spanX, spanY, spanZ) / cell));
  } else {
    const cells = Math.max(1, Math.round(n / TARGET_PER_CELL));
    side = Math.max(1, Math.round(Math.cbrt(cells)));
  }
  const cellX = spanX / side;
  const cellY = spanY / side;
  const cellZ = spanZ / side;
  const cellMax = Math.max(cellX, cellY, cellZ);
  const nCells = side * side * side;

  // counting sort into CSR: count, prefix-sum, scatter. No arrays-of-arrays.
  const count = new Uint32Array(nCells);
  const keyOf = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const ix = clampCell((positions[i * 3]! - minX) / cellX, side);
    const iy = clampCell((positions[i * 3 + 1]! - minY) / cellY, side);
    const iz = clampCell((positions[i * 3 + 2]! - minZ) / cellZ, side);
    const k = (iz * side + iy) * side + ix;
    keyOf[i] = k;
    count[k] = count[k]! + 1;
  }
  const start = new Uint32Array(nCells + 1);
  for (let c = 0; c < nCells; c++) start[c + 1] = start[c]! + count[c]!;
  const order = new Uint32Array(n);
  const cur = start.slice(0, nCells);
  for (let i = 0; i < n; i++) {
    const k = keyOf[i]!;
    const at = cur[k]!;
    order[at] = i;
    cur[k] = at + 1;
  }

  return {
    positions, n, side, nCells,
    minX, minY, minZ, spanX, spanY, spanZ,
    cellX, cellY, cellZ, cellMax,
    start, order,
    stamp: new Uint32Array(nCells),
    epoch: 0,
  };
}

/** The world radius that `px` CSS pixels subtend at view depth `t` — the one pixel↔world
 *  conversion the picker shares with the shader (DESIGN.md §5). */
export function pxToWorld(px: number, t: number, tanHalf: number, halfH: number): number {
  return (px * tanHalf * t) / halfH;
}

/** Per-cell maximum of a per-node channel (stage 9): one float per cell, from one O(n) pass
 *  over the CSR cells. The ladder's candidate sweep prunes whole cells with it — a cell whose
 *  LARGEST node cannot project over the admission floor at the cell's nearest possible depth
 *  holds no candidate at all, so its members are never touched. Rebuild it whenever the grid
 *  is rebuilt or the channel is rewritten (both events, never per frame). Reuses `out` when
 *  it is at least nCells long (rebuilds under spin must not allocate a fresh megabyte each
 *  time); empty cells read 0. */
export function cellMaxOf(
  grid: Grid,
  valueOf: (i: number) => number,
  out?: Float32Array,
): Float32Array {
  const res = out !== undefined && out.length >= grid.nCells ? out : new Float32Array(grid.nCells);
  for (let k = 0; k < grid.nCells; k++) {
    let mx = 0;
    const hi = grid.start[k + 1]!;
    for (let q = grid.start[k]!; q < hi; q++) {
      const v = valueOf(grid.order[q]!);
      if (v > mx) mx = v;
    }
    res[k] = mx;
  }
  return res;
}

export interface PickOptions {
  /** tan(fovY / 2) of the active camera — same projection the vertex shader uses. */
  tanHalf: number;
  /** Half the viewport height in CSS pixels. */
  halfH: number;
  /** The falloff pivot distance, world units — the field derives it from the LIVE corpus
   *  (diagonal × D_REF_FRACTION), so it is required here: a baked-in default would be a
   *  second, disagreeing pixel unit (see PICK_DEFAULTS). Feed it the field's own value. */
  dRef: number;
  /** Pointer forgiveness in CSS pixels around the drawn disc. */
  hitPx?: number;
  /** Predicate so a filtered-out or faded node cannot be picked — this is what keeps the
   *  pick in step with what is on screen. Null means everything is pickable. */
  isVisible?: ((i: number) => boolean) | null;
  /** Per-node drawn size (world units at `dRef`); one callback, never a per-node array of
   *  objects. Defaults to a constant 1. */
  sizeOf?: (i: number) => number;
  /** Per-node hover/selection size boost — the shader's iBoost, which scales the glyph AND
   *  its clamp bounds, so the hit disc of an emphasised node must grow the same way (feed it
   *  Field.boostOf). Defaults to a constant 1 (nothing emphasised). */
  boostOf?: (i: number) => number;
  /** Shader sizing constants — MUST match the field material's (see PICK_DEFAULTS). */
  sizeScale?: number;
  minPx?: number;
  maxPx?: number;
  falloff?: number;
  /** Upper bound on node movement since the grid was built (see header: the rebuild story). */
  drift?: number;
}

/**
 * Cast a ray and return the front-most node whose drawn disc contains the cursor, or -1.
 *
 * COVERAGE BEFORE DEPTH: hits rank in two classes — a node whose DRAWN DISC actually contains
 * the cursor always beats one that merely falls inside the `hitPx` forgiveness cone, and depth
 * only breaks ties within a class. Ranking on depth alone lets a node 8 px off the cursor
 * steal the pick from the one directly under it just by being slightly nearer — the
 * predecessor measured that at 45 of 330 probes.
 *
 * @param origin  ray origin (world)
 * @param dir     unit ray direction
 */
export function pickRay(grid: Grid, origin: Vec3, dir: Vec3, opts: PickOptions): number {
  const {
    hitPx = PICK_DEFAULTS.hitPx,
    tanHalf, halfH, dRef,
    isVisible = null,
    sizeScale = PICK_DEFAULTS.sizeScale,
    minPx = PICK_DEFAULTS.minPx,
    maxPx = PICK_DEFAULTS.maxPx,
    drift = 0,
    falloff = PICK_DEFAULTS.falloff,
    sizeOf = unitSize,
    boostOf = unitSize,
  } = opts;
  const pos = grid.positions;
  const step = grid.cellMax * 0.75;

  /* How far to march, from the INDEXED BOX rather than from a stored diagonal. The
     predecessor's old bound was the frozen layout's diagonal × 1.6, and the camera sits at a
     distance that scales with spread — past about 2.6× spread the walk ended before reaching
     anything and nothing was clickable at all. A slab test against the grid's own box also
     skips the empty space in front of the field instead of marching through it. */
  /* The box is over node CENTRES, and a hit is the node's drawn DISC — which can be hundreds
     of world units wide. Testing the ray against the bare box rejects a legitimate click on
     the outer edge of a node at the rim, and rejects everything for a field thin in one axis
     (2D mode, where one span is nearly zero). So inflate by an upper bound on the hit radius:
     the radius a max-size glyph plus the slop subtends at the FURTHEST the box can be, since
     hit radius grows with depth and nothing in the box is deeper than that. */
  let farCorner2 = 0;
  for (let c = 0; c < 8; c++) {
    const x = grid.minX + ((c & 1) !== 0 ? grid.spanX : 0) - origin.x;
    const y = grid.minY + ((c & 2) !== 0 ? grid.spanY : 0) - origin.y;
    const z = grid.minZ + ((c & 4) !== 0 ? grid.spanZ : 0) - origin.z;
    const d2 = x * x + y * y + z * z;
    if (d2 > farCorner2) farCorner2 = d2;
  }
  const pad = drift + pxToWorld(maxPx + hitPx, Math.sqrt(farCorner2), tanHalf, halfH);

  let t0 = 0;
  let t1 = Infinity;
  for (let a = 0; a < 3; a++) {
    const o = a === 0 ? origin.x : a === 1 ? origin.y : origin.z;
    const d = a === 0 ? dir.x : a === 1 ? dir.y : dir.z;
    const mn = a === 0 ? grid.minX : a === 1 ? grid.minY : grid.minZ;
    const sp = a === 0 ? grid.spanX : a === 1 ? grid.spanY : grid.spanZ;
    const lo = mn - pad;
    const hi = mn + sp + pad;
    if (Math.abs(d) < 1e-9) {
      if (o < lo || o > hi) return -1;
      continue;
    }
    let near = (lo - o) / d;
    let far = (hi - o) / d;
    if (near > far) {
      const s = near;
      near = far;
      far = s;
    }
    if (near > t0) t0 = near;
    if (far < t1) t1 = far;
    if (t0 > t1) return -1; // the ray misses the field entirely
  }
  if (t1 < 0) return -1;
  t0 = Math.max(0, t0 - step);
  const maxT = Math.min(t1 + step, t0 + grid.cellMax * 4096); // runaway guard

  const epoch = ++grid.epoch;
  // Two classes, ranked in this order (see the doc comment above).
  let best = -1, bestT = Infinity; // covers the pixel outright
  let near = -1, nearT = Infinity; // inside the slop only

  for (let t = t0; t < maxT; t += step) {
    if (t > Math.min(bestT, nearT) + grid.cellMax) break;
    const px = origin.x + dir.x * t;
    const py = origin.y + dir.y * t;
    const pz = origin.z + dir.z * t;
    const tol = Math.max(pxToWorld(hitPx, Math.max(t, 1), tanHalf, halfH), grid.cellMax * 0.05);
    // `drift` is why a rotating field stays clickable: a node bucketed before the field
    // turned can be up to `drift` from where its cell says it is, so searching `drift`
    // further out is exactly the widening that guarantees we still find it.
    // `rr` can dwarf the grid when cells are tiny relative to the pixel tolerance (a
    // one-node field has cells ~2e-3 across); every offset outside the grid is rejected
    // anyway, so clamp the loop ranges to the cells that exist instead of iterating
    // billions of misses.
    const rr = Math.ceil((tol + step + drift) / grid.cellMax);
    const ix = clampCell((px - grid.minX) / grid.cellX, grid.side);
    const iy = clampCell((py - grid.minY) / grid.cellY, grid.side);
    const iz = clampCell((pz - grid.minZ) / grid.cellZ, grid.side);
    const aLo = Math.max(-rr, -iz), aHi = Math.min(rr, grid.side - 1 - iz);
    const bLo = Math.max(-rr, -iy), bHi = Math.min(rr, grid.side - 1 - iy);
    const cLo = Math.max(-rr, -ix), cHi = Math.min(rr, grid.side - 1 - ix);
    for (let a = aLo; a <= aHi; a++) {
      for (let b = bLo; b <= bHi; b++) {
        for (let c = cLo; c <= cHi; c++) {
          const xx = ix + c;
          const yy = iy + b;
          const zz = iz + a;
          const k = (zz * grid.side + yy) * grid.side + xx;
          if (grid.stamp[k] === epoch) continue;
          grid.stamp[k] = epoch;
          const lo = grid.start[k]!;
          const hi = grid.start[k + 1]!;
          for (let q = lo; q < hi; q++) {
            const i = grid.order[q]!;
            if (isVisible !== null && !isVisible(i)) continue;
            const dx = pos[i * 3]! - origin.x;
            const dy = pos[i * 3 + 1]! - origin.y;
            const dz = pos[i * 3 + 2]! - origin.z;
            const tt = dx * dir.x + dy * dir.y + dz * dir.z; // depth along the ray
            if (tt <= 0 || (tt >= bestT && tt >= nearT)) continue;
            const ex = dx - dir.x * tt;
            const ey = dy - dir.y * tt;
            const ez = dz - dir.z * tt;
            const perp2 = ex * ex + ey * ey + ez * ez;
            // The node's own drawn radius at this depth — character for character the vertex
            // shader's sizing, including `falloff` and its fixed reference depth `dRef`, so
            // the hit area is the disc you can see, by construction. At falloff 1 the dRef
            // terms cancel and this reduces to the plain projection; away from 1 they do
            // not, and a hit area computed without them is a different disc from the one on
            // screen. The boost scales the radius AND both clamp rails (the shader's
            // `clamp(px, iBoost·MIN, iBoost·MAX)`) — a hovered node pinned at NODE_MAX_PX
            // draws boost× larger, and clicking its outer annulus must still hit it.
            const bst = boostOf(i);
            const base = (sizeOf(i) * bst * sizeScale * halfH) / (tanHalf * dRef);
            const drawnPx = Math.min(
              maxPx * bst,
              Math.max(minPx * bst, base * Math.pow(dRef / tt, falloff)),
            );
            const rHit = pxToWorld(drawnPx, tt, tanHalf, halfH);
            if (perp2 <= rHit * rHit) {
              if (tt < bestT) {
                best = i;
                bestT = tt;
              }
            } else {
              const rSlop = pxToWorld(drawnPx + hitPx, tt, tanHalf, halfH);
              if (perp2 <= rSlop * rSlop && tt < nearT) {
                near = i;
                nearT = tt;
              }
            }
          }
        }
      }
    }
  }
  return best >= 0 ? best : near;
}

export interface NearestOptions {
  /** Same visibility predicate as the picker — room is measured to what is on screen. */
  isVisible?: ((i: number) => boolean) | null;
  /** Upper bound on node movement since the grid was built. */
  drift?: number;
}

/**
 * Distance from `pos` to the nearest node ANYWHERE in the field — the flycam's `roomAt`
 * callback consumes this (speed scales with free room ahead; proximity damps toward a floor).
 * Unlike the predecessor's ring-capped version this never gives up early: shells expand until
 * the whole indexed box is covered, so a camera far outside the field still gets the true
 * distance rather than Infinity. Returns Infinity only for an empty (or fully hidden) field.
 *
 * This runs once or twice EVERY FRAME (the flycam probes the current position and the step
 * destination), so the walk must stay cheap at the §9 home framing — camera well OUTSIDE the
 * indexed box — not just deep inside a dense field. Three properties keep it there:
 *
 *   * SHELLS, not cubes: iteration k visits only the cells at Chebyshev index-distance
 *     exactly k from the probe's (clamped) cell, so expanding the search never re-reads a
 *     cell — the predecessor-shaped cube-per-ring walk rescanned the entire clamped volume
 *     on every iteration, which from outside the box meant repeated O(n) full-field sweeps.
 *   * Per-cell pruning: a cell whose own box is further than best + drift cannot improve the
 *     answer, so its members are never distance-tested. Once a candidate lands, lateral cells
 *     cost one AABB check each.
 *   * A EUCLIDEAN stop bound: after shell k, any unscanned cell lies beyond the scanned slab
 *     along some axis AND inside the indexed box in the others, so its distance is at least
 *     √(escape² + the other axes' box distances²) — from outside the box that grows like
 *     √(boxDist² + (k·cell)²), which crosses `best` after a handful of shells, not after
 *     boxDist/cell of them. (A max-norm bound stalls there: it stays ≈ boxDist until
 *     k·cell alone exceeds best.) `drift` widens the stop and the prune, never the shells
 *     themselves — a stale bucket is searched exactly as far as it can have wandered.
 */
export function nearestDistance(grid: Grid, pos: Vec3, opts: NearestOptions = {}): number {
  const { isVisible = null, drift = 0 } = opts;
  if (grid.n === 0) return Infinity;
  const p = grid.positions;
  const side = grid.side;
  const { cellX, cellY, cellZ } = grid;
  const maxX = grid.minX + grid.spanX;
  const maxY = grid.minY + grid.spanY;
  const maxZ = grid.minZ + grid.spanZ;

  // Per-axis distance from pos to the indexed box (0 inside along that axis): the "other
  // axes" term of the stop bound, and the floor every unscanned cell's distance shares.
  const gx = pos.x < grid.minX ? grid.minX - pos.x : pos.x > maxX ? pos.x - maxX : 0;
  const gy = pos.y < grid.minY ? grid.minY - pos.y : pos.y > maxY ? pos.y - maxY : 0;
  const gz = pos.z < grid.minZ ? grid.minZ - pos.z : pos.z > maxZ ? pos.z - maxZ : 0;

  const ix0 = clampCell((pos.x - grid.minX) / cellX, side);
  const iy0 = clampCell((pos.y - grid.minY) / cellY, side);
  const iz0 = clampCell((pos.z - grid.minZ) / cellZ, side);

  let best = Infinity; // squared live distance of the best node so far
  let prune2 = Infinity; // (√best + drift)² — cells beyond this cannot improve best

  // The visit: AABB-prune the cell, then distance-test its members against LIVE positions.
  const scan = (xx: number, yy: number, zz: number): void => {
    const x0 = grid.minX + xx * cellX;
    const y0 = grid.minY + yy * cellY;
    const z0 = grid.minZ + zz * cellZ;
    const dxC = pos.x < x0 ? x0 - pos.x : pos.x > x0 + cellX ? pos.x - (x0 + cellX) : 0;
    const dyC = pos.y < y0 ? y0 - pos.y : pos.y > y0 + cellY ? pos.y - (y0 + cellY) : 0;
    const dzC = pos.z < z0 ? z0 - pos.z : pos.z > z0 + cellZ ? pos.z - (z0 + cellZ) : 0;
    if (dxC * dxC + dyC * dyC + dzC * dzC > prune2) return;
    const k = (zz * side + yy) * side + xx;
    const lo = grid.start[k]!;
    const hi = grid.start[k + 1]!;
    for (let q = lo; q < hi; q++) {
      const i = grid.order[q]!;
      if (isVisible !== null && !isVisible(i)) continue;
      const dx = p[i * 3]! - pos.x;
      const dy = p[i * 3 + 1]! - pos.y;
      const dz = p[i * 3 + 2]! - pos.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) {
        best = d2;
        const r = Math.sqrt(best) + drift; // one sqrt per improvement, not per cell
        prune2 = r * r;
      }
    }
  };

  // Shells stop existing once every axis is exhausted in both directions.
  const maxK = Math.max(ix0, side - 1 - ix0, iy0, side - 1 - iy0, iz0, side - 1 - iz0);

  for (let k = 0; k <= maxK; k++) {
    const aLo = Math.max(-k, -iz0), aHi = Math.min(k, side - 1 - iz0);
    const bLo = Math.max(-k, -iy0), bHi = Math.min(k, side - 1 - iy0);
    const cLo = Math.max(-k, -ix0), cHi = Math.min(k, side - 1 - ix0);
    for (let a = aLo; a <= aHi; a++) {
      const zz = iz0 + a;
      const aFace = a === -k || a === k;
      for (let b = bLo; b <= bHi; b++) {
        const yy = iy0 + b;
        if (aFace || b === -k || b === k) {
          for (let c = cLo; c <= cHi; c++) scan(ix0 + c, yy, zz); // a face row: full strip
        } else {
          // interior row: only the two x-extremes are new at this shell
          if (-k >= cLo) scan(ix0 - k, yy, zz);
          if (k <= cHi) scan(ix0 + k, yy, zz);
        }
      }
    }
    if (best === Infinity) continue; // nothing found yet — the bound cannot stop the walk

    /* The stop bound. After this shell the scanned region is, per axis, the world slab of
       cells [i0−k, i0+k]; an unscanned in-grid cell escapes that slab along at least one
       axis, where its gap from pos is at least the distance to the first unscanned cell
       boundary on whichever side still has cells (exact, including pos's intra-cell offset
       and any outside-the-box overhang). Along the other axes it is still inside the box, so
       those contribute their g-floors. The true lower bound is the min over escape axes. */
    const exHi = ix0 + k + 1 <= side - 1 ? grid.minX + (ix0 + k + 1) * cellX - pos.x : Infinity;
    const exLo = ix0 - k >= 1 ? pos.x - (grid.minX + (ix0 - k) * cellX) : Infinity;
    const eyHi = iy0 + k + 1 <= side - 1 ? grid.minY + (iy0 + k + 1) * cellY - pos.y : Infinity;
    const eyLo = iy0 - k >= 1 ? pos.y - (grid.minY + (iy0 - k) * cellY) : Infinity;
    const ezHi = iz0 + k + 1 <= side - 1 ? grid.minZ + (iz0 + k + 1) * cellZ - pos.z : Infinity;
    const ezLo = iz0 - k >= 1 ? pos.z - (grid.minZ + (iz0 - k) * cellZ) : Infinity;
    const ex = Math.min(exHi, exLo);
    const ey = Math.min(eyHi, eyLo);
    const ez = Math.min(ezHi, ezLo);
    const lb2 = Math.min(
      ex === Infinity ? Infinity : ex * ex + gy * gy + gz * gz,
      ey === Infinity ? Infinity : ey * ey + gx * gx + gz * gz,
      ez === Infinity ? Infinity : ez * ez + gx * gx + gy * gy,
    );
    if (lb2 === Infinity) break; // no escape anywhere: the whole grid is scanned
    const lbAdj = Math.sqrt(lb2) - drift; // a stale node can be up to drift nearer than its cell
    if (lbAdj > 0 && best <= lbAdj * lbAdj) break;
  }
  return best === Infinity ? Infinity : Math.sqrt(best);
}
