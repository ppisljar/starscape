/* Reference frames over the hierarchy pivot tree (DESIGN.md §3.5, §4) — the stage-6 CPU
 * REFERENCE implementation the GPU transform chain mirrors. Ported from vault-site-3d
 * field.js (`resolveFrames` / `recompose` / `frameAt` / `rideChain`) and main.js's ride
 * loop; the pinned invariants of its rotation.test.mjs are the spec.
 *
 * The model: every pivot of a compiled hierarchy (src/place/hierarchy.ts) is a potential
 * rotating frame — a centre, a unit axis, a rate. Frames nest along `pivots.parent`, and
 * the three hard-won rules of the predecessor are load-bearing here:
 *
 *   1. COMPOSE THE ROTATIONS, DO NOT NEST THE LOOPS (README §"Compose the rotations").
 *      Two levels compose as  p' = C + R_shape( A + R_bloom(p − A) − C ):  rotating a
 *      satellite about its UNROTATED anchor and then turning the whole result about the
 *      centre is exactly equivalent to turning the anchor first and carrying its
 *      satellites with it. So `advance` resolves ONE quaternion per PIVOT per tick
 *      (hundreds), never one per node, and positioning a node is a single quaternion
 *      apply however deep the nesting — which is precisely the shape the vertex shader
 *      needs: `quat` / `centre` here are the uniform arrays it indexes by the node's
 *      pivot-chain attribute, and `transformPoint` is the arithmetic it replicates.
 *
 *   2. "IS ANYTHING TURNING" IS NOT "HAS ANYTHING EVER TURNED". Angles are integrated,
 *      so testing the accumulated angle keeps the whole pipeline hot forever once a
 *      frame has spun (the predecessor measured 117 fps still vs 83 fps because of
 *      exactly this). `advance` tests whether an angle MOVED THIS TICK (some rate ≠ 0
 *      and |rate·dt| above epsilon) and skips the entire resolve when none did;
 *      `activeThisTick()` reports that same answer so callers can skip their own work.
 *      `displaced()` is the separate question — "is any frame parked off its base
 *      angle" — which decides whether transforms are still non-identity while idle.
 *
 *   3. RIDE INCREMENTALLY, NEVER BY CANCELLING (README §"Ride incrementally"). Making
 *      the frame you are inside look still by subtracting its ACCUMULATED angle jumps
 *      the instant you cross the boundary, because that angle has been accruing since
 *      load. `rideDelta` instead returns the frame chain's PER-TICK rigid motion — the
 *      rotation the chain underwent during the last `advance`, evaluated at the camera —
 *      so the correction starts at zero on entry and grows only with the frame's own
 *      motion: entering mid-flight is seamless, and so is leaving.
 *
 * O(pivots) BY CONSTRUCTION (§4: per-frame CPU is O(pivots), never O(nodes)): the
 * factory consumes only a PivotTable, whose every array is pivot-length; `advance(dt)`
 * takes one scalar; no method accepts or allocates a node-length array. Nodes follow in
 * the vertex shader. `transformPoint` exists for tests, picking rebuilds and telemetry —
 * per-point work the caller chooses to do, never work `advance` does behind its back.
 *
 * Determinism (§10.1): no RNG, no clock. The caller owns time; the same dt sequence over
 * the same table yields byte-identical `quat` / `centre` / `axisWorld` buffers forever.
 * Angles accumulate in Float64; the exposed buffers are Float32 — the width the GPU
 * uniform upload reads, so the reference rounds exactly where the shader does.
 *
 * Live inputs, owned state: `rate` and `axis` are re-read from the table every advance —
 * a UI spin toggle writes there and the frame parks/resumes from its integrated angle
 * (never snaps to `now × rate`). Axes must stay unit length if rewritten (the hierarchy
 * compiler normalises them once). The integrated angle is this module's own state.
 */

import type { PivotTable } from '../place/hierarchy';

/** Below this, a per-tick angle is "did not move" — the predecessor's 1e-9 threshold,
 *  kept: it is far under one float32 ulp of any visible rotation, and far over the
 *  arithmetic noise of `rate × dt`. */
export const ANGLE_EPS = 1e-9;

/** Writable xyz/quaternion sink — tests hand in plain arrays, the renderer hands in the
 *  slices it uploads. */
export type NumArray = Float32Array | Float64Array | number[];

/** `rideDelta` output layout: 7 numbers. */
export const RIDE_DELTA_SIZE = 7;

export interface FrameSet {
  /** Pivot count — every exposed buffer is this length × 3 or × 4. */
  readonly count: number;
  /** 4p, xyzw — composed WORLD rotation per pivot (own ⊗ all ancestors). The GPU's
   *  quaternion uniform array, rebuilt by `advance` only when something moved. */
  readonly quat: Float32Array;
  /** 3p — current world centre per pivot: the unrotated centre carried by every
   *  ancestor's rotation. Equals `pivots.world` until something turns. */
  readonly centre: Float32Array;
  /** 3p — current world spin axis per pivot (the table's axis, carried by ancestors). */
  readonly axisWorld: Float32Array;
  /**
   * Integrate every spinning pivot's angle by `dt` (caller's time unit — the renderer
   * owns the clock) and, ONLY if some angle actually moved this tick, re-resolve the
   * frame tree: one quaternion per pivot, parents before children (the table guarantees
   * parent id < child id, so one forward pass composes the chain). Returns the same
   * answer `activeThisTick()` then reports. A zero dt, or all rates zero, costs an
   * O(pivots) rate scan and nothing else — the idle scene is free.
   */
  advance(dt: number): boolean;
  /** Did the last `advance` move any angle? (some rate ≠ 0 AND |rate·dt| > ANGLE_EPS.)
   *  False before the first advance, false the moment rates drop to zero — regardless
   *  of how long things spun before. This is the "skip everything downstream" test. */
  activeThisTick(): boolean;
  /** Is any frame parked off its base angle? True after spinning stops (the field stays
   *  composed at the parked rotation); false only when every accumulated angle is ~0. */
  displaced(): boolean;
  /** The composed chain quaternion of `pivot` (world rotation: own ⊗ ancestors), xyzw
   *  into `out` (length ≥ 4). `pivot` −1 (the world) writes identity. */
  quaternionOf(pivot: number, out: NumArray, outOffset?: number): NumArray;
  /**
   * The composed chain transform (§4): where the unrotated world-space point `p` of a
   * node whose pivot chain ends at `pivot` sits NOW. One quaternion apply however deep
   * the chain — `out = centre[pivot] + quat[pivot] ⊗ (p − world0[pivot])` — exactly the
   * vertex-shader arithmetic (p is the endpoint-buffer position, `world0` the pivot's
   * compile-time world centre). `pivot` −1 copies p through untouched.
   */
  transformPoint(pivot: number, p: ArrayLike<number>, out: NumArray, pOffset?: number, outOffset?: number): NumArray;
  /**
   * The deepest pivot whose sphere (current carried centre, compile-time radius)
   * contains `pos`, or −1 for the world. Depth ties break toward the sphere the point
   * is most deeply inside (smallest d/r — the predecessor's frameAt lesson: generous
   * overlapping siblings must not capture a point that barely grazes them). Deepest-
   * first is safe here where it was not in the predecessor because hierarchy radii nest
   * by construction (child ≤ parent, guard-railed in the compiler), so a child sphere
   * never swallows its parent's centre the way tightened bounding spheres did.
   */
  frameOfPoint(pos: ArrayLike<number>, offset?: number): number;
  /**
   * The PER-TICK incremental rigid motion a camera riding `pivot`'s frame applies to
   * itself — rule 3 above: never cancellation of accumulated angle. Walks the chain
   * root → pivot and composes each frame's LAST-advance rotation (rate × last dt, about
   * its current world axis and centre) evaluated at `cameraPos`. Writes RIDE_DELTA_SIZE
   * numbers into `out`: [0..3] the delta quaternion xyzw (premultiply onto the camera
   * orientation), [4..6] the position displacement (add to the camera position). A
   * pivot of −1 (nothing to ride in the world frame), a tick that moved nothing, or no
   * advance yet all yield identity + zero — which is exactly why entry is seamless.
   */
  rideDelta(pivot: number, cameraPos: ArrayLike<number>, out: NumArray, outOffset?: number): NumArray;
}

/** Build a FrameSet over a compiled hierarchy's pivot table. Reads `world`, `parent`,
 *  `level`, `radius` once as structure; `axis` and `rate` live (every advance). All
 *  internal state is pivot-length — nothing here can scale with node count. */
export function createFrameSet(pivots: PivotTable): FrameSet {
  const P = pivots.count;
  const parent = pivots.parent;
  const world0 = pivots.world; // compile-time world centres — frozen input, never written
  const radius = pivots.radius;
  const axis = pivots.axis; // live: unit axes, re-read each resolve
  const rate = pivots.rate; // live: a spin toggle writes here

  /* Owned state — all O(pivots), allocated once. */
  const angle = new Float64Array(P); // integrated angle per pivot (f64: no drift)
  const quat = new Float32Array(P * 4);
  const centre = new Float32Array(P * 3);
  const axisWorld = new Float32Array(P * 3);
  for (let p = 0; p < P; p++) quat[p * 4 + 3] = 1; // identity
  centre.set(world0);
  axisWorld.set(axis);

  let lastDt = 0; // dt of the most recent advance — what rideDelta's increments use
  let active = false; // did the last advance move any angle

  /* Chain scratch for rideDelta, sized by the deepest pivot level (≤ 256: level is a
   * Uint8). Allocated once; rideDelta never allocates. */
  let maxDepth = 1;
  for (let p = 0; p < P; p++) {
    const d = pivots.level[p]! + 1;
    if (d > maxDepth) maxDepth = d;
  }
  const chain = new Int32Array(maxDepth);

  /* ---- quaternion kernels (f64 locals, xyzw) ------------------------------------- */

  const qa = new Float64Array(4); // general-purpose quaternion scratch
  const qb = new Float64Array(4);
  const v3 = new Float64Array(3); // general-purpose vector scratch

  /** out = axis-angle(ax,ay,az unit, ang). */
  function qAxisAngle(ax: number, ay: number, az: number, ang: number, out: Float64Array): void {
    const h = ang * 0.5;
    const s = Math.sin(h);
    out[0] = ax * s;
    out[1] = ay * s;
    out[2] = az * s;
    out[3] = Math.cos(h);
  }

  /** out = a ⊗ b (apply b's rotation first, then a's — Hamilton product). Safe when
   *  `out` aliases `a` or `b`. */
  function qMul(a: Float64Array, b: Float64Array, out: Float64Array): void {
    const ax = a[0]!, ay = a[1]!, az = a[2]!, aw = a[3]!;
    const bx = b[0]!, by = b[1]!, bz = b[2]!, bw = b[3]!;
    out[0] = aw * bx + ax * bw + ay * bz - az * by;
    out[1] = aw * by - ax * bz + ay * bw + az * bx;
    out[2] = aw * bz + ax * by - ay * bx + az * bw;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
  }

  /** v ← q ⊗ v: rotate (v3[0..2]) in place by q. t = 2(q×v); v' = v + w·t + q×t. */
  function qRotate(qx: number, qy: number, qz: number, qw: number, v: Float64Array): void {
    const x = v[0]!, y = v[1]!, z = v[2]!;
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    v[0] = x + qw * tx + (qy * tz - qz * ty);
    v[1] = y + qw * ty + (qz * tx - qx * tz);
    v[2] = z + qw * tz + (qx * ty - qy * tx);
  }

  /* ---- the per-tick resolve: one forward pass, parents precede children ---------- */

  function resolve(): void {
    for (let p = 0; p < P; p++) {
      const a3 = p * 3;
      const a4 = p * 4;
      const ax = axis[a3]!, ay = axis[a3 + 1]!, az = axis[a3 + 2]!;
      // own rotation about the frame's compile-time axis, at its integrated angle
      qAxisAngle(ax, ay, az, angle[p]!, qa);
      const par = parent[p]!;
      if (par < 0) {
        // a root spins in place: centre and axis are its own, rotation is just `own`
        quat[a4] = qa[0]!;
        quat[a4 + 1] = qa[1]!;
        quat[a4 + 2] = qa[2]!;
        quat[a4 + 3] = qa[3]!;
        centre[a3] = world0[a3]!;
        centre[a3 + 1] = world0[a3 + 1]!;
        centre[a3 + 2] = world0[a3 + 2]!;
        axisWorld[a3] = ax;
        axisWorld[a3 + 1] = ay;
        axisWorld[a3 + 2] = az;
        continue;
      }
      // parent already resolved (lower id): centre and axis ride it, rotation composes
      const p3 = par * 3;
      const p4 = par * 4;
      const px = quat[p4]!, py = quat[p4 + 1]!, pz = quat[p4 + 2]!, pw = quat[p4 + 3]!;
      // composed = parent ⊗ own
      qb[0] = px; qb[1] = py; qb[2] = pz; qb[3] = pw;
      qMul(qb, qa, qa);
      quat[a4] = qa[0]!;
      quat[a4 + 1] = qa[1]!;
      quat[a4 + 2] = qa[2]!;
      quat[a4 + 3] = qa[3]!;
      // centre: the compile-time offset from the parent, carried by the parent's
      // rotation, hung off the parent's CURRENT centre
      v3[0] = world0[a3]! - world0[p3]!;
      v3[1] = world0[a3 + 1]! - world0[p3 + 1]!;
      v3[2] = world0[a3 + 2]! - world0[p3 + 2]!;
      qRotate(px, py, pz, pw, v3);
      centre[a3] = v3[0]! + centre[p3]!;
      centre[a3 + 1] = v3[1]! + centre[p3 + 1]!;
      centre[a3 + 2] = v3[2]! + centre[p3 + 2]!;
      // axis: carried by the parent's rotation (a child of a tipped galaxy tips too)
      v3[0] = ax; v3[1] = ay; v3[2] = az;
      qRotate(px, py, pz, pw, v3);
      axisWorld[a3] = v3[0]!;
      axisWorld[a3 + 1] = v3[1]!;
      axisWorld[a3 + 2] = v3[2]!;
    }
  }

  /* ---- the API -------------------------------------------------------------------- */

  /** A pivot id past the table is a caller bug; answering NaN geometry for it would be
   *  the silent-missing-node failure hash.ts documents. Loud instead (§10.5 spirit). */
  function checkPivot(pivot: number): void {
    if (pivot >= P || !Number.isInteger(pivot)) {
      throw new RangeError(`FrameSet: pivot ${pivot} out of range (count ${P})`);
    }
  }

  function advance(dt: number): boolean {
    lastDt = dt;
    // Rule 2: test the per-tick delta, not the accumulated angle. One O(pivots) scan;
    // when nothing moved, nothing else runs — not the integrate, not the resolve.
    let moved = false;
    for (let p = 0; p < P; p++) {
      if (Math.abs(rate[p]! * dt) > ANGLE_EPS) {
        moved = true;
        break;
      }
    }
    active = moved;
    if (!moved) return false;
    for (let p = 0; p < P; p++) angle[p] = angle[p]! + rate[p]! * dt;
    resolve();
    return true;
  }

  function activeThisTick(): boolean {
    return active;
  }

  function displaced(): boolean {
    for (let p = 0; p < P; p++) {
      if (Math.abs(angle[p]!) > ANGLE_EPS) return true;
    }
    return false;
  }

  function quaternionOf(pivot: number, out: NumArray, outOffset = 0): NumArray {
    checkPivot(pivot);
    if (pivot < 0) {
      out[outOffset] = 0;
      out[outOffset + 1] = 0;
      out[outOffset + 2] = 0;
      out[outOffset + 3] = 1;
      return out;
    }
    const a4 = pivot * 4;
    out[outOffset] = quat[a4]!;
    out[outOffset + 1] = quat[a4 + 1]!;
    out[outOffset + 2] = quat[a4 + 2]!;
    out[outOffset + 3] = quat[a4 + 3]!;
    return out;
  }

  function transformPoint(
    pivot: number,
    p: ArrayLike<number>,
    out: NumArray,
    pOffset = 0,
    outOffset = 0,
  ): NumArray {
    checkPivot(pivot);
    const x = p[pOffset]!, y = p[pOffset + 1]!, z = p[pOffset + 2]!;
    if (pivot < 0) {
      out[outOffset] = x;
      out[outOffset + 1] = y;
      out[outOffset + 2] = z;
      return out;
    }
    const a3 = pivot * 3;
    const a4 = pivot * 4;
    // offset from the frame's UNROTATED centre, rotated by the composed chain, hung off
    // the frame's CURRENT centre — the one-apply form of the nested composition
    v3[0] = x - world0[a3]!;
    v3[1] = y - world0[a3 + 1]!;
    v3[2] = z - world0[a3 + 2]!;
    qRotate(quat[a4]!, quat[a4 + 1]!, quat[a4 + 2]!, quat[a4 + 3]!, v3);
    out[outOffset] = v3[0]! + centre[a3]!;
    out[outOffset + 1] = v3[1]! + centre[a3 + 1]!;
    out[outOffset + 2] = v3[2]! + centre[a3 + 2]!;
    return out;
  }

  function frameOfPoint(pos: ArrayLike<number>, offset = 0): number {
    const x = pos[offset]!, y = pos[offset + 1]!, z = pos[offset + 2]!;
    let best = -1;
    let bestLevel = -1;
    let bestScore = Infinity; // d/r among same-level candidates
    for (let p = 0; p < P; p++) {
      const r = radius[p]!;
      const a3 = p * 3;
      const dx = x - centre[a3]!;
      const dy = y - centre[a3 + 1]!;
      const dz = z - centre[a3 + 2]!;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > r) continue;
      const lvl = pivots.level[p]!;
      const score = d / Math.max(r, 1e-6);
      if (lvl > bestLevel || (lvl === bestLevel && score < bestScore)) {
        best = p;
        bestLevel = lvl;
        bestScore = score;
      }
    }
    return best;
  }

  function rideDelta(
    pivot: number,
    cameraPos: ArrayLike<number>,
    out: NumArray,
    outOffset = 0,
  ): NumArray {
    checkPivot(pivot);
    const cx = cameraPos[0]!, cy = cameraPos[1]!, cz = cameraPos[2]!;
    // identity until something in the chain moved this tick
    qb[0] = 0; qb[1] = 0; qb[2] = 0; qb[3] = 1;
    let px = cx, py = cy, pz = cz;
    if (pivot >= 0) {
      // chain root → pivot (walk up, fill backward; depth ≤ maxDepth by construction)
      let d = chain.length;
      for (let f = pivot; f >= 0; f = parent[f]!) chain[--d] = f;
      for (let k = d; k < chain.length; k++) {
        const f = chain[k]!;
        const ang = rate[f]! * lastDt; // THIS tick's increment — rule 3, never accumulated
        if (Math.abs(ang) <= ANGLE_EPS) continue;
        const a3 = f * 3;
        qAxisAngle(axisWorld[a3]!, axisWorld[a3 + 1]!, axisWorld[a3 + 2]!, ang, qa);
        // rotate the carried position about this frame's current centre
        v3[0] = px - centre[a3]!;
        v3[1] = py - centre[a3 + 1]!;
        v3[2] = pz - centre[a3 + 2]!;
        qRotate(qa[0]!, qa[1]!, qa[2]!, qa[3]!, v3);
        px = v3[0]! + centre[a3]!;
        py = v3[1]! + centre[a3 + 1]!;
        pz = v3[2]! + centre[a3 + 2]!;
        qMul(qa, qb, qb); // delta composes outward-in, same order as the positions
      }
    }
    out[outOffset] = qb[0]!;
    out[outOffset + 1] = qb[1]!;
    out[outOffset + 2] = qb[2]!;
    out[outOffset + 3] = qb[3]!;
    out[outOffset + 4] = px - cx;
    out[outOffset + 5] = py - cy;
    out[outOffset + 6] = pz - cz;
    return out;
  }

  return {
    count: P,
    quat,
    centre,
    axisWorld,
    advance,
    activeThisTick,
    displaced,
    quaternionOf,
    transformPoint,
    frameOfPoint,
    rideDelta,
  };
}
