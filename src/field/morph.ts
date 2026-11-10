/* The pure halves of the stage-6 GPU transform pipeline (DESIGN.md §4) — everything here
 * runs headless (vitest, node) with no GPU and no three.js import. The stores/materials
 * consume what these produce; test/morph.test.ts pins the arithmetic.
 *
 * The pipeline these feed, restated from §4:
 *
 *     GPU per frame:  p = spread( chain( mix(A, B, t) ) )
 *
 * with A/B the two endpoint position sets, `t` the morph uniform, `chain` the composed
 * frame-chain transform (one quaternion apply — src/anim/frames.ts rule 1), and `spread`
 * a uniform scale about the field centroid. Per-frame CPU is O(pivots): everything in
 * this file is O(n) but runs on EVENTS (retarget, projection flip, spin-stop snapshot),
 * never per frame.
 *
 * House rules (§10): loops, typed arrays, no RNG, no clock — same input, byte-identical
 * output, pinned by FNV goldens.
 */

import type { FrameSet } from '../anim/frames';

/** The fixed maximum frame-chain depth the shader honours (DESIGN §3.4: start at 4, a
 *  constant to raise). Levels at or below this depth spin; deeper levels stop spinning
 *  INDEPENDENTLY — their nodes ride the deepest ancestor frame within the cap (packed by
 *  {@link packChainIds}), which is exactly the §3.4 contract. Placement depth itself is
 *  unlimited. */
export const CHAIN_CAP = 4;

/**
 * Pack the per-node pivot-chain attribute (§4: "frame quaternions are a uniform
 * array/texture indexed by a per-node pivot-chain attribute").
 *
 * ONE id per node, not four: the FrameSet already composes every ancestor rotation into
 * each pivot's world quaternion (frames.ts rule 1 — compose the rotations, do not nest
 * the loops), so the shader applies the whole chain, however deep, as a single
 * quaternion+offset indexed by this id. The "chain" lives in the composition, done once
 * per PIVOT on the CPU, never once per node in the shader.
 *
 * The id written is the node's leaf pivot CLAMPED to the active chain depth
 * `min(chainDepth, cap)`: a pivot at level ≥ depth contributes no independent spin (its
 * own rate may not even exist in the shader's budget), and because a never-spinning
 * child's composed quaternion equals its parent's, walking up to the deepest ancestor
 * with `level < depth` yields the identical transform with fewer distinct indices.
 * `chainDepth` 0 (no level declares spin) packs every node as −1: the world frame.
 *
 * Returns an Int32Array of pivot ids (−1 = world/identity). The GPU attribute form is
 * `id + 1` as float (0 = identity texel) — the stores own that shift; CPU code keeps
 * honest ids so `frames.transformPoint(ids[i], …)` is directly the reference transform.
 */
export function packChainIds(
  nodePivot: Uint32Array,
  parent: Int32Array,
  level: Uint8Array,
  chainDepth: number,
  cap: number = CHAIN_CAP,
): Int32Array {
  const n = nodePivot.length;
  const out = new Int32Array(n);
  const depth = Math.min(chainDepth, cap);
  if (depth <= 0) {
    out.fill(-1);
    return out;
  }
  for (let i = 0; i < n; i++) {
    let p: number = nodePivot[i]!;
    while (p >= 0 && level[p]! >= depth) p = parent[p]!;
    out[i] = p;
  }
  return out;
}

/**
 * The retarget bake — the standard morph-retarget trick (§4, deliverable: "A becomes the
 * current blend snapshot"). Writes `mix(a, b, t)` INTO `a`, with exact endpoints:
 *
 *   - t ≤ 0 → `a` untouched (bit-exact: the morph never started);
 *   - t ≥ 1 → `a` becomes a bit-exact copy of `b` (never `a + (b−a)·1`, which float
 *     arithmetic does NOT guarantee lands on `b` — endpoint exactness is what lets a
 *     round trip restore byte-identical coordinates, §4);
 *   - otherwise the plain lerp, rounded to f32 by the store itself.
 *
 * Continuity by construction: immediately after the bake, `mix(a′, anything, 0) = a′`
 * equals the pre-bake `mix(a, b, t)` — the field never snaps on retarget.
 */
export function bakeMix(a: Float32Array, b: Float32Array, t: number): void {
  if (a.length !== b.length) {
    throw new Error(`bakeMix: length mismatch (${a.length} vs ${b.length})`);
  }
  if (t <= 0) return;
  if (t >= 1) {
    a.set(b);
    return;
  }
  for (let i = 0; i < a.length; i++) a[i] = a[i]! + (b[i]! - a[i]!) * t;
}

/** The field centroid of a 3n position buffer — the centre the spread scale is about
 *  (§4). Float64 accumulation (10⁶ f32 sums drift), result returned as the f32-rounded
 *  values the GPU uniform will hold, so CPU reference and shader agree on the centre. */
export function centroidOf(positions: Float32Array, n: number): [number, number, number] {
  if (n <= 0) return [0, 0, 0];
  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (let i = 0; i < n; i++) {
    sx += positions[i * 3]!;
    sy += positions[i * 3 + 1]!;
    sz += positions[i * 3 + 2]!;
  }
  return [Math.fround(sx / n), Math.fround(sy / n), Math.fround(sz / n)];
}

/**
 * The 2D projection target (§3.6: dimensionality is a destination, not a branch): the
 * live arrangement with one axis dropped onto the plane through `value` (typically the
 * centroid's coordinate on that axis). Writes a NEW buffer unless `out` is given; the
 * input is never touched — the 3D target stays pristine, which is what makes the
 * 2d → 3d round trip land byte-exact (§4: stages compose, nothing mutates in place).
 */
export function flattenAxis(
  positions: Float32Array,
  n: number,
  axis: 0 | 1 | 2,
  value: number,
  out?: Float32Array,
): Float32Array {
  const dst = out ?? new Float32Array(n * 3);
  if (dst !== positions) dst.set(positions.subarray(0, n * 3));
  const v = Math.fround(value);
  for (let i = 0; i < n; i++) dst[i * 3 + axis] = v;
  return dst;
}

/**
 * The CPU reference for the spread stage: a uniform scale about `centroid`. At s = 1 it
 * is an EXACT identity (copied through, not `c + (p−c)·1` — float arithmetic does not
 * round-trip that), which is the §4 rule "dragging spread back to 1.0 restores byte-exact
 * coordinates" made literal. The tween lands exactly on its target (src/anim/tween.ts),
 * so the rule holds end to end.
 */
export function spreadPoint(
  p: ArrayLike<number>,
  centroid: readonly [number, number, number],
  s: number,
  out: Float32Array | number[],
  pOffset = 0,
  outOffset = 0,
): void {
  const x = p[pOffset]!;
  const y = p[pOffset + 1]!;
  const z = p[pOffset + 2]!;
  if (s === 1) {
    out[outOffset] = x;
    out[outOffset + 1] = y;
    out[outOffset + 2] = z;
    return;
  }
  out[outOffset] = centroid[0] + (x - centroid[0]) * s;
  out[outOffset + 1] = centroid[1] + (y - centroid[1]) * s;
  out[outOffset + 2] = centroid[2] + (z - centroid[2]) * s;
}

/**
 * The CPU reference for the FIELD-GLOBAL root-frame rotation (DESIGN §3.5 top level): rotate
 * `p` about `centroid` by quaternion `q` (xyzw) — the shader's last stage (stores.ts worldOf),
 * applied after spread about the SAME centre. At the identity quaternion (0,0,0,1) it is an
 * EXACT pass-through (t = 0 below, so no arithmetic moves the point), which keeps an unspun
 * field byte-stable through this stage — the same guarantee spreadPoint makes at s = 1. The
 * kernel is `v' = v + w·t + q×t` with `t = 2 q×v`, the frames.ts rotation, f64 locals.
 */
export function fieldSpinPoint(
  p: ArrayLike<number>,
  centroid: readonly [number, number, number],
  q: readonly [number, number, number, number],
  out: Float32Array | number[],
  pOffset = 0,
  outOffset = 0,
): void {
  const x = p[pOffset]! - centroid[0];
  const y = p[pOffset + 1]! - centroid[1];
  const z = p[pOffset + 2]! - centroid[2];
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  out[outOffset] = centroid[0] + x + qw * tx + (qy * tz - qz * ty);
  out[outOffset + 1] = centroid[1] + y + qw * ty + (qz * tx - qx * tz);
  out[outOffset + 2] = centroid[2] + z + qw * tz + (qx * ty - qy * tx);
}

/**
 * Fold a FrameSet's per-pivot (quat, centre) + compile-time world0 into the two arrays
 * the shader actually indexes — the one-apply form of frames.transformPoint:
 *
 *     transformPoint(p) = centre + q ⊗ (p − world0)  =  q ⊗ p + (centre − q ⊗ world0)
 *                                                       \________ offset ________/
 *
 * so the vertex shader does ONE rotate and ONE add per node, whatever the chain depth.
 * O(pivots), run once per tick while anything turns (§4) — this IS the per-frame CPU.
 *
 * Writes `count + 1` entries: slot 0 is the identity (quat 0,0,0,1; offset 0) — the
 * texel every chain-less node (id −1 → attribute 0) reads, branchlessly. Slot p+1 is
 * pivot p. `outQuat` is 4(count+1) long, `outOffset` 3(count+1) (callers may hand wider
 * slabs; only these prefixes are written).
 */
export function packPivotUniforms(
  frames: FrameSet,
  world0: Float32Array,
  outQuat: Float32Array,
  outOffset: Float32Array,
): void {
  const P = frames.count;
  const quat = frames.quat;
  const centre = frames.centre;
  outQuat[0] = 0;
  outQuat[1] = 0;
  outQuat[2] = 0;
  outQuat[3] = 1;
  outOffset[0] = 0;
  outOffset[1] = 0;
  outOffset[2] = 0;
  for (let p = 0; p < P; p++) {
    const a4 = p * 4;
    const a3 = p * 3;
    const qx = quat[a4]!;
    const qy = quat[a4 + 1]!;
    const qz = quat[a4 + 2]!;
    const qw = quat[a4 + 3]!;
    const o4 = (p + 1) * 4;
    const o3 = (p + 1) * 3;
    outQuat[o4] = qx;
    outQuat[o4 + 1] = qy;
    outQuat[o4 + 2] = qz;
    outQuat[o4 + 3] = qw;
    // q ⊗ world0: t = 2(q×v); v' = v + w·t + q×t  (the frames.ts kernel, f64 locals)
    const x = world0[a3]!;
    const y = world0[a3 + 1]!;
    const z = world0[a3 + 2]!;
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    outOffset[o3] = centre[a3]! - (x + qw * tx + (qy * tz - qz * ty));
    outOffset[o3 + 1] = centre[a3 + 1]! - (y + qw * ty + (qz * tx - qx * tz));
    outOffset[o3 + 2] = centre[a3 + 2]! - (z + qw * tz + (qx * ty - qy * tx));
  }
}
