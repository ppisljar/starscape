/* SPATIAL facet colouring (DESIGN.md §5 theming, §7 ramps, §10 house rules) — a colour-assignment
 * mode where each facet-value CLUSTER is coloured by its SPATIAL POSITION through a continuous
 * ramp, so spatially-adjacent clusters share hue (a smooth gradient) instead of the scattered
 * colours a categorical palette gives a high-cardinality facet.
 *
 * Where field/palette.ts paintFacetColors maps a value's INDEX straight to a palette slot (two
 * adjacent clusters whose value indices happen to be far apart get wildly different colours), this
 * reads GEOMETRY: it computes, per facet value, the CENTROID of that value's members from the
 * current positions, orders the values along a 3D space-filling curve over those centroids, and
 * maps each value to a ramp colour by its RANK in that order. Neighbouring clusters get neighbouring
 * ranks, hence neighbouring ramp colours.
 *
 * SPACE-FILLING CURVE — a 3D HILBERT curve (Skilling's AxesToTranspose, 2004). Chosen over Morton /
 * Z-order deliberately: the goal is a SMOOTH gradient, and Z-order has visible seams — crossing a
 * high-order bit boundary jumps the curve across the whole volume, so two spatially-adjacent
 * clusters can land far apart in rank and show a hard hue discontinuity. The Hilbert curve has no
 * such long jumps (consecutive indices are always spatially adjacent), so adjacency in space maps to
 * adjacency in rank far more faithfully — which is exactly "nearby clusters share hue".
 *
 * BOUNDED and DETERMINISTIC (§10). One O(n) pass accumulates per-value centroid sums; the per-value
 * work (bounding box, Hilbert key, rank sort, ramp sample) is O(V + V·log V) in the VALUE count, not
 * the node count; a final O(n) pass paints. No Math.random, no Date.now: identical positions →
 * identical colours, forever (ties in the Hilbert key break on value index, so the order is total
 * and stable). It is POSITION-DEPENDENT, so the field recomputes it on every arrangement change
 * (field.ts finishRetarget), exactly as the local-density channel recomputes.
 */

import { rampAt } from '../theme/palettes';
import type { RampLike } from '../theme/palettes';

/** Bits per axis for the Hilbert quantisation (per-axis grid resolution = 2^bits). 16 bits gives a
 *  65 536³ lattice — far finer than any realistic cluster count, so distinct centroids almost never
 *  collide into one cell, while the 48-bit Hilbert index (3·16) stays an exact double (< 2^53). */
export const SPATIAL_HILBERT_BITS = 16;

/** 3D Hilbert distance for integer lattice coordinates (each in [0, 2^bits)), via Skilling's
 *  AxesToTranspose transform followed by a bit-interleave of the transposed axes into one scalar.
 *  Pure; the ONLY use of the result is to ORDER values, so the absolute magnitude is irrelevant —
 *  only that it is a valid Hilbert ordering (consecutive indices spatially adjacent). */
export function hilbertIndex3(x: number, y: number, z: number, bits: number): number {
  // the transposed coordinate triple (a, b, c), mutated by Skilling's AxesToTranspose. Plain
  // locals rather than an array — the per-dimension inner loop is unrolled so dim 0's self-exchange
  // (a no-op) is simply dropped.
  let a = x | 0;
  let b = y | 0;
  let c = z | 0;
  const M = 1 << (bits - 1);
  // inverse undo of the excess work: invert (coord & Q) or exchange the low bits into a
  for (let Q = M; Q > 1; Q >>= 1) {
    const P = Q - 1;
    if (a & Q) a ^= P; // dim 0: invert (its exchange branch is a no-op)
    if (b & Q) {
      a ^= P; // dim 1
    } else {
      const t = (a ^ b) & P;
      a ^= t;
      b ^= t;
    }
    if (c & Q) {
      a ^= P; // dim 2
    } else {
      const t = (a ^ c) & P;
      a ^= t;
      c ^= t;
    }
  }
  // Gray encode across dimensions, then the final fix-up pass
  b ^= a;
  c ^= b;
  let t = 0;
  for (let Q = M; Q > 1; Q >>= 1) {
    if (c & Q) t ^= Q - 1;
  }
  a ^= t;
  b ^= t;
  c ^= t;
  // interleave the transposed triple into one index, most-significant bit first
  let d = 0;
  for (let bit = bits - 1; bit >= 0; bit--) {
    d = d * 2 + ((a >> bit) & 1);
    d = d * 2 + ((b >> bit) & 1);
    d = d * 2 + ((c >> bit) & 1);
  }
  return d;
}

/**
 * Paint per-node colours by SPATIAL facet ordering: each facet value's members share one ramp
 * colour, and the values are ordered along a 3D Hilbert curve over their centroids so adjacent
 * clusters get adjacent ramp colours (a smooth spatial gradient). Writes into `out` when given
 * (3n), else allocates — the field reuses one buffer across recomputes (§10.2; the per-value
 * scratch is O(values), allocated per call because it is tiny).
 *
 * @param codes      per-node facet value index (the chosen colour facet's data).
 * @param valueCount the facet's value-dictionary size (codes ∈ [0, valueCount)).
 * @param positions  3n floats, the geometry to read centroids from (the field feeds its live target).
 * @param n          node count.
 * @param ramp       the continuous ramp the clusters are coloured through (the user's chosen ramp).
 */
export function spatialFacetColors(
  codes: ArrayLike<number>,
  valueCount: number,
  positions: Float32Array,
  n: number,
  ramp: RampLike,
  out?: Float32Array,
): Float32Array {
  const colors = out !== undefined && out.length >= n * 3 ? out : new Float32Array(n * 3);
  if (n === 0 || valueCount === 0) return colors;

  /* ---- pass 1 (O(n)): per-value centroid sums + member counts. */
  const cx = new Float64Array(valueCount);
  const cy = new Float64Array(valueCount);
  const cz = new Float64Array(valueCount);
  const count = new Uint32Array(valueCount);
  for (let i = 0; i < n; i++) {
    const v = codes[i]!;
    cx[v] = cx[v]! + positions[i * 3]!;
    cy[v] = cy[v]! + positions[i * 3 + 1]!;
    cz[v] = cz[v]! + positions[i * 3 + 2]!;
    count[v] = count[v]! + 1;
  }

  /* ---- the PRESENT values (count > 0) and their centroids; plus the centroid bounding box.
     Values with no members never colour a node, so they are left out of the ordering entirely. */
  const present: number[] = [];
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (let v = 0; v < valueCount; v++) {
    const c = count[v]!;
    if (c === 0) continue;
    const x = cx[v]! / c;
    const y = cy[v]! / c;
    const z = cz[v]! / c;
    cx[v] = x; // overwrite the sums with the centroid (reused below)
    cy[v] = y;
    cz[v] = z;
    present.push(v);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const P = present.length;
  const vcol = new Float32Array(valueCount * 3); // per-value colour table
  if (P === 1) {
    // one cluster: a single point on the ramp (start), nothing to order.
    const only = present[0]!;
    const col = rampAt(ramp, 0);
    vcol[only * 3] = col[0];
    vcol[only * 3 + 1] = col[1];
    vcol[only * 3 + 2] = col[2];
  } else {
    /* Normalise each centroid into the CLUSTER-CENTROID bounding box (the tightest box that still
       contains every cluster — using it, rather than the looser full-node box, spreads the clusters
       across the whole Hilbert range for maximum ordering resolution), quantise to the lattice, and
       take the Hilbert index. A degenerate axis (all centroids share a coordinate) spans 0 — guard
       it so the quantiser maps every cluster to 0 on that axis rather than dividing by zero. */
    const bits = SPATIAL_HILBERT_BITS;
    const scale = (1 << bits) - 1;
    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const spanZ = maxZ - minZ;
    const invX = spanX > 0 ? scale / spanX : 0;
    const invY = spanY > 0 ? scale / spanY : 0;
    const invZ = spanZ > 0 ? scale / spanZ : 0;
    const key = new Float64Array(P);
    for (let p = 0; p < P; p++) {
      const v = present[p]!;
      const qx = Math.round((cx[v]! - minX) * invX);
      const qy = Math.round((cy[v]! - minY) * invY);
      const qz = Math.round((cz[v]! - minZ) * invZ);
      key[p] = hilbertIndex3(qx, qy, qz, bits);
    }
    /* Rank the present values by (Hilbert key, value index) — the value-index tiebreak makes the
       order total and deterministic even when two clusters quantise to one cell (§10). */
    const order = new Uint32Array(P);
    for (let p = 0; p < P; p++) order[p] = p;
    order.sort((i, j) => {
      const d = key[i]! - key[j]!;
      if (d !== 0) return d < 0 ? -1 : 1;
      return present[i]! - present[j]!;
    });
    /* Each value's ramp colour is its RANK percentile through the ramp — the full ramp is swept
       across the clusters in spatial order, so neighbours in space are neighbours on the ramp. */
    const denom = P - 1;
    for (let r = 0; r < P; r++) {
      const v = present[order[r]!]!;
      const col = rampAt(ramp, r / denom);
      vcol[v * 3] = col[0];
      vcol[v * 3 + 1] = col[1];
      vcol[v * 3 + 2] = col[2];
    }
  }

  /* ---- pass 2 (O(n)): paint each node its value's colour. */
  for (let i = 0; i < n; i++) {
    const v = codes[i]! * 3;
    colors[i * 3] = vcol[v]!;
    colors[i * 3 + 1] = vcol[v + 1]!;
    colors[i * 3 + 2] = vcol[v + 2]!;
  }
  return colors;
}
