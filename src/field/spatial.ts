/* SPATIAL facet colouring (DESIGN.md §5 theming, §7 ramps, §10 house rules) — a colour-assignment
 * mode where each facet-value CLUSTER is coloured by its SPATIAL POSITION through a continuous
 * ramp, so spatially-adjacent clusters share hue (a smooth gradient) instead of the scattered
 * colours a categorical palette gives a high-cardinality facet.
 *
 * Where field/palette.ts paintFacetColors maps a value's INDEX straight to a palette slot (two
 * adjacent clusters whose value indices happen to be far apart get wildly different colours), this
 * reads GEOMETRY: it computes, per facet value, the CENTROID of that value's members from the
 * current positions, PROJECTS each centroid onto the dominant spatial axis, and maps that scalar
 * (min..max → 0..1) to a ramp colour. Neighbouring clusters get neighbouring projections, hence
 * neighbouring ramp colours.
 *
 * PRINCIPAL-AXIS PROJECTION — not a space-filling curve. The predecessor ordered centroids along a
 * 3D Hilbert curve and coloured by RANK. That reduced small jumps but could not remove the curve's
 * TOP-LEVEL FOLD: any space-filling curve folds back on itself, so two clusters adjacent in 3D can
 * sit at opposite ends of the 1D order and, at the fold, opposite ends of the ramp — a visible
 * HARD COLOUR SEAM across the field (confirmed on synth 1M, colour=system, order=spatial). A fold
 * is intrinsic to mapping 3D→1D by a curve; the only seam-free map is a MONOTONIC one. Projecting
 * onto a single axis is exactly that: t = (centroid·û − min)/(max − min) is monotone in position
 * along û, so continuity in 3D becomes continuity in colour with no discontinuity anywhere — a
 * clean directional gradient that sweeps the WHOLE ramp from one end of the field to the other.
 * The axis û is the first principal component of the cluster centroids (the direction of greatest
 * spread), so the gradient runs along the field's longest dimension and uses the ramp's full range;
 * a degenerate/isotropic spread falls back deterministically to the dominant coordinate axis.
 *
 * BOUNDED and DETERMINISTIC (§10). One O(n) pass accumulates per-value centroid sums; the per-value
 * work (centroid mean, 3×3 covariance, closed-form eigensolve, projection) is O(V) in the VALUE
 * count, not the node count; a final O(n) pass paints. No Math.random, no Date.now: identical
 * positions → identical colours, forever (the principal axis is a pure function of the covariance,
 * with a canonical sign so v and −v never trade places between runs). It is POSITION-DEPENDENT, so
 * the field recomputes it on every arrangement change (field.ts finishRetarget), exactly as the
 * local-density channel recomputes.
 */

import { rampAt } from '../theme/palettes';
import type { RampLike } from '../theme/palettes';

/** The dominant axis (first principal component) of a symmetric 3×3 covariance matrix, as a unit
 *  vector. Returns the eigenvector of the LARGEST eigenvalue via the closed-form symmetric-3×3
 *  eigensolve (Smith 1961 / the standard trigonometric method for the eigenvalues, null-space cross
 *  products for the eigenvector) — pure and deterministic, no iteration. The matrix is given by its
 *  six distinct entries: diagonal (xx, yy, zz) and off-diagonal (xy, xz, yz). Absolute scale is
 *  irrelevant (only the direction is used), so unnormalised covariance SUMS may be passed. The sign
 *  is canonicalised (largest-magnitude component made positive) so the result is byte-stable. */
export function principalAxis3(
  xx: number,
  yy: number,
  zz: number,
  xy: number,
  xz: number,
  yz: number,
): [number, number, number] {
  // sign-canonicalise a direction so (v, −v) never trade places between runs (§10 determinism):
  // flip so the largest-magnitude component is positive; ties break toward the earlier axis.
  const canon = (x: number, y: number, z: number): [number, number, number] => {
    const ax = Math.abs(x),
      ay = Math.abs(y),
      az = Math.abs(z);
    const lead = ax >= ay && ax >= az ? x : ay >= az ? y : z;
    const s = lead < 0 ? -1 : 1;
    return [x * s, y * s, z * s];
  };
  const unit = (x: number, y: number, z: number): [number, number, number] => {
    const m = Math.hypot(x, y, z);
    return m > 0 ? [x / m, y / m, z / m] : [1, 0, 0];
  };

  const p1 = xy * xy + xz * xz + yz * yz;
  if (p1 === 0) {
    // already diagonal: the dominant axis is the coordinate axis of the largest diagonal entry.
    if (xx >= yy && xx >= zz) return [1, 0, 0];
    return yy >= zz ? [0, 1, 0] : [0, 0, 1];
  }

  // closed-form largest eigenvalue of the symmetric matrix (the trigonometric method).
  const q = (xx + yy + zz) / 3;
  const p2 = (xx - q) * (xx - q) + (yy - q) * (yy - q) + (zz - q) * (zz - q) + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  // B = (A − qI)/p; r = det(B)/2 ∈ [−1, 1] (clamped against round-off).
  const b11 = (xx - q) / p,
    b22 = (yy - q) / p,
    b33 = (zz - q) / p;
  const b12 = xy / p,
    b13 = xz / p,
    b23 = yz / p;
  const detB =
    b11 * (b22 * b33 - b23 * b23) -
    b12 * (b12 * b33 - b23 * b13) +
    b13 * (b12 * b23 - b22 * b13);
  const r = Math.max(-1, Math.min(1, detB / 2));
  const phi = Math.acos(r) / 3;
  const eig1 = q + 2 * p * Math.cos(phi); // the LARGEST eigenvalue

  // eigenvector of eig1: a non-zero vector in the null space of (A − eig1·I). The three rows of
  // that matrix span the plane orthogonal to the eigenvector, so the eigenvector is the cross
  // product of any two independent rows — take the pair with the largest cross magnitude for
  // numerical stability (near a repeated eigenvalue one pair can be degenerate).
  const r0x = xx - eig1,
    r0y = xy,
    r0z = xz;
  const r1x = xy,
    r1y = yy - eig1,
    r1z = yz;
  const r2x = xz,
    r2y = yz,
    r2z = zz - eig1;
  const cross = (
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
  ): [number, number, number] => [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
  const c01 = cross(r0x, r0y, r0z, r1x, r1y, r1z);
  const c02 = cross(r0x, r0y, r0z, r2x, r2y, r2z);
  const c12 = cross(r1x, r1y, r1z, r2x, r2y, r2z);
  const n01 = c01[0] * c01[0] + c01[1] * c01[1] + c01[2] * c01[2];
  const n02 = c02[0] * c02[0] + c02[1] * c02[1] + c02[2] * c02[2];
  const n12 = c12[0] * c12[0] + c12[1] * c12[1] + c12[2] * c12[2];
  let best = c01,
    bestN = n01;
  if (n02 > bestN) {
    best = c02;
    bestN = n02;
  }
  if (n12 > bestN) {
    best = c12;
    bestN = n12;
  }
  if (bestN === 0) {
    // isotropic / repeated eigenvalue: no unique principal axis. Fall back deterministically to
    // the largest-diagonal coordinate axis, matching the p1 === 0 branch's spirit.
    if (xx >= yy && xx >= zz) return [1, 0, 0];
    return yy >= zz ? [0, 1, 0] : [0, 0, 1];
  }
  const [ux, uy, uz] = unit(best[0], best[1], best[2]);
  return canon(ux, uy, uz);
}

/**
 * Paint per-node colours by SPATIAL facet ordering: each facet value's members share one ramp
 * colour, and the values are placed on the ramp by PROJECTING their centroids onto the clusters'
 * dominant spatial axis (first principal component), so adjacent clusters get adjacent ramp colours
 * — a smooth, seam-free directional gradient. Writes into `out` when given (3n), else allocates —
 * the field reuses one buffer across recomputes (§10.2; the per-value scratch is O(values),
 * allocated per call because it is tiny).
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

  /* ---- the PRESENT values (count > 0) and their centroids. Values with no members never colour a
     node, so they are left out of the projection entirely. */
  const present: number[] = [];
  for (let v = 0; v < valueCount; v++) {
    const c = count[v]!;
    if (c === 0) continue;
    cx[v] = cx[v]! / c; // overwrite the sums with the centroid (reused below)
    cy[v] = cy[v]! / c;
    cz[v] = cz[v]! / c;
    present.push(v);
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
    /* The centroid MEAN and the 3×3 covariance of the present centroids (each cluster weighted
       equally — one point apiece, so a huge cluster and a tiny one count the same for ORDERING). */
    let mx = 0,
      my = 0,
      mz = 0;
    for (let i = 0; i < P; i++) {
      const v = present[i]!;
      mx += cx[v]!;
      my += cy[v]!;
      mz += cz[v]!;
    }
    mx /= P;
    my /= P;
    mz /= P;
    let sxx = 0,
      syy = 0,
      szz = 0,
      sxy = 0,
      sxz = 0,
      syz = 0;
    for (let i = 0; i < P; i++) {
      const v = present[i]!;
      const dx = cx[v]! - mx;
      const dy = cy[v]! - my;
      const dz = cz[v]! - mz;
      sxx += dx * dx;
      syy += dy * dy;
      szz += dz * dz;
      sxy += dx * dy;
      sxz += dx * dz;
      syz += dy * dz;
    }

    /* The dominant axis û (first principal component): the direction the clusters spread along.
       Projecting onto it is a MONOTONIC 3D→1D map — continuous in space, hence continuous in
       colour, with no fold and therefore no seam (unlike a space-filling curve). */
    const [ux, uy, uz] = principalAxis3(sxx, syy, szz, sxy, sxz, syz);

    /* Project each centroid onto û and normalise min..max → 0..1 so the gradient sweeps the WHOLE
       ramp across the field's longest dimension. */
    const proj = new Float64Array(P);
    let minP = Infinity,
      maxP = -Infinity;
    for (let i = 0; i < P; i++) {
      const v = present[i]!;
      const t = (cx[v]! - mx) * ux + (cy[v]! - my) * uy + (cz[v]! - mz) * uz;
      proj[i] = t;
      if (t < minP) minP = t;
      if (t > maxP) maxP = t;
    }
    const span = maxP - minP;
    const inv = span > 0 ? 1 / span : 0; // degenerate (all projections equal) → everyone at t = 0
    for (let i = 0; i < P; i++) {
      const v = present[i]!;
      const col = rampAt(ramp, (proj[i]! - minP) * inv);
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
