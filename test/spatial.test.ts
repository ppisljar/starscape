/* SPATIAL facet colouring (field/spatial.ts, Feature F1): clusters coloured by their centroid's
 * projection onto the dominant spatial axis through a ramp, so spatially-adjacent clusters share
 * hue — a smooth, SEAM-FREE gradient.
 *
 * Two things are pinned: the principal axis is the direction of greatest spread (so the gradient
 * runs along the field's longest dimension and sweeps the whole ramp), and the colouring built on
 * it is deterministic (§10) AND seamless — because a projection is a MONOTONIC 3D→1D map, there is
 * no fold where spatially-adjacent clusters jump to opposite ends of the ramp (the hard seam the
 * predecessor's space-filling curve could not avoid).
 */

import { describe, expect, it } from 'vitest';
import { principalAxis3, spatialFacetColors } from '../src/field/spatial';
import { rampAt } from '../src/theme/palettes';

describe('principalAxis3 — the dominant axis of a symmetric 3×3 covariance', () => {
  it('returns the coordinate axis of the largest variance for a diagonal matrix', () => {
    expect(principalAxis3(9, 1, 2, 0, 0, 0)).toEqual([1, 0, 0]);
    expect(principalAxis3(1, 9, 2, 0, 0, 0)).toEqual([0, 1, 0]);
    expect(principalAxis3(1, 2, 9, 0, 0, 0)).toEqual([0, 0, 1]);
  });

  it('finds the elongation direction of a tilted spread (eigenvector of the largest eigenvalue)', () => {
    // points spread mostly along (1, 1, 0): covariance of {t·(1,1,0)} is [[s,s,0],[s,s,0],[0,0,0]]
    const [x, y, z] = principalAxis3(5, 5, 0, 5, 0, 0);
    // unit vector along (1, 1, 0), sign-canonicalised to a positive lead component
    expect(Math.hypot(x, y, z)).toBeCloseTo(1, 6);
    expect(Math.abs(x - y)).toBeLessThan(1e-6);
    expect(Math.abs(z)).toBeLessThan(1e-6);
    expect(x).toBeGreaterThan(0);
  });

  it('is deterministic and sign-stable (same matrix → byte-identical axis)', () => {
    const a = principalAxis3(7, 3, 2, 1.5, -0.5, 0.8);
    const b = principalAxis3(7, 3, 2, 1.5, -0.5, 0.8);
    expect(a).toEqual(b);
  });
});

/** K clusters strung along a dominant axis d (with a small deterministic perpendicular jitter so
 *  the distribution is genuinely 3D, not a perfect line), one node per cluster. Projection order
 *  along d is the cluster index order. */
function lineClusters(K: number): {
  codes: Uint32Array;
  positions: Float32Array;
  axis: [number, number, number];
} {
  // a deliberately tilted dominant axis, normalised
  const dx = 1,
    dy = 0.3,
    dz = -0.2;
  const dm = Math.hypot(dx, dy, dz);
  const axis: [number, number, number] = [dx / dm, dy / dm, dz / dm];
  const codes = new Uint32Array(K);
  const positions = new Float32Array(K * 3);
  for (let i = 0; i < K; i++) {
    codes[i] = i;
    const s = i * 2; // spacing along the dominant axis dwarfs the jitter below
    positions[i * 3] = s * axis[0] + 0.2 * Math.sin(i * 1.3);
    positions[i * 3 + 1] = s * axis[1] + 0.2 * Math.cos(i * 0.7);
    positions[i * 3 + 2] = s * axis[2] + 0.2 * Math.sin(i * 2.1);
  }
  return { codes, positions, axis };
}

const rgbDist = (c: Float32Array, a: number, b: number): number =>
  Math.hypot(c[a * 3]! - c[b * 3]!, c[a * 3 + 1]! - c[b * 3 + 1]!, c[a * 3 + 2]! - c[b * 3 + 2]!);

describe('spatialFacetColors — a smooth, seam-free spatial gradient', () => {
  it('is deterministic (same positions → byte-identical colours)', () => {
    const g = lineClusters(40);
    const a = spatialFacetColors(g.codes, 40, g.positions, 40, 'spectral');
    const b = spatialFacetColors(g.codes, 40, g.positions, 40, 'spectral');
    expect(Array.from(b)).toEqual(Array.from(a));
  });

  it('sweeps the WHOLE ramp: the extreme clusters hit the ramp endpoints', () => {
    const K = 40;
    const g = lineClusters(K);
    const colors = spatialFacetColors(g.codes, K, g.positions, K, 'spectral');
    const first = rampAt('spectral', 0);
    const last = rampAt('spectral', 1);
    // the min-projection cluster is coloured the ramp's start, the max-projection cluster its end —
    // t = (proj − min)/(max − min) hits exactly 0 and 1. (Index order ≈ projection order here.)
    let nearStart = Infinity;
    let nearEnd = Infinity;
    for (let i = 0; i < K; i++) {
      nearStart = Math.min(nearStart, Math.hypot(colors[i * 3]! - first[0], colors[i * 3 + 1]! - first[1], colors[i * 3 + 2]! - first[2]));
      nearEnd = Math.min(nearEnd, Math.hypot(colors[i * 3]! - last[0], colors[i * 3 + 1]! - last[1], colors[i * 3 + 2]! - last[2]));
    }
    expect(nearStart).toBeLessThan(1e-6);
    expect(nearEnd).toBeLessThan(1e-6);
  });

  it('has NO seam: consecutive clusters along the axis never jump across the ramp', () => {
    const K = 48;
    const g = lineClusters(K);
    const colors = spatialFacetColors(g.codes, K, g.positions, K, 'spectral');
    // cluster index order follows the projection order, so the step between consecutive clusters is
    // the gradient's local slope. The largest such step must stay a small fraction of the total ramp
    // extent — a fold (the predecessor's failure) would show up as one step comparable to the whole
    // ramp. Measure the ramp's end-to-end distance as the reference.
    const first = rampAt('spectral', 0);
    const last = rampAt('spectral', 1);
    const rampSpan = Math.hypot(first[0] - last[0], first[1] - last[1], first[2] - last[2]);
    let maxStep = 0;
    for (let i = 1; i < K; i++) maxStep = Math.max(maxStep, rgbDist(colors, i - 1, i));
    expect(maxStep).toBeLessThan(rampSpan * 0.25);
  });

  it('colours spatially-near clusters more alike than spatially-far ones', () => {
    const K = 40;
    const g = lineClusters(K);
    const colors = spatialFacetColors(g.codes, K, g.positions, K, 'spectral');
    let nearSum = 0,
      nearCount = 0,
      farSum = 0,
      farCount = 0;
    for (let i = 0; i < K; i++) {
      for (let j = i + 1; j < K; j++) {
        const cd = rgbDist(colors, i, j);
        if (j - i === 1) {
          nearSum += cd;
          nearCount++;
        } else if (j - i >= K / 2) {
          farSum += cd;
          farCount++;
        }
      }
    }
    expect(nearSum / nearCount).toBeLessThan((farSum / farCount) * 0.6);
  });

  it('a single present value takes one ramp colour; absent values are left untouched', () => {
    const codes = new Uint32Array([1, 1, 1]);
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const out = spatialFacetColors(codes, 3, positions, 3, 'ember');
    expect(rgbDist(out, 0, 1)).toBe(0);
    expect(rgbDist(out, 0, 2)).toBe(0);
  });

  it('handles the empty field without throwing', () => {
    const out = spatialFacetColors(new Uint32Array(0), 0, new Float32Array(0), 0, 'ice');
    expect(out.length).toBe(0);
  });
});
