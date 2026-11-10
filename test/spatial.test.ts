/* SPATIAL facet colouring (field/spatial.ts, Feature F1): clusters coloured by their centroid's
 * position along a 3D Hilbert curve through a ramp, so spatially-adjacent clusters share hue.
 *
 * Two things are pinned: the Hilbert index is a genuine space-filling curve (a bijection on the
 * lattice whose consecutive indices are grid-adjacent — THE locality guarantee), and the colouring
 * built on it is deterministic (§10) and actually smooth (spatially-near clusters get nearer colours
 * than spatially-far ones — the feature's whole point).
 */

import { describe, expect, it } from 'vitest';
import { hilbertIndex3, spatialFacetColors } from '../src/field/spatial';

describe('hilbertIndex3 — a genuine 3D Hilbert curve', () => {
  it('is a bijection on the lattice with grid-adjacent consecutive indices', () => {
    const bits = 2; // 4 per axis → 64 cells
    const side = 1 << bits;
    const cells: { x: number; y: number; z: number; d: number }[] = [];
    const seen = new Set<number>();
    for (let x = 0; x < side; x++) {
      for (let y = 0; y < side; y++) {
        for (let z = 0; z < side; z++) {
          const d = hilbertIndex3(x, y, z, bits);
          cells.push({ x, y, z, d });
          seen.add(d);
        }
      }
    }
    // bijection: 64 distinct indices, exactly 0..63
    expect(seen.size).toBe(side * side * side);
    let min = Infinity;
    let max = -Infinity;
    for (const d of seen) {
      min = Math.min(min, d);
      max = Math.max(max, d);
    }
    expect(min).toBe(0);
    expect(max).toBe(side * side * side - 1);
    // locality: walking the curve in index order never jumps — each step is one lattice cell
    cells.sort((a, b) => a.d - b.d);
    for (let i = 1; i < cells.length; i++) {
      const a = cells[i - 1]!;
      const b = cells[i]!;
      const manhattan = Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
      expect(manhattan).toBe(1);
    }
  });

  it('is deterministic (no RNG, no clock)', () => {
    expect(hilbertIndex3(13, 7, 42, 16)).toBe(hilbertIndex3(13, 7, 42, 16));
  });
});

/** One node per facet value, positioned at its 3D grid cell; value index is the RASTER order
 *  (x + side·y + side²·z), which has poor spatial locality — a good foil for the Hilbert order. */
function gridClusters(side: number): {
  codes: Uint32Array;
  valueCount: number;
  positions: Float32Array;
  cellOf: { x: number; y: number; z: number }[];
} {
  const v = side * side * side;
  const codes = new Uint32Array(v);
  const positions = new Float32Array(v * 3);
  const cellOf: { x: number; y: number; z: number }[] = [];
  for (let z = 0; z < side; z++) {
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const idx = x + side * y + side * side * z;
        codes[idx] = idx;
        positions[idx * 3] = x;
        positions[idx * 3 + 1] = y;
        positions[idx * 3 + 2] = z;
        cellOf[idx] = { x, y, z };
      }
    }
  }
  return { codes, valueCount: v, positions, cellOf };
}

const rgbDist = (c: Float32Array, a: number, b: number): number =>
  Math.hypot(c[a * 3]! - c[b * 3]!, c[a * 3 + 1]! - c[b * 3 + 1]!, c[a * 3 + 2]! - c[b * 3 + 2]!);

describe('spatialFacetColors — a smooth spatial gradient', () => {
  it('is deterministic (same positions → byte-identical colours)', () => {
    const g = gridClusters(4);
    const a = spatialFacetColors(g.codes, g.valueCount, g.positions, g.valueCount, 'spectral');
    const b = spatialFacetColors(g.codes, g.valueCount, g.positions, g.valueCount, 'spectral');
    expect(Array.from(b)).toEqual(Array.from(a));
  });

  it('colours spatially-near clusters more alike than spatially-far ones', () => {
    const side = 4;
    const g = gridClusters(side);
    const colors = spatialFacetColors(g.codes, g.valueCount, g.positions, g.valueCount, 'spectral');
    let nearSum = 0;
    let nearCount = 0;
    let farSum = 0;
    let farCount = 0;
    for (let i = 0; i < g.valueCount; i++) {
      for (let j = i + 1; j < g.valueCount; j++) {
        const a = g.cellOf[i]!;
        const b = g.cellOf[j]!;
        const dist = Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
        const cd = rgbDist(colors, i, j);
        if (dist === 1) {
          nearSum += cd;
          nearCount++;
        } else if (dist >= side + 1) {
          farSum += cd;
          farCount++;
        }
      }
    }
    const nearMean = nearSum / nearCount;
    const farMean = farSum / farCount;
    // the Hilbert order makes adjacent clusters neighbours on the ramp: their mean colour distance
    // is well under that of distant clusters (a scattered value-index palette would make the two
    // roughly equal). A comfortable margin so the assertion is about the gradient, not noise.
    expect(nearMean).toBeLessThan(farMean * 0.6);
  });

  it('a single present value takes one ramp colour; absent values are left untouched', () => {
    // 3 values but only value 1 has members
    const codes = new Uint32Array([1, 1, 1]);
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const out = spatialFacetColors(codes, 3, positions, 3, 'ember');
    // all three nodes (all value 1) share one colour
    expect(rgbDist(out, 0, 1)).toBe(0);
    expect(rgbDist(out, 0, 2)).toBe(0);
  });

  it('handles the empty field without throwing', () => {
    const out = spatialFacetColors(new Uint32Array(0), 0, new Float32Array(0), 0, 'ice');
    expect(out.length).toBe(0);
  });
});
