/* The local-density channel (src/field/density.ts) and its tie-collapsing rank + ramp helpers.
 * Headless, pure — no GPU, no three.js: a per-node neighbourhood count from a uniform grid, then
 * the midrank normalisation and ramp binding the field colours by. The look itself is browser-
 * tuned; these pin the signal's correctness and determinism (DESIGN §10).
 */

import { describe, expect, it } from 'vitest';
import { localDensity, DENSITY_TARGET_PER_CELL } from '../src/field/density';
import { rankPercentiles, rankPercentilesTied } from '../src/field/channels';
import { rampColors, rampColorsTied } from '../src/theme/palettes';

/** A corpus of two tight blobs FAR apart: `dense` nodes near the origin, `sparse` nodes near
 *  (1e6, 1e6, 1e6). The gap dwarfs the blobs, so whatever cell size the grid picks the two land in
 *  distant cells whose 3×3×3 blocks never overlap — a clean crowded-vs-lonely contrast that does
 *  not depend on the auto cell-sizing. Deterministic (a fixed hash walk, no RNG). */
function twoRegions(dense: number, sparse: number): {
  pos: Float32Array;
  n: number;
  dense: number;
  sparse: number;
} {
  const n = dense + sparse;
  const pos = new Float32Array(n * 3);
  const blob = (i: number, cx: number, cy: number, cz: number): void => {
    const h = (((i + 1) * 2654435761) >>> 0) || 1;
    pos[i * 3] = cx + ((h & 255) / 255 - 0.5) * 2;
    pos[i * 3 + 1] = cy + (((h >> 8) & 255) / 255 - 0.5) * 2;
    pos[i * 3 + 2] = cz + (((h >> 16) & 255) / 255 - 0.5) * 2;
  };
  for (let i = 0; i < dense; i++) blob(i, 0, 0, 0);
  for (let j = 0; j < sparse; j++) blob(dense + j, 1e6, 1e6, 1e6);
  return { pos, n, dense, sparse };
}

describe('localDensity (the geometric channel)', () => {
  it('counts a crowded neighbourhood higher than a lonely one', () => {
    const { pos, n, dense, sparse } = twoRegions(400, 8);
    const d = localDensity(pos, n);
    expect(d.length).toBe(n);
    let clusterMin = Infinity;
    for (let i = 0; i < dense; i++) clusterMin = Math.min(clusterMin, d[i]!);
    let isoMax = 0;
    for (let i = dense; i < n; i++) isoMax = Math.max(isoMax, d[i]!);
    expect(clusterMin).toBeGreaterThan(isoMax); // dense ≫ sparse
    expect(isoMax).toBeLessThanOrEqual(sparse); // the far blob's block holds only its own few
  });

  it('is deterministic: identical positions → byte-identical counts', () => {
    const { pos, n } = twoRegions(300, 30);
    expect(localDensity(pos, n)).toEqual(localDensity(pos, n));
  });

  it('every count is finite and ≥ 1 (a node is in its own block)', () => {
    const { pos, n } = twoRegions(200, 50);
    const d = localDensity(pos, n);
    for (let i = 0; i < n; i++) {
      expect(Number.isFinite(d[i]!)).toBe(true);
      expect(d[i]!).toBeGreaterThanOrEqual(1);
    }
  });

  it('reuses a caller buffer (the field recolours in place across retargets)', () => {
    const { pos, n } = twoRegions(100, 10);
    const out = new Float32Array(n);
    const ret = localDensity(pos, n, out);
    expect(ret).toBe(out);
  });

  it('handles n = 0 and n = 1', () => {
    expect(localDensity(new Float32Array(0), 0).length).toBe(0);
    expect(Array.from(localDensity(Float32Array.from([1, 2, 3]), 1))).toEqual([1]);
  });

  it('is scale-invariant in RANK: a uniform scale of the field cannot change the ordering', () => {
    // density colours by rank, and the grid measures its own extent — so scaling every
    // coordinate (what `spread` does) must leave the rank-normalised result identical.
    const { pos, n } = twoRegions(250, 25);
    const big = new Float32Array(pos.length);
    for (let i = 0; i < pos.length; i++) big[i] = pos[i]! * 7.5;
    const a = rankPercentilesTied(localDensity(pos, n), n);
    const b = rankPercentilesTied(localDensity(big, n), n);
    expect(a).toEqual(b);
  });

  it('exports a sane default cell target', () => {
    expect(DENSITY_TARGET_PER_CELL).toBeGreaterThan(0);
  });
});

describe('rankPercentilesTied (ties share one colour)', () => {
  it('collapses a tie group to its midrank instead of an index gradient', () => {
    // three equal values would rank 0, 0.5, 1 under the index tiebreak; tied they all share
    // the midrank (0 + 2)/2 / (n−1) = 0.5
    const p = rankPercentilesTied(Float32Array.from([5, 5, 5]), 3);
    expect(Array.from(p)).toEqual([0.5, 0.5, 0.5]);
  });

  it('a sparse plateau reads as ONE low value, not a spread (the #5 blue-nodes property)', () => {
    // 6 sparse (value 1), 2 dense (values 2, 3): every sparse node must share the SAME low
    // percentile — the midrank of ranks 0..5 = 2.5/7 ≈ 0.357 — so they colour identically
    const p = rankPercentilesTied(Float32Array.from([1, 1, 1, 1, 1, 1, 2, 3]), 8);
    for (let i = 0; i < 6; i++) expect(p[i]!).toBeCloseTo(2.5 / 7, 6); // f32 storage
    expect(p[6]!).toBeCloseTo(6 / 7, 6);
    expect(p[7]!).toBe(1);
  });

  it('with no ties it equals rankPercentiles', () => {
    const data = Float32Array.from([10, 30, 20, 40]);
    expect(rankPercentilesTied(data, 4)).toEqual(rankPercentiles(data, 4));
  });

  it('handles n = 0 and n = 1', () => {
    expect(rankPercentilesTied(new Float32Array(0), 0).length).toBe(0);
    expect(Array.from(rankPercentilesTied(Float32Array.from([7]), 1))).toEqual([1]);
  });
});

describe('rampColorsTied (the density → colour binding)', () => {
  it('paints a sparse plateau one flat colour (no false blue→orange order within it)', () => {
    const colors = rampColorsTied(Float32Array.from([1, 1, 1, 1, 5]), 5, 'iceFire');
    // the four tied sparse nodes share a colour exactly
    for (let c = 0; c < 3; c++) {
      expect(colors[c]).toBe(colors[3 + c]);
      expect(colors[c]).toBe(colors[6 + c]);
      expect(colors[c]).toBe(colors[9 + c]);
    }
    // and the sparse colour is cooler (more blue than red) than the lone dense node's
    expect(colors[2]!).toBeGreaterThan(colors[0]!); // sparse: blue > red
    expect(colors[12]!).toBeGreaterThan(colors[14]!); // dense (value 5, top rank): red > blue
  });

  it('matches rampColors when there are no ties', () => {
    const data = Float32Array.from([3, 1, 2, 4]);
    expect(rampColorsTied(data, 4, 'nebula')).toEqual(rampColors(data, 4, 'nebula'));
  });
});
