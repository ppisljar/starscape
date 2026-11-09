/* cellMaxOf (src/pick/grid.ts) — the stage-9 per-cell channel maximum the candidate sweep
 * prunes whole cells with. Pure over the CSR grid; pinned against a brute-force recompute. */

import { describe, expect, it } from 'vitest';
import { buildGrid, cellMaxOf } from '../src/pick/grid';
import { hash } from '../src/core/hash';

function corpus(n: number, seed: number): { pos: Float32Array; size: Float32Array } {
  const pos = new Float32Array(n * 3);
  const size = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = hash(i, seed) * 100;
    pos[i * 3 + 1] = hash(i, seed + 1) * 100;
    pos[i * 3 + 2] = hash(i, seed + 2) * 100;
    size[i] = 1.8 + hash(i, seed + 3) * 20;
  }
  return { pos, size };
}

describe('cellMaxOf', () => {
  it('matches a brute-force per-cell maximum, empty cells read 0', () => {
    const { pos, size } = corpus(500, 7);
    const grid = buildGrid(pos, 500);
    const got = cellMaxOf(grid, (i) => size[i]!);
    expect(got.length).toBe(grid.nCells);
    // brute force over the same CSR cells
    for (let k = 0; k < grid.nCells; k++) {
      let mx = 0;
      for (let q = grid.start[k]!; q < grid.start[k + 1]!; q++) {
        mx = Math.max(mx, size[grid.order[q]!]!);
      }
      expect(got[k]).toBe(mx);
    }
    // every node's cell maximum dominates its own value
    for (let i = 0; i < 500; i++) {
      // find i's cell through the order array (the grid owns the bucketing)
      for (let k = 0; k < grid.nCells; k++) {
        for (let q = grid.start[k]!; q < grid.start[k + 1]!; q++) {
          if (grid.order[q] === i) expect(got[k]!).toBeGreaterThanOrEqual(size[i]!);
        }
      }
    }
  });

  it('reuses a caller buffer when large enough and overwrites stale content', () => {
    const { pos, size } = corpus(64, 3);
    const grid = buildGrid(pos, 64);
    const buf = new Float32Array(grid.nCells).fill(123);
    const got = cellMaxOf(grid, (i) => size[i]!, buf);
    expect(got).toBe(buf); // no allocation on the rebuild path
    const fresh = cellMaxOf(grid, (i) => size[i]!);
    expect(Array.from(got.subarray(0, grid.nCells))).toEqual(Array.from(fresh));
    // too-small buffers are not written past their end — a fresh array comes back instead
    const small = new Float32Array(1);
    expect(cellMaxOf(grid, (i) => size[i]!, small)).not.toBe(small);
  });
});
