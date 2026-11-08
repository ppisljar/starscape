import { describe, expect, it } from 'vitest';
import { buildCSR } from '../src/core/csr';

describe('buildCSR', () => {
  // 6 nodes; node 5 has no links. Degrees: 0→3, 1→2, 2→3, 3→1, 4→1, 5→0.
  const links = Uint32Array.from([0, 1, 1, 2, 2, 3, 0, 2, 4, 0]);
  const csr = buildCSR(6, links);

  it('offsets are the degree prefix sum', () => {
    expect(Array.from(csr.off)).toEqual([0, 3, 5, 8, 9, 10, 10]);
    expect(csr.nbr.length).toBe(10); // 2m
    expect(csr.edge.length).toBe(10);
  });

  it('neighbour sets match the hand-built graph', () => {
    const nbrsOf = (i: number) =>
      Array.from(csr.nbr.subarray(csr.off[i]!, csr.off[i + 1]!)).sort((a, b) => a - b);
    expect(nbrsOf(0)).toEqual([1, 2, 4]);
    expect(nbrsOf(1)).toEqual([0, 2]);
    expect(nbrsOf(2)).toEqual([0, 1, 3]);
    expect(nbrsOf(3)).toEqual([2]);
    expect(nbrsOf(4)).toEqual([0]);
    expect(nbrsOf(5)).toEqual([]);
  });

  it('edge indices round-trip: each slot names a link whose endpoints are {i, nbr}', () => {
    for (let i = 0; i < 6; i++) {
      for (let k = csr.off[i]!; k < csr.off[i + 1]!; k++) {
        const e = csr.edge[k]!;
        const a = links[e * 2]!;
        const b = links[e * 2 + 1]!;
        const j = csr.nbr[k]!;
        expect([a, b].sort((x, y) => x - y)).toEqual([i, j].sort((x, y) => x - y));
      }
    }
    // every link index appears exactly twice (once per endpoint)
    const seen = new Uint32Array(links.length >> 1);
    for (let k = 0; k < csr.edge.length; k++) {
      const e = csr.edge[k]!;
      seen[e] = seen[e]! + 1;
    }
    expect(Array.from(seen)).toEqual([2, 2, 2, 2, 2]);
  });

  it('handles the empty graph', () => {
    const empty = buildCSR(0, new Uint32Array(0));
    expect(Array.from(empty.off)).toEqual([0]);
    expect(empty.nbr.length).toBe(0);
    expect(empty.edge.length).toBe(0);
  });

  it('handles nodes with links absent entirely', () => {
    const c = buildCSR(3, new Uint32Array(0));
    expect(Array.from(c.off)).toEqual([0, 0, 0, 0]);
    expect(c.nbr.length).toBe(0);
  });
});
