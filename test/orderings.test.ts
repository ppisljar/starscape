/* The ordering library (src/place/orderings.ts): byFacet, byColumn, byAdjacency, byHash.
 *
 * The ONE contract every ordering carries (src/place/types.ts): its result is a
 * PERMUTATION of the equal-spaced quantiles (rank+0.5)/count — an ordering seats members,
 * never spaces them. Each suite below pins that plus the ordering's own rule, every
 * tie-break by member id (never by the caller's buffer order), and determinism goldens.
 */

import { describe, expect, it } from 'vitest';
import { buildGraph } from '../src/core/graph';
import { fnv1a } from '../src/core/fnv';
import { hash } from '../src/core/hash';
import { placeGroup } from '../src/place/engine';
import { getOrdering, registerShape } from '../src/place/registry';
import '../src/place/orderings';
import type { Graph, GraphInput } from '../src/core/types';
import type { PlaceInput, Shape } from '../src/place/types';

/* ---------------------------------------------------------------- fixtures */

/** The small hand-checkable graph, 12 nodes:
 *  chain 0-1-2-3-4, hub 5 linked to {0,1,2,6}, pair 7-8, isolated 9, pair 10-11.
 *  Degrees: 0:2  1:3  2:3  3:2  4:1  5:4  6:1  7:1  8:1  9:0  10:1  11:1. */
function smallGraph(): Graph {
  const input: GraphInput = {
    n: 12,
    links: Uint32Array.from([0, 1, 1, 2, 2, 3, 3, 4, 5, 0, 5, 1, 5, 2, 5, 6, 7, 8, 10, 11]),
    columns: {
      // ties on purpose: nodes 2 and 7 share 4; nodes 0 and 9 share 8
      words: Float32Array.from([8, 2, 4, 16, 1, 32, 64, 4, 128, 8, 256, 512]),
    },
    facets: {
      kind: {
        values: ['a', 'b', 'c'],
        data: Uint8Array.from([0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]),
      },
    },
  };
  return buildGraph(input);
}

const G = smallGraph();

function inputOf(members: number[], graph: Graph | null = G, salt = 7): PlaceInput {
  return {
    members: Uint32Array.from(members),
    count: members.length,
    graph,
    salt,
    radius: 10,
    out: new Float32Array(members.length * 3),
    outOffset: 0,
  };
}

/** rank_k from the quantile q_k = (rank + 0.5)/count. */
function ranksOf(q: Float32Array, count: number): number[] {
  const r: number[] = [];
  for (let k = 0; k < count; k++) r.push(Math.round(q[k]! * count - 0.5));
  return r;
}

/** The permutation contract: sorted(q) === the equal-spaced grid, exactly (both sides
 *  take the same single f64→f32 rounding, so equality is exact, not approximate). */
function expectPermutation(q: Float32Array, count: number): void {
  const got = Array.from(q.subarray(0, count)).sort((a, b) => a - b);
  const want: number[] = [];
  for (let r = 0; r < count; r++) want.push(Math.fround((r + 0.5) / count));
  expect(got).toEqual(want);
}

/* ---------------------------------------------------------------- byFacet */

describe('byFacet', () => {
  const byFacet = getOrdering('byFacet');
  const SP = { facet: 'kind' };

  it('emits a permutation of the equal-spaced quantiles', () => {
    const inp = inputOf([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expectPermutation(byFacet.assign(inp, {}, SP), inp.count);
  });

  it('seats contiguous sectors in facet-value order, ascending id inside a sector', () => {
    // deliberately scrambled member buffer — the result may not depend on it
    const members = [7, 0, 11, 2, 5, 9, 4, 8];
    // kind of each:  b  a   a  a  a  c  b  a
    const inp = inputOf(members);
    const q = byFacet.assign(inp, {}, SP);
    const rank = ranksOf(q, members.length);
    const rankOf = (id: number) => rank[members.indexOf(id)]!;
    // sector 'a' (value 0): ids 0,2,5,8,11 ascending -> ranks 0..4
    expect([0, 2, 5, 8, 11].map(rankOf)).toEqual([0, 1, 2, 3, 4]);
    // sector 'b' (value 1): ids 4,7 -> ranks 5,6
    expect([4, 7].map(rankOf)).toEqual([5, 6]);
    // sector 'c' (value 2): id 9 -> rank 7
    expect(rankOf(9)).toBe(7);
  });

  it('is independent of the member buffer order', () => {
    const a = byFacet.assign(inputOf([3, 6, 1, 10, 0]), {}, SP);
    const b = byFacet.assign(inputOf([0, 1, 3, 6, 10]), {}, SP);
    const members = [3, 6, 1, 10, 0];
    const sortedMembers = [0, 1, 3, 6, 10];
    for (let k = 0; k < members.length; k++) {
      expect(a[k]).toBe(b[sortedMembers.indexOf(members[k]!)]);
    }
  });

  it('throws descriptively on null graph, missing key, unknown facet, bad member id', () => {
    expect(() => byFacet.assign(inputOf([0, 1], null), {}, SP)).toThrowError(/graph is null/);
    expect(() => byFacet.assign(inputOf([0, 1]), {}, {})).toThrowError(/stringParams\.facet/);
    expect(() => byFacet.assign(inputOf([0, 1]), {}, undefined)).toThrowError(
      /stringParams\.facet/,
    );
    expect(() => byFacet.assign(inputOf([0, 1]), {}, { facet: 'nope' })).toThrowError(
      /unknown facet "nope" — known: kind/,
    );
    expect(() => byFacet.assign(inputOf([0, 99]), {}, SP)).toThrowError(/member id 99/);
  });
});

/* ---------------------------------------------------------------- byColumn */

describe('byColumn', () => {
  const byColumn = getOrdering('byColumn');
  const SP = { column: 'words' };

  it('emits a permutation of the equal-spaced quantiles', () => {
    const inp = inputOf([11, 3, 7, 0, 5, 9, 2]);
    expectPermutation(byColumn.assign(inp, {}, SP), inp.count);
  });

  it('is a rank gradient: quantiles ascend with the column value', () => {
    const members = [10, 4, 6, 1, 3]; // words: 256, 1, 64, 2, 16
    const inp = inputOf(members);
    const q = byColumn.assign(inp, {}, SP);
    // sort members by their quantile; column values must be non-decreasing
    const order = members
      .map((id, k) => ({ id, q: q[k]! }))
      .sort((x, y) => x.q - y.q)
      .map((e) => e.id);
    expect(order).toEqual([4, 1, 3, 6, 10]); // words 1, 2, 16, 64, 256
  });

  it('breaks ties by member id', () => {
    // words: node 2 -> 4, node 7 -> 4 (tie); node 0 -> 8, node 9 -> 8 (tie)
    const members = [9, 7, 2, 0];
    const q = byColumn.assign(inputOf(members), {}, SP);
    const rank = ranksOf(q, members.length);
    const rankOf = (id: number) => rank[members.indexOf(id)]!;
    expect(rankOf(2)).toBe(0); // 4, lower id first
    expect(rankOf(7)).toBe(1); // 4
    expect(rankOf(0)).toBe(2); // 8, lower id first
    expect(rankOf(9)).toBe(3); // 8
  });

  it('works on the degree column buildGraph computed', () => {
    const members = [4, 5, 0]; // degree 1, 4, 2
    const q = byColumn.assign(inputOf(members), {}, { column: 'degree' });
    expect(ranksOf(q, 3)).toEqual([0, 2, 1]);
  });

  it('a column mixing NaN with numbers stays a total order: values ascend, NaNs seat last by id', () => {
    // NaN compares false both ways, so a comparator that sent NaN pairs to the id
    // tie-break while numbers compared by value was INTRANSITIVE — the emitted order was
    // whatever the engine's sort felt like (the measured ranks put the SMALLEST value at
    // the top quantile). The NaN partition pins the spec-defined order instead.
    const nanGraph = buildGraph({
      n: 6,
      links: new Uint32Array(0),
      columns: { w: Float32Array.from([5, NaN, 3, NaN, 4, 1]) },
    });
    const members = [0, 1, 2, 3, 4, 5];
    const q = byColumn.assign(inputOf(members, nanGraph), {}, { column: 'w' });
    // numbers by value: 1 (m5), 3 (m2), 4 (m4), 5 (m0); then NaNs by id: m1, m3
    expect(ranksOf(q, 6)).toEqual([3, 4, 1, 5, 2, 0]);
    // and the seating cannot depend on the caller's buffer order
    const shuffled = [3, 5, 0, 4, 2, 1];
    const q2 = byColumn.assign(inputOf(shuffled, nanGraph), {}, { column: 'w' });
    const rank2 = ranksOf(q2, 6);
    const rank1 = ranksOf(q, 6);
    for (let k = 0; k < 6; k++) {
      expect(rank2[k]).toBe(rank1[members.indexOf(shuffled[k]!)]);
    }
  });

  it('an all-NaN column degenerates to the id order', () => {
    const nanGraph = buildGraph({
      n: 3,
      links: new Uint32Array(0),
      columns: { w: Float32Array.from([NaN, NaN, NaN]) },
    });
    const q = byColumn.assign(inputOf([2, 0, 1], nanGraph), {}, { column: 'w' });
    expect(ranksOf(q, 3)).toEqual([2, 0, 1]);
  });

  it('throws descriptively on null graph, missing key, unknown column', () => {
    expect(() => byColumn.assign(inputOf([0], null), {}, SP)).toThrowError(/graph is null/);
    expect(() => byColumn.assign(inputOf([0]), {}, {})).toThrowError(/stringParams\.column/);
    expect(() => byColumn.assign(inputOf([0]), {}, { column: 'nope' })).toThrowError(
      /unknown column "nope" — known: degree, words/,
    );
  });
});

/* ---------------------------------------------------------------- byAdjacency */

describe('byAdjacency', () => {
  const byAdjacency = getOrdering('byAdjacency');

  it('emits a permutation of the equal-spaced quantiles', () => {
    const inp = inputOf([8, 2, 11, 5, 0, 7]);
    expectPermutation(byAdjacency.assign(inp, {}), inp.count);
  });

  it('starts at the highest-degree member and seats its component before the rest', () => {
    const members = [8, 4, 0, 7, 5, 2, 6, 1, 3]; // scrambled; hub 5 has degree 4
    const inp = inputOf(members);
    const q = byAdjacency.assign(inp, {});
    const rank = ranksOf(q, members.length);
    const rankOf = (id: number) => rank[members.indexOf(id)]!;
    expect(rankOf(5)).toBe(0); // BFS root: the hub
    // 5's component within the group: {0,1,2,3,4,5,6} -> ranks 0..6
    for (const id of [0, 1, 2, 3, 4, 6]) expect(rankOf(id)).toBeLessThan(7);
    // unreached pair 7,8: appended by ascending id
    expect(rankOf(7)).toBe(7);
    expect(rankOf(8)).toBe(8);
  });

  it('breaks the degree tie at the start by lowest id', () => {
    // group {0,1,2}: full-graph degrees 2,3,3 — tie between 1 and 2, 1 wins
    const q = byAdjacency.assign(inputOf([2, 1, 0]), {});
    const rank = ranksOf(q, 3);
    expect(rank[1]).toBe(0); // member 1
  });

  it('never leaves the member set: paths through non-members do not exist', () => {
    // 0 and 2 are linked only via 1 or 5, neither in the group -> 2 (deg 3) roots,
    // 0 is unreached and appended
    const q = byAdjacency.assign(inputOf([0, 2]), {});
    expect(ranksOf(q, 2)).toEqual([1, 0]);
  });

  it('keeps linked members nearby on a chain (the seriation point of it)', () => {
    // pure chain 0-1-…-7; BFS from an internal start grows both ways, so linked
    // pairs end up at most 2 ranks apart — the property, not an exact order
    const chain = buildGraph({
      n: 8,
      links: Uint32Array.from([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]),
    });
    const members = [5, 1, 7, 3, 0, 6, 2, 4];
    const inp = inputOf(members, chain);
    const q = byAdjacency.assign(inp, {});
    const rank = ranksOf(q, members.length);
    const rankOf = (id: number) => rank[members.indexOf(id)]!;
    for (let i = 0; i + 1 < 8; i++) {
      expect(Math.abs(rankOf(i) - rankOf(i + 1))).toBeLessThanOrEqual(2);
    }
  });

  it('a group with no in-group links falls back to ascending id order', () => {
    const members = [9, 4, 7]; // mutually unlinked
    const q = byAdjacency.assign(inputOf(members), {});
    const rank = ranksOf(q, 3);
    const rankOf = (id: number) => rank[members.indexOf(id)]!;
    // 4 roots (degree 1, lowest id among the degree-1 tie 4 vs 7); then 7, 9 by id
    expect(rankOf(4)).toBe(0);
    expect(rankOf(7)).toBe(1);
    expect(rankOf(9)).toBe(2);
  });

  it('throws on a null graph', () => {
    expect(() => byAdjacency.assign(inputOf([0, 1], null), {})).toThrowError(/graph is null/);
  });
});

/* ---------------------------------------------------------------- byHash */

describe('byHash', () => {
  const byHash = getOrdering('byHash');

  it('emits a permutation of the equal-spaced quantiles', () => {
    const members: number[] = [];
    for (let i = 0; i < 97; i++) members.push(i * 5 + 2);
    const inp = inputOf(members, null);
    expectPermutation(byHash.assign(inp, {}), inp.count);
  });

  it('actually shuffles (not the identity), yet deterministically', () => {
    const members: number[] = [];
    for (let i = 0; i < 64; i++) members.push(i);
    const a = byHash.assign(inputOf(members, null, 3), {});
    const b = byHash.assign(inputOf(members, null, 3), {});
    expect(Array.from(b)).toEqual(Array.from(a));
    let moved = 0;
    for (let k = 0; k < 64; k++) {
      if (a[k] !== Math.fround((k + 0.5) / 64)) moved++;
    }
    expect(moved).toBeGreaterThan(48); // a real shuffle displaces nearly everyone
  });

  it('different group salts deal different hands; params.shuffle re-deals in place', () => {
    const members = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const a = byHash.assign(inputOf(members, null, 1), {});
    const b = byHash.assign(inputOf(members, null, 2), {});
    const c = byHash.assign(inputOf(members, null, 1), { shuffle: 1 });
    expect(Array.from(a)).not.toEqual(Array.from(b));
    expect(Array.from(a)).not.toEqual(Array.from(c));
  });

  it('keys off member ids, not buffer positions', () => {
    const a = byHash.assign(inputOf([10, 20, 30], null, 5), {});
    const b = byHash.assign(inputOf([30, 10, 20], null, 5), {});
    // member 10's quantile is the same wherever it sits in the buffer
    expect(b[1]).toBe(a[0]);
    expect(b[2]).toBe(a[1]);
    expect(b[0]).toBe(a[2]);
  });
});

/* ---------------------------------------------------------------- through the engine */

describe('stringParams plumbing through placeGroup', () => {
  /** dim-1 passthrough: x = u0, so the engine's composed axis-0 values are readable. */
  const lineOrd: Shape = {
    key: 'test:lineOrd',
    dim: 1,
    place(u, _radius, count, out, outOffset): void {
      for (let k = 0; k < count; k++) {
        const o = outOffset + k * 3;
        out[o] = u[k]!;
        out[o + 1] = 0;
        out[o + 2] = 0;
      }
    },
  };
  registerShape(lineOrd);

  it('spec.stringParams reaches the ordering', () => {
    const members = [7, 0, 11, 2, 5];
    const inp = inputOf(members);
    placeGroup(
      { shape: 'test:lineOrd', ordering: 'byFacet', stringParams: { facet: 'kind' } },
      inp,
    );
    const direct = getOrdering('byFacet').assign(inputOf(members), {}, { facet: 'kind' });
    for (let k = 0; k < members.length; k++) expect(inp.out[k * 3]).toBe(direct[k]);
  });

  it('a spec that omits a required string key fails loudly at placeGroup', () => {
    const inp = inputOf([0, 1, 2]);
    expect(() => placeGroup({ shape: 'test:lineOrd', ordering: 'byColumn' }, inp)).toThrowError(
      /stringParams\.column/,
    );
  });
});

/* ---------------------------------------------------------------- determinism goldens */

describe('determinism goldens (DESIGN §10.9)', () => {
  /** A 200-node hash-grown graph: preferential chain + extra links, one facet, one
   *  column — all from hash(i, salt), so this corpus is pinned forever. */
  function goldenGraph(): Graph {
    const n = 200;
    const links: number[] = [];
    for (let i = 1; i < n; i++) {
      links.push(Math.floor(hash(i, 1) * i), i);
    }
    for (let e = 0; e < 100; e++) {
      const a = Math.floor(hash(e, 2) * n);
      const b = Math.floor(hash(e, 3) * n);
      if (a !== b) links.push(a, b);
    }
    const kind = new Uint8Array(n);
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      kind[i] = Math.floor(hash(i, 4) * 4);
      w[i] = Math.fround(hash(i, 5) * 100);
    }
    return buildGraph({
      n,
      links: Uint32Array.from(links),
      columns: { w },
      facets: { kind: { values: ['a', 'b', 'c', 'd'], data: kind } },
    });
  }

  const GG = goldenGraph();
  /** A scrambled ~70% member subset, deterministic via hash. */
  const MEMBERS = (() => {
    const picked: number[] = [];
    for (let i = 0; i < GG.n; i++) if (hash(i, 6) < 0.7) picked.push(i);
    return picked.sort((a, b) => hash(a, 7) - hash(b, 7));
  })();

  function qOf(key: string, stringParams?: Record<string, string>): Float32Array {
    const inp = inputOf(MEMBERS, GG, 42);
    const q = getOrdering(key).assign(inp, {}, stringParams);
    return q.slice(0, MEMBERS.length); // own the bytes before fingerprinting
  }

  // Recorded from the first run and pinned. A change means an ordering's byte output
  // changed: deliberate only alongside a documented contract change.
  const GOLDENS: Array<[string, Record<string, string> | undefined, number]> = [
    ['byFacet', { facet: 'kind' }, 2779558060],
    ['byColumn', { column: 'w' }, 72763576],
    ['byAdjacency', undefined, 1411291296],
    ['byHash', undefined, 4005005020],
  ];

  for (const [key, sp, golden] of GOLDENS) {
    it(`${key}: fixed graph + members pin a recorded FNV-1a digest`, () => {
      const q = qOf(key, sp);
      expectPermutation(q, MEMBERS.length);
      expect(fnv1a(qOf(key, sp))).toBe(fnv1a(q)); // twice, byte-identical
      expect(fnv1a(q)).toBe(golden);
    });
  }
});
