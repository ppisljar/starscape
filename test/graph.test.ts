import { describe, expect, it } from 'vitest';
import { buildGraph, incident, neighbours } from '../src/core/graph';
import { hash } from '../src/core/hash';
import type { GraphInput } from '../src/core/types';

/* FNV-1a over the little-endian bytes of Uint32Arrays — the golden-test fingerprint
 * (DESIGN.md §10.9: fixed input → byte-identical buffers, forever). */
function fnv1a(arrays: Uint32Array[]): number {
  let h = 0x811c9dc5;
  for (const arr of arrays) {
    for (let i = 0; i < arr.length; i++) {
      let v = arr[i]!;
      for (let b = 0; b < 4; b++) {
        h = (h ^ (v & 0xff)) >>> 0;
        h = Math.imul(h, 0x01000193) >>> 0;
        v >>>= 8;
      }
    }
  }
  return h >>> 0;
}

describe('buildGraph: dedup', () => {
  it('collapses a->b, b->a, duplicates, and self-links into unique undirected pairs', () => {
    // directed soup: 0->1, 1->0, 2->2 (self), 0->1 again, 1->2, 2->1
    const g = buildGraph({
      n: 4,
      links: Uint32Array.from([0, 1, 1, 0, 2, 2, 0, 1, 1, 2, 2, 1]),
    });
    // no positions/facet -> no reordering, so the buffer is the canonical pairs in
    // ascending (lo, hi) order — the bucketed dedup emits them sorted, not first-seen
    expect(Array.from(g.links)).toEqual([0, 1, 1, 2]);
    expect(g.counts).toEqual({ nodes: 4, links: 2 });
  });

  it('canonicalises orientation: a pair arrives as (lo, hi) regardless of direction', () => {
    const g = buildGraph({ n: 5, links: Uint32Array.from([4, 1, 3, 0]) });
    expect(Array.from(g.links)).toEqual([0, 3, 1, 4]);
  });

  it('self-link-only input: zero links, zero degrees, empty CSR', () => {
    const g = buildGraph({ n: 3, links: Uint32Array.from([0, 0, 1, 1, 2, 2]) });
    expect(g.counts.links).toBe(0);
    expect(g.links.length).toBe(0);
    expect(Array.from(g.columns['degree']!.data)).toEqual([0, 0, 0]);
    expect(Array.from(g.csr.off)).toEqual([0, 0, 0, 0]);
    expect(g.csr.nbr.length).toBe(0);
    expect(g.csr.edge.length).toBe(0);
  });

  it('handles the empty graph', () => {
    const g = buildGraph({ n: 0, links: new Uint32Array(0) });
    expect(g.counts).toEqual({ nodes: 0, links: 0 });
    expect(g.links.length).toBe(0);
    expect(Array.from(g.csr.off)).toEqual([0]);
  });
});

describe('buildGraph: columns', () => {
  it('computes degree over the deduped undirected links', () => {
    // after dedup: (0,1), (1,2) — degrees 1, 2, 1, 0
    const g = buildGraph({
      n: 4,
      links: Uint32Array.from([0, 1, 1, 0, 2, 2, 0, 1, 1, 2]),
    });
    const degree = g.columns['degree']!;
    expect(degree.key).toBe('degree');
    expect(Array.from(degree.data)).toEqual([1, 2, 1, 0]);
  });

  it('a provided degree column is passed through, not recomputed', () => {
    const mine = Uint32Array.from([7, 7, 7]);
    const g = buildGraph({
      n: 3,
      links: Uint32Array.from([0, 1]),
      columns: { degree: mine },
    });
    expect(g.columns['degree']!.data).toBe(mine);
  });

  it('other provided columns pass through by reference', () => {
    const words = Float32Array.from([1, 2, 3]);
    const g = buildGraph({ n: 3, links: new Uint32Array(0), columns: { words } });
    expect(g.columns['words']!.data).toBe(words);
    expect(g.columns['degree']).toBeDefined(); // degree still computed alongside
  });
});

describe('buildGraph: facets', () => {
  it('facets pass through by reference with their key stamped on', () => {
    const data = Uint8Array.from([0, 1, 0]);
    const values = ['note', 'person'];
    const g = buildGraph({
      n: 3,
      links: new Uint32Array(0),
      facets: { type: { values, data } },
    });
    expect(g.facets['type']!.data).toBe(data);
    expect(g.facets['type']!.values).toBe(values);
    expect(g.facets['type']!.key).toBe('type');
  });

  it('labelOf uses the provided source, or "#i" without one — no per-node strings', () => {
    const labelled = buildGraph({ n: 2, links: new Uint32Array(0), labels: (i) => `page ${i}` });
    expect(labelled.labelOf(1)).toBe('page 1');
    const bare = buildGraph({ n: 2, links: new Uint32Array(0) });
    expect(bare.labelOf(0)).toBe('#0');
    expect(bare.labelOf(1)).toBe('#1');
  });
});

describe('buildGraph: validation', () => {
  it('throws on an out-of-range link endpoint, naming the offending link index', () => {
    const bad = Uint32Array.from([0, 1, 2, 9, 1, 0]);
    expect(() => buildGraph({ n: 3, links: bad })).toThrowError(/link 1\b.*\b9\b/);
  });

  it('throws on a facet whose data length is not n', () => {
    expect(() =>
      buildGraph({
        n: 4,
        links: new Uint32Array(0),
        facets: { type: { values: ['x'], data: new Uint8Array(3) } },
      }),
    ).toThrowError(/facet "type"/);
  });

  it('throws on a facet code outside its values table, naming facet/node/code/size', () => {
    expect(() =>
      buildGraph({
        n: 2,
        links: new Uint32Array(0),
        facets: { type: { values: ['note'], data: Uint8Array.from([0, 7]) } },
      }),
    ).toThrowError(/facet "type" code 7 at node 1 .*1 entries/);
  });

  it('throws on a column whose length is not n', () => {
    expect(() =>
      buildGraph({ n: 4, links: new Uint32Array(0), columns: { w: new Uint32Array(5) } }),
    ).toThrowError(/column "w"/);
  });

  it('throws on positions of the wrong length', () => {
    expect(() =>
      buildGraph({ n: 3, positions: new Float32Array(8), links: new Uint32Array(0) }),
    ).toThrowError(/positions has 8 floats, expected 3n = 9/);
  });

  it('throws on an odd-length links buffer instead of silently dropping the stray element', () => {
    expect(() =>
      buildGraph({ n: 3, links: Uint32Array.from([0, 1, 2]) }),
    ).toThrowError(/odd length 3/);
  });

  it('throws when orderByFacet names no facet', () => {
    expect(() =>
      buildGraph(
        { n: 2, positions: new Float32Array(6), links: new Uint32Array(0) },
        { orderByFacet: 'nope' },
      ),
    ).toThrowError(/orderByFacet "nope"/);
  });
});

describe('buildGraph: link ordering and CSR agreement', () => {
  it('orders intra-group links before inter-group when orderByFacet is given', () => {
    // groups: {0,1} and {2,3}; intra links (0,1), (2,3); inter link (0,2).
    // The inter link is also the SHORTEST — ordering must still put it last.
    const positions = Float32Array.from([
      0, 0, 0, // 0
      9, 0, 0, // 1   (0,1) long intra
      1, 0, 0, // 2   (0,2) short inter
      8, 0, 0, // 3   (2,3) long intra
    ]);
    const g = buildGraph(
      {
        n: 4,
        positions,
        links: Uint32Array.from([0, 2, 0, 1, 2, 3]),
        facets: { group: { values: ['a', 'b'], data: Uint8Array.from([0, 0, 1, 1]) } },
      },
      { orderByFacet: 'group' },
    );
    expect(Array.from(g.links)).toEqual([0, 1, 2, 3, 0, 2]);
  });

  it('orderByFacet WITHOUT positions still orders intra-group before inter', () => {
    // The intra-before-inter half of the contract needs no geometry: a budget cutting the
    // tail must drop the inter-group chord, not an intra-group edge, positions or not.
    const g = buildGraph(
      {
        n: 4,
        links: Uint32Array.from([0, 2, 0, 1, 2, 3]),
        facets: { grp: { values: ['a', 'b'], data: Uint8Array.from([0, 0, 1, 1]) } },
      },
      { orderByFacet: 'grp' },
    );
    expect(g.positions).toBeNull();
    // dedup yields (0,1), (0,2), (2,3); ordering demotes the inter link (0,2) to the end
    expect(Array.from(g.links)).toEqual([0, 1, 2, 3, 0, 2]);
  });

  it('positions WITHOUT orderByFacet: length-only ordering through the full pipeline', () => {
    // (0,1) spans 50 units, (1,2) spans 1 — short before long, no facet involved
    const positions = Float32Array.from([0, 0, 0, 50, 0, 0, 51, 0, 0]);
    const g = buildGraph({ n: 3, positions, links: Uint32Array.from([0, 1, 1, 2]) });
    expect(Array.from(g.links)).toEqual([1, 2, 0, 1]);
  });

  it('CSR is built from the ORDERED buffer: edge indices resolve to the links they sit at', () => {
    const n = 40;
    const input = synthInput(n, 120);
    const g = buildGraph(input, { orderByFacet: 'g' });
    for (let i = 0; i < n; i++) {
      const nbrs = neighbours(g, i);
      const edges = incident(g, i);
      expect(edges.length).toBe(nbrs.length);
      for (let k = 0; k < edges.length; k++) {
        const e = edges[k]!;
        const a = g.links[e * 2]!;
        const b = g.links[e * 2 + 1]!;
        // the edge index names a link in the ORDERED buffer whose endpoints are {i, nbr}
        expect([a, b].sort((x, y) => x - y)).toEqual([i, nbrs[k]!].sort((x, y) => x - y));
      }
    }
  });

  it('neighbours() and incident() are subarray views, not copies', () => {
    const g = buildGraph({ n: 3, links: Uint32Array.from([0, 1, 1, 2]) });
    const nb = neighbours(g, 1);
    const ic = incident(g, 1);
    expect(nb.buffer).toBe(g.csr.nbr.buffer); // same backing store — zero allocation of data
    expect(ic.buffer).toBe(g.csr.edge.buffer);
    expect(nb.byteOffset).toBe(g.csr.off[1]! * 4);
    expect(Array.from(nb).sort((a, b) => a - b)).toEqual([0, 2]);
    expect(Array.from(neighbours(g, 0))).toEqual([1]);
    expect(Array.from(neighbours(g, 2))).toEqual([1]);
  });
});

/* A deterministic synthetic input, pure function of (n, m) via hash() — the only sanctioned
 * randomness (DESIGN.md §10.1). Includes self-links and duplicates on purpose. */
function synthInput(n: number, m: number): GraphInput {
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = hash(i, 1) * 100;
    positions[i * 3 + 1] = hash(i, 2) * 100;
    positions[i * 3 + 2] = hash(i, 3) * 100;
  }
  const links = new Uint32Array(m * 2);
  for (let e = 0; e < m; e++) {
    links[e * 2] = Math.floor(hash(e, 10) * n);
    links[e * 2 + 1] = Math.floor(hash(e, 11) * n);
  }
  const g = new Uint8Array(n);
  for (let i = 0; i < n; i++) g[i] = Math.floor(hash(i, 20) * 4);
  return { n, positions, links, facets: { g: { values: ['0', '1', '2', '3'], data: g } } };
}

describe('buildGraph: determinism (golden)', () => {
  it('fixed input -> byte-identical links, CSR and degree, pinned by FNV-1a', () => {
    const a = buildGraph(synthInput(64, 200), { orderByFacet: 'g' });
    const b = buildGraph(synthInput(64, 200), { orderByFacet: 'g' });
    const parts = (g: typeof a) => [
      g.links,
      g.csr.off,
      g.csr.nbr,
      g.csr.edge,
      g.columns['degree']!.data as Uint32Array,
    ];
    const ha = fnv1a(parts(a));
    const hb = fnv1a(parts(b));
    expect(ha).toBe(hb);
    // Golden: pins the exact bytes across refactors — links AND the full CSR (off, nbr,
    // edge) AND the computed degree column, so a permuted scatter order or a changed
    // degree pass cannot hide behind unchanged offsets. If this moves, dedup order, link
    // ranking, CSR layout, or degree computation changed — a breaking change, not noise.
    // Re-recorded for the typed-array dedup (pairs now emitted in (lo, hi)-sorted order,
    // not first-seen) and the widened fingerprint; the prior links+off-only golden was
    // 1724496879.
    expect(ha).toBe(72153371);
  });

  it('does not mutate its inputs: links, positions, facet data unchanged byte-for-byte', () => {
    const input = synthInput(64, 200);
    const u8 = (x: Uint32Array | Float32Array | Uint16Array | Uint8Array) =>
      new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
    const fnvBytes = (bts: Uint8Array) => {
      let h = 0x811c9dc5;
      for (let i = 0; i < bts.length; i++) {
        h = (h ^ bts[i]!) >>> 0;
        h = Math.imul(h, 0x01000193) >>> 0;
      }
      return h >>> 0;
    };
    const facetData = input.facets!['g']!.data;
    const before = [
      fnvBytes(u8(input.links)),
      fnvBytes(u8(input.positions!)),
      fnvBytes(u8(facetData)),
    ];
    buildGraph(input, { orderByFacet: 'g' });
    const after = [
      fnvBytes(u8(input.links)),
      fnvBytes(u8(input.positions!)),
      fnvBytes(u8(facetData)),
    ];
    expect(after).toEqual(before); // §4: stages compose; nothing mutates in place
  });
});
