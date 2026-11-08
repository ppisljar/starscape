/* The Graph builder (DESIGN.md §3.1): validate, dedup directed links into undirected pairs,
 * compute degree if absent, rank links by worth, THEN build the CSR — in that order, because
 * the CSR stores edge indices into the link buffer. Permute links after building adjacency and
 * every `csr.edge[k]` points at somebody else's link, so hovering a node would light a
 * stranger's edges. The predecessor documents this with "ranked BEFORE the adjacency is built".
 *
 * Everything here is a loop over typed arrays: no per-node objects, no spreads over
 * node-length arrays (a spread into a call blows the stack at 10⁶ — measured), no RNG, no
 * clock. Same input, byte-identical output.
 */

import { buildCSR } from './csr';
import { orderLinks } from './linkOrder';
import type { Facet, Graph, GraphInput, NumericColumn } from './types';

/* One shared closure for unlabelled graphs — never a string (or closure) per node. */
function defaultLabel(i: number): string {
  return '#' + i;
}

/** Compile a GraphInput into the shape every downstream module reads.
 *
 *  - Validates link endpoints (< n), facet/column lengths (=== n), position length (=== 3n).
 *  - Dedups directed links into unique undirected pairs, dropping self-links; pairs come out
 *    canonicalised as (lo, hi) in first-seen order.
 *  - Computes a `degree` column over the deduped undirected links unless one was provided.
 *  - Orders links by worth (intra-group before inter, short before long) when positions
 *    exist, grouping by `opts.orderByFacet`; without positions there is no length to rank
 *    by and the deduped order stands.
 *  - Builds the CSR from the ORDERED buffer, so edge indices match what gets drawn. */
export function buildGraph(input: GraphInput, opts?: { orderByFacet?: string }): Graph {
  const n = input.n;
  const positions = input.positions ?? null;
  if (positions && positions.length !== n * 3) {
    throw new Error(`buildGraph: positions has ${positions.length} floats, expected 3n = ${n * 3}`);
  }
  if (input.links.length % 2 !== 0) {
    // 2m pairs by contract; a stray element means the producer truncated — fail loudly rather
    // than silently dropping whichever endpoint fell off (silent truncation lies, §10.4).
    throw new Error(`buildGraph: links has odd length ${input.links.length}, expected 2m pairs`);
  }

  /* ---- validate + dedup in one pass -------------------------------------------------
     The predecessor keyed its dedup set on `a * n + b`, which silently collapses distinct
     pairs once n*n passes 2^53 (n ≈ 9.5e7) — Number can no longer represent every key. We
     use a split key instead: a Map from the low endpoint to the Set of high endpoints. All
     keys stay small exact integers at any n, and no per-link string is ever allocated (the
     other safe option, `${a}:${b}` keys, pays a string per link). The structure is transient
     — dropped before this function returns. */
  const links = input.links;
  const m0 = links.length >> 1;
  const seen = new Map<number, Set<number>>();
  const buf = new Uint32Array(m0 * 2); // upper bound; sliced tight below
  let w = 0;
  for (let e = 0; e < m0; e++) {
    const a = links[e * 2]!;
    const b = links[e * 2 + 1]!;
    if (a >= n) throw new Error(`buildGraph: link ${e} endpoint ${a} out of range (n = ${n})`);
    if (b >= n) throw new Error(`buildGraph: link ${e} endpoint ${b} out of range (n = ${n})`);
    if (a === b) continue; // self-link: no geometry, no adjacency information
    const lo = a < b ? a : b;
    const hi = a < b ? b : a;
    let set = seen.get(lo);
    if (set === undefined) {
      set = new Set<number>();
      seen.set(lo, set);
    }
    if (set.has(hi)) continue; // a->b and b->a are the same physical line
    set.add(hi);
    buf[w] = lo;
    buf[w + 1] = hi;
    w += 2;
  }
  const deduped = w === buf.length ? buf : buf.slice(0, w);
  const m = w >> 1;

  /* ---- columns (degree computed if absent — a loop, never a spread) ------------------ */
  const columns: Record<string, NumericColumn> = {};
  if (input.columns) {
    for (const key of Object.keys(input.columns)) {
      const data = input.columns[key]!;
      if (data.length !== n) {
        throw new Error(`buildGraph: column "${key}" has ${data.length} values, expected n = ${n}`);
      }
      columns[key] = { key, data };
    }
  }
  if (columns['degree'] === undefined) {
    const degree = new Uint32Array(n);
    for (let e = 0; e < m; e++) {
      const a = deduped[e * 2]!;
      const b = deduped[e * 2 + 1]!;
      degree[a] = degree[a]! + 1;
      degree[b] = degree[b]! + 1;
    }
    columns['degree'] = { key: 'degree', data: degree };
  }

  /* ---- facets ------------------------------------------------------------------------ */
  const facets: Record<string, Facet> = {};
  if (input.facets) {
    for (const key of Object.keys(input.facets)) {
      const f = input.facets[key]!;
      if (f.data.length !== n) {
        throw new Error(`buildGraph: facet "${key}" has ${f.data.length} values, expected n = ${n}`);
      }
      facets[key] = { key, values: f.values, data: f.data };
    }
  }

  /* ---- rank, THEN adjacency ----------------------------------------------------------
     orderLinks measures worth partly as length, so without a frozen layout there is nothing
     to rank by and the deduped order stands. With positions, group by the requested facet
     (intra-group links outrank inter-group ones) or by length alone. */
  let group: ArrayLike<number> | null = null;
  if (opts?.orderByFacet !== undefined) {
    const f = facets[opts.orderByFacet];
    if (f === undefined) {
      throw new Error(`buildGraph: orderByFacet "${opts.orderByFacet}" names no facet`);
    }
    group = f.data;
  }
  const ordered = positions ? orderLinks(n, deduped, group, positions) : deduped;
  const csr = buildCSR(n, ordered);

  return {
    n,
    positions,
    links: ordered,
    csr,
    columns,
    facets,
    labelOf: input.labels ?? defaultLabel,
    counts: { nodes: n, links: m },
  };
}

/** Neighbours of node i, as a subarray view — no allocation. */
export function neighbours(g: Graph, i: number): Uint32Array {
  return g.csr.nbr.subarray(g.csr.off[i]!, g.csr.off[i + 1]!);
}

/** Link indices incident to node i (into `g.links`), as a subarray view — no allocation. */
export function incident(g: Graph, i: number): Uint32Array {
  return g.csr.edge.subarray(g.csr.off[i]!, g.csr.off[i + 1]!);
}
