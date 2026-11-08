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
 *  - Validates link endpoints (< n), facet/column lengths (=== n), facet codes (< the
 *    facet's values table), position length (=== 3n).
 *  - Dedups directed links into unique undirected pairs, dropping self-links; pairs come out
 *    canonicalised as (lo, hi) in ascending (lo, hi) order.
 *  - Computes a `degree` column over the deduped undirected links unless one was provided.
 *  - Orders links by worth (intra-group before inter, short before long) when positions
 *    or a group facet exist, grouping by `opts.orderByFacet`; without positions every
 *    length ranks 0, so ordering degenerates to intra-before-inter. With neither positions
 *    nor a facet the deduped order stands.
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

  /* ---- validate + dedup, typed arrays only -------------------------------------------
     NOT a Map<lo, Set<hi>>: that shape allocates up to n JS Set objects plus one entry per
     link — measured at n=1e6, m=4e6 it alone cost 1304 ms and ~263 MB of transient heap,
     a large bite out of the §9 boot budget and a literal §10.2 violation (no per-node JS
     objects). Instead: canonicalise each pair to (lo, hi), counting-sort the pairs by lo
     into CSR-style buckets (two linear passes over two Uint32Arrays), sort each bucket's
     hi values in place (buckets are small — avg degree), then emit unique pairs in one
     linear pass. No per-node or per-link JS object anywhere.

     The predecessor keyed its dedup set on `a * n + b`, which silently collapses distinct
     pairs once n*n passes 2^53 (n ≈ 9.5e7); lo/hi stay exact at any n. NOTE: pairs come out
     sorted by (lo, hi), not in first-seen order — the order is deterministic either way. */
  const links = input.links;
  const m0 = links.length >> 1;
  const start = new Uint32Array(n + 1); // start[lo + 1] counts pairs whose low end is lo
  for (let e = 0; e < m0; e++) {
    const a = links[e * 2]!;
    const b = links[e * 2 + 1]!;
    if (a >= n) throw new Error(`buildGraph: link ${e} endpoint ${a} out of range (n = ${n})`);
    if (b >= n) throw new Error(`buildGraph: link ${e} endpoint ${b} out of range (n = ${n})`);
    if (a === b) continue; // self-link: no geometry, no adjacency information
    const lo = a < b ? a : b;
    start[lo + 1] = start[lo + 1]! + 1;
  }
  for (let i = 0; i < n; i++) start[i + 1] = start[i + 1]! + start[i]!; // prefix sum: bucket starts
  const total = n > 0 ? start[n]! : 0;
  const his = new Uint32Array(total); // hi endpoints, bucketed by lo
  const cur = start.slice(0, n); // write cursors; `start` itself must survive
  for (let e = 0; e < m0; e++) {
    const a = links[e * 2]!;
    const b = links[e * 2 + 1]!;
    if (a === b) continue;
    const lo = a < b ? a : b;
    const c = cur[lo]!;
    his[c] = a < b ? b : a;
    cur[lo] = c + 1;
  }
  const buf = new Uint32Array(total * 2); // upper bound; sliced tight below
  let w = 0;
  for (let lo = 0; lo < n; lo++) {
    const s = start[lo]!;
    const t = start[lo + 1]!;
    if (t - s > 1) his.subarray(s, t).sort(); // numeric ascending; in place, bucket-local
    let prev = -1;
    for (let k = s; k < t; k++) {
      const hi = his[k]!;
      if (hi === prev) continue; // a->b and b->a (and repeats) are the same physical line
      prev = hi;
      buf[w] = lo;
      buf[w + 1] = hi;
      w += 2;
    }
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
      // every code must index into the values table — an out-of-range code would surface
      // later as `values[data[i]] === undefined` in whatever channel binds the facet:
      // an invisible category instead of a boot-time error (silent, not loud).
      const tableSize = f.values.length;
      for (let i = 0; i < n; i++) {
        if (f.data[i]! >= tableSize) {
          throw new Error(
            `buildGraph: facet "${key}" code ${f.data[i]} at node ${i} is out of range ` +
              `(values table has ${tableSize} entries)`,
          );
        }
      }
      facets[key] = { key, values: f.values, data: f.data };
    }
  }

  /* ---- rank, THEN adjacency ----------------------------------------------------------
     orderLinks ranks by the inter bit and by length. With positions, group by the requested
     facet (intra-group links outrank inter-group ones) or by length alone. Without positions
     the intra-before-inter half of the contract still applies — a facet alone orders the
     buffer (every length ranks 0). Only with neither does the deduped order stand. */
  let group: ArrayLike<number> | null = null;
  if (opts?.orderByFacet !== undefined) {
    const f = facets[opts.orderByFacet];
    if (f === undefined) {
      throw new Error(`buildGraph: orderByFacet "${opts.orderByFacet}" names no facet`);
    }
    group = f.data;
  }
  const ordered = positions || group ? orderLinks(n, deduped, group, positions ?? null) : deduped;
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
