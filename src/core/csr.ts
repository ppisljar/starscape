/* Compressed-sparse-row adjacency over the undirected link set.
 *
 * Two passes, no intermediate arrays-of-arrays: count degrees, prefix-sum into offsets, then
 * scatter. O(V+E) and ~8 bytes per link total, against roughly 130 bytes per link for the
 * arrays-of-pairs shape that the usual graph libraries hand you (DESIGN.md §3.1). Nothing here
 * allocates per node.
 */

export interface CSR {
  /** n+1 offsets into `nbr`/`edge`; node i's slots are [off[i], off[i+1]). */
  off: Uint32Array;
  /** Neighbour node index per slot. */
  nbr: Uint32Array;
  /** Which link index produced each slot, so hovering a node can light its edges too. */
  edge: Uint32Array;
}

/** Build the CSR for `n` nodes over `links` (2m node-index pairs). Each undirected link
 *  appears once in `links` and lands in both endpoints' adjacency. */
export function buildCSR(n: number, links: Uint32Array): CSR {
  const m = links.length >> 1;
  const deg = new Uint32Array(n);
  for (let e = 0; e < m; e++) {
    const a = links[e * 2]!;
    const b = links[e * 2 + 1]!;
    deg[a] = deg[a]! + 1;
    deg[b] = deg[b]! + 1;
  }
  const off = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) off[i + 1] = off[i]! + deg[i]!;
  const total = off[n]!;
  const nbr = new Uint32Array(total);
  const edge = new Uint32Array(total);
  const cur = off.slice(0, n); // write cursors; `off` itself must survive untouched
  for (let e = 0; e < m; e++) {
    const a = links[e * 2]!;
    const b = links[e * 2 + 1]!;
    const ca = cur[a]!;
    nbr[ca] = b;
    edge[ca] = e;
    cur[a] = ca + 1;
    const cb = cur[b]!;
    nbr[cb] = a;
    edge[cb] = e;
    cur[b] = cb + 1;
  }
  return { off, nbr, edge };
}
