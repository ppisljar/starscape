/* Core data shapes (DESIGN.md §3.1): columns, never objects. A Graph is flat typed arrays
 * indexed by node id — ~8 bytes per link, not the ~130 of object-shaped graph libraries —
 * plus lazy callbacks for anything stringy. Nothing in here holds a per-node JS object,
 * closure, or eager string (§10.2).
 */

import type { CSR } from './csr';

/** A categorical column a visual channel can bind to: `values[data[i]]` names node i's
 *  category. The dictionary is per-facet, so the per-node storage is one small int. */
export interface Facet {
  key: string;
  values: string[];
  data: Uint8Array | Uint16Array | Uint32Array;
}

/** A numeric column (degree, words, rank, …), one value per node. */
export interface NumericColumn {
  key: string;
  data: Float32Array | Uint32Array;
}

/** What `buildGraph` consumes. Links may be directed, duplicated, and self-looped — the
 *  builder canonicalises them. Positions are an optional frozen layout (3n floats). */
export interface GraphInput {
  n: number;
  positions?: Float32Array;
  /** 2m node-index pairs, as produced (directed, possibly with duplicates/self-links). */
  links: Uint32Array;
  columns?: Record<string, Float32Array | Uint32Array>;
  facets?: Record<string, { values: string[]; data: Uint8Array | Uint16Array | Uint32Array }>;
  /** Lazy label source — called on demand, never materialised into 10⁶ eager strings. */
  labels?: (i: number) => string;
}

/** The compiled graph every downstream module reads. `links` is the deduped, undirected,
 *  importance-ordered buffer; `csr.edge` indices agree with it (rank before adjacency).
 *  Nothing downstream may assume more than `n` nodes exist (DESIGN.md §12) — chunked and
 *  procedural sources present this same shape per chunk. */
export interface Graph {
  n: number;
  positions: Float32Array | null;
  links: Uint32Array;
  csr: CSR;
  columns: Record<string, NumericColumn>;
  facets: Record<string, Facet>;
  labelOf: (i: number) => string;
  counts: { nodes: number; links: number };
}
