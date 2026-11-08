/** starscape — graph visualization for very large graphs. Further modules arrive in later
 *  stages; what is exported here is the stage-2 core data model (DESIGN.md §3.1). The synth
 *  corpus generator (`src/synth/`) is test/demo fuel, deliberately not part of this surface. */
export const VERSION = '0.0.0';

export { hash, gauss } from './core/hash';
export { buildCSR } from './core/csr';
export type { CSR } from './core/csr';
export { orderLinks } from './core/linkOrder';
export { buildGraph, neighbours, incident } from './core/graph';
export type { Facet, Graph, GraphInput, NumericColumn } from './core/types';
