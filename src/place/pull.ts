/* "Pull the rest in" (DESIGN.md §3.4's optional `pull`): attach every node to its nearest
 * forced anchor by multi-source BFS over links — ported from the predecessor's subset
 * partition (vault-site-3d arrange.js `bySubset`), minus the member-set restriction (the
 * hierarchy compiler owns scoping; this module answers for one whole graph).
 *
 * Why it exists, in the predecessor's words: forcing a subset onto a shape draws a wire
 * next to a cloud with nothing joining them. Pull each remaining node to the anchor it
 * actually links to and the shape grows blooms — each bloom is literally what that anchor
 * is about, and the shape finally carries information.
 *
 * What is kept from the port, because each line was a lesson:
 *   - HOP COUNT IS REAL INFORMATION. hop 0 is the anchor itself, hop 1 means "directly
 *     about that anchor", and a layout seats the direct links closest in. It is returned,
 *     not discarded.
 *   - THE FRONTIER IS SORTED PER RING ("so traversal cannot depend on adjacency order").
 *     Within one BFS depth, contested nodes go to whichever frontier node claims them
 *     first — so each ring is expanded in ascending node-id order, making ties break by
 *     id, deterministically, whatever order the CSR happened to be built in.
 *   - UNREACHABLE NODES FALL BACK TO THE SPATIALLY NEAREST ANCHOR in the frozen layout —
 *     "the only honest answer available, and it at least keeps the page near the
 *     neighbours it used to have" — with hop = maxHop + 1, one past every real hop, so a
 *     consumer can seat the strays at the rim. On a graph WITHOUT positions there is no
 *     "near", so strays attach to the lowest-id anchor: arbitrary, but deterministic and
 *     documented, which beats arbitrary and quiet.
 *
 * Determinism (§10.1): output is a pure function of (graph, anchor SET). The anchor
 * array is sorted and deduplicated internally, so caller assembly order cannot leak into
 * the result — the classic iteration-order trap, closed at the door.
 *
 * Costs: BFS is O(n + slots). The stray fallback is O(strays × anchors), same as the
 * predecessor — strays are other-component residue and anchors are a chosen few, so the
 * product stays small in practice; a spatial grid can replace it if a corpus proves
 * otherwise.
 */

import type { Graph } from '../core/types';

/** `hops` is a Uint8Array, so recorded hop counts SATURATE here; attachment itself is
 *  exact at any depth (a hop-300 node still reaches its true anchor — only the recorded
 *  count caps). 255 hops is far past anything a layout distinguishes visually. */
export const HOP_MAX = 255;

export interface AnchorAttachment {
  /** Per node: the ANCHOR NODE ID it attached to (anchors attach to themselves). */
  anchorOf: Uint32Array;
  /** Per node: link-hops to its anchor — 0 for the anchor itself, 1 for direct links,
   *  …, `maxHop + 1` for strays placed by the spatial fallback (saturating at HOP_MAX). */
  hops: Uint8Array;
}

/**
 * Attach every node of `graph` to one of `anchors` by multi-source BFS over the CSR,
 * carrying hop counts; nodes no path reaches fall back to the spatially nearest anchor
 * (see the module header for the no-positions rule).
 *
 * Throws on an empty anchor list and on out-of-range anchor ids. Duplicate anchors are
 * tolerated (the anchor SET is what matters).
 */
export function attachToAnchors(graph: Graph, anchors: Uint32Array): AnchorAttachment {
  const n = graph.n;
  if (anchors.length === 0) {
    throw new Error('attachToAnchors: anchors is empty — nothing to attach to');
  }

  /* Sorted, deduplicated seed list: determinism must not depend on how the caller
   * assembled the anchor array. */
  const sorted = anchors.slice();
  sorted.sort();
  let u = 0;
  let prev = -1;
  for (let k = 0; k < sorted.length; k++) {
    const a = sorted[k]!;
    if (a >= n) {
      throw new RangeError(`attachToAnchors: anchor ${a} out of range (n = ${n})`);
    }
    if (a !== prev) {
      sorted[u++] = a;
      prev = a;
    }
  }
  const seeds = sorted.subarray(0, u);

  const anchorOf = new Uint32Array(n);
  const hops = new Uint8Array(n);
  const claimed = new Uint8Array(n);
  const queue = new Uint32Array(n);
  let tail = 0;
  for (let k = 0; k < u; k++) {
    const a = seeds[k]!;
    claimed[a] = 1;
    anchorOf[a] = a; // hop 0, already zero
    queue[tail++] = a;
  }

  /* Ring-by-ring BFS. Claims happen during expansion (first claimant wins), and each
   * freshly claimed ring is sorted ascending before IT expands — the ported determinism
   * rule: within a depth, ties go to the lowest-id frontier node, never to whatever
   * adjacency order the CSR delivered. */
  const off = graph.csr.off;
  const nbr = graph.csr.nbr;
  let head = 0;
  let depth = 0;
  let maxHop = 0;
  while (head < tail) {
    const ringEnd = tail;
    depth++;
    const hop = depth < HOP_MAX ? depth : HOP_MAX;
    for (; head < ringEnd; head++) {
      const i = queue[head]!;
      const e1 = off[i + 1]!;
      for (let e = off[i]!; e < e1; e++) {
        const j = nbr[e]!;
        if (claimed[j] === 0) {
          claimed[j] = 1;
          anchorOf[j] = anchorOf[i]!;
          hops[j] = hop;
          queue[tail++] = j;
        }
      }
    }
    if (tail > ringEnd) {
      maxHop = depth;
      queue.subarray(ringEnd, tail).sort();
    }
  }

  /* Strays: no path to any anchor (another component, or an isolated node). Spatially
   * nearest anchor when the graph carries a frozen layout; the lowest-id anchor when it
   * does not. Ties break to the lower anchor id (seeds ascend; strict `<` keeps the
   * first). Hop is one past every real hop, saturating like the rest. */
  const pos = graph.positions;
  const strayHop = maxHop + 1 < HOP_MAX ? maxHop + 1 : HOP_MAX;
  for (let i = 0; i < n; i++) {
    if (claimed[i] !== 0) continue;
    let best = seeds[0]!;
    if (pos !== null) {
      const io = i * 3;
      const ix = pos[io]!;
      const iy = pos[io + 1]!;
      const iz = pos[io + 2]!;
      let bd = Infinity;
      for (let k = 0; k < u; k++) {
        const a = seeds[k]!;
        const ao = a * 3;
        const dx = ix - pos[ao]!;
        const dy = iy - pos[ao + 1]!;
        const dz = iz - pos[ao + 2]!;
        const d = dx * dx + dy * dy + dz * dz;
        if (d < bd) {
          bd = d;
          best = a;
        }
      }
    }
    anchorOf[i] = best;
    hops[i] = strayHop;
  }

  return { anchorOf, hops };
}
