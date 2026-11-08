/* Hover / selection focus state — the pure index math behind "light the neighbourhood without
 * repainting the field" (DESIGN.md §6), ported from vault-site-3d src/field.js (setHover /
 * applyNodes / applyLinks).
 *
 * Two INDEPENDENT states, which is the model the predecessor's viewers settled on:
 *
 *   SELECTION dims the field to a focus set (the node + its CSR neighbourhood); everything
 *     else recedes to `dimOthers`. It changes on a click, so its repaint is a full sweep —
 *     O(V+E) on an event, never per frame.
 *
 *   HOVER is an overlay. It never dims anything; it lifts the hovered node, its neighbours
 *     and its own edges ON TOP of whatever the selection is doing — and it is applied
 *     incrementally over the hovered node's DEGREE (the predecessor's hard rule: hover changes
 *     on pointermove, and resweeping all nodes and links on pointermove is how a viewer dies
 *     at 10⁵⁺). setHover touches exactly deg(prev) + deg(next) + 2 node slots and
 *     deg(prev) + deg(next) edge slots, nothing else — never the whole colour buffer.
 *
 * This module owns no GPU objects: it writes node alpha/boost into typed arrays it hands the
 * caller and emits edge alphas through a caller-supplied writer, so the whole state machine
 * runs headless under vitest. The Field wires the arrays to instanced attributes and the
 * writer to the link colour buffer. House rules: the focus set is a Uint8Array mask, not a
 * Set of boxed indices (§10.2); every sweep is a loop (§10.3); no randomness, no clock.
 */

import type { CSR } from '../core/csr';
import { LINK_BASE_ALPHA } from './constants';

/** Size emphasis on the selected node (the vertex stage scales the glyph by boost). */
export const BOOST_SELECT = 1.75;
/** Size emphasis on the hovered node. */
export const BOOST_HOVER = 1.45;
/** Size emphasis on the hovered node's neighbours — enough to see the neighbourhood light
 *  up, not enough to read as a second hover. */
export const BOOST_NEAR = 1.18;
/** Resting boost / alpha: 1 everywhere, nothing emphasised, nothing dimmed. */
export const FOCUS_REST = 1.0;

/** Alpha the rest of the field recedes to behind a selection. The predecessor's default,
 *  argued up from 0.085 — which was a blackout: the context you selected the page FROM
 *  disappeared, defeating the point of selecting it inside a map. */
export const DIM_OTHERS = 0.22;
/** Alpha of a link under the cursor (the hover overlay on its incident edges). */
export const LINK_HOVER_ALPHA = 0.85;
/** Alpha of a link joining two focus-set members while a selection is active. */
export const LINK_FOCUS_ALPHA = 0.7;
/** Alpha of every other link while a selection is active — present, not gone. */
export const LINK_DIM_ALPHA = 0.012;

export interface FocusOpts {
  /** Resting link alpha (no selection active). Default LINK_BASE_ALPHA. */
  linkBase?: number;
  /** See DIM_OTHERS. */
  dimOthers?: number;
  /** See LINK_HOVER_ALPHA. */
  linkHover?: number;
  /** See LINK_FOCUS_ALPHA. */
  linkFocus?: number;
  /** See LINK_DIM_ALPHA. */
  linkDim?: number;
}

export interface Focus {
  /** Hovered node index, -1 for none. */
  readonly hover: number;
  /** Selected node index, -1 for none. */
  readonly select: number;
  /** Per-node alpha (n): 1 lit, `dimOthers` dimmed behind a selection. Owned here, consumed
   *  by the node layer as the iAlpha instanced attribute. */
  readonly nodeAlpha: Float32Array;
  /** Per-node size boost (n): 1 resting, BOOST_* emphasised. The iBoost attribute. */
  readonly nodeBoost: Float32Array;
  /** Move the hover. O(deg(prev) + deg(next)). Returns false when nothing changed. */
  setHover(next: number): boolean;
  /** Move the selection (full sweep — runs on a click, never per frame). -1 clears,
   *  restoring every alpha to rest. Returns false when nothing changed. */
  setSelect(next: number): boolean;
  /** Re-emit every node alpha/boost and every edge alpha from the current state — for after
   *  something external rewrote the buffers underneath (e.g. a facet recolour). */
  repaint(): void;
  /** Edge e's alpha under the selection state alone (hover overlay excluded). */
  baseEdgeAlpha(e: number): number;
}

/**
 * Create the focus state over a compiled graph's adjacency.
 *
 * @param n          node count
 * @param csr        adjacency (off/nbr/edge) — `edge[k]` indexes the ordered link buffer
 * @param links      the ordered 2m link-endpoint buffer `csr.edge` agrees with
 * @param writeEdge  sink for edge alphas — the Field writes them into the link colour
 *                   buffer's two alpha slots; tests write them into a plain Float32Array
 */
export function createFocus(
  n: number,
  csr: CSR,
  links: Uint32Array,
  writeEdge: (e: number, alpha: number) => void,
  opts: FocusOpts = {},
): Focus {
  const linkBase = opts.linkBase ?? LINK_BASE_ALPHA;
  const dimOthers = opts.dimOthers ?? DIM_OTHERS;
  const linkHover = opts.linkHover ?? LINK_HOVER_ALPHA;
  const linkFocus = opts.linkFocus ?? LINK_FOCUS_ALPHA;
  const linkDim = opts.linkDim ?? LINK_DIM_ALPHA;
  const linkCount = links.length >> 1;

  const nodeAlpha = new Float32Array(n).fill(FOCUS_REST);
  const nodeBoost = new Float32Array(n).fill(FOCUS_REST);
  /** 1 for the selected node and its neighbours, else 0; meaningful only while hasFocus. */
  const mask = new Uint8Array(n);
  let hover = -1;
  let select = -1;
  let hasFocus = false;

  const baseBoost = (i: number): number => (i === select ? BOOST_SELECT : FOCUS_REST);

  function baseEdgeAlpha(e: number): number {
    if (!hasFocus) return linkBase;
    return mask[links[e * 2]!]! !== 0 && mask[links[e * 2 + 1]!]! !== 0 ? linkFocus : linkDim;
  }

  /** Lift (on) or drop (off) one node's hover overlay: the node, its CSR neighbours, its
   *  incident edges. O(degree) — the ported repaint-only-the-degree mechanism. Dropping
   *  resets neighbours to their base boost; when the old and new hover share neighbours the
   *  caller lifts the new overlay AFTERWARDS, so shared ones come straight back up. */
  function touch(i: number, on: boolean): void {
    if (i < 0) return;
    nodeBoost[i] = on ? BOOST_HOVER : baseBoost(i);
    for (let k = csr.off[i]!; k < csr.off[i + 1]!; k++) {
      const j = csr.nbr[k]!;
      const e = csr.edge[k]!;
      nodeBoost[j] = on ? Math.max(nodeBoost[j]!, BOOST_NEAR) : baseBoost(j);
      writeEdge(e, on ? Math.max(baseEdgeAlpha(e), linkHover) : baseEdgeAlpha(e));
    }
  }

  function setHover(next: number): boolean {
    if (next === hover) return false;
    touch(hover, false);
    hover = next;
    touch(next, true);
    // The selected node keeps its emphasis whatever the hover does around it (it may have
    // just been flattened as a neighbour of the previous hover).
    if (select >= 0) nodeBoost[select] = select === hover ? BOOST_HOVER : BOOST_SELECT;
    return true;
  }

  /** Full-sweep node pass: selection dim + the current hover overlay re-lifted on top. */
  function applyNodes(): void {
    for (let i = 0; i < n; i++) {
      nodeAlpha[i] = hasFocus ? (mask[i] !== 0 ? FOCUS_REST : dimOthers) : FOCUS_REST;
      nodeBoost[i] = i === hover ? BOOST_HOVER : baseBoost(i);
    }
    if (hover >= 0) {
      for (let k = csr.off[hover]!; k < csr.off[hover + 1]!; k++) {
        const j = csr.nbr[k]!;
        if (j !== hover) nodeBoost[j] = Math.max(nodeBoost[j]!, BOOST_NEAR);
      }
      if (select >= 0 && select !== hover) nodeBoost[select] = BOOST_SELECT;
    }
  }

  /** Full-sweep link pass: selection alphas, then the hover overlay on top. */
  function applyLinks(): void {
    for (let e = 0; e < linkCount; e++) writeEdge(e, baseEdgeAlpha(e));
    if (hover >= 0) {
      for (let k = csr.off[hover]!; k < csr.off[hover + 1]!; k++) {
        writeEdge(csr.edge[k]!, Math.max(baseEdgeAlpha(csr.edge[k]!), linkHover));
      }
    }
  }

  function setSelect(next: number): boolean {
    if (next === select) return false;
    // Rebuild the focus mask incrementally: clear the old neighbourhood, set the new —
    // O(deg(old) + deg(new)), so the mask never needs an O(n) clear on a click.
    if (select >= 0) {
      mask[select] = 0;
      for (let k = csr.off[select]!; k < csr.off[select + 1]!; k++) mask[csr.nbr[k]!] = 0;
    }
    select = next;
    hasFocus = next >= 0;
    if (next >= 0) {
      mask[next] = 1;
      for (let k = csr.off[next]!; k < csr.off[next + 1]!; k++) mask[csr.nbr[k]!] = 1;
    }
    applyNodes();
    applyLinks();
    return true;
  }

  return {
    get hover() {
      return hover;
    },
    get select() {
      return select;
    },
    nodeAlpha,
    nodeBoost,
    setHover,
    setSelect,
    repaint(): void {
      applyNodes();
      applyLinks();
    },
    baseEdgeAlpha,
  };
}
