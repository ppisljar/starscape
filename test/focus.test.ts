/* Headless tests for the hover/selection focus state (src/field/focus.ts) — the pure index
 * math behind DESIGN.md §6: hover lights the neighbourhood via CSR without repainting the
 * field; selection dims everything outside the focus set; the two states are independent.
 *
 * The repaint-only-the-degree property is asserted literally: writeEdge is instrumented and
 * its call count per hover change must be exactly deg(prev) + deg(next).
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { buildCSR } from '../src/core/csr';
import {
  createFocus,
  BOOST_HOVER,
  BOOST_NEAR,
  BOOST_SELECT,
  DIM_OTHERS,
  FOCUS_REST,
  LINK_DIM_ALPHA,
  LINK_FOCUS_ALPHA,
  LINK_HOVER_ALPHA,
} from '../src/field/focus';
import { LINK_BASE_ALPHA } from '../src/field/constants';

/* Five nodes, four links: a triangle 0-1-2 and a detached pair 3-4.
 *   deg: 0→2, 1→2, 2→2, 3→1, 4→1
 *   edges: e0 = 0-1, e1 = 0-2, e2 = 1-2, e3 = 3-4 */
const N = 5;
const LINKS = Uint32Array.from([0, 1, 0, 2, 1, 2, 3, 4]);
const M = LINKS.length >> 1;

interface Rig {
  focus: ReturnType<typeof createFocus>;
  edgeAlpha: Float32Array;
  writes: number[];
}

function rig(opts: Parameters<typeof createFocus>[4] = {}): Rig {
  const edgeAlpha = new Float32Array(M);
  const writes: number[] = [];
  const focus = createFocus(
    N,
    buildCSR(N, LINKS),
    LINKS,
    (e, alpha) => {
      writes.push(e);
      edgeAlpha[e] = alpha;
    },
    opts,
  );
  focus.repaint(); // seed the edge alphas, as the Field does at boot
  writes.length = 0;
  return { focus, edgeAlpha, writes };
}

let r: Rig;
beforeEach(() => {
  r = rig();
});

describe('resting state', () => {
  it('starts with every node lit at rest and every edge at the base alpha', () => {
    expect(Array.from(r.focus.nodeAlpha)).toEqual([1, 1, 1, 1, 1]);
    expect(Array.from(r.focus.nodeBoost)).toEqual([1, 1, 1, 1, 1]);
    for (let e = 0; e < M; e++) expect(r.edgeAlpha[e]).toBeCloseTo(LINK_BASE_ALPHA, 6);
    expect(r.focus.hover).toBe(-1);
    expect(r.focus.select).toBe(-1);
  });
});

describe('hover: light the degree, repaint the degree', () => {
  it('lifts the node, its CSR neighbours, and its incident edges', () => {
    expect(r.focus.setHover(0)).toBe(true);
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_HOVER, 6);
    expect(r.focus.nodeBoost[1]).toBeCloseTo(BOOST_NEAR, 6);
    expect(r.focus.nodeBoost[2]).toBeCloseTo(BOOST_NEAR, 6);
    expect(r.edgeAlpha[0]).toBeCloseTo(LINK_HOVER_ALPHA, 6); // 0-1
    expect(r.edgeAlpha[1]).toBeCloseTo(LINK_HOVER_ALPHA, 6); // 0-2
    expect(r.edgeAlpha[2]).toBeCloseTo(LINK_BASE_ALPHA, 6); // 1-2: NOT incident to 0
    // hover never dims anything
    expect(Array.from(r.focus.nodeAlpha)).toEqual([1, 1, 1, 1, 1]);
  });

  it('touches nothing outside the hovered degree', () => {
    r.focus.setHover(0);
    expect(r.focus.nodeBoost[3]).toBe(FOCUS_REST);
    expect(r.focus.nodeBoost[4]).toBe(FOCUS_REST);
    expect(r.edgeAlpha[3]).toBeCloseTo(LINK_BASE_ALPHA, 6);
    // the repaint-only-the-degree property, literally: deg(0) = 2 edge writes, no more
    expect(r.writes).toEqual([0, 1]);
  });

  it('moving the hover writes exactly deg(prev) + deg(next) edges', () => {
    r.focus.setHover(0);
    r.writes.length = 0;
    r.focus.setHover(3);
    // drop 0's two incident edges, lift 3's one
    expect(r.writes.length).toBe(2 + 1);
    expect(r.focus.nodeBoost[0]).toBe(FOCUS_REST);
    expect(r.focus.nodeBoost[1]).toBe(FOCUS_REST);
    expect(r.focus.nodeBoost[2]).toBe(FOCUS_REST);
    expect(r.focus.nodeBoost[3]).toBeCloseTo(BOOST_HOVER, 6);
    expect(r.focus.nodeBoost[4]).toBeCloseTo(BOOST_NEAR, 6);
    expect(r.edgeAlpha[0]).toBeCloseTo(LINK_BASE_ALPHA, 6);
    expect(r.edgeAlpha[3]).toBeCloseTo(LINK_HOVER_ALPHA, 6);
  });

  it('a hover between adjacent nodes keeps the shared lift correct', () => {
    r.focus.setHover(1);
    r.focus.setHover(2); // 1 and 2 are neighbours
    expect(r.focus.nodeBoost[2]).toBeCloseTo(BOOST_HOVER, 6);
    expect(r.focus.nodeBoost[1]).toBeCloseTo(BOOST_NEAR, 6); // demoted from hover to neighbour
    expect(r.edgeAlpha[2]).toBeCloseTo(LINK_HOVER_ALPHA, 6); // 1-2 incident to both
  });

  it('clearing the hover restores rest exactly', () => {
    r.focus.setHover(0);
    expect(r.focus.setHover(-1)).toBe(true);
    expect(Array.from(r.focus.nodeBoost)).toEqual([1, 1, 1, 1, 1]);
    for (let e = 0; e < M; e++) expect(r.edgeAlpha[e]).toBeCloseTo(LINK_BASE_ALPHA, 6);
  });

  it('re-hovering the same node is a no-op and says so', () => {
    r.focus.setHover(2);
    r.writes.length = 0;
    expect(r.focus.setHover(2)).toBe(false);
    expect(r.writes.length).toBe(0);
  });
});

describe('selection: dim the rest of the field', () => {
  it('keeps the node and its neighbourhood lit, dims everything else to dimOthers', () => {
    expect(r.focus.setSelect(0)).toBe(true);
    expect(r.focus.nodeAlpha[0]).toBe(FOCUS_REST);
    expect(r.focus.nodeAlpha[1]).toBe(FOCUS_REST);
    expect(r.focus.nodeAlpha[2]).toBe(FOCUS_REST);
    expect(r.focus.nodeAlpha[3]).toBeCloseTo(DIM_OTHERS, 6);
    expect(r.focus.nodeAlpha[4]).toBeCloseTo(DIM_OTHERS, 6);
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_SELECT, 6);
  });

  it('link alphas follow the focus set: inside bright, outside present-but-dim', () => {
    r.focus.setSelect(0);
    expect(r.edgeAlpha[0]).toBeCloseTo(LINK_FOCUS_ALPHA, 6); // 0-1: both in the set
    expect(r.edgeAlpha[1]).toBeCloseTo(LINK_FOCUS_ALPHA, 6); // 0-2
    expect(r.edgeAlpha[2]).toBeCloseTo(LINK_FOCUS_ALPHA, 6); // 1-2: both neighbours
    expect(r.edgeAlpha[3]).toBeCloseTo(LINK_DIM_ALPHA, 6); // 3-4: outside, dimmed not gone
  });

  it('moving the selection rebuilds the set correctly', () => {
    r.focus.setSelect(0);
    r.focus.setSelect(3);
    expect(r.focus.nodeAlpha[0]).toBeCloseTo(DIM_OTHERS, 6);
    expect(r.focus.nodeAlpha[1]).toBeCloseTo(DIM_OTHERS, 6);
    expect(r.focus.nodeAlpha[3]).toBe(FOCUS_REST);
    expect(r.focus.nodeAlpha[4]).toBe(FOCUS_REST);
    expect(r.focus.nodeBoost[3]).toBeCloseTo(BOOST_SELECT, 6);
    expect(r.focus.nodeBoost[0]).toBe(FOCUS_REST);
    expect(r.edgeAlpha[0]).toBeCloseTo(LINK_DIM_ALPHA, 6);
    expect(r.edgeAlpha[3]).toBeCloseTo(LINK_FOCUS_ALPHA, 6);
  });

  it('Escape (select -1) restores the resting field byte-exactly', () => {
    r.focus.setSelect(0);
    expect(r.focus.setSelect(-1)).toBe(true);
    expect(Array.from(r.focus.nodeAlpha)).toEqual([1, 1, 1, 1, 1]);
    expect(Array.from(r.focus.nodeBoost)).toEqual([1, 1, 1, 1, 1]);
    for (let e = 0; e < M; e++) expect(r.edgeAlpha[e]).toBeCloseTo(LINK_BASE_ALPHA, 6);
  });

  it('re-selecting the same node is a no-op and says so', () => {
    r.focus.setSelect(1);
    r.writes.length = 0;
    expect(r.focus.setSelect(1)).toBe(false);
    expect(r.writes.length).toBe(0);
  });

  it('honours a custom dimOthers level', () => {
    const custom = rig({ dimOthers: 0.5 });
    custom.focus.setSelect(0);
    expect(custom.focus.nodeAlpha[3]).toBeCloseTo(0.5, 6);
  });
});

describe('hover and selection are independent states (DESIGN.md §6)', () => {
  it('hovering elsewhere overlays the dimmed field without disturbing the selection', () => {
    r.focus.setSelect(0);
    r.focus.setHover(3);
    // the overlay lifts 3's neighbourhood ON TOP of the dim
    expect(r.focus.nodeBoost[3]).toBeCloseTo(BOOST_HOVER, 6);
    expect(r.focus.nodeBoost[4]).toBeCloseTo(BOOST_NEAR, 6);
    expect(r.edgeAlpha[3]).toBeCloseTo(LINK_HOVER_ALPHA, 6); // brighter than its dim base
    // the selection is untouched: still emphasised, still the same alphas
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_SELECT, 6);
    expect(r.focus.nodeAlpha[3]).toBeCloseTo(DIM_OTHERS, 6); // hover never un-dims
    // clearing the hover returns to the SELECTION state, not to rest
    r.focus.setHover(-1);
    expect(r.focus.nodeBoost[3]).toBe(FOCUS_REST);
    expect(r.edgeAlpha[3]).toBeCloseTo(LINK_DIM_ALPHA, 6);
    expect(r.focus.nodeAlpha[0]).toBe(FOCUS_REST);
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_SELECT, 6);
  });

  it('hovering the selected node: hover wins on the glyph, selection returns on leave', () => {
    r.focus.setSelect(0);
    r.focus.setHover(0);
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_HOVER, 6);
    r.focus.setHover(-1);
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_SELECT, 6);
  });

  it('hovering a neighbour of the selected node never demotes the selection emphasis', () => {
    r.focus.setSelect(0);
    r.focus.setHover(1); // 0 is a neighbour of 1
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_SELECT, 6); // not flattened to BOOST_NEAR
    r.focus.setHover(-1);
    expect(r.focus.nodeBoost[0]).toBeCloseTo(BOOST_SELECT, 6);
  });

  it('selecting while hovered keeps the hover overlay live', () => {
    r.focus.setHover(3);
    r.focus.setSelect(0);
    expect(r.focus.nodeBoost[3]).toBeCloseTo(BOOST_HOVER, 6); // sweep re-lifted the hover
    expect(r.focus.nodeBoost[4]).toBeCloseTo(BOOST_NEAR, 6);
    expect(r.edgeAlpha[3]).toBeCloseTo(LINK_HOVER_ALPHA, 6);
    expect(r.focus.hover).toBe(3);
    expect(r.focus.select).toBe(0);
  });
});

describe('determinism', () => {
  it('an identical event sequence reproduces identical buffers', () => {
    const a = rig();
    const b = rig();
    for (const f of [a, b]) {
      f.focus.setHover(0);
      f.focus.setSelect(2);
      f.focus.setHover(4);
      f.focus.setHover(1);
      f.focus.setSelect(-1);
    }
    expect(a.focus.nodeAlpha).toEqual(b.focus.nodeAlpha);
    expect(a.focus.nodeBoost).toEqual(b.focus.nodeBoost);
    expect(a.edgeAlpha).toEqual(b.edgeAlpha);
  });
});
