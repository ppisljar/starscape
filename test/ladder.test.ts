/* Headless tests for the zoom-ladder state machine (src/ladder/*, DESIGN.md §6) — the ported
 * logic of vault-site-3d src/zoom.js and its README's invariants:
 *   - one unclamped projected-size number drives everything; thresholds exact at the boundary;
 *   - TWO progress variables: glyph/named keys off the node's own apparent size, cards key off
 *     a FIXED world radius via distance-in-spacings;
 *   - label admission by rank band (earn 20 / keep 32), mount-immediate with a dt-accumulated
 *     minimum lifetime (the port's `since` + labelDwell), published in rank order;
 *   - card budget 5, best-ranked win; a pin bypasses budget, open threshold and candidate
 *     filter — pinning a non-candidate still pins;
 *   - attention follows the cursor; Escape retreats and suspends auto-open until real user
 *     intent (cursor move or pin) — never because another page drifted into range;
 *   - flying backwards runs the tiers in reverse frame-for-frame; fully deterministic, no clock.
 */

import { describe, expect, it } from 'vitest';
import { glyphPx } from '../src/field/channels';
import { DEFAULT_FOV, FALLOFF } from '../src/field/constants';
import * as C from '../src/ladder/constants';
import {
  cardGrowT,
  cardPx,
  cardRadiusOf,
  cardStandoff,
  createLadder,
  fovPxOf,
  projectedPx,
  standoffOf,
} from '../src/ladder/ladder';
import type { Ladder, LadderState, LadderView, TierEvents } from '../src/ladder/types';

const INF = Number.POSITIVE_INFINITY;

interface Row {
  node: number;
  px: number;
  dist: number;
  cur?: number;
}

function viewOf(
  rows: Row[],
  over: { spacing?: number; pinned?: number; cursorMoved?: boolean } = {},
): LadderView {
  const node: number[] = [];
  const px: number[] = [];
  const dist: number[] = [];
  const cur: number[] = [];
  for (const r of rows) {
    node.push(r.node);
    px.push(r.px);
    dist.push(r.dist);
    cur.push(r.cur ?? INF);
  }
  return {
    candidates: { count: rows.length, node, projectedPx: px, distance: dist, cursorPx: cur },
    spacing: over.spacing ?? 100,
    pinned: over.pinned ?? -1,
    cursorMoved: over.cursorMoved ?? false,
  };
}

/** Candidates whose rank equals their position (descending px, no cursor, far away). */
function ranked(nodesInRankOrder: number[]): LadderView {
  return viewOf(nodesInRankOrder.map((node, k) => ({ node, px: 100 - k, dist: 10_000 })));
}

function namedOf(st: LadderState): number[] {
  return Array.from(st.named.subarray(0, st.namedCount));
}
function cardNodes(st: LadderState): number[] {
  return st.cards.map((c) => c.node);
}
function evList(ev: TierEvents): { enter: number[]; exit: number[] } {
  return {
    enter: Array.from(ev.enter.subarray(0, ev.enterCount)),
    exit: Array.from(ev.exit.subarray(0, ev.exitCount)),
  };
}
function snap(st: LadderState): unknown {
  return {
    named: namedOf(st),
    cards: st.cards.map((c) => ({ node: c.node, tier: c.tier, growT: c.growT })),
    open: st.open,
    suspended: st.suspended,
  };
}

// The spacing used throughout, and the distance thresholds computed EXACTLY as the ladder
// computes them — boundary tests feed these very values.
const SP = 100;
const D_CARD = C.CARD_SPACINGS * SP;
const D_LOCK = C.LOCK_SPACINGS * SP;
const D_OPEN = C.OPEN_SPACINGS * SP;

describe('constants', () => {
  it('exports the contract numbers', () => {
    expect(C.NAMED_PX).toBe(13);
    expect(C.EARN).toBe(20);
    expect(C.KEEP).toBe(32);
    expect(C.DWELL_MS).toBe(600);
    expect(C.NAMED_CAP).toBe(34);
    expect(C.CARD_SPACINGS).toBe(2.2);
    expect(C.LOCK_SPACINGS).toBe(1.0);
    expect(C.OPEN_SPACINGS).toBe(0.45);
    expect(C.CARDS).toBe(5);
    expect(C.OPEN_BUDGET).toBe(1);
    expect(C.OPEN_HOLD_MS).toBe(600);
    expect(C.LOCK_W).toBe(440);
    expect(C.LOCK_H).toBe(320);
  });

  it('derives the predecessor CARD_R from spacing: 0.3125 × 192 = 60', () => {
    expect(C.CARD_R_SPACINGS).toBe(0.3125);
    expect(cardRadiusOf(192)).toBe(60);
  });

  it('is overridable via options, resolved on the instance', () => {
    const lad = createLadder({ cards: 3, lockW: 500 });
    expect(lad.config.cards).toBe(3);
    expect(lad.config.lockW).toBe(500);
    expect(lad.config.lockH).toBe(C.LOCK_H);
    expect(lad.config.keep).toBe(C.KEEP);
  });
});

describe('the projection helpers mirror the field shader', () => {
  it('projectedPx equals 2 × glyphPx at the exported FALLOFF (= 1, the true projection)', () => {
    expect(FALLOFF).toBe(1);
    const h = 900;
    const F = fovPxOf(h, DEFAULT_FOV);
    const halfH = h / 2;
    const tanHalf = Math.tan((DEFAULT_FOV * Math.PI) / 360);
    const cases: Array<[number, number, number]> = [
      [5, 120, 400],
      [22, 37.5, 999],
      [1.8, 2000, 10],
    ];
    for (const [r, d, dRef] of cases) {
      expect(projectedPx(r, d, F)).toBeCloseTo(2 * glyphPx(r, d, halfH, tanHalf, dRef, FALLOFF), 9);
    }
  });

  it('standoffOf inverts projectedPx; cardPx / cardStandoff agree through the fixed radius', () => {
    const F = fovPxOf(800);
    expect(projectedPx(7, standoffOf(7, 54, F), F)).toBeCloseTo(54, 9);
    expect(cardPx(192, cardStandoff(192, 320, F), F)).toBeCloseTo(320, 9);
  });

  it('cardGrowT is 0 at the card distance, 1 at the lock, linear in projected size', () => {
    expect(cardGrowT(D_CARD, D_CARD, D_LOCK)).toBe(0);
    expect(cardGrowT(D_LOCK, D_CARD, D_LOCK)).toBe(1);
    expect(cardGrowT(D_LOCK / 2, D_CARD, D_LOCK)).toBe(1); // clamped past the lock
    // halfway in 1/d space, not in distance
    const dMid = 2 / (1 / D_CARD + 1 / D_LOCK);
    expect(cardGrowT(dMid, D_CARD, D_LOCK)).toBeCloseTo(0.5, 12);
  });
});

describe('tier thresholds, exact at the boundary', () => {
  const opts = { dwellMs: 0, openHoldMs: 0 };

  it('named: a glyph of exactly NAMED_PX ranks, a hair under never does', () => {
    const lad = createLadder(opts);
    let st = lad.update(0.016, viewOf([{ node: 1, px: C.NAMED_PX - 1e-9, dist: 500 }]));
    expect(st.namedCount).toBe(0);
    st = lad.update(0.016, viewOf([{ node: 1, px: C.NAMED_PX, dist: 500 }]));
    expect(namedOf(st)).toEqual([1]);
    expect(evList(st.events.named).enter).toEqual([1]);
  });

  it('card fires at exactly CARD_SPACINGS × spacing, not a hair above', () => {
    const lad = createLadder(opts);
    let st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: D_CARD + 0.001 }]));
    expect(st.cards.length).toBe(0);
    st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: D_CARD }]));
    expect(st.cards.length).toBe(1);
    expect(st.cards[0]).toMatchObject({ node: 1, tier: 'card', growT: 0 });
    expect(evList(st.events.card).enter).toEqual([1]);
  });

  it('lock fires at exactly LOCK_SPACINGS × spacing with growT 1', () => {
    const lad = createLadder(opts);
    let st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: D_LOCK + 0.001 }]));
    expect(st.cards[0]!.tier).toBe('card');
    expect(st.cards[0]!.growT).toBeLessThan(1);
    st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: D_LOCK }]));
    expect(st.cards[0]).toMatchObject({ node: 1, tier: 'lock', growT: 1 });
    expect(evList(st.events.lock).enter).toEqual([1]);
  });

  it('open fires at exactly OPEN_SPACINGS × spacing (an empty slot opens immediately)', () => {
    const lad = createLadder(opts);
    let st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: D_OPEN + 0.001 }]));
    expect(st.open).toBe(-1);
    st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: D_OPEN }]));
    expect(st.open).toBe(1);
    expect(st.cards[0]).toMatchObject({ node: 1, tier: 'open', growT: 1 });
    expect(evList(st.events.open).enter).toEqual([1]);
  });
});

describe('named admission: rank band, mount-immediate, minimum lifetime', () => {
  // a small band keeps the corpora readable: earn inside top 2, keep until outside top 4
  const band = { earn: 2, keep: 4 };

  it('a label mounts the TICK its node enters the earn band (the port: names appear the frame glyphs cross the gate)', () => {
    const lad = createLadder(band);
    const st = lad.update(0.016, ranked([7, 8, 9, 10]));
    expect(namedOf(st)).toEqual([7, 8]); // the top-2 earn immediately, rank 2 does not
    expect(evList(st.events.named).enter).toEqual([7, 8]);
  });

  it('a label younger than DWELL_MS cannot drop, however far it falls — the minimum lifetime', () => {
    const lad = createLadder(band);
    lad.update(0.2, ranked([7, 8])); // 7 mounts, age 0
    const gone = ranked([1, 2, 3, 4, 5]); // 7 not even a candidate any more
    let st = lad.update(0.2, gone);
    expect(namedOf(st)).toContain(7); // age 200 ms — held by the grace
    st = lad.update(0.2, gone);
    expect(namedOf(st)).toContain(7); // age 400 ms — still held
    st = lad.update(0.2, gone);
    expect(namedOf(st)).not.toContain(7); // age 600 ms ≥ DWELL — drops
    expect(evList(st.events.named).exit).toContain(7);
  });

  it('past the grace a label drops the tick it leaves the keep band — the EARN < KEEP band is the hysteresis', () => {
    const lad = createLadder(band);
    lad.update(0.6, ranked([7, 8])); // mounts
    lad.update(0.6, ranked([7, 8])); // ages past the grace
    let st = lad.update(0.016, ranked([7, 1, 2, 3])); // inside keep: stays, indefinitely
    expect(namedOf(st)).toContain(7);
    st = lad.update(0.016, ranked([1, 2, 3, 4, 7])); // rank 4: outside keep (top 4 = 0..3)
    expect(namedOf(st)).not.toContain(7); // one tick outside — drops at once
    expect(evList(st.events.named).exit).toContain(7);
  });

  it('publishes the live set in RANK order — a later, better-ranked label goes first', () => {
    const lad = createLadder({ earn: 5, keep: 8 });
    lad.update(0.016, ranked([7])); // the leaf mounts first
    const st = lad.update(0.016, ranked([9, 7])); // the hub earns later, outranking it
    expect(namedOf(st)).toEqual([9, 7]); // rank order, not mount order — the overlay's
    //                                      greedy declutter gives the hub the pixels
  });

  it('earn/keep asymmetry at the DEFAULT band: rank 25 keeps a label but never earns one', () => {
    const lad = createLadder({ dwellMs: 0 });
    const order: number[] = [];
    for (let i = 0; i < 40; i++) order.push(i);
    let st = lad.update(0.016, ranked(order));
    expect(namedOf(st)).toEqual(order.slice(0, C.EARN)); // exactly the top 20, in rank order
    expect(namedOf(st)).not.toContain(25);
    // node 0 drifts to rank 25 — inside keep (32): the label stays
    const kept = order.filter((n) => n !== 0);
    kept.splice(25, 0, 0);
    st = lad.update(0.016, ranked(kept));
    expect(namedOf(st)).toContain(0);
    // node 0 drifts to rank 35 — outside keep: with dwell 0 it drops at once
    const dropped = order.filter((n) => n !== 0);
    dropped.splice(35, 0, 0);
    st = lad.update(0.016, ranked(dropped));
    expect(namedOf(st)).not.toContain(0);
  });

  it('the hard cap bites in rank order', () => {
    const lad = createLadder({ dwellMs: 0, earn: 5, keep: 6, namedCap: 3 });
    const st = lad.update(0.016, ranked([4, 5, 6, 7, 8]));
    expect(namedOf(st)).toEqual([4, 5, 6]);
  });
});

describe('attention follows the cursor', () => {
  it('cursor proximity re-ranks otherwise equal candidates', () => {
    const lad = createLadder({ dwellMs: 0, cards: 1 });
    const st = lad.update(
      0.016,
      viewOf([
        { node: 1, px: 20, dist: 150, cur: 500 }, // outside the pull radius
        { node: 2, px: 20, dist: 150, cur: 0 }, // on the cursor: score × (1 + CURSOR_PULL)
      ]),
    );
    expect(cardNodes(st)).toEqual([2]);
  });

  it('the pull is the ported linear falloff: ×1.45 at half radius, gone at the radius', () => {
    const lad = createLadder({ dwellMs: 0, cards: 2 });
    // px 20 pulled at half radius → 20 × (1 + 0.9 × 0.5) = 29 > 28; at the radius 20 < 28
    let st = lad.update(
      0.016,
      viewOf([
        { node: 1, px: 28, dist: 150 },
        { node: 2, px: 20, dist: 150, cur: C.CURSOR_RADIUS_PX / 2 },
      ]),
    );
    expect(cardNodes(st)).toEqual([2, 1]);
    st = lad.update(
      0.016,
      viewOf([
        { node: 1, px: 28, dist: 150 },
        { node: 2, px: 20, dist: 150, cur: C.CURSOR_RADIUS_PX },
      ]),
    );
    expect(cardNodes(st)).toEqual([1, 2]);
  });
});

describe('the card budget', () => {
  it('six candidates within card distance → five cards, best-RANKED win, in rank order', () => {
    const lad = createLadder({ dwellMs: 0 });
    const rows: Row[] = [];
    for (let k = 0; k < 6; k++) rows.push({ node: 10 + k, px: 40 - k, dist: 150 });
    const st = lad.update(0.016, viewOf(rows));
    expect(st.cards.length).toBe(C.CARDS);
    expect(cardNodes(st)).toEqual([10, 11, 12, 13, 14]); // 15, worst-ranked, dropped
    expect(evList(st.events.card).enter).toEqual([10, 11, 12, 13, 14]);
  });
});

describe('a pin outranks the ladder', () => {
  it('a pinned node not even in the candidates still reports open', () => {
    const lad = createLadder();
    const st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: 5000 }], { pinned: 42 }));
    expect(st.cards.length).toBe(1);
    expect(st.cards[0]).toMatchObject({ node: 42, tier: 'open', growT: 1 });
    expect(st.open).toBe(42);
    // mounting straight at open fires all three enters, for the overlay's mount order
    expect(evList(st.events.card).enter).toEqual([42]);
    expect(evList(st.events.lock).enter).toEqual([42]);
    expect(evList(st.events.open).enter).toEqual([42]);
  });

  it('bypasses the candidate (glyph-px) filter AND the open threshold', () => {
    const lad = createLadder();
    // glyph far under NAMED_PX, distance far past every card tier — still open
    const st = lad.update(0.016, viewOf([{ node: 3, px: 2, dist: 900 }], { pinned: 3 }));
    expect(st.open).toBe(3);
    expect(st.cards[0]).toMatchObject({ node: 3, tier: 'open', growT: 1 });
  });

  it('rides on top of the card budget: CARDS + 1 slots, pin first', () => {
    const lad = createLadder({ dwellMs: 0 });
    const rows: Row[] = [];
    for (let k = 0; k < 6; k++) rows.push({ node: 10 + k, px: 40 - k, dist: 150 });
    const st = lad.update(0.016, viewOf(rows, { pinned: 99 }));
    expect(st.cards.length).toBe(C.CARDS + 1);
    expect(cardNodes(st)).toEqual([99, 10, 11, 12, 13, 14]);
  });

  it('unpinning releases the slot and fires the exits', () => {
    const lad = createLadder();
    lad.update(0.016, viewOf([], { pinned: 42 }));
    const st = lad.update(0.016, viewOf([]));
    expect(st.open).toBe(-1);
    expect(st.cards.length).toBe(0);
    expect(evList(st.events.open).exit).toEqual([42]);
    expect(evList(st.events.lock).exit).toEqual([42]);
    expect(evList(st.events.card).exit).toEqual([42]);
  });
});

describe('the open slot (budget 1)', () => {
  const near = (node: number, px: number): Row => ({ node, px, dist: 40 });

  it('an empty slot opens immediately; a transfer waits out OPEN_HOLD_MS', () => {
    const lad = createLadder({ dwellMs: 0 });
    let st = lad.update(0.016, viewOf([near(1, 20)]));
    expect(st.open).toBe(1);
    const both = () => viewOf([near(1, 20), near(2, 30)]); // 2 outranks 1
    st = lad.update(0.25, both());
    expect(st.open).toBe(1); // held 250 ms — no transfer yet
    st = lad.update(0.25, both());
    expect(st.open).toBe(1); // 500 ms
    st = lad.update(0.25, both());
    expect(st.open).toBe(2); // 750 ms ≥ 600 — transfers
    expect(evList(st.events.open).exit).toEqual([1]);
    expect(evList(st.events.open).enter).toEqual([2]);
  });

  it('auto-close also waits out the hold (sliding back past the threshold)', () => {
    const lad = createLadder({ dwellMs: 0 });
    lad.update(0.016, viewOf([near(1, 20)]));
    const back = () => viewOf([{ node: 1, px: 20, dist: 60 }]); // past dOpen, still carded
    let st = lad.update(0.25, back());
    expect(st.open).toBe(1); // 250 ms — the body holds
    expect(st.cards[0]!.tier).toBe('open');
    st = lad.update(0.25, back());
    expect(st.open).toBe(1); // 500 ms
    st = lad.update(0.25, back());
    expect(st.open).toBe(-1); // 750 ms ≥ 600 — closes
    expect(st.cards[0]!.tier).toBe('lock'); // the card itself remains, one tier down
    expect(evList(st.events.open).exit).toEqual([1]);
  });

  it('the open card keeps its mount when it loses the budget race', () => {
    const lad = createLadder({ dwellMs: 0 });
    let st = lad.update(0.016, viewOf([near(1, 13)]));
    expect(st.open).toBe(1);
    const rows: Row[] = [near(1, 13)];
    for (let k = 0; k < 5; k++) rows.push({ node: 10 + k, px: 40 - k, dist: 150 });
    st = lad.update(0.016, viewOf(rows));
    expect(st.open).toBe(1);
    expect(st.cards.length).toBe(C.CARDS); // the budget contract still holds
    expect(cardNodes(st)).toEqual([10, 11, 12, 13, 1]); // 1 displaced the worst-ranked slot
    expect(st.cards[4]!.tier).toBe('open');
  });
});

describe('escape retreats', () => {
  const near = () => viewOf([{ node: 1, px: 20, dist: 40 }]);

  it('closes the open slot and suspends auto-open — the same page cannot reclaim it', () => {
    const lad = createLadder({ dwellMs: 0 });
    lad.update(0.016, near());
    lad.retreat();
    let st = lad.update(0.016, near());
    expect(st.open).toBe(-1);
    expect(st.suspended).toBe(true);
    expect(evList(st.events.open).exit).toEqual([1]);
    st = lad.update(10, near()); // however long you hover there
    expect(st.open).toBe(-1);
    expect(st.suspended).toBe(true);
  });

  it('a reported cursor move clears the suspension', () => {
    const lad = createLadder({ dwellMs: 0 });
    lad.update(0.016, near());
    lad.retreat();
    lad.update(0.016, near());
    const st = lad.update(0.016, viewOf([{ node: 1, px: 20, dist: 40 }], { cursorMoved: true }));
    expect(st.suspended).toBe(false);
    expect(st.open).toBe(1);
  });

  it('a DIFFERENT node coming into range does NOT clear it — the retreat must not appear to open something', () => {
    // The port's setOpenEnabled(false): only user input re-enables. If a different page
    // could claim the slot, the retreat flight past other pages would open one of them.
    const lad = createLadder({ dwellMs: 0 });
    lad.update(0.016, near());
    lad.retreat();
    lad.update(0.016, near());
    let st = lad.update(0.016, viewOf([{ node: 2, px: 30, dist: 40 }]));
    expect(st.suspended).toBe(true);
    expect(st.open).toBe(-1);
    // the cursor move (user intent) clears it, and THEN node 2 may open
    st = lad.update(0.016, viewOf([{ node: 2, px: 30, dist: 40 }], { cursorMoved: true }));
    expect(st.suspended).toBe(false);
    expect(st.open).toBe(2);
  });

  it('pinning clears the suspension', () => {
    const lad = createLadder({ dwellMs: 0 });
    lad.update(0.016, near());
    lad.retreat();
    const st = lad.update(0.016, viewOf([], { pinned: 7 }));
    expect(st.suspended).toBe(false);
    expect(st.open).toBe(7);
  });
});

describe('flying backwards runs the ladder in reverse, frame for frame', () => {
  it('a symmetric distance sequence produces a palindromic state sequence', () => {
    // dwell and hold are the only deliberate hysteresis on top of the pure s-driven tiers;
    // zeroed, the state is a pure function of camera distance and must mirror exactly.
    const lad = createLadder({ dwellMs: 0, openHoldMs: 0 });
    const F = fovPxOf(800, DEFAULT_FOV);
    const r = 2; // world radius: crosses NAMED_PX between d = 250 and d = 230
    const approach = [300, 250, 230, 220, 150, 100, 60, 45, 30];
    const seq = approach.concat([...approach].reverse());
    const snaps = seq.map((d) =>
      JSON.stringify(
        snap(lad.update(0.016, viewOf([{ node: 5, px: projectedPx(r, d, F), dist: d }]))),
      ),
    );
    for (let i = 0; i < snaps.length; i++) {
      expect(snaps[i]).toBe(snaps[snaps.length - 1 - i]);
    }
    // and it actually climbed the whole ladder on the way in
    expect(snaps.some((s) => s.includes('"open":5'))).toBe(true);
    expect(snaps.some((s) => s.includes('"tier":"card"'))).toBe(true);
    expect(snaps.some((s) => s.includes('"tier":"lock"'))).toBe(true);
    expect(snaps[0]).toBe(JSON.stringify({ named: [], cards: [], open: -1, suspended: false }));
  });
});

describe('determinism and house rules', () => {
  function scripted(lad: Ladder): string[] {
    const out: string[] = [];
    for (let t = 0; t < 120; t++) {
      const rows: Row[] = [];
      for (let n = 0; n < 12; n++) {
        rows.push({
          node: n,
          px: 8 + (30 * ((t * 13 + n * 29) % 37)) / 37,
          dist: 30 + (400 * ((t * 7 + n * 11) % 23)) / 23,
          cur: n === t % 12 ? 40 : INF,
        });
      }
      if (t === 40 || t === 95) lad.retreat();
      const st = lad.update(0.05, viewOf(rows, {
        pinned: t >= 60 && t < 80 ? 3 : -1,
        cursorMoved: t % 9 === 0,
      }));
      out.push(
        JSON.stringify({
          s: snap(st),
          e: [st.events.named, st.events.card, st.events.lock, st.events.open].map(evList),
        }),
      );
    }
    return out;
  }

  it('two instances fed the same (dt, view) sequence agree tick for tick', () => {
    expect(scripted(createLadder())).toEqual(scripted(createLadder()));
  });

  it('reuses its state, arrays and slot objects — fixed scratch, nothing per tick', () => {
    const lad = createLadder({ dwellMs: 0 });
    const v = () =>
      viewOf([
        { node: 1, px: 20, dist: 150 },
        { node: 2, px: 19, dist: 150 },
      ]);
    const st1 = lad.update(0.016, v());
    const named1 = st1.named;
    const cards1 = st1.cards;
    const slot0 = st1.cards[0];
    const events1 = st1.events;
    const enter1 = st1.events.card.enter;
    const st2 = lad.update(0.016, v());
    expect(st2).toBe(st1);
    expect(st2.named).toBe(named1);
    expect(st2.cards).toBe(cards1);
    expect(st2.cards[0]).toBe(slot0);
    expect(st2.events).toBe(events1);
    expect(st2.events.card.enter).toBe(enter1);
  });

  it('the ladder sources contain no clock and no RNG (§10.1)', async () => {
    // The tsconfig deliberately carries no Node types (a browser-targeted library on lib
    // ES2022+DOM), so the one Node API this audit needs is imported with a non-literal
    // specifier and typed locally, rather than dragging @types/node into the project.
    const fsModule = 'node:fs';
    const { readFileSync } = (await import(/* @vite-ignore */ fsModule)) as {
      readFileSync: (path: URL, encoding: 'utf8') => string;
    };
    const banned = /(Date\s*\.\s*now|performance\s*\.\s*now|Math\s*\.\s*random)\s*\(/;
    for (const f of ['constants.ts', 'types.ts', 'ladder.ts']) {
      const src = readFileSync(new URL(`../src/ladder/${f}`, import.meta.url), 'utf8');
      expect(src).not.toMatch(banned);
    }
  });
});
