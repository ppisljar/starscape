/* The zoom-ladder state machine (DESIGN.md §6), ported from vault-site-3d src/zoom.js — the
 * LOGIC, headless: no DOM, no three.js, no camera, no clock.
 *
 * One continuous mechanism, not five screens. Everything is driven by ONE unclamped number —
 * the projected size a node WOULD have if nothing clamped it. If the box locks and the content
 * tiers also key off box size, you have frozen the variable the tiers depend on and everything
 * past the lock stops progressing; so the unclamped size keeps growing and only what gets
 * drawn is clamped. Because that number is a pure function of camera position there is no
 * animation state, nothing to cancel, and reversing the camera reverses the ladder frame for
 * frame (the dwell and the open-hold are the only deliberate hysteresis on top).
 *
 * TWO progress variables, not one (the non-obvious part, measured on a real corpus):
 *   - the glyph/named ladder keys off the node's OWN apparent size (`projectedPx`), so a
 *     402-link hub announces itself from further away than a leaf — importance buys reach;
 *   - the card ladder keys off a FIXED world radius (CARD_R_SPACINGS × spacing) — node radii
 *     span 12×, and keying cards to node size put leaf pages' open tier inside the camera's
 *     soft stop, i.e. small pages could never open at all. Here that collapses to comparing
 *     `distance` against spacings directly (the fixed-radius projection is monotone in 1/d, so
 *     the pixel thresholds and the distance thresholds are the same test); the fixed radius
 *     survives in the exported projection helpers (`cardPx`, `cardStandoff`) for gotos/HUDs.
 *
 * House rules (§10): no Math.random / Date.now / performance.now — time arrives as dt through
 * update(dt) and dwells are dt-accumulated; no per-node JS objects (live labels are parallel
 * typed arrays; the ≤ CARDS+1 card slots are a fixed pool); reductions are loops;
 * zero allocation per update beyond fixed scratch (grown geometrically, never per tick).
 */

import { DEFAULT_FOV } from '../field/constants';
import {
  CARDS,
  CARD_R_SPACINGS,
  CARD_SPACINGS,
  CURSOR_PULL,
  CURSOR_RADIUS_PX,
  DWELL_MS,
  EARN,
  KEEP,
  LOCK_H,
  LOCK_SPACINGS,
  LOCK_W,
  MIN_DISTANCE,
  NAMED_CAP,
  NAMED_PX,
  OPEN_HOLD_MS,
  OPEN_SPACINGS,
} from './constants';
import type {
  CardSlot,
  Ladder,
  LadderEvents,
  LadderOptions,
  LadderState,
  LadderView,
  TierEvents,
} from './types';

/* ---- the pure projection helpers ---------------------------------------------------------
 * These mirror the field shader's maths (src/field/channels.ts glyphPx) so the ladder, the
 * picker and the shader measure the same thing. At the shipped FALLOFF = 1 the shader's
 * projection is the true one, and `projectedPx` below equals exactly 2 × glyphPx — asserted
 * in test/ladder.test.ts against the real constants. */

/** The projection factor: halfH / tan(fov/2), CSS px. Compute once per resize, pass to the
 *  helpers below. (`fovDeg` is the VERTICAL field of view, like the renderer's.) */
export function fovPxOf(viewportHeightCss: number, fovDeg: number = DEFAULT_FOV): number {
  return viewportHeightCss / 2 / Math.tan((fovDeg * Math.PI) / 360);
}

/** UNCLAMPED projected DIAMETER of a sphere, CSS px: 2·r·fovPx/d — the number the whole ladder
 *  keys off. For the glyph/named ladder pass the node's EFFECTIVE world radius (its size
 *  channel × NODE_SCALE × boost, the radius the shader actually draws). Never clamp this; the
 *  clamp belongs to what is drawn, not to what is measured.
 *  (The field's FALLOFF is 1.0 — the true projection, where glyphPx's dRef pivot cancels;
 *  with a non-1 falloff use glyphPx directly. The test asserts the equivalence.) */
export function projectedPx(worldRadius: number, distance: number, fovPx: number): number {
  return (2 * worldRadius * fovPx) / Math.max(distance, MIN_DISTANCE);
}

/** Inverse of `projectedPx`: the distance at which a sphere of `worldRadius` reads `px`. */
export function standoffOf(worldRadius: number, px: number, fovPx: number): number {
  return (2 * worldRadius * fovPx) / px;
}

/** The card ladder's fixed world radius, derived from the spacing (never from node size). */
export function cardRadiusOf(spacing: number, k: number = CARD_R_SPACINGS): number {
  return spacing * k;
}

/** Projected size of the fixed card radius at `distance` — what the overlay's pre-lock box
 *  tracks. Uniform across pages, by design. */
export function cardPx(spacing: number, distance: number, fovPx: number): number {
  return projectedPx(cardRadiusOf(spacing), distance, fovPx);
}

/** Distance at which the CARD ladder reads `px` — what a goto should stand off by. */
export function cardStandoff(spacing: number, px: number, fovPx: number): number {
  return standoffOf(cardRadiusOf(spacing), px, fovPx);
}

/** Card growth 0→1 between the card and lock distances, LINEAR IN PROJECTED SIZE (1/d), which
 *  is how the predecessor's box grew in pixels — not linear in distance. Clamped. */
export function cardGrowT(distance: number, dCard: number, dLock: number): number {
  const inv = 1 / Math.max(distance, MIN_DISTANCE);
  const invCard = 1 / dCard;
  const span = 1 / dLock - invCard;
  if (!(span > 0)) return distance <= dLock ? 1 : 0;
  const t = (inv - invCard) / span;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/* ---- deterministic rank: heapsort, in place, no allocation --------------------------------
 * ord[0..m) are candidate positions; order is score DESCENDING, ties by position ASCENDING —
 * a total order, so the result is unique whatever heapsort's instability does, and the rank
 * band means the same thing on every engine (Array#sort stability is not relied on). */

/** cmp < 0 ⇔ a sorts before b (higher score first, then lower position). */
function cmpOrd(a: number, b: number, score: Float64Array): number {
  const d = score[b]! - score[a]!;
  return d !== 0 ? d : a - b;
}

function siftDown(ord: Uint32Array, start: number, end: number, score: Float64Array): void {
  let root = start;
  for (;;) {
    let child = root * 2 + 1;
    if (child > end) return;
    if (child + 1 <= end && cmpOrd(ord[child]!, ord[child + 1]!, score) < 0) child++;
    if (cmpOrd(ord[root]!, ord[child]!, score) < 0) {
      const t = ord[root]!;
      ord[root] = ord[child]!;
      ord[child] = t;
      root = child;
    } else return;
  }
}

function heapSort(ord: Uint32Array, m: number, score: Float64Array): void {
  for (let s = (m - 2) >> 1; s >= 0; s--) siftDown(ord, s, m - 1, score);
  for (let e = m - 1; e > 0; e--) {
    const t = ord[0]!;
    ord[0] = ord[e]!;
    ord[e] = t;
    siftDown(ord, 0, e - 1, score);
  }
}

/* ---- the machine ------------------------------------------------------------------------ */

const LEVEL: Record<CardSlot['tier'], number> = { card: 1, lock: 2, open: 3 };

function tierEvents(capacity: number): TierEvents {
  return {
    enter: new Uint32Array(capacity),
    enterCount: 0,
    exit: new Uint32Array(capacity),
    exitCount: 0,
  };
}

export function createLadder(opts: LadderOptions = {}): Ladder {
  const earn = opts.earn ?? EARN;
  const cfg: Required<LadderOptions> = {
    namedPx: opts.namedPx ?? NAMED_PX,
    earn,
    // keep < earn would mean "earned and instantly outside the keep band" — clamp, the band
    // is a hysteresis and hysteresis cannot be negative.
    keep: Math.max(opts.keep ?? KEEP, earn),
    dwellMs: opts.dwellMs ?? DWELL_MS,
    namedCap: opts.namedCap ?? NAMED_CAP,
    cardSpacings: opts.cardSpacings ?? CARD_SPACINGS,
    lockSpacings: opts.lockSpacings ?? LOCK_SPACINGS,
    openSpacings: opts.openSpacings ?? OPEN_SPACINGS,
    cards: opts.cards ?? CARDS,
    openHoldMs: opts.openHoldMs ?? OPEN_HOLD_MS,
    lockW: opts.lockW ?? LOCK_W,
    lockH: opts.lockH ?? LOCK_H,
    cardRadiusSpacings: opts.cardRadiusSpacings ?? CARD_R_SPACINGS,
    cursorRadiusPx: opts.cursorRadiusPx ?? CURSOR_RADIUS_PX,
    cursorPull: opts.cursorPull ?? CURSOR_PULL,
  };

  // ---- fixed scratch: candidate rank (grows geometrically, never shrinks, never per tick)
  let cap = 0;
  let ord = new Uint32Array(0);
  let score = new Float64Array(0);
  function ensure(n: number): void {
    if (n <= cap) return;
    cap = Math.max(n, cap * 2, 64);
    ord = new Uint32Array(cap);
    score = new Float64Array(cap);
  }

  // ---- named tier: live labels, parallel typed arrays (§10.2). liveAgeMs is ms since
  // mount — the minimum-lifetime grace (the port's `since`); liveRank is per-tick scratch
  // for publishing the set in rank order.
  const liveNode = new Uint32Array(cfg.namedCap);
  const liveAgeMs = new Float64Array(cfg.namedCap);
  const liveRank = new Int32Array(cfg.namedCap);
  let liveCount = 0;

  // ---- card tier: a fixed pool of ≤ CARDS+1 slots, plus last tick's set for the event diff
  const slotCap = cfg.cards + 1;
  const slotPool: CardSlot[] = [];
  for (let i = 0; i < slotCap; i++) slotPool.push({ node: -1, tier: 'card', growT: 0 });
  const slotDist = new Float64Array(slotCap);
  const cardsOut: CardSlot[] = [];
  const prevNode = new Int32Array(slotCap);
  const prevLevel = new Uint8Array(slotCap);
  let prevCount = 0;

  // ---- the open slot
  let openNode = -1;
  let openHeldMs = 0;
  let suspended = false;

  const events: LadderEvents = {
    named: tierEvents(cfg.namedCap),
    card: tierEvents(slotCap),
    lock: tierEvents(slotCap),
    open: tierEvents(slotCap),
  };

  const state: LadderState = {
    named: liveNode,
    namedCount: 0,
    cards: cardsOut,
    open: -1,
    suspended: false,
    events,
  };

  function update(dt: number, view: LadderView): LadderState {
    const dtMs = dt * 1000;
    const cand = view.candidates;
    const count = cand.count;
    const pinned = view.pinned;
    const nodeCol = cand.node;
    const pxCol = cand.projectedPx;
    const distCol = cand.distance;
    const curCol = cand.cursorPx;
    const dCard = cfg.cardSpacings * view.spacing;
    const dLock = cfg.lockSpacings * view.spacing;
    const dOpen = cfg.openSpacings * view.spacing;
    ensure(count);

    // ---- 1. the candidate filter + score. Attention follows the cursor: proximity to the
    // pointer pulls a candidate's score up (never down), so what you point at outranks what
    // merely happens to be big. A pinned node bypasses the px gate entirely — pinning
    // something that is not a candidate must still pin.
    let m = 0;
    let pinnedPos = -1;
    let openPos = -1;
    for (let c = 0; c < count; c++) {
      const node = nodeCol[c]!;
      if (node === pinned) pinnedPos = c;
      if (node === openNode) openPos = c;
      const px = pxCol[c]!;
      if (px < cfg.namedPx && node !== pinned) continue;
      const cur = curCol[c]!;
      const pull =
        cur < cfg.cursorRadiusPx ? 1 + cfg.cursorPull * (1 - cur / cfg.cursorRadiusPx) : 1;
      score[c] = px * pull;
      ord[m++] = c;
    }
    heapSort(ord, m, score);

    // ---- 2. named: drops. A mounted label is held for its first dwellMs whatever its rank
    // (the minimum LIFETIME — the port's `since` + labelDwell); past that grace it drops the
    // tick it sits outside the top `keep` ranks (or left the candidates entirely). The
    // EARN < KEEP rank band is the hysteresis: a label earned at rank 19 survives drifting
    // to rank 31, and the band + grace measured 0 label re-entries per second over 3.9 s
    // of flight in the predecessor (zoom.js updateLabels — the exit test is an
    // instantaneous band test after the grace, never an accumulated out-dwell).
    const evNamed = events.named;
    evNamed.enterCount = 0;
    evNamed.exitCount = 0;
    const keepN = Math.min(cfg.keep, m);
    let w = 0;
    for (let l = 0; l < liveCount; l++) {
      const node = liveNode[l]!;
      let inKeep = false;
      for (let k = 0; k < keepN; k++) {
        if (nodeCol[ord[k]!]! === node) {
          inKeep = true;
          break;
        }
      }
      const age = liveAgeMs[l]! + dtMs;
      if (!inKeep && age >= cfg.dwellMs) {
        evNamed.exit[evNamed.exitCount++] = node;
        continue; // dropped — not carried forward
      }
      liveNode[w] = node;
      liveAgeMs[w] = age;
      w++;
    }
    liveCount = w;

    // ---- 3. named: mounts. A node entering the top `earn` ranks is named THAT tick — the
    // port mounted the frame a glyph crossed the gate, so flying at a region names it
    // immediately and a sub-dwell flyby still names what it passes; the dwell above is a
    // minimum lifetime, not an entry delay. Rank order, so when the cap bites the
    // better-ranked candidate wins the slot.
    const earnN = Math.min(cfg.earn, m);
    for (let k = 0; k < earnN && liveCount < cfg.namedCap; k++) {
      const node = nodeCol[ord[k]!]!;
      let isLive = false;
      for (let l = 0; l < liveCount; l++) {
        if (liveNode[l]! === node) {
          isLive = true;
          break;
        }
      }
      if (isLive) continue;
      liveNode[liveCount] = node;
      liveAgeMs[liveCount] = 0;
      liveCount++;
      evNamed.enter[evNamed.enterCount++] = node;
    }

    // ---- 3b. named: publish in rank order. The overlay declutters labels greedily in this
    // order, so "already placed" must mean "more important" (the port walked the sorted
    // candidate list); insertion order would let an old leaf's label hide a newly-earned
    // hub's indefinitely. Nodes absent from the candidates rank last (they are riding out
    // their drop grace). Early-out once every live node is ranked; insertion sort — the
    // set is ≤ namedCap and already mostly sorted tick to tick.
    for (let l = 0; l < liveCount; l++) liveRank[l] = m + l;
    for (let k = 0, found = 0; k < m && found < liveCount; k++) {
      const node = nodeCol[ord[k]!]!;
      for (let l = 0; l < liveCount; l++) {
        if (liveNode[l]! === node) {
          if (liveRank[l]! >= m) {
            liveRank[l] = k;
            found++;
          }
          break;
        }
      }
    }
    for (let a = 1; a < liveCount; a++) {
      const r = liveRank[a]!;
      const nd = liveNode[a]!;
      const ag = liveAgeMs[a]!;
      let b = a - 1;
      while (b >= 0 && liveRank[b]! > r) {
        liveRank[b + 1] = liveRank[b]!;
        liveNode[b + 1] = liveNode[b]!;
        liveAgeMs[b + 1] = liveAgeMs[b]!;
        b--;
      }
      liveRank[b + 1] = r;
      liveNode[b + 1] = nd;
      liveAgeMs[b + 1] = ag;
    }

    // ---- 4. cards. Pinned first and ON TOP of the budget (a pin outranks the ladder: it
    // bypasses the budget, the open threshold and the candidate filter — tier `open`, growT 1,
    // whether or not the node is even in the candidates). Then rank order, distance-gated by
    // the FIXED card radius (≤ cardSpacings × spacing), until the budget fills — so with six
    // nodes oversubscribing five slots, the best-ranked five win.
    let nSlots = 0;
    if (pinned >= 0) {
      const s = slotPool[nSlots]!;
      s.node = pinned;
      s.tier = 'open';
      s.growT = 1;
      slotDist[nSlots] = pinnedPos >= 0 ? distCol[pinnedPos]! : 0;
      nSlots++;
    }
    const budget = pinned >= 0 ? cfg.cards + 1 : cfg.cards;
    for (let k = 0; k < m && nSlots < budget; k++) {
      const c = ord[k]!;
      const node = nodeCol[c]!;
      if (node === pinned) continue;
      const d = distCol[c]!;
      if (d > dCard) continue;
      const s = slotPool[nSlots]!;
      s.node = node;
      s.tier = d <= dLock ? 'lock' : 'card';
      s.growT = cardGrowT(d, dCard, dLock);
      slotDist[nSlots] = d;
      nSlots++;
    }

    // ---- 5. the open card keeps its mount. Losing the budget race must not unmount a card
    // someone is reading (the predecessor force-kept it in `want`): if the open node missed
    // the cut but is still inside the card distance, it takes (or replaces) the LAST slot —
    // the worst-ranked one — keeping the ≤ budget contract. Fallen past the card distance (or
    // out of the candidates), it closes now; the ladder below card tier has nothing to hold.
    if (pinned < 0 && openNode >= 0) {
      let has = false;
      for (let s = 0; s < nSlots; s++) {
        if (slotPool[s]!.node === openNode) {
          has = true;
          break;
        }
      }
      if (!has) {
        const d = openPos >= 0 ? distCol[openPos]! : Infinity;
        if (d <= dCard && budget > 0) {
          const at = nSlots < budget ? nSlots++ : nSlots - 1;
          const s = slotPool[at]!;
          s.node = openNode;
          s.tier = d <= dLock ? 'lock' : 'card';
          s.growT = cardGrowT(d, dCard, dLock);
          slotDist[at] = d;
        } else {
          openNode = -1;
          openHeldMs = 0;
        }
      }
    }

    // ---- 6. the open slot (budget 1). A pin IS the open slot and clears any suspension
    // (Escape closed something; selecting something is a new intent). Otherwise: best-ranked
    // card within the open distance wins; an EMPTY slot opens immediately, but transferring
    // or auto-closing waits out openHoldMs so two pages straddling the threshold cannot fight
    // over the slot every frame. While suspended (retreat), NOTHING auto-opens — not even a
    // different page the retreat flight passes (the port's setOpenEnabled(false): a page
    // claiming the slot mid-retreat makes the gesture appear to OPEN something instead of
    // closing something) — until real user intent: a reported cursor move, or a pin.
    if (pinned >= 0) {
      if (openNode !== pinned) {
        openNode = pinned;
        openHeldMs = 0;
      } else {
        openHeldMs += dtMs;
      }
      suspended = false;
    } else {
      if (openNode >= 0) openHeldMs += dtMs;
      let top = -1;
      for (let s = 0; s < nSlots; s++) {
        if (slotDist[s]! <= dOpen) {
          top = slotPool[s]!.node;
          break;
        }
      }
      if (suspended && view.cursorMoved) {
        suspended = false;
      }
      if (!suspended) {
        if (openNode < 0) {
          if (top >= 0) {
            openNode = top;
            openHeldMs = 0;
          }
        } else if (top >= 0 && top !== openNode) {
          if (openHeldMs >= cfg.openHoldMs) {
            openNode = top;
            openHeldMs = 0;
          }
        } else if (top !== openNode && openHeldMs >= cfg.openHoldMs) {
          // top === -1 here: the open node slid back out past the open distance
          openNode = -1;
          openHeldMs = 0;
        }
      }
    }
    if (openNode >= 0) {
      for (let s = 0; s < nSlots; s++) {
        const slot = slotPool[s]!;
        if (slot.node === openNode) {
          slot.tier = 'open';
          slot.growT = 1;
          break;
        }
      }
    }

    // ---- 7. events: diff against last tick's card set, one enter/exit per tier boundary
    // crossed (absent counts as level 0, so a pin mounting straight at open fires all three
    // enters and an open card unmounting fires all three exits — mount order for the overlay).
    const evCard = events.card;
    const evLock = events.lock;
    const evOpen = events.open;
    evCard.enterCount = 0;
    evCard.exitCount = 0;
    evLock.enterCount = 0;
    evLock.exitCount = 0;
    evOpen.enterCount = 0;
    evOpen.exitCount = 0;
    for (let s = 0; s < nSlots; s++) {
      const slot = slotPool[s]!;
      const lvl = LEVEL[slot.tier];
      let was = 0;
      for (let p = 0; p < prevCount; p++) {
        if (prevNode[p]! === slot.node) {
          was = prevLevel[p]!;
          break;
        }
      }
      if (was < 1) evCard.enter[evCard.enterCount++] = slot.node;
      if (was < 2 && lvl >= 2) evLock.enter[evLock.enterCount++] = slot.node;
      if (was < 3 && lvl >= 3) evOpen.enter[evOpen.enterCount++] = slot.node;
      if (was >= 3 && lvl < 3) evOpen.exit[evOpen.exitCount++] = slot.node;
      if (was >= 2 && lvl < 2) evLock.exit[evLock.exitCount++] = slot.node;
    }
    for (let p = 0; p < prevCount; p++) {
      const node = prevNode[p]!;
      let still = false;
      for (let s = 0; s < nSlots; s++) {
        if (slotPool[s]!.node === node) {
          still = true;
          break;
        }
      }
      if (still) continue;
      const was = prevLevel[p]!;
      if (was >= 3) evOpen.exit[evOpen.exitCount++] = node;
      if (was >= 2) evLock.exit[evLock.exitCount++] = node;
      evCard.exit[evCard.exitCount++] = node;
    }
    prevCount = nSlots;
    for (let s = 0; s < nSlots; s++) {
      const slot = slotPool[s]!;
      prevNode[s] = slot.node;
      prevLevel[s] = LEVEL[slot.tier];
    }

    // ---- 8. publish (same objects every tick — fixed scratch, nothing allocated)
    cardsOut.length = nSlots;
    for (let s = 0; s < nSlots; s++) cardsOut[s] = slotPool[s]!;
    state.namedCount = liveCount;
    state.open = openNode;
    state.suspended = suspended;
    return state;
  }

  function retreat(): void {
    suspended = true;
    openNode = -1;
    openHeldMs = 0;
  }

  return { update, retreat, state, config: cfg };
}
