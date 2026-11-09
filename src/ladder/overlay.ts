/* The zoom ladder's DOM overlay (DESIGN.md §6, stage 7 part 2) — the presentation half of the
 * machine in ./ladder.ts. The ladder decides WHICH nodes are named / carded / open; this
 * module mounts the DOM for exactly those sets and projects their world positions to CSS px
 * each frame — never all nodes, only the ≤ namedCap labels and ≤ cards+1 cards currently live.
 *
 * Presentation rules ported from vault-site-3d (src/zoom.js + style.css), restated because
 * each one was learned on a real corpus:
 *
 *   - THE BOX GROWS IN PIXELS, THE TYPE NEVER SCALES. A card is a screen rectangle whose
 *     width/height in px track the ladder's growT and freeze at lockW × lockH; nothing is
 *     ever CSS-scaled, so type is crisp at every tier and "the interior resolves" is
 *     mechanical truth: more content simply fits. (A transform:scale on type blurs exactly
 *     during the transition you most want to look good.)
 *   - The open tier swaps the ASYNC body into the locked rect: loading state first, and a
 *     stale-response guard — the camera can outrun a provider, and a body resolving for a
 *     node the card no longer shows must be dropped, not rendered.
 *   - Height follows the content until the body opens (a locked-size box around two lines
 *     of summary is mostly empty, and empty chrome reads as something failing to load).
 *   - Cards clamp inside the stage: they slide along the edge instead of hanging off it,
 *     which is what every map label does and what stops a half-visible panel reading as a
 *     rendering fault.
 *   - Labels and cards declutter greedily in priority order (state.cards is already pinned-
 *     first-then-rank; state.named is published by the ladder in rank order), so "already
 *     placed" means "more important"; a node holding a card hides its own label — a card is
 *     a better label than a label.
 *   - The pinned card carries a pin affordance (the one pointer-interactive piece of chrome
 *     beyond the scrolling body); Escape is the CONSUMER's gesture — the overlay only renders.
 *
 * House rules (§10): zero per-frame allocation in the projection maths (one reused scratch
 * point, reused box scratch arrays grown geometrically); DOM writes only on change — every
 * written style is dirty-checked against cached NUMBERS (position, quantised opacity,
 * z-index), so no transform/opacity string is even built for an unchanged frame; no clock,
 * no RNG. The CSS ships as a string export + injectStyles() helper — a library must not
 * fetch external files.
 */

import { LOCK_H, LOCK_W } from './constants';
import type { LadderState } from './types';
import type { ContentProvider } from './provider';

/* ---- presentation constants (exported once — §5/§10) ------------------------------------ */

/** The card's mount size, CSS px — the box the card tier is born at (predecessor: 150×54). */
export const CARD_MIN_W = 150;
export const CARD_MIN_H = 54;

/** Stage margin cards are clamped inside, CSS px. */
export const EDGE_MARGIN = 8;

/** Label anchor offset right of the glyph centre, CSS px. */
export const LABEL_DX = 10;

/** Label declutter padding, CSS px (x / y). */
export const LABEL_PAD_X = 6;
export const LABEL_PAD_Y = 3;

/** A card covered more than this fraction by a better-ranked card is hidden (open and
 *  pinned cards never hide — what you are reading does not declutter away). */
export const COVER_MAX = 0.3;

/** Cards fade in over this leading fraction of their growth (predecessor's 0.22 ramp). */
export const CARD_FADE_T = 0.22;

/** The summary earns its space between these growT bounds — at mount size the box cannot
 *  hold it, at lock it reads at full strength. */
export const SUM_IN_T = 0.35;
export const SUM_FULL_T = 0.75;

/** Labels further off-screen than this are hidden rather than positioned, CSS px. */
export const LABEL_CULL_PX = 40;

/* ---- pure helpers (tested headless in test/overlay.test.ts) ----------------------------- */

/** A projected screen point. Reused — the overlay owns one scratch instance. */
export interface ProjectedPoint {
  /** CSS px from the canvas rect's left / top. */
  x: number;
  y: number;
  /** Clip-space w — the view depth for a standard perspective matrix. */
  depth: number;
}

/**
 * Project a world point through a column-major clip matrix (projection × worldInverse, the
 * three.js `Matrix4.elements` layout) to CSS px in a width × height rect. Returns false when
 * the point is behind the camera (clip w ≤ 0) — x/y are then meaningless and zeroed.
 * Zero allocation: writes into the caller's `out`.
 */
export function projectPoint(
  m: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
  width: number,
  height: number,
  out: ProjectedPoint,
): boolean {
  const cx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
  const cy = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
  const cw = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  out.depth = cw;
  if (!(cw > 0)) {
    out.x = 0;
    out.y = 0;
    return false;
  }
  const inv = 1 / cw;
  out.x = (cx * inv * 0.5 + 0.5) * width;
  out.y = (-cy * inv * 0.5 + 0.5) * height;
  return true;
}

export interface Dims {
  w: number;
  h: number;
}

/** The card box for a growth fraction: CARD_MIN at growT 0, lockW × lockH at 1, linear and
 *  clamped — growT is already linear in projected size (ladder.ts cardGrowT), so this IS
 *  "the box grows in pixels with approach, then freezes". Integer px (crisp borders). */
export function cardDims(growT: number, lockW: number, lockH: number, out: Dims): Dims {
  const t = growT < 0 ? 0 : growT > 1 ? 1 : growT;
  out.w = Math.round(CARD_MIN_W + (lockW - CARD_MIN_W) * t);
  out.h = Math.round(CARD_MIN_H + (lockH - CARD_MIN_H) * t);
  return out;
}

export interface Point2 {
  x: number;
  y: number;
}

/** Clamp a w × h box into a width × height stage with `margin` all round. A box wider than
 *  the stage pins to the leading margin (predecessor behaviour: Math.max(M, …)). */
export function clampBox(
  x: number,
  y: number,
  w: number,
  h: number,
  width: number,
  height: number,
  margin: number,
  out: Point2,
): Point2 {
  const mx = Math.max(margin, width - w - margin);
  const my = Math.max(margin, height - h - margin);
  out.x = x < margin ? margin : x > mx ? mx : x;
  out.y = y < margin ? margin : y > my ? my : y;
  return out;
}

/** Fraction of box A (ax, ay, aw, ah) covered by box B — the card declutter test. */
export function coverFrac(
  ax: number,
  ay: number,
  aw: number,
  ah: number,
  bx: number,
  by: number,
  bw: number,
  bh: number,
): number {
  const ox = Math.min(ax + aw, bx + bw) - Math.max(ax, bx);
  if (ox <= 0) return 0;
  const oy = Math.min(ay + ah, by + bh) - Math.max(ay, by);
  if (oy <= 0) return 0;
  const area = aw * ah;
  return area > 0 ? (ox * oy) / area : 0;
}

/* ---- the stylesheet ---------------------------------------------------------------------- */

/** The overlay's CSS, fresh and minimal (dark-ground defaults matching BG_DARK). No
 *  transform:scale anywhere and every font a fixed px size — type never scales. Consumers
 *  with their own look override the ss-* classes instead of editing this string. */
export const OVERLAY_CSS = `
.ss-labels,.ss-cards{position:absolute;inset:0;overflow:hidden;pointer-events:none}
.ss-cards{z-index:2}
.ss-label{position:absolute;left:0;top:0;display:flex;align-items:center;gap:5px;
  font:10.5px/1.4 ui-monospace,Menlo,Consolas,monospace;color:#aab3c8;white-space:nowrap;
  max-width:260px;text-shadow:0 0 7px rgba(2,4,9,.95),0 0 3px rgba(2,4,9,.95);
  transition:opacity .18s linear;pointer-events:none;opacity:0}
.ss-label i{width:6px;height:6px;border-radius:50%;background:currentColor;flex:none}
.ss-label span{overflow:hidden;text-overflow:ellipsis}
.ss-card{position:absolute;left:0;top:0;box-sizing:border-box;overflow:hidden;
  display:flex;flex-direction:column;gap:5px;padding:9px 11px;border-radius:5px;
  border:1px solid #2b3658;border-top:2px solid #59a0ff;
  background:linear-gradient(180deg,rgba(13,19,34,.93),rgba(7,10,19,.96));
  color:#cdd6e4;font:11px/1.5 ui-monospace,Menlo,Consolas,monospace;
  box-shadow:0 20px 60px -28px #000;transition:opacity .18s linear;pointer-events:none}
.ss-card.ss-locked{box-shadow:0 26px 70px -26px #000,0 0 26px -12px rgba(89,160,255,.4)}
.ss-card.ss-open{pointer-events:auto}
.ss-card.ss-pinned{border-color:#59a0ff}
.ss-bar{display:flex;justify-content:space-between;align-items:center;gap:8px;flex:none;
  font-size:8.5px;letter-spacing:.14em;text-transform:uppercase;color:#5c6688}
.ss-pin{display:none;border:1px solid #2b3658;border-radius:3px;background:none;cursor:pointer;
  color:#8d98b5;font:inherit;font-size:8.5px;letter-spacing:.1em;text-transform:uppercase;
  padding:1px 6px}
.ss-pin:hover{color:#e6ecf8;border-color:#59a0ff}
.ss-card.ss-pinned .ss-pin{display:inline-block}
.ss-title{margin:0;flex:none;font-size:12.5px;line-height:1.3;font-weight:500;color:#e6ecf8}
.ss-sum{margin:0;flex:none;font-size:10.5px;line-height:1.5;color:#8d98b5;
  transition:opacity .2s linear}
.ss-card.ss-open .ss-sum{display:none}
.ss-body{display:none;flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;margin-top:4px;
  padding-top:7px;border-top:1px solid #1a2238;font-size:11px;line-height:1.62;color:#aeb6c8;
  scrollbar-width:thin}
.ss-card.ss-open .ss-body{display:block}
.ss-body p{margin:0 0 8px}
.ss-body.ss-loading{color:#5c6688;font-style:italic}
`;

/** The id the injected style tag carries — injection is idempotent. */
export const OVERLAY_STYLE_ID = 'starscape-ladder-css';

/** Inject OVERLAY_CSS into the document once (no-op when already present). Call before
 *  creating an overlay; a consumer bundling its own styles simply never calls this. */
export function injectStyles(doc: Document = document): void {
  if (doc.getElementById(OVERLAY_STYLE_ID)) return;
  const el = doc.createElement('style');
  el.id = OVERLAY_STYLE_ID;
  el.textContent = OVERLAY_CSS;
  doc.head.appendChild(el);
}

/* ---- the overlay ------------------------------------------------------------------------- */

/** Everything the overlay needs for one frame. The matrix/positions references may be the
 *  same objects every frame (they are read, never retained beyond the call). */
export interface OverlayView {
  /** Column-major clip matrix: projection × worldInverse (three.js Matrix4.elements). */
  matrix: ArrayLike<number>;
  /** Current node positions, 3n — where the nodes are DRAWN (the consumer's settled view
   *  snapshot), not necessarily the frozen layout. */
  positions: ArrayLike<number>;
  /** The canvas rect, CSS px. The container must coincide with the canvas. */
  width: number;
  height: number;
  /** Pinned node (−1 none): its card shows the pin affordance and never declutters away. */
  pinned: number;
}

export interface OverlayOptions {
  /** The overlay root. Must be a positioned element exactly covering the canvas (the demo
   *  uses position:fixed; inset:0). Two layers (labels, cards) are created inside it. */
  container: HTMLElement;
  provider: ContentProvider;
  /** The locked card rect, CSS px — pass the ladder's config when overridden. */
  lockW?: number;
  lockH?: number;
  /** The pin affordance's click handler — the consumer clears its selection here. */
  onUnpin?: () => void;
}

export interface Overlay {
  /** Mount/unmount per `state.events`, then lay out the CURRENT named/card sets. Call once
   *  per frame with the ladder's fresh state. */
  update(state: LadderState, view: OverlayView): void;
  /** Live DOM element counts, for stats overlays. */
  readonly labelCount: number;
  readonly cardCount: number;
  dispose(): void;
}

interface LabelRec {
  node: number;
  el: HTMLElement;
  text: HTMLElement;
  /** Measured box, cached at mount (one layout read per mount, never per frame). */
  w: number;
  h: number;
  // dirty-check caches — numbers, so an unchanged frame builds no strings at all (§10)
  bx: number;
  by: number;
  on: boolean;
}

interface CardRec {
  node: number;
  el: HTMLElement;
  bar: HTMLElement;
  title: HTMLElement;
  sum: HTMLElement;
  body: HTMLElement;
  /** Stale-response token: bumped on every (re)mount and body request; a resolution whose
   *  token no longer matches is dropped — the node changed meanwhile. */
  req: number;
  /** Node the body content belongs to (−1 none) — reopening the same node skips the fetch. */
  bodyFor: number;
  // dirty-check caches — DOM is written only when one of these actually changes, and they
  // are NUMBERS: the style string is built only after the numeric compare says it differs
  bx: number;
  by: number;
  w: number;
  hFix: number; // fixed height px, −1 for 'auto'
  mh: number;
  opQ: number; // opacity in hundredths (0..100), −1 unknown
  sumQ: number; // summary opacity in hundredths, −1 unknown
  zN: number; // z-index, −1000 unknown
  bits: number; // bit0 locked · bit1 open · bit2 pinned
}

export function createOverlay(opts: OverlayOptions): Overlay {
  const doc = opts.container.ownerDocument;
  const provider = opts.provider;
  const lockW = opts.lockW ?? LOCK_W;
  const lockH = opts.lockH ?? LOCK_H;

  const labelLayer = doc.createElement('div');
  labelLayer.className = 'ss-labels';
  const cardLayer = doc.createElement('div');
  cardLayer.className = 'ss-cards';
  opts.container.appendChild(labelLayer);
  opts.container.appendChild(cardLayer);

  const labels = new Map<number, LabelRec>();
  const cards = new Map<number, CardRec>();
  const labelPool: LabelRec[] = [];
  const cardPool: CardRec[] = [];

  /* ---- fixed scratch: one projected point, box arrays grown geometrically (§10) --------- */
  const P: ProjectedPoint = { x: 0, y: 0, depth: 0 };
  const DIMS: Dims = { w: 0, h: 0 };
  const BOX: Point2 = { x: 0, y: 0 };
  let labelBoxes: Float64Array = new Float64Array(64 * 4);
  let cardBoxes: Float64Array = new Float64Array(8 * 4);
  const ensureBoxes = (buf: Float64Array, k: number): Float64Array =>
    k * 4 <= buf.length ? buf : new Float64Array(Math.max(buf.length * 2, k * 4));

  /* ---- mounting --------------------------------------------------------------------------- */

  function mountLabel(i: number): void {
    if (labels.has(i)) return;
    let rec = labelPool.pop();
    if (!rec) {
      const el = doc.createElement('div');
      el.className = 'ss-label';
      const dot = doc.createElement('i');
      const text = doc.createElement('span');
      el.appendChild(dot);
      el.appendChild(text);
      rec = { node: -1, el, text, w: 0, h: 0, bx: Infinity, by: Infinity, on: false };
    }
    rec.node = i;
    rec.text.textContent = provider.card(i).title;
    rec.bx = Infinity; // no cached position — the first layout pass always writes
    rec.by = Infinity;
    rec.on = false; // mounts hidden; the first layout pass fades it in (CSS transition)
    rec.el.style.opacity = '0';
    rec.el.style.transform = '';
    labelLayer.appendChild(rec.el);
    // one layout read per mount, cached — never re-measured per frame
    rec.w = rec.el.offsetWidth || 120;
    rec.h = rec.el.offsetHeight || 16;
    labels.set(i, rec);
  }

  function unmountLabel(i: number): void {
    const rec = labels.get(i);
    if (!rec) return;
    labels.delete(i);
    rec.el.remove();
    rec.node = -1;
    labelPool.push(rec);
  }

  function mountCard(i: number): void {
    if (cards.has(i)) return;
    let rec = cardPool.pop();
    if (!rec) {
      const el = doc.createElement('div');
      el.className = 'ss-card';
      const bar = doc.createElement('div');
      bar.className = 'ss-bar';
      const meta = doc.createElement('span');
      const pin = doc.createElement('button');
      pin.className = 'ss-pin';
      pin.type = 'button';
      pin.textContent = 'pinned · esc';
      bar.appendChild(meta);
      bar.appendChild(pin);
      const title = doc.createElement('h3');
      title.className = 'ss-title';
      const sum = doc.createElement('p');
      sum.className = 'ss-sum';
      const body = doc.createElement('div');
      body.className = 'ss-body';
      el.appendChild(bar);
      el.appendChild(title);
      el.appendChild(sum);
      el.appendChild(body);
      const r: CardRec = {
        node: -1, el, bar: meta, title, sum, body,
        req: 0, bodyFor: -1,
        bx: Infinity, by: Infinity, w: -1, hFix: -2, mh: -1,
        opQ: -1, sumQ: -1, zN: -1000, bits: -1,
      };
      pin.addEventListener('click', (ev: Event): void => {
        ev.stopPropagation();
        if (opts.onUnpin) opts.onUnpin();
      });
      // a drag starting on a card must not fly the camera underneath it
      el.addEventListener('pointerdown', (ev: Event): void => ev.stopPropagation());
      rec = r;
    }
    rec.node = i;
    rec.req++;
    rec.bodyFor = -1;
    const c = provider.card(i);
    rec.bar.textContent = `#${i}`;
    rec.title.textContent = c.title;
    rec.sum.textContent = c.summary ?? '';
    rec.body.textContent = '';
    rec.body.classList.remove('ss-loading');
    rec.el.classList.remove('ss-locked', 'ss-open', 'ss-pinned');
    rec.bits = 0;
    rec.bx = Infinity;
    rec.by = Infinity;
    rec.w = -1;
    rec.hFix = -2;
    rec.mh = -1;
    rec.opQ = 0; // style.opacity is set to '0' just below
    rec.sumQ = -1;
    rec.zN = -1000;
    rec.el.style.opacity = '0';
    rec.el.style.transform = '';
    rec.el.style.width = '';
    rec.el.style.height = '';
    rec.el.style.maxHeight = '';
    rec.el.style.zIndex = '';
    rec.sum.style.opacity = '';
    cardLayer.appendChild(rec.el);
    cards.set(i, rec);
  }

  function unmountCard(i: number): void {
    const rec = cards.get(i);
    if (!rec) return;
    cards.delete(i);
    rec.req++; // any in-flight body resolution is now stale
    rec.body.textContent = ''; // free the body DOM
    rec.el.remove();
    rec.node = -1;
    rec.bodyFor = -1;
    cardPool.push(rec);
  }

  /* The open tier: swap the async body in. Loading state immediately; the resolution is
   * guarded by (token, node) — if the slot transferred, the card unmounted, or the card was
   * recycled for another node while the provider worked, the stale body is dropped. */
  function openBody(i: number): void {
    const rec = cards.get(i);
    if (!rec) return;
    if (rec.bodyFor === i) return; // already loaded (or loading) for this very node
    rec.bodyFor = i;
    const token = ++rec.req;
    rec.body.textContent = 'loading…';
    rec.body.classList.add('ss-loading');
    provider.body(i).then(
      (res: string | HTMLElement): void => {
        if (rec.req !== token || rec.node !== i) return; // stale — the node changed meanwhile
        rec.body.classList.remove('ss-loading');
        if (typeof res === 'string') {
          rec.body.innerHTML = res;
        } else {
          rec.body.textContent = '';
          rec.body.appendChild(res);
        }
      },
      (): void => {
        if (rec.req !== token || rec.node !== i) return;
        rec.body.classList.remove('ss-loading');
        rec.body.textContent = 'failed to load';
      },
    );
  }

  /* ---- per-frame layout -------------------------------------------------------------------- */

  function update(state: LadderState, view: OverlayView): void {
    const ev = state.events;
    /* exits before enters. open.exit needs no handler: the ss-open class is synced from the
     * slot tier below, and the body content deliberately STAYS — sliding back out and in
     * again reopens instantly (openBody skips the fetch while bodyFor still matches). */
    for (let k = 0; k < ev.card.exitCount; k++) unmountCard(ev.card.exit[k]!);
    for (let k = 0; k < ev.named.exitCount; k++) unmountLabel(ev.named.exit[k]!);
    for (let k = 0; k < ev.named.enterCount; k++) mountLabel(ev.named.enter[k]!);
    for (let k = 0; k < ev.card.enterCount; k++) mountCard(ev.card.enter[k]!);
    for (let k = 0; k < ev.open.enterCount; k++) openBody(ev.open.enter[k]!);

    const m = view.matrix;
    const pos = view.positions;
    const W = view.width;
    const H = view.height;

    /* -- cards, in state.cards order (already priority order: pinned first, then rank) ------ */
    const slots = state.cards;
    cardBoxes = ensureBoxes(cardBoxes, slots.length);
    let placedC = 0;
    for (let s = 0; s < slots.length; s++) {
      const slot = slots[s]!;
      const rec = cards.get(slot.node);
      if (!rec) continue;
      const isOpen = slot.tier === 'open';
      const isPinned = slot.node === view.pinned;
      const bits = (slot.tier !== 'card' ? 1 : 0) | (isOpen ? 2 : 0) | (isPinned ? 4 : 0);
      if (bits !== rec.bits) {
        rec.el.classList.toggle('ss-locked', (bits & 1) !== 0);
        rec.el.classList.toggle('ss-open', (bits & 2) !== 0);
        rec.el.classList.toggle('ss-pinned', (bits & 4) !== 0);
        rec.bits = bits;
      }
      const o = slot.node * 3;
      if (!projectPoint(m, pos[o]!, pos[o + 1]!, pos[o + 2]!, W, H, P)) {
        if (rec.opQ !== 0) {
          rec.el.style.opacity = '0';
          rec.opQ = 0;
        }
        continue; // behind the camera — hidden, but the mount (and an open body) survives
      }
      cardDims(slot.growT, lockW, lockH, DIMS);
      const w = DIMS.w;
      const h = DIMS.h;
      if (w !== rec.w) {
        rec.el.style.width = `${w}px`;
        rec.w = w;
      }
      if (h !== rec.mh) {
        rec.el.style.maxHeight = `${h}px`;
        rec.mh = h;
      }
      // height follows the content until the body opens (see header)
      const hFix = isOpen ? h : -1;
      if (hFix !== rec.hFix) {
        rec.el.style.height = hFix < 0 ? 'auto' : `${hFix}px`;
        rec.hFix = hFix;
      }
      clampBox(P.x - w / 2, P.y - h / 2, w, h, W, H, EDGE_MARGIN, BOX);
      const bx = Math.round(BOX.x);
      const by = Math.round(BOX.y);
      // numbers compared first — the transform string is only built when they moved
      if (bx !== rec.bx || by !== rec.by) {
        rec.el.style.transform = `translate3d(${bx}px,${by}px,0)`;
        rec.bx = bx;
        rec.by = by;
      }
      // declutter against better-priority cards already placed
      let covered = 0;
      for (let b = 0; b < placedC; b++) {
        const c = coverFrac(
          bx, by, w, h,
          cardBoxes[b * 4]!, cardBoxes[b * 4 + 1]!, cardBoxes[b * 4 + 2]!, cardBoxes[b * 4 + 3]!,
        );
        if (c > covered) covered = c;
      }
      if (covered > COVER_MAX && !isOpen && !isPinned) {
        if (rec.opQ !== 0) {
          rec.el.style.opacity = '0';
          rec.opQ = 0;
        }
        continue;
      }
      cardBoxes[placedC * 4] = bx;
      cardBoxes[placedC * 4 + 1] = by;
      cardBoxes[placedC * 4 + 2] = w;
      cardBoxes[placedC * 4 + 3] = h;
      placedC++;
      // opacity quantised to hundredths (what toFixed(2) renders) and compared as a number
      const opQ =
        isOpen || isPinned
          ? 100
          : Math.round((slot.growT >= CARD_FADE_T ? 1 : slot.growT / CARD_FADE_T) * 100);
      if (opQ !== rec.opQ) {
        rec.el.style.opacity = opQ >= 100 ? '1' : (opQ / 100).toFixed(2);
        rec.opQ = opQ;
      }
      // the summary earns its space once the box can hold it
      const st = (slot.growT - SUM_IN_T) / (SUM_FULL_T - SUM_IN_T);
      const sumQ =
        isOpen || isPinned ? 100 : Math.round((st <= 0 ? 0 : st >= 1 ? 1 : st) * 100);
      if (sumQ !== rec.sumQ) {
        rec.sum.style.opacity = sumQ >= 100 ? '1' : (sumQ / 100).toFixed(2);
        rec.sumQ = sumQ;
      }
      // slots are priority-ordered, so priority IS the stacking order; open on top
      const zN = isOpen ? 30 : 20 - s;
      if (zN !== rec.zN) {
        rec.el.style.zIndex = String(zN);
        rec.zN = zN;
      }
    }

    /* -- labels, in state.named order (the ladder publishes it in RANK order, so the greedy
       declutter gives the pixels to the more important label — port: zoom.js walked the
       sorted candidate list) ----------------------------------------------------------------- */
    labelBoxes = ensureBoxes(labelBoxes, state.namedCount);
    let placedL = 0;
    for (let k = 0; k < state.namedCount; k++) {
      const i = state.named[k]!;
      const rec = labels.get(i);
      if (!rec) continue;
      const o = i * 3;
      // a card of its own is a better label than a label — suppress the duplicate
      let show =
        !cards.has(i) && projectPoint(m, pos[o]!, pos[o + 1]!, pos[o + 2]!, W, H, P);
      let x = 0;
      let y = 0;
      if (show) {
        x = P.x + LABEL_DX;
        y = P.y - rec.h / 2;
        if (
          x > W + LABEL_CULL_PX || x + rec.w < -LABEL_CULL_PX ||
          y > H + LABEL_CULL_PX || y + rec.h < -LABEL_CULL_PX
        ) {
          show = false;
        }
      }
      if (show) {
        // greedy declutter: an overlap with an already-placed (earlier-live) label hides it
        for (let b = 0; b < placedL; b++) {
          const bx = labelBoxes[b * 4]!;
          const by = labelBoxes[b * 4 + 1]!;
          const bw = labelBoxes[b * 4 + 2]!;
          const bh = labelBoxes[b * 4 + 3]!;
          if (
            x < bx + bw + LABEL_PAD_X && x + rec.w + LABEL_PAD_X > bx &&
            y < by + bh + LABEL_PAD_Y && y + rec.h + LABEL_PAD_Y > by
          ) {
            show = false;
            break;
          }
        }
      }
      if (show) {
        labelBoxes[placedL * 4] = x;
        labelBoxes[placedL * 4 + 1] = y;
        labelBoxes[placedL * 4 + 2] = rec.w;
        labelBoxes[placedL * 4 + 3] = rec.h;
        placedL++;
        const lx = Math.round(x);
        const ly = Math.round(y);
        if (lx !== rec.bx || ly !== rec.by) {
          rec.el.style.transform = `translate3d(${lx}px,${ly}px,0)`;
          rec.bx = lx;
          rec.by = ly;
        }
        if (!rec.on) {
          rec.el.style.opacity = '1';
          rec.on = true;
        }
      } else if (rec.on) {
        rec.el.style.opacity = '0';
        rec.on = false;
      }
    }
  }

  return {
    update,
    get labelCount(): number {
      return labels.size;
    },
    get cardCount(): number {
      return cards.size;
    },
    dispose(): void {
      labels.clear();
      cards.clear();
      labelPool.length = 0;
      cardPool.length = 0;
      labelLayer.remove();
      cardLayer.remove();
    },
  };
}
