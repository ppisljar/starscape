/* Types for the zoom-ladder state machine (DESIGN.md §6): mote → named → card → lock → open,
 * pure and headless. The ladder owns no DOM, no camera and no clock — the consumer projects
 * candidates (what is in front of the camera and how big it WOULD be, unclamped) and the
 * ladder answers which nodes are named, which hold cards, at what tier, and what changed this
 * tick. Content itself arrives elsewhere (`getCard(i)` / `getBody(i)` on the overlay — stage 7
 * part 2); this machine only decides WHEN.
 */

/** A card's tier. `card` grows with approach, `lock` is the frozen LOCK_W×LOCK_H rect, `open`
 *  is the body resolving inside it. The tiers are nested: open implies locked implies card. */
export type CardTier = 'card' | 'lock' | 'open';

/** The per-tick candidate set, COLUMNAR (§10.2: no per-node objects): parallel arrays, the
 *  first `count` entries valid. The consumer typically fills these from the nodes in front of
 *  the camera (frustum-tested, depth > 0); the ladder applies its own gates on top. */
export interface LadderCandidates {
  /** How many leading entries of the four columns are valid this tick. */
  count: number;
  /** Node index per candidate. */
  node: ArrayLike<number>;
  /** UNCLAMPED projected glyph diameter, CSS px — the node's own apparent size (the effective
   *  world radius the shader draws: size × NODE_SCALE × boost — through `projectedPx`). This
   *  is the glyph/named ladder's progress variable; it is deliberately NOT the card ladder's. */
  projectedPx: ArrayLike<number>;
  /** Camera distance, world units — view-forward DEPTH (clip w ≡ −z_view), the divisor the
   *  field shader sizes glyphs by, not the Euclidean range (which over-measures off-axis
   *  nodes by 1/cosθ). The card ladder's progress variable, compared against multiples of
   *  `spacing` — a fixed world radius, never the node's own size. */
  distance: ArrayLike<number>;
  /** Screen distance from the cursor to the node, CSS px. Pass Infinity (or anything ≥
   *  CURSOR_RADIUS_PX) when there is no cursor — scores are then unpulled. */
  cursorPx: ArrayLike<number>;
}

/** Everything the ladder sees in one tick. Nothing is retained across calls. */
export interface LadderView {
  candidates: LadderCandidates;
  /** The measured page spacing, world units — the unit every card-tier threshold is in. */
  spacing: number;
  /** Pinned (selected) node, -1 for none. A pin bypasses the candidate filter, the card
   *  budget, the open threshold and (in the overlay) the declutter: it reports `open` even
   *  when the node is not in `candidates` at all. */
  pinned: number;
  /** True when the cursor moved since the last update — this is what clears a retreat()'s
   *  auto-open suspension. */
  cursorMoved: boolean;
}

/** One card slot. Slot objects are pooled and REUSED across ticks — copy, never retain. */
export interface CardSlot {
  node: number;
  tier: CardTier;
  /** 0 at the card threshold → 1 at the lock, linear in projected (inverse-distance) size —
   *  the box grows in PIXELS with approach, then freezes. Always 1 for lock/open/pinned. */
  growT: number;
}

/** Enter/exit node lists for one tier this tick. Fixed buffers, reused — the first
 *  `enterCount`/`exitCount` entries are valid, the rest stale. */
export interface TierEvents {
  enter: Uint32Array;
  enterCount: number;
  exit: Uint32Array;
  exitCount: number;
}

/** What changed this tick, per tier, for the overlay to mount/unmount. Transitions are
 *  reported per boundary crossed: a card mounting straight at `open` (a pin) fires card.enter,
 *  lock.enter AND open.enter; an open card unmounting fires all three exits. */
export interface LadderEvents {
  named: TierEvents;
  card: TierEvents;
  lock: TierEvents;
  open: TierEvents;
}

/** The ladder's answer. One object, reused every tick (zero per-update allocation) — read it
 *  before the next update or copy what you keep. */
export interface LadderState {
  /** Live (named) node indices: the first `namedCount` entries of this reused array, in RANK
   *  order (score descending; nodes absent from this tick's candidates last) — the overlay
   *  declutters greedily in this order, so "already placed" means "more important". */
  named: Uint32Array;
  namedCount: number;
  /** Cards in priority order (pinned first, then rank order), ≤ CARDS + 1 (pin rides on top
   *  of the budget). Pooled objects — see CardSlot. */
  cards: CardSlot[];
  /** The node holding the single open slot, -1 for none. */
  open: number;
  /** True while auto-open is suspended by retreat() — cleared only by real user intent: a
   *  reported cursor move, or a pin. */
  suspended: boolean;
  events: LadderEvents;
}

/** Every constant in src/ladder/constants.ts, overridable per instance. Units unchanged:
 *  *_Px / *Ms / lockW / lockH are CSS px / milliseconds; *Spacings are page-spacings. */
export interface LadderOptions {
  namedPx?: number;
  earn?: number;
  keep?: number;
  dwellMs?: number;
  namedCap?: number;
  cardSpacings?: number;
  lockSpacings?: number;
  openSpacings?: number;
  cards?: number;
  openHoldMs?: number;
  lockW?: number;
  lockH?: number;
  cardRadiusSpacings?: number;
  cursorRadiusPx?: number;
  cursorPull?: number;
  /* ---- per-tier master switches (default on). Each SUPPRESSES one whole tier — a true OFF the
   * tier thresholds alone cannot reach (a threshold only moves WHERE a tier fires, never whether
   * it fires at all). They are read live every tick, so flipping one off drops that tier's set
   * the same frame, emitting the exits the overlay needs to unmount. The tiers nest, so the
   * switches compose: cards off implies no open (no box to open), open off leaves the box but
   * never resolves the body. */
  /** When false the named (label) tier emits nothing — no node ever earns a label, and any live
   *  one is dropped (exited) this tick. */
  showLabels?: boolean;
  /** When false the card budget is 0: no card/lock slot is produced, not even a pinned node's,
   *  and (the tiers nesting) nothing opens. */
  showCards?: boolean;
  /** When false the open tier never fires: auto-open is inert and a pin locks its card but never
   *  resolves the body. Leaves the card/lock tiers untouched. */
  showOpen?: boolean;
}

export interface Ladder {
  /** Advance the machine by `dt` SECONDS (the library-wide unit — tween and flycam agree) and
   *  evaluate this tick's view. Returns the reused LadderState. No clock is ever read: a fixed
   *  (dt, view) sequence replays byte-identically (§10.1). */
  update(dt: number, view: LadderView): LadderState;
  /** Escape. Closes the open slot NOW (the ladder transition then runs in reverse as the
   *  consumer's camera retreats — the tiers are pure functions of distance) and suspends
   *  auto-open, so the retreat cannot fly past another page and have it instantly claim the
   *  slot — which would make the gesture appear to OPEN something. The suspension clears
   *  only on real user intent: a reported cursor move, or a pin (the port re-enabled on
   *  flycam input / flyTo / pin, never because some other page drifted into range). */
  retreat(): void;
  /** The same object `update` returns. */
  readonly state: LadderState;
  /** The resolved options (defaults filled in) — the overlay reads lockW/lockH here. */
  readonly config: Required<LadderOptions>;
}
