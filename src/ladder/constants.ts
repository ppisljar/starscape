/* The zoom ladder's numbers (DESIGN.md §6) — mote → named → card → lock → open — in ONE module,
 * exported once (§5: one pixel unit; §10: constants exported once). Every value is either CSS
 * pixels or page-SPACINGS, never world units and never device pixels: "a card appears about two
 * pages away" survives a change of corpus size, field scale, viewport or field-of-view, where
 * "a card appears at 54 px" does not (the predecessor's first pixel-threshold version put cards
 * at 1,179 world units — four page-spacings out, open while you were still outside the region).
 *
 * All of these are defaults, overridable per-instance via LadderOptions (the stage-8 config
 * schema will put a knob on each). The numbers are the predecessor's shipped defaults
 * (vault-site-3d src/config.js), argued over on a real corpus.
 */

/** The named tier's glyph gate: a node may earn a label once its UNCLAMPED projected diameter
 *  reaches this many CSS px. This is also the ladder's candidate filter — below it a node is a
 *  mote and nothing more — and the one gate a pin must bypass (see README "A pin outranks the
 *  ladder": framing a neighbourhood can put the selected node's own glyph under this floor). */
export const NAMED_PX = 13;

/** Label admission is a RANK BAND, not a threshold: in the dense core a hundred nodes cross any
 *  absolute pixel gate at the same moment, while rank is density-independent. A label is EARNED
 *  the tick its node ranks inside the top EARN candidates (by score) — mounting is immediate,
 *  the port's semantics: names appear the frame glyphs cross the gate… */
export const EARN = 20;

/** …and KEPT until it falls outside the top KEEP. EARN < KEEP is the hysteresis: a label earned
 *  at rank 19 survives drifting to rank 31, and a fresh node at rank 25 never earns. The band
 *  and the dwell measured 0 label re-entries per second over 3.9 s of flight. */
export const KEEP = 32;

/** Minimum label LIFETIME, milliseconds — dt-accumulated, never wall-clock (§10.1). A freshly
 *  mounted label is held this long whatever its rank; past the grace it drops the tick it sits
 *  outside the keep band. This is the predecessor's shipped semantics (zoom.js: mount on
 *  entering the band, `since` + labelDwell before an out-of-band drop) — NOT an entry dwell,
 *  which would name every approach 600 ms late and a fast flyby not at all. */
export const DWELL_MS = 600;

/** Hard cap on simultaneously named nodes, whatever the band admits. */
export const NAMED_CAP = 34;

/** The card tier fires at this many page-spacings of camera distance. */
export const CARD_SPACINGS = 2.2;

/** The lock tier: the card rect stops growing and freezes at LOCK_W × LOCK_H. */
export const LOCK_SPACINGS = 1.0;

/** The open tier: the body resolves inside the locked rect. */
export const OPEN_SPACINGS = 0.45;

/** Card budget: at most this many ladder-driven cards at once. A pinned node's card rides ON
 *  TOP of this budget (≤ CARDS + 1 slots total) — a pin outranks every budget. */
export const CARDS = 5;

/** Open budget. There is exactly one open slot; the constant exists so the budget is stated,
 *  not implied (§10.4: every budget is reported, none is silent). */
export const OPEN_BUDGET = 1;

/** The open slot cannot transfer to another node, or auto-close, until it has been held this
 *  long (dt-accumulated). Opening into an EMPTY slot is immediate — the hold exists so two
 *  pages straddling the open threshold cannot fight over the slot every frame (the
 *  predecessor's MIN_OPEN_MS). */
export const OPEN_HOLD_MS = 600;

/** The locked card rect, CSS px. Fixed by design: past the lock the box stops growing and the
 *  INTERIOR resolves — type is never scaled, more content simply fits. */
export const LOCK_W = 440;
export const LOCK_H = 320;

/** The card ladder's FIXED world radius, as a fraction of the page spacing (predecessor:
 *  CARD_R = 60 on a 192-spacing vault). Two progress variables, not one: the glyph/named ladder
 *  keys off the node's OWN apparent size (importance buys reach), but a card is the same object
 *  whatever page is inside it — node radii span 12× on a real corpus, and keying cards to node
 *  size put leaf pages' open distance inside the camera's own soft stop, so they could never
 *  open at all. Every page opens at the same distance, by design. */
export const CARD_R_SPACINGS = 0.3125;

/** Attention follows the CURSOR, not the screen centre (centre-gating is a VR idiom that stops
 *  you opening what you point at): a candidate's score is boosted by up to CURSOR_PULL as the
 *  pointer approaches, fading linearly to nothing at CURSOR_RADIUS_PX away. */
export const CURSOR_RADIUS_PX = 260;
export const CURSOR_PULL = 0.9;

/** Floor on any distance fed to the projection — the same floor the field's glyphPx applies,
 *  so the ladder and the shader can never disagree about a node on the camera's nose. */
export const MIN_DISTANCE = 0.001;
