/* Boot quality (DESIGN.md §5: "boot quality chosen from corpus size — no feedback loop").
 *
 * The one thing known before anything draws is how many nodes there are, and the predecessor's
 * SCALE.md already measured where each decade lands — so the starting notch is a pure function
 * of (n, devicePixelRatio), decided once at boot. NO GOVERNOR: the predecessor's adaptive
 * version read a 500 ms fps average and adjusted the link budget, and the adjustment rewrote
 * the measurement — it oscillated at ~1 Hz and read as the links blinking. A boot-time choice
 * cannot oscillate, and the user overrides it by dragging the same config keys (`linkBudget`,
 * `renderScale`, `flatPx` — the schema's SESSION_KEYS: re-derived every boot, never persisted,
 * because a budget tuned for 10⁶ pages is simply wrong for 10³ of them).
 *
 * The notch ladder is the predecessor's, ported (vault-site-3d main.js LADDER/startingStep,
 * summarised in its SCALE.md) and then RE-MEASURED on this renderer (starscape SCALE.md):
 * everything up to 10⁴ (measured: that decade is free), half the links at 10⁵ (they were 55
 * of its 73 ms, and the half that goes is the long cross-group tail the shader already
 * fades), and at 10⁶ 1:1 device pixels, a raised flat-glyph threshold and 4% of the links —
 * tighter than the predecessor's 8-to-16% because a measured link here costs ~100 ns/frame in
 * a dense arrangement (half fragment fill, half the vertex-pull fetches), and 10⁶ spinning
 * galaxies only clear the §9 30 fps floor below ~200 k drawn links. flatPx must RISE with the
 * corpus because the px clamp pins a distant field at NODE_MIN_PX (2.2 px) — above the
 * shipped FLAT_PX of 2, so the cheap flat branch never fires until the threshold is lifted
 * over the clamp floor. At 10⁶ the halo is most of the node fill and none of the information.
 *
 * Pure and headless — the tests pin the policy; the demo applies it to the config store.
 */

import { FLAT_PX } from './constants';

export interface BootQuality {
  /** Fraction of the importance-ordered link buffer to draw (the `linkBudget` config key). */
  linkFraction: number;
  /** Multiplier on devicePixelRatio (the `renderScale` config key). 1 = native. */
  renderScale: number;
  /** Flat-glyph threshold, CSS px (the `flatPx` config key) — glyphs projected under this
   *  draw the cheap no-halo branch. */
  flatPx: number;
}

/** The notch table, smallest corpus first. Thresholds are the predecessor's startingStep
 *  boundaries; each row restates what its SCALE.md measured that decade to need. */
const NOTCHES: ReadonlyArray<{ atLeast: number; q: Omit<BootQuality, 'renderScale'> & { dpr1: boolean } }> = [
  { atLeast: 0, q: { linkFraction: 1, flatPx: FLAT_PX, dpr1: false } },
  { atLeast: 50_000, q: { linkFraction: 0.5, flatPx: 4, dpr1: false } },
  { atLeast: 300_000, q: { linkFraction: 1 / 6, flatPx: 7, dpr1: false } },
  { atLeast: 1_000_000, q: { linkFraction: 0.04, flatPx: 9, dpr1: true } },
];

/** Floor on the renderScale multiplier — the schema row's own rail (config/schema.ts). A
 *  dpr-3 display asking for 1:1 would need 0.33; clamping to the rail keeps the stored value
 *  one the panel can represent. */
export const RENDER_SCALE_FLOOR = 0.5;

/** The starting quality for a corpus of `n` nodes on a `devicePixelRatio`-`dpr` display.
 *  Deterministic, no clock, no feedback — callers apply it once at boot as session DEFAULTS
 *  the user overrides through the very same config keys. */
export function bootQuality(n: number, dpr: number): BootQuality {
  let pick = NOTCHES[0]!.q;
  for (const notch of NOTCHES) {
    if (n >= notch.atLeast) pick = notch.q;
  }
  const renderScale = pick.dpr1 ? Math.max(RENDER_SCALE_FLOOR, 1 / Math.max(dpr, 1)) : 1;
  return { linkFraction: pick.linkFraction, renderScale, flatPx: pick.flatPx };
}
