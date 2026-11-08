/* Facet → palette, as flat data (DESIGN.md §5 tier 1: themes are pure data). A palette is one
 * Float32Array of 3·k RGB floats in [0, 1] — the predecessor's `pal` shape — never an array of
 * per-value objects. Pure functions, headless-testable, no three.js import.
 */

/** The golden-ratio hue step: adjacent facet values land on hues that never neighbour each
 *  other, however many values the facet has (the predecessor's GOLDEN constant). */
export const GOLDEN = 0.618033988749895;

const PALETTE_S = 0.72;
const PALETTE_L = 0.58;

/** HSL → RGB, all components in [0, 1]. The CSS/three.js convention; pure so tests can pin it. */
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const hue = ((h % 1) + 1) % 1;
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const channel = (t0: number): number => {
    const t = ((t0 % 1) + 1) % 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * 6 * (2 / 3 - t);
    return p;
  };
  return [channel(hue + 1 / 3), channel(hue), channel(hue - 1 / 3)];
}

/** Build the categorical palette for a facet with `count` values: flat RGB triplets, value v's
 *  colour at [v*3 .. v*3+2]. Golden-angle hues at fixed saturation/lightness — deterministic,
 *  unbounded in k, and guaranteed-distinct neighbours. (Named per-corpus palettes are theme
 *  data and arrive with the theming system at stage 8; this is the basics.) */
export function facetPalette(count: number): Float32Array {
  const pal = new Float32Array(count * 3);
  for (let v = 0; v < count; v++) {
    const [r, g, b] = hslToRgb(v * GOLDEN, PALETTE_S, PALETTE_L);
    pal[v * 3] = r;
    pal[v * 3 + 1] = g;
    pal[v * 3 + 2] = b;
  }
  return pal;
}

/** Paint per-node colours from a facet's codes through a palette, into (or as) a 3n buffer.
 *  A loop, one write per channel — no per-node allocation. */
export function paintFacetColors(
  codes: ArrayLike<number>,
  n: number,
  pal: Float32Array,
  out?: Float32Array,
): Float32Array {
  const colors = out ?? new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const v = codes[i]! * 3;
    colors[i * 3] = pal[v]!;
    colors[i * 3 + 1] = pal[v + 1]!;
    colors[i * 3 + 2] = pal[v + 2]!;
  }
  return colors;
}
