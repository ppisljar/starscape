/* Palettes and ramps as data (DESIGN.md §7: "Palettes/ramps ship as data (including
 * colorblind-safe sets); facet → categorical palette, numeric column → ramp").
 *
 * Two different kinds of colour table, deliberately kept apart (the predecessor's rule):
 *   - a RAMP maps a NUMBER to a colour — stops interpolated, order is meaning;
 *   - a PALETTE maps a CATEGORY to a colour — discrete entries, a ramp would imply an
 *     ordering the values do not have.
 *
 * Everything here is pure data + pure functions: no three.js import, headless-testable, and
 * the binding helpers produce exactly the flat buffers the field already consumes — a 3·k
 * palette buffer for paintFacetColors, or a 3·n per-node colour buffer written in a loop
 * (§10: no per-node objects, no spreads over node-length arrays).
 *
 * RAMPS and rampAt are ported from vault-site-3d src/config.js, values verbatim.
 */

import { rankPercentiles } from '../field/channels';
import { paintFacetColors } from '../field/palette';

/** One colour: RGB in [0, 1]. */
export type RGB = readonly [number, number, number];
/** A ramp: ordered stops, interpolated in linear RGB. */
export type RampStops = ReadonlyArray<RGB>;
/** A categorical palette: discrete entries, value v → entry v (wrapping). */
export type Palette = ReadonlyArray<RGB>;

/* ---- ramps (numeric column → colour) ----------------------------------------------------- */

/* Predecessor notes carried with the data:
 *   aurora — NO white stop. In the reference image the white is not a HUE, it is where
 *   additive blending piles up and the hot cores burn out — a luminance effect. A white stop
 *   spent a third of the palette on desaturated nodes (measured: 53% of the field in one hue
 *   bucket). The ramp stays saturated; brightness does the white.
 *   vault — the original site's type colours, from its CSS custom properties (earthy and
 *   desaturated on purpose: chosen to sit on paper, where the saturated ramps turn garish). */
export const RAMPS = {
  aurora: [
    [0.05, 0.26, 0.4],
    [0.08, 0.68, 0.66],
    [0.33, 0.45, 0.92],
    [0.8, 0.26, 0.74],
    [1.0, 0.72, 0.28],
  ],
  ice: [
    [0.02, 0.06, 0.16],
    [0.13, 0.36, 0.66],
    [0.55, 0.8, 0.98],
    [0.95, 0.99, 1.0],
  ],
  ember: [
    [0.1, 0.02, 0.04],
    [0.56, 0.1, 0.1],
    [0.95, 0.45, 0.1],
    [1.0, 0.9, 0.55],
  ],
  /* cool→hot by degree — the nebula look's defining ramp (§7 "hot-by-degree"). Low-degree
   * leaves are a cool blue, high-degree hubs a hot orange that, where they crowd and SUM under
   * additive blending, bloom and tonemap into the white-orange dense core of the reference. The
   * stops are chosen so the BULK of the field (low ranks) reads blue and only the rare high-rank
   * hubs go warm — the rank-normalised channel (rampColors) already pushes the mass toward 0. */
  nebula: [
    [0.16, 0.36, 0.86],
    [0.26, 0.56, 0.98],
    [0.62, 0.68, 0.95],
    [0.98, 0.56, 0.22],
    [1.0, 0.86, 0.58],
  ],
  spectral: [
    [0.23, 0.07, 0.4],
    [0.13, 0.42, 0.66],
    [0.2, 0.7, 0.52],
    [0.86, 0.84, 0.28],
    [0.9, 0.33, 0.24],
  ],
  mono: [
    [0.08, 0.09, 0.12],
    [0.55, 0.58, 0.64],
    [1.0, 1.0, 1.0],
  ],
  /* viridis-like: perceptually-ordered dark-purple → teal → yellow, colorblind-safe in the
   * sense that lightness carries the order (stops are the usual viridis anchor colours). */
  viridis: [
    [0.267, 0.005, 0.329],
    [0.283, 0.141, 0.458],
    [0.254, 0.265, 0.53],
    [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.15],
    [0.993, 0.906, 0.144],
  ],
  vault: [
    [0.122, 0.42, 0.322],
    [0.541, 0.42, 0.137],
    [0.247, 0.373, 0.525],
    [0.635, 0.306, 0.212],
    [0.42, 0.298, 0.502],
    [0.49, 0.42, 0.365],
  ],
} as const satisfies Record<string, RampStops>;

export type RampName = keyof typeof RAMPS;
export const RAMP_NAMES = Object.keys(RAMPS) as RampName[];

/** The shipped default ramp — the ONE name the config schema's ramp row, the Field's
 *  setColorColumn default argument and the unknown-name fallback below all reference (§10:
 *  constants exported once; three independent 'aurora' strings were three chances for a
 *  look exported on one host to replay differently on another, with nothing throwing). */
export const DEFAULT_RAMP: RampName = 'aurora';

/** A ramp argument: a shipped name or your own stops — "many palettes, trivially extensible". */
export type RampLike = RampName | RampStops;

const rampStops = (ramp: RampLike): RampStops =>
  // predecessor rampAt behaviour: an unknown name falls back to the default, not a throw
  typeof ramp === 'string'
    ? ((RAMPS as Record<string, RampStops>)[ramp] ?? RAMPS[DEFAULT_RAMP])
    : ramp;

/** Sample a ramp at t ∈ [0, 1] (predecessor's rampAt, logic preserved: t clamps into
 *  [0, 0.9999] so the segment index never overruns the last pair). */
export function rampAt(ramp: RampLike, t: number): [number, number, number] {
  const r = rampStops(ramp);
  const x = Math.min(0.9999, Math.max(0, t)) * (r.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = r[i]!;
  const b = r[Math.min(r.length - 1, i + 1)]!;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** The `uniform` colour mode's one flat hue (config schema COLOR_MODES; blueprint's "the
 *  single hue IS the look" — theme/presets.ts). Shipped as data (§7) so the named presets
 *  that lean on it render the same in every host, instead of each host inventing its own
 *  blue and "blueprint" meaning a different picture per consumer. */
export const UNIFORM_RGB: RGB = [0.42, 0.62, 0.95];

/* ---- categorical palettes (facet → colour) ------------------------------------------------ */

const hx = (h: number): RGB => [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255];

/* The predecessor's layer palette (vault-site-3d config.js VAULT_TYPE_COLOR), in its own
 * order: source, note, person, system, synthesis, term, other. The correspondence to the
 * original site is by FOLDER, not by list position — reproducing it on another corpus means
 * matching value NAMES (see namedPaletteBuffer), never walking values through the list. */
export const VAULT_LAYER_NAMES = [
  'source',
  'note',
  'person',
  'system',
  'synthesis',
  'term',
  'other',
] as const;

export const PALETTES = {
  vaultLayer: [
    [0.122, 0.42, 0.322], // source    (the original's guide green,   --c-guide  #1F6B52)
    [0.541, 0.42, 0.137], // note      (course gold,                  --c-course #8A6B23)
    [0.247, 0.373, 0.525], // person   (handbook blue,                --c-handbook #3F5F86)
    [0.42, 0.298, 0.502], // system    (tool purple,                  --c-tool   #6B4C80)
    [0.49, 0.42, 0.365], // synthesis  (res taupe,                    --c-res    #7D6B5D)
    [0.635, 0.306, 0.212], // term     (term rust,                    --c-term   #A24E36)
    [0.082, 0.126, 0.106], // other    (root ink — no original group corresponds)
  ],
  /* Okabe & Ito's 8-colour set (colorblindness.org / Wong, Nature Methods 8:441, 2011) —
   * distinguishable under protanopia, deuteranopia and tritanopia. Values are the canonical
   * sRGB bytes: black #000000, orange #E69F00, sky blue #56B4E9, bluish green #009E73,
   * yellow #F0E442, blue #0072B2, vermillion #D55E00, reddish purple #CC79A7. */
  okabeIto: [
    hx(0x000000),
    hx(0xe69f00),
    hx(0x56b4e9),
    hx(0x009e73),
    hx(0xf0e442),
    hx(0x0072b2),
    hx(0xd55e00),
    hx(0xcc79a7),
  ],
  /* Paul Tol's "bright" qualitative scheme (personal.sron.nl/~pault, 2021), colorblind-safe:
   * blue #4477AA, cyan #66CCEE, green #228833, yellow #CCBB44, red #EE6677, purple #AA3377,
   * grey #BBBBBB. */
  tolBright: [
    hx(0x4477aa),
    hx(0x66ccee),
    hx(0x228833),
    hx(0xccbb44),
    hx(0xee6677),
    hx(0xaa3377),
    hx(0xbbbbbb),
  ],
  /* Paul Tol's "muted" qualitative scheme (same source), colorblind-safe, 9 colours + the
   * pale grey he reserves for "bad data": indigo #332288, cyan #88CCEE, teal #44AA99,
   * green #117733, olive #999933, sand #DDCC77, rose #CC6677, wine #882255, purple #AA4499,
   * pale grey #DDDDDD. */
  tolMuted: [
    hx(0x332288),
    hx(0x88ccee),
    hx(0x44aa99),
    hx(0x117733),
    hx(0x999933),
    hx(0xddcc77),
    hx(0xcc6677),
    hx(0x882255),
    hx(0xaa4499),
    hx(0xdddddd),
  ],
} as const satisfies Record<string, Palette>;

export type PaletteName = keyof typeof PALETTES;
export const PALETTE_NAMES = Object.keys(PALETTES) as PaletteName[];

/* ---- binding helpers: the flat buffers the field consumes -------------------------------- */

/** Flatten a palette for a facet with `count` values: value v's colour at [3v .. 3v+2] —
 *  the exact shape field/palette.ts paintFacetColors (and setColorFacet behind it) consumes.
 *  More values than entries wraps (entry v % length): honest repetition beats inventing
 *  colours a hand-picked set never contained. */
export function paletteBuffer(palette: Palette, count: number): Float32Array {
  const k = palette.length;
  if (k === 0) throw new Error('paletteBuffer: empty palette');
  const out = new Float32Array(count * 3);
  for (let v = 0; v < count; v++) {
    const c = palette[v % k]!;
    out[v * 3] = c[0];
    out[v * 3 + 1] = c[1];
    out[v * 3 + 2] = c[2];
  }
  return out;
}

/** Flatten a palette keyed by VALUE NAME: facet value `values[v]` takes `byName[values[v]]`,
 *  or `fallback` when the name is unknown. This is the predecessor's hard-won correspondence
 *  rule: its 'by layer' mode once walked facet values through a ramp by POSITION, which put
 *  `system` on the rust stop and `synthesis` on the purple one — both wrong even though every
 *  stop was used. Names, not positions. */
export function namedPaletteBuffer(
  values: readonly string[],
  byName: Readonly<Record<string, RGB>>,
  fallback: RGB,
): Float32Array {
  const out = new Float32Array(values.length * 3);
  for (let v = 0; v < values.length; v++) {
    const c = byName[values[v]!] ?? fallback;
    out[v * 3] = c[0];
    out[v * 3 + 1] = c[1];
    out[v * 3 + 2] = c[2];
  }
  return out;
}

/** Facet → palette binding: per-node 3n colours from a facet's codes through a palette.
 *  (paintFacetColors with the palette flattened for you.) */
export function facetColors(
  codes: ArrayLike<number>,
  n: number,
  valueCount: number,
  palette: Palette,
  out?: Float32Array,
): Float32Array {
  return paintFacetColors(codes, n, paletteBuffer(palette, valueCount), out);
}

/** Column → ramp binding: per-node 3n colours, RANK-normalised through the ramp — the same
 *  percentile machinery the size channel uses (field/channels.ts rankPercentiles), for the
 *  same reason: real columns span orders of magnitude, so a linear map leaves almost every
 *  node on the ramp's floor while rank sweeps the whole ramp whatever the distribution. */
export function rampColors(
  data: ArrayLike<number>,
  n: number,
  ramp: RampLike,
  out?: Float32Array,
): Float32Array {
  const colors = out ?? new Float32Array(n * 3);
  const pct = rankPercentiles(data, n);
  const stops = rampStops(ramp);
  for (let i = 0; i < n; i++) {
    // rampAt inlined (no per-node tuple allocation — §10.2)
    const x = Math.min(0.9999, Math.max(0, pct[i]!)) * (stops.length - 1);
    const s = Math.floor(x);
    const f = x - s;
    const a = stops[s]!;
    const b = stops[Math.min(stops.length - 1, s + 1)]!;
    colors[i * 3] = a[0] + (b[0] - a[0]) * f;
    colors[i * 3 + 1] = a[1] + (b[1] - a[1]) * f;
    colors[i * 3 + 2] = a[2] + (b[2] - a[2]) * f;
  }
  return colors;
}
