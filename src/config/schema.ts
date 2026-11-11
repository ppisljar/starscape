/* The config schema (DESIGN.md §7): every tunable is a row — group, key, label, kind, range or
 * options, default, unit — and everything else derives from the rows. The panel renders them,
 * DEFAULTS falls out of them, presets and looks diff against them, persistence validates
 * against them. One source; the predecessor proved the engine (vault-site-3d src/config.js),
 * whose MACHINERY is ported in ./store — the CONTENTS below are designed fresh for what this
 * library actually has today (§7: nothing copied wholesale).
 *
 * Two conventions, both §7 contract:
 *   - every DISTANCE is in page-spacings or CSS px, never raw world units, so a number keeps
 *     meaning the same thing when the corpus or the field scale changes. `'world'` exists in
 *     the Unit type exactly so a test can assert no row uses it.
 *   - a row's default REFERENCES the constant that module already exports (field/constants,
 *     ladder/constants, focus, flycam) — never a restated copy of the number. The constants
 *     modules stay the single source of truth (§10: constants exported once); the schema is a
 *     knob on each, not a second home for its value.
 */

import {
  BODY_R,
  CORE_HOT,
  FALLOFF,
  FLAT_PX,
  HALO_GAIN,
  HALO_PX,
  LINK_BASE_ALPHA,
  LINK_LEN_FLOOR,
  LINK_LEN_SOFT,
  NODE_MIN_PX,
  NODE_SCALE,
  SIZE_CURVE,
  SPHERE,
} from '../field/constants';
import {
  DIM_OTHERS,
  LINK_DIM_ALPHA,
  LINK_FOCUS_ALPHA,
  LINK_HOVER_ALPHA,
} from '../field/focus';
import {
  CARDS,
  CARD_R_SPACINGS,
  CARD_SPACINGS,
  DWELL_MS,
  EARN,
  KEEP,
  LOCK_H,
  LOCK_SPACINGS,
  LOCK_W,
  NAMED_CAP,
  NAMED_PX,
  OPEN_SPACINGS,
} from '../ladder/constants';
import { FLYCAM_DEFAULTS } from '../cameras/flycam';
import { EXPOSURE_DEFAULT, TONEMAP_OPS } from '../field/tonemap';
import { DEFAULT_RAMP, PALETTE_NAMES, RAMP_NAMES } from '../theme/palettes';
import { VISUAL_PRESETS } from '../theme/presets';

/** What a row's number is measured in. `'world'` is in the type ONLY so tests can assert it
 *  never appears in the schema (§7: distances are spacings or px — world units do not survive
 *  a change of corpus). `'none'` covers dimensionless knobs: fractions, multipliers, ranks,
 *  alphas, rates and times. */
export type Unit = 'world' | 'spacings' | 'px' | 'none';

/** A config value: ranges are numbers, selects are option strings, toggles booleans. */
export type ConfigValue = number | string | boolean;

/** The live config: key → value, defaults overlaid by whatever was touched. */
export type Config = Record<string, ConfigValue>;

interface RowBase {
  /** Panel section — purely presentational grouping. */
  group: string;
  /** The config key. Unique across the schema (asserted by deriveDefaults). */
  key: string;
  /** Human label the panel shows. */
  label: string;
  unit: Unit;
}

export interface RangeRow extends RowBase {
  kind: 'range';
  min: number;
  max: number;
  step: number;
  default: number;
  /** Hard DOMAIN bound (exclusive), distinct from the rail: values ≤ floor are invalid
   *  everywhere, because the owning setter throws on them (Field.setTonemap requires
   *  exposure > 0, Field.setSpread requires spread > 0) — a pasted look that stored one
   *  would poison persistence and crash every subsequent boot. Most rows have none:
   *  min/max stay panel rails a deliberate paste may overshoot (store.ts validValue). */
  floor?: number;
  /** OPTIONAL-CAP rows: at or above this value the control is "off" (no cap) rather than a
   *  number — the panel readout prints "off". The slider's TOP detent is this value (= the
   *  shipped default), so the knob reads as an optional cap you drag DOWN to impose; the
   *  consumer maps the off state to its subsystem's uncapped sentinel (nodeMaxPx → the node
   *  shader's NODE_MAX_PX_OFF). Omitted on ordinary rows, which always print a number. */
  offAt?: number;
}

export interface SelectRow extends RowBase {
  kind: 'select';
  options: readonly string[];
  default: string;
}

export interface ToggleRow extends RowBase {
  kind: 'toggle';
  default: boolean;
}

export type SchemaRow = RangeRow | SelectRow | ToggleRow;

/** One blending setting for both layers (field.ts setBlend). */
export const BLENDS = ['normal', 'additive'] as const;

/** Highlight roll-off curves — field/tonemap.ts TONEMAP_OPS, THE list, not a restatement:
 *  the drift between a name and its shader index is the predecessor's silent-no-op bug, so
 *  the operator module stays the one source (same derive pattern as PALETTES below). */
export const TONEMAPS = TONEMAP_OPS;

/** Node colour source: a facet's categorical palette ('facet'); a numeric column
 *  rank-normalised through the `ramp` row ('column' — the degree column, the one buildGraph
 *  guarantees; a panel extends the choice from the live graph, exactly as SIZE_CHANNELS does);
 *  one flat colour ('uniform'); or LOCAL SPATIAL DENSITY ('density' — a geometric channel the
 *  field computes from the current positions, field/density.ts, rank-normalised through the
 *  `ramp` the same way 'column' is: crowded neighbourhoods hot, sparse ones cool — the nebula
 *  look's defining signal, and the one 'column' by degree cannot paint under additive blending). */
export const COLOR_MODES = ['facet', 'column', 'uniform', 'density'] as const;

/** Categorical palettes: 'golden' is the golden-angle generator (field/palette.ts); the named
 *  sets are theme data (theme/palettes.ts — the predecessor layer palette and the
 *  colorblind-safe sets), listed from their one source so the two lists can never drift. */
export const PALETTES: readonly string[] = ['golden', ...PALETTE_NAMES];

/** The ground. 'dark' is BG_DARK, 'light' BG_LIGHT, 'black' BG_BLACK (all field/constants — the
 *  named grounds must mean the same colour in every host); blending follows the ground (§5). 'black'
 *  is the pure-#000 ground the additive nebula look needs so the negative space between clusters
 *  reads truly black (BG_DARK's near-black lifts to grey under the glow). */
export const BACKGROUNDS = ['dark', 'light', 'black'] as const;

/** Node size channel the field can bind without corpus knowledge: the degree column that
 *  buildGraph guarantees, or equal dots. (Other numeric columns are corpus-specific; a panel
 *  extends this list from the live graph, the schema ships what always exists.) */
export const SIZE_CHANNELS = ['degree', 'uniform'] as const;

/** WHICH facet colours the nodes when colorMode is 'facet' (Field.setColorFacet). Facets are
 *  corpus-specific, so the schema ships only 'type' (the facet nearly every corpus carries,
 *  and today's hardcoded default) and a panel WIDENS the options from the live graph's facets —
 *  exactly the SIZE_CHANNELS extension point. Inert in every other colour mode. */
export const COLOR_FACETS = ['type'] as const;

/** WHICH facet binds node GLYPH SHAPE (Field.setShapeFacet): 'none' (every node a disc) plus
 *  the facets a panel widens in from the live graph, same pattern as COLOR_FACETS. 'none' is
 *  the only corpus-agnostic option, so it is the shipped default. */
export const SHAPE_FACETS = ['none'] as const;

/** The schema. Order is panel order; rows share a group by adjacency. */
export const SCHEMA: readonly SchemaRow[] = [
  /* ---- the zoom ladder (§6): where each tier fires. Distances in page-spacings. ----------
     The two master switches first: a tier threshold only moves WHERE a tier fires, never whether
     it fires at all, so a true OFF needs its own toggle (ladder honours showCards/showOpen). */
  { group: 'ladder', key: 'showCards', label: 'cards', kind: 'toggle', default: true, unit: 'none' },
  { group: 'ladder', key: 'showOpen', label: 'full page view', kind: 'toggle', default: true, unit: 'none' },
  { group: 'ladder', key: 'tierCard', label: 'card appears', kind: 'range', min: 0.5, max: 12, step: 0.1, default: CARD_SPACINGS, unit: 'spacings' },
  { group: 'ladder', key: 'tierLock', label: 'box locks', kind: 'range', min: 0.2, max: 6, step: 0.05, default: LOCK_SPACINGS, unit: 'spacings' },
  { group: 'ladder', key: 'tierOpen', label: 'body opens', kind: 'range', min: 0.1, max: 3, step: 0.05, default: OPEN_SPACINGS, unit: 'spacings' },
  { group: 'ladder', key: 'cardRadius', label: 'card world radius', kind: 'range', min: 0.05, max: 2, step: 0.0125, default: CARD_R_SPACINGS, unit: 'spacings' },
  { group: 'ladder', key: 'maxCards', label: 'cards at once', kind: 'range', min: 1, max: 12, step: 1, default: CARDS, unit: 'none' },
  { group: 'ladder', key: 'lockW', label: 'locked width', kind: 'range', min: 220, max: 760, step: 10, default: LOCK_W, unit: 'px' },
  { group: 'ladder', key: 'lockH', label: 'locked height', kind: 'range', min: 160, max: 640, step: 10, default: LOCK_H, unit: 'px' },

  /* ---- the label band: earn/keep hysteresis, cap, dwell (ladder/constants). --------------
     The master switch first — off suppresses the whole tier, below any px gate (ladder honours
     showLabels: no node ever earns a label, live ones are dropped). */
  { group: 'labels', key: 'showLabels', label: 'labels', kind: 'toggle', default: true, unit: 'none' },
  { group: 'labels', key: 'namedPx', label: 'label from', kind: 'range', min: 4, max: 60, step: 1, default: NAMED_PX, unit: 'px' },
  { group: 'labels', key: 'labelEarn', label: 'earn inside top', kind: 'range', min: 2, max: 60, step: 1, default: EARN, unit: 'none' },
  { group: 'labels', key: 'labelKeep', label: 'keep until top', kind: 'range', min: 4, max: 90, step: 1, default: KEEP, unit: 'none' },
  { group: 'labels', key: 'labelCap', label: 'hard cap', kind: 'range', min: 0, max: 90, step: 1, default: NAMED_CAP, unit: 'none' },
  { group: 'labels', key: 'labelDwell', label: 'dwell', kind: 'range', min: 0, max: 2000, step: 50, default: DWELL_MS, unit: 'none' },

  /* ---- the nodes: size channel, px clamps, glyph body, falloff (field/constants). -------- */
  { group: 'nodes', key: 'sizeChannel', label: 'size', kind: 'select', options: SIZE_CHANNELS, default: 'degree', unit: 'none' },
  /* Which facet binds the node glyph SHAPE (setShapeFacet). 'none' = all discs; a panel widens
     the options from the live graph's facets. Best read at a larger node size / lower density. */
  { group: 'nodes', key: 'shapeFacet', label: 'shape by', kind: 'select', options: SHAPE_FACETS, default: 'none', unit: 'none' },
  { group: 'nodes', key: 'nodeScale', label: 'node size', kind: 'range', min: 0.05, max: 10, step: 0.05, default: NODE_SCALE, unit: 'none' },
  { group: 'nodes', key: 'sizeCurve', label: 'size contrast', kind: 'range', min: 0.15, max: 3, step: 0.05, default: SIZE_CURVE, unit: 'none' },
  { group: 'nodes', key: 'nodeMinPx', label: 'never smaller than', kind: 'range', min: 0.2, max: 10, step: 0.1, default: NODE_MIN_PX, unit: 'px' },
  /* An OPTIONAL upper size cap, OFF by default. The old finite clamp was what stopped a node
     ever filling the frame — approach it and, once projected px hit the cap, the shader held
     screen size constant by shrinking the world radius, so the node appeared to "push" / back
     away. So the shipped default is uncapped — the slider's TOP detent (default === offAt) reads
     "off": fly up to ANY node and it grows without bound and fills the view. Drag the slider DOWN
     to re-impose a finite cap (CSS px). The consumer maps the "off" detent to the node shader's
     NODE_MAX_PX_OFF sentinel (Field.setNodeStyle({maxPx})); the shader's own unstyled default is
     that same sentinel, so an unconfigured field is uncapped too. */
  { group: 'nodes', key: 'nodeMaxPx', label: 'size cap', kind: 'range', min: 40, max: 2000, step: 10, default: 2000, offAt: 2000, unit: 'px' },
  { group: 'nodes', key: 'falloff', label: 'shrink with distance', kind: 'range', min: 0, max: 2.5, step: 0.05, default: FALLOFF, unit: 'none' },
  { group: 'nodes', key: 'bodyR', label: 'solid part', kind: 'range', min: 0.15, max: 1, step: 0.01, default: BODY_R, unit: 'none' },
  /* Fake-sphere shading: 0 = a flat disc (the shipped look), 1 = the body reads as a lit ball
     (a hemisphere normal + Lambert + soft specular, faked per-fragment — no geometry). The paper
     look turns it on; the dark looks keep it at 0, so they are visually unchanged. */
  { group: 'nodes', key: 'sphere', label: 'sphere shading', kind: 'range', min: 0, max: 1, step: 0.05, default: SPHERE, unit: 'none' },
  { group: 'nodes', key: 'haloPx', label: 'halo width', kind: 'range', min: 0, max: 24, step: 0.5, default: HALO_PX, unit: 'px' },
  { group: 'nodes', key: 'haloGain', label: 'halo strength', kind: 'range', min: 0, max: 2, step: 0.05, default: HALO_GAIN, unit: 'none' },
  { group: 'nodes', key: 'coreHot', label: 'core burns to white', kind: 'range', min: 0, max: 1, step: 0.02, default: CORE_HOT, unit: 'none' },

  /* ---- colour and ground: mode, background, palette, blend, tonemap. --------------------- */
  { group: 'colour', key: 'colorMode', label: 'colour', kind: 'select', options: COLOR_MODES, default: 'facet', unit: 'none' },
  /* WHICH facet colours the nodes in 'facet' mode (setColorFacet). The schema ships 'type';
     a panel widens the options from the live graph. Inert in column/uniform/density modes. */
  { group: 'colour', key: 'colorFacet', label: 'colour facet', kind: 'select', options: COLOR_FACETS, default: 'type', unit: 'none' },
  { group: 'colour', key: 'background', label: 'ground', kind: 'select', options: BACKGROUNDS, default: 'dark', unit: 'none' },
  { group: 'colour', key: 'palette', label: 'palette', kind: 'select', options: PALETTES, default: 'golden', unit: 'none' },
  { group: 'colour', key: 'blend', label: 'blending', kind: 'select', options: BLENDS, default: 'normal', unit: 'none' },
  { group: 'colour', key: 'tonemap', label: 'tone curve', kind: 'select', options: TONEMAPS, default: 'none', unit: 'none' },
  /* The ramp serves the 'column' colour mode (theme/palettes.ts RAMPS — a NUMBER to a colour;
     categorical facets keep discrete palettes, a ramp would imply an order they lack). */
  { group: 'colour', key: 'ramp', label: 'colour ramp', kind: 'select', options: RAMP_NAMES, default: DEFAULT_RAMP, unit: 'none' },
  /* Pre-curve gain, applied before the roll-off and ONLY when an operator is on (op 'none'
     is a true bypass — field/tonemap.ts). floor: setTonemap throws on exposure ≤ 0. */
  { group: 'colour', key: 'exposure', label: 'exposure', kind: 'range', min: 0.01, max: 4, step: 0.005, default: EXPOSURE_DEFAULT, unit: 'none', floor: 0 },
  /* Sublinear highlight compression on the accumulated additive HDR, applied PRE-bloom in the
     post chain (post/bloom.ts). 0 is off — an exact linear pass-through, today's look — and 1
     the strongest roll-off: it curves luminance by pow(L, g) (hue preserved) so a dense core
     stops blowing out to a flat white disc while the lifted sparse arms bloom. Only the bloom
     post-process reads it; with the glow off it has no effect. Distinct from 'tone curve' above
     (the post-SUM operator) — this crushes the density dynamic range BEFORE the bloom sees it. */
  { group: 'colour', key: 'compression', label: 'highlight compression', kind: 'range', min: 0, max: 1, step: 0.05, default: 0, unit: 'none' },

  /* ---- the links: alphas (field/constants + focus) and the budget fraction. -------------- */
  { group: 'links', key: 'linkBase', label: 'resting', kind: 'range', min: 0, max: 0.4, step: 0.002, default: LINK_BASE_ALPHA, unit: 'none' },
  { group: 'links', key: 'linkFocus', label: 'in focus', kind: 'range', min: 0, max: 1, step: 0.02, default: LINK_FOCUS_ALPHA, unit: 'none' },
  { group: 'links', key: 'linkHover', label: 'under the cursor', kind: 'range', min: 0, max: 1, step: 0.02, default: LINK_HOVER_ALPHA, unit: 'none' },
  { group: 'links', key: 'linkDim', label: 'behind a selection', kind: 'range', min: 0, max: 0.2, step: 0.002, default: LINK_DIM_ALPHA, unit: 'none' },
  { group: 'links', key: 'dimOthers', label: 'rest of the field', kind: 'range', min: 0.02, max: 0.7, step: 0.01, default: DIM_OTHERS, unit: 'none' },
  { group: 'links', key: 'lenSoft', label: 'long-edge fade over', kind: 'range', min: 0.01, max: 1, step: 0.01, default: LINK_LEN_SOFT, unit: 'none' },
  { group: 'links', key: 'lenFloor', label: 'longest edges keep', kind: 'range', min: 0, max: 1, step: 0.02, default: LINK_LEN_FLOOR, unit: 'none' },
  /* Fraction of the importance-ordered buffer drawn (the §5 draw-range budget). 1 = all of
     them — the boot quality is chosen from the corpus size, so this is session-scoped. */
  { group: 'links', key: 'linkBudget', label: 'links drawn', kind: 'range', min: 0.02, max: 1, step: 0.02, default: 1, unit: 'none' },

  /* ---- motion: spread, spin rates. Rates in rad/s — the renderer owns the clock (§10). ---
     spread's floor: setSpread throws on k ≤ 0 (scale about the centroid needs a scale). */
  { group: 'motion', key: 'spread', label: 'spread', kind: 'range', min: 0.05, max: 4, step: 0.05, default: 1, unit: 'none', floor: 0 },
  { group: 'motion', key: 'cameraDrift', label: 'camera drifts', kind: 'toggle', default: false, unit: 'none' },
  { group: 'motion', key: 'driftRate', label: 'drift', kind: 'range', min: 0, max: 0.3, step: 0.002, default: FLYCAM_DEFAULTS.spinRate, unit: 'none' },
  /* Hierarchy frame rate. 0 is the compiled default — a level spins only when its spec says
     so (hierarchy.ts: rate 0 where none declared), so the knob idles until a spec declares. */
  { group: 'motion', key: 'levelSpin', label: 'groups turn', kind: 'range', min: 0, max: 0.8, step: 0.005, default: 0, unit: 'none' },

  /* ---- flight: the flycam's tunables (cameras/flycam FLYCAM_DEFAULTS). ------------------- */
  { group: 'flight', key: 'speedK', label: 'speed', kind: 'range', min: 0.4, max: 10, step: 0.1, default: FLYCAM_DEFAULTS.speedK, unit: 'none' },
  { group: 'flight', key: 'mouseSens', label: 'drag sensitivity', kind: 'range', min: 0.001, max: 0.02, step: 0.0005, default: FLYCAM_DEFAULTS.mouseSens, unit: 'none' },
  { group: 'flight', key: 'turnRate', label: 'turn rate', kind: 'range', min: 0.2, max: 4, step: 0.05, default: FLYCAM_DEFAULTS.turnRate, unit: 'none' },
  { group: 'flight', key: 'easeLin', label: 'throttle ease', kind: 'range', min: 0.01, max: 0.6, step: 0.01, default: FLYCAM_DEFAULTS.easeLin, unit: 'none' },

  /* ---- the frame budget's blunt levers — session-scoped, see SESSION_KEYS. --------------- */
  { group: 'perf', key: 'renderScale', label: 'render scale', kind: 'range', min: 0.5, max: 2, step: 0.05, default: 1, unit: 'none' },
  { group: 'perf', key: 'flatPx', label: 'flat glyph under', kind: 'range', min: 0, max: 24, step: 0.5, default: FLAT_PX, unit: 'px' },
];

/** Session-scoped keys: auto-chosen from the corpus/device at every boot rather than
 *  remembered. A link budget tuned for a million pages is simply wrong for sixteen hundred,
 *  and restoring it after a corpus switch presents as a rendering bug, not a preference
 *  (the predecessor's hard-won rule, ported). Never persisted, never in a look. */
export const SESSION_KEYS: readonly string[] = ['linkBudget', 'renderScale', 'flatPx'];

/** Derive key → default from schema rows. The derivation lives here so DEFAULTS can never
 *  drift from the schema; duplicate keys throw at module load — a duplicate would make "the
 *  default" ambiguous and the panel render two knobs over one value. */
export function deriveDefaults(rows: readonly SchemaRow[]): Config {
  const out: Config = {};
  for (const row of rows) {
    if (row.key in out) throw new Error(`config schema: duplicate key "${row.key}"`);
    out[row.key] = row.default;
  }
  return out;
}

/** The shipped defaults — every value referenced from its owning constants module. */
export const DEFAULTS: Readonly<Config> = Object.freeze(deriveDefaults(SCHEMA));

/** A preset: a DIFF from DEFAULTS, so it reads as a statement about what makes the look what
 *  it is, and every knob it has no opinion about keeps following the code. */
export interface Preset {
  label: string;
  /** What this look is — the picker shows it, tests require it. */
  note: string;
  set: Readonly<Config>;
}

/** Shipped looks — the theme module's six visual presets (theme/presets.ts), which is the
 *  ONE shipped registry: a second same-named record here would be whole-preset copy drift,
 *  exactly what the header forbids row-by-row ('nebula' meaning two different renderings
 *  depending on which import a consumer reached first). User looks layer the same way. */
export const PRESETS: Readonly<Record<string, Preset>> = VISUAL_PRESETS;
