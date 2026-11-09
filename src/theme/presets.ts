/* Visual presets (DESIGN.md §7): "presets and user 'looks' are DIFFS from defaults". Each
 * entry is a named diff over the stage-8 config schema keys (src/config/schema.ts — every key
 * set here is a real row, test-enforced) plus a one-line statement of what makes that look
 * what it is — the predecessor's convention (vault-site-3d config.js PRESETS), where the diff
 * doubles as the explanation. Ported in SPIRIT: the keys are this schema's, the numbers are
 * the predecessor's where the knob survived, and each look keeps the one or two decisions its
 * README says actually make it (e.g. paper: "the single biggest difference between the two
 * looks" is the glow going to zero, "and it is not the palette").
 *
 * Pure data — applying a preset is the config store's job (a diff over DEFAULTS: everything a
 * preset does not name returns to the default, so a look stays reproducible rather than
 * "whatever was set plus this"). These load through the store like any look:
 * createConfigStore({ presets: VISUAL_PRESETS }).
 *
 * Two inherited invariants, test-enforced in theme.test.ts:
 *   - a preset that asks for additive blending MUST also name a tonemap operator — additive
 *     is the only way to blow a dense core out to flat white, so 'additive' + 'none' ships
 *     the exact bug the roll-off (field/tonemap.ts) exists to fix;
 *   - blending is a property of the GROUND (§5): a light-ground look is 'normal', a glow
 *     look is 'additive', and both layers follow the one setting.
 */

import type { Preset } from '../config/schema';

/** A visual preset — exactly the config store's Preset shape (schema-key diff + note). */
export type VisualPreset = Preset;

export const VISUAL_PRESETS = {
  field: {
    label: 'field',
    note: 'the default — categorical hues on the dark ground, links readable; an empty diff by definition',
    set: {},
  },

  paper: {
    label: 'paper',
    note: 'ink on paper: light ground, matte solid dots, zero glow — normal blend because blending is a property of the ground (§5)',
    set: {
      background: 'light',
      blend: 'normal',
      tonemap: 'none',
      // On a light ground a white-hot core is invisible and a halo is just haze, so both go
      // to zero: these dots are ink, not light (the predecessor's "single biggest difference
      // between the two looks, and it is not the palette").
      coreHot: 0,
      haloGain: 0.08,
      haloPx: 1,
      bodyR: 0.88,
      nodeScale: 1.5,
      nodeMinPx: 1.8,
      nodeMaxPx: 220,
      linkBase: 0.16,
      lenSoft: 0.5,
      lenFloor: 0.3,
    },
  },

  nebula: {
    label: 'nebula',
    note: 'black-ground additive glow with hot cores — big soft halos SUM past white where nodes crowd, and agx rolls the blaze off while keeping its hue',
    set: {
      background: 'dark',
      blend: 'additive',
      tonemap: 'agx',
      exposure: 1.8,
      // A BIG soft sprite with a TINY hot core, not a tiny speck: large overlapping halos
      // reproduce the reference's MECHANISM (overlap → brightness) at any corpus density —
      // the predecessor measured the literal "tiny bright specks" reading as grey mush.
      nodeScale: 2.8,
      nodeMinPx: 5,
      nodeMaxPx: 300,
      coreHot: 0.95,
      haloGain: 1.7,
      bodyR: 0.22,
      haloPx: 11,
      falloff: 1.1,
      linkBase: 0.03,
      lenSoft: 0.1,
      lenFloor: 0.02,
    },
  },

  blueprint: {
    label: 'blueprint',
    note: 'structure over atmosphere: one flat hue, depth deliberately flattened (falloff well under 1 breaks perspective so the whole graph is legible at once), links you can actually trace',
    set: {
      background: 'dark',
      blend: 'normal',
      tonemap: 'none',
      colorMode: 'uniform', // the single hue IS the look
      falloff: 0.45,
      nodeScale: 0.8,
      nodeMinPx: 1.8,
      nodeMaxPx: 90,
      coreHot: 0.25,
      haloGain: 0.18,
      haloPx: 1.5,
      bodyR: 0.78,
      linkBase: 0.17,
      lenSoft: 0.9,
      lenFloor: 0.6,
    },
  },

  ember: {
    label: 'ember',
    note: 'hubs burn, leaves smoulder — colour by the degree column through the ember ramp (rank-normalised, field setColorColumn), so importance IS the colour',
    set: {
      background: 'dark',
      blend: 'additive',
      tonemap: 'filmic',
      colorMode: 'column', // the degree column, rank-normalised…
      ramp: 'ember', // …through the ember ramp — the look's defining pair
      nodeScale: 1.0,
      nodeMinPx: 1.2,
      nodeMaxPx: 200,
      coreHot: 0.8,
      haloGain: 0.9,
      haloPx: 7,
      bodyR: 0.4,
      linkBase: 0.045,
      lenSoft: 0.12,
      lenFloor: 0.04,
    },
  },

  survey: {
    label: 'survey',
    note: 'everything named, nothing glowing — the label band opened wide over solid un-haloed dots; for reading the corpus rather than flying it',
    set: {
      background: 'dark',
      blend: 'normal',
      tonemap: 'none',
      falloff: 0.7,
      nodeScale: 0.9,
      nodeMinPx: 2.4,
      nodeMaxPx: 140,
      coreHot: 0.3,
      haloGain: 0.25,
      haloPx: 2,
      bodyR: 0.8,
      // label-heavy: the glyph threshold drops and the rank band + cap open right up
      // (defaults: namedPx 13, earn 20, keep 32, cap 34 — ladder/constants)
      namedPx: 7,
      labelEarn: 48,
      labelKeep: 70,
      labelCap: 70,
      labelDwell: 900,
      linkBase: 0.05,
      lenSoft: 0.3,
      lenFloor: 0.08,
    },
  },
} as const satisfies Record<string, VisualPreset>;

export type VisualPresetName = keyof typeof VISUAL_PRESETS;
export const VISUAL_PRESET_NAMES = Object.keys(VISUAL_PRESETS) as VisualPresetName[];
