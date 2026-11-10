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
    note: 'the signature look — black ground, additive glyphs SUMMING in an HDR buffer, a screen-space bloom carrying the glow, and the accumulated blaze rolled off POST-SUM to a true black. Reinhard, not agx: agx lifts small values off zero and greys the negative space, where reinhard maps 0→0 and keeps the space between clusters truly black. Colour is cool→hot BY DEGREE (the `nebula` ramp), so crowded hubs burn white-orange cores while leaves stay blue. The consumer turns the glow on when this look is picked; the bloom is what makes it a nebula rather than a white blob.',
    set: {
      background: 'dark',
      blend: 'additive',
      // The post-sum roll-off (the bloom chain applies it once over the whole accumulated
      // field). Reinhard for a clean black point — see the note; agx washed the negative space.
      tonemap: 'reinhard',
      // Deliberately LOW: the roll-off runs once over the entire additive SUM, which is large,
      // so a low gain is what keeps the sparse field black and lets only the dense cores bloom
      // bright. A per-fragment look would want ~1; this is the whole-field post value. Tuned on
      // a 1M synth corpus against the reference nebula.
      exposure: 0.11,
      // colour is the degree column through the cool→hot `nebula` ramp (rank-normalised, the
      // field's setColorColumn): importance IS the hue, so hubs read warm and leaves blue — the
      // reference's palette, reproduced by data rather than painted. (Additive summing only keeps
      // the hue where the per-pixel density is low — the sparse cluster edges; dense cores blow to
      // a white bloom, exactly as the reference's hot centres do.)
      colorMode: 'column',
      ramp: 'nebula',
      // TINY bright additive dots, NOT big halos: at 1M zoomed out every glyph is a sub-pixel
      // additive point and the BLOOM, not a per-node halo, carries the glow. Big halos (the
      // predecessor's instinct) pile into a solid white sheet at this density — the exact blob the
      // bloom exists to replace. The hot-core + halo terms below only fire when you fly IN
      // (non-flat glyphs), giving each near node a bright centre that seeds the bloom up close.
      nodeScale: 1.0,
      nodeMinPx: 1.3,
      nodeMaxPx: 280,
      coreHot: 0.95,
      haloGain: 1.2,
      bodyR: 0.5,
      haloPx: 4,
      falloff: 1.0,
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
