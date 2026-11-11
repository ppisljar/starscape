/* The config engine (DESIGN.md §7), pinning the predecessor's hard-won invariants
 * (vault-site-3d test/config.test.mjs) on the ported machinery:
 *
 *   - only TOUCHED keys persist, so a default improved in the code actually reaches a browser
 *     that saved settings before (the one that was broken, and cost rounds of chasing it);
 *   - presets and looks are DIFFS from the defaults, applied over them;
 *   - session keys never persist and never ride a look or an import;
 *   - unknown stored keys are dropped AND reported — returned, never console.warn'd;
 *   - storage is injected (a Map here) — the library touches no global at import time;
 *   - schema sanity: distances are spacings or px, never raw world units, and every default
 *     REFERENCES its owning constants module rather than restating the number.
 */
import { describe, expect, it } from 'vitest';
import {
  BLENDS,
  DEFAULTS,
  PRESETS,
  SCHEMA,
  SESSION_KEYS,
  TONEMAPS,
  deriveDefaults,
} from '../src/config/schema';
import type { SchemaRow } from '../src/config/schema';
import {
  CUSTOM,
  createConfigStore,
  mapStorage,
  validValue,
} from '../src/config/store';
import type { StorageLike } from '../src/config/store';
import {
  CARD_SPACINGS,
  DWELL_MS,
  EARN,
  KEEP,
  LOCK_SPACINGS,
  NAMED_CAP,
  OPEN_SPACINGS,
} from '../src/ladder/constants';
import {
  FALLOFF,
  LINK_BASE_ALPHA,
  NODE_MIN_PX,
} from '../src/field/constants';
import { DIM_OTHERS, LINK_FOCUS_ALPHA, LINK_HOVER_ALPHA } from '../src/field/focus';
import { FLYCAM_DEFAULTS } from '../src/cameras/flycam';
import { TONEMAP_OPS } from '../src/field/tonemap';
import { VISUAL_PRESETS } from '../src/theme/presets';

/** A store over a fresh Map — what every test uses; no window, no localStorage anywhere. */
const fresh = (): { store: ReturnType<typeof createConfigStore>; map: Map<string, string> } => {
  const map = new Map<string, string>();
  return { store: createConfigStore({ storage: mapStorage(map) }), map };
};

describe('schema → defaults derivation', () => {
  it('DEFAULTS carries exactly one entry per row, equal to the row default', () => {
    expect(Object.keys(DEFAULTS).length).toBe(SCHEMA.length);
    for (const row of SCHEMA) expect(DEFAULTS[row.key]).toBe(row.default);
  });

  it('deriveDefaults throws on a duplicate key rather than silently shadowing', () => {
    const row = SCHEMA[0]!;
    expect(() => deriveDefaults([row, row])).toThrow(/duplicate/);
  });

  it('every default REFERENCES its owning constants module (spot checks across them all)', () => {
    // ladder distances + label band ← src/ladder/constants
    expect(DEFAULTS['tierCard']).toBe(CARD_SPACINGS);
    expect(DEFAULTS['tierLock']).toBe(LOCK_SPACINGS);
    expect(DEFAULTS['tierOpen']).toBe(OPEN_SPACINGS);
    expect(DEFAULTS['labelEarn']).toBe(EARN);
    expect(DEFAULTS['labelKeep']).toBe(KEEP);
    expect(DEFAULTS['labelCap']).toBe(NAMED_CAP);
    expect(DEFAULTS['labelDwell']).toBe(DWELL_MS);
    // node channel ← src/field/constants
    expect(DEFAULTS['nodeMinPx']).toBe(NODE_MIN_PX);
    // the size cap ships OFF: its default sits at the slider's top "off" detent (offAt), so an
    // unconfigured node grows without bound as you approach (the shader's uncapped default).
    const maxRow = SCHEMA.find((r) => r.key === 'nodeMaxPx');
    expect(maxRow?.kind === 'range' && maxRow.offAt).toBeDefined();
    expect(DEFAULTS['nodeMaxPx']).toBe(maxRow?.kind === 'range' ? maxRow.offAt : undefined);
    expect(DEFAULTS['falloff']).toBe(FALLOFF);
    // link alphas ← src/field/constants + src/field/focus
    expect(DEFAULTS['linkBase']).toBe(LINK_BASE_ALPHA);
    expect(DEFAULTS['linkFocus']).toBe(LINK_FOCUS_ALPHA);
    expect(DEFAULTS['linkHover']).toBe(LINK_HOVER_ALPHA);
    expect(DEFAULTS['dimOthers']).toBe(DIM_OTHERS);
    // flight ← src/cameras/flycam
    expect(DEFAULTS['speedK']).toBe(FLYCAM_DEFAULTS.speedK);
    expect(DEFAULTS['easeLin']).toBe(FLYCAM_DEFAULTS.easeLin);
    expect(DEFAULTS['driftRate']).toBe(FLYCAM_DEFAULTS.spinRate);
  });
});

describe('schema row sanity', () => {
  it('no distance row is in raw world units — spacings or px only (§7)', () => {
    for (const row of SCHEMA) {
      expect(row.unit, `row "${row.key}" must never be world units`).not.toBe('world');
    }
  });

  it('every range row has a coherent rail and a default inside it', () => {
    for (const row of SCHEMA) {
      if (row.kind !== 'range') continue;
      expect(row.min, row.key).toBeLessThan(row.max);
      expect(row.step, row.key).toBeGreaterThan(0);
      expect(row.default, row.key).toBeGreaterThanOrEqual(row.min);
      expect(row.default, row.key).toBeLessThanOrEqual(row.max);
      // a domain floor sits strictly below the rail — the rail stays a UI affordance
      if (row.floor !== undefined) expect(row.floor, row.key).toBeLessThan(row.min);
    }
  });

  it('the option lists ARE their owning modules’ lists — same reference, no restated copy', () => {
    // the predecessor's silent-no-op bug was a tonemap name drifting from its shader index
    expect(TONEMAPS).toBe(TONEMAP_OPS);
    // one shipped preset registry: the schema's PRESETS is the theme module's six looks,
    // not a second same-named record that renders differently (whole-preset copy drift)
    expect(PRESETS).toBe(VISUAL_PRESETS);
  });

  it('every select row defaults to one of its own options', () => {
    for (const row of SCHEMA) {
      if (row.kind !== 'select') continue;
      expect(row.options, row.key).toContain(row.default);
    }
  });

  it('only range rows carry a distance unit — a select or toggle measures nothing', () => {
    for (const row of SCHEMA) {
      if (row.kind !== 'range') expect(row.unit, row.key).toBe('none');
    }
  });

  it('session keys all name real rows', () => {
    const keys = new Set(SCHEMA.map((r) => r.key));
    for (const k of SESSION_KEYS) expect(keys.has(k), k).toBe(true);
  });

  it('every shipped preset sets only real keys, with values its row accepts, and has a note', () => {
    const byKey = new Map<string, SchemaRow>(SCHEMA.map((r) => [r.key, r]));
    for (const [name, p] of Object.entries(PRESETS)) {
      expect(p.note, `preset ${name} needs a note saying what it is`).toBeTruthy();
      for (const [k, v] of Object.entries(p.set)) {
        const row = byKey.get(k);
        expect(row, `preset ${name} sets unknown key ${k}`).toBeDefined();
        expect(validValue(row!, v), `preset ${name}: ${k} = ${String(v)}`).toBe(true);
        expect(SESSION_KEYS, `preset ${name} must not pin session key ${k}`).not.toContain(k);
      }
    }
    expect(BLENDS).toContain('normal');
    expect(TONEMAPS).toContain('none');
  });
});

describe('persistence through an injected storage', () => {
  it('a fresh load is exactly the defaults, with no warnings', () => {
    const { store } = fresh();
    expect(store.load()).toEqual([]);
    for (const k of Object.keys(DEFAULTS)) expect(store.values[k]).toBe(DEFAULTS[k]);
  });

  it('a changed value survives a reload through the SAME Map — storage is truly injected', () => {
    const { store, map } = fresh();
    store.set('speedK', 7.7);
    store.save();
    const again = createConfigStore({ storage: mapStorage(map) });
    again.load();
    expect(again.get('speedK')).toBe(7.7);
  });

  it('an UNTOUCHED key follows the code, even after other settings were saved', () => {
    // THE one the predecessor broke: saving the whole config froze every default.
    const { store, map } = fresh();
    store.set('nodeScale', 2.5); // the user drags one slider...
    store.save(); // ...which must not freeze every other default too
    const again = createConfigStore({ storage: mapStorage(map) });
    again.load();
    expect(again.get('nodeScale')).toBe(2.5); // the deliberate change sticks
    expect(again.get('falloff')).toBe(DEFAULTS['falloff']); // the rest follows the code
    expect(again.touched.has('falloff')).toBe(false);
  });

  it('session-scoped keys are never persisted', () => {
    const { store, map } = fresh();
    store.set('linkBudget', 0.25);
    store.set('renderScale', 0.5);
    store.save();
    const again = createConfigStore({ storage: mapStorage(map) });
    again.load();
    expect(again.get('linkBudget')).toBe(DEFAULTS['linkBudget']); // the corpus picks it every boot
    expect(again.get('renderScale')).toBe(DEFAULTS['renderScale']);
  });

  it('a corrupt blob loads the defaults and REPORTS, instead of breaking the viewer', () => {
    const map = new Map<string, string>([['starscape:cfg:v1', '{not json']]);
    const store = createConfigStore({ storage: mapStorage(map) });
    const warnings = store.load();
    expect(warnings.length).toBeGreaterThan(0);
    expect(store.get('nodeScale')).toBe(DEFAULTS['nodeScale']);
  });

  it('unknown keys in stored data are dropped and returned as warnings — no console', () => {
    const map = new Map<string, string>([
      [
        'starscape:cfg:v1',
        JSON.stringify({
          preset: 'field',
          touched: ['speedK', 'notAKey'],
          v: { speedK: 5, notAKey: 7 },
        }),
      ],
    ]);
    const store = createConfigStore({ storage: mapStorage(map) });
    const warnings = store.load();
    expect(store.get('speedK')).toBe(5);
    expect('notAKey' in store.values).toBe(false);
    expect(warnings.some((w) => w.includes('notAKey'))).toBe(true);
  });

  it('an ill-typed stored value is dropped with a warning, not trusted into a uniform', () => {
    const map = new Map<string, string>([
      [
        'starscape:cfg:v1',
        JSON.stringify({ touched: ['speedK', 'blend'], v: { speedK: 'fast', blend: 'screen' } }),
      ],
    ]);
    const store = createConfigStore({ storage: mapStorage(map) });
    const warnings = store.load();
    expect(store.get('speedK')).toBe(DEFAULTS['speedK']);
    expect(store.get('blend')).toBe(DEFAULTS['blend']); // 'screen' names no blend option
    expect(warnings.length).toBe(2);
  });

  it('works with no storage at all — memory-only, nothing throws', () => {
    const store = createConfigStore({});
    expect(store.load()).toEqual([]);
    store.set('speedK', 9);
    store.save(); // a no-op, not an error
    expect(store.get('speedK')).toBe(9);
  });
});

describe('presets, looks, and layering', () => {
  it('a preset applies as a diff over the defaults', () => {
    const { store } = fresh();
    expect(store.applyPreset('paper')).toBe(true);
    expect(store.get('background')).toBe('light'); // the preset's own keys
    expect(store.get('speedK')).toBe(DEFAULTS['speedK']); // everything else: the default
    expect(store.preset).toBe('paper');
    expect(store.applyPreset('no-such-look')).toBe(false);
  });

  it('layering: preset, then user diff on top, then session — each layer survives correctly', () => {
    const { store, map } = fresh();
    // layer 1: a preset
    store.applyPreset('nebula');
    // layer 2: the user's own touch on top of it
    store.set('speedK', 6.5);
    // layer 3: session state the boot derived
    store.set('linkBudget', 0.3);
    store.save();

    const again = createConfigStore({ storage: mapStorage(map) });
    again.load();
    expect(again.preset).toBe('nebula');
    expect(again.get('blend')).toBe('additive'); // the preset's key persisted (it was touched)
    expect(again.get('speedK')).toBe(6.5); // the user diff persisted on top
    expect(again.get('linkBudget')).toBe(DEFAULTS['linkBudget']); // session never persisted
    expect(again.get('turnRate')).toBe(DEFAULTS['turnRate']); // untouched keys follow the code
  });

  it('a preset resets keys a previous look had touched — apply is reproducible', () => {
    const { store } = fresh();
    store.set('nodeScale', 3.3);
    store.applyPreset('field'); // the empty diff: exactly the defaults
    expect(store.get('nodeScale')).toBe(DEFAULTS['nodeScale']);
    expect(store.touched.size).toBe(0);
  });

  it('a look NEVER rides OR resets the session keys — the boot perf notch survives it', () => {
    // The regression: applying a look reset the boot-derived session keys to their 1:1
    // defaults, so the consumer then drew every link at native dpr and tanked the framerate
    // zoomed in. Session keys are the corpus/device's, orthogonal to the visual look.
    const { store } = fresh();
    store.set('linkBudget', 0.04); // the 10⁶ boot notch
    store.set('renderScale', 0.5);
    store.set('flatPx', 9);
    store.applyPreset('nebula'); // the look that must NOT touch them
    expect(store.get('linkBudget')).toBe(0.04);
    expect(store.get('renderScale')).toBe(0.5);
    expect(store.get('flatPx')).toBe(9);
    expect(store.get('blend')).toBe('additive'); // the look's own keys still land
    // ...and a saved look is just as inert on them
    store.saveLook('mine');
    store.set('renderScale', 0.75);
    expect(store.applyLook('mine')).toBe(true);
    expect(store.get('renderScale')).toBe(0.75); // untouched by the look
  });

  it('a saved look stores only the diff, round-trips, and forgets nothing it never touched', () => {
    const { store, map } = fresh();
    store.set('haloPx', 9);
    store.set('background', 'light');
    store.saveLook('mine');
    const raw = JSON.parse(map.get('starscape:looks:v1')!) as Record<string, unknown>;
    expect(Object.keys(raw['mine'] as object).sort()).toEqual(['background', 'haloPx']);

    const again = createConfigStore({ storage: mapStorage(map) });
    expect(again.applyLook('mine')).toBe(true);
    expect(again.get('haloPx')).toBe(9);
    expect(again.get('background')).toBe('light');
    expect(again.get('nodeScale')).toBe(DEFAULTS['nodeScale']); // picks up improved defaults
  });

  it('deleting a look removes it, leaves the others, and refuses the built-ins', () => {
    const { store } = fresh();
    store.set('haloPx', 4);
    store.saveLook('a');
    store.set('haloPx', 5);
    store.saveLook('b');
    expect(store.deleteLook('a')).toBe(true);
    expect(store.allLooks().saved).toEqual(['b']);
    expect(store.deleteLook('paper')).toBe(false); // shipped presets are not yours to delete
    expect(store.applyLook('paper')).toBe(true); // ...and still apply
  });

  it('allLooks lists the built-ins (with notes) and the saved ones separately', () => {
    const { store } = fresh();
    store.set('haloPx', 7);
    store.saveLook('mine');
    const { built, saved } = store.allLooks();
    expect(built.some(([k]) => k === 'nebula')).toBe(true);
    for (const [, , note] of built) expect(note).toBeTruthy();
    expect(saved).toEqual(['mine']);
  });

  it('applying CUSTOM is a no-op readout, never a reset', () => {
    const { store } = fresh();
    store.set('haloPx', 11);
    expect(store.applyLook(CUSTOM)).toBe(true);
    expect(store.get('haloPx')).toBe(11); // picking "custom" must not wipe what it describes
  });
});

describe('export / import', () => {
  it('diff round-trip: apply(export(x)) reproduces x on every persistable key', () => {
    const { store } = fresh();
    store.set('tierCard', 3.7);
    store.set('blend', 'additive');
    store.set('cameraDrift', true);
    store.set('linkBudget', 0.4); // session — must NOT ride the export
    const blob = store.exportLook();
    expect(Object.keys(blob).sort()).toEqual(['blend', 'cameraDrift', 'tierCard']);

    const other = fresh().store;
    const { applied, warnings } = other.importLook(blob);
    expect(warnings).toEqual([]);
    expect(applied).toEqual(blob);
    for (const k of Object.keys(DEFAULTS)) {
      if (k === 'linkBudget') continue; // session — deliberately not carried
      expect(other.values[k], k).toBe(k in blob ? blob[k] : DEFAULTS[k]);
    }
    expect(other.exportLook()).toEqual(blob); // export ∘ import is the identity on diffs
  });

  it('import drops unknown and session keys with warnings, applies the rest', () => {
    const { store } = fresh();
    const { applied, warnings } = store.importLook({
      tierCard: 4,
      notAKey: 7,
      linkBudget: 0.1,
      speedK: 'nope',
    });
    expect(applied).toEqual({ tierCard: 4 });
    expect(store.get('tierCard')).toBe(4);
    expect(store.get('linkBudget')).toBe(DEFAULTS['linkBudget']);
    expect(warnings.some((w) => w.includes('notAKey'))).toBe(true);
    expect(warnings.some((w) => w.includes('linkBudget'))).toBe(true);
    expect(warnings.some((w) => w.includes('speedK'))).toBe(true);
  });

  it('importing garbage applies nothing and says so', () => {
    const { store } = fresh();
    store.set('haloPx', 3);
    const { applied, warnings } = store.importLook('[not an object]');
    expect(applied).toBeNull();
    expect(warnings.length).toBe(1);
    expect(store.get('haloPx')).toBe(3); // a failed import changes nothing
  });
});

describe('the store API guards', () => {
  it('get throws on an unknown key — a typo must not become NaN in a shader', () => {
    const { store } = fresh();
    expect(() => store.get('nodeScail')).toThrow(/unknown key/);
  });

  it('set refuses unknown keys and ill-typed values, and leaves no mark', () => {
    const { store } = fresh();
    expect(store.set('nodeScail', 1)).toBe(false);
    expect(store.set('blend', 'screen')).toBe(false);
    expect(store.set('speedK', Number.NaN)).toBe(false);
    expect(store.touched.size).toBe(0);
  });

  it('a row floor is validation, the rail is not: exposure/spread ≤ 0 never reach a setter', () => {
    // Field.setTonemap throws on exposure ≤ 0 and Field.setSpread on spread ≤ 0 — a pasted
    // look storing one used to persist FIRST and crash every boot after. The floor refuses
    // it at the validation layer; rail overshoot stays a deliberate choice (predecessor rule).
    const { store } = fresh();
    expect(store.set('exposure', 0)).toBe(false);
    expect(store.set('exposure', -2)).toBe(false);
    expect(store.set('spread', 0)).toBe(false);
    expect(store.set('exposure', 0.01)).toBe(true); // under the 0.1 rail, above the floor
    const { applied, warnings } = store.importLook({ exposure: 0, spread: -1, tierCard: 3 });
    expect(applied).toEqual({ tierCard: 3 });
    expect(warnings.some((w) => w.includes('exposure'))).toBe(true);
    expect(warnings.some((w) => w.includes('spread'))).toBe(true);
  });

  it('a throwing storage adapter is survived on read and write', () => {
    const broken: StorageLike = {
      getItem: () => {
        throw new Error('nope');
      },
      setItem: () => {
        throw new Error('nope');
      },
      removeItem: () => {
        throw new Error('nope');
      },
    };
    const store = createConfigStore({ storage: broken });
    expect(store.load()).toEqual([]); // unreadable = fresh, not fatal
    store.set('speedK', 4);
    store.save(); // private mode: a nicety lost, not an exception
    expect(store.get('speedK')).toBe(4);
  });
});
