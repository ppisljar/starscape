/* Theming data (src/theme): ramps + sampler (predecessor RAMPS/rampAt pins), categorical
 * palettes (predecessor layer palette + the colorblind-safe sets, values pinned to their
 * published sources), the binding helpers that produce the flat buffers the field consumes,
 * and the visual presets (diffs over the stage-8 schema keys).
 *
 * The preset ↔ schema cross-check imports src/config/schema DYNAMICALLY: that module is being
 * written in parallel. Absent, the check skips with a loud message — the gate re-runs
 * everything once it lands, which is when a preset naming a key the schema does not have
 * becomes a failure instead of a note.
 */

import { describe, expect, it } from 'vitest';
import {
  PALETTES,
  PALETTE_NAMES,
  RAMPS,
  RAMP_NAMES,
  VAULT_LAYER_NAMES,
  facetColors,
  namedPaletteBuffer,
  paletteBuffer,
  rampAt,
  rampColors,
} from '../src/theme/palettes';
import type { RGB } from '../src/theme/palettes';
import { VISUAL_PRESETS, VISUAL_PRESET_NAMES } from '../src/theme/presets';
import { TONEMAP_OPS } from '../src/field/tonemap';

/* The stage-8 schema, when the parallel module has landed. */
const schemaMod: Record<string, unknown> | null = await import('../src/config/schema')
  .then((m) => m as Record<string, unknown>)
  .catch(() => null);

describe('rampAt (predecessor sampler, values verbatim)', () => {
  it('pins ramp STARTS exactly (t = 0 is the first stop)', () => {
    expect(rampAt('aurora', 0)).toEqual([0.05, 0.26, 0.4]);
    expect(rampAt('ice', 0)).toEqual([0.02, 0.06, 0.16]);
    expect(rampAt('ember', 0)).toEqual([0.1, 0.02, 0.04]);
    expect(rampAt('mono', 0)).toEqual([0.08, 0.09, 0.12]);
  });

  it('pins midpoints: t = 0.5 lands where the stop maths says', () => {
    // aurora has 5 stops → x = 0.5·4 = 2 exactly: stop 2, no interpolation
    expect(rampAt('aurora', 0.5)).toEqual([0.33, 0.45, 0.92]);
    // ice has 4 stops → x = 1.5: halfway between stops 1 and 2
    const [r, g, b] = rampAt('ice', 0.5);
    expect(r).toBeCloseTo((0.13 + 0.55) / 2, 10);
    expect(g).toBeCloseTo((0.36 + 0.8) / 2, 10);
    expect(b).toBeCloseTo((0.66 + 0.98) / 2, 10);
  });

  it('pins ramp ENDS (t = 1 clamps to 0.9999 — predecessor logic preserved)', () => {
    // within 4·10⁻⁴·span of the last stop, by construction
    const end = rampAt('aurora', 1);
    expect(end[0]).toBeCloseTo(1.0, 3);
    expect(end[1]).toBeCloseTo(0.72, 3);
    expect(end[2]).toBeCloseTo(0.28, 3);
    const mono = rampAt('mono', 1);
    for (const c of mono) expect(c).toBeCloseTo(1.0, 3);
  });

  it('clamps t outside [0, 1] instead of overrunning the stops', () => {
    expect(rampAt('ember', -3)).toEqual(rampAt('ember', 0));
    expect(rampAt('ember', 7)).toEqual(rampAt('ember', 1));
  });

  it('falls back to aurora on an unknown name (predecessor behaviour), and accepts raw stops', () => {
    expect(rampAt('no-such-ramp' as never, 0.25)).toEqual(rampAt('aurora', 0.25));
    const stops: RGB[] = [
      [0, 0, 0],
      [1, 1, 1],
    ];
    const mid = rampAt(stops, 0.5);
    for (const c of mid) expect(c).toBeCloseTo(0.5, 10);
  });

  it('ships the cool→hot nebula ramp (blue leaves → orange/white hubs)', () => {
    expect(RAMP_NAMES).toContain('nebula');
    const cool = rampAt('nebula', 0);
    const hot = rampAt('nebula', 1);
    expect(cool[2]).toBeGreaterThan(cool[0]); // cool end: blue dominates red (leaves)
    expect(hot[0]).toBeGreaterThan(hot[2]); // hot end: red dominates blue (hubs)
  });

  it('ships the iceFire density ramp (blue sparse → orange dense → white cores)', () => {
    expect(RAMP_NAMES).toContain('iceFire');
    const cool = rampAt('iceFire', 0);
    const hot = rampAt('iceFire', 0.75);
    const top = rampAt('iceFire', 1);
    expect(cool[2]).toBeGreaterThan(cool[0]); // sparse end: blue dominates red
    expect(hot[0]).toBeGreaterThan(hot[2]); // dense end: red/orange dominates blue
    for (const c of top) expect(c).toBeCloseTo(1, 2); // densest cores: white
  });

  it('ships the viridis-like ramp alongside the predecessor set', () => {
    expect(RAMP_NAMES).toContain('viridis');
    expect(RAMPS.viridis.length).toBeGreaterThanOrEqual(5);
    // dark-purple start, yellow end — the perceptual order IS the point
    expect(rampAt('viridis', 0)[2]).toBeGreaterThan(rampAt('viridis', 0)[1]); // blue > green
    expect(rampAt('viridis', 1)[0]).toBeGreaterThan(0.9); // hot yellow
    expect(rampAt('viridis', 1)[2]).toBeLessThan(0.3);
  });
});

describe('categorical palettes (values pinned to their published sources)', () => {
  it('ports the predecessor layer palette, in its order, with the root ink for "other"', () => {
    const p = PALETTES.vaultLayer;
    expect(p.length).toBe(7);
    expect(VAULT_LAYER_NAMES.length).toBe(7);
    expect(p[0]).toEqual([0.122, 0.42, 0.322]); // source — the original's guide green
    expect(p[5]).toEqual([0.635, 0.306, 0.212]); // term — rust
    expect(p[6]).toEqual([0.082, 0.126, 0.106]); // other — root ink
  });

  it('pins Okabe–Ito (8 colours, canonical sRGB bytes)', () => {
    const p = PALETTES.okabeIto;
    expect(p.length).toBe(8);
    const want: Array<[number, number, number]> = [
      [0, 0, 0], // black
      [230, 159, 0], // orange
      [86, 180, 233], // sky blue
      [0, 158, 115], // bluish green
      [240, 228, 66], // yellow
      [0, 114, 178], // blue
      [213, 94, 0], // vermillion
      [204, 121, 167], // reddish purple
    ];
    for (let v = 0; v < 8; v++) {
      for (let c = 0; c < 3; c++) expect(p[v]![c]).toBeCloseTo(want[v]![c]! / 255, 10);
    }
  });

  it('pins Paul Tol bright (7) and muted (10)', () => {
    expect(PALETTES.tolBright.length).toBe(7);
    expect(PALETTES.tolMuted.length).toBe(10);
    // tolBright[0] = #4477AA, tolBright[4] = #EE6677
    expect(PALETTES.tolBright[0]).toEqual([0x44 / 255, 0x77 / 255, 0xaa / 255]);
    expect(PALETTES.tolBright[4]).toEqual([0xee / 255, 0x66 / 255, 0x77 / 255]);
    // tolMuted[0] = #332288 indigo, tolMuted[9] = #DDDDDD pale grey ("bad data")
    expect(PALETTES.tolMuted[0]).toEqual([0x33 / 255, 0x22 / 255, 0x88 / 255]);
    expect(PALETTES.tolMuted[9]).toEqual([0xdd / 255, 0xdd / 255, 0xdd / 255]);
  });

  it('every shipped palette stays in [0, 1] (the shader consumes these raw)', () => {
    for (const name of PALETTE_NAMES) {
      for (const entry of PALETTES[name]) {
        for (const c of entry) {
          expect(c).toBeGreaterThanOrEqual(0);
          expect(c).toBeLessThanOrEqual(1);
        }
      }
    }
  });
});

describe('paletteBuffer (the flat 3·k layout the field consumes)', () => {
  it('lays value v at [3v .. 3v+2]', () => {
    const buf = paletteBuffer(PALETTES.okabeIto, 8);
    expect(buf.length).toBe(24);
    expect(buf[3]).toBeCloseTo(230 / 255, 6); // value 1 = orange, f32 storage
    expect(buf[4]).toBeCloseTo(159 / 255, 6);
    expect(buf[5]).toBe(0);
  });

  it('wraps when the facet has more values than the palette has entries', () => {
    const buf = paletteBuffer(PALETTES.tolBright, 9);
    expect(buf.length).toBe(27);
    // entry 7 wraps to 0, entry 8 to 1
    for (let c = 0; c < 3; c++) {
      expect(buf[7 * 3 + c]).toBe(buf[c]);
      expect(buf[8 * 3 + c]).toBe(buf[3 + c]);
    }
  });

  it('refuses an empty palette loudly', () => {
    expect(() => paletteBuffer([], 3)).toThrow(/empty/);
  });
});

describe('namedPaletteBuffer (correspondence by NAME, never by list position)', () => {
  it('maps facet value names through the record, fallback for strangers', () => {
    const byName: Record<string, RGB> = {
      note: [0.541, 0.42, 0.137],
      term: [0.635, 0.306, 0.212],
    };
    const buf = namedPaletteBuffer(['term', 'mystery', 'note'], byName, [1, 0, 1]);
    expect(Array.from(buf.subarray(0, 3))).toEqual([
      Math.fround(0.635),
      Math.fround(0.306),
      Math.fround(0.212),
    ]);
    expect(Array.from(buf.subarray(3, 6))).toEqual([1, 0, 1]); // the fallback
    expect(buf[6]).toBeCloseTo(0.541, 6);
  });
});

describe('facetColors / rampColors (the per-node 3n buffers)', () => {
  it('facetColors paints codes through the palette', () => {
    const colors = facetColors(Uint8Array.from([0, 1, 0]), 3, 2, PALETTES.tolBright);
    expect(colors.length).toBe(9);
    expect(colors[0]).toBeCloseTo(0x44 / 255, 6);
    expect(colors[3]).toBeCloseTo(0x66 / 255, 6); // node 1 = tolBright[1] cyan
    expect(colors[6]).toBe(colors[0]);
  });

  it('rampColors is RANK-normalised: an outlier cannot flatten the band (sizeOf machinery)', () => {
    // data [5, 1e9, 7]: linear mapping would pin nodes 0 and 2 to the ramp floor together;
    // rank puts 5 → t=0, 7 → t=0.5, 1e9 → t=1
    const colors = rampColors(Float32Array.from([5, 1e9, 7]), 3, 'mono');
    // node 0: first stop
    expect(colors[0]).toBeCloseTo(0.08, 6);
    expect(colors[1]).toBeCloseTo(0.09, 6);
    expect(colors[2]).toBeCloseTo(0.12, 6);
    // node 2 (value 7, mid RANK): exactly the middle stop of the 3-stop ramp
    expect(colors[6]).toBeCloseTo(0.55, 6);
    expect(colors[7]).toBeCloseTo(0.58, 6);
    expect(colors[8]).toBeCloseTo(0.64, 6);
    // node 1 (the outlier): the top of the ramp (t=1 → within the 0.9999 clamp of white)
    for (let c = 0; c < 3; c++) expect(colors[3 + c]!).toBeGreaterThan(0.999);
  });

  it('writes into a caller-provided buffer (the field repaints in place)', () => {
    const out = new Float32Array(9);
    const ret = rampColors(Float32Array.from([1, 2, 3]), 3, 'ember', out);
    expect(ret).toBe(out);
    expect(out[0]).toBeCloseTo(0.1, 6); // ember floor
  });

  it('matches rampAt pointwise (the sampler and the batch loop cannot disagree)', () => {
    const data = Float32Array.from([10, 30, 20, 40]);
    const colors = rampColors(data, 4, 'spectral');
    const pct = [0, 2 / 3, 1 / 3, 1];
    for (let i = 0; i < 4; i++) {
      const want = rampAt('spectral', pct[i]!);
      for (let c = 0; c < 3; c++) expect(colors[i * 3 + c]).toBeCloseTo(want[c]!, 6);
    }
  });
});

describe('visual presets (diffs from defaults — §7)', () => {
  it('ships the six looks', () => {
    for (const name of ['field', 'paper', 'nebula', 'blueprint', 'ember', 'survey']) {
      expect(VISUAL_PRESET_NAMES).toContain(name);
    }
  });

  it('field is the default: an empty diff by definition', () => {
    expect(Object.keys(VISUAL_PRESETS.field.set)).toEqual([]);
  });

  it('every preset carries its one-line doc', () => {
    for (const name of VISUAL_PRESET_NAMES) {
      expect(VISUAL_PRESETS[name].note.length).toBeGreaterThan(10);
      expect(VISUAL_PRESETS[name].label).toBe(name);
    }
  });

  it('every ADDITIVE preset rolls its highlights off (the predecessor invariant)', () => {
    // additive blending is the only way to blow a dense core out to flat white — a preset
    // that asks for it and leaves the operator at 'none' ships the bug tonemap.ts fixes.
    for (const name of VISUAL_PRESET_NAMES) {
      const set = VISUAL_PRESETS[name].set as Record<string, unknown>;
      if (set['blend'] === 'additive') {
        expect(set['tonemap'], `${name} is additive with no roll-off`).toBeTypeOf('string');
        expect(set['tonemap']).not.toBe('none');
      }
    }
  });

  it('every tonemap a preset names is a real operator; every ramp a real ramp', () => {
    for (const name of VISUAL_PRESET_NAMES) {
      const set = VISUAL_PRESETS[name].set as Record<string, unknown>;
      if (typeof set['tonemap'] === 'string') {
        expect(TONEMAP_OPS as readonly string[]).toContain(set['tonemap']);
      }
      if (typeof set['ramp'] === 'string') {
        expect(RAMP_NAMES as readonly string[]).toContain(set['ramp']);
      }
    }
  });

  it('paper: light ground, normal blend — blending is a property of the ground (§5)', () => {
    const set = VISUAL_PRESETS.paper.set;
    expect(set.background).toBe('light');
    expect(set.blend).toBe('normal');
    expect(set.coreHot).toBe(0); // ink, not light
    expect(set.haloGain as number).toBeLessThan(0.1); // no glow — the look's actual mechanism
  });

  it('nebula: additive + reinhard (true black) + hot cores + local-density colour', () => {
    const set = VISUAL_PRESETS.nebula.set;
    expect(set.blend).toBe('additive');
    // reinhard, not agx: the bloom post-pass applies the roll-off post-sum, and reinhard's 0→0
    // black point keeps the negative space black where agx lifts it to grey (measured on 1M).
    expect(set.tonemap).toBe('reinhard');
    expect(set.coreHot).toBeGreaterThanOrEqual(0.9);
    expect(set.haloGain as number).toBeGreaterThan(1); // the hot halo that seeds the bloom up close
    // a pure-#000 ground: BG_DARK lifts to grey under additive, the reference is true black.
    expect(set.background).toBe('black');
    // cool→hot by LOCAL SPATIAL DENSITY is the look's defining colour decision — the geometric
    // signal degree cannot paint under additive blending (field/density.ts + the iceFire ramp).
    expect(set.colorMode).toBe('density');
    expect(set.ramp).toBe('iceFire');
  });

  it('blueprint: a single hue', () => {
    expect(VISUAL_PRESETS.blueprint.set.colorMode).toBe('uniform');
    expect(VISUAL_PRESETS.blueprint.set.falloff as number).toBeLessThan(1); // flat depth
  });

  it('ember: colour by the degree column through the ember ramp', () => {
    expect(VISUAL_PRESETS.ember.set.colorMode).toBe('column');
    expect(VISUAL_PRESETS.ember.set.ramp).toBe('ember');
  });

  it('survey: no glow, label-heavy', () => {
    const set = VISUAL_PRESETS.survey.set;
    expect(set.blend).toBe('normal');
    expect(set.haloGain as number).toBeLessThan(0.3);
    expect(set.labelCap as number).toBeGreaterThanOrEqual(50);
    expect(set.labelEarn as number).toBeGreaterThan(20); // band opened past the default
  });
});

describe('preset ↔ schema key cross-check (stage-8 schema, parallel module)', () => {
  if (schemaMod === null) {
    console.warn(
      '\n*** theme.test: src/config/schema is NOT present at this runtime — preset diffs were ' +
        'NOT cross-checked against the schema keys. This is expected only while the parallel ' +
        'schema module is unwritten; the gate re-runs all tests once it lands. ***\n',
    );
    it.skip('SKIPPED (loudly): src/config/schema has not landed yet', () => undefined);
    return;
  }

  /** Pull the schema's key set from whichever shape the module exports: a DEFAULTS record
   *  (key → default) or SCHEMA rows (objects with .key, or [group, key, …] tuples). */
  const schemaKeys = (): Set<string> | null => {
    const defaults = schemaMod['DEFAULTS'];
    if (defaults !== null && typeof defaults === 'object' && !Array.isArray(defaults)) {
      const keys = Object.keys(defaults as Record<string, unknown>);
      if (keys.length > 0) return new Set(keys);
    }
    const schema = schemaMod['SCHEMA'];
    if (Array.isArray(schema) && schema.length > 0) {
      const keys = new Set<string>();
      for (const row of schema as unknown[]) {
        if (Array.isArray(row) && typeof row[1] === 'string') keys.add(row[1]);
        else if (
          row !== null &&
          typeof row === 'object' &&
          typeof (row as { key?: unknown }).key === 'string'
        ) {
          keys.add((row as { key: string }).key);
        }
      }
      if (keys.size > 0) return keys;
    }
    return null;
  };

  it('every key every preset sets exists in the schema', () => {
    const keys = schemaKeys();
    expect(
      keys,
      'src/config/schema landed but exports neither a DEFAULTS record nor SCHEMA rows this ' +
        'test knows how to read — update theme.test.ts schemaKeys() to match its real shape',
    ).not.toBeNull();
    const missing: string[] = [];
    for (const name of VISUAL_PRESET_NAMES) {
      for (const key of Object.keys(VISUAL_PRESETS[name].set)) {
        if (!keys!.has(key)) missing.push(`${name}.${key}`);
      }
    }
    expect(missing, `preset keys the schema does not know: ${missing.join(' ')}`).toEqual([]);
  });

  it('every value every preset sets is one its row accepts (kind, options, rails)', () => {
    const schema = schemaMod['SCHEMA'];
    if (!Array.isArray(schema)) return; // key-only shapes were handled above
    interface RowLike {
      key?: unknown;
      kind?: unknown;
      options?: unknown;
      min?: unknown;
      max?: unknown;
    }
    const rows = new Map<string, RowLike>();
    for (const row of schema as RowLike[]) {
      if (row !== null && typeof row === 'object' && typeof row.key === 'string') {
        rows.set(row.key, row);
      }
    }
    const bad: string[] = [];
    for (const name of VISUAL_PRESET_NAMES) {
      for (const [key, v] of Object.entries(VISUAL_PRESETS[name].set)) {
        const row = rows.get(key);
        if (!row) continue; // the key test above already reports these
        if (row.kind === 'select') {
          if (!(Array.isArray(row.options) && row.options.includes(v))) {
            bad.push(`${name}.${key}=${String(v)}`);
          }
        } else if (row.kind === 'range') {
          if (
            typeof v !== 'number' ||
            (typeof row.min === 'number' && v < row.min) ||
            (typeof row.max === 'number' && v > row.max)
          ) {
            bad.push(`${name}.${key}=${String(v)}`);
          }
        } else if (row.kind === 'toggle' && typeof v !== 'boolean') {
          bad.push(`${name}.${key}=${String(v)}`);
        }
      }
    }
    expect(bad, `preset values their rows refuse: ${bad.join(' ')}`).toEqual([]);
  });

  it('the schema offers every shipped ramp and every shipped named palette', () => {
    // the two lists are built FROM theme data (schema.ts imports RAMP_NAMES/PALETTE_NAMES),
    // so this is the no-drift pin from the other side
    const schema = schemaMod['SCHEMA'];
    if (!Array.isArray(schema)) return;
    const row = (schema as Array<{ key?: unknown; options?: unknown }>).find(
      (r) => r.key === 'ramp',
    );
    expect(row, 'the schema has no ramp row').toBeDefined();
    for (const name of RAMP_NAMES) expect(row!.options as string[]).toContain(name);
    const pal = (schema as Array<{ key?: unknown; options?: unknown }>).find(
      (r) => r.key === 'palette',
    );
    expect(pal, 'the schema has no palette row').toBeDefined();
    for (const name of PALETTE_NAMES) expect(pal!.options as string[]).toContain(name);
  });
});
