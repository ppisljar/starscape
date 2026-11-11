/* The tuning panel's PURE parts (stage 8) — everything that runs headless: schema grouping,
 * control parsing/formatting, the look-picker model, and the copy/paste text path (the diff
 * discipline: copy emits ONLY the changed keys, paste applies a diff over DEFAULTS). The DOM
 * half is thin glue over these, exercised in the browser (the stage's validation pass).
 */
import { describe, expect, it } from 'vitest';
import { DEFAULTS, SCHEMA, SESSION_KEYS } from '../src/config/schema';
import type { RangeRow, SchemaRow } from '../src/config/schema';
import { CUSTOM, createConfigStore, mapStorage } from '../src/config/store';
import { VISUAL_PRESETS } from '../src/theme/presets';
import {
  PANEL_CSS,
  PANEL_STYLE_ID,
  copyText,
  decimalsOf,
  formatValue,
  groupRows,
  lookOptions,
  parseRow,
  pasteText,
} from '../src/panel/panel';

const freshStore = () =>
  createConfigStore({ storage: mapStorage(), presets: VISUAL_PRESETS });

describe('groupRows', () => {
  it('covers every schema row exactly once, in schema order', () => {
    const groups = groupRows(SCHEMA);
    const flat: SchemaRow[] = [];
    for (const g of groups) {
      for (const row of g.rows) {
        expect(row.group).toBe(g.group); // a row never lands under a foreign header
        flat.push(row);
      }
    }
    expect(flat).toEqual([...SCHEMA]); // same rows, same order — the schema IS the panel order
  });

  it('splits by ADJACENCY: a new header whenever the group name changes', () => {
    const rows = SCHEMA.slice();
    const groups = groupRows(rows);
    let k = 0;
    for (let i = 1; i < rows.length; i++) {
      if (rows[i]!.group !== rows[i - 1]!.group) k++;
    }
    expect(groups.length).toBe(k + 1);
  });
});

describe('decimalsOf / formatValue', () => {
  it('prints at the step precision — a slider must never read 0.15000000000000002', () => {
    expect(decimalsOf(1)).toBe(0);
    expect(decimalsOf(0.1)).toBe(1);
    expect(decimalsOf(0.05)).toBe(2);
    expect(decimalsOf(0.002)).toBe(3);
    expect(decimalsOf(0.0125)).toBe(4);
    expect(decimalsOf(50)).toBe(0);
    expect(decimalsOf(0)).toBe(0); // degenerate steps fall back to integers
    expect(decimalsOf(Infinity)).toBe(0);
  });

  it('formats ranges by their own step, selects and toggles verbatim', () => {
    const range: RangeRow = {
      group: 'g', key: 'k', label: 'k', kind: 'range',
      min: 0, max: 1, step: 0.05, default: 0, unit: 'none',
    };
    expect(formatValue(range, 0.1 + 0.05)).toBe('0.15'); // the float that prints as 0.15000…2
    expect(formatValue(range, 2)).toBe('2.00');
    const select = SCHEMA.find((r) => r.kind === 'select')!;
    expect(formatValue(select, 'additive')).toBe('additive');
    const toggle = SCHEMA.find((r) => r.kind === 'toggle')!;
    expect(formatValue(toggle, true)).toBe('true');
  });

  it('every shipped range default round-trips through its own format', () => {
    for (const row of SCHEMA) {
      if (row.kind !== 'range') continue;
      // An optional-cap row whose default sits at/above offAt reads "off" by design — a
      // display-only word the panel never parses back (it writes the slider's number), so it
      // is exempt from the numeric round-trip. It must actually SAY "off".
      if (row.offAt !== undefined && row.default >= row.offAt) {
        expect(formatValue(row, row.default)).toBe('off');
        continue;
      }
      // formatting the default then parsing it back must not move the value: the panel
      // writes what it shows, and a lossy readout would mark untouched knobs as touched
      const printed = formatValue(row, row.default);
      expect(Number(printed)).toBeCloseTo(row.default, 6);
    }
  });

  it('an optional-cap row prints "off" at/above offAt and a number below it', () => {
    const cap: RangeRow = {
      group: 'g', key: 'k', label: 'k', kind: 'range',
      min: 40, max: 2000, step: 10, default: 2000, offAt: 2000, unit: 'px',
    };
    expect(formatValue(cap, 2000)).toBe('off');
    expect(formatValue(cap, 5000)).toBe('off');
    expect(formatValue(cap, 1990)).toBe('1990');
    expect(formatValue(cap, 400)).toBe('400');
  });
});

describe('parseRow', () => {
  const range = SCHEMA.find((r): r is RangeRow => r.kind === 'range')!;
  const select = SCHEMA.find((r) => r.kind === 'select')!;
  const toggle = SCHEMA.find((r) => r.kind === 'toggle')!;

  it('ranges parse finite numbers and refuse the rest', () => {
    expect(parseRow(range, '2.5')).toBe(2.5);
    expect(parseRow(range, 'abc')).toBeNull();
    expect(parseRow(range, 'Infinity')).toBeNull();
  });

  it('selects accept only their own options', () => {
    const opt = (select as { options: readonly string[] }).options[0]!;
    expect(parseRow(select, opt)).toBe(opt);
    expect(parseRow(select, 'no-such-option')).toBeNull();
  });

  it('toggles parse the two boolean words', () => {
    expect(parseRow(toggle, 'true')).toBe(true);
    expect(parseRow(toggle, 'false')).toBe(false);
    expect(parseRow(toggle, 'yes')).toBeNull();
  });
});

describe('lookOptions', () => {
  it('CUSTOM leads, built-ins next (store order), saved looks last', () => {
    const store = freshStore();
    store.set('bodyR', 0.3);
    store.saveLook('mine');
    const opts = lookOptions(store.allLooks());
    expect(opts[0]!.value).toBe(CUSTOM);
    expect(opts[0]!.kind).toBe('custom');
    const builtNames = opts.filter((o) => o.kind === 'built').map((o) => o.value);
    expect(builtNames).toEqual(Object.keys(VISUAL_PRESETS));
    const saved = opts.filter((o) => o.kind === 'saved');
    expect(saved.map((o) => o.value)).toEqual(['mine']);
    // notes ride along — the picker's explanation line comes from the same single source
    expect(opts.find((o) => o.value === 'nebula')!.note).toBe(VISUAL_PRESETS.nebula.note);
  });
});

describe('copyText — ONLY the changed keys', () => {
  it('emits exactly the touched non-session diff, as JSON', () => {
    const store = freshStore();
    store.set('bodyR', 0.3);
    store.set('blend', 'additive');
    store.set('cameraDrift', true);
    const parsed = JSON.parse(copyText(store)) as Record<string, unknown>;
    expect(parsed).toEqual({ bodyR: 0.3, blend: 'additive', cameraDrift: true });
  });

  it('never carries a session key, and an untouched store copies as {}', () => {
    const store = freshStore();
    expect(JSON.parse(copyText(store))).toEqual({});
    for (const k of SESSION_KEYS) store.set(k, 0.5);
    expect(JSON.parse(copyText(store))).toEqual({}); // linkBudget & co are session-scoped
  });

  it('a key dragged back to its default drops OUT of the copy — the diff is live, not a log', () => {
    const store = freshStore();
    store.set('haloPx', 9);
    store.set('haloPx', DEFAULTS['haloPx']!);
    expect(JSON.parse(copyText(store))).toEqual({});
  });
});

describe('pasteText — the copy round trip', () => {
  it('paste(copy(x)) reproduces x for every persistable key', () => {
    const a = freshStore();
    a.set('bodyR', 0.31);
    a.set('tonemap', 'agx');
    a.set('labelCap', 70);
    const b = freshStore();
    const res = pasteText(b, copyText(a));
    expect(res.ok).toBe(true);
    expect(res.warnings).toEqual([]);
    expect(b.get('bodyR')).toBe(0.31);
    expect(b.get('tonemap')).toBe('agx');
    expect(b.get('labelCap')).toBe(70);
    // everything the diff does not name follows the code — paste is an overlay on DEFAULTS
    expect(b.get('haloPx')).toBe(DEFAULTS['haloPx']);
    expect([...b.touched].sort()).toEqual(['bodyR', 'labelCap', 'tonemap']);
  });

  it('drops unknown keys and session keys WITH warnings, keeps the rest', () => {
    const store = freshStore();
    const res = pasteText(store, JSON.stringify({ bodyR: 0.4, nope: 1, linkBudget: 0.1 }));
    expect(res.ok).toBe(true);
    expect(store.get('bodyR')).toBe(0.4);
    expect(res.warnings.some((w) => w.includes('nope'))).toBe(true);
    expect(res.warnings.some((w) => w.includes('linkBudget'))).toBe(true);
    expect(store.get('linkBudget')).toBe(DEFAULTS['linkBudget']);
  });

  it('refuses non-JSON and non-object clipboards without touching the store', () => {
    const store = freshStore();
    store.set('bodyR', 0.33);
    expect(pasteText(store, 'not json').ok).toBe(false);
    expect(pasteText(store, '[1,2,3]').ok).toBe(false);
    expect(store.get('bodyR')).toBe(0.33); // a failed paste is not a reset
  });
});

describe('the stylesheet export', () => {
  it('ships as a string with the ss-cfg classes and a stable style id', () => {
    expect(PANEL_CSS).toContain('.ss-cfg');
    expect(PANEL_CSS).toContain('.ss-cfg-r');
    expect(PANEL_STYLE_ID).toBe('starscape-panel-css');
  });
});
