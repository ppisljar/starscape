/* Config persistence, presets and user looks (DESIGN.md §7) — the MACHINERY of the
 * predecessor's vault-site-3d src/config.js, ported with its hard-won rules intact:
 *
 *   - ONLY TOUCHED KEYS PERSIST. Saving the whole config froze all forty-odd defaults into
 *     storage the first time anyone moved a slider, after which every default improved in the
 *     code was invisible on that machine — several rounds of "I changed the default / it looks
 *     the same" were exactly this. An untouched knob follows the code; a deliberate choice
 *     sticks (pinned by the predecessor's test/config.test.mjs, re-pinned in
 *     test/config.test.ts).
 *   - PRESETS AND LOOKS ARE DIFFS from DEFAULTS, never full dumps: a look captured today still
 *     picks up a default improved tomorrow in every knob it had no opinion about, and the
 *     stored diff doubles as the explanation of what makes the look what it is.
 *   - SESSION_KEYS NEVER PERSIST and never ride a look: they are re-derived from the corpus
 *     and device at every boot (a link budget tuned at 10⁶ is simply wrong at 10³, and
 *     restoring it presents as a rendering bug).
 *   - UNKNOWN OR ILL-TYPED STORED KEYS ARE DROPPED, NOT TRUSTED — and the drops are RETURNED
 *     as warnings, not console.warn'd: the library performs no side effects on a channel the
 *     host cannot intercept (§10.4's spirit — nothing silent, nothing noisy).
 *   - STORAGE IS INJECTED. The library never touches `window` or `localStorage` at import
 *     time; the localStorage adapter resolves lazily when ASKED, tests hand in a Map, and a
 *     host with neither (SSR, a worker) passes nothing and simply gets no persistence.
 *   - A CORRUPT BLOB IS NOT WORTH A BROKEN VIEWER: parse failures load the defaults (with a
 *     warning) and write failures (private mode, quota) are swallowed.
 */

import { PRESETS as SHIPPED_PRESETS, SCHEMA, SESSION_KEYS } from './schema';
import type { Config, ConfigValue, Preset, SchemaRow } from './schema';

/** The storage contract — localStorage's shape, satisfiable by a Map in two lines. */
export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** The browser adapter, resolved LAZILY — importing this module never touches globals, so the
 *  library loads headless (§ packaging: demo/SSR/tests all consume the same build). Returns
 *  null where no localStorage exists (node, workers, storage-disabled browsers). */
export function localStorageAdapter(): StorageLike | null {
  try {
    const g = globalThis as { localStorage?: StorageLike };
    return g.localStorage ?? null;
  } catch {
    return null; // some embeddings throw on the mere property access
  }
}

/** A Map-backed adapter — what tests inject, and a useful in-memory fallback. */
export function mapStorage(map: Map<string, string> = new Map()): StorageLike {
  return {
    getItem: (k) => (map.has(k) ? map.get(k)! : null),
    setItem: (k, v) => {
      map.set(k, String(v));
    },
    removeItem: (k) => {
      map.delete(k);
    },
  };
}

/** The picker's value when the live settings match no named look — a READOUT, never a look
 *  you can apply (applying it must be a no-op, or picking "custom" would wipe the very
 *  settings the word describes). */
export const CUSTOM = '~custom';

/** A diff from DEFAULTS: what presets, looks, export and import all speak. */
export type Diff = Config;

/** What importLook/load hand back alongside their result: every stored key that was dropped
 *  and why. Returned, never logged — the host decides whether anyone hears about it. */
export type Warnings = string[];

export interface ConfigStoreOptions {
  /** Where to persist. Omit (or pass null) for a memory-only store — everything works,
   *  nothing survives a reload. */
  storage?: StorageLike | null;
  /** The schema to validate against. Default: the shipped SCHEMA. */
  schema?: readonly SchemaRow[];
  /** Named looks that ship with the host. Default: the shipped PRESETS. */
  presets?: Readonly<Record<string, Preset>>;
  /** Keys excluded from every persistence path. Default: the shipped SESSION_KEYS. */
  sessionKeys?: readonly string[];
  /** Storage slot for the config blob. Version the key when the schema breaks compatibility
   *  (the predecessor was on :cfg:v5 for exactly this reason). */
  configKey?: string;
  /** Storage slot for saved looks. */
  looksKey?: string;
  /** The preset a fresh store boots into. Default 'field'. */
  defaultPreset?: string;
}

export interface LookList {
  /** [name, label, note] per shipped preset, in declaration order. */
  built: [string, string, string][];
  /** Saved look names, sorted. */
  saved: string[];
}

export interface ConfigStore {
  /** key → default, derived from the schema (frozen). */
  readonly defaults: Readonly<Config>;
  /** The live values. Read freely; write through set() so touch-tracking holds. */
  readonly values: Config;
  /** The named look the values started from (CUSTOM once they match no name). */
  preset: string;
  /** Keys deliberately changed away from their default — the ONLY keys save() persists. */
  readonly touched: ReadonlySet<string>;

  /** Read one value. Throws on a key the schema does not name — a typo'd read returning
   *  `undefined` would propagate NaN into a shader uniform three modules later. */
  get(key: string): ConfigValue;
  /** Write one value and mark it touched. Returns false (and changes nothing) when the key
   *  is unknown or the value fails the row's kind/options check. */
  set(key: string, value: ConfigValue): boolean;
  /** Mark a key touched without changing it (a panel that mutated `values` directly). */
  touch(key: string): void;

  /** Load from storage: defaults overlaid by the touched keys of the stored blob. Returns
   *  the warnings for everything dropped (unknown keys, ill-typed values, corrupt blob). */
  load(): Warnings;
  /** Persist the touched, non-session keys (and the preset name). */
  save(): void;

  /** Reset to DEFAULTS + the preset's diff. Everything the preset does not name returns to
   *  the default — that is what makes applying it reproducible. False when no such preset. */
  applyPreset(name: string): boolean;
  /** Store the current non-session diff from DEFAULTS under `name` (user look). */
  saveLook(name: string): void;
  /** Delete a saved look. Shipped presets are not yours to delete — returns false. */
  deleteLook(name: string): boolean;
  /** Apply a saved look, or fall through to a shipped preset of that name. CUSTOM is a
   *  readout: applying it is a no-op returning true. */
  applyLook(name: string): boolean;
  /** Everything a picker should offer: shipped presets, then saved looks. */
  allLooks(): LookList;
  /** The current config as a portable diff from DEFAULTS (session keys excluded). */
  exportLook(): Diff;
  /** Apply a pasted diff: unknown keys, ill-typed values and session keys are dropped and
   *  reported; everything else overlays DEFAULTS. Null diff (not an object) applies
   *  nothing. apply(export(x)) reproduces x for every persistable key. */
  importLook(obj: unknown): { applied: Diff | null; warnings: Warnings };
}

/** Does `value` fit `row`? Ranges want finite numbers, selects one of their own options,
 *  toggles booleans. Min/max are panel rails, not validation — the predecessor let a pasted
 *  look overshoot a slider deliberately (the rail is a UI affordance, the value a choice).
 *  A row's `floor` is NOT a rail: it is the subsystem's own domain (its setter throws at or
 *  below it — schema.ts RangeRow), so a value past it is refused here, where it can be
 *  dropped with a warning, instead of detonating in a Field setter after it was persisted
 *  (a pasted exposure 0 used to crash every boot until localStorage was cleared by hand). */
export function validValue(row: SchemaRow, value: ConfigValue): boolean {
  switch (row.kind) {
    case 'range':
      return (
        typeof value === 'number' &&
        Number.isFinite(value) &&
        (row.floor === undefined || value > row.floor)
      );
    case 'select':
      return typeof value === 'string' && row.options.includes(value);
    case 'toggle':
      return typeof value === 'boolean';
  }
}

interface StoredBlob {
  preset?: unknown;
  touched?: unknown;
  v?: unknown;
}

export function createConfigStore(opts: ConfigStoreOptions = {}): ConfigStore {
  const schema = opts.schema ?? SCHEMA;
  const presets = opts.presets ?? SHIPPED_PRESETS;
  const sessionKeys = new Set(opts.sessionKeys ?? SESSION_KEYS);
  const storage = opts.storage ?? null;
  const configKey = opts.configKey ?? 'starscape:cfg:v1';
  const looksKey = opts.looksKey ?? 'starscape:looks:v1';
  const defaultPreset = opts.defaultPreset ?? 'field';

  const rowOf = new Map<string, SchemaRow>();
  const defaults: Config = {};
  for (const row of schema) {
    if (rowOf.has(row.key)) throw new Error(`createConfigStore: duplicate key "${row.key}"`);
    rowOf.set(row.key, row);
    defaults[row.key] = row.default;
  }
  const values: Config = { ...defaults };
  const touched = new Set<string>();
  let preset = defaultPreset;

  const read = (slot: string): string | null => {
    if (!storage) return null;
    try {
      return storage.getItem(slot);
    } catch {
      return null;
    }
  };
  const write = (slot: string, blob: string): void => {
    if (!storage) return;
    try {
      storage.setItem(slot, blob);
    } catch {
      /* private mode / quota — persistence is a nicety, never worth an exception */
    }
  };

  /** Overlay a diff onto DEFAULTS, validating each entry; returns the accepted diff.
   *  Session keys are dropped silently when `warnSession` is false (loading one back is
   *  normal staleness) and reported when true (an import claiming one is user-visible). */
  const overlay = (
    diff: Record<string, unknown>,
    warnings: Warnings,
    source: string,
    warnSession: boolean,
  ): Diff => {
    const applied: Diff = {};
    for (const key of Object.keys(diff)) {
      const row = rowOf.get(key);
      if (!row) {
        warnings.push(`${source}: unknown key "${key}" dropped`);
        continue;
      }
      if (sessionKeys.has(key)) {
        if (warnSession) warnings.push(`${source}: session key "${key}" dropped (never persisted)`);
        continue;
      }
      const v = diff[key];
      if (!validValue(row, v as ConfigValue)) {
        warnings.push(`${source}: "${key}" holds ${JSON.stringify(v)}, not a ${row.kind} value — dropped`);
        continue;
      }
      applied[key] = v as ConfigValue;
    }
    /* Reset every NON-session key to its default, then overlay the diff. SESSION_KEYS
       (linkBudget / renderScale / flatPx) are orthogonal to the LOOK: the corpus and device
       chose them at boot (the §top invariant, "never ride a look"), and a look neither sets
       them — overlay dropped them from the incoming diff above — NOR clears them here. Resetting
       them to their 1:1 defaults on every applyPreset/applyLook silently DEFEATED the boot perf
       notch: picking the nebula look reset linkBudget 0.04→1 and renderScale 0.5→1, so the
       consumer's applyAll then drew all 3.9M links at dpr 2 and tanked the framerate zoomed in. */
    for (const k of Object.keys(defaults)) {
      if (!sessionKeys.has(k)) values[k] = defaults[k]!;
    }
    for (const k of [...touched]) {
      if (!sessionKeys.has(k)) touched.delete(k);
    }
    for (const k of Object.keys(applied)) {
      values[k] = applied[k]!;
      touched.add(k);
    }
    return applied;
  };

  /** The current non-session diff from the defaults — what a look stores and export emits. */
  const diffNow = (): Diff => {
    const out: Diff = {};
    for (const key of Object.keys(defaults)) {
      if (sessionKeys.has(key)) continue;
      if (values[key] !== defaults[key]) out[key] = values[key]!;
    }
    return out;
  };

  const loadLooks = (): Record<string, Diff> => {
    const raw = read(looksKey);
    if (raw === null) return {};
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
      const out: Record<string, Diff> = {};
      for (const [name, v] of Object.entries(parsed as Record<string, unknown>)) {
        if (v !== null && typeof v === 'object' && !Array.isArray(v)) out[name] = v as Diff;
      }
      return out;
    } catch {
      return {};
    }
  };
  const writeLooks = (looks: Record<string, Diff>): void => {
    write(looksKey, JSON.stringify(looks));
  };

  const store: ConfigStore = {
    defaults,
    values,
    get preset(): string {
      return preset;
    },
    set preset(name: string) {
      preset = name;
    },
    touched,

    get(key: string): ConfigValue {
      if (!rowOf.has(key)) throw new Error(`config: unknown key "${key}"`);
      return values[key]!;
    },

    set(key: string, value: ConfigValue): boolean {
      const row = rowOf.get(key);
      if (!row || !validValue(row, value)) return false;
      values[key] = value;
      touched.add(key);
      return true;
    },

    touch(key: string): void {
      if (rowOf.has(key)) touched.add(key);
    },

    load(): Warnings {
      const warnings: Warnings = [];
      const raw = read(configKey);
      if (raw === null) {
        overlay({}, warnings, 'load', false);
        preset = defaultPreset;
        return warnings;
      }
      let blob: StoredBlob = {};
      try {
        const parsed: unknown = JSON.parse(raw);
        if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
          blob = parsed as StoredBlob;
        } else {
          warnings.push('load: stored config is not an object — defaults loaded');
        }
      } catch {
        warnings.push('load: stored config is not JSON — defaults loaded');
      }
      const saved =
        blob.v !== null && typeof blob.v === 'object' && !Array.isArray(blob.v)
          ? (blob.v as Record<string, unknown>)
          : {};
      /* The touched list decides what loads; keys present in `v` but not in `touched` stay
         on the code's default (predecessor semantics). An old blob without a touched list
         treats every saved key as touched. */
      const touchedList = Array.isArray(blob.touched)
        ? (blob.touched as unknown[]).filter((k): k is string => typeof k === 'string')
        : Object.keys(saved);
      const diff: Record<string, unknown> = {};
      for (const k of touchedList) {
        if (k in saved) diff[k] = saved[k];
        else warnings.push(`load: touched key "${k}" has no stored value — dropped`);
      }
      for (const k of Object.keys(saved)) {
        if (!touchedList.includes(k)) warnings.push(`load: untouched key "${k}" ignored (follows the code)`);
      }
      overlay(diff, warnings, 'load', false);
      preset = typeof blob.preset === 'string' ? blob.preset : defaultPreset;
      return warnings;
    },

    save(): void {
      const keys: string[] = [];
      const v: Config = {};
      for (const k of touched) {
        if (sessionKeys.has(k)) continue;
        keys.push(k);
        v[k] = values[k]!;
      }
      write(configKey, JSON.stringify({ preset, touched: keys, v }));
    },

    applyPreset(name: string): boolean {
      const p = presets[name];
      if (!p) return false;
      // Shipped presets are validated by tests, not at every apply: overlay still guards.
      overlay(p.set as Record<string, unknown>, [], `preset ${name}`, false);
      preset = name;
      return true;
    },

    saveLook(name: string): void {
      const looks = loadLooks();
      looks[name] = diffNow();
      writeLooks(looks);
      preset = name;
    },

    deleteLook(name: string): boolean {
      const looks = loadLooks();
      if (!(name in looks)) return false;
      delete looks[name];
      writeLooks(looks);
      if (preset === name) preset = defaultPreset;
      return true;
    },

    applyLook(name: string): boolean {
      if (!name || name === CUSTOM) return true; // a readout, not a look — never a reset
      const look = loadLooks()[name];
      if (look) {
        overlay(look as Record<string, unknown>, [], `look ${name}`, false);
        preset = name;
        return true;
      }
      return store.applyPreset(name);
    },

    allLooks(): LookList {
      return {
        built: Object.entries(presets).map(([k, p]) => [k, p.label || k, p.note || '']),
        saved: Object.keys(loadLooks()).sort(),
      };
    },

    exportLook(): Diff {
      return diffNow();
    },

    importLook(obj: unknown): { applied: Diff | null; warnings: Warnings } {
      if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
        return { applied: null, warnings: ['import: not a config diff'] };
      }
      const warnings: Warnings = [];
      const applied = overlay(obj as Record<string, unknown>, warnings, 'import', true);
      preset = defaultPreset;
      return { applied, warnings };
    },
  };

  return store;
}
