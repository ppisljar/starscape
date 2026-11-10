/* The tuning panel (DESIGN.md §7, stage 8) — the optional `panel` module that renders the
 * config schema. The MECHANISM is ported from vault-site-3d src/config.js buildPanel (~line
 * 786) and its README "Tuning it"; the DOM and CSS are fresh (ss-cfg classes, CSS shipped as
 * a string export + injectPanelStyles(), like the ladder overlay — a library must not fetch
 * external files).
 *
 * The ported rules, each one earned on a real corpus:
 *
 *   - GENERATED FROM THE SCHEMA. Every control is a schema row (group, key, label, kind,
 *     range, default, unit) — add a row and it appears, no UI code. The panel never knows
 *     what a key MEANS; routing a change to a subsystem is the host's job (onChange).
 *   - LIVE ON DRAG. A range row writes through store.set on every input event; the host's
 *     onChange(key) fires immediately. No apply button — the field is the preview.
 *   - COPY IS THE DIFF. The copy button puts ONLY the changed keys on the clipboard as JSON
 *     (store.exportLook(): a diff from DEFAULTS, session keys excluded) — what paste and a
 *     saved look both expect. A diff from the current preset would only reapply on top of
 *     that same preset.
 *   - RESET RESTORES THE SHIPPED DEFAULTS (applies the default preset — an empty diff), and
 *     PASTE imports a diff from the clipboard rather than opening a prompt() — a modal blocks
 *     the render loop, and in an automated session blocks everything.
 *   - CUSTOM IS A READOUT, never a look you can apply: the moment you touch a slider the
 *     picker must say "custom" (a name still showing after four changes labels something no
 *     longer on screen), and picking "custom" changes nothing.
 *   - THE PICKER IS EXPOSED (lookSel) so a host masthead can mirror it instead of keeping a
 *     second list that drifts.
 *   - KEYBOARD TOGGLE (~), ignored while typing in an editable element.
 *
 * Persistence is the store's job: the panel calls store.save() after every change, so touched
 * keys (and the preset name) survive a reload exactly as the store's rules dictate.
 *
 * The pure halves (grouping, parsing, formatting, the picker model, copy/paste text) are
 * exported for headless tests (test/panel.test.ts) — the DOM half is thin glue over them.
 */

import { SCHEMA } from '../config/schema';
import type { ConfigValue, SchemaRow } from '../config/schema';
import { CUSTOM } from '../config/store';
import type { ConfigStore, LookList, Warnings } from '../config/store';

/* ---- pure parts (headless-testable) ------------------------------------------------------ */

/** One panel section: adjacent schema rows sharing a group (schema order is panel order). */
export interface PanelGroup {
  group: string;
  rows: SchemaRow[];
}

/** Split rows into sections by ADJACENCY (the predecessor's walk: a new header whenever the
 *  group name changes), so the schema's order is the panel's order with no second list. */
export function groupRows(rows: readonly SchemaRow[]): PanelGroup[] {
  const out: PanelGroup[] = [];
  let cur: PanelGroup | null = null;
  for (const row of rows) {
    if (cur === null || cur.group !== row.group) {
      cur = { group: row.group, rows: [] };
      out.push(cur);
    }
    cur.rows.push(row);
  }
  return out;
}

/** Decimal places needed to print multiples of `step` exactly (0.05 → 2, 1 → 0), capped at 6
 *  so a pathological step cannot produce a 17-digit readout. */
export function decimalsOf(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0;
  let d = 0;
  let s = step;
  while (d < 6 && Math.abs(s - Math.round(s)) > 1e-9) {
    s *= 10;
    d++;
  }
  return d;
}

/** The value readout: ranges print at the step's precision (a slider on step 0.05 must never
 *  read "0.15000000000000002"); selects and toggles print verbatim. */
export function formatValue(row: SchemaRow, value: ConfigValue): string {
  if (row.kind === 'range' && typeof value === 'number') {
    return value.toFixed(decimalsOf(row.step));
  }
  return String(value);
}

/** Parse a control's raw string back into the row's value type. Null means "not a value for
 *  this row" — the caller leaves the store untouched (store.set would refuse it anyway; the
 *  panel just avoids the round trip). */
export function parseRow(row: SchemaRow, raw: string): ConfigValue | null {
  switch (row.kind) {
    case 'range': {
      const v = Number(raw);
      return Number.isFinite(v) ? v : null;
    }
    case 'select':
      return row.options.includes(raw) ? raw : null;
    case 'toggle':
      return raw === 'true' ? true : raw === 'false' ? false : null;
  }
}

/** One entry of the look picker. */
export interface LookOption {
  value: string;
  label: string;
  note: string;
  kind: 'custom' | 'built' | 'saved';
}

/** The picker model: CUSTOM leads (the moment you touch a slider that is what you are looking
 *  at and the picker has to say so), then the shipped presets, then your saved looks. */
export function lookOptions(looks: LookList): LookOption[] {
  const out: LookOption[] = [
    {
      value: CUSTOM,
      label: '— custom —',
      note: 'your own settings — save them to keep them',
      kind: 'custom',
    },
  ];
  for (const [name, label, note] of looks.built) {
    out.push({ value: name, label: label || name, note, kind: 'built' });
  }
  for (const name of looks.saved) {
    out.push({ value: name, label: name, note: 'your saved look', kind: 'saved' });
  }
  return out;
}

/** What the copy button writes: ONLY the changed keys, as pretty JSON — a portable diff from
 *  DEFAULTS (session keys excluded), the exact shape paste and saved looks consume. */
export function copyText(store: ConfigStore): string {
  return JSON.stringify(store.exportLook(), null, 2);
}

/** Apply pasted text: parse, import (unknown keys / ill-typed values / session keys dropped
 *  and reported by the store), succeed only when a diff actually applied. */
export function pasteText(
  store: ConfigStore,
  text: string,
): { ok: boolean; warnings: Warnings } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, warnings: ['paste: not JSON'] };
  }
  const { applied, warnings } = store.importLook(parsed);
  return { ok: applied !== null, warnings };
}

/* ---- the stylesheet ----------------------------------------------------------------------- */

/** The panel's CSS — fresh, dark-chrome (the panel is chrome, not ground: it stays dark on a
 *  light field, like every HUD). Shipped as a string + injectPanelStyles(), the overlay's
 *  convention; consumers restyle by overriding ss-cfg-* classes. */
export const PANEL_CSS = `
.ss-cfg{position:fixed;top:10px;right:10px;bottom:10px;width:304px;z-index:30;overflow-y:auto;
  box-sizing:border-box;padding:10px 12px 14px;border:1px solid #27314a;border-radius:6px;
  background:rgba(8,11,20,.93);color:#cdd6e4;font:11px/1.5 ui-monospace,Menlo,Consolas,monospace;
  user-select:none;scrollbar-width:thin}
.ss-cfg-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.ss-cfg-t{letter-spacing:.14em;color:#8fa4c8}
.ss-cfg button{background:#0d1322;color:#8d98b5;border:1px solid #27314a;border-radius:3px;
  font:inherit;font-size:10px;padding:1px 7px;cursor:pointer}
.ss-cfg button:hover{color:#e6ecf8;border-color:#59a0ff}
.ss-cfg-a{display:flex;gap:4px}
.ss-cfg-p{display:flex;flex-direction:column;gap:3px;margin-bottom:6px}
.ss-cfg-note{color:#5c6688;font-style:italic;min-height:15px;font-size:10px}
.ss-cfg-save{display:flex;gap:4px;margin-bottom:4px}
.ss-cfg-save input{flex:1;min-width:0}
.ss-cfg select,.ss-cfg input[type=text]{background:#0d1322;color:#cdd6e4;border:1px solid #27314a;
  border-radius:3px;font:inherit;padding:1px 4px}
.ss-cfg-g{margin:11px 0 3px;letter-spacing:.16em;text-transform:uppercase;color:#5c6688;
  font-size:9.5px;border-bottom:1px solid #1a2238;padding-bottom:2px}
.ss-cfg-r{display:grid;grid-template-columns:104px 1fr 46px 40px;gap:6px;align-items:center;
  margin:2px 0}
.ss-cfg-l{color:#aab3c8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ss-cfg-v{font-weight:400;color:#e6ecf8;text-align:right;font-size:10px}
.ss-cfg-u{color:#5c6688;font-style:normal;font-size:9px}
.ss-cfg input[type=range]{width:100%;min-width:0;accent-color:#59a0ff;margin:0;height:14px}
.ss-cfg-r select{grid-column:2/span 3;width:100%}
.ss-cfg-r input[type=checkbox]{justify-self:start;accent-color:#59a0ff;margin:0}
.ss-cfg-inert{opacity:.45}
.ss-cfg-inert .ss-cfg-l{font-style:italic}
`;

/** The id the injected style tag carries — injection is idempotent. */
export const PANEL_STYLE_ID = 'starscape-panel-css';

/** Inject PANEL_CSS once (no-op when present). A consumer bundling its own styles never
 *  calls this. */
export function injectPanelStyles(doc: Document = document): void {
  if (doc.getElementById(PANEL_STYLE_ID)) return;
  const el = doc.createElement('style');
  el.id = PANEL_STYLE_ID;
  el.textContent = PANEL_CSS;
  doc.head.appendChild(el);
}

/* ---- the panel ---------------------------------------------------------------------------- */

export interface PanelOptions {
  /** The config store the panel reads and writes (all persistence flows through it). */
  store: ConfigStore;
  /** The rows to render. Default: the shipped SCHEMA. Must be the schema the store validates
   *  against, or set() will refuse the panel's own writes. */
  schema?: readonly SchemaRow[];
  /** Fired after ONE key changed through a control (live, on drag) — route it to its
   *  subsystem. The store already holds the new value. */
  onChange?: (key: string) => void;
  /** Fired after a BULK rewrite (preset/look pick, reset, paste-import): every key may have
   *  moved, so re-apply the whole routing table. */
  onBulk?: (source: 'preset' | 'reset' | 'import', name?: string) => void;
  /** Visibility toggle key. '~' (also matched as Backquote) by default; null disables. */
  hotkey?: string | null;
  /** The look `reset` returns to — its default preset should be the empty diff, making reset
   *  "the shipped defaults". Default 'field'. */
  resetTo?: string;
  /** Keys the HOST's routing does not reach in this build (the demo's UNROUTED_KEYS): their
   *  rows render greyed with an explanatory tooltip. The control still works — the value
   *  validates, persists, exports and rides looks — but the field will not change, and a
   *  live-looking dead knob reads as a rendering bug (§10.4's spirit: a gap must be visible
   *  where the user is looking, not only on a debug handle). */
  inertKeys?: readonly string[];
  /** The document to build in. Default: the global one. */
  document?: Document;
}

export interface Panel {
  /** The <aside> root — append it to your page. Starts hidden; toggle with ~ or toggle(). */
  el: HTMLElement;
  /** Show/hide (force true = show). */
  toggle(force?: boolean): void;
  /** Refresh every control, the picker value and the note from the store — call after the
   *  host changed values behind the panel's back. */
  sync(): void;
  /** Mark the settings as matching no named look (host-driven change). Returns true when the
   *  picker actually moved to CUSTOM. */
  markCustom(): boolean;
  /** Replace the inert key set and repaint the affected rows LIVE — the host calls this when a
   *  mode change makes a control momentarily meaningless (e.g. the palette is dead while colour
   *  order is 'spatial', the ramp dead while it is 'value'), so the greying tracks the real state
   *  instead of being fixed at build time. The control stays usable; only its "no live effect"
   *  styling changes. */
  setInert(keys: readonly string[]): void;
  /** The look picker, exposed so a host masthead can mirror it (one list, two places). */
  readonly lookSel: HTMLSelectElement;
  dispose(): void;
}

/** Build the tuning panel over a config store. Generated from the schema: add a row there and
 *  a control appears here, no UI code. */
export function createPanel(opts: PanelOptions): Panel {
  const store = opts.store;
  const schema = opts.schema ?? SCHEMA;
  const doc = opts.document ?? document;
  const onChange = opts.onChange ?? ((): void => {});
  const onBulk = opts.onBulk ?? ((): void => {});
  const hotkey = opts.hotkey === undefined ? '~' : opts.hotkey;
  const resetTo = opts.resetTo ?? 'field';
  const inert = new Set(opts.inertKeys ?? []);

  const el = doc.createElement('aside');
  el.className = 'ss-cfg';
  el.hidden = true;

  /* ---- header: title + copy / paste / reset / close -------------------------------------- */
  const head = doc.createElement('div');
  head.className = 'ss-cfg-h';
  const title = doc.createElement('span');
  title.className = 'ss-cfg-t';
  title.textContent = '// TUNING';
  const actions = doc.createElement('span');
  actions.className = 'ss-cfg-a';
  const btn = (label: string, titleText: string): HTMLButtonElement => {
    const b = doc.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.title = titleText;
    actions.appendChild(b);
    return b;
  };
  const copyBtn = btn('copy', 'Copy this look as JSON — only the keys you changed');
  const pasteBtn = btn('paste', 'Apply a look from JSON on the clipboard');
  const resetBtn = btn('reset', 'Back to the shipped defaults');
  const closeBtn = btn('×', 'Close (~)');
  head.appendChild(title);
  head.appendChild(actions);
  el.appendChild(head);

  /* ---- the look picker + note ------------------------------------------------------------- */
  const pickerWrap = doc.createElement('div');
  pickerWrap.className = 'ss-cfg-p';
  const sel = doc.createElement('select');
  sel.className = 'ss-cfg-look';
  const note = doc.createElement('i');
  note.className = 'ss-cfg-note';
  pickerWrap.appendChild(sel);
  pickerWrap.appendChild(note);
  el.appendChild(pickerWrap);

  /* ---- save / delete a look ---------------------------------------------------------------- */
  const saveWrap = doc.createElement('div');
  saveWrap.className = 'ss-cfg-save';
  const nameIn = doc.createElement('input');
  nameIn.type = 'text';
  nameIn.placeholder = 'name this look';
  nameIn.spellcheck = false;
  const saveBtn = doc.createElement('button');
  saveBtn.type = 'button';
  saveBtn.textContent = 'save';
  saveBtn.title = 'Save the current settings under that name';
  const dropBtn = doc.createElement('button');
  dropBtn.type = 'button';
  dropBtn.textContent = 'delete';
  dropBtn.title = 'Delete the selected saved look';
  saveWrap.appendChild(nameIn);
  saveWrap.appendChild(saveBtn);
  saveWrap.appendChild(dropBtn);
  el.appendChild(saveWrap);

  /* ---- the generated rows ------------------------------------------------------------------ */
  const body = doc.createElement('div');
  body.className = 'ss-cfg-s';
  el.appendChild(body);

  interface RowRec {
    row: SchemaRow;
    input: HTMLInputElement | HTMLSelectElement;
    label: HTMLElement;
    show: () => void;
  }
  const rows = new Map<string, RowRec>();

  /** Apply the inert styling to one row's label from the live `inert` set (shared by the initial
   *  build and setInert): greyed, italic label, explanatory tooltip when inert; plain otherwise. */
  const paintInert = (key: string, label: HTMLElement): void => {
    if (inert.has(key)) {
      label.classList.add('ss-cfg-inert');
      label.title = `${key}: no live effect in this mode — the value still saves and rides looks`;
    } else {
      label.classList.remove('ss-cfg-inert');
      label.removeAttribute('title');
    }
  };

  for (const section of groupRows(schema)) {
    const h = doc.createElement('div');
    h.className = 'ss-cfg-g';
    h.textContent = section.group;
    body.appendChild(h);
    for (const row of section.rows) {
      const label = doc.createElement('label');
      label.className = `ss-cfg-r ss-cfg-${row.kind}`;
      const l = doc.createElement('span');
      l.className = 'ss-cfg-l';
      l.textContent = row.label;
      l.title = row.key;
      paintInert(row.key, label);
      label.appendChild(l);

      let input: HTMLInputElement | HTMLSelectElement;
      if (row.kind === 'range') {
        const r = doc.createElement('input');
        r.type = 'range';
        r.min = String(row.min);
        r.max = String(row.max);
        r.step = String(row.step);
        input = r;
      } else if (row.kind === 'select') {
        const s = doc.createElement('select');
        for (const o of row.options) {
          const opt = doc.createElement('option');
          opt.value = o;
          opt.textContent = o;
          s.appendChild(opt);
        }
        input = s;
      } else {
        const c = doc.createElement('input');
        c.type = 'checkbox';
        input = c;
      }
      label.appendChild(input);

      const out = doc.createElement('b');
      out.className = 'ss-cfg-v';
      const unit = doc.createElement('i');
      unit.className = 'ss-cfg-u';
      unit.textContent = row.unit === 'none' ? '' : row.unit;
      if (row.kind !== 'select') {
        label.appendChild(out);
        label.appendChild(unit);
      }

      const show = (): void => {
        const v = store.get(row.key);
        if (row.kind === 'toggle') {
          (input as HTMLInputElement).checked = v === true;
          out.textContent = '';
        } else {
          input.value = String(v);
          out.textContent = row.kind === 'range' ? formatValue(row, v) : '';
        }
      };

      input.addEventListener('input', () => {
        const value: ConfigValue | null =
          row.kind === 'toggle'
            ? (input as HTMLInputElement).checked
            : parseRow(row, input.value);
        if (value === null) return;
        if (!store.set(row.key, value)) return;
        mark(); // one changed setting and this is no longer the look it was named after
        store.save();
        show();
        onChange(row.key);
      });

      rows.set(row.key, { row, input, label, show });
      body.appendChild(label);
    }
  }

  /* ---- picker plumbing --------------------------------------------------------------------- */

  const noteOf = new Map<string, string>();

  /** Rebuild the picker options: CUSTOM, built-ins, saved looks (lookOptions order). */
  function fillLooks(): void {
    const model = lookOptions(store.allLooks());
    noteOf.clear();
    sel.textContent = '';
    let builtGroup: HTMLOptGroupElement | null = null;
    let savedGroup: HTMLOptGroupElement | null = null;
    for (const o of model) {
      noteOf.set(o.value, o.note);
      const opt = doc.createElement('option');
      opt.value = o.value;
      opt.textContent = o.label;
      if (o.kind === 'custom') {
        sel.appendChild(opt);
      } else if (o.kind === 'built') {
        if (!builtGroup) {
          builtGroup = doc.createElement('optgroup');
          builtGroup.label = 'built in';
          sel.appendChild(builtGroup);
        }
        builtGroup.appendChild(opt);
      } else {
        if (!savedGroup) {
          savedGroup = doc.createElement('optgroup');
          savedGroup.label = 'saved';
          sel.appendChild(savedGroup);
        }
        savedGroup.appendChild(opt);
      }
    }
    syncPicker();
  }

  const showNote = (): void => {
    note.textContent = noteOf.get(sel.value) ?? '';
  };

  const hasOption = (v: string): boolean => {
    for (let i = 0; i < sel.options.length; i++) if (sel.options[i]!.value === v) return true;
    return false;
  };

  const syncPicker = (): void => {
    sel.value = hasOption(store.preset) ? store.preset : CUSTOM;
    showNote();
  };

  /** The internal custom-marker (no save — callers sequence their own persistence). */
  const mark = (): boolean => {
    if (store.preset === CUSTOM) return false;
    store.preset = CUSTOM;
    syncPicker();
    return true;
  };

  const syncAll = (): void => {
    for (const rec of rows.values()) rec.show();
    syncPicker();
  };

  /* ---- actions ------------------------------------------------------------------------------ */

  sel.addEventListener('change', () => {
    // CUSTOM is a readout, not a look — picking it changes nothing, it is already what you have
    if (sel.value === CUSTOM) {
      showNote();
      return;
    }
    // one path for both: a saved look if there is one by that name, otherwise a built-in
    store.applyLook(sel.value);
    store.save();
    syncAll();
    onBulk('preset', sel.value);
  });

  closeBtn.addEventListener('click', () => {
    el.hidden = true;
  });

  saveBtn.addEventListener('click', () => {
    const name = nameIn.value.trim();
    if (!name) {
      nameIn.focus();
      return;
    }
    store.saveLook(name); // stores the diff AND makes it the current preset
    store.save();
    nameIn.value = '';
    fillLooks();
  });

  dropBtn.addEventListener('click', () => {
    if (!store.deleteLook(sel.value)) return; // built-ins (and CUSTOM) are not yours to delete
    store.save();
    fillLooks();
  });

  copyBtn.addEventListener('click', () => {
    const txt = copyText(store);
    const clip = doc.defaultView?.navigator.clipboard;
    const done = (ok: boolean): void => {
      copyBtn.textContent = ok ? 'copied' : 'see console';
      // the fallback dumps the diff where an automated session can still read it
      if (!ok) console.log(txt);
      setTimeout(() => {
        copyBtn.textContent = 'copy';
      }, 1200);
    };
    if (clip) {
      clip.writeText(txt).then(
        () => done(true),
        () => done(false),
      );
    } else {
      done(false);
    }
  });

  pasteBtn.addEventListener('click', () => {
    /* Import by reading the clipboard rather than opening a prompt() — a modal blocks the
       render loop and, in an automated session, blocks everything. */
    const clip = doc.defaultView?.navigator.clipboard;
    if (!clip) {
      note.textContent = 'no clipboard access';
      return;
    }
    clip.readText().then(
      (text) => {
        const res = pasteText(store, text);
        if (!res.ok) {
          note.textContent = 'clipboard did not hold a look';
          return;
        }
        store.save();
        fillLooks();
        syncAll();
        onBulk('import');
      },
      () => {
        note.textContent = 'clipboard did not hold a look';
      },
    );
  });

  resetBtn.addEventListener('click', () => {
    store.applyLook(resetTo); // the default preset's diff is empty — the shipped defaults
    store.save();
    syncAll();
    onBulk('reset');
  });

  /* ---- the keyboard toggle ------------------------------------------------------------------ */

  const editable = (t: unknown): boolean => {
    const n = t as { closest?: (sel: string) => unknown } | null;
    return !!(
      n &&
      typeof n.closest === 'function' &&
      n.closest('input,textarea,select,[contenteditable=""],[contenteditable="true"]')
    );
  };
  const onKey = (ev: KeyboardEvent): void => {
    if (hotkey === null) return;
    const hit = ev.key === hotkey || (hotkey === '~' && (ev.key === '`' || ev.code === 'Backquote'));
    if (!hit || editable(ev.target)) return;
    ev.preventDefault();
    el.hidden = !el.hidden;
  };
  const win = doc.defaultView;
  if (hotkey !== null && win) win.addEventListener('keydown', onKey);

  fillLooks();
  syncAll();

  return {
    el,
    toggle(force?: boolean): void {
      el.hidden = force === undefined ? !el.hidden : !force;
    },
    sync: syncAll,
    markCustom(): boolean {
      if (!mark()) return false;
      store.save();
      return true;
    },
    setInert(keys: readonly string[]): void {
      inert.clear();
      for (const k of keys) inert.add(k);
      for (const rec of rows.values()) paintInert(rec.row.key, rec.label);
    },
    lookSel: sel,
    dispose(): void {
      if (hotkey !== null && win) win.removeEventListener('keydown', onKey);
      el.remove();
      rows.clear();
    },
  };
}
