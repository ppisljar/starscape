/* The tuning-panel module (DESIGN.md §7/§8, stage 8): the schema-generated panel (./panel)
 * plus the config engine it renders — schema rows, defaults, presets, the persisting store —
 * re-exported here so `starscape/panel` is self-sufficient for a consumer wiring a panel.
 * One name is adjusted at this barrel: the schema's palette OPTION LIST ships as
 * PALETTE_OPTIONS so it cannot collide with the theme module's PALETTES data record. */

export * from './panel';

export {
  BACKGROUNDS,
  BLENDS,
  COLOR_MODES,
  DEFAULTS,
  PALETTES as PALETTE_OPTIONS,
  PRESETS,
  SCHEMA,
  SESSION_KEYS,
  SIZE_CHANNELS,
  TONEMAPS,
  deriveDefaults,
} from '../config/schema';
export type {
  Config,
  ConfigValue,
  Preset,
  RangeRow,
  SchemaRow,
  SelectRow,
  ToggleRow,
  Unit,
} from '../config/schema';

export {
  CUSTOM,
  createConfigStore,
  localStorageAdapter,
  mapStorage,
  validValue,
} from '../config/store';
export type {
  ConfigStore,
  ConfigStoreOptions,
  Diff,
  LookList,
  StorageLike,
  Warnings,
} from '../config/store';
