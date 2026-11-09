/* The placement registries (DESIGN.md §3.3): one per knob kind — shape, ordering, density,
 * scatter. Primitives, presets and user extensions all land in the same four maps, which is
 * the extension story: `registerShape(myShape)` and `{ shape: 'myShape' }` just works.
 *
 * Lookups throw on an unknown key, naming the kind and the registered keys — a misspelled
 * knob must fail at compile-the-spec time with a message you can act on, not place nothing
 * and render a blank group.
 *
 * Re-registering a key OVERWRITES (last registration wins). That is deliberate: presets and
 * apps are allowed to replace a built-in ("my `cloud` is flatter"), and the alternative —
 * throwing on collision — would make every hot-reload a crash.
 */

import type { Shape, Ordering, Density, Scatter } from './types';

/** One knob kind's registry: a Map plus the error message that names it. */
function makeRegistry<T extends { key: string }>(kind: string) {
  const map = new Map<string, T>();
  return {
    register(entry: T): void {
      map.set(entry.key, entry);
    },
    get(key: string): T {
      const hit = map.get(key);
      if (hit === undefined) {
        const known = [...map.keys()].sort().join(', ');
        throw new Error(`unknown ${kind} "${key}" — known: ${known || '(none)'}`);
      }
      return hit;
    },
    /** The registered keys, sorted — for panels and error reporting. */
    keys(): string[] {
      return [...map.keys()].sort();
    },
  };
}

const shapes = makeRegistry<Shape>('shape');
const orderings = makeRegistry<Ordering>('ordering');
const densities = makeRegistry<Density>('density');
const scatters = makeRegistry<Scatter>('scatter');

/** Register a shape under its own `key` (overwrites an existing registration). */
export const registerShape = (s: Shape): void => shapes.register(s);
/** Resolve a shape key; throws naming the kind and the known keys when absent. */
export const getShape = (key: string): Shape => shapes.get(key);
/** The registered shape keys, sorted. */
export const shapeKeys = (): string[] => shapes.keys();

/** Register an ordering under its own `key` (overwrites an existing registration). */
export const registerOrdering = (o: Ordering): void => orderings.register(o);
/** Resolve an ordering key; throws naming the kind and the known keys when absent. */
export const getOrdering = (key: string): Ordering => orderings.get(key);
/** The registered ordering keys, sorted. */
export const orderingKeys = (): string[] => orderings.keys();

/** Register a density under its own `key` (overwrites an existing registration). */
export const registerDensity = (d: Density): void => densities.register(d);
/** Resolve a density key; throws naming the kind and the known keys when absent. */
export const getDensity = (key: string): Density => densities.get(key);
/** The registered density keys, sorted. */
export const densityKeys = (): string[] => densities.keys();

/** Register a scatter under its own `key` (overwrites an existing registration). */
export const registerScatter = (s: Scatter): void => scatters.register(s);
/** Resolve a scatter key; throws naming the kind and the known keys when absent. */
export const getScatter = (key: string): Scatter => scatters.get(key);
/** The registered scatter keys, sorted. */
export const scatterKeys = (): string[] => scatters.keys();
