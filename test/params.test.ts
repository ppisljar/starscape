/* Param-metadata tests (DESIGN.md §3.3, §7): the self-describing `ParamDesc[]` each shape /
 * density / scatter declares, exposed by `shapeParams` / `densityParams` / `scatterParams`.
 *
 * The load-bearing invariant is THE DEFAULT CONTRACT (src/place/types.ts): a param left at
 * its declared `default` must produce output byte-identical to omitting it — otherwise a
 * panel that seeds every control at `default` would silently change an untouched scene. We
 * prove it at the knob level: for every plain `?? default` param, running the knob with
 * `{[key]: default}` equals running it with `{}`.
 *
 * Three knobs read a param whose fallback is NOT a constant and so cannot satisfy that
 * equality by a sent value — they are tested for metadata presence / documented default and
 * excluded from the equality sweep (a consumer omits them at default instead, per each
 * knob's doc): ball.radialPower (a `cbrt`, not `pow(⅓)`), dendrite.tiers (count-derived
 * auto), and mixture.m<i> (presence flags — omitting m0 throws).
 */

import { describe, expect, it } from 'vitest';
import { placeGroup } from '../src/place/engine';
import '../src/place/shapes';
import '../src/place/densities';
import '../src/place/scatters';
import '../src/place/orderings';
import {
  densityParams,
  getDensity,
  getScatter,
  orderingKeys,
  scatterParams,
  shapeKeys,
  shapeParams,
} from '../src/place/registry';
import type { Params, PlacementSpec } from '../src/place/types';

const R = 10;
const SALT = 7;

/** Place members 0..n-1 (ascending) with a shape + params, byIndex/uniform/none around it. */
function place(shape: string, params: Params, n = 64): Float32Array {
  const members = new Uint32Array(n);
  for (let i = 0; i < n; i++) members[i] = i;
  const out = new Float32Array(n * 3);
  const spec: PlacementSpec = { shape, params };
  placeGroup(spec, { members, count: n, graph: null, salt: SALT, radius: R, out, outOffset: 0 });
  return out;
}

const bytesEqual = (a: Float32Array, b: Float32Array): boolean => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

/* Shapes whose declared defaults are plain `?? default` fallbacks and whose param set is
 * fully equality-testable by the sweep below (the exceptions get their own cases). */
const SHAPE_EXCLUDE: Record<string, Set<string>> = {
  ball: new Set(['radialPower']), // cbrt, not pow(⅓) — see module header
  dendrite: new Set(['tiers']), // count-derived auto — see module header
};

describe('param metadata — retrievability', () => {
  it('paramless knobs report an empty array', () => {
    expect(shapeParams('ring')).toEqual([]);
    expect(shapeParams('line')).toEqual([]);
    expect(shapeParams('shell')).toEqual([]);
    expect(densityParams('uniform')).toEqual([]);
    expect(scatterParams('none')).toEqual([]);
  });

  it('declared knobs hand back their ParamDesc list by key', () => {
    expect(shapeParams('helix').map((p) => p.key)).toEqual(['turns', 'coil']);
    expect(shapeParams('spiralArm').map((p) => p.key)).toEqual(['arms', 'swirl', 'bar', 'radialPower']);
    expect(shapeParams('torus').map((p) => p.key)).toEqual(['tube']);
    expect(shapeParams('box').map((p) => p.key)).toEqual(['aspectY', 'aspectZ']);
    expect(shapeParams('cloud').map((p) => p.key)).toEqual(['sigma']);
    expect(shapeParams('torusKnot').map((p) => p.key)).toEqual(['p', 'q', 'tube']);
    expect(shapeParams('superShape').map((p) => p.key)).toEqual(['m', 'n1', 'n2', 'n3']);
    expect(shapeParams('phyllotaxis').map((p) => p.key)).toEqual(['jitter']);
    expect(shapeParams('cone').map((p) => p.key)).toEqual(['aperture', 'hollow']);
    expect(densityParams('gaussian').map((p) => p.key)).toEqual(['center', 'sigma']);
    expect(densityParams('beta').map((p) => p.key)).toEqual(['a', 'b']);
    expect(densityParams('edges').map((p) => p.key)).toEqual(['strength']);
    expect(densityParams('triangular').map((p) => p.key)).toEqual(['peak']);
    expect(scatterParams('uniform').map((p) => p.key)).toEqual(['amount']);
    expect(scatterParams('turbulence').map((p) => p.key)).toEqual(['scale', 'amp', 'octaves']);
    expect(scatterParams('swirl').map((p) => p.key)).toEqual(['strength']);
    expect(densityParams('mixture').map((p) => p.key)).toEqual([
      'm0', 's0', 'w0', 'm1', 's1', 'w1', 'm2', 's2', 'w2', 'm3', 's3', 'w3',
    ]);
    expect(scatterParams('noiseField').map((p) => p.key)).toEqual(['scale', 'amp']);
  });

  it('every ParamDesc is well-formed (finite default within [min, max], positive step)', () => {
    for (const key of shapeKeys()) {
      for (const p of shapeParams(key)) {
        expect(Number.isFinite(p.default), `${key}.${p.key} default`).toBe(true);
        expect(p.min).toBeLessThanOrEqual(p.default);
        expect(p.default).toBeLessThanOrEqual(p.max);
        expect(p.step).toBeGreaterThan(0);
        expect(typeof p.label).toBe('string');
      }
    }
  });

  it('orderingKeys enumerates the registered orderings', () => {
    expect(orderingKeys()).toEqual(
      ['byAdjacency', 'byColumn', 'byFacet', 'byHash', 'byIndex'],
    );
  });
});

describe('THE DEFAULT CONTRACT — a param at its default equals omitting it', () => {
  it('shapes: place({[key]: default}) is byte-identical to place({})', () => {
    for (const key of shapeKeys()) {
      const descs = shapeParams(key);
      if (descs.length === 0) continue;
      const base = place(key, {});
      const exclude = SHAPE_EXCLUDE[key];
      for (const p of descs) {
        if (exclude?.has(p.key)) continue;
        const withDefault = place(key, { [p.key]: p.default });
        expect(bytesEqual(withDefault, base), `shape ${key}.${p.key}=${p.default}`).toBe(true);
      }
    }
  });

  it('densities: warp(q, {[key]: default}) equals warp(q, {}) across q', () => {
    const qs = [0.01, 0.17, 0.33, 0.5, 0.66, 0.83, 0.99];
    for (const key of ['gaussian', 'exponential', 'powerLaw', 'beta', 'edges', 'triangular'] as const) {
      const d = getDensity(key);
      for (const p of densityParams(key)) {
        for (const q of qs) {
          expect(d.warp(q, { [p.key]: p.default }), `${key}.${p.key} @ q=${q}`).toBe(
            d.warp(q, {}),
          );
        }
      }
    }
  });

  it('mixture: s<i>/w<i> at default do not move the warp (their component present)', () => {
    const d = getDensity('mixture');
    const qs = [0.05, 0.3, 0.5, 0.7, 0.95];
    // all four components on, so every s<i>/w<i> is read; adding one at its default is a no-op.
    const base: Params = { m0: 0, m1: 0.35, m2: 0.65, m3: 1 };
    for (const p of densityParams('mixture')) {
      if (p.key.startsWith('m')) continue; // means are presence flags, not ?? defaults
      for (const q of qs) {
        expect(d.warp(q, { ...base, [p.key]: p.default }), `mixture ${p.key} @ q=${q}`).toBe(
          d.warp(q, base),
        );
      }
    }
  });

  it('scatters: offset with {[key]: default} equals offset with {}', () => {
    for (const key of ['gaussian', 'discFalloff', 'noiseField', 'uniform', 'turbulence', 'swirl'] as const) {
      const s = getScatter(key);
      const seed = (): Float32Array => Float32Array.from([2.5, -1.25, 3.75]);
      for (const p of scatterParams(key)) {
        const a = seed();
        const b = seed();
        s.offset(3, 42, SALT, R, a, 0, { [p.key]: p.default });
        s.offset(3, 42, SALT, R, b, 0, {});
        expect(bytesEqual(a, b), `scatter ${key}.${p.key}=${p.default}`).toBe(true);
      }
    }
  });
});

describe('documented non-constant defaults', () => {
  it('ball.radialPower default is ⅓ (the cbrt value) and omitting it keeps the exact cbrt', () => {
    const rp = shapeParams('ball').find((p) => p.key === 'radialPower')!;
    expect(rp.default).toBe(1 / 3);
    // omitting radialPower uses Math.cbrt; sending ⅓ uses Math.pow — close but not pinned
    // equal, which is exactly why a consumer omits it at default (module header).
  });

  it('dendrite.tiers default is the 0 "auto" sentinel', () => {
    const t = shapeParams('dendrite').find((p) => p.key === 'tiers')!;
    expect(t.default).toBe(0);
  });

  it('mixture needs m0 — omitting every mean throws, so m<i> are presence flags', () => {
    expect(() => getDensity('mixture').warp(0.5, {})).toThrow();
    expect(() => getDensity('mixture').warp(0.5, { m0: 0 })).not.toThrow();
  });
});
