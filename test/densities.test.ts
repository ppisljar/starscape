/* The density library (src/place/densities.ts): gaussian, exponential, powerLaw, mixture
 * — inverse-CDF warps on [0,1) (DESIGN §3.3 knob 3).
 *
 * The load-bearing suites: each density's warp of 10k equal-spaced quantiles reproduces
 * its analytic CDF (inverse-transform sampling is exact, so the bounds are deterministic
 * and tight); ring + gaussian produces §3.3's canonical bunching — max bucket at the
 * center, min at the antipode; every output survives the engine's Float32 u buffer
 * strictly below 1; FNV goldens pin the bytes (§10.9).
 */

import { describe, expect, it } from 'vitest';
import { fnv1a } from '../src/core/fnv';
import { placeGroup } from '../src/place/engine';
import { getDensity, registerShape } from '../src/place/registry';
import '../src/place/densities';
import type { Density, Params, PlaceInput, Shape } from '../src/place/types';

const N = 10_000;

/** warp applied over N equal-spaced quantiles (k+0.5)/N — exactly what the engine feeds. */
function warped(d: Density, params: Params, n = N): Float64Array {
  const w = new Float64Array(n);
  for (let k = 0; k < n; k++) w[k] = d.warp((k + 0.5) / n, params);
  return w;
}

/** Empirical CDF of `ws` at x. */
function empiricalCdf(ws: Float64Array, x: number): number {
  let c = 0;
  for (let k = 0; k < ws.length; k++) if (ws[k]! <= x) c++;
  return c / ws.length;
}

/** Φ, the standard normal CDF — the test's own analytic reference (erf by
 *  Abramowitz–Stegun; the gaussian density under test goes through probitApprox, a
 *  different code path, so the comparison is between two independent routes). */
function phi(z: number): number {
  const x = Math.abs(z) * Math.SQRT1_2;
  const t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return z < 0 ? 0.5 * (1 - erf) : 0.5 * (1 + erf);
}

/** |empirical − analytic| at nine interior checkpoints. Inverse-transform sampling makes
 *  the empirical CDF exact to 1/N, so 2e-3 is loose yet fully deterministic. */
function expectMatchesCdf(ws: Float64Array, F: (x: number) => number): void {
  for (let i = 1; i <= 9; i++) {
    const x = i / 10;
    expect(Math.abs(empiricalCdf(ws, x) - F(x))).toBeLessThan(2e-3);
  }
}

const gaussian = getDensity('gaussian');
const exponential = getDensity('exponential');
const powerLaw = getDensity('powerLaw');
const mixture = getDensity('mixture');
const uniform = getDensity('uniform');

/* ---------------------------------------------------------------- CDF agreement */

describe('empirical histogram matches the analytic CDF', () => {
  it('uniform: identity', () => {
    expectMatchesCdf(warped(uniform, {}), (x) => x);
  });

  it('gaussian(0.5, 0.1): Φ((x−c)/σ)', () => {
    expectMatchesCdf(warped(gaussian, { center: 0.5, sigma: 0.1 }), (x) => phi((x - 0.5) / 0.1));
  });

  it('exponential(3): truncated-renormalised 1−e^(−λx)', () => {
    const lam = 3;
    const Z = 1 - Math.exp(-lam);
    expectMatchesCdf(warped(exponential, { rate: lam }), (x) => (1 - Math.exp(-lam * x)) / Z);
  });

  it('exponential(−2): mirrored — crowds toward 1', () => {
    const lam = -2;
    const Z = 1 - Math.exp(-lam);
    expectMatchesCdf(warped(exponential, { rate: lam }), (x) => (1 - Math.exp(-lam * x)) / Z);
  });

  it('powerLaw(0.5): F(x) = x^0.5 (mass at the core)', () => {
    expectMatchesCdf(warped(powerLaw, { alpha: 0.5 }), (x) => Math.sqrt(x));
  });

  it('powerLaw(2): F(x) = x² (mass at the rim)', () => {
    expectMatchesCdf(warped(powerLaw, { alpha: 2 }), (x) => x * x);
  });

  it('mixture(two bumps): renormalised weighted Φ sum', () => {
    const params = { m0: 0.3, s0: 0.08, w0: 2, m1: 0.75, s1: 0.05, w1: 1 };
    const W = (x: number): number => 2 * phi((x - 0.3) / 0.08) + phi((x - 0.75) / 0.05);
    const w0 = W(0);
    const span = W(1) - w0;
    expectMatchesCdf(warped(mixture, params), (x) => (W(x) - w0) / span);
  });

  it('mixture with one component agrees with gaussian (independent inversions)', () => {
    // probit path (gaussian) vs bisection-of-Φ path (mixture): two implementations of
    // the same inverse must meet to well under a float32 step
    for (let k = 0; k < 200; k++) {
      const q = (k + 0.5) / 200;
      const a = gaussian.warp(q, { center: 0.5, sigma: 0.1 });
      const b = mixture.warp(q, { m0: 0.5, s0: 0.1 });
      expect(Math.abs(a - b)).toBeLessThan(1e-4);
    }
  });
});

/* ---------------------------------------------------------------- §3.3 bunching */

describe('ring + gaussian bunching (DESIGN §3.3)', () => {
  function buckets(ws: Float64Array, nb: number): number[] {
    const counts = new Array<number>(nb).fill(0);
    for (let k = 0; k < ws.length; k++) {
      const w = ws[k]!;
      expect(w).toBeGreaterThanOrEqual(0);
      expect(w).toBeLessThan(1);
      counts[Math.floor(w * nb)]!++;
    }
    return counts;
  }

  it('max bucket at the center, min at the antipode', () => {
    const ws = warped(gaussian, { center: 0.25, sigma: 0.12 });
    const counts = buckets(ws, 20);
    const centerBucket = 5; // 0.25 * 20
    const antipodeBucket = 15; // 0.75 * 20
    for (let b = 0; b < 20; b++) {
      expect(counts[centerBucket]!).toBeGreaterThanOrEqual(counts[b]!);
      expect(counts[antipodeBucket]!).toBeLessThanOrEqual(counts[b]!);
    }
    expect(counts[centerBucket]!).toBeGreaterThan(counts[antipodeBucket]! + N / 20);
  });

  it('center 0 wraps: mass on BOTH ends of [0,1), thinnest at 0.5', () => {
    const ws = warped(gaussian, { center: 0, sigma: 0.1 });
    const counts = buckets(ws, 20);
    expect(counts[0]!).toBeGreaterThan(N / 10); // the two half-bumps, circularly adjacent
    expect(counts[19]!).toBeGreaterThan(N / 10);
    for (let b = 0; b < 20; b++) expect(counts[10]!).toBeLessThanOrEqual(counts[b]!);
  });
});

/* ---------------------------------------------------------------- warp contract */

describe('the warp contract: monotone, inside [0,1), f32-survivable', () => {
  const CASES: Array<[string, Density, Params]> = [
    ['gaussian (tails inside)', gaussian, { center: 0.5, sigma: 0.08 }],
    ['exponential', exponential, { rate: 3 }],
    ['powerLaw', powerLaw, { alpha: 0.5 }],
    ['mixture', mixture, { m0: 0.3, s0: 0.1, w0: 1, m1: 0.8, s1: 0.06, w1: 0.5 }],
  ];

  for (const [name, d, params] of CASES) {
    it(`${name} is monotone non-decreasing over the quantile grid`, () => {
      let prev = -Infinity;
      for (let k = 0; k < 2001; k++) {
        const v = d.warp((k + 0.5) / 2001, params);
        expect(v).toBeGreaterThanOrEqual(prev);
        prev = v;
      }
    });
  }

  it('outputs stay below 1 even after float32 rounding, for hostile params too', () => {
    const HOSTILE: Array<[Density, Params]> = [
      [gaussian, { center: 0.999, sigma: 5 }], // wrap engaged hard
      [exponential, { rate: 60 }],
      [exponential, { rate: -60 }],
      [powerLaw, { alpha: 1000 }], // q^(1/1000) → hugs 1
      [mixture, { m0: 1.5, s0: 0.1 }], // mean outside the axis
    ];
    for (const [d, params] of HOSTILE) {
      for (const q of [1e-9, 0.001, 0.5, 0.999, (N - 0.5) / N, 1 - 2 ** -53]) {
        const w = d.warp(q, params);
        expect(w).toBeGreaterThanOrEqual(0);
        expect(Math.fround(w)).toBeLessThan(1); // survives the engine's f32 u buffer
      }
    }
  });

  it('degenerate-to-uniform limits answer the identity', () => {
    for (const q of [0.1, 0.25, 0.5, 0.9]) {
      expect(exponential.warp(q, { rate: 1e-12 })).toBe(q);
      expect(powerLaw.warp(q, { alpha: 1 })).toBe(q);
    }
  });

  it('spec errors throw descriptively', () => {
    expect(() => gaussian.warp(0.5, { sigma: -1 })).toThrowError(/sigma must be >= 0/);
    expect(() => exponential.warp(0.5, { rate: Infinity })).toThrowError(/rate must be finite/);
    expect(() => powerLaw.warp(0.5, { alpha: 0 })).toThrowError(/alpha must be > 0/);
    expect(() => powerLaw.warp(0.5, { alpha: -2 })).toThrowError(/alpha must be > 0/);
    expect(() => mixture.warp(0.5, {})).toThrowError(/no components/);
    expect(() => mixture.warp(0.5, { m0: 0.5, s0: 0 })).toThrowError(/s0 must be > 0/);
    expect(() => mixture.warp(0.5, { m0: 0.5, w0: -1 })).toThrowError(/w0 must be > 0/);
  });
});

/* ---------------------------------------------------------------- through the engine */

describe('through placeGroup', () => {
  /** dim-1 passthrough: x = u0, exposing the warped axis-0 parameter. */
  const lineDen: Shape = {
    key: 'test:lineDen',
    dim: 1,
    place(u, _radius, count, out, outOffset): void {
      for (let k = 0; k < count; k++) {
        const o = outOffset + k * 3;
        out[o] = u[k]!;
        out[o + 1] = 0;
        out[o + 2] = 0;
      }
    },
  };
  registerShape(lineDen);

  function inputOf(count: number): PlaceInput {
    const members = new Uint32Array(count);
    for (let k = 0; k < count; k++) members[k] = k;
    return {
      members,
      count,
      graph: null,
      salt: 7,
      radius: 10,
      out: new Float32Array(count * 3),
      outOffset: 0,
    };
  }

  it('density key resolves and warps the byIndex quantiles', () => {
    const inp = inputOf(32);
    const params = { center: 0.25, sigma: 0.1 };
    placeGroup({ shape: 'test:lineDen', density: 'gaussian', params }, inp);
    for (let k = 0; k < 32; k++) {
      expect(inp.out[k * 3]).toBe(Math.fround(gaussian.warp((k + 0.5) / 32, params)));
    }
  });
});

/* ---------------------------------------------------------------- determinism goldens */

describe('determinism goldens (DESIGN §10.9)', () => {
  // One Float32Array of all four densities' warps over 64 quantiles each, FNV-pinned.
  // Recorded from the first run; a change means a density's byte output changed —
  // deliberate only alongside a documented contract change.
  const GOLDEN_FNV = 4156393982;

  function goldenRun(): Float32Array {
    const specs: Array<[Density, Params]> = [
      [gaussian, { center: 0.3, sigma: 0.12 }],
      [exponential, { rate: 4 }],
      [powerLaw, { alpha: 0.7 }],
      [mixture, { m0: 0.25, s0: 0.07, w0: 1, m1: 0.7, s1: 0.1, w1: 2 }],
    ];
    const buf = new Float32Array(specs.length * 64);
    let i = 0;
    for (const [d, params] of specs) {
      for (let k = 0; k < 64; k++) buf[i++] = d.warp((k + 0.5) / 64, params);
    }
    return buf;
  }

  it('pure: two runs are byte-identical', () => {
    expect(Array.from(goldenRun())).toEqual(Array.from(goldenRun()));
  });

  it('GOLDEN: fixed params + quantiles pin a recorded FNV-1a digest', () => {
    const buf = goldenRun();
    for (let i = 0; i < buf.length; i++) {
      expect(buf[i]!).toBeGreaterThanOrEqual(0);
      expect(buf[i]!).toBeLessThan(1);
    }
    expect(fnv1a(buf)).toBe(GOLDEN_FNV);
  });
});
