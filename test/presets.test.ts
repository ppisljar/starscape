import { describe, expect, it } from 'vitest';
import '../src/place/engine'; // built-in knobs (cloud / byIndex / uniform / none)
import '../src/place/scatters'; // gaussian / discFalloff / noiseField
import { getPreset, preset, presetKeys, registerPreset } from '../src/place/presets';
import { placeGroup } from '../src/place/engine';
import { densityKeys, orderingKeys, scatterKeys, shapeKeys } from '../src/place/registry';
import { fnv1a } from '../src/core/fnv';
import type { PlaceInput, PlacementSpec } from '../src/place/types';

/* The shape/ordering/density libraries are sibling stage-5 modules; load whichever have
 * landed (file-URL dynamic import so an absent module is a skip, not a crash — and not a
 * compile-time dependency on files this test does not own). The integration test at the
 * bottom is skipIf-gated on the registry actually holding every key the presets name, so
 * it starts passing the moment the sibling modules exist. */
for (const mod of ['shapes', 'orderings', 'densities']) {
  try {
    await import(/* @vite-ignore */ new URL(`../src/place/${mod}.ts`, import.meta.url).href);
  } catch {
    /* not landed yet — the integration test below skips itself */
  }
}

const BUILT_INS = ['spiralGalaxy', 'barredGalaxy', 'elliptical', 'globularCluster', 'irregular'];

/** Do the registries hold every knob a spec names (defaults included)? */
function knobsRegistered(spec: PlacementSpec): boolean {
  return (
    shapeKeys().includes(spec.shape) &&
    orderingKeys().includes(spec.ordering ?? 'byIndex') &&
    densityKeys().includes(spec.density ?? 'uniform') &&
    scatterKeys().includes(spec.scatter ?? 'none')
  );
}

const allKnobsReady = BUILT_INS.every((name) => knobsRegistered(getPreset(name)));

describe('preset registry', () => {
  it('ships the five built-in looks', () => {
    const keys = presetKeys();
    for (const name of BUILT_INS) expect(keys).toContain(name);
  });

  it('returns FROZEN specs — params included', () => {
    for (const name of BUILT_INS) {
      const spec = getPreset(name);
      expect(Object.isFrozen(spec)).toBe(true);
      if (spec.params !== undefined) expect(Object.isFrozen(spec.params)).toBe(true);
    }
  });

  it('a frozen spec actually rejects mutation (ESM is strict mode: throws, not ignores)', () => {
    const spec = getPreset('spiralGalaxy');
    expect(() => {
      (spec as { shape: string }).shape = 'cloud';
    }).toThrow(TypeError);
    expect(() => {
      (spec.params as Record<string, number>)['swirl'] = 99;
    }).toThrow(TypeError);
    expect(spec.shape).toBe('spiralArm');
    expect(spec.params?.['swirl']).toBe(2.6);
  });

  it('throws on an unknown preset, naming the known ones', () => {
    expect(() => getPreset('spiralgalaxy')).toThrow(/unknown preset "spiralgalaxy"/);
    expect(() => getPreset('spiralgalaxy')).toThrow(/spiralGalaxy/);
  });

  it('`preset` is the §3.4 spelling of getPreset', () => {
    expect(preset('elliptical')).toBe(getPreset('elliptical'));
  });

  it('re-registering a name overwrites, and freezes the new spec', () => {
    registerPreset('test:look', { shape: 'cloud', params: { sigma: 0.5 } });
    registerPreset('test:look', { shape: 'cloud', params: { sigma: 0.25 } });
    const spec = getPreset('test:look');
    expect(spec.params?.['sigma']).toBe(0.25);
    expect(Object.isFrozen(spec)).toBe(true);
    expect(Object.isFrozen(spec.params)).toBe(true);
  });
});

describe('the built-in looks are the right data bundles', () => {
  it('spiralGalaxy: spiral arms, exponential radial crowding, disc-falloff thickness, swirl', () => {
    const s = getPreset('spiralGalaxy');
    expect(s.shape).toBe('spiralArm');
    expect(s.density).toBe('exponential');
    expect(s.densityAxis).toBe(0);
    expect(s.scatter).toBe('discFalloff');
    expect(s.ordering).toBe('byHash');
    // the predecessor's DEFAULT_MORPH: 3 arms, swirl 2.6; its bulge 0.10 ports as `h`
    expect(s.params?.['arms']).toBe(3);
    expect(s.params?.['swirl']).toBe(2.6);
    expect(s.params?.['h']).toBe(0.1); // discFalloff half-thickness = the ported bulge
    expect(s.params?.['rate']).toBeGreaterThan(0); // exponential gets a real rate
    // the escape hatch: the warp output IS the radius fraction — without it the shape's
    // √t area correction composes into the warp and the exponential becomes a gaussian
    expect(s.params?.['radialPower']).toBe(1);
  });

  it('barredGalaxy: the spiral bundle plus a bar the arms launch from', () => {
    const s = getPreset('barredGalaxy');
    expect(s.shape).toBe('spiralArm');
    expect(s.density).toBe('exponential');
    expect(s.scatter).toBe('discFalloff');
    // bar half-length as a fraction of R — predecessor DEFAULT_MORPH 0.3, hashed 0.26–0.52
    expect(s.params?.['bar']).toBeGreaterThanOrEqual(0.26);
    expect(s.params?.['bar']).toBeLessThanOrEqual(0.52);
    expect(s.params?.['arms']).toBe(2); // a bar has two ends
    // the unbarred preset must NOT carry a bar — that is the whole difference
    expect(getPreset('spiralGalaxy').params?.['bar']).toBeUndefined();
  });

  it('elliptical: ball + powerLaw on the RADIAL axis, r = R·q^0.62, squashed', () => {
    const s = getPreset('elliptical');
    expect(s.shape).toBe('ball');
    expect(s.density).toBe('powerLaw');
    expect(s.densityAxis).toBe(2); // the ball's radial axis — 0 is the azimuth
    expect(s.scatter ?? 'none').toBe('none'); // relaxed: no structure to scatter in
    // powerLaw warps q^(1/alpha); with radialPower 1 the predecessor's r = R·t^0.62 law
    // therefore needs alpha = 1/0.62 (the law lives in the inverse CDF, not alpha itself)
    expect(s.params?.['alpha']).toBeCloseTo(1 / 0.62, 12);
    expect(s.params?.['radialPower']).toBe(1);
    // the squash (the predecessor's `ell`), within its E3–E6 hashed range
    expect(s.params?.['aspectY']).toBeGreaterThanOrEqual(0.42);
    expect(s.params?.['aspectY']).toBeLessThanOrEqual(0.78);
  });

  it('globularCluster: ball + half-gaussian radial profile centred on the core', () => {
    const s = getPreset('globularCluster');
    expect(s.shape).toBe('ball');
    // mixture with one component AT the core is the truncated half-gaussian: it
    // renormalises over [0,1] instead of wrapping (gaussian center-0 would put half the
    // mass at the rim), which is the only built-in way to crowd a radial axis gaussianly
    expect(s.density).toBe('mixture');
    expect(s.densityAxis).toBe(2); // radial, not azimuth
    expect(s.params?.['m0']).toBe(0);
    expect(s.params?.['s0']).toBeGreaterThan(0);
    expect(s.params?.['s0']).toBeLessThan(1);
    expect(s.params?.['radialPower']).toBe(1);
  });

  it('irregular: cloud + noiseField, low frequency and high amplitude = a few big clumps', () => {
    const s = getPreset('irregular');
    expect(s.shape).toBe('cloud');
    expect(s.scatter).toBe('noiseField');
    const scale = s.params?.['scale'] ?? NaN;
    const amp = s.params?.['amp'] ?? NaN;
    expect(scale).toBeGreaterThan(0);
    expect(scale).toBeLessThanOrEqual(4); // low frequency: clumps, not static
    expect(amp).toBeGreaterThanOrEqual(0.3); // strong enough to break the cloud's symmetry
  });

  it('every built-in is PURE DATA: knob keys, numeric params, nothing else', () => {
    const allowed = new Set(['shape', 'ordering', 'density', 'densityAxis', 'scatter', 'params']);
    for (const name of BUILT_INS) {
      const spec = getPreset(name) as unknown as Record<string, unknown>;
      for (const key of Object.keys(spec)) {
        expect(allowed.has(key), `${name}.${key} is not a PlacementSpec field`).toBe(true);
        expect(typeof spec[key]).not.toBe('function');
      }
      const params = getPreset(name).params ?? {};
      for (const [k, v] of Object.entries(params)) {
        expect(typeof v, `${name}.params.${k}`).toBe('number');
        expect(Number.isFinite(v), `${name}.params.${k}`).toBe(true);
      }
    }
  });
});

/* The integration tests: run every built-in preset through the real engine once the
 * sibling shape/ordering/density modules have registered their keys. Skipped — visibly,
 * not silently green — until they land; the stage gate re-runs everything. */

const N_RUN = 500;
const RUN_SALT = 1234;
const RUN_RADIUS = 100;

/** The canonical integration run: n=500, members i·3+1, salt 1234, radius 100. */
function runPreset(spec: PlacementSpec): Float32Array {
  const inp: PlaceInput = {
    members: Uint32Array.from({ length: N_RUN }, (_, i) => i * 3 + 1),
    count: N_RUN,
    graph: null,
    salt: RUN_SALT,
    radius: RUN_RADIUS,
    out: new Float32Array(N_RUN * 3),
    outOffset: 0,
  };
  placeGroup(spec, inp);
  return inp.out;
}

/* RECORDED FNV-1a constants over `runPreset` — one per built-in look (§10.9: real
 * goldens, not two-runs-agree, so a byte-level geometry drift in any single preset fails
 * here even when it stays deterministic, finite and spread out). A change means that
 * look's geometry changed: deliberate only alongside a documented preset change. */
const GOLDENS: Record<string, number> = {
  spiralGalaxy: 1762130960,
  barredGalaxy: 3573674580,
  elliptical: 2200506835,
  globularCluster: 3079042052,
  irregular: 40727252,
};

describe('presets through the engine', () => {
  it.skipIf(!allKnobsReady)(
    'every built-in look places a real group: finite, deterministic, spread out',
    () => {
      for (const name of BUILT_INS) {
        const spec = getPreset(name);
        const a = runPreset(spec);
        for (let i = 0; i < a.length; i++) {
          expect(Number.isFinite(a[i]), `${name}: coordinate ${i} not finite`).toBe(true);
        }
        // deterministic: byte-identical on a second compile (§10, DESIGN golden rule)
        expect(fnv1a(runPreset(spec)), `${name}: not deterministic`).toBe(fnv1a(a));
        // ... and pinned to the RECORDED constant, per preset
        expect(fnv1a(a), `${name}: golden moved — geometry changed`).toBe(GOLDENS[name]);
        // spread out: a look that stacks everyone on one point is a broken bundle
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (let k = 0; k < N_RUN; k++) {
          minX = Math.min(minX, a[3 * k]!);
          maxX = Math.max(maxX, a[3 * k]!);
          minY = Math.min(minY, a[3 * k + 1]!);
          maxY = Math.max(maxY, a[3 * k + 1]!);
        }
        expect(maxX - minX, `${name}: degenerate x extent`).toBeGreaterThan(1);
        expect(maxY - minY, `${name}: degenerate y extent`).toBeGreaterThan(0.1);
      }
    },
  );

  it.skipIf(!allKnobsReady)('spiralGalaxy: equal arms AND the asked-for radial crowding', () => {
    const spec = getPreset('spiralGalaxy');
    const arms = spec.params!['arms']!;
    const swirl = spec.params!['swirl']!;
    const rate = spec.params!['rate']!;
    const a = runPreset(spec);
    const perArm = new Array<number>(arms).fill(0);
    let core = 0;
    for (let k = 0; k < N_RUN; k++) {
      const x = a[3 * k]!;
      const z = a[3 * k + 2]!;
      const r = Math.hypot(x, z) / RUN_RADIUS; // discFalloff displaces y only
      if (r < 1 / 3) core++;
      // de-swirl to the arm's launch angle; the residue picks the arm
      const d = Math.atan2(z, x) - r * swirl;
      const arm = (Math.round((d / (2 * Math.PI)) * arms) % arms + arms) % arms;
      perArm[arm]!++;
    }
    // the fold is declared, so the density CANNOT re-assign arms: populations are exact
    // thirds (±1), not the 66/25/9% split the undeclared fold produced
    for (const c of perArm) {
      expect(c).toBeGreaterThanOrEqual(Math.floor(N_RUN / arms) - 1);
      expect(c).toBeLessThanOrEqual(Math.ceil(N_RUN / arms) + 1);
    }
    // radialPower 1 hands the warp through: P(r < 1/3) = truncated-exp CDF at 1/3
    const want = (1 - Math.exp(-rate / 3)) / (1 - Math.exp(-rate)); // ≈ 0.665
    expect(core / N_RUN).toBeGreaterThan(want - 0.05);
    expect(core / N_RUN).toBeLessThan(want + 0.05);
  });

  it.skipIf(!allKnobsReady)('barredGalaxy: the bar EXISTS and the arms launch from its ends', () => {
    const spec = getPreset('barredGalaxy');
    const bar = spec.params!['bar']!;
    const a = runPreset(spec);
    // bar members lie on the ±X spokes: z is sin(0)/sin(π) scaled — zero to f64 dust —
    // while arm members carry a real swirl angle; 1e-6 cleanly separates the two
    let onBar = 0;
    let barMaxX = 0;
    let armMaxR = 0;
    for (let k = 0; k < N_RUN; k++) {
      const x = a[3 * k]!;
      const z = a[3 * k + 2]!;
      if (Math.abs(z) < 1e-6 * RUN_RADIUS) {
        onBar++;
        barMaxX = Math.max(barMaxX, Math.abs(x));
      } else {
        armMaxR = Math.max(armMaxR, Math.hypot(x, z));
      }
    }
    // the exponential crowds the core, so MORE than `bar` of the members sit in the bar
    expect(onBar / N_RUN).toBeGreaterThan(0.4);
    expect(onBar / N_RUN).toBeLessThan(0.8);
    expect(barMaxX).toBeLessThanOrEqual(bar * RUN_RADIUS * (1 + 1e-6)); // bar ends at bar·R
    expect(armMaxR).toBeGreaterThan(0.6 * RUN_RADIUS); // arms continue well beyond it
    // and the bar param DOES something: dropping it must change the bytes
    const bare: PlacementSpec = { ...spec, params: { ...spec.params } };
    delete (bare.params as Record<string, number>)['bar'];
    expect(fnv1a(runPreset(bare))).not.toBe(fnv1a(a));
  });

  it.skipIf(!allKnobsReady)('elliptical: the r = R·q^0.62 concentration, squashed along Y', () => {
    const spec = getPreset('elliptical');
    const ay = spec.params!['aspectY']!;
    const a = runPreset(spec);
    let core = 0;
    let maxAbsY = 0;
    let maxRxz = 0;
    for (let k = 0; k < N_RUN; k++) {
      const x = a[3 * k]!;
      const y = a[3 * k + 1]! / ay; // un-squash to read the radial law
      const z = a[3 * k + 2]!;
      if (Math.hypot(x, y, z) < RUN_RADIUS / 2) core++;
      maxAbsY = Math.max(maxAbsY, Math.abs(a[3 * k + 1]!));
      maxRxz = Math.max(maxRxz, Math.hypot(x, z));
    }
    // P(r < R/2) = 0.5^(1/0.62) ≈ 0.327 — NOT the 0.125 of a uniform ball
    expect(core / N_RUN).toBeGreaterThan(0.25);
    expect(core / N_RUN).toBeLessThan(0.41);
    // the squash is real: Y stays inside aspectY·R while the disc plane fills out
    expect(maxAbsY).toBeLessThanOrEqual(ay * RUN_RADIUS * (1 + 1e-5));
    expect(maxRxz).toBeGreaterThan(0.8 * RUN_RADIUS);
  });

  it.skipIf(!allKnobsReady)('globularCluster: crowds the core, round, no rim pile-up', () => {
    const a = runPreset(getPreset('globularCluster'));
    let core = 0;
    let rim = 0;
    for (let k = 0; k < N_RUN; k++) {
      const r = Math.hypot(a[3 * k]!, a[3 * k + 1]!, a[3 * k + 2]!) / RUN_RADIUS;
      if (r < 0.5) core++;
      if (r > 0.9) rim++;
    }
    // half-gaussian σ 0.35: ≈85% inside R/2 (a uniform ball has 12.5%; the OLD wrapped
    // gaussian put literally half the members at the rim)
    expect(core / N_RUN).toBeGreaterThan(0.75);
    expect(rim / N_RUN).toBeLessThan(0.05);
  });
});
