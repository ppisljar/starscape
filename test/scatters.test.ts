import { describe, expect, it } from 'vitest';
import '../src/place/scatters'; // registers gaussian / discFalloff / noiseField
import { valueNoise3 } from '../src/place/scatters';
import { placeGroup } from '../src/place/engine';
import { getScatter, registerShape, scatterKeys } from '../src/place/registry';
import { fnv1a } from '../src/core/fnv';
import { hash } from '../src/core/hash';
import type { PlaceInput, Scatter, Shape } from '../src/place/types';

const SALT = 7;
const R = 10;

/** Run `scatter.offset` for members 0..n-1 over a copy of `base`, returning the result. */
function scattered(
  scatter: Scatter,
  base: Float32Array,
  params: Record<string, number>,
  salt = SALT,
  radius = R,
): Float32Array {
  const out = base.slice();
  const n = out.length / 3;
  for (let k = 0; k < n; k++) scatter.offset(k, k, salt, radius, out, 3 * k, params);
  return out;
}

/** n members all placed at the origin — pure displacement becomes directly readable. */
const zeros = (n: number): Float32Array => new Float32Array(n * 3);

describe('scatter registry', () => {
  it('registers the full scatter family next to the built-in none', () => {
    const keys = scatterKeys();
    for (const k of [
      'none',
      'gaussian',
      'discFalloff',
      'noiseField',
      'uniform',
      'turbulence',
      'swirl',
    ]) {
      expect(keys).toContain(k);
    }
  });
});

describe('gaussian', () => {
  const g = getScatter('gaussian');

  it('is deterministic: same members, salt and params → byte-identical output', () => {
    const a = scattered(g, zeros(500), { thickness: 0.2 });
    const b = scattered(g, zeros(500), { thickness: 0.2 });
    expect(fnv1a(a)).toBe(fnv1a(b));
  });

  it('keys off the MEMBER id, not the slot: a member keeps its scatter when the group changes', () => {
    const a = new Float32Array(3);
    const b = new Float32Array(3);
    g.offset(0, 42, SALT, R, a, 0, { thickness: 0.2 }); // member 42 first in its group
    g.offset(9, 42, SALT, R, b, 0, { thickness: 0.2 }); // member 42 tenth in another
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('ADDS to the placed position rather than overwriting it', () => {
    const base = new Float32Array([1, 2, 3]);
    const delta = scattered(g, zeros(1), { thickness: 0.2 });
    const out = scattered(g, base, { thickness: 0.2 });
    expect(out[0]).toBeCloseTo(1 + delta[0]!, 5);
    expect(out[1]).toBeCloseTo(2 + delta[1]!, 5);
    expect(out[2]).toBeCloseTo(3 + delta[2]!, 5);
  });

  it('has per-axis σ ≈ thickness·radius, isotropic, and sane extreme bounds', () => {
    const n = 4000;
    const thickness = 0.2;
    const sigma = thickness * R;
    const out = scattered(g, zeros(n), { thickness });
    for (let axis = 0; axis < 3; axis++) {
      let sum = 0;
      let sumSq = 0;
      let maxAbs = 0;
      for (let k = 0; k < n; k++) {
        const v = out[3 * k + axis]!;
        sum += v;
        sumSq += v * v;
        maxAbs = Math.max(maxAbs, Math.abs(v));
      }
      const mean = sum / n;
      const variance = sumSq / n - mean * mean;
      expect(Math.abs(mean)).toBeLessThan(sigma * 0.1); // centred
      expect(variance).toBeGreaterThan(sigma * sigma * 0.85); // σ within ±~8%
      expect(variance).toBeLessThan(sigma * sigma * 1.15);
      expect(maxAbs).toBeLessThan(sigma * 6); // a 6σ draw at n=4000 would be a broken hash
    }
  });

  it('thickness 0 displaces nothing', () => {
    const out = scattered(g, zeros(50), { thickness: 0 });
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(0);
  });
});

describe('discFalloff', () => {
  const d = getScatter('discFalloff');
  const H = 0.3;

  /** n members on a flat disc in the xz-plane, radial distance spread over (0, R). */
  function disc(n: number): Float32Array {
    const base = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) {
      const rho = ((k + 0.5) / n) * R;
      const th = k * 2.399963229728653;
      base[3 * k] = Math.cos(th) * rho;
      base[3 * k + 2] = Math.sin(th) * rho;
    }
    return base;
  }

  it('is deterministic', () => {
    const a = scattered(d, disc(400), { h: H });
    const b = scattered(d, disc(400), { h: H });
    expect(fnv1a(a)).toBe(fnv1a(b));
  });

  it('displaces along y only — x and z stay the shape’s', () => {
    const base = disc(200);
    const out = scattered(d, base, { h: H });
    for (let k = 0; k < 200; k++) {
      expect(out[3 * k]).toBe(base[3 * k]);
      expect(out[3 * k + 2]).toBe(base[3 * k + 2]);
    }
  });

  it('THINS with radius: outer-band |y| < inner-band |y| on average', () => {
    const n = 1200;
    const base = disc(n);
    const out = scattered(d, base, { h: H });
    let inner = 0;
    let innerN = 0;
    let outer = 0;
    let outerN = 0;
    for (let k = 0; k < n; k++) {
      const rho = Math.hypot(out[3 * k]!, out[3 * k + 2]!) / R;
      const ay = Math.abs(out[3 * k + 1]!);
      if (rho < 1 / 3) {
        inner += ay;
        innerN++;
      } else if (rho > 2 / 3) {
        outer += ay;
        outerN++;
      }
    }
    expect(innerN).toBeGreaterThan(100);
    expect(outerN).toBeGreaterThan(100);
    // thick(ρ→0) ≈ (h+0.02)R vs thick(ρ→1) → 0.02R: the bands must be far apart, not
    // merely ordered — a falloff you need a t-test to see is not a galaxy disc.
    expect(outer / outerN).toBeLessThan((inner / innerN) * 0.5);
  });

  it('never exceeds the central half-thickness, and keeps the rim floor', () => {
    const n = 1200;
    const out = scattered(d, disc(n), { h: H });
    let rimMax = 0;
    for (let k = 0; k < n; k++) {
      const rho = Math.hypot(out[3 * k]!, out[3 * k + 2]!) / R;
      const ay = Math.abs(out[3 * k + 1]!);
      expect(ay).toBeLessThanOrEqual(R * (H + 0.02));
      if (rho > 0.95) rimMax = Math.max(rimMax, ay);
    }
    expect(rimMax).toBeGreaterThan(0); // the 0.02·R floor: the rim is thin, not razor-flat
    expect(rimMax).toBeLessThanOrEqual(R * (H * (1 - 0.95 * 0.95) + 0.02) + 1e-6);
  });

  it('reads the position the SHAPE wrote (read-after-shape, through the engine)', () => {
    // A radial line shape: member k lands at x = u0·R. If discFalloff reads out[] before
    // offsetting — the contract — thickness must track |x| of the point the shape placed.
    const radialLine: Shape = {
      key: 'test:radialLine',
      dim: 1,
      place(u, radius, count, out, outOffset): void {
        for (let k = 0; k < count; k++) {
          const o = outOffset + 3 * k;
          out[o] = u[k]! * radius;
          out[o + 1] = 0;
          out[o + 2] = 0;
        }
      },
    };
    registerShape(radialLine);
    const n = 600;
    const inp: PlaceInput = {
      members: Uint32Array.from({ length: n }, (_, i) => i),
      count: n,
      graph: null,
      salt: SALT,
      radius: R,
      out: new Float32Array(n * 3),
      outOffset: 0,
    };
    placeGroup({ shape: 'test:radialLine', scatter: 'discFalloff', params: { h: H } }, inp);
    let inner = 0;
    let innerN = 0;
    let outer = 0;
    let outerN = 0;
    for (let k = 0; k < n; k++) {
      const rho = Math.abs(inp.out[3 * k]!) / R;
      const ay = Math.abs(inp.out[3 * k + 1]!);
      if (rho < 1 / 3) {
        inner += ay;
        innerN++;
      } else if (rho > 2 / 3) {
        outer += ay;
        outerN++;
      }
    }
    expect(outer / outerN).toBeLessThan((inner / innerN) * 0.5);
  });
});

describe('noiseField', () => {
  const nf = getScatter('noiseField');

  it('valueNoise3 is deterministic and stays in [0, 1)', () => {
    for (let s = 0; s < 200; s++) {
      const x = (hash(s, 1) - 0.5) * 20;
      const y = (hash(s, 2) - 0.5) * 20;
      const z = (hash(s, 3) - 0.5) * 20;
      const v = valueNoise3(x, y, z, 99);
      expect(v).toBe(valueNoise3(x, y, z, 99));
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('is CONTINUOUS: nearby samples differ less than far ones, on a deterministic set', () => {
    const samples = 300;
    const NEAR = 0.05; // a twentieth of a lattice cell
    const FAR = 7.37; // several cells away, off-axis of the lattice
    let nearSum = 0;
    let farSum = 0;
    let nearMax = 0;
    for (let s = 0; s < samples; s++) {
      const x = (hash(s, 11) - 0.5) * 16;
      const y = (hash(s, 12) - 0.5) * 16;
      const z = (hash(s, 13) - 0.5) * 16;
      const v = valueNoise3(x, y, z, 5);
      const dNear = Math.abs(valueNoise3(x + NEAR, y + NEAR, z + NEAR, 5) - v);
      const dFar = Math.abs(valueNoise3(x + FAR, y + FAR, z + FAR, 5) - v);
      nearSum += dNear;
      farSum += dFar;
      nearMax = Math.max(nearMax, dNear);
    }
    // Continuity, twice over: the average near-step is a small fraction of the average
    // far-step, and even the WORST near-step stays small in absolute terms.
    expect(nearSum / samples).toBeLessThan((farSum / samples) * 0.3);
    expect(nearMax).toBeLessThan(0.35);
  });

  it('exact lattice points equal their corner hash — the interpolation anchors are real', () => {
    // At integer coordinates the trilinear weights collapse to one corner, so moving by a
    // full cell changes the value to an INDEPENDENT hash: lattice-point values must span
    // most of [0,1) rather than huddle (a broken fold would correlate neighbours).
    let min = 1;
    let max = 0;
    for (let i = -10; i <= 10; i++) {
      const v = valueNoise3(i, -i, 2 * i, 31);
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    expect(min).toBeLessThan(0.2);
    expect(max).toBeGreaterThan(0.8);
  });

  it('is a FIELD: two members at the same position displace identically', () => {
    const params = { scale: 3, amp: 0.4 };
    const a = new Float32Array([2, -1, 4]);
    const b = new Float32Array([2, -1, 4]);
    nf.offset(0, 17, SALT, R, a, 0, params);
    nf.offset(5, 23456, SALT, R, b, 0, params); // different member, same spot
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('respects the amplitude bound: |Δ| ≤ amp·radius per axis', () => {
    const params = { scale: 4, amp: 0.3 };
    const n = 800;
    const base = new Float32Array(n * 3);
    for (let i = 0; i < base.length; i++) base[i] = (hash(i, 77) - 0.5) * 2 * R;
    const out = scattered(nf, base, params);
    for (let k = 0; k < n * 3; k++) {
      expect(Math.abs(out[k]! - base[k]!)).toBeLessThanOrEqual(params.amp * R + 1e-6);
    }
    // ... and actually uses a decent share of it (an inert field passes any bound).
    let maxAbs = 0;
    for (let k = 0; k < n * 3; k++) maxAbs = Math.max(maxAbs, Math.abs(out[k]! - base[k]!));
    expect(maxAbs).toBeGreaterThan(params.amp * R * 0.5);
  });

  it('is deterministic end to end', () => {
    const base = new Float32Array(300 * 3);
    for (let i = 0; i < base.length; i++) base[i] = (hash(i, 7) - 0.5) * 2 * R;
    const a = scattered(nf, base, { scale: 3, amp: 0.4 });
    const b = scattered(nf, base, { scale: 3, amp: 0.4 });
    expect(fnv1a(a)).toBe(fnv1a(b));
  });

  /* RECORDED goldens (§10.9) — constants, not two-runs-agree: every other check in this
   * suite compares a run against itself, so a silent byte-level change to the noise
   * lattice (new primes, a different fold, a smoothstep tweak) would pass the whole
   * suite as long as it stayed deterministic, bounded and continuous. These two pins are
   * what fails instead. A change here means the FIELD changed: deliberate only alongside
   * a documented scatter change. */
  const VALUE_NOISE_GOLDEN_FNV = 113932759;
  const NOISE_FIELD_GOLDEN_FNV = 2172301370;

  it('GOLDEN: valueNoise3 over a fixed sample set pins a recorded FNV-1a digest', () => {
    const buf = new Float32Array(256);
    for (let s = 0; s < 256; s++) {
      const x = (hash(s, 41) - 0.5) * 16;
      const y = (hash(s, 42) - 0.5) * 16;
      const z = (hash(s, 43) - 0.5) * 16;
      buf[s] = valueNoise3(x, y, z, 99);
    }
    expect(fnv1a(buf)).toBe(VALUE_NOISE_GOLDEN_FNV);
  });

  it('GOLDEN: noiseField displacement over a fixed base pins a recorded FNV-1a digest', () => {
    const base = new Float32Array(300 * 3);
    for (let i = 0; i < base.length; i++) base[i] = (hash(i, 7) - 0.5) * 2 * R;
    const out = scattered(nf, base, { scale: 3, amp: 0.4 });
    expect(fnv1a(out)).toBe(NOISE_FIELD_GOLDEN_FNV);
  });
});

describe('uniform (box jitter)', () => {
  const u = getScatter('uniform');

  it('is deterministic and keys off the MEMBER id, not the slot', () => {
    const a = new Float32Array(3);
    const b = new Float32Array(3);
    u.offset(0, 42, SALT, R, a, 0, { amount: 0.2 });
    u.offset(9, 42, SALT, R, b, 0, { amount: 0.2 }); // same member, different slot
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(fnv1a(scattered(u, zeros(400), { amount: 0.2 }))).toBe(
      fnv1a(scattered(u, zeros(400), { amount: 0.2 })),
    );
  });

  it('respects the per-axis box bound |Δ| ≤ amount·radius, and fills it', () => {
    const amount = 0.2;
    const out = scattered(u, zeros(3000), { amount });
    let maxAbs = 0;
    for (let i = 0; i < out.length; i++) {
      expect(Math.abs(out[i]!)).toBeLessThanOrEqual(amount * R + 1e-6);
      maxAbs = Math.max(maxAbs, Math.abs(out[i]!));
    }
    expect(maxAbs).toBeGreaterThan(amount * R * 0.9); // a uniform box actually reaches its edges
  });

  it('is isotropic: the three axes share one spread (box, not a slab)', () => {
    const out = scattered(u, zeros(3000), { amount: 0.2 });
    const v = [0, 0, 0];
    for (let k = 0; k < 3000; k++) for (let a = 0; a < 3; a++) v[a]! += out[3 * k + a]! ** 2;
    for (const s of v) expect(s / 3000).toBeGreaterThan(0); // all axes move
    expect(Math.max(...v) / Math.min(...v)).toBeLessThan(1.3); // and by roughly the same amount
  });

  it('amount 0 displaces nothing', () => {
    const out = scattered(u, zeros(50), { amount: 0 });
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(0);
  });

  const UNIFORM_GOLDEN_FNV = 29262312;
  it('GOLDEN: pins a recorded FNV-1a digest', () => {
    expect(fnv1a(scattered(u, zeros(300), { amount: 0.2 }))).toBe(UNIFORM_GOLDEN_FNV);
  });
});

describe('turbulence (fBm)', () => {
  const tb = getScatter('turbulence');

  /** n members scattered through space (turbulence samples a field at the placed point). */
  function cloud(n: number): Float32Array {
    const base = new Float32Array(n * 3);
    for (let i = 0; i < base.length; i++) base[i] = (hash(i, 55) - 0.5) * 2 * R;
    return base;
  }

  it('is a FIELD: two members at the same position displace identically', () => {
    const params = { scale: 3, amp: 0.4, octaves: 4 };
    const a = new Float32Array([2, -1, 4]);
    const b = new Float32Array([2, -1, 4]);
    tb.offset(0, 17, SALT, R, a, 0, params);
    tb.offset(5, 98765, SALT, R, b, 0, params); // different member, same spot
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('respects the per-axis bound |Δ| ≤ amp·radius (octave-amplitude normalised), and uses it', () => {
    const params = { scale: 3, amp: 0.3, octaves: 5 };
    const base = cloud(1000);
    const out = scattered(tb, base, params);
    let maxAbs = 0;
    for (let k = 0; k < out.length; k++) {
      const d = Math.abs(out[k]! - base[k]!);
      expect(d).toBeLessThanOrEqual(params.amp * R + 1e-6);
      maxAbs = Math.max(maxAbs, d);
    }
    expect(maxAbs).toBeGreaterThan(params.amp * R * 0.4);
  });

  it('adds detail over a single octave: more octaves changes the displacement', () => {
    const base = cloud(400);
    const one = scattered(tb, base, { scale: 3, amp: 0.3, octaves: 1 });
    const four = scattered(tb, base, { scale: 3, amp: 0.3, octaves: 4 });
    let moved = 0;
    for (let k = 0; k < base.length; k++) if (Math.abs(one[k]! - four[k]!) > 1e-4) moved++;
    expect(moved).toBeGreaterThan(base.length / 2);
  });

  it('octaves clamps (≥1): a 0 request behaves as a single octave, never a NaN', () => {
    const base = cloud(100);
    const a = scattered(tb, base, { scale: 3, amp: 0.3, octaves: 0 });
    const b = scattered(tb, base, { scale: 3, amp: 0.3, octaves: 1 });
    for (let k = 0; k < base.length; k++) expect(Number.isFinite(a[k]!)).toBe(true);
    expect(fnv1a(a)).toBe(fnv1a(b));
  });

  it('amp 0 displaces nothing', () => {
    const base = cloud(50);
    const out = scattered(tb, base, { amp: 0 });
    for (let k = 0; k < base.length; k++) expect(out[k]).toBe(base[k]);
  });

  const TURB_GOLDEN_FNV = 3543823139;
  it('GOLDEN: pins a recorded FNV-1a digest', () => {
    const base = cloud(300);
    expect(fnv1a(scattered(tb, base, { scale: 3, amp: 0.4, octaves: 4 }))).toBe(TURB_GOLDEN_FNV);
  });
});

describe('swirl (tangential shear)', () => {
  const sw = getScatter('swirl');

  /** n members off the Y axis (a rotation of an on-axis point does nothing). */
  function ring(n: number): Float32Array {
    const base = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) {
      const th = k * 2.399963229728653;
      const rho = (0.3 + 0.7 * ((k + 0.5) / n)) * R;
      base[3 * k] = Math.cos(th) * rho;
      base[3 * k + 1] = (hash(k, 3) - 0.5) * R; // some height, to prove y is untouched
      base[3 * k + 2] = Math.sin(th) * rho;
    }
    return base;
  }

  it('is deterministic and keys off the member id', () => {
    const a = new Float32Array([3, 1, 4]);
    const b = new Float32Array([3, 1, 4]);
    sw.offset(0, 77, SALT, R, a, 0, { strength: 0.8 });
    sw.offset(6, 77, SALT, R, b, 0, { strength: 0.8 }); // same member, different slot
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('is a pure rotation about Y: |p| and y are preserved, x/z move', () => {
    const base = ring(1000);
    const out = scattered(sw, base, { strength: 0.6 });
    let moved = 0;
    for (let k = 0; k < 1000; k++) {
      const before = Math.hypot(base[3 * k]!, base[3 * k + 1]!, base[3 * k + 2]!);
      const after = Math.hypot(out[3 * k]!, out[3 * k + 1]!, out[3 * k + 2]!);
      expect(after).toBeCloseTo(before, 3); // magnitude preserved (within f32)
      expect(out[3 * k + 1]).toBe(base[3 * k + 1]); // y exactly untouched
      if (Math.abs(out[3 * k]! - base[3 * k]!) > 1e-3) moved++;
    }
    expect(moved).toBeGreaterThan(900); // the shear is visible
  });

  it('reads the position the SHAPE wrote (an on-axis point cannot swirl)', () => {
    const onAxis = new Float32Array([0, 5, 0]);
    const out = scattered(sw, onAxis, { strength: 1.2 });
    expect(Array.from(out)).toEqual([0, 5, 0]); // ρ = 0 ⇒ no tangential displacement
  });

  it('strength 0 displaces nothing', () => {
    const base = ring(50);
    const out = scattered(sw, base, { strength: 0 });
    for (let k = 0; k < base.length; k++) expect(out[k]).toBe(base[k]);
  });

  const SWIRL_GOLDEN_FNV = 1063713914;
  it('GOLDEN: pins a recorded FNV-1a digest', () => {
    expect(fnv1a(scattered(sw, ring(300), { strength: 0.6 }))).toBe(SWIRL_GOLDEN_FNV);
  });
});
