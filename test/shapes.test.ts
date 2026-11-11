/* Stage-5 shape library tests (DESIGN.md §3.3, §10.9).
 *
 * Three families of pin per shape:
 *   - the radius bound, with each shape's documented tolerance (|p| ≤ radius for
 *     everything except `box`, whose documented bound is radius·√3);
 *   - shape-specific geometric invariants (ring: constant |xz| and y = 0; shell:
 *     constant |p|; helix: monotone y; spiralArm: two angular clusters; …);
 *   - determinism: a fixed member set + fixed spec → byte-identical output, pinned by
 *     FNV-1a goldens. If a golden moves, geometry moved — that is the alarm, not noise.
 *
 * Area/volume-uniformity checks are chi-square-ish bucket balances with LOOSE bounds on
 * deterministic inputs: the hashed parameter axes are fixed by (member, salt), so these
 * can never flake — they pin that the cos-weighted polar (shell/ball), sqrt radial (disc)
 * and cbrt radial (ball) corrections are present, not statistics.
 */

import { describe, expect, it } from 'vitest';
import { placeGroup } from '../src/place/engine';
import '../src/place/shapes'; // registers the §3.3 shape set as an import side effect
import '../src/place/densities'; // the fold test composes spiralArm with 'exponential'
import { dendriteTiers } from '../src/place/shapes';
import { fnv1a } from '../src/core/fnv';
import type { PlacementSpec } from '../src/place/types';

const R = 10;
const SALT = 7;

/** Place members 0..n-1 (ascending, so byIndex quantiles land in member order). */
function place(spec: PlacementSpec, n: number, salt = SALT, radius = R): Float32Array {
  const members = new Uint32Array(n);
  for (let i = 0; i < n; i++) members[i] = i;
  const out = new Float32Array(n * 3);
  placeGroup(spec, { members, count: n, graph: null, salt, radius, out, outOffset: 0 });
  return out;
}

const mag = (p: Float32Array, k: number): number =>
  Math.hypot(p[k * 3]!, p[k * 3 + 1]!, p[k * 3 + 2]!);

/** Pearson chi-square statistic of `values` bucketed into `bins` equal slices of
 *  [lo, hi). Values outside the range count as a failure via an out-of-range bin. */
function chi2(values: number[], lo: number, hi: number, bins: number): number {
  const counts = new Array<number>(bins).fill(0);
  let outside = 0;
  for (const v of values) {
    const b = Math.floor(((v - lo) / (hi - lo)) * bins);
    if (b < 0 || b >= bins) outside++;
    else counts[b] = counts[b]! + 1;
  }
  expect(outside).toBe(0);
  const exp = values.length / bins;
  let x2 = 0;
  for (let b = 0; b < bins; b++) x2 += (counts[b]! - exp) ** 2 / exp;
  return x2;
}

/* ---------------------------------------------------------------- radius bound
 * Every shape's output stays inside its documented bound. Tolerances: 1 + 1e-5 covers
 * f32 storage rounding on shapes whose math bounds |p| by radius exactly; box is the
 * documented radius·√3 exception. */
const BOUND: Array<{ spec: PlacementSpec; tol: number }> = [
  { spec: { shape: 'ring' }, tol: 1 + 1e-5 },
  { spec: { shape: 'line' }, tol: 1 + 1e-5 },
  { spec: { shape: 'helix' }, tol: 1 + 1e-5 },
  { spec: { shape: 'spiralArm' }, tol: 1 + 1e-5 },
  { spec: { shape: 'dendrite' }, tol: 1 + 1e-5 },
  { spec: { shape: 'torusKnot' }, tol: 1 + 1e-5 },
  { spec: { shape: 'shell' }, tol: 1 + 1e-5 },
  { spec: { shape: 'disc' }, tol: 1 + 1e-5 },
  { spec: { shape: 'torus' }, tol: 1 + 1e-5 },
  { spec: { shape: 'superShape' }, tol: 1 + 1e-5 },
  { spec: { shape: 'phyllotaxis' }, tol: 1 + 1e-5 },
  { spec: { shape: 'ball' }, tol: 1 + 1e-5 },
  { spec: { shape: 'box' }, tol: Math.sqrt(3) * (1 + 1e-5) },
  { spec: { shape: 'cone' }, tol: 1 + 1e-5 },
  // the superShape magnitude clamp holds for hostile exponents too (small n1 blows the raw radius)
  { spec: { shape: 'superShape', params: { m: 7, n1: 0.2, n2: 0.4, n3: 0.4 } }, tol: 1 + 1e-5 },
];

describe('radius bound', () => {
  for (const { spec, tol } of BOUND) {
    it(`${spec.shape}: |p| <= radius x ${tol.toFixed(5)}`, () => {
      const n = 257; // prime-ish, not a multiple of any shape's internal modulus
      const p = place(spec, n);
      for (let k = 0; k < n; k++) {
        const m = mag(p, k);
        expect(Number.isFinite(m)).toBe(true);
        expect(m).toBeLessThanOrEqual(R * tol);
      }
    });
  }
});

/* ---------------------------------------------------------------- determinism goldens
 * Fixed spec + fixed corpus → byte-identical buffers (§10.9). 32 members, salt 7,
 * radius 10. Pinned FNV-1a over the little-endian bytes of the output Float32Array. */
const GOLDEN: Array<{ spec: PlacementSpec; fnv: number }> = [
  { spec: { shape: 'ring' }, fnv: 4206107077 },
  { spec: { shape: 'line' }, fnv: 84391313 },
  { spec: { shape: 'helix', params: { turns: 3, coil: 0.4 } }, fnv: 253414273 },
  { spec: { shape: 'spiralArm', params: { arms: 2, swirl: 2.5 } }, fnv: 2298895269 },
  { spec: { shape: 'dendrite' }, fnv: 1806527155 },
  { spec: { shape: 'torusKnot', params: { p: 2, q: 3, tube: 0.3 } }, fnv: 3622944349 },
  { spec: { shape: 'shell' }, fnv: 929740504 },
  { spec: { shape: 'disc' }, fnv: 236750619 },
  { spec: { shape: 'torus', params: { tube: 0.25 } }, fnv: 2576968585 },
  { spec: { shape: 'superShape', params: { m: 6, n1: 1, n2: 1, n3: 1 } }, fnv: 2728218065 },
  { spec: { shape: 'phyllotaxis' }, fnv: 1668539647 },
  { spec: { shape: 'ball' }, fnv: 3773177055 },
  { spec: { shape: 'box' }, fnv: 1949661858 },
  { spec: { shape: 'cone', params: { aperture: 0.6, hollow: 0 } }, fnv: 2025580209 },
];

describe('determinism', () => {
  for (const { spec, fnv } of GOLDEN) {
    it(`${spec.shape}: pinned FNV golden, and a second run is byte-identical`, () => {
      const a = place(spec, 32);
      const b = place(spec, 32);
      expect(fnv1a(a)).toBe(fnv);
      expect(Array.from(a)).toEqual(Array.from(b));
    });
  }
});

/* ---------------------------------------------------------------- curve invariants */

describe('ring', () => {
  it('constant |xz| = radius, y = 0 exactly', () => {
    const p = place({ shape: 'ring' }, 64);
    for (let k = 0; k < 64; k++) {
      expect(p[k * 3 + 1]).toBe(0);
      expect(Math.hypot(p[k * 3]!, p[k * 3 + 2]!)).toBeCloseTo(R, 4);
    }
  });

  it('members spread over the full circle (no quadrant left empty)', () => {
    const p = place({ shape: 'ring' }, 64);
    const quad = [0, 0, 0, 0];
    for (let k = 0; k < 64; k++) {
      const a = Math.atan2(p[k * 3 + 2]!, p[k * 3]!);
      quad[(Math.floor(((a + Math.PI) / (2 * Math.PI)) * 4) + 4) % 4]! += 1;
    }
    for (const c of quad) expect(c).toBe(16);
  });
});

describe('line', () => {
  it('y = z = 0 exactly, x spans [-radius, radius) and is monotone in member order', () => {
    const n = 32;
    const p = place({ shape: 'line' }, n);
    for (let k = 0; k < n; k++) {
      expect(p[k * 3 + 1]).toBe(0);
      expect(p[k * 3 + 2]).toBe(0);
      expect(Math.abs(p[k * 3]!)).toBeLessThanOrEqual(R);
      if (k > 0) expect(p[k * 3]!).toBeGreaterThan(p[(k - 1) * 3]!);
    }
  });
});

describe('helix', () => {
  it('y strictly monotone in member order; |xz| constant at coil x radius', () => {
    const n = 128;
    const coil = 0.4;
    const p = place({ shape: 'helix', params: { turns: 3, coil } }, n);
    for (let k = 0; k < n; k++) {
      expect(Math.hypot(p[k * 3]!, p[k * 3 + 2]!)).toBeCloseTo(R * coil, 3);
      if (k > 0) expect(p[k * 3 + 1]!).toBeGreaterThan(p[(k - 1) * 3 + 1]!);
    }
  });

  it('completes the asked-for turns: angle advances ~2pi x turns root to tip', () => {
    const n = 512;
    const turns = 3;
    const p = place({ shape: 'helix', params: { turns } }, n);
    let total = 0;
    let prev = Math.atan2(p[2]!, p[0]!);
    for (let k = 1; k < n; k++) {
      const a = Math.atan2(p[k * 3 + 2]!, p[k * 3]!);
      let d = a - prev;
      if (d > Math.PI) d -= 2 * Math.PI;
      if (d < -Math.PI) d += 2 * Math.PI;
      total += d;
      prev = a;
    }
    expect(Math.abs(total)).toBeGreaterThan(2 * Math.PI * turns * 0.95);
    expect(Math.abs(total)).toBeLessThan(2 * Math.PI * turns * 1.05);
  });
});

describe('spiralArm', () => {
  it('arms = 2: a fixed radius band holds exactly two angular clusters, pi apart', () => {
    const n = 512;
    const swirl = 2.5;
    const p = place({ shape: 'spiralArm', params: { arms: 2, swirl } }, n);
    let onArm0 = 0;
    let onArm1 = 0;
    let inBand = 0;
    for (let k = 0; k < n; k++) {
      expect(p[k * 3 + 1]).toBe(0); // flat: the shape owns no thickness, scatter does
      const r = Math.hypot(p[k * 3]!, p[k * 3 + 2]!);
      if (r < 0.55 * R || r > 0.65 * R) continue;
      inBand++;
      // de-swirl: theta = arm*pi + (r/R)*swirl, so this residue must be ~0 or ~pi
      const d = Math.atan2(p[k * 3 + 2]!, p[k * 3]!) - (r / R) * swirl;
      expect(Math.abs(Math.sin(d))).toBeLessThan(1e-3);
      if (Math.cos(d) > 0.9) onArm0++;
      else if (Math.cos(d) < -0.9) onArm1++;
    }
    expect(inBand).toBeGreaterThan(20); // the band is populated at all
    expect(onArm0).toBeGreaterThan(0); // …by BOTH arms
    expect(onArm1).toBeGreaterThan(0);
    expect(onArm0 + onArm1).toBe(inBand); // …and by nothing in between
  });

  it('radial profile is area-corrected: r^2/R^2 quantiles are uniform per arm', () => {
    const n = 512;
    const p = place({ shape: 'spiralArm', params: { arms: 2, swirl: 2.5 } }, n);
    const r2: number[] = [];
    for (let k = 0; k < n; k++) {
      r2.push((p[k * 3]! ** 2 + p[k * 3 + 2]! ** 2) / (R * R));
    }
    // r = R*sqrt(t) with t the within-arm quantile, so r^2/R^2 is uniform on [0,1)
    expect(chi2(r2, 0, 1 + 1e-9, 8)).toBeLessThan(1);
  });

  it('declares its arm fold, so a density crowds radially WITHOUT re-assigning arms', () => {
    const n = 600;
    const arms = 3;
    const swirl = 2.0;
    const p = place({
      shape: 'spiralArm',
      params: { arms, swirl, radialPower: 1 },
      density: 'exponential', // rate defaults to 3
    }, n);
    const perArm = [0, 0, 0];
    let core = 0;
    for (let k = 0; k < n; k++) {
      const r = Math.hypot(p[k * 3]!, p[k * 3 + 2]!) / R;
      if (r < 1 / 3) core++;
      const d = Math.atan2(p[k * 3 + 2]!, p[k * 3]!) - r * swirl;
      perArm[((Math.round((d / (2 * Math.PI)) * arms) % arms) + arms) % arms]!++;
    }
    // equal arms, exactly (quantiles split the fold deterministically): 200 each —
    // the undeclared-fold bug measured 66/25/9% here
    expect(perArm).toEqual([200, 200, 200]);
    // …while the exponential still owns the radial profile, full strength
    const want = (1 - Math.exp(-1)) / (1 - Math.exp(-3)); // F(1/3) at rate 3 ≈ 0.665
    expect(core / n).toBeGreaterThan(want - 0.05);
    expect(core / n).toBeLessThan(want + 0.05);
  });

  it('bar: t < bar lies on straight spokes; arms launch FROM the bar ends, continuously', () => {
    const n = 512;
    const bar = 0.4;
    const swirl = 2.5;
    const p = place({ shape: 'spiralArm', params: { arms: 2, swirl, bar } }, n);
    let onBar = 0;
    let barMax = 0;
    let armMinR = Infinity;
    for (let k = 0; k < n; k++) {
      const x = p[k * 3]!;
      const z = p[k * 3 + 2]!;
      expect(p[k * 3 + 1]).toBe(0); // still flat — thickness is scatter's job
      const r = Math.hypot(x, z);
      expect(r).toBeLessThanOrEqual(R * (1 + 1e-5)); // the bound survives the bar
      if (Math.abs(z) < 1e-6 * R) {
        onBar++;
        barMax = Math.max(barMax, Math.abs(x));
      } else {
        armMinR = Math.min(armMinR, r);
        // arms wind from the bar-end anchor: de-swirling by (r/R - bar) recovers 0 or pi
        const d = Math.atan2(z, x) - (r / R - bar) * swirl;
        expect(Math.abs(Math.sin(d))).toBeLessThan(1e-3);
      }
    }
    // uniform density: the bar holds ~bar of the members, exactly as the predecessor
    expect(onBar / n).toBeGreaterThan(bar - 0.05);
    expect(onBar / n).toBeLessThan(bar + 0.05);
    expect(barMax).toBeLessThanOrEqual(bar * R * (1 + 1e-6)); // spokes end at bar·R
    expect(armMinR).toBeGreaterThanOrEqual(bar * R * (1 - 1e-3)); // arms start there
  });

  it('bar: 0 (the default) is byte-identical to the pre-bar spiral (golden above pins it)', () => {
    const a = place({ shape: 'spiralArm', params: { arms: 2, swirl: 2.5 } }, 64);
    const b = place({ shape: 'spiralArm', params: { arms: 2, swirl: 2.5, bar: 0 } }, 64);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

describe('dendrite', () => {
  it('tiers heuristic: pinned handover points (ported from arrange.js)', () => {
    expect(dendriteTiers(1)).toBe(2); // the max(8, …) floor
    expect(dendriteTiers(100)).toBe(2);
    expect(dendriteTiers(500)).toBe(4);
    expect(dendriteTiers(1_000_000)).toBe(9); // the depth cap
  });

  it('the trunk is a real segment along +Y: trunk members at x = z = 0, y > 0', () => {
    const n = 512;
    const tiers = dendriteTiers(n); // 4 -> 31 segments
    const segs = (1 << (tiers + 1)) - 1;
    const p = place({ shape: 'dendrite' }, n);
    let trunkCount = 0;
    for (let k = 0; k < n; k += segs) {
      // byIndex quantiles make rank k land on segment k % segs; k % segs === 0 is trunk
      expect(Math.abs(p[k * 3]!)).toBeLessThan(1e-9);
      expect(Math.abs(p[k * 3 + 2]!)).toBeLessThan(1e-9);
      expect(p[k * 3 + 1]!).toBeGreaterThan(0);
      trunkCount++;
    }
    expect(trunkCount).toBeGreaterThan(10);
  });

  it('branches actually reach outward (not a blob at the root)', () => {
    const n = 512;
    const p = place({ shape: 'dendrite' }, n);
    let far = 0;
    for (let k = 0; k < n; k++) if (mag(p, k) > 0.5 * R) far++;
    expect(far).toBeGreaterThan(n / 8);
  });

  it('params.tiers overrides the heuristic (fewer segments, trunk still clean)', () => {
    const n = 128;
    const p = place({ shape: 'dendrite', params: { tiers: 2 } }, n); // 7 segments
    for (let k = 0; k < n; k += 7) {
      expect(Math.abs(p[k * 3]!)).toBeLessThan(1e-9);
      expect(Math.abs(p[k * 3 + 2]!)).toBeLessThan(1e-9);
    }
  });
});

/* ---------------------------------------------------------------- surface invariants */

describe('shell', () => {
  it('constant |p| = radius', () => {
    const n = 256;
    const p = place({ shape: 'shell' }, n);
    for (let k = 0; k < n; k++) expect(mag(p, k)).toBeCloseTo(R, 3);
  });

  it('area-uniform: equal-area y bands balance (cos-weighted polar mapping)', () => {
    const n = 4096;
    const p = place({ shape: 'shell' }, n);
    // Archimedes: surface area between two y planes is proportional to their y gap, so
    // equal-width y bins are equal-AREA bins; uniform density must balance them.
    const ys: number[] = [];
    for (let k = 0; k < n; k++) ys.push(p[k * 3 + 1]!);
    expect(chi2(ys, -R - 1e-6, R + 1e-6, 8)).toBeLessThan(30); // df 7, exp ~7; loose
  });

  it('azimuth comes from the ordering axis: sectors are exactly balanced', () => {
    const n = 4096;
    const p = place({ shape: 'shell' }, n);
    const sect = new Array<number>(8).fill(0);
    for (let k = 0; k < n; k++) {
      const a = Math.atan2(p[k * 3 + 2]!, p[k * 3]!);
      const b = (Math.floor(((a + Math.PI) / (2 * Math.PI)) * 8) + 8) % 8;
      sect[b] = sect[b]! + 1;
    }
    for (const c of sect) expect(c).toBe(n / 8); // quantiles are exact, not statistical
  });
});

describe('disc', () => {
  it('flat: y = 0 exactly; r <= radius', () => {
    const n = 256;
    const p = place({ shape: 'disc' }, n);
    for (let k = 0; k < n; k++) {
      expect(p[k * 3 + 1]).toBe(0);
      expect(Math.hypot(p[k * 3]!, p[k * 3 + 2]!)).toBeLessThanOrEqual(R * (1 + 1e-5));
    }
  });

  it('area-uniform by default: equal-area annuli (equal r^2 bins) balance', () => {
    const n = 4096;
    const p = place({ shape: 'disc' }, n);
    const r2: number[] = [];
    for (let k = 0; k < n; k++) r2.push((p[k * 3]! ** 2 + p[k * 3 + 2]! ** 2) / (R * R));
    expect(chi2(r2, 0, 1 + 1e-6, 8)).toBeLessThan(30);
  });

  it('radialPower overrides the sqrt correction (the galaxy-preset escape hatch)', () => {
    const n = 4096;
    const p = place({ shape: 'disc', params: { radialPower: 1 } }, n);
    // with radialPower 1 the (uniform) axis-1 coordinate IS the radius fraction, so r/R
    // is uniform — and r^2/R^2 is therefore NOT (center-heavy).
    const r1: number[] = [];
    for (let k = 0; k < n; k++) r1.push(Math.hypot(p[k * 3]!, p[k * 3 + 2]!) / R);
    expect(chi2(r1, 0, 1 + 1e-6, 8)).toBeLessThan(30);
    let inner = 0;
    for (const r of r1) if (r * r < 0.25) inner++;
    expect(inner).toBeGreaterThan(n * 0.4); // ~half the members inside quarter the area
  });
});

describe('torus', () => {
  it('constant distance from the spine circle = tube x radius; bounded by radius', () => {
    const n = 256;
    const tube = 0.25;
    const p = place({ shape: 'torus', params: { tube } }, n);
    const spine = R * (1 - tube);
    for (let k = 0; k < n; k++) {
      const rxz = Math.hypot(p[k * 3]!, p[k * 3 + 2]!);
      const d = Math.hypot(rxz - spine, p[k * 3 + 1]!);
      expect(d).toBeCloseTo(R * tube, 3);
    }
  });
});

/* ---------------------------------------------------------------- volume invariants */

describe('ball', () => {
  it('volume-uniform: equal-volume radial bins ((|p|/R)^3) balance', () => {
    const n = 4096;
    const p = place({ shape: 'ball' }, n);
    const v: number[] = [];
    for (let k = 0; k < n; k++) v.push((mag(p, k) / R) ** 3);
    expect(chi2(v, 0, 1 + 1e-6, 8)).toBeLessThan(30);
  });

  it('y distribution is symmetric-ish (cos-weighted polar, no pole crowding)', () => {
    const n = 4096;
    const p = place({ shape: 'ball' }, n);
    let up = 0;
    for (let k = 0; k < n; k++) if (p[k * 3 + 1]! > 0) up++;
    expect(up).toBeGreaterThan(n * 0.45);
    expect(up).toBeLessThan(n * 0.55);
  });

  it('radialPower overrides the cbrt correction (the preset escape hatch, as disc)', () => {
    const n = 4096;
    const p = place({ shape: 'ball', params: { radialPower: 1 } }, n);
    // with radialPower 1 the (uniform) axis-2 coordinate IS the radius fraction, so r/R
    // is uniform — center-heavy against volume ((r/R)^3 is NOT uniform)
    const r1: number[] = [];
    for (let k = 0; k < n; k++) r1.push(mag(p, k) / R);
    expect(chi2(r1, 0, 1 + 1e-6, 8)).toBeLessThan(30);
    let inner = 0;
    for (const r of r1) if (r < 0.5) inner++;
    expect(inner).toBeGreaterThan(n * 0.4); // ~half inside an eighth of the volume
  });

  it('aspectY squashes along Y only (the predecessor’s elliptical `ell`)', () => {
    const n = 2048;
    const ay = 0.5;
    const a = place({ shape: 'ball' }, n);
    const b = place({ shape: 'ball', params: { aspectY: ay } }, n);
    let maxAbsY = 0;
    for (let k = 0; k < n; k++) {
      // same point, y scaled by exactly aspectY; x and z byte-identical
      expect(b[k * 3]).toBe(a[k * 3]);
      expect(b[k * 3 + 2]).toBe(a[k * 3 + 2]);
      expect(b[k * 3 + 1]).toBe(Math.fround(a[k * 3 + 1]! * ay));
      maxAbsY = Math.max(maxAbsY, Math.abs(b[k * 3 + 1]!));
    }
    expect(maxAbsY).toBeLessThanOrEqual(R * ay);
    expect(maxAbsY).toBeGreaterThan(R * ay * 0.8); // squashed, not collapsed
  });

  it('aspectY 1 (the default) is byte-identical to the unsquashed ball', () => {
    const a = place({ shape: 'ball' }, 64);
    const b = place({ shape: 'ball', params: { aspectY: 1 } }, 64);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

describe('box', () => {
  it('axis-aligned, half-extent = radius per axis; corners inside radius x sqrt(3)', () => {
    const n = 512;
    const p = place({ shape: 'box' }, n);
    for (let k = 0; k < n; k++) {
      expect(Math.abs(p[k * 3]!)).toBeLessThanOrEqual(R);
      expect(Math.abs(p[k * 3 + 1]!)).toBeLessThanOrEqual(R);
      expect(Math.abs(p[k * 3 + 2]!)).toBeLessThanOrEqual(R);
    }
  });

  it('aspectY/aspectZ squash the box into a slab', () => {
    const n = 512;
    const p = place({ shape: 'box', params: { aspectY: 0.25, aspectZ: 0.5 } }, n);
    let maxY = 0;
    let maxZ = 0;
    for (let k = 0; k < n; k++) {
      maxY = Math.max(maxY, Math.abs(p[k * 3 + 1]!));
      maxZ = Math.max(maxZ, Math.abs(p[k * 3 + 2]!));
    }
    expect(maxY).toBeLessThanOrEqual(R * 0.25);
    expect(maxZ).toBeLessThanOrEqual(R * 0.5);
    expect(maxY).toBeGreaterThan(R * 0.2); // squashed, not collapsed
  });
});

/* ---------------------------------------------------------------- added curve: torusKnot */

describe('torusKnot', () => {
  it('every member lies ON the torus tube (the knot threads a torus of minor radius tube·R)', () => {
    const n = 400;
    const tube = 0.3;
    const p = place({ shape: 'torusKnot', params: { p: 2, q: 3, tube } }, n);
    const scale = R / (1 + tube);
    for (let k = 0; k < n; k++) {
      const rho = Math.hypot(p[k * 3]!, p[k * 3 + 2]!); // distance from the Y axis
      const u = rho / scale - 1;
      const v = p[k * 3 + 1]! / scale;
      expect(Math.hypot(u, v)).toBeCloseTo(tube, 3); // on the tube surface
      expect(mag(p, k)).toBeLessThanOrEqual(R * (1 + 1e-5));
    }
  });

  it('winds p times about the Y axis: the azimuth advances ~2π·p root to tip', () => {
    const n = 2000;
    const pw = 2;
    const q = 3;
    const pts = place({ shape: 'torusKnot', params: { p: pw, q, tube: 0.3 } }, n);
    let total = 0;
    let prev = Math.atan2(pts[2]!, pts[0]!);
    for (let k = 1; k < n; k++) {
      const a = Math.atan2(pts[k * 3 + 2]!, pts[k * 3]!);
      let d = a - prev;
      if (d > Math.PI) d -= 2 * Math.PI;
      if (d < -Math.PI) d += 2 * Math.PI;
      total += d;
      prev = a;
    }
    expect(Math.abs(total)).toBeGreaterThan(2 * Math.PI * pw * 0.95);
    expect(Math.abs(total)).toBeLessThan(2 * Math.PI * pw * 1.05);
  });
});

/* ---------------------------------------------------------------- added surfaces */

describe('superShape', () => {
  it('bounded by R and NOT a sphere: |p| varies across the lobes (unlike shell)', () => {
    const n = 1024;
    const p = place({ shape: 'superShape', params: { m: 6, n1: 1, n2: 1, n3: 1 } }, n);
    let lo = Infinity;
    let hi = 0;
    for (let k = 0; k < n; k++) {
      const m = mag(p, k);
      expect(m).toBeLessThanOrEqual(R * (1 + 1e-5));
      lo = Math.min(lo, m);
      hi = Math.max(hi, m);
    }
    expect(hi).toBeGreaterThan(0.9 * R); // the lobe tips reach the sphere
    expect(hi - lo).toBeGreaterThan(0.1 * R); // …and the valleys dip in — a lobed surface
  });

  it('a hostile small-n1 exponent set stays on/inside the sphere (the clamp holds)', () => {
    const n = 512;
    const p = place({ shape: 'superShape', params: { m: 5, n1: 0.2, n2: 0.3, n3: 0.3 } }, n);
    for (let k = 0; k < n; k++) {
      expect(Number.isFinite(mag(p, k))).toBe(true);
      expect(mag(p, k)).toBeLessThanOrEqual(R * (1 + 1e-5));
    }
  });
});

describe('phyllotaxis', () => {
  it('flat disc (y = 0), r ≤ R, and the golden-angle step between consecutive indices', () => {
    const n = 500;
    const GOLDEN = 2.399963229728653;
    const p = place({ shape: 'phyllotaxis' }, n);
    for (let k = 0; k < n; k++) {
      expect(p[k * 3 + 1]).toBe(0); // flat in the XZ plane
      expect(Math.hypot(p[k * 3]!, p[k * 3 + 2]!)).toBeLessThanOrEqual(R * (1 + 1e-5));
    }
    // members 0..n-1 are ascending, so index k seats at angle k·goldenAngle: consecutive
    // members differ by the golden angle (mod 2π) — the sunflower's defining step.
    for (let k = 10; k < 20; k++) {
      const a0 = Math.atan2(p[k * 3 + 2]!, p[k * 3]!);
      const a1 = Math.atan2(p[(k + 1) * 3 + 2]!, p[(k + 1) * 3]!);
      let d = a1 - a0;
      while (d <= -Math.PI) d += 2 * Math.PI;
      while (d > Math.PI) d -= 2 * Math.PI;
      let g = GOLDEN;
      while (g > Math.PI) g -= 2 * Math.PI; // golden angle folded into (−π, π]
      expect(Math.abs(d - g)).toBeLessThan(1e-3);
    }
  });

  it('area-correct: r²/R² quantiles are ~uniform (the √ radial packing)', () => {
    const n = 4096;
    const p = place({ shape: 'phyllotaxis' }, n);
    const r2: number[] = [];
    for (let k = 0; k < n; k++) r2.push((p[k * 3]! ** 2 + p[k * 3 + 2]! ** 2) / (R * R));
    expect(chi2(r2, 0, 1 + 1e-6, 8)).toBeLessThan(30);
  });

  it('jitter 0 (the default) is byte-identical to omitting it (u1 is a no-op)', () => {
    const a = place({ shape: 'phyllotaxis' }, 64);
    const b = place({ shape: 'phyllotaxis', params: { jitter: 0 } }, 64);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it('jitter > 0 perturbs the points (the rings soften)', () => {
    const a = place({ shape: 'phyllotaxis' }, 256);
    const b = place({ shape: 'phyllotaxis', params: { jitter: 0.8 } }, 256);
    let moved = 0;
    for (let k = 0; k < 256; k++) {
      if (Math.hypot(a[k * 3]! - b[k * 3]!, a[k * 3 + 2]! - b[k * 3 + 2]!) > 1e-4) moved++;
    }
    expect(moved).toBeGreaterThan(200); // most members shifted
  });
});

/* ---------------------------------------------------------------- added volume: cone */

describe('cone', () => {
  it('apex up at +Y, base below; every member inside the cone envelope and the sphere', () => {
    const n = 1000;
    const aperture = 0.6;
    const p = place({ shape: 'cone', params: { aperture } }, n);
    const t = Math.tan(aperture);
    const t2 = t * t;
    const yB = (R * (t2 - 1)) / (t2 + 1);
    const H = R - yB;
    const rb = H * t;
    let highY = -Infinity;
    for (let k = 0; k < n; k++) {
      const y = p[k * 3 + 1]!;
      const rho = Math.hypot(p[k * 3]!, p[k * 3 + 2]!);
      expect(y).toBeLessThanOrEqual(R + 1e-4); // never above the apex
      expect(y).toBeGreaterThanOrEqual(yB - 1e-4); // never below the base
      // in-plane radius bounded by the cone's radius at this height: rb·(R − y)/H
      expect(rho).toBeLessThanOrEqual((rb * (R - y)) / H + 1e-4);
      expect(mag(p, k)).toBeLessThanOrEqual(R * (1 + 1e-5));
      highY = Math.max(highY, y);
    }
    expect(highY).toBeGreaterThan(0.9 * R); // members reach up toward the apex
  });

  it('hollow 1 lays every member ON the lateral surface (ρ = rb·(R−y)/H)', () => {
    const n = 800;
    const aperture = 0.6;
    const p = place({ shape: 'cone', params: { aperture, hollow: 1 } }, n);
    const t = Math.tan(aperture);
    const t2 = t * t;
    const yB = (R * (t2 - 1)) / (t2 + 1);
    const H = R - yB;
    const rb = H * t;
    for (let k = 0; k < n; k++) {
      const y = p[k * 3 + 1]!;
      const rho = Math.hypot(p[k * 3]!, p[k * 3 + 2]!);
      expect(rho).toBeCloseTo((rb * (R - y)) / H, 3); // on the slant, not inside it
    }
  });

  it('hollow 0 (the default) is byte-identical to omitting it', () => {
    const a = place({ shape: 'cone' }, 64);
    const b = place({ shape: 'cone', params: { hollow: 0 } }, 64);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

/* ---------------------------------------------------------------- spec plumbing */

describe('spec plumbing', () => {
  it('densityAxis beyond the shape dim throws (curves have only axis 0)', () => {
    expect(() => place({ shape: 'ring', densityAxis: 1 }, 4)).toThrow(/densityAxis/);
  });

  it('all §3.3 shapes — the originals and the new set — resolve via placeGroup', () => {
    for (const key of [
      'ring',
      'line',
      'helix',
      'spiralArm',
      'dendrite',
      'torusKnot',
      'shell',
      'disc',
      'torus',
      'superShape',
      'phyllotaxis',
      'ball',
      'box',
      'cone',
      'cloud', // the engine built-in the set completes
    ]) {
      expect(() => place({ shape: key }, 8)).not.toThrow();
    }
  });
});
