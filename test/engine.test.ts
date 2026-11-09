import { describe, expect, it } from 'vitest';
import { placeGroup } from '../src/place/engine';
import {
  getShape,
  registerDensity,
  registerOrdering,
  registerScatter,
  registerShape,
} from '../src/place/registry';
import { probitApprox } from '../src/place/stat';
import { fnv1a } from '../src/core/fnv';
import { hash } from '../src/core/hash';
import type { Density, Ordering, PlaceInput, Scatter, Shape } from '../src/place/types';

/** A PlaceInput over the given member ids, with a fresh zeroed output buffer. */
function inputOf(members: number[], salt = 7, radius = 10): PlaceInput {
  return {
    members: Uint32Array.from(members),
    count: members.length,
    graph: null,
    salt,
    radius,
    out: new Float32Array(members.length * 3),
    outOffset: 0,
  };
}

/* ---------------------------------------------------------------- instrumented knobs
 * Each logs its calls into `log` so the composition ORDER itself can be pinned, and uses
 * dyadic values (exact in f32) so the value checks below are exact equality, not "close". */
const log: string[] = [];

const logOrdering: Ordering = {
  key: 'test:logOrdering',
  assign(inp) {
    log.push('assign');
    const q = new Float32Array(inp.count);
    for (let k = 0; k < inp.count; k++) q[k] = (k + 0.5) / inp.count;
    return q;
  },
};

const squareDensity: Density = {
  key: 'test:square',
  warp(q) {
    log.push('warp');
    return q * q;
  },
};

/** dim-1 passthrough: x = u0 * radius, y = z = 0. */
const line1: Shape = {
  key: 'test:line1',
  dim: 1,
  place(u, radius, count, out, outOffset) {
    log.push('place');
    for (let k = 0; k < count; k++) {
      const o = outOffset + k * 3;
      out[o] = u[k]! * radius;
      out[o + 1] = 0;
      out[o + 2] = 0;
    }
  },
};

/** dim-3 passthrough: (x, y, z) = (u0, u1, u2), radius ignored. */
const pass3: Shape = {
  key: 'test:pass3',
  dim: 3,
  place(u, _radius, count, out, outOffset) {
    for (let k = 0; k < count; k++) {
      const b = k * 3;
      const o = outOffset + k * 3;
      out[o] = u[b]!;
      out[o + 1] = u[b + 1]!;
      out[o + 2] = u[b + 2]!;
    }
  },
};

/** Constant warp — a density only a test would love; makes "which axis got warped" legible. */
const threeQuarters: Density = {
  key: 'test:threeQuarters',
  warp() {
    return 0.75;
  },
};

/** ADDS 1000 to x — proves scatter runs last and composes onto the shape's output. */
const plus1000: Scatter = {
  key: 'test:plus1000',
  offset(_k, _member, _salt, _radius, out, o) {
    log.push('offset');
    out[o] = out[o]! + 1000;
  },
};

/** An ordering that must never run — pins "zero members returns before the pipeline". */
const bomb: Ordering = {
  key: 'test:bomb',
  assign() {
    throw new Error('ordering ran on an empty group');
  },
};

registerOrdering(logOrdering);
registerDensity(squareDensity);
registerShape(line1);
registerShape(pass3);
registerDensity(threeQuarters);
registerScatter(plus1000);
registerOrdering(bomb);

describe('placeGroup composition', () => {
  it('runs ordering -> density -> shape -> scatter, in that order', () => {
    log.length = 0;
    const inp = inputOf([0, 1, 2, 3]);
    placeGroup(
      {
        shape: 'test:line1',
        ordering: 'test:logOrdering',
        density: 'test:square',
        scatter: 'test:plus1000',
      },
      inp,
    );
    expect(log).toEqual([
      'assign',
      'warp',
      'warp',
      'warp',
      'warp',
      'place',
      'offset',
      'offset',
      'offset',
      'offset',
    ]);
  });

  it('composes the VALUES as pos = shape(density(ordering)) + scatter', () => {
    const inp = inputOf([0, 1, 2, 3], 7, 10);
    placeGroup(
      {
        shape: 'test:line1',
        ordering: 'test:logOrdering',
        density: 'test:square',
        scatter: 'test:plus1000',
      },
      inp,
    );
    // quantiles 1/8, 3/8, 5/8, 7/8 are dyadic (exact in f32); squared and ×10, +1000 —
    // every step exact, so exact equality is the honest assertion.
    const expected = [0.125, 0.375, 0.625, 0.875].map((q) => Math.fround(1000 + q * q * 10));
    for (let k = 0; k < 4; k++) {
      expect(inp.out[k * 3]).toBe(expected[k]);
      expect(inp.out[k * 3 + 1]).toBe(0); // scatter only touched x
      expect(inp.out[k * 3 + 2]).toBe(0);
    }
  });

  it('writes at outOffset and leaves the rest of the buffer alone', () => {
    const out = new Float32Array(5 * 3).fill(123);
    const inp: PlaceInput = {
      members: Uint32Array.from([4, 5, 6]),
      count: 3,
      graph: null,
      salt: 7,
      radius: 10,
      out,
      outOffset: 3, // skip member slot 0; place 3 members into slots 1..3
    };
    placeGroup({ shape: 'test:line1', ordering: 'test:logOrdering' }, inp);
    expect(out[0]).toBe(123);
    expect(out[1]).toBe(123);
    expect(out[2]).toBe(123);
    // two roundings, matching the pipeline: the quantile lands in the f32 u buffer first,
    // then the shape's f64 product lands in the f32 out buffer
    expect(out[3]).toBe(Math.fround(Math.fround(0.5 / 3) * 10));
    expect(out[12]).toBe(123);
    expect(out[13]).toBe(123);
    expect(out[14]).toBe(123);
  });
});

describe('densityAxis routing', () => {
  const members = [5, 9, 13];
  const salt = 11;

  /** What the engine should produce on a passthrough dim-3 shape for a given densityAxis. */
  function expected(axis: 0 | 1 | 2): number[][] {
    const n = members.length;
    return members.map((m, k) => {
      const q = Math.fround((k + 0.5) / n); // byIndex over ascending members
      const h1 = hash(m, salt + 101);
      const h2 = hash(m, salt + 102);
      return [
        Math.fround(axis === 0 ? 0.75 : q),
        Math.fround(axis === 1 ? 0.75 : h1),
        Math.fround(axis === 2 ? 0.75 : h2),
      ];
    });
  }

  for (const axis of [0, 1, 2] as const) {
    it(`densityAxis ${axis} warps exactly axis ${axis}`, () => {
      const inp = inputOf(members, salt);
      placeGroup(
        { shape: 'test:pass3', density: 'test:threeQuarters', densityAxis: axis },
        inp,
      );
      const want = expected(axis);
      for (let k = 0; k < 3; k++) {
        expect(inp.out[k * 3]).toBe(want[k]![0]);
        expect(inp.out[k * 3 + 1]).toBe(want[k]![1]);
        expect(inp.out[k * 3 + 2]).toBe(want[k]![2]);
      }
    });
  }

  it('defaults to axis 0 (the ordering axis)', () => {
    const a = inputOf(members, salt);
    const b = inputOf(members, salt);
    placeGroup({ shape: 'test:pass3', density: 'test:threeQuarters' }, a);
    placeGroup({ shape: 'test:pass3', density: 'test:threeQuarters', densityAxis: 0 }, b);
    expect(Array.from(a.out)).toEqual(Array.from(b.out));
  });

  it('throws on a densityAxis the shape does not have', () => {
    const inp = inputOf(members, salt);
    expect(() =>
      placeGroup({ shape: 'test:line1', densityAxis: 1 }, inp),
    ).toThrowError(/densityAxis 1 out of range .* "test:line1" \(dim 1\)/);
  });
});

describe('the [0,1) cap on the u buffer', () => {
  it('a hash within 2^-25 of 1 reaches the shape as 1-2^-24, never as 1.0f', () => {
    // hash(19847011, 101) = 0.99999998766… and hash(29435555, 102) = 0.99999999767…
    // both round to exactly 1.0 in a Float32Array slot — the engine must cap them, or
    // any shape trusting the documented [0, 1) domain (floor(u*K) table indexing…) reads
    // out of bounds. salt 0 makes the axis salts exactly 101/102.
    expect(Math.fround(hash(19847011, 0 + 101))).toBe(1); // the hazard is real
    expect(Math.fround(hash(29435555, 0 + 102))).toBe(1);
    const inp = inputOf([19847011, 29435555], 0, 1);
    placeGroup({ shape: 'test:pass3' }, inp); // passthrough: out = u verbatim
    const UNIT_MAX = Math.fround(1 - 2 ** -24);
    for (let i = 0; i < inp.out.length; i++) {
      expect(inp.out[i]!).toBeLessThan(1);
      expect(inp.out[i]!).toBeLessThanOrEqual(UNIT_MAX);
    }
    // the capped coordinates are exactly UNIT_MAX, not some other redistribution
    expect(inp.out[1]).toBe(UNIT_MAX); // member 19847011, axis 1
    expect(inp.out[5]).toBe(UNIT_MAX); // member 29435555, axis 2
  });
});

/* ------------------------------------------------------------------ folded axis 0 */

/** dim-1 passthrough DECLARING a 2-segment fold (Shape.folds): x = u0, so the warped
 *  parameter the engine built is directly readable. */
const folded2: Shape = {
  key: 'test:folded2',
  dim: 1,
  folds() {
    return 2;
  },
  place(u, _radius, count, out, outOffset) {
    for (let k = 0; k < count; k++) {
      const o = outOffset + k * 3;
      out[o] = u[k]!;
      out[o + 1] = 0;
      out[o + 2] = 0;
    }
  },
};
registerShape(folded2);

describe('Shape.folds: an axis-0 density warps WITHIN each segment', () => {
  it('seg is preserved and the fraction is warped — all dyadic, exact equality', () => {
    const inp = inputOf([0, 1, 2, 3], 7, 1);
    placeGroup(
      { shape: 'test:folded2', ordering: 'test:logOrdering', density: 'test:square' },
      inp,
    );
    // quantiles 1/8, 3/8, 5/8, 7/8 → segs 0,0,1,1 with fracs 1/4, 3/4, 1/4, 3/4;
    // square-warped fracs 1/16, 9/16; u = (seg + warp(frac))/2 — every step dyadic
    const want = [1 / 32, 9 / 32, 17 / 32, 25 / 32];
    for (let k = 0; k < 4; k++) expect(inp.out[k * 3]).toBe(Math.fround(want[k]!));
  });

  it('the uniform density is byte-exact the identity on a folded shape too', () => {
    const a = inputOf([0, 1, 2], 7, 1);
    const b = inputOf([0, 1, 2], 7, 1);
    placeGroup({ shape: 'test:folded2', ordering: 'test:logOrdering' }, a);
    placeGroup({ shape: 'test:line1', ordering: 'test:logOrdering' }, b);
    // same passthrough geometry at radius 1… except line1 multiplies by radius — radius
    // is 1, so the quantiles must come through bit-identical on both paths
    expect(Array.from(a.out)).toEqual(Array.from(b.out));
  });

  it('a non-zero densityAxis ignores the fold (folds speaks only for axis 0)', () => {
    // folded2 is dim 1, so use pass3 extended by a fold declaration
    const foldedPass3: Shape = {
      key: 'test:foldedPass3',
      dim: 3,
      folds() {
        return 4;
      },
      place: pass3.place,
    };
    registerShape(foldedPass3);
    const a = inputOf([5, 9, 13], 11);
    placeGroup({ shape: 'test:foldedPass3', density: 'test:threeQuarters', densityAxis: 1 }, a);
    for (let k = 0; k < 3; k++) expect(a.out[k * 3 + 1]).toBe(0.75);
  });
});

describe('unknown keys', () => {
  it('names the kind, the bad key, and the known keys', () => {
    const inp = inputOf([1, 2]);
    expect(() => placeGroup({ shape: 'nope' }, inp)).toThrowError(/unknown shape "nope"/);
    expect(() => placeGroup({ shape: 'nope' }, inp)).toThrowError(/cloud/);
    expect(() => placeGroup({ shape: 'cloud', ordering: 'nope' }, inp)).toThrowError(
      /unknown ordering "nope"/,
    );
    expect(() => placeGroup({ shape: 'cloud', ordering: 'nope' }, inp)).toThrowError(/byIndex/);
    expect(() => placeGroup({ shape: 'cloud', density: 'nope' }, inp)).toThrowError(
      /unknown density "nope"/,
    );
    expect(() => placeGroup({ shape: 'cloud', density: 'nope' }, inp)).toThrowError(/uniform/);
    expect(() => placeGroup({ shape: 'cloud', scatter: 'nope' }, inp)).toThrowError(
      /unknown scatter "nope"/,
    );
    expect(() => placeGroup({ shape: 'cloud', scatter: 'nope' }, inp)).toThrowError(/none/);
  });

  it('getShape resolves what registerShape stored', () => {
    expect(getShape('cloud').dim).toBe(3);
    expect(getShape('test:pass3')).toBe(pass3);
  });
});

describe('empty and tiny groups', () => {
  it('zero members: validates the spec but runs no knob and writes nothing', () => {
    const out = new Float32Array(6).fill(123);
    const inp: PlaceInput = {
      members: new Uint32Array(0),
      count: 0,
      graph: null,
      salt: 1,
      radius: 5,
      out,
      outOffset: 0,
    };
    // the bomb ordering would throw if the pipeline ran
    placeGroup({ shape: 'cloud', ordering: 'test:bomb' }, inp);
    expect(Array.from(out)).toEqual([123, 123, 123, 123, 123, 123]);
    // …but a spec typo still surfaces, members or not
    expect(() => placeGroup({ shape: 'nope' }, inp)).toThrowError(/unknown shape/);
    expect(() => placeGroup({ shape: 'test:line1', densityAxis: 2 }, inp)).toThrowError(
      /densityAxis/,
    );
  });

  it('one member: finite, deterministic, and centred on the ordering median', () => {
    const a = inputOf([77], 3, 9);
    const b = inputOf([77], 3, 9);
    placeGroup({ shape: 'cloud' }, a);
    placeGroup({ shape: 'cloud' }, b);
    // quantile (0+0.5)/1 = 0.5 -> probit 0 -> x exactly 0
    expect(a.out[0]).toBe(0);
    expect(Number.isFinite(a.out[1])).toBe(true);
    expect(Number.isFinite(a.out[2])).toBe(true);
    expect(Array.from(a.out)).toEqual(Array.from(b.out));
  });
});

describe('determinism', () => {
  // Recorded from the first run and pinned (DESIGN.md §10.9). A change here means the
  // engine's byte output changed: deliberate only alongside a documented contract change.
  const GOLDEN_FNV = 239392534;

  function goldenRun(): Float32Array {
    const members: number[] = [];
    for (let i = 0; i < 64; i++) members.push(i * 3 + 1);
    const inp = inputOf(members, 42, 100);
    placeGroup({ shape: 'cloud' }, inp);
    return inp.out;
  }

  it('same input -> byte-identical output, across unrelated interleaved work', () => {
    const a = goldenRun();
    // a different-size group in between must not leak through the reused u scratch
    const other = inputOf([2, 4], 9, 3);
    placeGroup({ shape: 'cloud' }, other);
    const b = goldenRun();
    expect(fnv1a(b)).toBe(fnv1a(a));
    expect(Array.from(b)).toEqual(Array.from(a));
  });

  it('GOLDEN: fixed spec + fixed members pin a recorded FNV-1a digest', () => {
    const out = goldenRun();
    for (let i = 0; i < out.length; i++) expect(Number.isFinite(out[i])).toBe(true);
    expect(fnv1a(out)).toBe(GOLDEN_FNV);
  });

  it('different salts decorrelate groups', () => {
    const a = inputOf([0, 1, 2, 3, 4, 5], 1, 10);
    const b = inputOf([0, 1, 2, 3, 4, 5], 2, 10);
    placeGroup({ shape: 'cloud' }, a);
    placeGroup({ shape: 'cloud' }, b);
    expect(fnv1a(a.out)).not.toBe(fnv1a(b.out));
  });
});

describe('built-ins', () => {
  it('byIndex ranks by member id, not by position in the members buffer', () => {
    const inp = inputOf([30, 10, 20], 7, 1);
    placeGroup({ shape: 'test:line1' }, inp); // defaults: byIndex, uniform, none
    expect(inp.out[0]).toBeCloseTo(2.5 / 3, 6); // id 30 -> rank 2
    expect(inp.out[3]).toBeCloseTo(0.5 / 3, 6); // id 10 -> rank 0
    expect(inp.out[6]).toBeCloseTo(1.5 / 3, 6); // id 20 -> rank 1
  });

  it('byIndex emits a permutation of the equal-spaced quantiles', () => {
    const inp = inputOf([9, 3, 7, 1, 5], 7, 1);
    placeGroup({ shape: 'test:line1' }, inp);
    const got = [0, 1, 2, 3, 4].map((k) => inp.out[k * 3]!).sort((x, y) => x - y);
    const want = [0, 1, 2, 3, 4].map((r) => Math.fround((r + 0.5) / 5));
    expect(got).toEqual(want);
  });

  it('cloud is a gaussian ball: finite everywhere, mean ~0, std ~ radius/3 per axis', () => {
    const members: number[] = [];
    for (let i = 0; i < 4000; i++) members.push(i);
    const inp = inputOf(members, 5, 30); // sigma = 10
    placeGroup({ shape: 'cloud' }, inp);
    for (let a = 0; a < 3; a++) {
      let s = 0;
      let ss = 0;
      for (let k = 0; k < 4000; k++) {
        const v = inp.out[k * 3 + a]!;
        expect(Number.isFinite(v)).toBe(true);
        s += v;
        ss += v * v;
      }
      const mean = s / 4000;
      const std = Math.sqrt(ss / 4000 - mean * mean);
      expect(Math.abs(mean)).toBeLessThan(0.5);
      expect(std).toBeGreaterThan(9);
      expect(std).toBeLessThan(11);
    }
  });

  it('cloud sizes from params.sigma as a fraction of radius', () => {
    const a = inputOf([0, 1, 2, 3], 7, 30);
    const b = inputOf([0, 1, 2, 3], 7, 30);
    placeGroup({ shape: 'cloud' }, a);
    placeGroup({ shape: 'cloud', params: { sigma: 2 / 3 } }, b);
    for (let i = 0; i < a.out.length; i++) {
      expect(b.out[i]).toBe(Math.fround(a.out[i]! * 2));
    }
  });
});

describe('probitApprox', () => {
  // (p, x) with p = Phi(x): round-trips the standard normal CDF at textbook points.
  const TABLE: Array<[number, number]> = [
    [0.5, 0],
    [0.8413447460685429, 1],
    [0.9772498680518208, 2],
    [0.9986501019683699, 3],
    [0.15865525393145707, -1],
    [0.022750131948179207, -2],
    [0.0013498980316300933, -3],
    [0.75, 0.6744897501960817],
    [0.9, 1.2815515655446004],
    [0.95, 1.6448536269514722],
    [0.975, 1.959963984540054],
    [0.99, 2.3263478740408408],
    [0.999, 3.090232306167813],
  ];

  it('matches known quantiles to 1e-6', () => {
    for (const [p, x] of TABLE) {
      expect(Math.abs(probitApprox(p) - x)).toBeLessThan(1e-6);
    }
  });

  it('is antisymmetric about 0.5', () => {
    for (const p of [0.001, 0.01, 0.1, 0.3, 0.45, 0.49999]) {
      expect(Math.abs(probitApprox(1 - p) + probitApprox(p))).toBeLessThan(1e-9);
    }
  });

  it('is monotone across the branch joins', () => {
    let prev = -Infinity;
    for (let i = 1; i < 2000; i++) {
      const v = probitApprox(i / 2000);
      expect(v).toBeGreaterThan(prev);
      prev = v;
    }
  });

  it('answers the closures honestly: 0 -> -Inf, 1 -> +Inf, NaN -> NaN', () => {
    expect(probitApprox(0)).toBe(-Infinity);
    expect(probitApprox(-0.5)).toBe(-Infinity);
    expect(probitApprox(1)).toBe(Infinity);
    expect(probitApprox(1.5)).toBe(Infinity);
    expect(Number.isNaN(probitApprox(NaN))).toBe(true);
  });

  it('is pure: same input, same output', () => {
    for (const [p] of TABLE) expect(probitApprox(p)).toBe(probitApprox(p));
  });
});
