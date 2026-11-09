/* Stage-6 reference frames (DESIGN.md §3.5, §4): the FrameSet over the hierarchy pivot
 * tree — the CPU reference the GPU transform chain mirrors.
 *
 * The pinned invariants are the predecessor's (vault-site-3d README §"Reference frames",
 * §"The two things worth knowing", rotation.test.mjs):
 *   - compose-vs-nest: p' = C + R₁(A + R₂(p−A) − C) equals rotating the anchor first and
 *     carrying its satellites — one quaternion per PIVOT, verified against an independent
 *     f64 Rodrigues implementation, in the literal two-level form AND the general
 *     recursive form;
 *   - riding: a point in the frame the camera rides moves ~0 camera-relative while a
 *     sibling frame sweeps (the README telemetry scenario: 0.0 vs 1,387);
 *   - seamless entry: riding applies the PER-TICK increment, never cancellation of
 *     accumulated angle, so starting to ride mid-flight produces zero instantaneous jump;
 *   - the 117-vs-83 fps lesson: "is anything turning" tests the per-tick delta, so
 *     activeThisTick() is false the moment rates are zero, however long things spun;
 *   - determinism under a fixed dt sequence (byte-identical buffers, FNV-pinned);
 *   - O(pivots): the API admits no node-length data by construction.
 */

import { describe, expect, it } from 'vitest';
import '../src/place/engine'; // built-ins: cloud / byIndex / uniform / none
import '../src/place/shapes';
import '../src/place/orderings';
import '../src/place/densities';
import '../src/place/scatters';
import { ANGLE_EPS, createFrameSet, RIDE_DELTA_SIZE } from '../src/anim/frames';
import { compileHierarchy } from '../src/place/hierarchy';
import type { LevelSpec, PivotTable } from '../src/place/hierarchy';
import { buildGraph } from '../src/core/graph';
import { generateSynth } from '../src/synth/generate';
import { fnv1a } from '../src/core/fnv';

/* ------------------------------------------------------------------ fixtures */

/** Hand-built pivot tables — the FrameSet consumes exactly the PivotTable shape the
 *  hierarchy compiler emits, so tests can state a tree in five lines. `c` is the LOCAL
 *  centre (world for roots); `world` is resolved by the same forward pass the compiler
 *  uses (parents precede children). Axes are normalised here, as the compiler does. */
interface Row {
  c: [number, number, number];
  parent: number;
  level: number;
  radius: number;
  axis?: [number, number, number];
  rate?: number;
}

function makeTable(rows: Row[]): PivotTable {
  const P = rows.length;
  const centre = new Float32Array(P * 3);
  const world = new Float32Array(P * 3);
  const parent = new Int32Array(P);
  const level = new Uint8Array(P);
  const radius = new Float32Array(P);
  const axis = new Float32Array(P * 3);
  const rate = new Float32Array(P);
  for (let p = 0; p < P; p++) {
    const r = rows[p]!;
    centre[p * 3] = r.c[0];
    centre[p * 3 + 1] = r.c[1];
    centre[p * 3 + 2] = r.c[2];
    parent[p] = r.parent;
    level[p] = r.level;
    radius[p] = r.radius;
    const a = r.axis ?? [0, 1, 0];
    const len = Math.hypot(a[0], a[1], a[2]);
    axis[p * 3] = a[0] / len;
    axis[p * 3 + 1] = a[1] / len;
    axis[p * 3 + 2] = a[2] / len;
    rate[p] = r.rate ?? 0;
    const o = p * 3;
    if (r.parent < 0) {
      world[o] = centre[o]!;
      world[o + 1] = centre[o + 1]!;
      world[o + 2] = centre[o + 2]!;
    } else {
      const po = r.parent * 3;
      world[o] = centre[o]! + world[po]!;
      world[o + 1] = centre[o + 1]! + world[po + 1]!;
      world[o + 2] = centre[o + 2]! + world[po + 2]!;
    }
  }
  return { count: P, centre, world, parent, level, radius, axis, rate };
}

/* f64 reference maths — deliberately a DIFFERENT implementation from the module's
 * quaternion kernels (Rodrigues' formula), so agreement pins the maths, not the code. */

type V3 = [number, number, number];

function norm(a: V3): V3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / l, a[1] / l, a[2] / l];
}
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const dist = (a: V3, b: V3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** v rotated by angle t about unit axis u: v·cos t + (u×v)·sin t + u·(u·v)(1−cos t). */
function rodrigues(u: V3, t: number, v: V3): V3 {
  const c = Math.cos(t);
  const s = Math.sin(t);
  const d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const cr: V3 = [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  return [
    v[0] * c + cr[0] * s + u[0] * d * (1 - c),
    v[1] * c + cr[1] * s + u[1] * d * (1 - c),
    v[2] * c + cr[2] * s + u[2] * d * (1 - c),
  ];
}

type Q = [number, number, number, number];

const qAA = (u: V3, t: number): Q => {
  const s = Math.sin(t / 2);
  return [u[0] * s, u[1] * s, u[2] * s, Math.cos(t / 2)];
};
const qMul = (a: Q, b: Q): Q => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
const qConj = (q: Q): Q => [-q[0], -q[1], -q[2], q[3]];
function qRot(q: Q, v: V3): V3 {
  const tx = 2 * (q[1] * v[2] - q[2] * v[1]);
  const ty = 2 * (q[2] * v[0] - q[0] * v[2]);
  const tz = 2 * (q[0] * v[1] - q[1] * v[0]);
  return [
    v[0] + q[3] * tx + (q[1] * tz - q[2] * ty),
    v[1] + q[3] * ty + (q[2] * tx - q[0] * tz),
    v[2] + q[3] * tz + (q[0] * ty - q[1] * tx),
  ];
}

/** The recursive NEST-form reference: rotate about the frame's own UNROTATED centre,
 *  then hand the result to the parent — the long-hand composition transformPoint's
 *  single quaternion apply must equal, chain-deep. Angles are rate × elapsed. */
function refTransform(tab: PivotTable, pivot: number, p: V3, elapsed: number): V3 {
  if (pivot < 0) return p;
  const o = pivot * 3;
  const c0: V3 = [tab.world[o]!, tab.world[o + 1]!, tab.world[o + 2]!];
  const u: V3 = [tab.axis[o]!, tab.axis[o + 1]!, tab.axis[o + 2]!];
  const spun = rodrigues(u, tab.rate[pivot]! * elapsed, sub(p, c0));
  return refTransform(tab, tab.parent[pivot]!, add(c0, spun), elapsed);
}

const expectVec = (got: ArrayLike<number>, want: V3, digits: number): void => {
  expect(got[0]).toBeCloseTo(want[0], digits);
  expect(got[1]).toBeCloseTo(want[1], digits);
  expect(got[2]).toBeCloseTo(want[2], digits);
};

/* The synth-backed hierarchy for the integration / determinism / O(pivots) tests —
 * same corpus the stage-5 goldens pinned. */
const SYNTH = buildGraph(generateSynth({ n: 2000, seed: 7 }));
const LEVELS: LevelSpec[] = [
  { groupBy: 'galaxy', place: { shape: 'shell' }, spin: { rate: 0.4 } },
  { groupBy: 'system', place: { shape: 'cloud' }, spin: { rate: 1.1, axis: [0, 0, 1] } },
  { place: { shape: 'cloud' } },
];
const compiled = (): ReturnType<typeof compileHierarchy> =>
  compileHierarchy(SYNTH, LEVELS, { spacing: 10 });

/* ------------------------------------------------------------------ compose vs nest */

describe('compose-vs-nest (the README §"Compose the rotations" identity)', () => {
  // two levels: anchor A under centre C, both spinning about non-trivial axes
  const U1 = norm([0.2, 1, -0.3]);
  const U2 = norm([1, 0.5, 2]);
  const C: V3 = [0, 0, 0];
  const A: V3 = [2, 0, 0];
  const W1 = 0.3;
  const W2 = 0.7;
  const DT = 0.5;
  const tab = makeTable([
    { c: C, parent: -1, level: 0, radius: 10, axis: U1, rate: W1 },
    { c: A, parent: 0, level: 1, radius: 3, axis: U2, rate: W2 },
  ]);
  const fs = createFrameSet(tab);
  fs.advance(DT);
  const p0: V3 = [2.5, 0.4, -0.2];

  it('transformPoint equals C + R1(A + R2(p−A) − C), the literal two-level form', () => {
    const composed = add(C, rodrigues(U1, W1 * DT, sub(add(A, rodrigues(U2, W2 * DT, sub(p0, A))), C)));
    const got = fs.transformPoint(1, p0, [0, 0, 0]);
    expectVec(got, composed, 5);
  });

  it('… and equals anchor-rotated-first carrying its satellite (the identity itself)', () => {
    // turn the anchor about C first, then place the satellite with BOTH rotations
    const carriedAnchor = add(C, rodrigues(U1, W1 * DT, sub(A, C)));
    const nested = add(carriedAnchor, rodrigues(U1, W1 * DT, rodrigues(U2, W2 * DT, sub(p0, A))));
    // the two reference forms agree to f64 (the identity holds numerically) …
    const composed = add(C, rodrigues(U1, W1 * DT, sub(add(A, rodrigues(U2, W2 * DT, sub(p0, A))), C)));
    expect(dist(nested, composed)).toBeLessThan(1e-12);
    // … and the FrameSet matches them with its one quaternion per pivot
    const got = fs.transformPoint(1, p0, [0, 0, 0]);
    expectVec(got, nested, 5);
  });

  it('quaternionOf(child) is the composed chain q1 ⊗ q2', () => {
    const want = qMul(qAA(U1, W1 * DT), qAA(U2, W2 * DT));
    const got = fs.quaternionOf(1, [0, 0, 0, 0]);
    for (let k = 0; k < 4; k++) expect(got[k]).toBeCloseTo(want[k]!, 6);
  });

  it('the root itself rotates about C with q1 alone', () => {
    const want = add(C, rodrigues(U1, W1 * DT, sub(p0, C)));
    expectVec(fs.transformPoint(0, p0, [0, 0, 0]), want, 5);
  });

  it('three deep: transformPoint equals the recursive nest-form reference', () => {
    const deep = makeTable([
      { c: [1, 2, 3], parent: -1, level: 0, radius: 40, axis: [0.2, 1, -0.3], rate: 0.3 },
      { c: [8, 0, 0], parent: 0, level: 1, radius: 12, axis: [1, 0.5, 2], rate: 0.7 },
      { c: [0, 3, 1], parent: 1, level: 2, radius: 4, axis: [0, 0, 1], rate: -1.2 },
    ]);
    const f = createFrameSet(deep);
    let elapsed = 0;
    for (const dt of [0.1, 0.25, 1 / 60]) {
      f.advance(dt);
      elapsed += dt;
    }
    const pt: V3 = [9.5, 5.6, 4.1]; // a point hanging off the innermost frame
    for (let pivot = 0; pivot < 3; pivot++) {
      expectVec(
        f.transformPoint(pivot, pt, [0, 0, 0]),
        refTransform(deep, pivot, pt, elapsed),
        4,
      );
    }
  });

  it('angle integration accumulates: two half-steps equal one full step', () => {
    const a = createFrameSet(tab);
    a.advance(0.2);
    a.advance(0.3);
    const b = createFrameSet(tab);
    b.advance(0.5);
    const pa = a.transformPoint(1, p0, [0, 0, 0]);
    const pb = b.transformPoint(1, p0, [0, 0, 0]);
    expectVec(pa, [pb[0]!, pb[1]!, pb[2]!] as V3, 5);
  });

  it('pivot −1 is the world: identity copy, identity quaternion', () => {
    expect(fs.transformPoint(-1, p0, [0, 0, 0])).toEqual([2.5, 0.4, -0.2]);
    expect(fs.quaternionOf(-1, [9, 9, 9, 9])).toEqual([0, 0, 0, 1]);
  });

  it('before any advance every transform is the identity (to f32 round-off)', () => {
    const f = createFrameSet(tab);
    expectVec(f.transformPoint(1, p0, [0, 0, 0]), p0, 6);
    expect(Array.from(f.quat.subarray(0, 4))).toEqual([0, 0, 0, 1]);
    expect(Array.from(f.centre)).toEqual(Array.from(tab.world));
  });

  it('an out-of-range pivot throws rather than answering NaN geometry', () => {
    expect(() => fs.transformPoint(99, p0, [0, 0, 0])).toThrow(/out of range/);
    expect(() => fs.quaternionOf(2, [0, 0, 0, 0])).toThrow(/out of range/);
    expect(() => fs.rideDelta(7, p0, new Float64Array(7))).toThrow(/out of range/);
  });
});

/* ------------------------------------------------------------------ riding */

/** Root at rest holding two spinning siblings — the README telemetry scenario. */
const siblings = (): PivotTable =>
  makeTable([
    { c: [0, 0, 0], parent: -1, level: 0, radius: 50, rate: 0 },
    { c: [30, 0, 0], parent: 0, level: 1, radius: 8, rate: 0.5 }, // A — ridden
    { c: [-30, 0, 0], parent: 0, level: 1, radius: 8, rate: 0.5 }, // B — sweeps
  ]);

describe('riding a frame (README telemetry: own frame 0.0, sibling sweeps)', () => {
  it('a point in the ridden frame holds still camera-relative; a sibling point sweeps', () => {
    const fs = createFrameSet(siblings());
    const pA: V3 = [33, 1, 0]; // a node of frame A (pivot 1)
    const pB: V3 = [-27, 1, 0]; // a node of frame B (pivot 2)
    let cam: V3 = [31, 0.5, 2];
    let camQ: Q = [0, 0, 0, 1];
    expect(fs.frameOfPoint(cam)).toBe(1); // inside A — that is the frame to ride

    const rel = (p: ArrayLike<number>): V3 =>
      qRot(qConj(camQ), sub([p[0]!, p[1]!, p[2]!], cam));

    const d = new Float64Array(RIDE_DELTA_SIZE);
    const pw = new Float64Array(3);
    let relA0: V3 | null = null;
    let relB0: V3 | null = null;
    let maxDriftA = 0;
    let maxSweepB = 0;
    for (let tick = 0; tick < 60; tick++) {
      fs.advance(1 / 60);
      fs.rideDelta(1, cam, d);
      cam = [cam[0] + d[4]!, cam[1] + d[5]!, cam[2] + d[6]!];
      camQ = qMul([d[0]!, d[1]!, d[2]!, d[3]!], camQ);
      const ra = rel(fs.transformPoint(1, pA, pw));
      const rb = rel(fs.transformPoint(2, pB, pw));
      if (relA0 === null || relB0 === null) {
        relA0 = ra;
        relB0 = rb;
      } else {
        maxDriftA = Math.max(maxDriftA, dist(ra, relA0));
        maxSweepB = Math.max(maxSweepB, dist(rb, relB0));
      }
    }
    expect(maxDriftA).toBeLessThan(1e-3); // "moves 0.0" — float noise only
    expect(maxSweepB).toBeGreaterThan(0.5); // the sibling visibly streams past
    expect(maxSweepB / Math.max(maxDriftA, 1e-12)).toBeGreaterThan(1e3);
  });

  it('seamless entry: riding begun mid-flight moves the camera exactly like a frame point', () => {
    const fs = createFrameSet(siblings());
    // spin a long while with NOBODY riding — the accumulated angle grows large
    for (let tick = 0; tick < 100; tick++) fs.advance(1 / 60);
    const cam: V3 = [31, 0.5, 2];
    // the frame point currently AT the camera: p0 such that transformPoint(1, p0) = cam
    const q = fs.quaternionOf(1, [0, 0, 0, 0]) as number[];
    const c0: V3 = [fs.centre[3]!, fs.centre[4]!, fs.centre[5]!]; // A: root at rest → centre = world0
    const p0 = add([30, 0, 0], qRot(qConj(q as unknown as Q), sub(cam, c0)));
    // first ridden tick, entered exactly now
    fs.advance(1 / 60);
    const d = fs.rideDelta(1, cam, new Float64Array(RIDE_DELTA_SIZE));
    const camAfter: V3 = [cam[0] + d[4]!, cam[1] + d[5]!, cam[2] + d[6]!];
    const pointAfter = fs.transformPoint(1, p0, [0, 0, 0]);
    // zero instantaneous jump: the camera is carried exactly as the co-located point is
    expectVec(pointAfter, camAfter, 4);
  });

  it('the delta is the PER-TICK angle, never the accumulated one', () => {
    const fs = createFrameSet(siblings());
    for (let tick = 0; tick < 100; tick++) fs.advance(1 / 60); // ~0.83 rad accumulated
    fs.advance(1 / 60);
    const d = fs.rideDelta(1, [31, 0.5, 2], new Float64Array(RIDE_DELTA_SIZE));
    const ang = 2 * Math.acos(Math.min(1, Math.abs(d[3]!)));
    expect(ang).toBeCloseTo(0.5 / 60, 9); // this tick's rate·dt …
    expect(ang).toBeLessThan(0.01); // … nowhere near the 0.84 rad cancellation would apply
  });

  it('nested entry: the first correction is O(rate·dt), not O(accumulated angle)', () => {
    const tab = makeTable([
      { c: [0, 0, 0], parent: -1, level: 0, radius: 60, rate: 0.3 },
      { c: [30, 0, 0], parent: 0, level: 1, radius: 8, rate: 0.7, axis: [0, 0, 1] },
    ]);
    const fs = createFrameSet(tab);
    for (let tick = 0; tick < 200; tick++) fs.advance(1 / 60); // both frames far from home
    // the camera sits near the CARRIED child centre (where a rider would actually be)
    const cam: V3 = [fs.centre[3]! + 1, fs.centre[4]!, fs.centre[5]! + 1];
    fs.advance(1 / 60);
    const d = fs.rideDelta(1, cam, new Float64Array(RIDE_DELTA_SIZE));
    const step = Math.hypot(d[4]!, d[5]!, d[6]!);
    // per-tick: ≲ (|w1|·30 + |w2|·2)·dt ≈ 0.17 units; cancelling ~3.3 rad of accumulated
    // angle at radius 30 would displace tens of units
    expect(step).toBeLessThan(0.5);
    expect(step).toBeGreaterThan(0); // it does ride — the correction is real, just incremental
  });

  it('riding the world (pivot −1) and riding during a still tick are both identity', () => {
    const fs = createFrameSet(siblings());
    fs.advance(1 / 60);
    const w = fs.rideDelta(-1, [31, 0.5, 2], new Float64Array(RIDE_DELTA_SIZE));
    expect(Array.from(w)).toEqual([0, 0, 0, 1, 0, 0, 0]);
    fs.advance(0); // nothing moved this tick → nothing to ride
    const s = fs.rideDelta(1, [31, 0.5, 2], new Float64Array(RIDE_DELTA_SIZE));
    expect(Array.from(s)).toEqual([0, 0, 0, 1, 0, 0, 0]);
  });
});

/* ------------------------------------------------------------------ frameOfPoint */

describe('frameOfPoint: deepest containing sphere, at CURRENT centres', () => {
  it('deepest wins; outside everything is the world', () => {
    const fs = createFrameSet(siblings());
    expect(fs.frameOfPoint([31, 1, 0])).toBe(1); // inside A (and the root): A
    expect(fs.frameOfPoint([-29, 0, 1])).toBe(2); // inside B: B
    expect(fs.frameOfPoint([0, 0, 0])).toBe(0); // root only
    expect(fs.frameOfPoint([200, 0, 0])).toBe(-1); // nowhere
  });

  it('spheres ride the rotation: a carried frame is found where it IS now', () => {
    const tab = makeTable([
      { c: [0, 0, 0], parent: -1, level: 0, radius: 50, rate: Math.PI / 2 },
      { c: [30, 0, 0], parent: 0, level: 1, radius: 8, rate: 0 },
    ]);
    const fs = createFrameSet(tab);
    fs.advance(1); // root turned 90° about +y: child centre (30,0,0) → (0,0,−30)
    expect(fs.frameOfPoint([1, 0, -29])).toBe(1); // found at the carried position
    expect(fs.frameOfPoint([30, 0, 0])).toBe(0); // the old spot is just root interior now
    expectVec([fs.centre[3]!, fs.centre[4]!, fs.centre[5]!], [0, 0, -30], 4);
  });

  it('a depth tie goes to the sphere the point is most deeply inside (d/r)', () => {
    const tab = makeTable([
      { c: [0, 0, 0], parent: -1, level: 0, radius: 100, rate: 0 },
      { c: [10, 0, 0], parent: 0, level: 1, radius: 12, rate: 0 },
      { c: [20, 0, 0], parent: 0, level: 1, radius: 12, rate: 0 },
    ]);
    const fs = createFrameSet(tab);
    expect(fs.frameOfPoint([19, 0, 0])).toBe(2); // in both siblings — the nearer-in wins
    expect(fs.frameOfPoint([11, 0, 0])).toBe(1);
  });
});

/* ------------------------------------------------------------------ activity */

describe('activeThisTick: the per-tick delta, never the accumulated angle', () => {
  it('false before any advance; true only when a rate and a dt both move', () => {
    const fs = createFrameSet(siblings());
    expect(fs.activeThisTick()).toBe(false);
    expect(fs.advance(1 / 60)).toBe(true);
    expect(fs.activeThisTick()).toBe(true);
    expect(fs.advance(0)).toBe(false); // a dt that moved nothing
    expect(fs.activeThisTick()).toBe(false);
  });

  it('false when rates drop to 0 even after prior spinning — and the field stays parked', () => {
    const tab = siblings();
    const fs = createFrameSet(tab);
    for (let tick = 0; tick < 30; tick++) fs.advance(1 / 60);
    const parkedQuat = fnv1a(fs.quat);
    const parkedCentre = fnv1a(fs.centre);
    tab.rate.fill(0); // the spin toggle: rates are read live
    expect(fs.advance(1 / 60)).toBe(false); // the 117-vs-83 lesson: no work while idle
    expect(fs.activeThisTick()).toBe(false);
    expect(fs.displaced()).toBe(true); // parked off base — transforms stay composed
    expect(fnv1a(fs.quat)).toBe(parkedQuat); // byte-identical: resolve never ran
    expect(fnv1a(fs.centre)).toBe(parkedCentre);
  });

  it('a rate below the epsilon·dt threshold does not count as turning', () => {
    const tab = siblings();
    tab.rate.fill(ANGLE_EPS / 2);
    const fs = createFrameSet(tab);
    expect(fs.advance(1)).toBe(false);
  });

  it('switching a frame off and on resumes from the parked angle (integrated, not clocked)', () => {
    const tab = siblings();
    const a = createFrameSet(tab);
    a.advance(0.4);
    tab.rate.fill(0);
    a.advance(5); // a long idle stretch — must not advance anything
    tab.rate[1] = 0.5; // back on (the root stays at rest, as siblings() builds it)
    tab.rate[2] = 0.5;
    a.advance(0.1);
    const b = createFrameSet(siblings());
    b.advance(0.5); // the same total spun time, uninterrupted
    const pa = a.transformPoint(1, [33, 1, 0], [0, 0, 0]);
    const pb = b.transformPoint(1, [33, 1, 0], [0, 0, 0]);
    expectVec(pa, [pb[0]!, pb[1]!, pb[2]!] as V3, 5);
  });

  it('displaced() is false at base, and after spinning exactly back', () => {
    const tab = siblings();
    const fs = createFrameSet(tab);
    expect(fs.displaced()).toBe(false);
    fs.advance(0.25);
    expect(fs.displaced()).toBe(true);
    fs.advance(-0.25); // rewind — the angle returns to ~0
    expect(fs.displaced()).toBe(false);
  });
});

/* ------------------------------------------------------------------ determinism */

describe('determinism under a fixed dt sequence (§10.1)', () => {
  const DTS = [1 / 60, 1 / 30, 0.25, 0, 1 / 144, 0.5, 1 / 60];

  it('two runs over the synth hierarchy are byte-identical', () => {
    const run = (): { q: number; c: number; a: number } => {
      const fs = createFrameSet(compiled().pivots);
      for (const dt of DTS) fs.advance(dt);
      return { q: fnv1a(fs.quat), c: fnv1a(fs.centre), a: fnv1a(fs.axisWorld) };
    };
    const one = run();
    expect(run()).toEqual(one);
    // recorded from a run and pinned (§10.9 style, like the stage-5 hierarchy goldens):
    // a silent change to the frame maths fails here even if it stays self-consistent.
    // Re-recorded 2026-11-09 (was q 706161765, c 4101504357, a 2732415901): the synth
    // constant-density rescale re-deals the galaxy/system facets (~n^⅓ counts), so the
    // compiled pivot table this digests moved — the frame maths itself did not.
    expect(one).toEqual({ q: 2700951938, c: 2433892124, a: 1476580600 });
  });

  it('a different dt sequence lands elsewhere (the pin is not vacuous)', () => {
    const fs1 = createFrameSet(compiled().pivots);
    const fs2 = createFrameSet(compiled().pivots);
    for (const dt of DTS) fs1.advance(dt);
    for (const dt of DTS) fs2.advance(dt * 1.5);
    expect(fnv1a(fs1.quat)).not.toBe(fnv1a(fs2.quat));
  });
});

/* ------------------------------------------------------------------ O(pivots) */

describe('O(pivots) by construction (§4: per-frame CPU never O(nodes))', () => {
  /* `advance` cannot touch a node-length array because none can reach it: the factory
   * consumes only the PivotTable (every field pivot-length), `advance` takes one scalar,
   * and every buffer the set owns is pivot-length. Asserted structurally — there is no
   * argument through which node data could enter, which is the by-construction claim. */
  it('the whole API surface is pivot-sized over a 2000-node corpus', () => {
    const r = compiled();
    const P = r.pivots.count;
    expect(P).toBeGreaterThan(0);
    expect(P).toBeLessThan(SYNTH.n / 4); // pivots are groups: far fewer than nodes
    // the factory's ONLY input: a table whose every array is pivot-length
    expect(createFrameSet.length).toBe(1);
    expect(r.pivots.centre.length).toBe(P * 3);
    expect(r.pivots.world.length).toBe(P * 3);
    expect(r.pivots.parent.length).toBe(P);
    expect(r.pivots.level.length).toBe(P);
    expect(r.pivots.radius.length).toBe(P);
    expect(r.pivots.axis.length).toBe(P * 3);
    expect(r.pivots.rate.length).toBe(P);
    const fs = createFrameSet(r.pivots);
    // advance takes one scalar dt — no argument admits node-length data
    expect(fs.advance.length).toBe(1);
    // everything the set owns is pivot-length too
    expect(fs.count).toBe(P);
    expect(fs.quat.length).toBe(P * 4);
    expect(fs.centre.length).toBe(P * 3);
    expect(fs.axisWorld.length).toBe(P * 3);
  });

  it('per-NODE work stays outside advance: transformPoint is the caller’s per-point verb', () => {
    // the reference transform exists and works per point, but it is a separate call the
    // caller makes (tests, pick rebuilds, telemetry) — advance never iterates nodes
    const r = compiled();
    const fs = createFrameSet(r.pivots);
    fs.advance(0.2);
    const out = [0, 0, 0];
    const i = 123;
    fs.transformPoint(r.nodePivot[i]!, r.positions, out, i * 3);
    expect(out.every((v) => Number.isFinite(v))).toBe(true);
    // chain consistency: the node's transformed position matches the nest-form reference
    const want = refTransform(
      r.pivots,
      r.nodePivot[i]!,
      [r.positions[i * 3]!, r.positions[i * 3 + 1]!, r.positions[i * 3 + 2]!],
      0.2,
    );
    expectVec(out, want, 3);
  });
});
