/* The stage-5 shape library (DESIGN.md §3.3): the parametric manifolds — curves, surfaces,
 * volumes — registered under their design names. Importing this module registers them all.
 *
 * A shape here is a pure function of (u, radius, params) and NOTHING else (see ./types):
 * no member ids, no salt, no hashing. Everything the predecessor's slot functions drew
 * from `rnd(seed, …)` — ring hair, arm jitter, branch thickness — is deliberately ABSENT:
 * that is scatter's job now, so that the same spiral can be bare or fuzzy without two
 * spiral shapes existing. The logic itself is ported from
 * `SecondBrain/tools/vault-site-3d/src/arrange.js` (galaxySlot's spiral math, dendriteTiers
 * / dendriteGeom / dendriteWalk), translated to the four-knob contract.
 *
 * Shared conventions, stated once:
 *   - ORIENTATION is canonical, Y up: flat shapes (ring, disc, spiralArm, torus) lie in
 *     the XZ plane; axial shapes (helix, dendrite, line's normal) grow along Y. The
 *     hierarchy compiler (§3.4) owns orientation — inclination, inherited planes — later.
 *   - SIZE comes from `radius` alone. Each shape documents its bound: most fit inside the
 *     sphere |p| ≤ radius exactly; `box` is the one exception (|p| ≤ radius·√3).
 *   - Every u coordinate arrives in [0, 1) (engine contract); shapes never read outside
 *     the first `dim * count` entries and never write into `u`.
 *   - House rules (§10): no Math.random, no Date.now, no per-member objects, loops only.
 *     Same input → byte-identical output, pinned by FNV goldens in test/shapes.test.ts.
 */

import { registerShape } from './registry';
import type { Shape } from './types';

const TAU = Math.PI * 2;

/* ================================================================== curves (dim 1) */

/** `ring` — a circle of radius `radius` in the XZ plane.
 *
 *  Param axes: u0 = angle fraction; θ = 2π·u0, starting at +X and winding toward +Z.
 *  Bound: |p| = radius exactly, y = 0 exactly. Thickness (the predecessor's "hair of
 *  depth" that kept an edge-on ring visible) is a scatter, not baked in here. */
const ring: Shape = {
  key: 'ring',
  dim: 1,
  place(u, radius, count, out, outOffset): void {
    for (let k = 0; k < count; k++) {
      const th = u[k]! * TAU;
      const o = outOffset + k * 3;
      out[o] = Math.cos(th) * radius;
      out[o + 1] = 0;
      out[o + 2] = Math.sin(th) * radius;
    }
  },
};

/** `line` — a straight segment along the X axis.
 *
 *  Param axes: u0 = position along the segment; x = (2·u0 − 1)·radius, y = z = 0.
 *  Bound: |p| ≤ radius exactly (x spans [-radius, radius)). It lies along X rather than Y
 *  so that a line of groups reads side-by-side under the canonical Y-up camera; the
 *  hierarchy compiler reorients it like any other shape. */
const line: Shape = {
  key: 'line',
  dim: 1,
  place(u, radius, count, out, outOffset): void {
    for (let k = 0; k < count; k++) {
      const o = outOffset + k * 3;
      out[o] = (2 * u[k]! - 1) * radius;
      out[o + 1] = 0;
      out[o + 2] = 0;
    }
  },
};

/** `helix` — a coil climbing the Y axis.
 *
 *  Param axes: u0 = progress root-to-tip; it drives BOTH the height (linearly, so y is
 *  strictly monotone in u0) and the winding angle θ = 2π·turns·u0.
 *  Params: `turns` — full revolutions over the length (default 3);
 *          `coil`  — coil radius as a fraction of `radius` (default 0.4).
 *  Size policy: the coil radius is coil·radius and the half-height is
 *  sqrt(radius² − (coil·radius)²), so every point satisfies |p| ≤ radius exactly —
 *  a fatter coil is a shorter one. `coil` ≥ 1 degenerates to a flat ring (half-height 0). */
const helix: Shape = {
  key: 'helix',
  dim: 1,
  params: [
    { key: 'turns', label: 'turns', default: 3, min: 0.5, max: 12, step: 0.5 },
    { key: 'coil', label: 'coil', default: 0.4, min: 0.05, max: 1, step: 0.05 },
  ],
  place(u, radius, count, out, outOffset, params): void {
    const turns = params['turns'] ?? 3;
    const coil = params['coil'] ?? 0.4;
    const rc = radius * coil;
    const h = Math.sqrt(Math.max(0, radius * radius - rc * rc));
    for (let k = 0; k < count; k++) {
      const t = u[k]!;
      const th = t * turns * TAU;
      const o = outOffset + k * 3;
      out[o] = Math.cos(th) * rc;
      out[o + 1] = (2 * t - 1) * h;
      out[o + 2] = Math.sin(th) * rc;
    }
  },
};

/** `spiralArm` — a multi-arm logarithmic-ish spiral in the XZ plane: the bare bone of the
 *  predecessor's spiral galaxy (arrange.js `galaxySlot`), arms + swirl + optional bar, no
 *  jitter and no bulge thickness — those are scatter.
 *
 *  Param axes: u0 is read in two parts — `floor(u0·arms)` picks the ARM, and the
 *  fractional remainder `t = u0·arms − arm` is the progress OUT along that arm. The fold
 *  is DECLARED (`folds` below), so the engine applies an axis-0 density within each arm:
 *  equal-spaced quantiles fill every arm equally (±1 member, the predecessor's round-robin
 *  `k % arms` in four-knob form) and a density shapes each arm's radial profile with the
 *  full run — it never re-assigns arms (undeclared, an exponential warp measured 66/25/9%
 *  arm populations; that bug is why Shape.folds exists).
 *
 *  Radius r = radius·(bar + (1−bar)·tt^radialPower) over the arm run tt = (t−bar)/(1−bar);
 *  with the defaults (bar 0, radialPower 0.5) that is the predecessor's r = radius·√t area
 *  correction — uniform density along the parameter is then uniform over the implied disc.
 *  A preset warping the radial profile overrides `radialPower: 1` so the warped parameter
 *  IS the radius fraction, same escape hatch as `disc`. Angle θ = arm·2π/arms +
 *  (r/radius − bar)·swirl: each arm starts at its even angular offset — anchored AT the
 *  bar rim when there is one — and winds by `swirl` radians over its radial run.
 *
 *  The bar (arrange.js galaxySlot's `barred` branches, ported): members with t < bar lie
 *  ON a straight spoke along their arm's launch angle, centre → bar end at bar·radius
 *  (for the canonical arms 2: one straight bar through the core along ±X), and the arms
 *  launch from the bar ends — continuous at r = bar·radius, which is the whole visual
 *  difference between a barred spiral and an ordinary one. The predecessor's ±0.05R bar
 *  wobble and bulge thickness are scatter's job here.
 *
 *  Params: `arms`        — arm count (default 2, min 1, rounded);
 *          `swirl`       — how tightly the arms wind, in radians at the rim (default 2.5);
 *          `bar`         — bar half-length as a fraction of radius (default 0 = no bar;
 *                          clamped to [0, 0.95]; meant for arms 2 — more arms make it a
 *                          spoked core);
 *          `radialPower` — the arm-run radial exponent (default 0.5, the area correction).
 *  Bound: |p| ≤ radius exactly (r ≤ radius, y = 0). */
const spiralArm: Shape = {
  key: 'spiralArm',
  dim: 1,
  params: [
    { key: 'arms', label: 'arms', default: 2, min: 1, max: 8, step: 1 },
    { key: 'swirl', label: 'swirl', default: 2.5, min: 0, max: 10, step: 0.25 },
    { key: 'bar', label: 'bar', default: 0, min: 0, max: 0.95, step: 0.05 },
    { key: 'radialPower', label: 'radialPwr', default: 0.5, min: 0.1, max: 2, step: 0.05 },
  ],
  folds(params): number {
    return Math.max(1, Math.round(params['arms'] ?? 2));
  },
  place(u, radius, count, out, outOffset, params): void {
    const arms = Math.max(1, Math.round(params['arms'] ?? 2));
    const swirl = params['swirl'] ?? 2.5;
    const bar = Math.min(0.95, Math.max(0, params['bar'] ?? 0));
    const rp = params['radialPower'] ?? 0.5;
    const invR = 1 / Math.max(radius, 1e-9);
    for (let k = 0; k < count; k++) {
      const s = u[k]! * arms;
      let arm = Math.floor(s);
      if (arm >= arms) arm = arms - 1; // u0 < 1, but guard f32 rounding at the seam
      const t = s - arm;
      const armTh = (arm / arms) * TAU;
      const o = outOffset + k * 3;
      if (t < bar) {
        // in the bar: on the straight spoke under this arm's launch angle, centre → end
        const d = t * radius;
        out[o] = Math.cos(armTh) * d;
        out[o + 1] = 0;
        out[o + 2] = Math.sin(armTh) * d;
        continue;
      }
      const tt = (t - bar) / (1 - bar);
      // rp === 0.5 keeps Math.sqrt (byte-stable default; pow(x, 0.5) is not pinned to it)
      const run = rp === 0.5 ? Math.sqrt(tt) : Math.pow(tt, rp);
      const r = radius * (bar + (1 - bar) * run);
      const th = armTh + (r * invR - bar) * swirl;
      out[o] = Math.cos(th) * r;
      out[o + 1] = 0;
      out[o + 2] = Math.sin(th) * r;
    }
  },
};

/* ------------------------------------------------------------------ dendrite
 * Ported from arrange.js: a diffusion-limited-aggregation LOOK with a closed form. A real
 * DLA grows by random walk and is iterative, which placement cannot be; but a DLA looks
 * like a bifurcating tree, and a bifurcating tree has a closed form. A branch index read
 * as a binary-heap address IS a path: its leading 1 is the root, each later bit picks a
 * fork, so a position is the sum of one step per bit, each step the previous direction
 * rotated and shortened. O(log k) per member, no state, no iteration. */

/** How many branch tiers a group of `m` members can fill (arrange.js `dendriteTiers`):
 *  depth chosen so each SEGMENT carries ~12 members — enough to read as a line rather
 *  than a dotted one. Solving for segments, not levels: a tree of depth d has 2^(d+1)−1
 *  segments, so the −1 matters. Exported because the default `tiers` param comes from it
 *  and tests pin the handover points. */
export function dendriteTiers(m: number): number {
  return Math.max(1, Math.min(9, Math.round(Math.log2(Math.max(8, m / 12))) - 1));
}

/** Rodrigues rotation of `v` about unit axis `k` by angle `a`, into `dst` (3 floats).
 *  `dst` may alias neither `v` nor `k`. */
function rot3(
  vx: number,
  vy: number,
  vz: number,
  kx: number,
  ky: number,
  kz: number,
  a: number,
  dst: Float64Array,
  d0: number,
): void {
  const c = Math.cos(a);
  const s = Math.sin(a);
  const cx = ky * vz - kz * vy;
  const cy = kz * vx - kx * vz;
  const cz = kx * vy - ky * vx;
  const kd = kx * vx + ky * vy + kz * vz;
  dst[d0] = vx * c + cx * s + kx * kd * (1 - c);
  dst[d0 + 1] = vy * c + cy * s + ky * kd * (1 - c);
  dst[d0 + 2] = vz * c + cz * s + kz * kd * (1 - c);
}

const GOLDEN_ANGLE = 2.399963229728653;

/* Walk scratch: [0..2] dir, [3..5] in-plane reference u, [6..8] rotation axis,
 * [9..11] rotated dir, [12..14] rotated u. Module-level is safe — placement is
 * synchronous and single-threaded (same argument as the engine's u scratch). */
const dw = new Float64Array(15);

/** `dendrite` — a branching frond growing up the Y axis (trunk along +Y), members lying
 *  ALONG the branches, not at the nodes between them (arrange.js `dendriteSlot` +
 *  `dendriteWalk`, with the hashed branch-thickness jitter removed — flesh is scatter's
 *  job, so the shape places points exactly on the branch centrelines).
 *
 *  Param axes (tree-structured): u0 is read as a rank r = floor(u0·count); the segment is
 *  r mod segs (so consecutive ranks interleave across all 2^(tiers+1)−1 segments and every
 *  branch gets members even when count < segs·12) and floor(r/segs) walks outward along
 *  that segment. A density warp on u0 therefore shifts members between whole branches —
 *  coarse on purpose; within-branch spacing is fixed by the tree.
 *
 *  Params: `tiers`  — fork depth (default `dendriteTiers(count)`, clamped to [1, 16]);
 *          `shrink` — each tier's length as a fraction of the one above (default 0.77 —
 *                     the predecessor's visibility fix: at 0.62 a fourth-level branch was
 *                     a speck; clamped to [0.05, 0.95]);
 *          `split`  — fork half-angle in radians (default 0.4);
 *          `twist`  — extra roll per tier added to the golden-angle phyllotaxis that
 *                     spreads forks out of plane (default 0.26).
 *
 *  Size policy: the trunk length is radius·(1−shrink)/(1−shrink^(tiers+1)), so the longest
 *  root-to-tip PATH sums to exactly `radius`; |p| ≤ radius by the triangle inequality. */
const dendrite: Shape = {
  key: 'dendrite',
  dim: 1,
  /* `tiers` default 0 is the REPRESENTATIVE "auto" sentinel: the real fallback is
   * `dendriteTiers(count)` (count-derived), which no constant can match. A caller OMITS
   * `tiers` (the default-equals-0 omit rule) to get the auto depth, and sends 1..9 to
   * override; note the shape itself reads a literally-sent 0 as ~1 tier (`0 ?? x` is 0),
   * so 0 means "auto" only by omission, never by value. The other three are plain `??
   * default` params. */
  params: [
    { key: 'tiers', label: 'tiers·0=auto', default: 0, min: 0, max: 9, step: 1 },
    { key: 'shrink', label: 'shrink', default: 0.77, min: 0.05, max: 0.95, step: 0.01 },
    { key: 'split', label: 'split', default: 0.4, min: 0, max: 1.5, step: 0.05 },
    { key: 'twist', label: 'twist', default: 0.26, min: 0, max: 1.5, step: 0.02 },
  ],
  place(u, radius, count, out, outOffset, params): void {
    const tiers = Math.max(1, Math.min(16, Math.round(params['tiers'] ?? dendriteTiers(count))));
    const shrink = Math.min(0.95, Math.max(0.05, params['shrink'] ?? 0.77));
    const split = params['split'] ?? 0.4;
    const twist = params['twist'] ?? 0.26;
    const segs = (1 << (tiers + 1)) - 1; // every node of a full binary tree is a segment
    const len0 = (radius * (1 - shrink)) / (1 - Math.pow(shrink, tiers + 1));
    const per = Math.max(1, Math.ceil(count / segs));

    for (let k = 0; k < count; k++) {
      let rank = Math.floor(u[k]! * count);
      if (rank >= count) rank = count - 1; // guard f32 rounding at u0 → 1⁻
      if (rank < 0) rank = 0;
      const seg = rank % segs;
      const along = (Math.floor(rank / segs) + 0.5) / per;

      /* ---- dendriteWalk(seg + 1, along), ported: the heap address's bits are the path.
       * The root IS a real segment — no turn before tier 0 (the predecessor's "tree with
       * no trunk" fix). */
      const idx = seg + 1;
      const depth = 31 - Math.clz32(idx);
      dw[0] = 0; dw[1] = 1; dw[2] = 0; // dir: the trunk grows along +Y (canonical up)
      dw[3] = 1; dw[4] = 0; dw[5] = 0; // in-plane reference
      let x = 0;
      let y = 0;
      let z = 0;
      let len = len0;
      for (let b = depth; b >= 0; b--) {
        if (b < depth) {
          const bit = (idx >> b) & 1;
          const phi = (depth - b) * GOLDEN_ANGLE + twist;
          // fork axis: the reference rolled about the current direction by the phyllotaxis
          rot3(dw[3]!, dw[4]!, dw[5]!, dw[0]!, dw[1]!, dw[2]!, phi, dw, 6);
          const turn = bit ? split : -split;
          rot3(dw[0]!, dw[1]!, dw[2]!, dw[6]!, dw[7]!, dw[8]!, turn, dw, 9);
          rot3(dw[3]!, dw[4]!, dw[5]!, dw[6]!, dw[7]!, dw[8]!, turn, dw, 12);
          dw[0] = dw[9]!; dw[1] = dw[10]!; dw[2] = dw[11]!;
          dw[3] = dw[12]!; dw[4] = dw[13]!; dw[5] = dw[14]!;
        }
        const step = b === 0 ? len * along : len; // the last segment only partway
        x += dw[0]! * step;
        y += dw[1]! * step;
        z += dw[2]! * step;
        if (b > 0) len *= shrink;
      }

      const o = outOffset + k * 3;
      out[o] = x;
      out[o + 1] = y;
      out[o + 2] = z;
    }
  },
};

/* ================================================================== surfaces (dim 2) */

/** `shell` — the surface of a sphere of radius `radius`, area-uniform.
 *
 *  Param axes: u0 = azimuth fraction (θ = 2π·u0 in the XZ plane, from +X toward +Z);
 *  u1 = cos-weighted polar coordinate: y/radius = 1 − 2·u1. Mapping u1 linearly to the
 *  COSINE of the polar angle is the area-uniform (Archimedes) mapping — equal u1 intervals
 *  cover equal sphere AREA, so uniform density in parameter space is uniform density on
 *  the surface, with no crowding at the poles. A density on axis 1 therefore reweights
 *  latitude by area, which is the intuitive reading.
 *  Bound: |p| = radius exactly. */
const shell: Shape = {
  key: 'shell',
  dim: 2,
  place(u, radius, count, out, outOffset): void {
    for (let k = 0; k < count; k++) {
      const b = k * 2;
      const th = u[b]! * TAU;
      const yn = 1 - 2 * u[b + 1]!;
      const rxz = Math.sqrt(Math.max(0, 1 - yn * yn)) * radius;
      const o = outOffset + k * 3;
      out[o] = Math.cos(th) * rxz;
      out[o + 1] = yn * radius;
      out[o + 2] = Math.sin(th) * rxz;
    }
  },
};

/** `disc` — a flat filled disc in the XZ plane (y = 0 exactly).
 *
 *  Param axes: u0 = angle fraction (θ = 2π·u0); u1 = radial coordinate, mapped through
 *  r = radius·u1^radialPower.
 *  Params: `radialPower` — the radial exponent (default 0.5, the sqrt AREA CORRECTION:
 *  equal u1 intervals then cover equal annulus area, so uniform density means uniform
 *  surface density instead of a crowded core).
 *  NOTE for presets: a galaxy preset sets `densityAxis: 1` to shape the radial profile
 *  (e.g. an exponential disc) and OVERRIDES this correction with `radialPower: 1`, taking
 *  the warped u1 as the radius fraction directly — otherwise the sqrt would distort the
 *  profile the density asked for. The two knobs compose as warp-then-power.
 *  Bound: |p| ≤ radius exactly. */
const disc: Shape = {
  key: 'disc',
  dim: 2,
  params: [{ key: 'radialPower', label: 'radialPwr', default: 0.5, min: 0.1, max: 2, step: 0.05 }],
  place(u, radius, count, out, outOffset, params): void {
    const pow = params['radialPower'] ?? 0.5;
    for (let k = 0; k < count; k++) {
      const b = k * 2;
      const th = u[b]! * TAU;
      const r = radius * Math.pow(u[b + 1]!, pow);
      const o = outOffset + k * 3;
      out[o] = Math.cos(th) * r;
      out[o + 1] = 0;
      out[o + 2] = Math.sin(th) * r;
    }
  },
};

/** `torus` — a torus around the Y axis, its spine circle in the XZ plane.
 *
 *  Param axes: u0 = angle around the main ring (θ = 2π·u0, from +X toward +Z);
 *  u1 = angle around the tube (φ = 2π·u1; φ = 0 points outward in the XZ plane, φ = π/4
 *  band rises toward +Y).
 *  Params: `tube` — tube radius as a fraction of `radius` (default 0.25, clamped to
 *  (0, 0.5]). Size policy: spine radius = radius·(1 − tube), tube radius = radius·tube,
 *  so the outer equator touches |p| = radius exactly and never exceeds it.
 *  (Not area-uniform in u0 — the inner rim is slightly denser than the outer, the usual
 *  parametric-torus bias; at tube ≤ 0.5·spine the effect is below what scatter hides.)
 *  Bound: |p| ≤ radius exactly. */
const torus: Shape = {
  key: 'torus',
  dim: 2,
  params: [{ key: 'tube', label: 'tube', default: 0.25, min: 0.02, max: 0.5, step: 0.01 }],
  place(u, radius, count, out, outOffset, params): void {
    const tube = Math.min(0.5, Math.max(1e-6, params['tube'] ?? 0.25));
    const spine = radius * (1 - tube);
    const rt = radius * tube;
    for (let k = 0; k < count; k++) {
      const b = k * 2;
      const th = u[b]! * TAU;
      const ph = u[b + 1]! * TAU;
      const rxz = spine + Math.cos(ph) * rt;
      const o = outOffset + k * 3;
      out[o] = Math.cos(th) * rxz;
      out[o + 1] = Math.sin(ph) * rt;
      out[o + 2] = Math.sin(th) * rxz;
    }
  },
};

/* ================================================================== volumes (dim 3) */

/** `ball` — a solid sphere, volume-uniform by default; optionally squashed.
 *
 *  Param axes: u0 = azimuth fraction (θ = 2π·u0); u1 = cos-weighted polar (as `shell`);
 *  u2 = radial coordinate, r = radius·u2^radialPower.
 *  Params: `radialPower` — the radial exponent (default ⅓, computed via `cbrt`: the
 *          CUBE-ROOT CORRECTION — equal u2 intervals cover equal VOLUME, so uniform
 *          density in parameter space is uniform density in the ball, the predecessor's
 *          "cbrt keeps a ball uniform"). A density on axis 2 shapes the radial mass
 *          profile; a preset doing that OVERRIDES this correction with `radialPower: 1`
 *          so the warped u2 IS the radius fraction — otherwise the cbrt composes INTO the
 *          asked-for profile and silently changes it (a powerLaw's exponent is divided by
 *          3, a gaussian stops being one). Same warp-then-power escape hatch as `disc`.
 *          `aspectY` — Y half-extent multiplier (default 1): the predecessor's elliptical
 *          `ell` squash, applied to the polar axis so a 0.6 ball reads as an ellipse.
 *  Bound: |p| ≤ radius·max(1, aspectY) — exactly `radius` for aspectY ≤ 1. */
const ball: Shape = {
  key: 'ball',
  dim: 3,
  /* `radialPower` default ⅓ is REPRESENTATIVE: the real fallback when the param is ABSENT is
   * `Math.cbrt(t)` (the volume-uniform cube root), which `Math.pow(t, 1/3)` does not reproduce
   * bit-for-bit. A caller OMITS `radialPower` (the default-equals-⅓ omit rule) to keep the
   * exact cbrt, and sends a value to switch to `pow`. `aspectY` is a plain `?? 1` param. */
  params: [
    { key: 'radialPower', label: 'radialPwr', default: 1 / 3, min: 0.1, max: 2, step: 0.05 },
    { key: 'aspectY', label: 'aspectY', default: 1, min: 0.1, max: 3, step: 0.1 },
  ],
  place(u, radius, count, out, outOffset, params): void {
    const rp = params['radialPower'];
    const ay = params['aspectY'] ?? 1;
    for (let k = 0; k < count; k++) {
      const b = k * 3;
      const th = u[b]! * TAU;
      const yn = 1 - 2 * u[b + 1]!;
      const t = u[b + 2]!;
      const r = radius * (rp === undefined ? Math.cbrt(t) : Math.pow(t, rp));
      const rxz = Math.sqrt(Math.max(0, 1 - yn * yn)) * r;
      const o = outOffset + k * 3;
      out[o] = Math.cos(th) * rxz;
      out[o + 1] = yn * r * ay;
      out[o + 2] = Math.sin(th) * rxz;
    }
  },
};

/** `box` — an axis-aligned solid box, volume-uniform.
 *
 *  Param axes: u0 → x, u1 → y, u2 → z, each mapped linearly to [-half, half).
 *  Params: `aspectY`, `aspectZ` — per-axis half-extent multipliers relative to X's
 *  (default 1 each: a cube), so a slab is `{ aspectY: 0.1 }` rather than a second shape.
 *  Size policy: the X half-extent is `radius`; Y and Z are radius·aspect.
 *  Bound — THE documented exception to |p| ≤ radius: a corner of the default cube sits at
 *  radius·√3. The radius is the half-extent, not a bounding-sphere radius, because a box
 *  whose sides shrink to radius/√3 would gap visibly next to a ball of the same spec. */
const box: Shape = {
  key: 'box',
  dim: 3,
  params: [
    { key: 'aspectY', label: 'aspectY', default: 1, min: 0.1, max: 3, step: 0.1 },
    { key: 'aspectZ', label: 'aspectZ', default: 1, min: 0.1, max: 3, step: 0.1 },
  ],
  place(u, radius, count, out, outOffset, params): void {
    const hy = radius * (params['aspectY'] ?? 1);
    const hz = radius * (params['aspectZ'] ?? 1);
    for (let k = 0; k < count; k++) {
      const b = k * 3;
      const o = outOffset + k * 3;
      out[o] = (2 * u[b]! - 1) * radius;
      out[o + 1] = (2 * u[b + 1]! - 1) * hy;
      out[o + 2] = (2 * u[b + 2]! - 1) * hz;
    }
  },
};

/* `cloud` (dim 3, gaussian ball) is the engine's built-in — registered in ./engine, not
 * duplicated here; it needed no extension for the §3.3 set. */

registerShape(ring);
registerShape(line);
registerShape(helix);
registerShape(spiralArm);
registerShape(dendrite);
registerShape(shell);
registerShape(disc);
registerShape(torus);
registerShape(ball);
registerShape(box);
