/* The placement engine (DESIGN.md §3.3): the ONE composition of the four knobs.
 *
 *     param = density.warp( ordering.assign(member) )    // where along the shape
 *     pos   = shape.at(param, radius) + scatter.offset(member)
 *
 * `placeGroup` is the only code that wires a shape to an ordering to a density to a
 * scatter; shapes/orderings/densities/scatters never see each other. The hierarchy
 * compiler (§3.4) calls it once per group — per spec change, never per frame (§4).
 *
 * Allocation discipline: beyond the parameter scratch below (grown monotonically, reused
 * across calls) the engine allocates nothing — no per-member objects, no spreads, loops
 * only (§10.2–3). Orderings own their returned quantile array; the engine copies out of it
 * immediately and drops the reference.
 */

import { hash } from '../core/hash';
import { probitApprox } from './stat';
import {
  getDensity,
  getOrdering,
  getScatter,
  getShape,
  registerDensity,
  registerOrdering,
  registerScatter,
  registerShape,
} from './registry';
import type { Density, Ordering, Params, PlaceInput, PlacementSpec, Scatter, Shape } from './types';

/** Shared empty params so a spec without `params` costs no allocation per group. */
const NO_PARAMS: Params = {};

/* ------------------------------------------------------------------ the [0,1) cap
 * Largest float32 strictly below 1. EVERY value written into the f32 u buffer goes
 * through `capU`, because the Shape contract promises coordinates in [0, 1) AFTER f32
 * storage: a raw `hash` draw like 0.99999998766 (hash(19847011, 101)) rounds to exactly
 * 1.0f, and so can a custom density's 1 − 2⁻⁵⁰. The library densities cap their own f64
 * outputs the same way (src/place/densities.ts), but hashed axes, `uniform`, and
 * user-registered densities land here unprotected — this is the engine's own fence. */
const UNIT_MAX = 1 - 2 ** -24;
const capU = (v: number): number => (v < UNIT_MAX ? v : UNIT_MAX);

/* ------------------------------------------------------------------ the u scratch
 * One buffer for parameter coordinates, grown monotonically and never shrunk: the compile
 * walks groups large and small, and reallocating per group would be O(groups) garbage for
 * nothing. Consequence, part of the Shape contract: the `u` a shape receives may be LONGER
 * than `dim * count`. Module-level state is safe because placement is synchronous and
 * single-threaded; nothing retains `u` past the `shape.place` call. */
let uScratch = new Float32Array(0);

function uBuffer(size: number): Float32Array {
  if (uScratch.length < size) {
    let cap = uScratch.length > 0 ? uScratch.length : 1024;
    while (cap < size) cap *= 2;
    uScratch = new Float32Array(cap);
  }
  return uScratch;
}

/**
 * Place one group's members (LOCAL coordinates, group centre at origin — see `PlaceInput`).
 *
 * Pipeline, in the pinned order:
 *   1. ordering.assign      → axis-0 quantiles, a permutation of (rank+0.5)/count
 *   2. density.warp         → applied to the axis `spec.densityAxis` selects (default 0);
 *                             axes ≥ 1 are hashed uniforms `hash(member, salt + 100 + axis)`.
 *                             On a shape declaring `folds` (spiralArm's arms), an axis-0
 *                             warp runs WITHIN each fold — see Shape.folds in ./types
 *   3. shape.place          → manifold positions from the warped parameters
 *   4. scatter.offset       → adds displacement per member, after the shape wrote
 *
 * Spec errors (unknown knob key, densityAxis ≥ shape.dim) throw even for an empty group —
 * a typo in a spec must not hide behind a group that happened to have no members.
 */
export function placeGroup(spec: PlacementSpec, inp: PlaceInput): void {
  const shape = getShape(spec.shape);
  const ordering = getOrdering(spec.ordering ?? 'byIndex');
  const density = getDensity(spec.density ?? 'uniform');
  const scatter = getScatter(spec.scatter ?? 'none');
  const params = spec.params ?? NO_PARAMS;
  const dim = shape.dim;
  const densityAxis = spec.densityAxis ?? 0;
  if (densityAxis >= dim) {
    throw new RangeError(
      `densityAxis ${densityAxis} out of range for shape "${shape.key}" (dim ${dim})`,
    );
  }

  const count = inp.count;
  if (count === 0) return;

  const { members, out, outOffset, radius, salt } = inp;
  const u = uBuffer(dim * count);

  // Axis 0: the ordering's quantiles, warped in place when the density targets axis 0.
  // `stringParams` reaches ONLY the ordering (the facet/column key for byFacet/byColumn —
  // see PlacementSpec.stringParams); the other three knobs stay numeric-only.
  const q = ordering.assign(inp, params, spec.stringParams);
  if (densityAxis === 0) {
    const folds = shape.folds !== undefined ? Math.max(1, Math.round(shape.folds(params))) : 1;
    if (folds > 1) {
      // Folded axis (Shape.folds): warp WITHIN each segment, so the density shapes every
      // segment's profile instead of dragging members across segment boundaries. Written
      // as q + (warp(frac) − frac)/folds — algebraically (seg + warp(frac))/folds, but
      // exactly the identity when the warp is (uniform always is, bytes matter here).
      for (let k = 0; k < count; k++) {
        const qk = q[k]!;
        const s = qk * folds;
        let seg = Math.floor(s);
        if (seg >= folds) seg = folds - 1; // f64 rounding guard at q → 1⁻
        const frac = s - seg;
        u[k * dim] = capU(qk + (density.warp(frac, params) - frac) / folds);
      }
    } else {
      for (let k = 0; k < count; k++) u[k * dim] = capU(density.warp(q[k]!, params));
    }
  } else {
    for (let k = 0; k < count; k++) u[k * dim] = capU(q[k]!);
  }

  // Axes 1..dim-1: hashed uniforms per member — `salt + 100 + axis` keeps the streams
  // independent of each other and of anything a scatter draws from the base salt.
  for (let a = 1; a < dim; a++) {
    const axisSalt = salt + 100 + a;
    if (densityAxis === a) {
      for (let k = 0; k < count; k++) {
        u[k * dim + a] = capU(density.warp(hash(members[k]!, axisSalt), params));
      }
    } else {
      for (let k = 0; k < count; k++) u[k * dim + a] = capU(hash(members[k]!, axisSalt));
    }
  }

  shape.place(u, radius, count, out, outOffset, params);

  for (let k = 0; k < count; k++) {
    scatter.offset(k, members[k]!, salt, radius, out, outOffset + 3 * k, params);
  }
}

/* ================================================================== built-ins
 * The minimal knob set that makes the engine testable standalone — registered as a side
 * effect of importing this module. The full libraries (ring/disc/spiral shapes, facet and
 * adjacency orderings, the density family, the scatter family) live in their own stage-5
 * modules and register themselves the same way. */

/** `byIndex`: members in ascending id order — member rank = position after sorting ids.
 *  Rank is by id, not by position in `members`, so the result cannot depend on how the
 *  caller happened to assemble the group (iteration order is the classic determinism leak,
 *  §10). The common case — members already ascending — is detected in one pass and skips
 *  the sort. */
const byIndex: Ordering = {
  key: 'byIndex',
  assign(inp: PlaceInput): Float32Array {
    const n = inp.count;
    const members = inp.members;
    const q = new Float32Array(n);
    let sorted = true;
    for (let k = 1; k < n; k++) {
      if (members[k]! < members[k - 1]!) {
        sorted = false;
        break;
      }
    }
    if (sorted) {
      for (let k = 0; k < n; k++) q[k] = (k + 0.5) / n;
      return q;
    }
    const idx = new Uint32Array(n);
    for (let k = 0; k < n; k++) idx[k] = k;
    idx.sort((x, y) => members[x]! - members[y]!);
    for (let r = 0; r < n; r++) q[idx[r]!] = (r + 0.5) / n;
    return q;
  },
};

/** `uniform`: the identity warp — equal-spaced quantiles stay equal-spaced. */
const uniform: Density = {
  key: 'uniform',
  warp(qv: number): number {
    return qv;
  },
};

/** `none`: no displacement off the manifold. */
const none: Scatter = {
  key: 'none',
  offset(): void {
    /* adds nothing, by design */
  },
};

/* Guard for feeding f32-rounded uniforms to the probit: a hash like (2³²−1)/2³² rounds to
 * exactly 1.0 in a Float32Array slot, and probit(1) is +Infinity — an infinite coordinate
 * is a silently missing node (src/core/hash.ts). probit(1e-7) ≈ ±5.2σ, far outside what a
 * cloud of any realistic size reaches anyway. */
const P_MIN = 1e-7;
const P_MAX = 1 - 1e-7;
const clampP = (p: number): number => (p < P_MIN ? P_MIN : p > P_MAX ? P_MAX : p);

/** `cloud`: a gaussian ball (dim 3). Each parameter axis becomes an independent normal
 *  deviate via the probit — statistically the `gauss(member, salt)` ball of the
 *  predecessor, but routed through parameter space so it composes: the ordering stratifies
 *  axis 0 (quantiles through the probit give a perfectly even gaussian sample), and a
 *  density can re-weight any axis before the shape sees it.
 *
 *  `params.sigma` is the standard deviation as a fraction of `radius` (default ⅓, so the
 *  ball's ~3σ extent is the group radius). */
const cloud: Shape = {
  key: 'cloud',
  dim: 3,
  place(u, radius, count, out, outOffset, params): void {
    const sigma = radius * (params['sigma'] ?? 1 / 3);
    for (let k = 0; k < count; k++) {
      const b = k * 3;
      const o = outOffset + k * 3;
      out[o] = probitApprox(clampP(u[b]!)) * sigma;
      out[o + 1] = probitApprox(clampP(u[b + 1]!)) * sigma;
      out[o + 2] = probitApprox(clampP(u[b + 2]!)) * sigma;
    }
  },
};

registerShape(cloud);
registerOrdering(byIndex);
registerDensity(uniform);
registerScatter(none);
