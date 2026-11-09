/* The scatter library (DESIGN.md §3.3 knob 4): displacement OFF the manifold — the flesh on
 * the shape's bone. A ring becomes a tube, a shell a fuzzy shell, a disc gains thickness
 * that thins toward the rim. (`none` is the engine's built-in; this module adds the rest.)
 *
 * THE READ-AFTER-SHAPE CONTRACT. The engine (./engine `placeGroup`) runs scatters strictly
 * AFTER `shape.place` has written every member's position, so a scatter may READ the
 * member's already-placed coordinates at `out[o..o+2]` before adding its displacement.
 * `discFalloff` and `noiseField` depend on that: the first thins with the radial distance
 * of the placed point, the second samples a spatial noise field AT the placed point. Both
 * remain read-modify-write — they add to the three slots, never overwrite them — so scatter
 * still composes with whatever the shape wrote (./types `Scatter`). Anyone calling a
 * scatter outside the engine must honour the same order: position first, scatter second.
 *
 * Determinism (§10.1): every draw is `hash`/`gauss(member, salt + C)` or a pure function of
 * the already-deterministic position — no RNG, no clock — so the same input produces
 * byte-identical buffers forever. The constants C below stay clear of `salt + 100…102`,
 * which the engine reserves for parameter axes (./types `PlaceInput.salt`).
 */

import { gauss, hash } from '../core/hash';
import { registerScatter } from './registry';
import type { Scatter } from './types';

/* Per-scatter salt offsets. Spread out so the streams cannot collide with each other, with
 * the engine's axis salts (+101, +102), or with `gauss`'s internal second draw (which sits
 * a huge constant away — src/core/hash.ts). */
const GAUSS_SALT = 11; // gaussian uses +11, +12, +13 (one per axis)
const DISC_SALT = 17; // discFalloff uses +17
const NOISE_SALT = 23; // noiseField uses +23, +24, +25 (one per displacement component)

/* ------------------------------------------------------------------ gaussian
 * Isotropic normal displacement: the predecessor's jitter ball (vault-site-3d arrange.js
 * used `rnd`-driven jitter around every slot), routed through `gauss` so the magnitude has
 * a real unit — a standard deviation.
 *
 * `params.thickness` is that standard deviation as a fraction of the group radius (default
 * 0.1): each axis adds `gauss(member, salt + axis) * thickness * radius`, three independent
 * streams, so the displacement is an isotropic 3D gaussian with σ = thickness·radius. */
const gaussianScatter: Scatter = {
  key: 'gaussian',
  offset(_k, member, salt, radius, out, o, params): void {
    const t = (params['thickness'] ?? 0.1) * radius;
    if (t === 0) return;
    out[o]! += gauss(member, salt + GAUSS_SALT) * t;
    out[o + 1]! += gauss(member, salt + GAUSS_SALT + 1) * t;
    out[o + 2]! += gauss(member, salt + GAUSS_SALT + 2) * t;
  },
};

/* ------------------------------------------------------------------ discFalloff
 * Galaxy-disc thickness that thins with radius: the look-defining line of the predecessor's
 * spiral (arrange.js galaxySlot):
 *
 *     thick = R * bulge * (1 - t) + R * 0.02        // t = quantile; r = R√t, so 1-t = 1-(r/R)²
 *
 * Ported against the placed point instead of the quantile — a scatter never sees parameter
 * space, but under the read-after-shape contract it can read where the member LANDED, and
 * `1 - t` is exactly `1 - (ρ/R)²` of that point. So: read `out[]` first, measure the
 * in-plane (xz) radial distance ρ, then add a uniform vertical (y) displacement inside
 *
 *     thick = radius * ( h * (1 - (ρ/radius)²) + 0.02 )
 *
 * `params.h` is the central half-thickness as a fraction of radius (default 0.12, the
 * predecessor's bulge range 0.10–0.36); the 0.02·radius floor keeps the rim a thin sheet
 * rather than a knife edge, exactly as in the predecessor. Displacement is y-only: the
 * vertical axis of a disc lying in the xz-plane, matching the shell/disc convention
 * elsewhere in this library (y is up). */
const DISC_FLOOR = 0.02;

const discFalloff: Scatter = {
  key: 'discFalloff',
  offset(_k, member, salt, radius, out, o, params): void {
    const h = params['h'] ?? 0.12;
    // READ the already-placed point (shape ran first — see the module header contract).
    const x = out[o]!;
    const z = out[o + 2]!;
    const rhoN = radius > 0 ? Math.min(1, Math.hypot(x, z) / radius) : 0;
    const thick = radius * (h * (1 - rhoN * rhoN) + DISC_FLOOR);
    out[o + 1]! += (hash(member, salt + DISC_SALT) * 2 - 1) * thick;
  },
};

/* ------------------------------------------------------------------ noiseField
 * Deterministic 3D VALUE NOISE: hash the quantised lattice coordinates (no permutation
 * tables, hence nothing seeded from Math.random — §10.1), smoothstep-interpolate the eight
 * cell corners. Classic spatial-hash fold of the three lattice ints into one 32-bit key,
 * then the shared `hash` turns key + salt into the corner value.
 */

/** Fold three lattice integers into one 32-bit key and hash it with `salt`. The three
 *  primes are the classic spatial-hash set; `| 0` keeps the key an int32 (hash applies
 *  ToInt32 anyway — this just makes the domain explicit). */
function latticeHash(ix: number, iy: number, iz: number, salt: number): number {
  const key = (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) | 0;
  return hash(key, salt);
}

/** Hermite smoothstep — C¹ across cell boundaries, which is what makes the field read as
 *  billows rather than a Minecraft block-world. */
const smooth = (f: number): number => f * f * (3 - 2 * f);

/**
 * 3D value noise in [0, 1): trilinear smoothstep interpolation of hashed lattice corners.
 * Pure function of (x, y, z, salt); coordinates are in LATTICE units (one cell = 1.0).
 * Continuous everywhere, equal to the corner hash exactly at integer coordinates.
 *
 * Exported for tests (continuity is pinned against this directly) and for anyone wanting
 * the raw field — e.g. a custom shape modulating its own geometry deterministically.
 */
export function valueNoise3(x: number, y: number, z: number, salt: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const fx = smooth(x - x0);
  const fy = smooth(y - y0);
  const fz = smooth(z - z0);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const z1 = z0 + 1;
  // Eight corners, three lerps — written out, no arrays, no per-call allocation.
  const c000 = latticeHash(x0, y0, z0, salt);
  const c100 = latticeHash(x1, y0, z0, salt);
  const c010 = latticeHash(x0, y1, z0, salt);
  const c110 = latticeHash(x1, y1, z0, salt);
  const c001 = latticeHash(x0, y0, z1, salt);
  const c101 = latticeHash(x1, y0, z1, salt);
  const c011 = latticeHash(x0, y1, z1, salt);
  const c111 = latticeHash(x1, y1, z1, salt);
  const x00 = c000 + (c100 - c000) * fx;
  const x10 = c010 + (c110 - c010) * fx;
  const x01 = c001 + (c101 - c001) * fx;
  const x11 = c011 + (c111 - c011) * fx;
  const y0v = x00 + (x10 - x00) * fy;
  const y1v = x01 + (x11 - x01) * fy;
  return y0v + (y1v - y0v) * fz;
}

/* The scatter: sample the field at the member's already-placed position (read-after-shape,
 * see the module header) and displace by one noise channel per axis. A FIELD, deliberately:
 * the displacement is a function of WHERE the member sits, not of who it is, so neighbours
 * move together and the group develops coherent clumps and voids — the irregular-galaxy
 * look the predecessor built from offset knots and lopsidedness (arrange.js `irregular`),
 * produced here by low-frequency noise instead so it composes with any shape.
 *
 * `params.scale`: lattice cells across one group radius (default 3) — lower = fewer, bigger
 * clumps. `params.amp`: maximum displacement per axis as a fraction of radius (default
 * 0.25); each axis adds `(noise*2 - 1) * amp * radius`, so |Δ| ≤ amp·radius per axis. */
const noiseField: Scatter = {
  key: 'noiseField',
  offset(_k, _member, salt, radius, out, o, params): void {
    const amp = (params['amp'] ?? 0.25) * radius;
    if (amp === 0 || radius <= 0) return;
    const f = (params['scale'] ?? 3) / radius;
    const x = out[o]! * f;
    const y = out[o + 1]! * f;
    const z = out[o + 2]! * f;
    out[o]! += (valueNoise3(x, y, z, salt + NOISE_SALT) * 2 - 1) * amp;
    out[o + 1]! += (valueNoise3(x, y, z, salt + NOISE_SALT + 1) * 2 - 1) * amp;
    out[o + 2]! += (valueNoise3(x, y, z, salt + NOISE_SALT + 2) * 2 - 1) * amp;
  },
};

registerScatter(gaussianScatter);
registerScatter(discFalloff);
registerScatter(noiseField);
