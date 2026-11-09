/* The density library (DESIGN.md §3.3 knob 3): distributions over parameter space,
 * applied as inverse-CDF warps of the ordered quantiles. Importing this module registers
 * `gaussian`, `exponential`, `powerLaw` and `mixture`; `uniform` (the identity) is the
 * engine's built-in and is deliberately not duplicated here.
 *
 * The inverse-CDF scheme: the ordering hands each member an equal-spaced quantile q, and
 * `warp(q) = F⁻¹(q)` seats the members so their spacing follows the density F′ — members
 * crowd where the density is high. Ring + gaussian = most members bunched at one point of
 * the ring, thinning toward the antipode (§3.3's canonical example, pinned in the tests).
 *
 * Every warp is a pure function of (q, params) — no RNG, no clock, no state (§10.1) — and
 * returns a value in [0, 1) that SURVIVES the engine's Float32 parameter buffer: outputs
 * are capped at the largest float32 below 1 (see `clampUnit`), because a f64 value like
 * 1 − 2⁻⁵⁰ rounds to exactly 1.0f in the u buffer and 1.0 is outside the domain shapes
 * are promised (src/core/hash.ts records where that class of bug leads).
 */

import { registerDensity } from './registry';
import { probitApprox } from './stat';
import type { Density, Params } from './types';

/* ------------------------------------------------------------------ numeric guards */

/** Largest float32 strictly below 1 — the output cap that keeps a warped parameter inside
 *  [0, 1) even after the engine stores it into its Float32Array u buffer. */
const UNIT_MAX = 1 - 2 ** -24;

/** Clamp a warp result into [0, UNIT_MAX]: pure float fuzz at the ends (a −1e-17 wrap
 *  residue, a bisection midpoint at 1 − 2⁻⁵⁰), never a redistribution of mass. */
function clampUnit(w: number): number {
  return w <= 0 ? 0 : w >= UNIT_MAX ? UNIT_MAX : w;
}

/* Probit input guard, same bounds (and same reasoning) as the engine's cloud shape:
 * probit(0)/probit(1) are ±Infinity by honest contract, and an infinite coordinate is a
 * silently missing node. probit(1e-7) ≈ ±5.2σ — beyond anything a placement reaches. */
const P_MIN = 1e-7;
const P_MAX = 1 - 1e-7;
const clampP = (p: number): number => (p < P_MIN ? P_MIN : p > P_MAX ? P_MAX : p);

/* ------------------------------------------------------------------ gaussian */

/** `gaussian(center, sigma)`: the normal distribution N(center, sigma²) on the unit
 *  parameter axis, WRAPPED onto the circle as `mod 1`.
 *
 *  `warp(q) = (center + sigma · probit(q)) mod 1`. Defaults: `params.center` 0.5,
 *  `params.sigma` 0.15 (≈3σ from the antipode, a visible bunching on a ring).
 *
 *  On the wrap: periodicity is the SHAPE AXIS's property, not this density's — a ring's
 *  angle axis is periodic, a line's is not, and the density cannot know which it feeds.
 *  It therefore always wraps: the lift `center + sigma·probit(q)` is strictly monotone,
 *  and `mod 1` preserves that order on the circle, so on a periodic axis the result is
 *  exactly the wrapped normal. On a NON-periodic axis, keep center ± ~3σ inside [0, 1]
 *  and the wrap never engages; let the tails cross an end and that tail mass re-enters
 *  from the other end (the circular reading), which is the honest alternative to piling
 *  it up against the boundary. */
const gaussian: Density = {
  key: 'gaussian',
  warp(q: number, params: Params): number {
    const center = params['center'] ?? 0.5;
    const sigma = params['sigma'] ?? 0.15;
    if (!(sigma >= 0)) throw new RangeError(`gaussian: sigma must be >= 0, got ${sigma}`);
    const x = center + sigma * probitApprox(clampP(q));
    return clampUnit(x - Math.floor(x)); // mod 1; clampUnit catches (1 − tiny) → 1.0 rounding
  },
};

/* ------------------------------------------------------------------ exponential */

/** `exponential(rate)`: the exponential distribution truncated to the unit interval and
 *  renormalised — CDF `F(x) = (1 − e^(−rate·x)) / (1 − e^(−rate))` on [0, 1], so unlike a
 *  raw exponential no mass is lost off the end of the axis.
 *
 *  `warp(q) = −ln(1 − q·(1 − e^(−rate))) / rate`. Default `params.rate` 3 (the density at
 *  0 is e³ ≈ 20× the density at 1). Positive rate crowds members toward 0 — on a disc's
 *  radial axis, toward the core (the classic exponential galaxy-disc profile §3.3 names);
 *  a negative rate mirrors the crowd toward 1; |rate| below 1e-9 is the uniform limit and
 *  is answered exactly as the identity. */
const exponential: Density = {
  key: 'exponential',
  warp(q: number, params: Params): number {
    const rate = params['rate'] ?? 3;
    if (!Number.isFinite(rate)) throw new RangeError(`exponential: rate must be finite, got ${rate}`);
    if (Math.abs(rate) < 1e-9) return clampUnit(q);
    return clampUnit(-Math.log(1 - q * (1 - Math.exp(-rate))) / rate);
  },
};

/* ------------------------------------------------------------------ powerLaw */

/** `powerLaw(alpha)`: density ∝ x^(alpha−1) on the unit interval — CDF `F(x) = x^alpha`,
 *  so `warp(q) = q^(1/alpha)`.
 *
 *  Default `params.alpha` 0.5. alpha < 1 crowds toward 0 with a density that diverges at
 *  the origin (the heavy-headed "power law" reading — a dense core with a long thinning
 *  tail); alpha = 1 is uniform; alpha > 1 crowds toward 1. alpha must be > 0 (the
 *  distribution is not normalisable otherwise) — anything else throws. */
const powerLaw: Density = {
  key: 'powerLaw',
  warp(q: number, params: Params): number {
    const alpha = params['alpha'] ?? 0.5;
    if (!(alpha > 0)) throw new RangeError(`powerLaw: alpha must be > 0, got ${alpha}`);
    return clampUnit(Math.pow(q, 1 / alpha));
  },
};

/* ------------------------------------------------------------------ mixture */

/* Φ(z), the standard normal CDF, via the Abramowitz–Stegun 7.1.26 erf approximation
 * (|error| < 1.5e-7 — invisible after the f32 parameter buffer). The forward direction of
 * what probitApprox inverts; lives here because only the mixture needs it. */
const ERF_A1 = 0.254829592;
const ERF_A2 = -0.284496736;
const ERF_A3 = 1.421413741;
const ERF_A4 = -1.453152027;
const ERF_A5 = 1.061405429;
const ERF_P = 0.3275911;
const SQRT1_2 = Math.SQRT1_2;

function normalCdf(z: number): number {
  const x = Math.abs(z) * SQRT1_2;
  const t = 1 / (1 + ERF_P * x);
  const erf =
    1 - ((((ERF_A5 * t + ERF_A4) * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t * Math.exp(-x * x);
  return z < 0 ? 0.5 * (1 - erf) : 0.5 * (1 + erf);
}

/* Component scratch, module-level and fixed-size: `warp` runs once per member and must
 * not allocate per call (§10.2). Single-threaded synchronous placement makes this safe. */
const MIX_MAX = 4;
const mixM = new Float64Array(MIX_MAX);
const mixS = new Float64Array(MIX_MAX);
const mixW = new Float64Array(MIX_MAX);
const KEY_M = ['m0', 'm1', 'm2', 'm3'] as const;
const KEY_S = ['s0', 's1', 's2', 's3'] as const;
const KEY_W = ['w0', 'w1', 'w2', 'w3'] as const;

/** Unnormalised mixture CDF at x over the `nc` components currently in the scratch. */
function mixCdf(x: number, nc: number): number {
  let s = 0;
  for (let c = 0; c < nc; c++) s += mixW[c]! * normalCdf((x - mixM[c]!) / mixS[c]!);
  return s;
}

/** `mixture`: up to 4 gaussian components, numeric-params-only (§7 — every param is a
 *  schema row): component i exists when `params.m<i>` (its mean) is present, with
 *  `params.s<i>` its sigma (default 0.1, must be > 0) and `params.w<i>` its weight
 *  (default 1, must be > 0; weights are relative, not required to sum to 1). At least one
 *  component is required — a mixture of nothing is a spec error and throws.
 *
 *  The mixture CDF has no closed-form inverse, so `warp` is a 50-step bisection of
 *  `F(x) = F(0) + q·(F(1) − F(0))` on [0, 1] — deterministic to below f64 epsilon, ~200
 *  normalCdf evaluations per member, run once per compile (§4: never per frame). The
 *  domain restriction to [0, 1] renormalises away any tail mass outside the axis (NOT
 *  wrapped — unlike `gaussian`, a multi-bump profile is usually composed on non-periodic
 *  axes; put means safely inside [0, 1] so each bump keeps its mass). */
const mixture: Density = {
  key: 'mixture',
  warp(q: number, params: Params): number {
    let nc = 0;
    for (let i = 0; i < MIX_MAX; i++) {
      const m = params[KEY_M[i]!];
      if (m === undefined) continue;
      const s = params[KEY_S[i]!] ?? 0.1;
      const w = params[KEY_W[i]!] ?? 1;
      if (!(s > 0)) throw new RangeError(`mixture: s${i} must be > 0, got ${s}`);
      if (!(w > 0)) throw new RangeError(`mixture: w${i} must be > 0, got ${w}`);
      mixM[nc] = m;
      mixS[nc] = s;
      mixW[nc] = w;
      nc++;
    }
    if (nc === 0) {
      throw new Error('mixture: no components — provide m0 (and optionally s0/w0, m1…m3)');
    }

    const f0 = mixCdf(0, nc);
    const target = f0 + q * (mixCdf(1, nc) - f0);
    let lo = 0;
    let hi = 1;
    for (let it = 0; it < 50; it++) {
      const mid = 0.5 * (lo + hi);
      if (mixCdf(mid, nc) < target) lo = mid;
      else hi = mid;
    }
    return clampUnit(0.5 * (lo + hi));
  },
};

registerDensity(gaussian);
registerDensity(exponential);
registerDensity(powerLaw);
registerDensity(mixture);

export { gaussian, exponential, powerLaw, mixture };
