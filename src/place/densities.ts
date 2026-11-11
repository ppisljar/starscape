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
  params: [
    { key: 'center', label: 'center', default: 0.5, min: 0, max: 1, step: 0.02 },
    { key: 'sigma', label: 'sigma', default: 0.15, min: 0.01, max: 0.5, step: 0.01 },
  ],
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
  params: [{ key: 'rate', label: 'rate', default: 3, min: -8, max: 8, step: 0.25 }],
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
  params: [{ key: 'alpha', label: 'alpha', default: 0.5, min: 0.1, max: 4, step: 0.05 }],
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
  /* Up to 4 gaussian components. Component i EXISTS only when `m<i>` is present (presence is
   * the toggle — there is no "off" numeric value), with `s<i>` (sigma, default 0.1) and
   * `w<i>` (weight, default 1) read only then. So `m<i>` defaults are REPRESENTATIVE means a
   * panel seeds when the user turns a component ON, not omit-at-default fallbacks (omitting
   * m0 throws — a mixture needs at least one component). `s<i>`/`w<i>` ARE plain `?? default`
   * params (0.1 / 1), read when their component is present. */
  params: [
    { key: 'm0', label: 'm0·mean', default: 0, min: 0, max: 1, step: 0.02 },
    { key: 's0', label: 's0·sigma', default: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    { key: 'w0', label: 'w0·wt', default: 1, min: 0.05, max: 4, step: 0.05 },
    { key: 'm1', label: 'm1·mean', default: 0.35, min: 0, max: 1, step: 0.02 },
    { key: 's1', label: 's1·sigma', default: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    { key: 'w1', label: 'w1·wt', default: 1, min: 0.05, max: 4, step: 0.05 },
    { key: 'm2', label: 'm2·mean', default: 0.65, min: 0, max: 1, step: 0.02 },
    { key: 's2', label: 's2·sigma', default: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    { key: 'w2', label: 'w2·wt', default: 1, min: 0.05, max: 4, step: 0.05 },
    { key: 'm3', label: 'm3·mean', default: 1, min: 0, max: 1, step: 0.02 },
    { key: 's3', label: 's3·sigma', default: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    { key: 'w3', label: 'w3·wt', default: 1, min: 0.05, max: 4, step: 0.05 },
  ],
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

/* ------------------------------------------------------------------ beta
 * The Beta(a, b) inverse CDF, by numeric inversion of the regularized incomplete beta Iₓ(a,b).
 * Iₓ has no elementary closed form, so this mirrors `mixture`: a monotone CDF evaluated by a
 * stable series (Lentz's continued fraction, the Numerical Recipes `betacf`) and inverted by
 * bisection — deterministic, run once per compile, not per frame (§4). Beta(1,1) is uniform,
 * Beta(2,2) a bell, a,b < 1 a U-shape, a ≠ b a skew; it subsumes powerLaw's x^(α−1) as the
 * b = 1 edge, but earns its place on the two-sided shapes those cannot reach.
 */

/* ln Γ(x) — Lanczos g = 7, n = 9, the standard coefficient set (|rel err| < 1e-13 for x > 0,
 * far inside the f32 parameter buffer). Pure: the incomplete beta needs the Beta normaliser. */
const LG_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

function lgamma(x: number): number {
  let g = LG_C[0]!;
  const xm1 = x - 1;
  for (let i = 1; i < 9; i++) g += LG_C[i]! / (xm1 + i);
  const tmp = xm1 + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (xm1 + 0.5) * Math.log(tmp) - tmp + Math.log(g);
}

/* Lentz's continued fraction for Iₓ(a,b), Numerical Recipes §6.4 `betacf`. Converges for
 * x < (a+1)/(a+b+2); the caller swaps to the complement past that so convergence is always fast. */
const BETACF_ITMAX = 200;
const BETACF_EPS = 3e-14;
const BETACF_FPMIN = 1e-300;

function betacf(a: number, b: number, x: number): number {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < BETACF_FPMIN) d = BETACF_FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= BETACF_ITMAX; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < BETACF_FPMIN) d = BETACF_FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < BETACF_FPMIN) c = BETACF_FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < BETACF_FPMIN) d = BETACF_FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < BETACF_FPMIN) c = BETACF_FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < BETACF_EPS) break;
  }
  return h;
}

/** Regularized incomplete beta Iₓ(a,b) ∈ [0,1] — the Beta(a,b) CDF, monotone in x. */
function betai(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a + b) - lgamma(a) - lgamma(b);
  const bt = Math.exp(lbeta + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** `beta(a, b)`: the Beta(a, b) distribution on [0, 1), inverse CDF by bisection on Iₓ(a,b).
 *  Defaults `a` 2, `b` 2 (a symmetric bell). a = b = 1 is uniform; a, b < 1 is U-shaped
 *  (mass at both ends); a ≠ b skews (a < b toward 0, a > b toward 1). Both must be > 0 — the
 *  distribution is not defined otherwise — anything else throws. */
const beta: Density = {
  key: 'beta',
  params: [
    { key: 'a', label: 'a', default: 2, min: 0.3, max: 8, step: 0.1 },
    { key: 'b', label: 'b', default: 2, min: 0.3, max: 8, step: 0.1 },
  ],
  warp(q: number, params: Params): number {
    const a = params['a'] ?? 2;
    const b = params['b'] ?? 2;
    if (!(a > 0)) throw new RangeError(`beta: a must be > 0, got ${a}`);
    if (!(b > 0)) throw new RangeError(`beta: b must be > 0, got ${b}`);
    // bisect Iₓ(a,b) = q on [0, 1]; 60 steps ⇒ interval < 2⁻⁶⁰, below the f32 buffer's step
    let lo = 0;
    let hi = 1;
    for (let it = 0; it < 60; it++) {
      const mid = 0.5 * (lo + hi);
      if (betai(a, b, mid) < q) lo = mid;
      else hi = mid;
    }
    return clampUnit(0.5 * (lo + hi));
  },
};

/* ------------------------------------------------------------------ edges (bathtub)
 * The OPPOSITE of gaussian: a U-shaped "bathtub" that crowds members toward BOTH ends of the
 * axis and empties the middle. Closed form — a signed power of the centred quantile, so no
 * inversion is needed. On a ring it gathers members at two antipodal poles; on a line it piles
 * them at both tips. */

/** `edges(strength)`: a symmetric bathtub inverse CDF, `warp(q) = ½(1 + sign(v)·|v|^(1/strength))`
 *  with v = 2q − 1. Default `strength` 2 (density ∝ strength at each end, 0 at the centre).
 *  strength = 1 is uniform; strength > 1 deepens the bathtub; strength must be > 0 (throws
 *  otherwise). The mirror image of a centred gaussian, which empties the ends and fills the
 *  middle. */
const edges: Density = {
  key: 'edges',
  params: [{ key: 'strength', label: 'strength', default: 2, min: 1, max: 6, step: 0.25 }],
  warp(q: number, params: Params): number {
    const strength = params['strength'] ?? 2;
    if (!(strength > 0)) throw new RangeError(`edges: strength must be > 0, got ${strength}`);
    const v = 2 * q - 1;
    const w = Math.sign(v) * Math.pow(Math.abs(v), 1 / strength);
    return clampUnit(0.5 * (1 + w));
  },
};

/* ------------------------------------------------------------------ triangular
 * A linear ramp to a single peak at `peak`, zero density at both ends — the simplest
 * asymmetric density, closed-form inverse CDF. peak toward 1 crowds the far end (a ramp up),
 * toward 0 the near end (a ramp down), at 0.5 a symmetric tent. */

/** `triangular(peak)`: the triangular distribution on [0, 1] with mode `peak` (default 0.5).
 *  CDF F(peak) = peak, so `warp(q) = √(q·peak)` for q ≤ peak and `1 − √((1−q)(1−peak))` above —
 *  continuous at the peak. `peak` is clamped to [0, 1] (peak = 1 recovers √q, the powerLaw(2)
 *  ramp; peak = 0 its mirror). */
const triangular: Density = {
  key: 'triangular',
  params: [{ key: 'peak', label: 'peak', default: 0.5, min: 0, max: 1, step: 0.05 }],
  warp(q: number, params: Params): number {
    const peak = Math.min(1, Math.max(0, params['peak'] ?? 0.5));
    const w = q < peak ? Math.sqrt(q * peak) : 1 - Math.sqrt((1 - q) * (1 - peak));
    return clampUnit(w);
  },
};

registerDensity(gaussian);
registerDensity(exponential);
registerDensity(powerLaw);
registerDensity(mixture);
registerDensity(beta);
registerDensity(edges);
registerDensity(triangular);

export { gaussian, exponential, powerLaw, mixture, beta, edges, triangular };
