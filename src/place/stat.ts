/* Deterministic statistics helpers for placement.
 *
 * The density knob (DESIGN.md §3.3) is an inverse-CDF warp, and the gaussian family needs
 * the inverse NORMAL CDF — which has no closed form. This is Peter Acklam's rational
 * approximation: three branches (lower tail / central / upper tail), relative error below
 * 1.15e-9 over the whole open interval, which is far inside Float32's ~1e-7 resolution —
 * the error is invisible by the time a coordinate lands in a GPU buffer.
 *
 * Pure functions of their arguments, like everything that produces geometry (§10.1): no
 * RNG, no clock, no state.
 */

/* Acklam's coefficients, verbatim. Central branch: a rational function in r = q² times q,
 * with q = p - 0.5. Tail branches: a rational function in q = sqrt(-2 ln p) (lower), and
 * the mirror image for the upper tail. */
const A0 = -3.969683028665376e1;
const A1 = 2.209460984245205e2;
const A2 = -2.759285104469687e2;
const A3 = 1.38357751867269e2;
const A4 = -3.066479806614716e1;
const A5 = 2.506628277459239;
const B0 = -5.447609879822406e1;
const B1 = 1.615858368580409e2;
const B2 = -1.556989798598866e2;
const B3 = 6.680131188771972e1;
const B4 = -1.328068155288572e1;
const C0 = -7.784894002430293e-3;
const C1 = -3.223964580411365e-1;
const C2 = -2.400758277161838;
const C3 = -2.549732539343734;
const C4 = 4.374664141464968;
const C5 = 2.938163982698783;
const D0 = 7.784695709041462e-3;
const D1 = 3.224671290700398e-1;
const D2 = 2.445134137142996;
const D3 = 3.754408661907416;

/** Where the central rational approximation hands over to the tail ones. */
const P_LOW = 0.02425;
const P_HIGH = 1 - P_LOW;

/**
 * The inverse of the standard normal CDF (the probit function), Acklam's approximation:
 * `probitApprox(Φ(x)) ≈ x` to better than 1.15e-9 relative error.
 *
 * Domain is the open interval (0, 1). The closures answer honestly rather than NaN-ing:
 * `p <= 0` → `-Infinity`, `p >= 1` → `+Infinity`, `NaN` → `NaN`. Callers feeding hashed
 * uniforms through a Float32Array must clamp first — a value like (2³²−1)/2³² rounds to
 * exactly 1.0 in f32, and an infinite coordinate is a silently missing node and then a NaN
 * camera trying to frame it (see src/core/hash.ts for that war story).
 */
export function probitApprox(p: number): number {
  if (Number.isNaN(p)) return NaN;
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p < P_LOW) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((C0 * q + C1) * q + C2) * q + C3) * q + C4) * q + C5) /
      ((((D0 * q + D1) * q + D2) * q + D3) * q + 1)
    );
  }
  if (p > P_HIGH) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(((((C0 * q + C1) * q + C2) * q + C3) * q + C4) * q + C5) /
      ((((D0 * q + D1) * q + D2) * q + D3) * q + 1)
    );
  }
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((A0 * r + A1) * r + A2) * r + A3) * r + A4) * r + A5) * q) /
    (((((B0 * r + B1) * r + B2) * r + B3) * r + B4) * r + 1)
  );
}
