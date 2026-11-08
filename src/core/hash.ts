/* Deterministic per-index randomness. Everything that scatters geometry draws from here —
 * never Math.random(), never a clock — so the same input produces byte-identical buffers
 * forever (DESIGN.md §10.1).
 *
 * The `>>> 0`s are load-bearing. `^` in JS yields a SIGNED int32, so without the final
 * coercion about half of all hashes come back negative. That is invisible until it reaches
 * Math.sqrt or Math.acos, which answer NaN — and a NaN position is not a visible mistake, it
 * is a silently missing node and then a NaN camera trying to frame it. The predecessor's
 * cosmos view lost 796 of 1,600 nodes to exactly this; the fix is cheap and the bug is not.
 */

/** Uniform in [0, 1), a pure function of (i, salt). Integer-mix finalizer over a
 *  golden-ratio-salted product; different salts give independent streams. */
export function hash(i: number, salt: number): number {
  let x = Math.imul(i ^ Math.imul(salt, 0x9e3779b1), 2654435761) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 2246822519) >>> 0;
  x ^= x >>> 13;
  return (x >>> 0) / 4294967296;
}

/** Standard normal (mean 0, variance 1), a pure function of (i, salt): Box–Muller over two
 *  decorrelated hash draws. `1 - hash(...)` maps [0,1) to (0,1] so the log never sees zero. */
export function gauss(i: number, salt: number): number {
  const u1 = 1 - hash(i, salt);
  const u2 = hash(i, (salt + 0x85ebca6b) | 0);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
