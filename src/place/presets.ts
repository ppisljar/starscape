/* Placement presets (DESIGN.md §3.3): named looks — "spiral galaxy", "elliptical" — as
 * PURE DATA bundles of the four knobs. A preset is a `PlacementSpec` and nothing else: no
 * code, no closures, no knob of its own. `hierarchy: [{ place: preset('spiralGalaxy') }]`
 * hands the engine the same kind of object a hand-written `place:` would.
 *
 * The registry mirrors ./registry's conventions: lookups throw naming the known keys,
 * re-registering overwrites (apps may replace a built-in look). What it adds: every
 * registered spec is FROZEN (spec and params both) — presets are shared data, read by
 * every group of every scene that names them, so a caller mutating `params` would be
 * spooky action at a distance across scenes. Customizing a preset means spreading it into
 * a fresh spec, not editing the registered one.
 *
 * The numbers are ported from the predecessor's galaxy morphology (vault-site-3d
 * arrange.js, `galaxyMorph` / `DEFAULT_MORPH` / `galaxySlot`): arms 2–4, swirl 1.5–4.5,
 * bulge 0.10–0.36, bar 0.26–0.52, elliptical concentration r = R·t^0.62 with squash
 * 0.42–0.78. Each preset pins that morph's DEFAULT; per-group variation (the predecessor
 * hashed these per cluster so no two galaxies matched) is the hierarchy compiler's story,
 * not the preset's — a preset is one look, reproducibly.
 */

import type { PlacementSpec } from './types';

const presets = new Map<string, PlacementSpec>();

/** Register a placement preset under `name` (overwrites an existing registration). The
 *  spec — params included — is frozen here and shared from then on; callers keep no right
 *  to mutate what they passed in. */
export function registerPreset(name: string, spec: PlacementSpec): void {
  if (spec.params !== undefined) Object.freeze(spec.params);
  presets.set(name, Object.freeze(spec));
}

/** Resolve a preset name to its (frozen) `PlacementSpec`; throws naming the known presets
 *  when absent — a misspelled look fails at spec-compile time with an actionable message,
 *  exactly like a misspelled knob key (./registry). */
export function getPreset(name: string): PlacementSpec {
  const hit = presets.get(name);
  if (hit === undefined) {
    const known = [...presets.keys()].sort().join(', ');
    throw new Error(`unknown preset "${name}" — known: ${known || '(none)'}`);
  }
  return hit;
}

/** The registered preset names, sorted — for panels and error reporting. */
export function presetKeys(): string[] {
  return [...presets.keys()].sort();
}

/** DESIGN §3.4 spelling: `place: preset('spiralGalaxy')`. Same lookup as `getPreset`. */
export const preset = getPreset;

/* ================================================================== the built-in looks */

/* A spiral disc: members along winding arms, crowded toward the core by the exponential
 * radial warp, with the disc thickness thinning toward the rim. spiralArm DECLARES its
 * arm fold, so the engine applies the exponential within each arm: the arms stay equally
 * populated (the predecessor's round-robin `k % arms`) and every arm runs its full radial
 * length with linear density ∝ e^(−rate·r/R) — `rate` 3 puts the scale length near
 * radius/3. `radialPower: 1` hands the warped parameter straight through as the radius
 * fraction (the disc/ball escape hatch) — the warp owns the whole radial profile; the √t
 * area correction would compose into it and turn the asked-for exponential into a
 * gaussian. `byHash` decorrelates arm position from member id — the predecessor jittered
 * each slot's angle for the same reason (galaxySlot's ±0.55 rad wobble); here the
 * sanctioned channel is a deterministic shuffle of the ordering. `arms`/`swirl` are the
 * predecessor's DEFAULT_MORPH (3 arms, swirl 2.6); its `bulge` 0.10 is ported as `h`, the
 * discFalloff central half-thickness (galaxySlot: thick = R·bulge·(1−t) + 0.02R). */
registerPreset('spiralGalaxy', {
  shape: 'spiralArm',
  ordering: 'byHash',
  density: 'exponential',
  densityAxis: 0,
  scatter: 'discFalloff',
  params: { arms: 3, swirl: 2.6, radialPower: 1, rate: 3, h: 0.1 },
});

/* The barred variant: the same bundle, plus a central bar (`bar` = bar half-length as a
 * fraction of R, the predecessor's DEFAULT_MORPH 0.3, hashed range 0.26–0.52) from whose
 * ENDS the arms launch — which is the whole visual difference between a barred spiral and
 * an ordinary one (arrange.js galaxySlot, the `barred` branches; spiralArm's `bar` is
 * that port). Two arms, because a bar has two ends: the predecessor's barred arms
 * alternated `k % 2` between them; here each arm owns one end of the bar. */
registerPreset('barredGalaxy', {
  shape: 'spiralArm',
  ordering: 'byHash',
  density: 'exponential',
  densityAxis: 0,
  scatter: 'discFalloff',
  params: { arms: 2, swirl: 2.6, bar: 0.3, radialPower: 1, rate: 3, h: 0.1 },
});

/* A spheroid with a de Vaucouleurs-ish concentration: most of the light near the middle,
 * a long faint tail outward, no arms because there is no disc (arrange.js `elliptical`).
 * The predecessor's radial law was r = R·t^0.62 — a power-law inverse CDF, so here it IS
 * a powerLaw density warping the ball's RADIAL axis (axis 2; axis 0 is the azimuth) with
 * `radialPower: 1` so the warp output is the radius fraction itself. powerLaw's warp is
 * q^(1/alpha), so alpha = 1/0.62 reproduces r = R·q^0.62 exactly — the law lives in the
 * inverse CDF, not in alpha verbatim (and the default cbrt would have divided the
 * exponent by 3 on top; both compositions are why the escape hatch exists). `aspectY` is
 * the squash that makes it read as an ellipse rather than a ball (the predecessor's
 * `ell`: DEFAULT_MORPH 0.6, hashed E3–E6 range 0.42–0.78). No scatter: an elliptical is
 * dynamically relaxed, structure would lie. */
registerPreset('elliptical', {
  shape: 'ball',
  density: 'powerLaw',
  densityAxis: 2,
  params: { alpha: 1 / 0.62, radialPower: 1, aspectY: 0.6 },
});

/* A globular cluster: a dense, nearly round swarm — half-gaussian radial profile (σ 0.35
 * of the radius) crowding members hard into the core and thinning smoothly outward, no
 * squash, no arms, no noise. The half-gaussian is `mixture` with its single component
 * centred AT the core: mixture renormalises over [0, 1] instead of wrapping, so the
 * below-zero half of the bell folds away and warp(q) climbs 0 → 1 with gaussian falloff
 * (≈85% of members inside radius/2) — `gaussian(center: 0)` would have WRAPPED half the
 * mass to the rim, the circular reading a radial axis must not get. On the radial axis
 * (2) with `radialPower: 1`, as `elliptical`. The roundest and smoothest of the looks,
 * on purpose: next to a spiral it reads as a different KIND of object, which is what
 * morphology is for. */
registerPreset('globularCluster', {
  shape: 'ball',
  density: 'mixture',
  densityAxis: 2,
  params: { m0: 0, s0: 0.35, radialPower: 1 },
});

/* A disturbed, clumpy, lopsided cloud (arrange.js `irregular`: offset knots, deliberate
 * lopsidedness). Here the clumps come from the noiseField scatter at low frequency and
 * high amplitude — displacement is a function of position, so neighbours herd together
 * into knots and the voids between them are real, exactly the clumpy asymmetry the
 * predecessor built by hand with 3 knots and a `lop` shift. scale 2 ≈ a couple of cells
 * across the group = a few big clumps; amp 0.5 is strong enough to break the cloud's
 * gaussian symmetry. */
registerPreset('irregular', {
  shape: 'cloud',
  scatter: 'noiseField',
  params: { scale: 2, amp: 0.5 },
});
