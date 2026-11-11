/* Placement contracts (DESIGN.md §3.3): four orthogonal knobs — shape, ordering, density,
 * scatter — composed by ONE engine (`placeGroup` in ./engine):
 *
 *     param = density.warp( ordering.assign(member) )    // where along the shape
 *     pos   = shape.at(param, radius) + scatter.offset(member)
 *
 * Every other stage-5 module codes against these interfaces, so they are deliberately exact
 * and boring. All of it obeys the house rules (§10): no Math.random / Date.now anywhere —
 * per-member randomness is `hash(member, salt)` from src/core/hash — no per-member JS
 * objects, and the same input always produces byte-identical output.
 */

import type { Graph } from '../core/types';

/** Numeric knob parameters, straight out of `PlacementSpec.params`. Numbers only — a param
 *  is a schema row (§7), and anything stringier belongs in the spec itself. Implementations
 *  read what they know (`params['sigma'] ?? defaultSigma`) and ignore the rest, so one
 *  params bag can feed all four knobs of a preset without cross-talk. */
export type Params = Record<string, number>;

/** Self-describing metadata for ONE numeric param a knob reads (DESIGN §7 — a param is a
 *  schema row). The knob declares its `params: ParamDesc[]` so a panel can enumerate and
 *  render a control per param WITHOUT hardcoding the knob's internals — the registry's
 *  `shapeParams`/`densityParams`/`scatterParams` getters hand these back by key.
 *
 *  THE DEFAULT CONTRACT: `default` MUST equal the knob's own inline `params[key] ?? default`
 *  fallback, so a param left at its declared default produces byte-identical output to
 *  omitting it entirely — the panel can seed every control at `default` and the untouched
 *  scene is unchanged. A few knobs read a param whose fallback is not a plain constant
 *  (ball's `radialPower` defaults to a `cbrt`, dendrite's `tiers` to a count-derived auto,
 *  mixture's `m<i>` are presence flags); those declare a REPRESENTATIVE default and the
 *  caller omits a param equal to its default rather than sending it (each knob's doc says
 *  so). */
export interface ParamDesc {
  /** The `params` key the knob reads (e.g. `'arms'`, `'sigma'`, `'tube'`). */
  key: string;
  /** Short human label for a panel (often the same as `key`). */
  label: string;
  /** The value the knob uses when the param is absent — see THE DEFAULT CONTRACT above. */
  default: number;
  /** Suggested slider/number minimum (advisory — the knob still validates its own domain). */
  min: number;
  /** Suggested slider/number maximum (advisory). */
  max: number;
  /** Suggested slider/number step (advisory). */
  step: number;
}

/** One group's placement job: everything the engine (and the knobs it drives) may know about
 *  the group. The engine writes **LOCAL** positions — the group's centre is the origin; the
 *  hierarchy compiler (§3.4) adds the parent transform. Nothing in here is retained past the
 *  `placeGroup` call. */
export interface PlaceInput {
  /** The group's members: node indices (or child-group pivot ids at interior hierarchy
   *  levels), in `[0, count)` of this buffer. The buffer may be longer than `count` (a
   *  reused slab); only the first `count` entries are meaningful. */
  members: Uint32Array;
  /** How many members to place. May be 0 — the engine then validates the spec and returns
   *  without touching `out`. */
  count: number;
  /** The compiled graph, for orderings that read columns/facets/CSR (`byColumn`,
   *  `byAdjacency`, …). `null` at levels where no graph applies (e.g. placing synthetic
   *  child-group centres); an ordering that needs it must throw a descriptive error on
   *  `null`, never guess. */
  graph: Graph | null;
  /** Deterministic stream id for this group. Distinct per group so sibling groups
   *  decorrelate; every draw inside is `hash(member, salt + k)` — never an RNG (§10.1).
   *  The engine reserves `salt + 100 + axis` for parameter axes ≥ 1; scatters receive the
   *  base `salt` and should offset it by their own constants. */
  salt: number;
  /** The group's radius in local units (the hierarchy compiler's `spacing × count^⅓`,
   *  floored — §3.4). Shapes size themselves from it; scatters scale displacement by it. */
  radius: number;
  /** Destination: `count * 3` floats written at `outOffset` as x,y,z per member, in member
   *  order (member `k` lands at `outOffset + 3k`). LOCAL coordinates, group centre at the
   *  origin. */
  out: Float32Array;
  /** First index in `out` this group owns. */
  outOffset: number;
}

/** A parametric manifold: parameter space U → local 3D positions.
 *
 *  A shape is a pure function of (u, radius, params) and NOTHING else — no member ids, no
 *  salt, no graph. That restriction is what makes the four knobs orthogonal: any per-member
 *  variation reaches a shape only through its `u` coordinates (which ordering + density
 *  control), or is added afterwards by scatter. A shape wanting "gaussian spread" converts
 *  its uniform `u` axes through `probitApprox` (./stat) rather than hashing members itself. */
export interface Shape {
  /** Registry key, e.g. `'ring'`, `'disc'`, `'cloud'`. */
  key: string;
  /** Dimension of parameter space U: 1 for curves, 2 for surfaces, 3 for volumes. The
   *  engine builds `dim` parameter axes per member. */
  dim: 1 | 2 | 3;
  /** Self-describing metadata for the numeric params `place` reads (see `ParamDesc`).
   *  Optional — a shape with no tunable params (ring, line, shell) omits it. */
  params?: ParamDesc[];
  /**
   * Optional: axis 0 is FOLDED into `folds(params)` equal segments — `floor(u0 · folds)`
   * picks the segment (a spiral's arm) and the fraction within it is the segment's own
   * run. Declaring it changes how a density composes: the engine applies an axis-0 warp
   * WITHIN each segment, so the density shapes every segment's profile instead of
   * dragging members across segment boundaries (a monotone warp of the whole folded axis
   * would re-assign segments — exponential on a 3-arm spiral measured 66/25/9% arm
   * populations before this existed). A shape that folds axis 0 without declaring it
   * accepts that coarse behaviour, as `dendrite` documents it does.
   */
  folds?(params: Params): number;
  /**
   * Map `count` parameter points onto the manifold.
   *
   * @param u      Parameter coordinates, `dim * count` floats, column-major per member:
   *               member `k`'s coordinates are `u[k*dim + a]` for axis `a` in `[0, dim)`.
   *               Every coordinate is in `[0, 1)`. The buffer is engine scratch and may be
   *               LONGER than `dim * count` — read exactly the first `dim * count` entries,
   *               and do not write into it.
   * @param radius The group radius; the shape's size policy maps U onto `[-radius, radius]`
   *               however it sees fit (a ring uses it as the circle radius, a cloud as its
   *               ~3σ extent, …).
   * @param count  Members to place.
   * @param out    Destination; write exactly `count * 3` floats at `outOffset` (member `k`
   *               at `outOffset + 3k`), LOCAL coordinates centred on the origin. Write every
   *               slot — the buffer arrives with arbitrary prior contents.
   * @param params Spec params; read known keys, ignore the rest.
   */
  place(
    u: Float32Array,
    radius: number,
    count: number,
    out: Float32Array,
    outOffset: number,
    params: Params,
  ): void;
}

/** Which member gets which parameter coordinate along axis 0 (the shape's primary axis —
 *  arc length on a curve, the first surface coordinate, …). */
export interface Ordering {
  /** Registry key, e.g. `'byIndex'`, `'byFacet'`, `'byAdjacency'`. */
  key: string;
  /**
   * Assign `inp.count` quantiles, one per member in member order: the returned array's
   * `[k]` is member `inp.members[k]`'s position along axis 0.
   *
   * `stringParams` is `PlacementSpec.stringParams`, passed through verbatim — the
   * facet/column KEY for `byFacet`/`byColumn`. Only orderings receive it (see the
   * `stringParams` doc below); orderings that don't need a key ignore it.
   *
   * CONTRACT: the result is a PERMUTATION of the equal-spaced quantiles `(rank + 0.5) /
   * count` for rank `0 … count-1` — an ordering decides only WHO sits WHERE, never how
   * tightly. Spacing is the density knob's job, and the inverse-CDF warp is only meaningful
   * over equal-spaced inputs; an ordering that bunches its outputs silently corrupts every
   * density it is composed with.
   *
   * The returned array is freshly produced per call (length ≥ `count`; entries beyond
   * `count` are meaningless). The engine reads `[0, count)` immediately and does not retain
   * the reference, so implementations may hand back an internal scratch they overwrite on
   * the next call.
   */
  assign(
    inp: PlaceInput,
    params: Params,
    stringParams?: Readonly<Record<string, string>>,
  ): Float32Array;
}

/** A distribution over parameter space, applied as an inverse-CDF warp of the ordered
 *  quantiles (§3.3): feed the equal-spaced quantile `q` through the distribution's inverse
 *  CDF and the members crowd where the density is high. Ring + gaussian = most members
 *  bunched at one point of the ring, thinning toward the antipode. */
export interface Density {
  /** Registry key, e.g. `'uniform'`, `'gaussian'`, `'powerLaw'`. */
  key: string;
  /** Self-describing metadata for the numeric params `warp` reads (see `ParamDesc`).
   *  Optional — `uniform` (the identity) omits it. */
  params?: ParamDesc[];
  /**
   * The inverse CDF on `[0, 1)`: monotone non-decreasing — OR circular-monotone, a
   * monotone lift taken `mod 1` (`gaussian` wraps its tails this way, because whether an
   * axis is periodic is the SHAPE axis's property and the density cannot know which it
   * feeds). The result must stay inside `[0, 1)` — shapes rely on the parameter domain,
   * so a density squashes and stretches it but never leaves it. On a periodic shape axis
   * (ring angle, torus angles, the azimuth of disc/shell/ball) a wrapped warp reads as
   * the wrapped distribution; on a NON-periodic axis keep the distribution's mass inside
   * [0, 1] (gaussian: center ± ~3σ) or the tail that crosses an end re-enters from the
   * other — a contiguous ordering can then be torn across the two ends, which is the
   * composition the caller asked for, not a library bug. `uniform` is the identity.
   * Pure: same `(q, params)` → same answer.
   */
  warp(q: number, params: Params): number;
}

/** Displacement OFF the manifold — the flesh on the shape's bone: a ring becomes a tube, a
 *  shell a fuzzy shell, a disc gains hashed thickness that thins with radius. */
export interface Scatter {
  /** Registry key, e.g. `'none'`, `'gaussian'`, `'discFalloff'`. */
  key: string;
  /** Self-describing metadata for the numeric params `offset` reads (see `ParamDesc`).
   *  Optional — `none` omits it. */
  params?: ParamDesc[];
  /**
   * ADD a displacement to one already-placed member — read-modify-write on the three slots,
   * never overwrite, so scatter composes with whatever the shape wrote.
   *
   * @param k      The member's position within the group, `0 … count-1` (for effects that
   *               vary along the shape, e.g. thinning with the parameter).
   * @param member The member's global id (`inp.members[k]`) — the hash key, so a member
   *               keeps its scatter when the group around it changes.
   * @param salt   The group's `PlaceInput.salt`; derive draws as `hash(member, salt + C)`
   *               with implementation-chosen constants `C` (avoid `100…102`, reserved by
   *               the engine for parameter axes).
   * @param radius The group radius, the natural scale for displacement magnitudes.
   * @param out    Position buffer; the member's x,y,z live at `o`, `o+1`, `o+2`.
   * @param o      Base index of the member's position in `out`.
   * @param params Spec params; read known keys, ignore the rest.
   */
  offset(
    k: number,
    member: number,
    salt: number,
    radius: number,
    out: Float32Array,
    o: number,
    params: Params,
  ): void;
}

/** One group's placement recipe — the four knob keys plus their numeric params. This is
 *  what a hierarchy level's `place:` holds, and what a preset ("spiralGalaxy", …) bundles. */
export interface PlacementSpec {
  /** Shape registry key. Required — there is no default manifold. */
  shape: string;
  /** Ordering registry key. Default `'byIndex'` (members in ascending id order). */
  ordering?: string;
  /** Density registry key. Default `'uniform'` (no warp). */
  density?: string;
  /** WHICH parameter axis the density warps. Default `0` (the ordering axis). A disc galaxy
   *  warps the radial axis: shape `'disc'` reads angle from axis 0 and radius from axis 1,
   *  so `density: 'exponential', densityAxis: 1` crowds members toward the core while the
   *  ordering still controls who sits at which angle. Must be `< shape.dim` — the engine
   *  throws on an axis the shape does not have, because silently not warping would be a
   *  spec typo you could only discover by squinting at a screenshot. */
  densityAxis?: 0 | 1 | 2;
  /** Scatter registry key. Default `'none'`. */
  scatter?: string;
  /** Numeric parameters, visible to all four knobs. Default `{}`. */
  params?: Params;
  /** String-valued parameters — the facet/column KEY an ordering reads (`byFacet` reads
   *  `stringParams.facet`, `byColumn` reads `stringParams.column`). Kept apart from
   *  `params` so `params` stays pure numbers (a param is a schema row, §7) and the two
   *  cannot be confused. Stage-5 plumbing decision: the engine hands this bag to
   *  `ordering.assign` as its third argument and to NOTHING else — shapes, densities and
   *  scatters are numeric-only by design, and widening them would reopen the stringly
   *  config the predecessor had. Default `{}` (orderings that need a key throw a
   *  descriptive error when it is absent). */
  stringParams?: Record<string, string>;
}
