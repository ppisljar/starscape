/* The hierarchy compiler (DESIGN.md §3.4): recursive levels — the feature the redesign
 * exists for. The predecessor hardcoded two levels (galaxy/system) and wrote the recursion
 * out longhand per shape (vault-site-3d arrange.js `layout`/`nest`); starscape takes a
 * LIST, and one compiler walks it.
 *
 * The model, restated from §3.4:
 *   - Level k partitions each level-(k−1) group by its `groupBy` facet. Only NON-EMPTY
 *     child groups exist (every dropped empty is counted in `stats.emptyDropped` — §10.4,
 *     silence lies). Each non-empty group becomes a PIVOT: a centre, a radius, a parent,
 *     a spin axis/rate — the per-group record the §4 GPU transform chain and the §3.5
 *     frame system consume. Pivots are O(groups), never O(nodes).
 *   - Each level's `place` spec seats its CHILD GROUP CENTRES inside the parent group:
 *     a graph-free PlaceInput (graph = null — orderings that read columns throw, by
 *     contract) whose members are the child pivot ids and whose radius is the PARENT's.
 *     Centres are LOCAL to the parent (engine contract); `pivots.world` is the resolved
 *     chain, one forward pass since parents always precede children.
 *   - The FINAL level is the leaf: no `groupBy`, it places actual nodes (graph ≠ null,
 *     members = node ids, radius = their leaf group's) into the endpoint buffer B.
 *   - Group radius = `spacing × count^(1/3)` — count is the group's NODE count, sublinear
 *     so one giant group cannot swallow the field — floored at `radiusFloor` (default
 *     `spacing × 2`, so a four-member group is still a place you can fly to).
 *
 * Determinism (§10.1): every salt is a pure mix of (opts.salt, level, parent pivot id) —
 * sibling groups decorrelate, and the same graph + levels + opts is byte-identical
 * forever, pinned by FNV goldens in test/hierarchy.test.ts. Partition order is facet-value
 * order inside parent-id order, so nothing depends on object-key or insertion order.
 *
 * Guard rails, all loud:
 *   - more than MAX_LEVELS levels throws (the shader's frame-chain depth is finite even
 *     though placement's is not — §3.4);
 *   - a `groupBy` naming no facet throws, naming the known facets;
 *   - a missing `groupBy` on a non-leaf level (or a stray one on the leaf) throws;
 *   - a child radius exceeding its parent's is scaled into it and counted in
 *     `stats.radiusScaled`, never silently. (Under the DEFAULT radius policy it cannot
 *     fire — a child's node count never exceeds its parent's and both share one floor, so
 *     child ≤ parent always — but a level's `radiusScale` override CAN push a child past
 *     its parent, and then this guard engages rather than silently overlapping.)
 */

import type { Graph } from '../core/types';
import { hash } from '../core/hash';
import { placeGroup } from './engine';
import { getPreset } from './presets';
import { getShape } from './registry';
import type { PlacementSpec } from './types';

/** Hard ceiling on `levels.length`. Placement itself has no depth limit by design (§3.4),
 *  but the stage-6 shader composes a fixed maximum frame-chain depth, and a spec deeper
 *  than anything the renderer can honour is a mistake worth hearing about at compile
 *  time. A constant to raise, deliberately exported. */
export const MAX_LEVELS = 8;

/** One level's rotating-frame request (§3.5): stamped onto every pivot the level's
 *  `groupBy` creates. Declaring spin on the LEAF level throws unless it is the only level
 *  (the leaf creates no pivots of its own to turn — put spin on the level whose groupBy
 *  makes the groups; a single-level hierarchy's synthetic whole-field group is the one
 *  exception, and there the spin turns the whole field as a frame). */
export interface SpinSpec {
  /** Angular rate (radians per time unit — the renderer owns the clock). Copied verbatim
   *  onto every pivot of the level; 0 is a legal "frame exists but idle". */
  rate: number;
  /** Rotation axis, normalised at compile time (a zero or non-finite axis throws).
   *  Default [0, 1, 0] — canonical up, matching the shape library's orientation. */
  axis?: [number, number, number];
}

/** The curated shape set a `{ random: true }` level draws from, in no significant order (§3.3).
 *  Real geometric manifolds with distinct silhouettes — a curve (`ring`, `helix`), surfaces
 *  (`shell`, `disc`, `torus`) and volumes (`ball`, `box`, `cloud`) — each usable with the default
 *  ordering and NO required params, so a group gets a recognisable shape whatever it is handed.
 *  Presets are deliberately excluded: they bundle an ordering/density a bare per-group pick would
 *  not feed. A level may override the set (`{ random: true, shapes: [...] }`). */
export const RANDOM_SHAPES = [
  'ring',
  'helix',
  'shell',
  'disc',
  'torus',
  'ball',
  'box',
  'cloud',
] as const;

/** A per-group RANDOM arrangement for a level (§3.3, §10.1): instead of ONE shape for the whole
 *  level, each group pivot the level fills draws a shape from `shapes` (default RANDOM_SHAPES)
 *  chosen by `hash(groupPivotId, salt)` — fully DETERMINISTIC (no RNG, no clock), so the same
 *  hierarchy + salt yields an identical assignment forever. Everything else the level does —
 *  ordering, density, scatter, params, and the level's own spread (`radiusScale`) and `spin` —
 *  is shared across groups through `base`; only the SHAPE varies. On the LEAF level each leaf
 *  group lays its nodes out with its own shape; on an interior level each parent group lays its
 *  child centres out with its own shape. */
export interface RandomPlace {
  /** The sentinel discriminant — must be the literal `true`. */
  random: true;
  /** Shape keys to pick from (each resolved and validated at compile time; an unknown key throws
   *  naming the registered shapes). Default RANDOM_SHAPES. */
  shapes?: readonly string[];
  /** The shared knobs applied to EVERY group (ordering / density / densityAxis / scatter /
   *  params) — a PlacementSpec without its `shape`, which the per-group pick supplies. A
   *  `densityAxis` is clamped into each chosen shape's dimension (the shapes vary in dim, so an
   *  axis valid for a volume would be out of range for a curve). Default `{}`. */
  base?: Omit<PlacementSpec, 'shape'>;
}

/** One hierarchy level (§3.4). Interior levels carry `groupBy`; the last level is the
 *  leaf and must not. */
export interface LevelSpec {
  /** Facet KEY to partition by. Required on every level but the last; forbidden on the
   *  last (the leaf places nodes — a trailing groupBy would create groups nothing ever
   *  places, which is a spec typo, not a request). */
  groupBy?: string;
  /** How this level seats its children (child group centres, or nodes at the leaf):
   *  a four-knob PlacementSpec, a registered preset by name (`{ preset: 'spiralGalaxy' }` —
   *  resolved through src/place/presets, unknown names throw at compile time), or a per-group
   *  RANDOM arrangement (`{ random: true }` — each group a different shape, see RandomPlace). */
  place: PlacementSpec | { preset: string } | RandomPlace;
  /** Optional rotating frame about this level's group pivots (§3.5). */
  spin?: SpinSpec;
  /** Scale applied to this level's computed group radii — AFTER the `spacing × count^⅓`
   *  policy (floor included), BEFORE the parent-fit guard, so a scale large enough to
   *  push a child past its parent is clamped into it and counted in `stats.radiusScaled`
   *  rather than overlapping silently. Scaling a level's radius is how far apart the
   *  things INSIDE its groups sit (child group centres, or leaf nodes): < 1 packs them
   *  around the group centre, > 1 spreads them. Default 1; must be finite and > 0.
   *  Like `spin`, it is forbidden on the leaf level of a multi-level hierarchy (the leaf
   *  creates no pivots of its own to scale — scale the level whose groupBy makes the
   *  groups); a single-level hierarchy's synthetic whole-field group is the exception. */
  radiusScale?: number;
}

export interface HierarchyOpts {
  /** The page spacing — the unit every radius is derived from (§7: distances keep meaning
   *  across corpora). Must be finite and > 0. */
  spacing: number;
  /** Radius floor. Default `spacing * 2` — the §3.4 floor that keeps a four-member group
   *  a place you can fly to. */
  radiusFloor?: number;
  /** Base of every derived salt (default 0): change it and every level re-deals, change
   *  nothing and the output is byte-identical forever. */
  salt?: number;
  /** Scales the root radius — the sphere the top level's group centres are seated in — so
   *  the outermost groups spread apart (or pack in) without changing how big each group is.
   *  Default 1. A level's own `radiusScale` governs how far its *contents* spread; this is
   *  the one knob above level 0, which no level's radiusScale can reach. Finite and > 0. */
  rootRadiusScale?: number;
}

/** The pivot table: one record per group, columnar (§3.1 discipline — typed arrays, no
 *  per-pivot objects). Parents always have LOWER ids than their children, so one forward
 *  pass resolves the chain (the predecessor's frame-ordering invariant, kept). */
export interface PivotTable {
  count: number;
  /** 3p — pivot centres LOCAL to their parent (root pivots: world). */
  centre: Float32Array;
  /** 3p — resolved world centres (centre + parent's world, one forward pass). */
  world: Float32Array;
  /** Parent pivot id, -1 for root (top-level groups hang off the implicit world). */
  parent: Int32Array;
  /** Hierarchy level of each pivot: 0 for the first groupBy level, and so on. */
  level: Uint8Array;
  /** Group radius after floor and parent-fit guard (see module header). */
  radius: Float32Array;
  /** 3p — unit spin axis per pivot (the level's, normalised; default 0,1,0). */
  axis: Float32Array;
  /** Spin rate per pivot (the level's; 0 where the level declared none). */
  rate: Float32Array;
}

export interface HierarchyStats {
  /** Groups created per groupBy level, in level order (length = levels.length − 1; the
   *  synthetic whole-field group of a leaf-only hierarchy is not a grouping level). */
  groupsPerLevel: number[];
  /** (parent × facet-value) combinations that held no members and were dropped — §3.4's
   *  "only non-empty child groups exist", reported rather than silent. */
  emptyDropped: number;
  /** Groups whose radius was scaled down into their parent's (the guard-rail backstop —
   *  see the module header; 0 under the default radius policy). */
  radiusScaled: number;
}

export interface CompiledHierarchy {
  /** 3n — the endpoint buffer B of DESIGN §4: every node's compiled position, world
   *  space, indexed by node id. */
  positions: Float32Array;
  pivots: PivotTable;
  /** Leaf pivot per node: the innermost group each node belongs to. Walking
   *  `pivots.parent` from it reaches -1 in at most `levels.length` steps. */
  nodePivot: Uint32Array;
  /** Frame-chain depth the shader must compose: 1 + the deepest level declaring `spin`,
   *  0 when none does. (Declared is what counts — a `rate: 0` frame still exists.) */
  chainDepth: number;
  stats: HierarchyStats;
}

/* ------------------------------------------------------------------ internals */

/** The deterministic per-group salt: a pure mix of (base, level, pivot id of the group
 *  being filled; -1 = the implicit root). Same double-odd-multiplier construction as the
 *  synth generator's stream salts — distinct inputs give decorrelated hash streams. */
function groupSalt(base: number, level: number, pivot: number): number {
  return (
    (Math.imul(base ^ Math.imul(level + 1, 0x9e3779b1), 0x85ebca6b) ^
      Math.imul(pivot + 2, 0xc2b2ae35)) |
    0
  );
}

/** Is this level's `place` the per-group RANDOM form? (narrows to RandomPlace). */
function isRandomPlace(place: LevelSpec['place']): place is RandomPlace {
  return 'random' in place && place.random === true;
}

/** A non-random `place:` → a concrete PlacementSpec; `{ preset }` resolves through the preset
 *  registry (throws naming the known presets on a miss — at compile time, not mid-placement).
 *  RANDOM levels resolve PER GROUP instead (randomSpecFor), so they never reach here. */
function resolvePlace(place: PlacementSpec | { preset: string }): PlacementSpec {
  return 'preset' in place ? getPreset(place.preset) : place;
}

/** Resolve the PlacementSpec for ONE group of a RANDOM level: hash the group pivot id into the
 *  level's shape set and bolt the chosen shape onto the shared `base` knobs. Deterministic
 *  (§10.1): `hash(groupPivot, salt)` with a shape-dedicated, level-decorrelated salt — same
 *  hierarchy + base salt ⇒ same shape every time. `groupPivot` is the pivot the level is FILLING
 *  (the parent whose children are being seated, or the leaf group whose nodes are being placed),
 *  the same id the group's placement salt already keys on. */
function randomSpecFor(
  rp: RandomPlace,
  groupPivot: number,
  level: number,
  salt: number,
): PlacementSpec {
  const set = rp.shapes ?? RANDOM_SHAPES;
  // A stream distinct from the placement draws that reuse `salt`, so the shape choice does not
  // correlate with where the members land. level-folded so the same pivot id at two levels differs.
  const stream = (salt ^ Math.imul(level + 1, 0x27d4eb2f)) | 0;
  const pick = (hash(groupPivot, stream) * set.length) | 0;
  const shape = set[pick < set.length ? pick : set.length - 1]!;
  const base = rp.base ?? {};
  const spec: PlacementSpec = { ...base, shape };
  if (base.densityAxis !== undefined) {
    // the shapes vary in dim; clamp a shared densityAxis into the chosen shape's range so a
    // volume-level axis does not throw on a curve/surface group.
    const dim = getShape(shape).dim;
    spec.densityAxis = Math.min(base.densityAxis, dim - 1) as 0 | 1 | 2;
  }
  return spec;
}

/* ------------------------------------------------------------------ the compiler */

/**
 * Compile a recursive hierarchy (§3.4) over a graph: partition level by level, size every
 * group, seat child group centres with each level's placement spec, seat nodes with the
 * leaf's, resolve the pivot chain to world space. Everything here runs once per spec
 * change — never per frame (§4); per-frame work downstream is O(pivots).
 *
 * A single-level `levels` (just a leaf) is legal: the whole graph becomes one synthetic
 * group at the origin (radius from n), so the leaf spec places the entire field and
 * `nodePivot` still gives every node a frame to ride.
 */
export function compileHierarchy(
  graph: Graph,
  levels: LevelSpec[],
  opts: HierarchyOpts,
): CompiledHierarchy {
  const L = levels.length;
  if (L === 0) {
    throw new Error('compileHierarchy: levels is empty — at least a leaf level is required');
  }
  if (L > MAX_LEVELS) {
    throw new Error(
      `compileHierarchy: ${L} levels exceeds MAX_LEVELS (${MAX_LEVELS}) — placement has no ` +
        'depth limit by design, but the renderer composes a bounded frame chain; raise the ' +
        'constant deliberately rather than overflowing it silently',
    );
  }
  const spacing = opts.spacing;
  if (!Number.isFinite(spacing) || spacing <= 0) {
    throw new Error(`compileHierarchy: opts.spacing must be finite and > 0, got ${spacing}`);
  }
  const radiusFloor = opts.radiusFloor ?? spacing * 2;
  if (!Number.isFinite(radiusFloor) || radiusFloor < 0) {
    throw new Error(
      `compileHierarchy: opts.radiusFloor must be finite and >= 0, got ${radiusFloor}`,
    );
  }
  const rootRadiusScale = opts.rootRadiusScale ?? 1;
  if (!Number.isFinite(rootRadiusScale) || rootRadiusScale <= 0) {
    throw new Error(
      `compileHierarchy: opts.rootRadiusScale must be finite and > 0, got ${rootRadiusScale}`,
    );
  }
  const baseSalt = (opts.salt ?? 0) | 0;
  const leafIndex = L - 1;
  const grouping = leafIndex; // number of groupBy levels

  /* ---- validate every level up front: a typo in level 5 must not come up only after
   * levels 0–4 placed half the field. Spin axes are normalised here, once per level. */
  /* placeSpecs[k] is the level's ONE resolved spec, or null when the level is RANDOM — then
   * randomSpecs[k] holds the RandomPlace and the spec is resolved per group at placement. */
  const placeSpecs: (PlacementSpec | null)[] = [];
  const randomSpecs: (RandomPlace | null)[] = [];
  const radiusScaleOf = new Float64Array(L).fill(1);
  const spinAxis = new Float64Array(L * 3);
  const spinRate = new Float64Array(L);
  const hasSpin = new Uint8Array(L);
  for (let k = 0; k < L; k++) {
    const lv = levels[k]!;
    if (k < leafIndex && lv.groupBy === undefined) {
      throw new Error(
        `compileHierarchy: level ${k} has no groupBy — only the last (leaf) level places ` +
          'nodes; every level above it must name the facet it partitions by',
      );
    }
    if (k === leafIndex && lv.groupBy !== undefined) {
      throw new Error(
        `compileHierarchy: the leaf level (index ${k}) must not have groupBy "${lv.groupBy}" ` +
          '— it places nodes, not groups; add a final leaf level if you meant to nest deeper',
      );
    }
    if (k < leafIndex) {
      const key = lv.groupBy!;
      if (graph.facets[key] === undefined) {
        const known = Object.keys(graph.facets).sort().join(', ');
        throw new Error(
          `compileHierarchy: level ${k} groupBy "${key}" names no facet — known: ` +
            `${known || '(none)'}`,
        );
      }
    }
    if (isRandomPlace(lv.place)) {
      const set = lv.place.shapes ?? RANDOM_SHAPES;
      if (set.length === 0) {
        throw new Error(`compileHierarchy: level ${k} random place has an empty shape set`);
      }
      for (const sk of set) getShape(sk); // validate every key now, naming the registered shapes
      placeSpecs.push(null);
      randomSpecs.push(lv.place);
    } else {
      placeSpecs.push(resolvePlace(lv.place));
      randomSpecs.push(null);
    }
    const rs = lv.radiusScale;
    if (rs !== undefined) {
      if (k === leafIndex && L > 1) {
        throw new Error(
          'compileHierarchy: radiusScale on the leaf level has no pivot of its own to ' +
            'scale — put it on the level whose groupBy creates the groups it should resize',
        );
      }
      if (!Number.isFinite(rs) || rs <= 0) {
        throw new Error(
          `compileHierarchy: level ${k} radiusScale must be finite and > 0, got ${rs}`,
        );
      }
      radiusScaleOf[k] = rs;
    }
    spinAxis[k * 3 + 1] = 1; // default axis: canonical up
    const spin = lv.spin;
    if (spin !== undefined) {
      if (k === leafIndex && L > 1) {
        throw new Error(
          'compileHierarchy: spin on the leaf level has no pivot of its own to turn — ' +
            'declare it on the level whose groupBy creates the groups it should spin',
        );
      }
      if (!Number.isFinite(spin.rate)) {
        throw new Error(`compileHierarchy: level ${k} spin.rate must be finite, got ${spin.rate}`);
      }
      if (spin.axis !== undefined) {
        const len = Math.hypot(spin.axis[0], spin.axis[1], spin.axis[2]);
        if (!Number.isFinite(len) || len === 0) {
          throw new Error(
            `compileHierarchy: level ${k} spin.axis must have finite non-zero length`,
          );
        }
        spinAxis[k * 3] = spin.axis[0] / len;
        spinAxis[k * 3 + 1] = spin.axis[1] / len;
        spinAxis[k * 3 + 2] = spin.axis[2] / len;
      }
      spinRate[k] = spin.rate;
      hasSpin[k] = 1;
    }
  }

  /* ---- phase 1: partition, level by level.
   *
   * Membership lives in ONE permutation of the node ids (`order`), with each group a
   * contiguous [start, count) range — the CSR trick applied to grouping, so no per-group
   * member arrays exist. Each level counting-sorts every parent's range by facet value
   * into `next` (stable, so members stay in ascending node-id order within every group —
   * which is what lets `byIndex` take its no-sort fast path at the leaf), then the
   * buffers swap. Counters are reset per parent by walking only the TOUCHED values, so a
   * level costs O(nodes + groups·log) regardless of how large the facet dictionary is.
   *
   * Pivot records accumulate in plain number arrays — O(groups), converted to typed
   * arrays below; the §10.2 rule is about per-NODE allocation, and groups are the thing
   * there are few of. */
  const n = graph.n;
  let order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  let next = new Uint32Array(n);

  const pParent: number[] = [];
  const pStart: number[] = [];
  const pCount: number[] = [];
  const levelFirst: number[] = [0]; // first pivot id of each pivot level; last entry = total
  const groupsPerLevel: number[] = [];
  let emptyDropped = 0;

  for (let k = 0; k < grouping; k++) {
    const facet = graph.facets[levels[k]!.groupBy!]!; // validated above
    const data = facet.data;
    const nv = facet.values.length;
    const counts = new Uint32Array(nv);
    const writePos = new Uint32Array(nv);
    const touched = new Uint32Array(nv);

    const firstParent = k === 0 ? 0 : levelFirst[k - 1]!;
    const lastParent = k === 0 ? 1 : levelFirst[k]!;
    for (let pi = firstParent; pi < lastParent; pi++) {
      // level 0 splits the one implicit root group (id -1) spanning every node
      const parentId = k === 0 ? -1 : pi;
      const s = k === 0 ? 0 : pStart[pi]!;
      const e = s + (k === 0 ? n : pCount[pi]!);

      let nt = 0;
      for (let x = s; x < e; x++) {
        const v = data[order[x]!]!;
        if (counts[v]! === 0) touched[nt++] = v;
        counts[v] = counts[v]! + 1;
      }
      if (nt > 1) touched.subarray(0, nt).sort(); // pivots in facet-value order, always
      emptyDropped += nv - nt;

      let w = s;
      for (let t = 0; t < nt; t++) {
        const v = touched[t]!;
        pParent.push(parentId);
        pStart.push(w);
        pCount.push(counts[v]!);
        writePos[v] = w;
        w += counts[v]!;
      }
      for (let x = s; x < e; x++) {
        const i = order[x]!;
        const v = data[i]!;
        next[writePos[v]!] = i;
        writePos[v] = writePos[v]! + 1;
      }
      for (let t = 0; t < nt; t++) counts[touched[t]!] = 0;
    }
    groupsPerLevel.push(pParent.length - levelFirst[k]!);
    levelFirst.push(pParent.length);
    const tmp = order;
    order = next;
    next = tmp;
  }

  if (grouping === 0) {
    // leaf-only hierarchy: one synthetic whole-field group so every node still has a
    // pivot to ride (and the leaf spec a radius to fill)
    pParent.push(-1);
    pStart.push(0);
    pCount.push(n);
    levelFirst.push(1);
  }

  /* ---- phase 2: the pivot table — radii, parents, levels, spin. */
  const P = pParent.length;
  const parent = new Int32Array(P);
  const level = new Uint8Array(P);
  const centre = new Float32Array(P * 3);
  const world = new Float32Array(P * 3);
  const radius = new Float32Array(P);
  const axis = new Float32Array(P * 3);
  const rate = new Float32Array(P);
  for (let p = 0; p < P; p++) parent[p] = pParent[p]!;
  for (let k = 0; k + 1 < levelFirst.length; k++) {
    const ax = spinAxis[k * 3]!;
    const ay = spinAxis[k * 3 + 1]!;
    const az = spinAxis[k * 3 + 2]!;
    const rt = spinRate[k]!;
    for (let p = levelFirst[k]!; p < levelFirst[k + 1]!; p++) {
      level[p] = k;
      axis[p * 3] = ax;
      axis[p * 3 + 1] = ay;
      axis[p * 3 + 2] = az;
      rate[p] = rt;
    }
  }

  const rootRadius =
    Math.max(radiusFloor, spacing * Math.cbrt(Math.max(n, 1))) * rootRadiusScale;
  let radiusScaled = 0;
  for (let p = 0; p < P; p++) {
    // the count^⅓ policy (floor included), then the level's radiusScale, THEN the
    // parent-fit guard — so an oversized scale is clamped loudly, never overlapped
    let r = Math.max(radiusFloor, spacing * Math.cbrt(pCount[p]!)) * radiusScaleOf[level[p]!]!;
    const pr = parent[p]! < 0 ? rootRadius : radius[parent[p]!]!;
    if (r > pr) {
      r = pr; // scaled into the parent — reported, never silent (§10.4)
      radiusScaled++;
    }
    radius[p] = r;
  }

  let chainDepth = 0;
  for (let k = 0; k + 1 < levelFirst.length; k++) {
    if (hasSpin[k] === 1) chainDepth = k + 1;
  }

  /* ---- phase 3: seat child group centres, level by level.
   * Children of one parent are consecutive pivot ids (allocated that way in phase 1), so
   * each group's members are a subarray of the identity id list and its output lands
   * directly in `centre` at its own offset — LOCAL to the parent, per the engine
   * contract. The PlaceInput radius is the PARENT's (§3.4: centres go inside the parent
   * radius); graph is null — these members are groups, not nodes. */
  const pivotIds = new Uint32Array(P);
  for (let p = 0; p < P; p++) pivotIds[p] = p;
  for (let k = 0; k < grouping; k++) {
    const spec = placeSpecs[k];
    const rp = randomSpecs[k];
    const lo = levelFirst[k]!;
    const hi = levelFirst[k + 1]!;
    let a = lo;
    while (a < hi) {
      const par = parent[a]!;
      let b = a + 1;
      while (b < hi && parent[b]! === par) b++;
      // RANDOM level: the parent group being filled picks its own shape (hash(par, salt)); a
      // plain level reuses its one resolved spec. Everything else about the call is identical.
      const gSalt = groupSalt(baseSalt, k, par);
      placeGroup(spec ?? randomSpecFor(rp!, par, k, gSalt), {
        members: pivotIds.subarray(a, b),
        count: b - a,
        graph: null,
        salt: gSalt,
        radius: par < 0 ? rootRadius : radius[par]!,
        out: centre,
        outOffset: a * 3,
      });
      a = b;
    }
  }
  // (the synthetic whole-field group of a leaf-only hierarchy stays at the origin)

  /* ---- phase 4: resolve world centres — parents precede children, one forward pass. */
  for (let p = 0; p < P; p++) {
    const pr = parent[p]!;
    const o = p * 3;
    if (pr < 0) {
      world[o] = centre[o]!;
      world[o + 1] = centre[o + 1]!;
      world[o + 2] = centre[o + 2]!;
    } else {
      const po = pr * 3;
      world[o] = centre[o]! + world[po]!;
      world[o + 1] = centre[o + 1]! + world[po + 1]!;
      world[o + 2] = centre[o + 2]! + world[po + 2]!;
    }
  }

  /* ---- phase 5: the leaf level — place actual nodes (graph ≠ null) into each leaf
   * group, then write world positions into the endpoint buffer by NODE id. The engine
   * writes in member order into a scratch slab (grown to the largest leaf group), because
   * the endpoint buffer is node-id-indexed and the two orders differ. */
  const positions = new Float32Array(n * 3);
  const nodePivot = new Uint32Array(n);
  const leafLo = levelFirst[levelFirst.length - 2]!;
  const leafHi = levelFirst[levelFirst.length - 1]!;
  let maxLeaf = 0;
  for (let q = leafLo; q < leafHi; q++) {
    if (pCount[q]! > maxLeaf) maxLeaf = pCount[q]!;
  }
  const scratch = new Float32Array(maxLeaf * 3);
  const leafSpec = placeSpecs[leafIndex];
  const leafRandom = randomSpecs[leafIndex];
  for (let q = leafLo; q < leafHi; q++) {
    const s = pStart[q]!;
    const c = pCount[q]!;
    // RANDOM leaf: each leaf group lays its nodes out with a shape of its own (hash(q, salt)).
    const gSalt = groupSalt(baseSalt, leafIndex, q);
    placeGroup(leafSpec ?? randomSpecFor(leafRandom!, q, leafIndex, gSalt), {
      members: order.subarray(s, s + c),
      count: c,
      graph,
      salt: gSalt,
      radius: radius[q]!,
      out: scratch,
      outOffset: 0,
    });
    const qo = q * 3;
    const wx = world[qo]!;
    const wy = world[qo + 1]!;
    const wz = world[qo + 2]!;
    for (let m = 0; m < c; m++) {
      const i = order[s + m]!;
      const b = m * 3;
      const o = i * 3;
      nodePivot[i] = q;
      positions[o] = scratch[b]! + wx;
      positions[o + 1] = scratch[b + 1]! + wy;
      positions[o + 2] = scratch[b + 2]! + wz;
    }
  }

  return {
    positions,
    pivots: { count: P, centre, world, parent, level, radius, axis, rate },
    nodePivot,
    chainDepth,
    stats: { groupsPerLevel, emptyDropped, radiusScaled },
  };
}
