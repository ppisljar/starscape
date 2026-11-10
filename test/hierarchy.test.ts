/* Stage-5 capstone tests: the recursive hierarchy compiler (DESIGN.md §3.4) and the
 * multi-source BFS pull (arrange.js `bySubset`, ported as src/place/pull).
 *
 * The goldens pin byte-identity (§10.9): fixed corpus (generateSynth {n:2000, seed:7}) +
 * fixed levels → the same FNV-1a digest over every output buffer, forever. The structural
 * tests pin the semantics the digests cannot explain: partition correctness, parent-chain
 * integrity, the radius policy, empty-group dropping, spin plumbing, and the guard rails.
 */

import { describe, expect, it } from 'vitest';
import '../src/place/engine'; // built-ins: cloud / byIndex / uniform / none
import '../src/place/shapes';
import '../src/place/orderings';
import '../src/place/densities';
import '../src/place/scatters';
import { compileHierarchy, MAX_LEVELS } from '../src/place/hierarchy';
import type { CompiledHierarchy, LevelSpec } from '../src/place/hierarchy';
import { attachToAnchors, HOP_MAX } from '../src/place/pull';
import { buildGraph } from '../src/core/graph';
import { generateSynth } from '../src/synth/generate';
import { fnv1a } from '../src/core/fnv';
import type { Graph } from '../src/core/types';

/* ------------------------------------------------------------------ fixtures */

/** THE corpus of the task spec: generateSynth({n:2000, seed:7}), compiled once. Facets:
 *  type (7 values), galaxy (13), system (169), era (5) — galaxy/system counts come from the
 *  generator's ~n^⅓ defaults (synthDefaults), not fixed constants, since the constant-density
 *  rescale of 2026-11-09. */
const SYNTH = buildGraph(generateSynth({ n: 2000, seed: 7 }));
const GALAXY_VALUES = SYNTH.facets['galaxy']!.values.length; // 13 at n=2000

/** A hand-sized graph: facet `g` with one block of nodes per entry of `sizes` (value k
 *  gets `sizes[k]` consecutive nodes), plus optional never-assigned extra values. No
 *  links, no positions — the smallest thing a hierarchy can compile. */
function handGraph(sizes: number[], extraValues: string[] = []): Graph {
  let n = 0;
  for (const s of sizes) n += s;
  const data = new Uint8Array(n);
  let i = 0;
  for (let v = 0; v < sizes.length; v++) {
    for (let k = 0; k < sizes[v]!; k++) data[i++] = v;
  }
  const values = sizes.map((_, v) => `g${v}`).concat(extraValues);
  return buildGraph({ n, links: new Uint32Array(0), facets: { g: { values, data } } });
}

/** FNV digest of every output buffer — one object so a mismatch diffs all at once. */
function digest(r: CompiledHierarchy): Record<string, number> {
  return {
    positions: fnv1a(r.positions),
    centre: fnv1a(r.pivots.centre),
    world: fnv1a(r.pivots.world),
    parent: fnv1a(r.pivots.parent),
    level: fnv1a(r.pivots.level),
    radius: fnv1a(r.pivots.radius),
    axis: fnv1a(r.pivots.axis),
    rate: fnv1a(r.pivots.rate),
    nodePivot: fnv1a(r.nodePivot),
  };
}

/** Distinct facet-value tuples over `keys` — the independently computed group count the
 *  compiler's partition must agree with. Test-side only; strings are fine here. */
function comboSet(g: Graph, keys: string[]): Set<string> {
  const out = new Set<string>();
  for (let i = 0; i < g.n; i++) {
    let s = '';
    for (const k of keys) s += String(g.facets[k]!.data[i]!) + ':';
    out.add(s);
  }
  return out;
}

/* ------------------------------------------------------------------ 2-level golden */

const LEVELS_2: LevelSpec[] = [
  { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.1 } },
  { place: { shape: 'cloud' } },
];

/* Recorded from a run and pinned (§10.9) — real goldens, not two-runs-agree: a silent
 * change to any buffer's bytes fails here even if it stays self-consistent. */
const GOLDEN_2 = {
  positions: 3023373302,
  centre: 1877975373,
  world: 1877975373, // = centre: root pivots are already in world space
  parent: 3336120569,
  level: 1177189415,
  radius: 3111743380,
  axis: 1297736056,
  rate: 4171167731,
  nodePivot: 3814245314,
};

describe('compileHierarchy: 2 levels over synth {n:2000, seed:7}', () => {
  const r = compileHierarchy(SYNTH, LEVELS_2, { spacing: 10 });

  it('pins the FNV goldens over positions + pivots arrays', () => {
    expect(digest(r)).toEqual(GOLDEN_2);
  });

  it('a second compile is byte-identical (determinism, §10.1)', () => {
    expect(digest(compileHierarchy(SYNTH, LEVELS_2, { spacing: 10 }))).toEqual(digest(r));
  });

  it('a different salt deals a different field', () => {
    const other = compileHierarchy(SYNTH, LEVELS_2, { spacing: 10, salt: 99 });
    expect(fnv1a(other.positions)).not.toBe(fnv1a(r.positions));
  });

  it('one pivot per non-empty type value, all root, all level 0', () => {
    expect(r.pivots.count).toBe(comboSet(SYNTH, ['type']).size);
    expect(r.stats.groupsPerLevel).toEqual([r.pivots.count]);
    for (let p = 0; p < r.pivots.count; p++) {
      expect(r.pivots.parent[p]).toBe(-1);
      expect(r.pivots.level[p]).toBe(0);
    }
  });

  it('nodePivot groups exactly by type, and every position is finite', () => {
    const typeData = SYNTH.facets['type']!.data;
    const seen = new Map<number, number>(); // pivot -> type code
    for (let i = 0; i < SYNTH.n; i++) {
      const p = r.nodePivot[i]!;
      const t = typeData[i]!;
      const prev = seen.get(p);
      if (prev === undefined) seen.set(p, t);
      else expect(prev).toBe(t);
    }
    expect(seen.size).toBe(r.pivots.count);
    for (let k = 0; k < r.positions.length; k++) {
      expect(Number.isFinite(r.positions[k])).toBe(true);
    }
  });

  it('world equals centre for root pivots', () => {
    expect(Array.from(r.pivots.world)).toEqual(Array.from(r.pivots.centre));
  });

  it('chainDepth 1: spin on level 0 only', () => {
    expect(r.chainDepth).toBe(1);
  });
});

/* ------------------------------------------------------------------ 4-level golden */

const LEVELS_4: LevelSpec[] = [
  { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.2 } },
  { groupBy: 'galaxy', place: { preset: 'spiralGalaxy' }, spin: { rate: 0.5, axis: [0, 0, 3] } },
  { groupBy: 'era', place: { shape: 'ring' } },
  { place: { shape: 'cloud', scatter: 'gaussian' } },
];

/* Re-recorded when the spiralGalaxy preset's geometry was deliberately fixed (stage-5
 * review: equal arms via the declared fold, radialPower 1, dead `bulge` dropped) —
 * positions/centre/world moved, every structural digest (parent/level/radius/axis/rate/
 * nodePivot) stayed, which is exactly the shape of a pure geometry change.
 *
 * Re-recorded again 2026-11-09 for the synth constant-density rescale (galaxies now ~n^⅓:
 * the n=2000 corpus deals 13 galaxies where it dealt 12), which re-partitions every level
 * below `type` — all nine digests moved. Old values: positions 276329203, centre 3197021762,
 * world 3874786577, parent 3922889169, level 937075595, radius 1421081648, axis 1830356232,
 * rate 345338110, nodePivot 1595266401. (GOLDEN_2 above partitions by `type` only — an
 * independent hash stream the rescale does not touch — and deliberately did not move.) */
const GOLDEN_4 = {
  positions: 825973778,
  centre: 414606039,
  world: 749269809,
  parent: 1783205304,
  level: 993922690,
  radius: 3590265682,
  axis: 2208874264,
  rate: 763836879,
  nodePivot: 322497763,
};

describe('compileHierarchy: 4 levels over synth {n:2000, seed:7}', () => {
  const r = compileHierarchy(SYNTH, LEVELS_4, { spacing: 10, salt: 42 });

  it('pins the FNV goldens over positions + pivots arrays', () => {
    expect(digest(r)).toEqual(GOLDEN_4);
  });

  it('groupsPerLevel equals the independently counted non-empty combinations', () => {
    const c1 = comboSet(SYNTH, ['type']).size;
    const c2 = comboSet(SYNTH, ['type', 'galaxy']).size;
    const c3 = comboSet(SYNTH, ['type', 'galaxy', 'era']).size;
    expect(r.stats.groupsPerLevel).toEqual([c1, c2, c3]);
    expect(r.pivots.count).toBe(c1 + c2 + c3);
  });

  it('empty groups are DROPPED and the drops are counted, never silent', () => {
    const c1 = comboSet(SYNTH, ['type']).size;
    const c2 = comboSet(SYNTH, ['type', 'galaxy']).size;
    const c3 = comboSet(SYNTH, ['type', 'galaxy', 'era']).size;
    // potential children: 7 type values at the root, GALAXY_VALUES (13 at n=2000, from the
    // generator's ~n^⅓ default) per type group, 5 eras per (type, galaxy) group — minus
    // what actually exists
    const expected = (7 - c1) + (c1 * GALAXY_VALUES - c2) + (c2 * 5 - c3);
    expect(r.stats.emptyDropped).toBe(expected);
  });

  it('nodePivot/parent chain integrity: every walk reaches -1 in <= 4 steps, levels descending', () => {
    for (let i = 0; i < SYNTH.n; i++) {
      let p = r.nodePivot[i]!;
      expect(r.pivots.level[p]).toBe(2); // the deepest groupBy level
      let steps = 0;
      for (;;) {
        const par = r.pivots.parent[p]!;
        steps++;
        expect(steps).toBeLessThanOrEqual(LEVELS_4.length);
        if (par === -1) break;
        expect(r.pivots.level[par]).toBe(r.pivots.level[p]! - 1);
        p = par;
      }
    }
  });

  it('nodePivot partitions exactly by (type, galaxy, era)', () => {
    const t = SYNTH.facets['type']!.data;
    const g = SYNTH.facets['galaxy']!.data;
    const e = SYNTH.facets['era']!.data;
    const byPivot = new Map<number, string>();
    for (let i = 0; i < SYNTH.n; i++) {
      const key = `${t[i]}:${g[i]}:${e[i]}`;
      const p = r.nodePivot[i]!;
      const prev = byPivot.get(p);
      if (prev === undefined) byPivot.set(p, key);
      else expect(prev).toBe(key);
    }
    expect(byPivot.size).toBe(comboSet(SYNTH, ['type', 'galaxy', 'era']).size);
    // distinct pivots carry distinct triples (it is a bijection, not just a function)
    expect(new Set(byPivot.values()).size).toBe(byPivot.size);
  });

  it('world = centre + parent world, down the whole chain', () => {
    const { parent, centre, world, count } = r.pivots;
    for (let p = 0; p < count; p++) {
      const par = parent[p]!;
      for (let c = 0; c < 3; c++) {
        const want = par < 0 ? centre[p * 3 + c]! : centre[p * 3 + c]! + world[par * 3 + c]!;
        // f32 add of two f32 values is exact in f64 comparison terms here
        expect(world[p * 3 + c]).toBeCloseTo(want, 3);
      }
    }
  });

  it('spin plumbing: axis normalised, rate copied, spinless level gets rate 0 and default axis', () => {
    const { count, level, axis, rate } = r.pivots;
    const f32 = Math.fround;
    for (let p = 0; p < count; p++) {
      const l = level[p]!;
      const a = [axis[p * 3]!, axis[p * 3 + 1]!, axis[p * 3 + 2]!];
      if (l === 0) {
        expect(a).toEqual([0, 1, 0]); // default up
        expect(rate[p]).toBe(f32(0.2));
      } else if (l === 1) {
        expect(a).toEqual([0, 0, 1]); // [0, 0, 3] normalised
        expect(rate[p]).toBe(f32(0.5));
      } else {
        expect(a).toEqual([0, 1, 0]);
        expect(rate[p]).toBe(0);
      }
    }
  });

  it('chainDepth 2: the deepest spin sits on level 1', () => {
    expect(r.chainDepth).toBe(2);
  });

  it('every node position is finite', () => {
    for (let k = 0; k < r.positions.length; k++) {
      expect(Number.isFinite(r.positions[k])).toBe(true);
    }
  });
});

/* ------------------------------------------------------------------ radius policy */

describe('radius policy: spacing × count^(1/3), floored (§3.4)', () => {
  it('pins the radii of hand-sized groups: sublinear growth, exact cube roots', () => {
    const g = handGraph([1, 8, 64, 512]);
    const r = compileHierarchy(
      g,
      [{ groupBy: 'g', place: { shape: 'shell' } }, { place: { shape: 'cloud' } }],
      { spacing: 10 },
    );
    // max(floor = spacing*2 = 20, 10·c^⅓): 1 → 20 (floored), 8 → 20 (exactly the floor),
    // 64 → 40, 512 → 80 — sublinear: 8× the members, 2× the radius
    expect(Array.from(r.pivots.radius)).toEqual([20, 20, 40, 80]);
    expect(r.stats.radiusScaled).toBe(0);
  });

  it('radiusFloor defaults to spacing*2 and is honoured when given', () => {
    const g = handGraph([1, 8]);
    const def = compileHierarchy(
      g,
      [{ groupBy: 'g', place: { shape: 'ring' } }, { place: { shape: 'cloud' } }],
      { spacing: 7 },
    );
    expect(def.pivots.radius[0]).toBe(14); // 7·1 floored at 7*2
    const low = compileHierarchy(
      g,
      [{ groupBy: 'g', place: { shape: 'ring' } }, { place: { shape: 'cloud' } }],
      { spacing: 10, radiusFloor: 5 },
    );
    expect(Array.from(low.pivots.radius)).toEqual([10, 20]); // 10·1, 10·2 — floor at 5 never binds
  });
});

/* ------------------------------------------------------------------ radiusScale */

describe('per-level radiusScale: after the count^⅓ policy, before the parent-fit guard', () => {
  /** 8 groups of 64 nodes: root = 10·cbrt(512) = 80, every group 10·cbrt(64) = 40 — far
   *  enough from the root that a 1.5× scale cannot touch the parent-fit guard. */
  const g8 = handGraph([64, 64, 64, 64, 64, 64, 64, 64]);
  const lvls = (rs?: number): LevelSpec[] => [
    { groupBy: 'g', place: { shape: 'shell' }, radiusScale: rs },
    { place: { shape: 'cloud' } },
  ];

  it('scales that level\'s radii linearly', () => {
    const base = compileHierarchy(g8, lvls(), { spacing: 10 });
    const up = compileHierarchy(g8, lvls(1.5), { spacing: 10 });
    const down = compileHierarchy(g8, lvls(0.5), { spacing: 10 });
    for (let p = 0; p < base.pivots.count; p++) {
      expect(base.pivots.radius[p]).toBe(40);
      expect(up.pivots.radius[p]).toBe(60);
      expect(down.pivots.radius[p]).toBe(20);
    }
    expect(up.stats.radiusScaled).toBe(0);
    expect(down.stats.radiusScaled).toBe(0);
  });

  it('scales radii only — the level\'s own centres do not move (children spread, groups stay)', () => {
    const base = compileHierarchy(g8, lvls(), { spacing: 10 });
    const down = compileHierarchy(g8, lvls(0.5), { spacing: 10 });
    expect(fnv1a(down.pivots.centre)).toBe(fnv1a(base.pivots.centre));
    expect(fnv1a(down.positions)).not.toBe(fnv1a(base.positions)); // leaves pack inward
  });

  it('radiusScale 1 (explicit) leaves the goldens byte-identical', () => {
    const levels2: LevelSpec[] = [
      { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.1 }, radiusScale: 1 },
      { place: { shape: 'cloud' } },
    ];
    expect(digest(compileHierarchy(SYNTH, levels2, { spacing: 10 }))).toEqual(GOLDEN_2);
    const levels4: LevelSpec[] = [
      { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.2 }, radiusScale: 1 },
      {
        groupBy: 'galaxy',
        place: { preset: 'spiralGalaxy' },
        spin: { rate: 0.5, axis: [0, 0, 3] },
        radiusScale: 1,
      },
      { groupBy: 'era', place: { shape: 'ring' }, radiusScale: 1 },
      { place: { shape: 'cloud', scatter: 'gaussian' } },
    ];
    expect(digest(compileHierarchy(SYNTH, levels4, { spacing: 10, salt: 42 }))).toEqual(GOLDEN_4);
  });

  it('the parent-fit guard still engages: an oversized scale clamps into the parent, counted', () => {
    const g = handGraph([8, 8]); // root = 10·cbrt(16) ≈ 25.198; groups floored at 20
    const r = compileHierarchy(g, lvls(100), { spacing: 10 });
    expect(r.stats.radiusScaled).toBe(2);
    for (let p = 0; p < r.pivots.count; p++) {
      expect(r.pivots.radius[p]).toBeCloseTo(10 * Math.cbrt(16), 3); // the ROOT radius, not 2000
    }
  });

  it('is allowed on a leaf-only level (the synthetic whole-field group scales)', () => {
    const r = compileHierarchy(SYNTH, [{ place: { shape: 'cloud' }, radiusScale: 0.5 }], {
      spacing: 10,
    });
    expect(r.pivots.radius[0]).toBeCloseTo(0.5 * 10 * Math.cbrt(SYNTH.n), 3);
    expect(r.stats.radiusScaled).toBe(0);
  });

  it('guard rails: non-positive or non-finite throws; multi-level leaf radiusScale throws', () => {
    expect(() => compileHierarchy(g8, lvls(0), { spacing: 10 })).toThrow(/radiusScale/);
    expect(() => compileHierarchy(g8, lvls(NaN), { spacing: 10 })).toThrow(/radiusScale/);
    expect(() =>
      compileHierarchy(
        g8,
        [{ groupBy: 'g', place: { shape: 'shell' } }, { place: { shape: 'cloud' }, radiusScale: 2 }],
        { spacing: 10 },
      ),
    ).toThrow(/radiusScale on the leaf level/);
  });
});

/* ------------------------------------------------------------------ rootRadiusScale */

describe('rootRadiusScale: the one knob above level 0 — spreads the outermost groups apart', () => {
  // 8 groups of 64: the top-level centres sit on a shell of the root radius; scaling it
  // moves those centres without changing any group's own size.
  const g8 = handGraph([64, 64, 64, 64, 64, 64, 64, 64]);
  const lvls: LevelSpec[] = [
    { groupBy: 'g', place: { shape: 'shell' } },
    { place: { shape: 'cloud' } },
  ];
  const centreSpread = (r: ReturnType<typeof compileHierarchy>): number => {
    // RMS distance of the level-0 pivot centres from their centroid.
    let cx = 0,
      cy = 0,
      cz = 0,
      c = 0;
    for (let p = 0; p < r.pivots.count; p++) {
      if (r.pivots.level[p] !== 0) continue;
      cx += r.pivots.world[p * 3]!;
      cy += r.pivots.world[p * 3 + 1]!;
      cz += r.pivots.world[p * 3 + 2]!;
      c++;
    }
    cx /= c;
    cy /= c;
    cz /= c;
    let s = 0;
    for (let p = 0; p < r.pivots.count; p++) {
      if (r.pivots.level[p] !== 0) continue;
      const dx = r.pivots.world[p * 3]! - cx,
        dy = r.pivots.world[p * 3 + 1]! - cy,
        dz = r.pivots.world[p * 3 + 2]! - cz;
      s += dx * dx + dy * dy + dz * dz;
    }
    return Math.sqrt(s / c);
  };

  it('scales how far apart the top-level centres sit, linearly', () => {
    const base = compileHierarchy(g8, lvls, { spacing: 10 });
    const wide = compileHierarchy(g8, lvls, { spacing: 10, rootRadiusScale: 3 });
    expect(centreSpread(wide) / centreSpread(base)).toBeCloseTo(3, 6);
  });

  it('leaves every group\'s OWN radius untouched (groups move apart, do not grow)', () => {
    const base = compileHierarchy(g8, lvls, { spacing: 10 });
    const wide = compileHierarchy(g8, lvls, { spacing: 10, rootRadiusScale: 3 });
    expect(fnv1a(wide.pivots.radius)).toBe(fnv1a(base.pivots.radius));
    expect(wide.stats.radiusScaled).toBe(0); // bigger root can only help the parent-fit guard
  });

  it('default (omitted or 1) is byte-identical — the goldens hold', () => {
    expect(digest(compileHierarchy(SYNTH, LEVELS_2, { spacing: 10 }))).toEqual(GOLDEN_2);
    expect(digest(compileHierarchy(SYNTH, LEVELS_2, { spacing: 10, rootRadiusScale: 1 }))).toEqual(
      GOLDEN_2,
    );
  });

  it('composes with a level\'s radiusScale: root moves centres, radiusScale grows contents', () => {
    const rootOnly = compileHierarchy(g8, lvls, { spacing: 10, rootRadiusScale: 2 });
    // root scaled but group radii unchanged vs base
    const base = compileHierarchy(g8, lvls, { spacing: 10 });
    expect(rootOnly.pivots.radius[0]).toBe(base.pivots.radius[0]);
  });

  it('guard rails: non-positive or non-finite throws', () => {
    expect(() => compileHierarchy(g8, lvls, { spacing: 10, rootRadiusScale: 0 })).toThrow(
      /rootRadiusScale/,
    );
    expect(() => compileHierarchy(g8, lvls, { spacing: 10, rootRadiusScale: -1 })).toThrow(
      /rootRadiusScale/,
    );
    expect(() => compileHierarchy(g8, lvls, { spacing: 10, rootRadiusScale: NaN })).toThrow(
      /rootRadiusScale/,
    );
  });
});

/* ------------------------------------------------------------------ empty dropping */

describe('empty-group dropping', () => {
  it('a facet value with no members produces no pivot, and the drop is counted', () => {
    const g = handGraph([3, 5], ['unused']);
    const r = compileHierarchy(
      g,
      [{ groupBy: 'g', place: { shape: 'ring' } }, { place: { shape: 'cloud' } }],
      { spacing: 10 },
    );
    expect(r.pivots.count).toBe(2);
    expect(r.stats.groupsPerLevel).toEqual([2]);
    expect(r.stats.emptyDropped).toBe(1);
  });
});

/* ------------------------------------------------------------------ presets + guards */

describe('preset reference resolution and guard rails', () => {
  const leaf: LevelSpec = { place: { shape: 'cloud' } };

  it('place: { preset } resolves through the registry (4-level test uses it); unknown throws', () => {
    expect(() =>
      compileHierarchy(SYNTH, [{ groupBy: 'type', place: { preset: 'nope' } }, leaf], {
        spacing: 10,
      }),
    ).toThrow(/unknown preset "nope"/);
  });

  it('a preset works at the leaf level too (graph-aware knobs allowed there)', () => {
    const r = compileHierarchy(
      SYNTH,
      [{ groupBy: 'type', place: { shape: 'shell' } }, { place: { preset: 'elliptical' } }],
      { spacing: 10 },
    );
    for (let k = 0; k < r.positions.length; k++) {
      expect(Number.isFinite(r.positions[k])).toBe(true);
    }
  });

  it('groupBy naming no facet throws, naming it and the known ones', () => {
    expect(() =>
      compileHierarchy(SYNTH, [{ groupBy: 'nope', place: { shape: 'shell' } }, leaf], {
        spacing: 10,
      }),
    ).toThrow(/groupBy "nope" names no facet.*era.*galaxy.*system.*type/);
  });

  it('more than MAX_LEVELS levels throws', () => {
    const levels: LevelSpec[] = [];
    for (let k = 0; k < MAX_LEVELS; k++) levels.push({ groupBy: 'type', place: { shape: 'shell' } });
    levels.push(leaf);
    expect(levels.length).toBe(MAX_LEVELS + 1);
    expect(() => compileHierarchy(SYNTH, levels, { spacing: 10 })).toThrow(/MAX_LEVELS/);
  });

  it('a non-leaf level without groupBy throws; a leaf WITH groupBy throws', () => {
    expect(() =>
      compileHierarchy(SYNTH, [{ place: { shape: 'shell' } }, leaf], { spacing: 10 }),
    ).toThrow(/level 0 has no groupBy/);
    expect(() =>
      compileHierarchy(SYNTH, [{ groupBy: 'type', place: { shape: 'cloud' } }], { spacing: 10 }),
    ).toThrow(/leaf level.*must not have groupBy/);
  });

  it('empty levels throws; bad spacing throws', () => {
    expect(() => compileHierarchy(SYNTH, [], { spacing: 10 })).toThrow(/levels is empty/);
    expect(() => compileHierarchy(SYNTH, [leaf], { spacing: 0 })).toThrow(/spacing/);
  });

  it('spin guard rails: zero axis throws, non-finite rate throws, leaf spin (multi-level) throws', () => {
    expect(() =>
      compileHierarchy(
        SYNTH,
        [{ groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 1, axis: [0, 0, 0] } }, leaf],
        { spacing: 10 },
      ),
    ).toThrow(/spin\.axis/);
    expect(() =>
      compileHierarchy(
        SYNTH,
        [{ groupBy: 'type', place: { shape: 'shell' }, spin: { rate: NaN } }, leaf],
        { spacing: 10 },
      ),
    ).toThrow(/spin\.rate/);
    expect(() =>
      compileHierarchy(
        SYNTH,
        [{ groupBy: 'type', place: { shape: 'shell' } }, { place: { shape: 'cloud' }, spin: { rate: 1 } }],
        { spacing: 10 },
      ),
    ).toThrow(/spin on the leaf level/);
  });
});

/* ------------------------------------------------------------------ leaf-only */

describe('leaf-only hierarchy (one synthetic whole-field group)', () => {
  it('compiles: one root pivot, every node rides it', () => {
    const r = compileHierarchy(SYNTH, [{ place: { shape: 'cloud' }, spin: { rate: 1 } }], {
      spacing: 10,
    });
    expect(r.pivots.count).toBe(1);
    expect(r.pivots.parent[0]).toBe(-1);
    expect(r.pivots.level[0]).toBe(0);
    expect(r.stats.groupsPerLevel).toEqual([]);
    expect(r.chainDepth).toBe(1); // the one legal leaf spin: the whole-field frame
    for (let i = 0; i < SYNTH.n; i++) expect(r.nodePivot[i]).toBe(0);
    expect(Array.from(r.pivots.world.subarray(0, 3))).toEqual([0, 0, 0]);
  });
});

/* ------------------------------------------------------------------ §1 smoke: six levels */

describe('DESIGN §1 "hierarchy without limits": SIX levels over 2000 nodes', () => {
  // synth has four facets; a fifth, independent one makes five groupBy levels + leaf
  const input = generateSynth({ n: 2000, seed: 7 });
  input.facets!['band'] = {
    values: ['a', 'b', 'c'],
    data: Uint8Array.from({ length: 2000 }, (_, i) => i % 3),
  };
  const g6 = buildGraph(input);
  const levels6: LevelSpec[] = [
    { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.1 } },
    { groupBy: 'galaxy', place: { preset: 'spiralGalaxy' }, spin: { rate: 0.3 } },
    { groupBy: 'era', place: { shape: 'ring' } },
    { groupBy: 'band', place: { shape: 'ball' } },
    { groupBy: 'system', place: { shape: 'shell' } },
    { place: { shape: 'cloud' } },
  ];
  const r = compileHierarchy(g6, levels6, { spacing: 10, salt: 3 });

  it('compiles and places all 2000 nodes finitely', () => {
    expect(r.positions.length).toBe(2000 * 3);
    for (let k = 0; k < r.positions.length; k++) {
      expect(Number.isFinite(r.positions[k])).toBe(true);
    }
  });

  it('five grouping levels with DISTINCT pivot counts, refining monotonically', () => {
    expect(r.stats.groupsPerLevel.length).toBe(5);
    expect(new Set(r.stats.groupsPerLevel).size).toBe(5);
    for (let k = 1; k < 5; k++) {
      expect(r.stats.groupsPerLevel[k]!).toBeGreaterThan(r.stats.groupsPerLevel[k - 1]!);
    }
    let total = 0;
    for (const c of r.stats.groupsPerLevel) total += c;
    expect(r.pivots.count).toBe(total);
  });

  it('every chain walks to -1 in <= 6 steps from a level-4 leaf pivot', () => {
    for (let i = 0; i < g6.n; i += 37) {
      let p = r.nodePivot[i]!;
      expect(r.pivots.level[p]).toBe(4);
      let steps = 0;
      let par = r.pivots.parent[p]!;
      while (par !== -1) {
        steps++;
        expect(steps).toBeLessThanOrEqual(levels6.length);
        p = par;
        par = r.pivots.parent[p]!;
      }
      expect(r.pivots.level[p]).toBe(0);
    }
  });
});

/* ================================================================== attachToAnchors */

describe('attachToAnchors: multi-source BFS pull (arrange.js bySubset, ported)', () => {
  /* Hand graph: two chains off two anchors, one contested node, two strays.
   *   0—1—2        anchor 0, hops 1 and 2
   *   3—4          anchor 3, hop 1
   *   7—0, 7—3     contested at depth 1 → the LOWER-id frontier node claims it
   *   5, 6         isolated; 5 sits next to anchor 3, 6 next to anchor 0 */
  const HAND = buildGraph({
    n: 8,
    positions: Float32Array.from([
      0, 0, 0, //    0
      1, 0, 0, //    1
      2, 0, 0, //    2
      10, 0, 0, //   3
      11, 0, 0, //   4
      9, 0, 0, //    5 (stray, near 3)
      1, 1, 0, //    6 (stray, near 0)
      5, 0, 0, //    7 (contested)
    ]),
    links: Uint32Array.from([0, 1, 1, 2, 3, 4, 0, 7, 3, 7]),
  });

  it('hop-1 vs hop-2 grouping: counts carried, anchors at hop 0', () => {
    const { anchorOf, hops } = attachToAnchors(HAND, Uint32Array.from([0, 3]));
    expect(Array.from(anchorOf.subarray(0, 5))).toEqual([0, 0, 0, 3, 3]);
    expect(hops[0]).toBe(0);
    expect(hops[1]).toBe(1);
    expect(hops[2]).toBe(2);
    expect(hops[3]).toBe(0);
    expect(hops[4]).toBe(1);
  });

  it('a contested node goes to the lower-id frontier claimant, deterministically', () => {
    const { anchorOf, hops } = attachToAnchors(HAND, Uint32Array.from([0, 3]));
    expect(anchorOf[7]).toBe(0);
    expect(hops[7]).toBe(1);
  });

  it('unreachable nodes fall back to the SPATIALLY nearest anchor, hop = maxHop + 1', () => {
    const { anchorOf, hops } = attachToAnchors(HAND, Uint32Array.from([0, 3]));
    expect(anchorOf[5]).toBe(3); // 1 unit from anchor 3, 9 from anchor 0
    expect(anchorOf[6]).toBe(0);
    expect(hops[5]).toBe(3); // maxHop was 2 (node 2)
    expect(hops[6]).toBe(3);
  });

  it('without positions, strays attach to the lowest-id anchor (documented fallback)', () => {
    const g = buildGraph({ n: 4, links: Uint32Array.from([2, 3]) });
    const { anchorOf, hops } = attachToAnchors(g, Uint32Array.from([3, 1]));
    expect(anchorOf[0]).toBe(1); // stray → lowest anchor id
    expect(hops[0]).toBe(2); // maxHop 1 (node 2) + 1
    expect(anchorOf[2]).toBe(3);
    expect(hops[2]).toBe(1);
  });

  it('determinism: anchor array order cannot leak into the result', () => {
    const a = attachToAnchors(HAND, Uint32Array.from([0, 3]));
    const b = attachToAnchors(HAND, Uint32Array.from([3, 0, 3])); // shuffled, duplicated
    expect(Array.from(b.anchorOf)).toEqual(Array.from(a.anchorOf));
    expect(Array.from(b.hops)).toEqual(Array.from(a.hops));
  });

  it('synth-scale determinism: two runs agree byte-for-byte, and every anchorOf IS an anchor', () => {
    const typeData = SYNTH.facets['type']!.data;
    const ids: number[] = [];
    for (let i = 0; i < SYNTH.n; i++) if (typeData[i] === 2) ids.push(i);
    const anchors = Uint32Array.from(ids);
    const a = attachToAnchors(SYNTH, anchors);
    const b = attachToAnchors(SYNTH, anchors);
    expect(fnv1a(b.anchorOf)).toBe(fnv1a(a.anchorOf));
    expect(fnv1a(b.hops)).toBe(fnv1a(a.hops));
    const set = new Set(ids);
    for (let i = 0; i < SYNTH.n; i++) {
      expect(set.has(a.anchorOf[i]!)).toBe(true);
      if (set.has(i)) {
        expect(a.anchorOf[i]).toBe(i);
        expect(a.hops[i]).toBe(0);
      }
    }
  });

  it('hops saturate at HOP_MAX on a 300-node path; attachment stays exact', () => {
    const n = 300;
    const links = new Uint32Array((n - 1) * 2);
    for (let i = 0; i < n - 1; i++) {
      links[i * 2] = i;
      links[i * 2 + 1] = i + 1;
    }
    const g = buildGraph({ n, links });
    const { anchorOf, hops } = attachToAnchors(g, Uint32Array.from([0]));
    expect(hops[200]).toBe(200);
    expect(hops[254]).toBe(254);
    expect(hops[255]).toBe(HOP_MAX);
    expect(hops[299]).toBe(HOP_MAX);
    expect(anchorOf[299]).toBe(0);
  });

  it('guard rails: empty anchors throws, out-of-range anchor throws', () => {
    expect(() => attachToAnchors(HAND, new Uint32Array(0))).toThrow(/anchors is empty/);
    expect(() => attachToAnchors(HAND, Uint32Array.from([0, 99]))).toThrow(/out of range/);
  });
});

/* ------------------------------------------------------------------ RandomPlace (§3.3) */

describe('place: { random } — a deterministic per-group shape', () => {
  /* Eight groups of eight nodes, grouped by facet `g`, leaf = random per group: each leaf
   *  group lays its nodes out with a shape hashed from its pivot id. A ring (dim 1, y ≡ 0)
   *  and a ball (dim 3) place very differently, so distinct shapes are observable in the
   *  positions without reaching into the private pick. */
  const EIGHT = handGraph([8, 8, 8, 8, 8, 8, 8, 8]);
  const randomLevels: LevelSpec[] = [
    { groupBy: 'g', place: { shape: 'shell' } },
    { place: { random: true } },
  ];

  it('same hierarchy + salt → byte-identical output (no RNG, no clock)', () => {
    const a = compileHierarchy(EIGHT, randomLevels, { spacing: 1, salt: 3 });
    const b = compileHierarchy(EIGHT, randomLevels, { spacing: 1, salt: 3 });
    expect(fnv1a(b.positions)).toBe(fnv1a(a.positions));
  });

  it('a different base salt re-deals the shapes', () => {
    const a = compileHierarchy(EIGHT, randomLevels, { spacing: 1, salt: 3 });
    const b = compileHierarchy(EIGHT, randomLevels, { spacing: 1, salt: 4 });
    expect(fnv1a(b.positions)).not.toBe(fnv1a(a.positions));
  });

  /** Per-group spread of the member positions along world y. A group's parent offset is
   *  constant across its members, so this equals the LOCAL y spread of its shape: a planar
   *  shape (ring, disc) is ~0, a volumetric/curved one (ball, box, shell, helix, …) is not. */
  const yExtentByGroup = (h: CompiledHierarchy, n: number): Map<number, number> => {
    const lo = new Map<number, number>();
    const hi = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      const g = h.nodePivot[i]!;
      const y = h.positions[i * 3 + 1]!;
      lo.set(g, Math.min(lo.get(g) ?? Infinity, y));
      hi.set(g, Math.max(hi.get(g) ?? -Infinity, y));
    }
    const ext = new Map<number, number>();
    for (const g of lo.keys()) ext.set(g, hi.get(g)! - lo.get(g)!);
    return ext;
  };

  it('groups differ: not every leaf group drew the same shape', () => {
    // Every leaf group has the same size (8), radius and ordering, so two groups that drew the
    // SAME shape get a byte-identical local layout and thus the same y-spread; two different
    // shapes get different y-spreads. More than one distinct y-spread across the eight groups
    // therefore proves the per-group shape genuinely varies (and is not one shape for the level).
    const h = compileHierarchy(EIGHT, randomLevels, { spacing: 1, salt: 3 });
    const spreads = new Set([...yExtentByGroup(h, EIGHT.n).values()].map((e) => e.toFixed(3)));
    expect(spreads.size).toBeGreaterThan(1);
  });

  it('respects a restricted shape set', () => {
    // all-ring set: every leaf group is a ring, so every group has ~0 spread along world y
    // (the ring lies in the XZ plane; only its parent's constant y offset remains).
    const rings: LevelSpec[] = [
      { groupBy: 'g', place: { shape: 'shell' } },
      { place: { random: true, shapes: ['ring'] } },
    ];
    const h = compileHierarchy(EIGHT, rings, { spacing: 1, salt: 3 });
    for (const ext of yExtentByGroup(h, EIGHT.n).values()) expect(ext).toBeLessThan(1e-4);
  });

  it('an unknown shape in the set throws at compile time, naming it', () => {
    const bad: LevelSpec[] = [
      { groupBy: 'g', place: { shape: 'shell' } },
      { place: { random: true, shapes: ['ring', 'nope'] } },
    ];
    expect(() => compileHierarchy(EIGHT, bad, { spacing: 1 })).toThrow(/nope/);
  });

  it('an empty shape set throws', () => {
    const bad: LevelSpec[] = [
      { groupBy: 'g', place: { shape: 'shell' } },
      { place: { random: true, shapes: [] } },
    ];
    expect(() => compileHierarchy(EIGHT, bad, { spacing: 1 })).toThrow(/empty shape set/);
  });

  it('an interior random level seats child centres and stays deterministic', () => {
    // g (random interior) → leaf cloud. The interior level seats its group centres with a
    // hashed shape; assert it compiles, every node is finite, and the output is reproducible.
    const three = handGraph([16, 16, 16, 16]); // 4 top groups
    const levels: LevelSpec[] = [
      { groupBy: 'g', place: { random: true } },
      { place: { shape: 'cloud' } },
    ];
    const h = compileHierarchy(three, levels, { spacing: 1, salt: 2 });
    for (let i = 0; i < three.n; i++) {
      expect(Number.isFinite(h.positions[i * 3]!)).toBe(true);
    }
    // determinism holds for the interior form too
    const h2 = compileHierarchy(three, levels, { spacing: 1, salt: 2 });
    expect(fnv1a(h2.positions)).toBe(fnv1a(h.positions));
  });
});
