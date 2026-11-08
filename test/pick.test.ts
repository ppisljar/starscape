/* Picking, tested where it can be tested: grid.ts imports only the field's pure constants
 * module, so the bugs that made the predecessor's clicks land on nothing are reproducible in
 * Node without a GPU or a browser. The pinned behaviours of vault-site-3d's test/pick.test.mjs
 * are mirrored here — hand-placed geometry with unambiguous expected picks, plus the
 * starscape-specific additions (coverage before depth isolated, nearestDistance vs brute
 * force, determinism, empty field). */
import { describe, expect, it } from 'vitest';
import { buildGrid, pickRay, nearestDistance, PICK_DEFAULTS } from '../src/pick/grid';
import type { Grid, PickOptions, Vec3 } from '../src/pick/grid';
import { hash } from '../src/core/hash';
import { NODE_MAX_PX, NODE_MIN_PX, NODE_SCALE } from '../src/field/constants';

const OPT = {
  hitPx: 9,
  tanHalf: Math.tan(((55 * Math.PI) / 180) / 2),
  halfH: 384,
  sizeScale: 1.15,
  minPx: 2.2,
  maxPx: 260,
  dRef: 400, // irrelevant at falloff 1 (the terms cancel); pickRay requires it regardless
};

interface Field {
  n: number;
  positions: Float32Array;
  sizes: Float32Array;
}

function optsFor(f: Field, extra: Partial<PickOptions> = {}): PickOptions {
  return { ...OPT, sizeOf: (i: number) => f.sizes[i]!, ...extra };
}

// 30 × 30 per sheet = 1,800 nodes, which puts the grid at 8 cells a side — the same order as
// a real corpus rather than the 4 a toy graph gives, and cell size is what decides whether a
// stale index still happens to find things.
const SIDE = 30, GAP = 100, PLANE = SIDE * SIDE;
const MID = Math.floor(SIDE / 2) * SIDE + Math.floor(SIDE / 2); // a node mid-sheet

/** Two sheets of SIDE × SIDE nodes: the front at z = 0, the back at z = -GAP.
 *  Node `i` is in the front sheet; `i + PLANE` is the node directly behind it. */
function makeSheets(): Field {
  const n = PLANE * 2;
  const positions = new Float32Array(n * 3);
  const sizes = new Float32Array(n).fill(12);
  const half = (SIDE - 1) / 2;
  for (let s = 0; s < 2; s++) {
    for (let y = 0; y < SIDE; y++) {
      for (let x = 0; x < SIDE; x++) {
        const i = s * PLANE + y * SIDE + x;
        positions[i * 3] = (x - half) * GAP;
        positions[i * 3 + 1] = (y - half) * GAP;
        positions[i * 3 + 2] = -s * GAP;
      }
    }
  }
  return { n, positions, sizes };
}

/** Look at node `i` head-on from `dist` in front of it (+z), cursor dead on its centre. */
function aim(f: Field, grid: Grid, i: number, dist: number, extra: Partial<PickOptions> = {}): number {
  const origin: Vec3 = {
    x: f.positions[i * 3]!,
    y: f.positions[i * 3 + 1]!,
    z: f.positions[i * 3 + 2]! + dist,
  };
  return pickRay(grid, origin, { x: 0, y: 0, z: -1 }, optsFor(f, extra));
}

/* A sphere shell of nodes — the right fixture for the drift (stale index) case, for two
 * reasons the predecessor learned the hard way: its extent is roughly cubic so no axis is
 * thin (on flat sheets a rotation pushes every node outside the z span, the index clamps them
 * into boundary cells, the lookup clamps to the same place, and a hopelessly stale index
 * still finds everything by accident); and aiming inward along a radius hits exactly one
 * node — its antipode is on the same ray but further along, and the contract is front-most.
 * Fibonacci spiral for the directions: evenly distributed, deterministic, no RNG. */
function makeShell(n = 1200, R = 2000): Field & { R: number; diagonal: number } {
  const positions = new Float32Array(n * 3);
  const sizes = new Float32Array(n).fill(12);
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = golden * i;
    positions[i * 3] = Math.cos(th) * r * R;
    positions[i * 3 + 1] = y * R;
    positions[i * 3 + 2] = Math.sin(th) * r * R;
  }
  return { n, positions, sizes, R, diagonal: 2 * R * Math.sqrt(3) };
}

/** Look inward at node `i` from `gap` outside the shell, aiming at where it IS NOW. The
 *  camera looks at what it can see; only the INDEX is stale. */
function aimIn(f: Field, grid: Grid, i: number, gap: number, extra: Partial<PickOptions> = {}): number {
  const x = f.positions[i * 3]!;
  const y = f.positions[i * 3 + 1]!;
  const z = f.positions[i * 3 + 2]!;
  const L = Math.hypot(x, y, z) || 1;
  const d: Vec3 = { x: x / L, y: y / L, z: z / L };
  const origin: Vec3 = { x: d.x * (L + gap), y: d.y * (L + gap), z: d.z * (L + gap) };
  return pickRay(grid, origin, { x: -d.x, y: -d.y, z: -d.z }, optsFor(f, extra));
}

function rotateY(f: Field, th: number): void {
  const c = Math.cos(th), s = Math.sin(th);
  for (let i = 0; i < f.n; i++) {
    const x = f.positions[i * 3]!;
    const z = f.positions[i * 3 + 2]!;
    f.positions[i * 3] = c * x + s * z;
    f.positions[i * 3 + 2] = -s * x + c * z;
  }
}

describe('buildGrid', () => {
  it('indexes the live positions into non-degenerate CSR cells', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n);
    expect(grid.start[grid.nCells]).toBe(f.n); // every node landed in exactly one cell
    let occupied = 0, biggest = 0;
    for (let c = 0; c < grid.nCells; c++) {
      const m = grid.start[c + 1]! - grid.start[c]!;
      if (m > 0) occupied++;
      if (m > biggest) biggest = m;
    }
    expect(biggest).toBeLessThan(f.n * 0.2);
    expect(occupied).toBeGreaterThan(8);
  });

  it('is deterministic: same positions, byte-identical grid', () => {
    const f = makeSheets();
    const a = buildGrid(f.positions, f.n);
    const b = buildGrid(f.positions, f.n);
    expect(Array.from(a.start)).toEqual(Array.from(b.start));
    expect(Array.from(a.order)).toEqual(Array.from(b.order));
    expect(a.cellMax).toBe(b.cellMax);
  });

  it('honours an explicit cell size', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n, 500);
    // span is ~2900 in x/y, so ~6 cells a side; the edge length lands near the request
    expect(grid.side).toBeGreaterThan(3);
    expect(grid.cellMax).toBeGreaterThan(250);
    expect(grid.cellMax).toBeLessThan(1000);
    expect(aim(f, grid, MID, 900)).toBe(MID);
  });

  it('survives an empty field', () => {
    const grid = buildGrid(new Float32Array(0), 0);
    expect(grid.nCells).toBe(1);
    expect(pickRay(grid, { x: 0, y: 0, z: 5 }, { x: 0, y: 0, z: -1 }, OPT)).toBe(-1);
    expect(nearestDistance(grid, { x: 0, y: 0, z: 0 })).toBe(Infinity);
  });
});

describe('pickRay', () => {
  it('hits the node under the cursor, far away', () => {
    const f = makeSheets();
    expect(aim(f, buildGrid(f.positions, f.n), MID, 5000)).toBe(MID);
  });

  it('hits it at every range, including very close', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n);
    for (const d of [25, 60, 120, 300, 900, 3000]) {
      expect(aim(f, grid, MID, d), `at d=${d}`).toBe(MID);
    }
  });

  // --- COVERAGE BEFORE DEPTH, isolated. A small node nearer the camera sits 20 world units
  //     off the cursor: outside its own drawn disc (r≈13.8 here) but inside the 9 px slop
  //     cone (r≈26). A larger node further along the same ray is dead under the cursor. Depth
  //     alone would pick the near one; the contract picks the one actually covering the pixel.
  function smallNearBigFar(offX: number): { f: Field; grid: Grid } {
    const positions = new Float32Array([offX, 0, 1000, 0, 0, 0]);
    const sizes = new Float32Array([12, 40]);
    const f = { n: 2, positions, sizes };
    return { f, grid: buildGrid(positions, f.n) };
  }
  const RO: Vec3 = { x: 0, y: 0, z: 2000 };
  const RD: Vec3 = { x: 0, y: 0, z: -1 };

  it('a covering node beats a nearer node that is only inside the slop cone', () => {
    const { f, grid } = smallNearBigFar(20);
    expect(pickRay(grid, RO, RD, optsFor(f))).toBe(1); // farther-but-covering wins
  });

  it('depth breaks the tie when both discs cover the cursor', () => {
    const { f, grid } = smallNearBigFar(5); // 5 < rHit ≈ 13.8: the near disc covers too
    expect(pickRay(grid, RO, RD, optsFor(f))).toBe(0); // closer-but-smaller wins
  });

  it('the slop-only node is still returned when nothing covers', () => {
    const positions = new Float32Array([20, 0, 1000]);
    const sizes = new Float32Array([12]);
    const f = { n: 1, positions, sizes };
    expect(pickRay(buildGrid(positions, 1), RO, RD, optsFor(f))).toBe(0);
  });

  it('a filtered-out node cannot be picked, and reveals the one behind it', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n);
    expect(aim(f, grid, MID, 900, { isVisible: (i) => i !== MID })).toBe(MID + PLANE);
  });

  /* Falloff, isolated the way the predecessor pinned it: `hitPx: 0` removes the forgiveness
   * cone (9 px at 7,000 units is 106 world units — far larger than the disc, so with it on
   * the test would measure the slop, not the disc), and one lone node removes any question
   * of a neighbour winning instead. */
  it('the hit disc tracks the drawn disc when falloff is not 1', () => {
    const positions = new Float32Array([0, 0, 0]);
    const sizes = new Float32Array([12]);
    const f = { n: 1, positions, sizes };
    const grid = buildGrid(positions, 1);
    const ro: Vec3 = { x: 40, y: 0, z: 7000 };
    const rd: Vec3 = { x: 0, y: 0, z: -1 };
    expect(
      pickRay(grid, ro, rd, optsFor(f, { hitPx: 0, dRef: 400, falloff: 1 })),
      '40 units off should fall outside a 2.2 px disc at true perspective',
    ).toBe(-1);
    expect(
      pickRay(grid, ro, rd, optsFor(f, { hitPx: 0, dRef: 400, falloff: 0.15 })),
      'and inside it once a low falloff draws the glyph much larger',
    ).toBe(0);
  });

  /* One pixel unit, one set of constants (DESIGN.md §5): the picker's sizing defaults must
   * BE the shader's — the predecessor's inline fallbacks (0.62 / 1.4 / 190) gave a default-
   * relying consumer a hit disc ~54% of the drawn disc. */
  it('PICK_DEFAULTS are the field material constants, not a second copy', () => {
    expect(PICK_DEFAULTS.sizeScale).toBe(NODE_SCALE);
    expect(PICK_DEFAULTS.minPx).toBe(NODE_MIN_PX);
    expect(PICK_DEFAULTS.maxPx).toBe(NODE_MAX_PX);
  });

  /* The shader scales the glyph radius AND its clamp rails by iBoost, so a hovered node
   * pinned at a rail still grows — and its hit disc must grow with it. hitPx: 0 so the test
   * measures the disc, not the slop; the node here is pinned at NODE_MIN_PX, the rail where
   * "boost scales the radius only" would change nothing at all. */
  it('a boosted node is hit on the annulus the boost added to its drawn disc', () => {
    const positions = new Float32Array([0, 0, 0]);
    const sizes = new Float32Array([12]);
    const f = { n: 1, positions, sizes };
    const grid = buildGrid(positions, 1);
    const ro: Vec3 = { x: 25, y: 0, z: 7000 }; // resting rHit ≈ 20.9, boosted ≈ 30.3
    const rd: Vec3 = { x: 0, y: 0, z: -1 };
    expect(
      pickRay(grid, ro, rd, optsFor(f, { hitPx: 0 })),
      '25 units off falls outside the resting 2.2 px disc',
    ).toBe(-1);
    expect(
      pickRay(grid, ro, rd, optsFor(f, { hitPx: 0, boostOf: () => 1.45 })),
      'and inside the hover-boosted disc (rails scale with the boost)',
    ).toBe(0);
  });

  it('still picks with the camera far beyond the field diagonal', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n);
    const diagonal = Math.hypot((SIDE - 1) * GAP, (SIDE - 1) * GAP, GAP);
    expect(aim(f, grid, MID, diagonal * 6)).toBe(MID);
  });

  it('returns -1 for a ray that misses the field', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n);
    expect(pickRay(grid, { x: 0, y: 0, z: 5000 }, { x: 0, y: 1, z: 0 }, optsFor(f))).toBe(-1);
  });

  it('is deterministic: the same ray picks the same node', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n);
    const a = aim(f, grid, MID, 900);
    const b = aim(f, grid, MID, 900);
    expect(a).toBe(b);
    expect(a).toBe(MID);
  });

  /* THE REBUILD STORY (DESIGN §6): the grid is rebuilt on morph completion, not per frame,
   * so while the field turns the index is stale. Rather than asserting the stale bug
   * reproduces at one probe — a node near the rotation axis barely moves and is found either
   * way — measure across the whole shell: a stale index loses some picks, and declaring the
   * movement bound as `drift` recovers every one of them. */
  function shellHitRate(f: Field, grid: Grid, drift: number): number {
    let hits = 0;
    for (let i = 0; i < f.n; i++) if (aimIn(f, grid, i, 1500, { drift }) === i) hits++;
    return hits;
  }

  it('declaring the drift recovers every pick a rotated field loses', () => {
    for (const th of [0.45, 1.2]) {
      const f = makeShell();
      const grid = buildGrid(f.positions, f.n); // index it, THEN turn it
      rotateY(f, th);
      const drift = th * f.diagonal * 0.5; // the bound a frame loop accumulates
      const without = shellHitRate(f, grid, 0);
      const withIt = shellHitRate(f, grid, drift);
      // The stale effect is real but partial — most nodes have not crossed a cell boundary
      // yet, which is exactly why this presented as "clicking sometimes does not work".
      expect(f.n - without, `stale index should lose some picks at ${th} rad`).toBeGreaterThan(0);
      expect(withIt, `declared drift should lose nothing at ${th} rad`).toBe(f.n);
    }
  });
});

describe('nearestDistance', () => {
  it('finds the true nearest node on a hand-placed field', () => {
    const f = makeSheets();
    const grid = buildGrid(f.positions, f.n);
    const d = nearestDistance(grid, {
      x: f.positions[MID * 3]! + 30,
      y: f.positions[MID * 3 + 1]!,
      z: f.positions[MID * 3 + 2]!,
    });
    expect(d).toBeCloseTo(30, 3);
  });

  // The flycam's roomAt contract: nearest node ANYWHERE, exact, from any probe point —
  // inside the cloud, at its edge, and far outside the indexed box.
  it('matches brute force on a hash-generated 1000-node cloud', () => {
    const n = 1000;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = hash(i, 1) * 1000;
      positions[i * 3 + 1] = hash(i, 2) * 1000;
      positions[i * 3 + 2] = hash(i, 3) * 1000;
    }
    const grid = buildGrid(positions, n);

    const probes: Vec3[] = [];
    for (let p = 0; p < 24; p++) {
      probes.push({
        x: hash(p, 11) * 2000 - 500, // spans [-500, 1500): inside, edge, and outside
        y: hash(p, 12) * 2000 - 500,
        z: hash(p, 13) * 2000 - 500,
      });
    }
    probes.push({ x: 500, y: 500, z: 500 }); // dead centre
    probes.push({ x: 5000, y: 5000, z: 5000 }); // far outside — the "anywhere" case

    for (const p of probes) {
      let best = Infinity;
      for (let i = 0; i < n; i++) {
        const dx = positions[i * 3]! - p.x;
        const dy = positions[i * 3 + 1]! - p.y;
        const dz = positions[i * 3 + 2]! - p.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) best = d2;
      }
      expect(nearestDistance(grid, p)).toBeCloseTo(Math.sqrt(best), 6);
    }
  });

  /* The §9 regression this walk is shaped around: the flycam probes roomAt once or twice
   * EVERY FRAME, and at the home framing the camera sits well outside the indexed box. The
   * cube-per-ring walk this replaced distance-tested essentially the whole field there —
   * repeatedly — which at 10⁶ nodes is tens of ms inside a <1 ms budget. The shell walk
   * with per-cell pruning must touch a small corner of the field and stop. */
  it('does not degenerate to full-field scans from far outside the box', () => {
    const n = 1000;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = hash(i, 1) * 1000;
      positions[i * 3 + 1] = hash(i, 2) * 1000;
      positions[i * 3 + 2] = hash(i, 3) * 1000;
    }
    const grid = buildGrid(positions, n);
    let tested = 0;
    const count = (_i: number): boolean => {
      tested++;
      return true;
    };
    const d = nearestDistance(grid, { x: 5000, y: 5000, z: 5000 }, { isVisible: count });
    expect(d).toBeGreaterThan(0); // it still finds the true nearest (pinned above vs brute force)
    expect(tested).toBeLessThan(n / 2); // …without sweeping the field to do it
  });

  it('respects the visibility predicate', () => {
    const positions = new Float32Array([0, 0, 0, 100, 0, 0]);
    const grid = buildGrid(positions, 2);
    expect(nearestDistance(grid, { x: 10, y: 0, z: 0 })).toBeCloseTo(10, 6);
    expect(nearestDistance(grid, { x: 10, y: 0, z: 0 }, { isVisible: (i) => i !== 0 }))
      .toBeCloseTo(90, 6);
    expect(nearestDistance(grid, { x: 10, y: 0, z: 0 }, { isVisible: () => false }))
      .toBe(Infinity);
  });

  it('stays correct against a stale index when drift is declared', () => {
    const f = makeShell(400, 1000);
    const grid = buildGrid(f.positions, f.n);
    rotateY(f, 0.6);
    const drift = 0.6 * f.diagonal * 0.5;
    // probe just outside a moved node: brute force vs grid with the bound declared
    const i = 123;
    const p: Vec3 = {
      x: f.positions[i * 3]! * 1.02,
      y: f.positions[i * 3 + 1]! * 1.02,
      z: f.positions[i * 3 + 2]! * 1.02,
    };
    let best = Infinity;
    for (let j = 0; j < f.n; j++) {
      const dx = f.positions[j * 3]! - p.x;
      const dy = f.positions[j * 3 + 1]! - p.y;
      const dz = f.positions[j * 3 + 2]! - p.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) best = d2;
    }
    expect(nearestDistance(grid, p, { drift })).toBeCloseTo(Math.sqrt(best), 6);
  });

  it('is deterministic', () => {
    const f = makeShell(200, 500);
    const grid = buildGrid(f.positions, f.n);
    const p: Vec3 = { x: 50, y: -20, z: 700 };
    expect(nearestDistance(grid, p)).toBe(nearestDistance(grid, p));
  });
});
