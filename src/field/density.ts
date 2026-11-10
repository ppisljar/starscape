/* Local spatial DENSITY as a per-node channel (DESIGN.md §5 theming, §10 house rules) — the
 * nebula look's defining signal. Where setColorColumn reads a DATA column (degree, words) and
 * setColorFacet a category, this reads GEOMETRY: how crowded each node's neighbourhood is in the
 * current positions. Crowded neighbourhoods read hot, sparse ones cool, once the counts are
 * rank-normalised through a cool→hot ramp (the field colours by this via rampColorsTied). This is
 * what makes an isolated-but-bright node BLUE — its colour tracks the emptiness around it, not its
 * own luminance or its degree.
 *
 * Why a geometric channel and not degree: additive blending already sums per-point hues, so in any
 * dense pile every colour tends toward white/orange regardless of each point's hue — a degree ramp
 * cannot paint the blue→orange SPATIAL gradient the reference nebula has, because the signal driving
 * the gradient (local spatial density) was never computed. This module computes it.
 *
 * BOUNDED and DETERMINISTIC (§10). One uniform grid over the positions (the pick/grid approach,
 * reused in spirit — extent in one O(n) pass, counting the occupancy of CSR-style cells, no
 * arrays-of-arrays, no per-node objects), then a 3×3×3 block-occupancy sum per cell as the local
 * density estimate. Cost is O(n + nCells·27) WHATEVER the clustering — a per-node fixed-radius
 * count instead would degrade to O(Σ occupancyₖ²) and spike in a galaxy core where one cell holds
 * thousands; block sums never do. No Math.random, no Date.now: identical positions → identical
 * counts, forever (so screenshots and golden tests reproduce).
 */

/** Average nodes per cell the grid aims for. ~2 sizes cells near the mean nearest-neighbour
 *  spacing (DESIGN's suggestion), so the 3×3×3 block is a tight local neighbourhood and the
 *  core/periphery contrast stays sharp rather than smeared across coarse cells — while halving
 *  the cell count (and the per-call allocations + the block sweep) versus one-per-cell, which is
 *  what keeps the recompute near the tens-of-ms the channel aims for at 1M. */
export const DENSITY_TARGET_PER_CELL = 2;

function clampCell(v: number, side: number): number {
  const c = v | 0;
  return c < 0 ? 0 : c >= side ? side - 1 : c;
}

export interface DensityOpts {
  /** Average nodes per cell (default {@link DENSITY_TARGET_PER_CELL}). Smaller → finer cells,
   *  sharper density contrast, more empty cells to sweep; larger → smoother, cheaper. */
  targetPerCell?: number;
}

/**
 * Per-node local density: for each node, the number of nodes occupying the 3×3×3 block of grid
 * cells centred on its own cell (itself included). Returned as raw counts (float) — the caller
 * rank-normalises them through a ramp (rampColorsTied) exactly as setColorColumn rank-normalises a
 * data column. Writes into `out` when it is at least `n` long (the field reuses one scratch buffer
 * across recomputes, DESIGN §10.2), else allocates.
 *
 * @param positions 3n floats, the geometry to measure (the field feeds its CURRENT target base).
 */
export function localDensity(
  positions: Float32Array,
  n: number,
  out?: Float32Array,
  opts: DensityOpts = {},
): Float32Array {
  const res = out !== undefined && out.length >= n ? out : new Float32Array(n);
  if (n === 0) return res;
  if (n === 1) {
    res[0] = 1; // a lone node: its own cell holds exactly itself
    return res;
  }

  const pad = 1e-3;
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3]!;
    const y = positions[i * 3 + 1]!;
    const z = positions[i * 3 + 2]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) {
    // all-NaN / empty guard: nothing to measure, every node equally dense
    res.fill(1, 0, n);
    return res;
  }
  const spanX = Math.max(maxX - minX, 2 * pad);
  const spanY = Math.max(maxY - minY, 2 * pad);
  const spanZ = Math.max(maxZ - minZ, 2 * pad);

  const target = opts.targetPerCell && opts.targetPerCell > 0 ? opts.targetPerCell : DENSITY_TARGET_PER_CELL;
  const side = Math.max(1, Math.round(Math.cbrt(Math.max(1, n / target))));
  const cellX = spanX / side;
  const cellY = spanY / side;
  const cellZ = spanZ / side;
  const nCells = side * side * side;

  /* occupancy per cell + each node's cell key — counting-sort style, typed arrays only. */
  const occ = new Uint32Array(nCells);
  const cellOf = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const ix = clampCell((positions[i * 3]! - minX) / cellX, side);
    const iy = clampCell((positions[i * 3 + 1]! - minY) / cellY, side);
    const iz = clampCell((positions[i * 3 + 2]! - minZ) / cellZ, side);
    const k = (iz * side + iy) * side + ix;
    cellOf[i] = k;
    occ[k] = occ[k]! + 1;
  }

  /* 3×3×3 block-occupancy per cell: the local density every member of that cell shares. The
     bounds-clamped neighbour walk is O(nCells·27), independent of how the nodes pile up. */
  const block = new Float32Array(nCells);
  for (let z = 0; z < side; z++) {
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        let sum = 0;
        const zlo = z > 0 ? z - 1 : 0;
        const zhi = z < side - 1 ? z + 1 : side - 1;
        const ylo = y > 0 ? y - 1 : 0;
        const yhi = y < side - 1 ? y + 1 : side - 1;
        const xlo = x > 0 ? x - 1 : 0;
        const xhi = x < side - 1 ? x + 1 : side - 1;
        for (let zz = zlo; zz <= zhi; zz++) {
          for (let yy = ylo; yy <= yhi; yy++) {
            const row = (zz * side + yy) * side;
            for (let xx = xlo; xx <= xhi; xx++) sum += occ[row + xx]!;
          }
        }
        block[(z * side + y) * side + x] = sum;
      }
    }
  }

  for (let i = 0; i < n; i++) res[i] = block[cellOf[i]!]!;
  return res;
}
