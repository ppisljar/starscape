/* The pure halves of the field's visual channels — everything here runs headless (vitest, node)
 * with no GPU and no three.js import. The TSL materials consume the buffers these produce.
 *
 * House rules honoured throughout (DESIGN.md §10): loops over node-length arrays, never
 * spreads; typed arrays only, no per-node objects; everything a pure function of its inputs.
 */

import {
  FALLOFF,
  NODE_MAX_PX,
  NODE_MIN_PX,
  SIZE_CURVE,
  SIZE_MAX_WORLD,
  SIZE_MIN_WORLD,
  SIZE_UNIFORM_WORLD,
} from './constants';

/** Rank-normalise a numeric column: out[i] = percentile of node i in [0, 1], ties broken by
 *  index (ascending), so the result is deterministic whatever the sort engine does with equal
 *  keys. Rank, not a linear map: words span 177..7,486 and rank spans four orders of magnitude
 *  on a real corpus, so a linear map leaves almost every node on the floor; ranking makes the
 *  size knob mean the same thing whichever column is behind it (ported predecessor reasoning).
 *
 *  Uint32Array for the order, not Array.from(...).sort — at 10⁶ a boxed index array costs ~8 MB
 *  and an unbox per compare. */
export function rankPercentiles(data: ArrayLike<number>, n: number): Float32Array {
  const out = new Float32Array(n);
  if (n === 0) return out;
  if (n === 1) {
    out[0] = 1; // the predecessor's pct = r/(n-1) convention: a lone node is the top of its rank
    return out;
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => data[a]! - data[b]! || a - b);
  const inv = 1 / (n - 1);
  for (let r = 0; r < n; r++) out[order[r]!] = r * inv;
  return out;
}

/** The size channel: world radii swept MIN→MAX by percentile^curve. One output buffer the
 *  caller owns; call again with another column to switch channels (what is big changes, how
 *  aggressively bigness is drawn does not). */
export function sizesFromColumn(
  data: ArrayLike<number>,
  n: number,
  curve: number = SIZE_CURVE,
): Float32Array {
  const pct = rankPercentiles(data, n);
  const span = SIZE_MAX_WORLD - SIZE_MIN_WORLD;
  for (let i = 0; i < n; i++) pct[i] = SIZE_MIN_WORLD + span * Math.pow(pct[i]!, curve);
  return pct;
}

/** The 'uniform' size channel: every node the same radius. */
export function uniformSizes(n: number): Float32Array {
  const out = new Float32Array(n);
  out.fill(SIZE_UNIFORM_WORLD);
  return out;
}

/** A glyph's projected radius in CSS px, UNCLAMPED — the exact expression the vertex stage
 *  evaluates, kept here so the picker (stage 4) and the zoom ladder (stage 7) measure the same
 *  thing the shader draws. One pixel unit, everywhere.
 *
 *  `halfHCss` is HALF the viewport height in CSS px; `tanHalf` is tan(fov/2); `depth` is the
 *  view-space distance (−mv.z); `dRef` pivots the falloff so turning that knob does not resize
 *  what is right in front of you. At falloff = 1 this is exactly the real projection:
 *  r·halfH / (tanHalf·d). */
export function glyphPx(
  sizeWorld: number,
  depth: number,
  halfHCss: number,
  tanHalf: number,
  dRef: number,
  falloff: number = FALLOFF,
): number {
  const d = Math.max(depth, 0.001);
  const base = (sizeWorld * halfHCss) / (tanHalf * dRef);
  return base * Math.pow(dRef / d, falloff);
}

/** The CSS-px clamp applied after projection (same constants the shader uses). */
export function clampGlyphPx(px: number): number {
  return Math.min(NODE_MAX_PX, Math.max(NODE_MIN_PX, px));
}

/** Link-budget draw-range maths: how many of the importance-ordered links to draw for a budget
 *  of k links (k is clamped into [0, total]). The buffer is non-indexed line-list vertices, two
 *  per link, so `setDrawRange(0, 2 * linkDrawCount(...))`. Because the buffer is ordered by
 *  worth (intra-group before inter, short before long — stage 2's orderLinks), "the first k"
 *  IS "the best k": the whole budget is a draw range, no second buffer, no per-frame sort. */
export function linkDrawCount(total: number, k: number): number {
  const kk = Math.round(k);
  return kk < 0 ? 0 : kk > total ? total : kk;
}

/** Vertices for `setDrawRange` over a non-indexed line list drawing `links` links. */
export function linkDrawVertices(links: number): number {
  return links * 2;
}

/** The live extent of a position buffer — centre, bounding radius about that centre, and the
 *  axis-aligned diagonal. One loop pass; fit the camera (and fog) to THIS, never to a stored
 *  extent (DESIGN.md §10.8). */
export function fieldExtent(
  positions: Float32Array,
  n: number,
): { centre: [number, number, number]; radius: number; diagonal: number } {
  if (n === 0) return { centre: [0, 0, 0], radius: 0, diagonal: 0 };
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3]!,
      y = positions[i * 3 + 1]!,
      z = positions[i * 3 + 2]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2,
    cy = (minY + maxY) / 2,
    cz = (minZ + maxZ) / 2;
  let r2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = positions[i * 3]! - cx,
      dy = positions[i * 3 + 1]! - cy,
      dz = positions[i * 3 + 2]! - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r2) r2 = d2;
  }
  const dx = maxX - minX,
    dy = maxY - minY,
    dz = maxZ - minZ;
  return {
    centre: [cx, cy, cz],
    radius: Math.sqrt(r2),
    diagonal: Math.sqrt(dx * dx + dy * dy + dz * dz),
  };
}
