/* The link layer: every link as one segment of ONE batched line list — one draw call for all
 * of them, with the budget as a draw range over the importance-ordered buffer (DESIGN.md §5).
 * Ported from vault-site-3d src/field.js (LINK_VERT / LINK_FRAG) into TSL.
 *
 * Endpoint positions are a STATIC COPY of the node position buffer, baked once at boot. That
 * is deliberate and temporary: the layout is frozen at stage 3, so there is nothing to
 * re-upload, ever. DESIGN.md §4's vertex pulling (links reading node positions from a shared
 * storage buffer/texture, so no link rewrite exists at all) arrives at stage 6 with the GPU
 * morph — not here.
 */

import {
  BufferAttribute,
  BufferGeometry,
  LineBasicNodeMaterial,
  LineSegments,
  NormalBlending,
} from 'three/webgpu';
import {
  attribute,
  clamp,
  max,
  mix,
  positionView,
  uniform,
  varying,
  vec4,
  Discard,
  Fn,
} from 'three/tsl';
import { LINK_LEN_FLOOR } from './constants';

export interface LinkLayer {
  lines: LineSegments;
  material: LineBasicNodeMaterial;
  /** Per-vertex RGBA (two vertices per link) — rewritten when the colour facet changes. */
  colors: Float32Array;
  colorAttr: BufferAttribute;
  uniforms: {
    fogNear: { value: number };
    fogFar: { value: number };
    /** World length over which a link's alpha eases to LINK_LEN_FLOOR of itself. */
    lenSoft: { value: number };
  };
  /** Draw the first k links of the ordered buffer (vertices = 2k). */
  setDrawCount(links: number): void;
  dispose(): void;
}

/** Bake the batched line list: positions and lengths once (frozen layout), colours whenever
 *  the palette changes. `links` is the importance-ordered 2m buffer from buildGraph, so the
 *  draw-range budget keeps the best prefix by construction. */
export function buildLinkLayer(
  links: Uint32Array,
  nodePositions: Float32Array,
  opts: { fogNear: number; fogFar: number; lenSoft: number },
): LinkLayer {
  const m = links.length >> 1;

  /* Positions and per-vertex lengths, baked exactly once — see the header comment for why a
     static copy is correct at this stage. `lLen` is the edge's world length: cost and clutter
     both track total edge PIXEL-length rather than edge count, so fading the long ones is the
     single cheapest thing that makes a dense graph legible. Rotation never changes a length,
     so baking is exact, not approximate. */
  const lpos = new Float32Array(m * 6);
  const llen = new Float32Array(m * 2);
  for (let e = 0; e < m; e++) {
    const a = links[e * 2]! * 3;
    const b = links[e * 2 + 1]! * 3;
    const ax = nodePositions[a]!,
      ay = nodePositions[a + 1]!,
      az = nodePositions[a + 2]!;
    const bx = nodePositions[b]!,
      by = nodePositions[b + 1]!,
      bz = nodePositions[b + 2]!;
    const o = e * 6;
    lpos[o] = ax;
    lpos[o + 1] = ay;
    lpos[o + 2] = az;
    lpos[o + 3] = bx;
    lpos[o + 4] = by;
    lpos[o + 5] = bz;
    const len = Math.hypot(ax - bx, ay - by, az - bz);
    llen[e * 2] = len;
    llen[e * 2 + 1] = len;
  }
  const colors = new Float32Array(m * 8); // RGBA per endpoint — per-link alpha lives here

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(lpos, 3));
  geo.setAttribute('lLen', new BufferAttribute(llen, 1));
  const colorAttr = new BufferAttribute(colors, 4);
  geo.setAttribute('lColor', colorAttr);

  const uFogNear = uniform(opts.fogNear);
  const uFogFar = uniform(opts.fogFar);
  const uLenSoft = uniform(opts.lenSoft);

  /* Vertex stage: long-link fade (baked length, a mix toward LINK_LEN_FLOOR) and fog from
     view depth. Fragment: colour through, discard what would not be visible anyway. */
  const lColor = varying(vec4(attribute('lColor', 'vec4')));
  const lLen = attribute('lLen', 'float');
  const lenFade = varying(mix(1.0, LINK_LEN_FLOOR, clamp(lLen.div(uLenSoft), 0.0, 1.0)));
  const vFog = varying(
    clamp(positionView.z.negate().sub(uFogNear).div(max(uFogFar.sub(uFogNear), 1.0)), 0.0, 1.0)
      .oneMinus(),
  );

  const material = new LineBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    blending: NormalBlending, // synced with the nodes by the Field — one ground setting
  });
  material.colorNode = Fn(() => {
    const a = lColor.w.mul(lenFade).mul(vFog);
    Discard(a.lessThan(0.003));
    return vec4(lColor.xyz, a);
  })();

  const lines = new LineSegments(geo, material);
  lines.frustumCulled = false;
  lines.renderOrder = 0;

  return {
    lines,
    material,
    colors,
    colorAttr,
    uniforms: { fogNear: uFogNear, fogFar: uFogFar, lenSoft: uLenSoft },
    setDrawCount(count: number) {
      geo.setDrawRange(0, count * 2);
    },
    dispose() {
      geo.dispose();
      material.dispose();
    },
  };
}
