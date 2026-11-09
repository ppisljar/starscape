/* The link layer: every link as one segment of ONE batched line list — one draw call for all
 * of them, with the budget as a draw range over the importance-ordered buffer (DESIGN.md §5).
 * Ported from vault-site-3d src/field.js (LINK_VERT / LINK_FRAG) into TSL.
 *
 * Stage 6 (DESIGN §4): the duplicated endpoint copies are GONE. Each vertex carries only the
 * NODE INDICES it concerns — its own endpoint in `position.x`, the opposite endpoint in
 * `position.y` — and pulls the actual coordinates from the shared position store through the
 * same GpuTransform the node layer uses (storage buffers on WebGPU, DataTexture texelFetch on
 * the WebGL2 fallback — the seam lives in ./stores). Morph, frame chains and spread all
 * resolve in the vertex stage, so rotating or morphing the field re-uploads NO link bytes —
 * the predecessor's one design flaw (SCALE.md "Rotation re-uploads the link buffer": 42 ms/f
 * at 10⁵) is structurally impossible here, not merely avoided.
 *
 * Why indices ride in `position`: a BufferGeometry needs a 'position' attribute anyway (it
 * is where every renderer reads the vertex count), so the two ids live there instead of
 * paying for a parallel attribute. Float32 carries integers exactly to 2²⁴ — loudly past the
 * §9 scale ceiling. The baked per-vertex length attribute is gone too: endpoints move under
 * the morph, so the long-link fade measures the CURRENT base-space length (both endpoints
 * are one extra pull away), times the spread — lengths scale with the field.
 *
 * Colours stay per-vertex RGBA exactly as before: the focus state owns every alpha slot and
 * its O(degree) update-range discipline (see field.ts) is untouched by this rewrite.
 */

import {
  BufferAttribute,
  BufferGeometry,
  LineBasicNodeMaterial,
  LineSegments,
  NormalBlending,
} from 'three/webgpu';
import type { Node } from 'three/webgpu';
import {
  attribute,
  clamp,
  int,
  length,
  max,
  mix,
  positionGeometry,
  positionView,
  uniform,
  varying,
  vec4,
  Discard,
  Fn,
} from 'three/tsl';
import { LINK_LEN_FLOOR } from './constants';
import type { GpuTransform } from './stores';
import type { TonemapRig } from './tonemap';

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

/** Build the batched line list. `links` is the importance-ordered 2m buffer from buildGraph
 *  (so the draw-range budget keeps the best prefix by construction); `transform` is the
 *  shared GPU transform both layers pull positions through. The ONLY geometry baked here is
 *  the pair of node ids per vertex — written once, never rewritten. */
export function buildLinkLayer(
  links: Uint32Array,
  transform: GpuTransform,
  opts: {
    fogNear: number;
    fogFar: number;
    lenSoft: number;
    /** The shared highlight roll-off rig (stage 8, ./tonemap) — same instance as the nodes,
     *  so the two draw calls can never roll off differently. */
    tonemap: TonemapRig;
  },
): LinkLayer {
  const m = links.length >> 1;

  /* Vertex v of edge e: x = its own node id, y = the opposite endpoint's id, z unused.
     Baked exactly once — positions are pulled, so nothing here ever goes stale. */
  const ids = new Float32Array(m * 6);
  for (let e = 0; e < m; e++) {
    const a = links[e * 2]!;
    const b = links[e * 2 + 1]!;
    const o = e * 6;
    ids[o] = a;
    ids[o + 1] = b;
    ids[o + 3] = b;
    ids[o + 4] = a;
  }
  const colors = new Float32Array(m * 8); // RGBA per endpoint — per-link alpha lives here

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(ids, 3));
  const colorAttr = new BufferAttribute(colors, 4);
  geo.setAttribute('lColor', colorAttr);

  const uFogNear = uniform(opts.fogNear);
  const uFogFar = uniform(opts.fogFar);
  const uLenSoft = uniform(opts.lenSoft);

  /* Vertex stage: pull both endpoints, place this one. The long-link fade reads the LIVE
     base-space separation scaled by the spread uniform, so a spread field fades like the
     longer field it visually is. The frame CHAIN is deliberately skipped for the
     measurement (one rotate per vertex saved): for endpoints in the SAME frame the chain
     is one shared rigid motion and the base length is exact; a link SPANNING two frames
     oscillates in world length with the spin phase while this fade stays at the
     compile-time separation — an accepted approximation of the §3.5 frame model, not an
     invariant. Fog from view depth, which derives from positionNode automatically. */
  const ownId = int(positionGeometry.x) as unknown as Node<'int'>;
  const otherId = int(positionGeometry.y) as unknown as Node<'int'>;
  const own = transform.worldOf(ownId);
  const liveLen = length(transform.baseOf(ownId).sub(transform.baseOf(otherId))).mul(
    transform.uniforms.spread as unknown as Node<'float'>,
  );

  const lColor = varying(vec4(attribute('lColor', 'vec4')));
  const lenFade = varying(mix(1.0, LINK_LEN_FLOOR, clamp(liveLen.div(uLenSoft), 0.0, 1.0)));
  const vFog = varying(
    clamp(positionView.z.negate().sub(uFogNear).div(max(uFogFar.sub(uFogNear), 1.0)), 0.0, 1.0)
      .oneMinus(),
  );

  const material = new LineBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    blending: NormalBlending, // synced with the nodes by the Field — one ground setting
  });
  material.positionNode = own;
  material.colorNode = Fn(() => {
    const a = lColor.w.mul(lenFade).mul(vFog);
    Discard(a.lessThan(0.003));
    // Stage 8 (§7): the shared highlight roll-off — identity when the op uniform is 0.
    return vec4(opts.tonemap.apply(lColor.xyz as unknown as Node<'vec3'>), a);
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
