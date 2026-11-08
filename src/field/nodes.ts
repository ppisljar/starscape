/* The node layer: every node as one instanced camera-facing quad — ONE draw call however many
 * nodes there are. Ported from vault-site-3d src/field.js (NODE_VERT / NODE_FRAG), translated
 * from raw GLSL to TSL so one material codebase compiles to WGSL on WebGPU and falls back to
 * GLSL/WebGL2 (DESIGN.md §5).
 *
 * What is ported at stage 3: the view-space billboard sized by the real projection (world
 * radius → projected CSS px → clamped → back to a world radius at this depth), the glyph
 * distance-field families, the pixel-unit halo with a solid body, the hot core, the fog, and
 * the cheap flat branch under FLAT_PX. Stage 4 adds the predecessor's two focus channels:
 * iAlpha (selection dim: 1 lit, dimOthers dimmed) and iBoost (hover/selection size emphasis —
 * it scales the glyph AND its clamp bounds, so an emphasised node grows even when pinned at
 * NODE_MIN_PX/NODE_MAX_PX, which is most of a distant field). What waits: rim bubbles, sphere
 * shading and the full theming (stage 8).
 */

import {
  Float32BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  NormalBlending,
  SpriteNodeMaterial,
} from 'three/webgpu';
import {
  Fn,
  If,
  attribute,
  clamp,
  float,
  length,
  max,
  min,
  mix,
  modelViewMatrix,
  pow,
  select,
  smoothstep,
  uniform,
  uv,
  varying,
  vec3,
  vec4,
  Discard,
} from 'three/tsl';
import {
  BODY_R,
  CORE_HOT,
  FALLOFF,
  FLAT_PX,
  HALO_GAIN,
  HALO_PX,
  NODE_MAX_PX,
  NODE_MIN_PX,
  NODE_SCALE,
} from './constants';

export interface NodeLayer {
  mesh: Mesh;
  material: SpriteNodeMaterial;
  /** Per-instance colour buffer (3n) — written by setColorFacet, flagged via colorAttr. */
  colors: Float32Array;
  colorAttr: InstancedBufferAttribute;
  /** Per-instance world-radius buffer (n) — written by setSizeColumn. */
  sizes: Float32Array;
  sizeAttr: InstancedBufferAttribute;
  shapeAttr: InstancedBufferAttribute;
  /** Per-instance focus alpha (n) — written by the focus state on selection change. */
  alphaAttr: InstancedBufferAttribute;
  /** Per-instance hover/selection boost (n) — written by the focus state on hover change. */
  boostAttr: InstancedBufferAttribute;
  uniforms: {
    /** HALF the viewport height, CSS px (one pixel unit — see constants.ts). */
    halfH: { value: number };
    /** tan(fov / 2). */
    tanHalf: { value: number };
    /** Falloff pivot distance, world units. */
    dRef: { value: number };
    fogNear: { value: number };
    fogFar: { value: number };
  };
  dispose(): void;
}

/** Build the instanced node quad. `positions` is the frozen 3n layout (uploaded once);
 *  `colors`/`sizes`/`shapes`/`alphas`/`boosts` are per-instance channels the Field (and its
 *  focus state) mutates in place. */
export function buildNodeLayer(
  n: number,
  positions: Float32Array,
  colors: Float32Array,
  sizes: Float32Array,
  shapes: Float32Array,
  alphas: Float32Array,
  boosts: Float32Array,
  opts: { fov: number; heightCss: number; dRef: number; fogNear: number; fogFar: number },
): NodeLayer {
  /* ---- geometry: one unit quad, instanced n times. Per-instance data lives in instanced
     attributes on the geometry — the step mode both backends read off the attribute class. */
  const geo = new InstancedBufferGeometry();
  geo.setAttribute(
    'position',
    new Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
  );
  geo.setAttribute('uv', new Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  geo.instanceCount = n;

  const posAttr = new InstancedBufferAttribute(positions, 3);
  const colorAttr = new InstancedBufferAttribute(colors, 3);
  const sizeAttr = new InstancedBufferAttribute(sizes, 1);
  const shapeAttr = new InstancedBufferAttribute(shapes, 1);
  const alphaAttr = new InstancedBufferAttribute(alphas, 1);
  const boostAttr = new InstancedBufferAttribute(boosts, 1);
  geo.setAttribute('iPos', posAttr);
  geo.setAttribute('iColor', colorAttr);
  geo.setAttribute('iSize', sizeAttr);
  geo.setAttribute('iShape', shapeAttr);
  geo.setAttribute('iAlpha', alphaAttr);
  geo.setAttribute('iBoost', boostAttr);

  /* ---- uniforms. CSS pixels throughout; devicePixelRatio is the renderer's business, once. */
  const uHalfH = uniform(opts.heightCss / 2);
  const uTanHalf = uniform(Math.tan((opts.fov * Math.PI) / 360));
  const uDRef = uniform(opts.dRef);
  const uFogNear = uniform(opts.fogNear);
  const uFogFar = uniform(opts.fogFar);

  /* ---- vertex stage: world radius → projected px → clamp → world radius at this depth.
     The clamp is done from the REAL projection, so it is exact rather than an approximation
     of itself. uFalloff bends the perspective about the fixed pivot uDRef (1 = true
     projection). SpriteNodeMaterial supplies the billboard: it offsets the quad corner in
     view space by scaleNode, exactly the predecessor's `mv.xy += position.xy * (r * 2)`. */
  const iPos = attribute('iPos', 'vec3');
  const iSize = attribute('iSize', 'float');
  const iBoost = attribute('iBoost', 'float');
  const centerView = modelViewMatrix.mul(vec4(iPos, 1.0));
  const depth = max(centerView.z.negate(), 0.001);
  /* Boost scales the world radius AND both clamp bounds (predecessor's NODE_VERT: `uMinPx *
     iBoost`, `uMaxPx * iBoost`) — most of a distant field sits pinned at NODE_MIN_PX, where
     scaling only the radius would make hover invisible exactly when it is most needed. */
  const rw = iSize.mul(NODE_SCALE).mul(iBoost);
  const base = rw.mul(uHalfH).div(uTanHalf.mul(uDRef));
  const px = base.mul(pow(uDRef.div(depth), FALLOFF));
  const pxc = clamp(px, iBoost.mul(NODE_MIN_PX), iBoost.mul(NODE_MAX_PX));
  const rWorld = pxc.mul(uTanHalf).mul(depth).div(uHalfH);

  /* Varyings: computed per vertex, constant across the quad (so interpolation is a no-op). */
  const vPx = varying(pxc);
  const vFog = varying(
    clamp(depth.sub(uFogNear).div(max(uFogFar.sub(uFogNear), 1.0)), 0.0, 1.0).oneMinus(),
  );
  const vColor = varying(vec3(attribute('iColor', 'vec3')));
  const vShape = varying(float(attribute('iShape', 'float')));
  const vAlpha = varying(float(attribute('iAlpha', 'float')));
  const vBoost = varying(float(iBoost));

  const material = new SpriteNodeMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: NormalBlending, // the Field syncs blending with the links — one ground setting
  });
  material.positionNode = iPos;
  material.scaleNode = rWorld.mul(2.0);

  /* ---- fragment stage. The distance metric IS the shape: one length() plus a couple of
     abs()es buys seven glyph families with no texture fetch (predecessor's formDist). All
     seven fields are cheap ALU, so a select chain (both sides evaluated) costs less than a
     real branch would save. */
  material.colorNode = Fn(() => {
    const p = uv().sub(0.5); // quad coordinate in [-0.5, 0.5], predecessor's vUv

    const out = vec4(0.0).toVar();

    If(vPx.lessThan(FLAT_PX), () => {
      /* LOD, the cheapest available: under a couple of px every glyph is the same blob and
         the halo is the expensive half of it. One flat colour, no smoothsteps, no form test. */
      const rr = length(p).mul(2.0);
      Discard(rr.greaterThan(1.0));
      const aa = vFog.mul(vAlpha);
      Discard(aa.lessThan(0.004));
      out.assign(vec4(vColor, aa));
    }).Else(() => {
      const ax = p.x.abs();
      const ay = p.y.abs();
      const ring = length(p).mul(2.0);
      // the seven families: disc, square, diamond, ring, cross, disc-in-cross, bar
      const dDisc = ring;
      const dSquare = max(ax, ay).mul(2.0);
      const dDiamond = ax.add(ay).mul(2.0);
      const dRing = ring.sub(0.62).abs().mul(2.7);
      const dCross = min(ax, ay).mul(8.0);
      const dDiscCross = max(ring, min(ax, ay).mul(8.0).oneMinus());
      const dBar = max(ax.mul(1.7), ay).mul(2.0);
      const r = select(
        vShape.lessThan(0.5),
        dDisc,
        select(
          vShape.lessThan(1.5),
          dSquare,
          select(
            vShape.lessThan(2.5),
            dDiamond,
            select(
              vShape.lessThan(3.5),
              dRing,
              select(vShape.lessThan(4.5), dCross, select(vShape.lessThan(5.5), dDiscCross, dBar)),
            ),
          ),
        ),
      );
      Discard(r.greaterThan(1.0));

      /* A SOLID body with a one-CSS-pixel edge, plus a halo whose width is also in pixels —
         edge softness is a property of the screen, not of the object. vPx is this glyph's
         radius in CSS px and r runs 0..1 across exactly that, so one pixel is 1/vPx in
         r units; no derivatives required. */
      const e = clamp(float(1.0).div(max(vPx, 1.0)), 0.004, 0.45);
      const body = smoothstep(float(BODY_R).sub(e), float(BODY_R).add(e), r).oneMinus();
      const hw = clamp(float(HALO_PX).div(max(vPx, 1.0)), e, max(1.0 - BODY_R, e));
      const halo = smoothstep(float(BODY_R), float(BODY_R).add(hw), r).oneMinus();
      const a = body.add(halo.mul(HALO_GAIN).mul(body.oneMinus())).mul(vFog).mul(vAlpha);
      Discard(a.lessThan(0.004));

      /* A small bright centre, not the whole body — with a solid body, reusing `body` here
         would wash the entire glyph toward white. Fog attenuates alpha only; mixing colour
         toward the ground as well dims twice (predecessor's measured mistake). The boost
         term (capped, predecessor's `min(vBoost, 1.6)`) burns a hovered/selected core
         hotter, so emphasis reads in colour as well as size. */
      const hot = smoothstep(0.0, BODY_R * 0.5, r).oneMinus();
      const c = mix(vColor, vec3(1.0), hot.mul(CORE_HOT).mul(min(vBoost, 1.6)));
      out.assign(vec4(c, a));
    });

    return out;
  })();

  const mesh = new Mesh(geo, material);
  mesh.frustumCulled = false; // one object covering the whole field; nothing to cull
  mesh.renderOrder = 1;

  return {
    mesh,
    material,
    colors,
    colorAttr,
    sizes,
    sizeAttr,
    shapeAttr,
    alphaAttr,
    boostAttr,
    uniforms: { halfH: uHalfH, tanHalf: uTanHalf, dRef: uDRef, fogNear: uFogNear, fogFar: uFogFar },
    dispose() {
      geo.dispose();
      material.dispose();
    },
  };
}
