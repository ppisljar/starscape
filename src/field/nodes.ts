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
 * NODE_MIN_PX, the floor most of a distant field sits at). The UPPER cap is off by default
 * (uMaxPx = NODE_MAX_PX_OFF): a node you approach grows without bound and fills the view —
 * the clamp holds only the lower rail. What waits: rim bubbles, sphere shading and the full
 * theming (stage 8).
 *
 * Stage 6: the node POSITION is no longer an instanced attribute — it is pulled from the
 * shared GpuTransform at `instanceIndex` (DESIGN §4): morph mix, frame chain and spread all
 * happen right here in the vertex stage, so spinning or morphing the field re-uploads
 * NOTHING node-shaped (the predecessor's recompose flaw, closed). The links pull from the
 * same store, so the two layers cannot disagree about where a node is.
 */

import {
  DynamicDrawUsage,
  Float32BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  NormalBlending,
  SpriteNodeMaterial,
  Vector3,
} from 'three/webgpu';
import type { DataArrayTexture, Node } from 'three/webgpu';
import {
  Fn,
  If,
  asin,
  atan,
  attribute,
  clamp,
  dot,
  float,
  instanceIndex,
  int,
  length,
  max,
  min,
  mix,
  modelViewMatrix,
  pow,
  select,
  smoothstep,
  sqrt,
  texture,
  uniform,
  uniformArray,
  uv,
  varying,
  vec2,
  vec3,
  vec4,
  Discard,
} from 'three/tsl';
import type { GpuTransform } from './stores';
import type { TonemapRig } from './tonemap';
import {
  BODY_R,
  CORE_HOT,
  FALLOFF,
  FLAT_PX,
  HALO_GAIN,
  HALO_PX,
  NODE_MAX_PX_OFF,
  NODE_MIN_PX,
  NODE_SCALE,
  SPHERE,
} from './constants';

/** The seven glyph families the node fragment shader renders, in `iShape` index order
 *  (0 = disc … 6 = bar) — the select chain in `material.colorNode` below IS this list. A
 *  facet's value index maps into it through Field.setShapeFacet. All seven are cheap distance
 *  fields with no texture fetch, but they only READ as distinct shapes at a decent projected
 *  size: a 2 px disc and a 2 px diamond are the same blob (the flat-LOD branch even collapses
 *  them on purpose), so shape-by-facet wants a larger node size / lower density to be legible. */
export const NODE_SHAPES = ['disc', 'square', 'diamond', 'ring', 'cross', 'discCross', 'bar'] as const;
/** One of the seven glyph family names (NODE_SHAPES[k]). */
export type NodeShape = (typeof NODE_SHAPES)[number];
/** How many glyph families `iShape` selects between — the clamp/wrap modulus for setShapeFacet. */
export const SHAPE_COUNT = NODE_SHAPES.length;

/** How the FAR billboards are optionally textured to match the near-mesh LOD's planet look, so a
 *  node does not pop from a flat facet colour (billboard) to a textured planet (mesh) as it
 *  promotes. 'off' (the default) is today's behaviour, byte-identical — no texture, no new
 *  attribute, no extra uniform. 'tint' paints each far node its planet's DOMINANT colour (one
 *  precomputed average RGB per texture layer, no per-pixel fetch). 'full' SAMPLES the planet
 *  texture on a hemisphere UV (the disc shows the planet's visible face), so a far node reads as
 *  a tiny flat planet matching the mesh it becomes up close. */
export type GlyphTextureMode = 'off' | 'tint' | 'full';

/** The data `Field.setGlyphTexture` hands the node layer to texture the billboards. `array` is the
 *  SHARED planet DataArrayTexture (the SAME instance the near-mesh LOD samples, so far and near are
 *  identical); `layers` is the per-node texture-layer index (length n — node i's layer, the same
 *  `facetCode % layerCount` the LOD uses, so the billboard and the mesh agree per node); `avg` is
 *  the per-layer average RGB (length `layerCount*3`, linear 0..1) driving the cheap 'tint' mode. */
export interface GlyphTextureParams {
  array: DataArrayTexture;
  layerCount: number;
  layers: Float32Array;
  avg: Float32Array;
  mode: GlyphTextureMode;
}

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
    /** Flat-glyph threshold, CSS px: under this projected radius the fragment takes the
     *  cheap no-halo branch. Live (stage 9): the boot quality notch raises it with corpus
     *  size — the px clamp pins a distant field at NODE_MIN_PX (2.2), ABOVE the shipped
     *  FLAT_PX of 2, so without the lift the LOD branch never fires at all. */
    flatPx: { value: number };
    /* ---- glyph style, GLOBAL (same for every node — cheap scalar uniforms, not per-node).
       All seven were compile-time constants until stage 10; now they are live so the tuning
       panel's node rows have something to write (Field.setNodeStyle). Defaults are the
       constants in constants.ts (DESIGN §10 single-source). */
    /** Solid part of the glyph body, fraction of the sprite radius (default BODY_R). */
    bodyR: { value: number };
    /** Halo reach beyond the body, CSS px (default HALO_PX). */
    haloPx: { value: number };
    /** Halo strength relative to the body (default HALO_GAIN). */
    haloGain: { value: number };
    /** How much the bright centre burns toward white (default CORE_HOT). */
    coreHot: { value: number };
    /** Projected radius lower clamp, CSS px (default NODE_MIN_PX). Always applied. */
    minPx: { value: number };
    /** OPTIONAL projected-radius upper cap, CSS px. OFF by default (NODE_MAX_PX_OFF) — the node
     *  grows without bound as you approach; a finite value re-imposes a cap. */
    maxPx: { value: number };
    /** Perspective bend on projected size; 1 = true projection (default FALLOFF). */
    falloff: { value: number };
    /** Fake-sphere body shading, 0..1 (default SPHERE = 0 flat disc). */
    sphere: { value: number };
  };
  /** OPT-IN billboard texturing (see GlyphTextureMode). Passing a 'tint'/'full' mode LAZILY adds
   *  the `iTexLayer` attribute, binds the shared texture array + per-layer average colours, and
   *  recompiles the glyph once (the textured variant); thereafter mode/layer/colour changes are
   *  live uniform/attribute writes with no recompile. Passing mode 'off' (or null) parks it — and
   *  until the FIRST non-off call the glyph is byte-identical to today (nothing generated/bound). */
  setGlyphTexture(params: GlyphTextureParams | null): void;
  dispose(): void;
}

/** Live glyph-style overrides for the node layer — every field optional, each falling back to
 *  its constants.ts default. Shared by the build-time boot path (opts.style) and the live
 *  setter (Field.setNodeStyle). All GLOBAL (one value for the whole field). */
export interface NodeStyle {
  /** Solid part of the glyph body, fraction of the sprite radius. */
  bodyR?: number;
  /** Halo reach beyond the body, CSS px. */
  haloPx?: number;
  /** Halo strength relative to the body. */
  haloGain?: number;
  /** How much the bright centre burns toward white. */
  coreHot?: number;
  /** Projected radius lower clamp, CSS px. */
  minPx?: number;
  /** Optional projected-radius upper cap, CSS px. Omit for uncapped (NODE_MAX_PX_OFF). */
  maxPx?: number;
  /** Perspective bend on projected size; 1 = true projection. */
  falloff?: number;
  /** Fake-sphere body shading, 0..1 (0 = flat disc, 1 = a lit billboard "ball"). */
  sphere?: number;
}

/** Build the instanced node quad. `transform` is the shared GPU transform the position is
 *  pulled through at `instanceIndex` (stage 6 — no position attribute exists any more);
 *  `colors`/`sizes`/`shapes`/`alphas`/`boosts` are per-instance channels the Field (and its
 *  focus state) mutates in place. */
export function buildNodeLayer(
  n: number,
  transform: GpuTransform,
  colors: Float32Array,
  sizes: Float32Array,
  shapes: Float32Array,
  alphas: Float32Array,
  boosts: Float32Array,
  opts: {
    fov: number;
    heightCss: number;
    dRef: number;
    fogNear: number;
    fogFar: number;
    /** The shared highlight roll-off rig (stage 8, ./tonemap) — same instance as the links,
     *  so the two draw calls can never roll off differently. */
    tonemap: TonemapRig;
    /** Initial glyph-style overrides (boot path — a saved look applied on load). Each field
     *  falls back to its constants.ts default; live changes go through Field.setNodeStyle. */
    style?: NodeStyle;
  },
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

  const colorAttr = new InstancedBufferAttribute(colors, 3);
  const sizeAttr = new InstancedBufferAttribute(sizes, 1);
  const shapeAttr = new InstancedBufferAttribute(shapes, 1);
  const alphaAttr = new InstancedBufferAttribute(alphas, 1);
  const boostAttr = new InstancedBufferAttribute(boosts, 1);
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
  const uFlatPx = uniform(FLAT_PX);
  /* Glyph-style uniforms, GLOBAL (one value per field, not per node). Each defaults to its
     constants.ts value (single source) and is overridable at boot via opts.style; all are
     live — writing .value needs no recompile (Field.setNodeStyle). */
  const s = opts.style ?? {};
  const uBodyR = uniform(s.bodyR ?? BODY_R);
  const uHaloPx = uniform(s.haloPx ?? HALO_PX);
  const uHaloGain = uniform(s.haloGain ?? HALO_GAIN);
  const uCoreHot = uniform(s.coreHot ?? CORE_HOT);
  const uMinPx = uniform(s.minPx ?? NODE_MIN_PX);
  /* The upper cap is OPTIONAL and OFF by default (NODE_MAX_PX_OFF, a sentinel past any real
     viewport): a node is clamped only from BELOW, so it grows without bound as you approach and
     fills the frame — no "pushing". A finite cap re-imposed via setNodeStyle({maxPx}) binds again. */
  const uMaxPx = uniform(s.maxPx ?? NODE_MAX_PX_OFF);
  const uFalloff = uniform(s.falloff ?? FALLOFF);
  /* Fake-sphere shading strength, GLOBAL. 0 = today's flat disc (the dark looks are unchanged);
     1 = the body reads as a lit ball. The shade is faked per-fragment from a hemisphere normal —
     no geometry, so the two-draw-call billboard path is untouched. */
  const uSphere = uniform(s.sphere ?? SPHERE);

  /* ---- OPT-IN billboard texturing (GlyphTextureMode). Everything here stays INERT until the
     consumer's first non-off setGlyphTexture, so the default glyph is byte-identical to today: no
     iTexLayer attribute on the geometry, no texture bound, and uGlyphTexMode never enters the
     compiled shader (the default colorNode does not reference it). On the first activation the
     textured colorNode variant is compiled ONCE; after that, switching off/tint/full is a bare
     uniform write and relayering is an attribute write — no recompile. */
  const uGlyphTexMode = uniform(0); // 0 off, 1 tint, 2 full (only read by the textured variant)
  let textured = false; // has the textured colorNode variant been compiled yet
  let texLayers: Float32Array | null = null; // per-node layer buffer backing the iTexLayer attribute
  let texLayerAttr: InstancedBufferAttribute | null = null;
  let texArray: DataArrayTexture | null = null; // the shared planet array (captured by the shader)
  let avgVecs: Vector3[] = []; // per-layer average colours (mutated live; uniformArray re-reads them)
  let uAvg: ReturnType<typeof uniformArray> | null = null;

  /* ---- vertex stage: world radius → projected px → clamp → world radius at this depth.
     The clamp is done from the REAL projection, so it is exact rather than an approximation
     of itself. uFalloff bends the perspective about the fixed pivot uDRef (1 = true
     projection). SpriteNodeMaterial supplies the billboard: it offsets the quad corner in
     view space by scaleNode, exactly the predecessor's `mv.xy += position.xy * (r * 2)`.

     The position is PULLED through the shared transform at instanceIndex (§4): the same
     node object feeds positionNode and the depth math, so the builder evaluates the morph +
     chain + spread exactly once per vertex. */
  const worldPos = transform.worldOf(int(instanceIndex) as unknown as Node<'int'>);
  const iSize = attribute('iSize', 'float');
  const iBoost = attribute('iBoost', 'float');
  const centerView = modelViewMatrix.mul(vec4(worldPos, 1.0));
  const depth = max(centerView.z.negate(), 0.001);
  /* Boost scales the world radius AND both clamp bounds (predecessor's NODE_VERT: `uMinPx *
     iBoost`, `uMaxPx * iBoost`) — most of a distant field sits pinned at NODE_MIN_PX, where
     scaling only the radius would make hover invisible exactly when it is most needed. */
  const rw = iSize.mul(NODE_SCALE).mul(iBoost);
  const base = rw.mul(uHalfH).div(uTanHalf.mul(uDRef));
  const px = base.mul(pow(uDRef.div(depth), uFalloff));
  /* The LOWER clamp always applies (a distant page stays ≥ NODE_MIN_PX, a mote). The UPPER clamp
     is OPTIONAL: `min(px, iBoost·uMaxPx)` with uMaxPx defaulting to the "off" sentinel, so by
     default it is a no-op and px (hence the drawn world radius) grows without bound as depth→0 —
     at falloff 1, pxc reduces to the true projection and rWorld back to the node's real world
     radius, so approaching it simply makes it bigger on screen. A finite uMaxPx re-arms the cap. */
  const pxFloored = max(px, iBoost.mul(uMinPx));
  const pxc = min(pxFloored, iBoost.mul(uMaxPx));
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
  material.positionNode = worldPos;
  material.scaleNode = rWorld.mul(2.0);

  /* ---- fragment stage. The distance metric IS the shape: one length() plus a couple of
     abs()es buys seven glyph families with no texture fetch (predecessor's formDist). All
     seven fields are cheap ALU, so a select chain (both sides evaluated) costs less than a
     real branch would save.

     The glyph is compiled in TWO variants (see setGlyphTexture): the DEFAULT, non-textured
     variant below is byte-identical to the shipped glyph — it references no texture, no avg
     uniform and no iTexLayer attribute, so `withTex === false` is exactly today's shader. The
     textured variant (`withTex === true`, compiled lazily on the first non-off setGlyphTexture)
     weaves the planet look in behind uGlyphTexMode: 'tint' repaints the body the planet's average
     colour (cheap — one uniform-array fetch, NO texture read), 'full' samples the planet texture
     on a hemisphere UV so the flat disc shows the planet's visible face — and 'off' (mode 0) is
     the facet colour again, so toggling modes after opt-in costs only a uniform write. */
  const INV_2PI = 0.159_154_943_091_895_4; // 1 / (2π): hemisphere longitude → [0.25, 0.75]
  const INV_PI = 0.318_309_886_183_790_7; // 1 / π:    hemisphere latitude  → [0, 1]
  const EMISSIVE_GAIN = 1.2; // far-dot glow for lava cracks / stars (the texture's A channel)
  const makeColorNode = (withTex: boolean): Node<'vec4'> => {
    // iTexLayer is per-INSTANCE: read it through a flat varying (constant across the quad, like
    // iShape/iColor). Only the textured variant references it, so the default geometry needs no
    // such attribute.
    const vTexLayer = withTex ? varying(float(attribute('iTexLayer', 'float'))) : null;
    return Fn(() => {
      const p = uv().sub(0.5); // quad coordinate in [-0.5, 0.5], predecessor's vUv

      const out = vec4(0.0).toVar();

      // the node's planet layer + its precomputed average colour (both cheap: no texture fetch).
      // Only ever evaluated in the textured variant. `.element` is typed Node<unknown>; coerce it
      // back to a vec3 node so the colour maths below type-check (a no-op at runtime).
      const layerI = withTex ? int(vTexLayer!) : null;
      const avgCol = withTex
        ? (uAvg!.element(layerI!) as unknown as ReturnType<typeof vec3>)
        : null;
      const mode = uGlyphTexMode;

      If(vPx.lessThan(uFlatPx), () => {
        /* LOD, the cheapest available: under a couple of px every glyph is the same blob and
           the halo is the expensive half of it. One flat colour, no smoothsteps, no form test. */
        const rr = length(p).mul(2.0);
        Discard(rr.greaterThan(1.0));
        const aa = vFog.mul(vAlpha);
        Discard(aa.lessThan(0.004));
        if (withTex) {
          // far/tiny: tint AND full collapse to the planet average — a texture fetch is invisible
          // at < 2 px, so even 'full' uses the cheap average here (the dominant-colour continuity
          // is all that reads at this size).
          out.assign(vec4(select(mode.lessThan(0.5), vColor, avgCol!), aa));
        } else {
          out.assign(vec4(vColor, aa));
        }
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
        const body = smoothstep(uBodyR.sub(e), uBodyR.add(e), r).oneMinus();
        const hw = clamp(uHaloPx.div(max(vPx, 1.0)), e, max(uBodyR.oneMinus(), e));
        const halo = smoothstep(uBodyR, uBodyR.add(hw), r).oneMinus();
        const a = body.add(halo.mul(uHaloGain).mul(body.oneMinus())).mul(vFog).mul(vAlpha);
        Discard(a.lessThan(0.004));

        /* The fake-sphere hemisphere normal over the disc (nz = sqrt(1 − ring²)): the sphere
           shade below lights it, and the 'full' texture mode maps it to an equirectangular UV so
           the flat disc shows the planet's visible (camera-facing) face. */
        const nx = p.x.mul(2.0);
        const ny = p.y.mul(2.0);
        const nz = sqrt(max(float(1.0).sub(ring.mul(ring)), 0.0));
        const N = vec3(nx, ny, nz);

        /* A small bright centre, not the whole body — with a solid body, reusing `body` here
           would wash the entire glyph toward white. Fog attenuates alpha only; mixing colour
           toward the ground as well dims twice (predecessor's measured mistake). The boost
           term (capped, predecessor's `min(vBoost, 1.6)`) burns a hovered/selected core
           hotter, so emphasis reads in colour as well as size. */
        const hot = smoothstep(0.0, uBodyR.mul(0.5), r).oneMinus();

        /* The body albedo, by glyph-texture mode. Off → the facet colour (today's path). Tint →
           the planet's average colour, so a far node matches the dominant colour of the mesh it
           becomes. Full → the sampled planet texel on the hemisphere UV, with a light facet-colour
           hint (the mesh's `mix(white, tint, 0.25)`) so the node still carries its colour, plus the
           texture's emissive mask (lava cracks / stars). Non-textured variant: just the facet
           colour, no branches — byte-identical to today. */
        const bodyCol = withTex ? vColor.toVar() : vColor;
        const emis = withTex ? vec3(0.0).toVar() : null;
        if (withTex) {
          If(mode.greaterThan(0.5), () => {
            bodyCol.assign(avgCol!); // tint (and the starting point full overrides below)
          });
          If(mode.greaterThan(1.5), () => {
            const uLon = atan(N.x, N.z).mul(INV_2PI).add(0.5); // longitude of the visible face
            const vLat = asin(clamp(N.y, -1.0, 1.0)).mul(INV_PI).add(0.5); // latitude
            const sampled = texture(texArray!, vec2(uLon, vLat)).depth(layerI!).level(float(0.0));
            bodyCol.assign(sampled.rgb.mul(mix(vec3(1.0, 1.0, 1.0), vColor, float(0.25))));
            emis!.assign(sampled.rgb.mul(sampled.a).mul(EMISSIVE_GAIN));
          });
        }
        const c = mix(bodyCol, vec3(1.0), hot.mul(uCoreHot).mul(min(vBoost, 1.6)));

        /* Fake-sphere body shading (uSphere): the camera-facing billowy body reads as a lit BALL,
           not a flat disc. A Lambert diffuse lit from the UPPER-LEFT (L and the half-vector H are
           the precomputed unit directions below) that keeps a 0.6 ambient floor so the shaded side
           stays matte ink rather than black, plus a soft specular glint. All faked in-fragment — NO
           geometry, so the two-draw-call path is untouched. uSphere lerps flat → shaded; at 0 this
           is the exact disc the dark looks ship, so field/nebula/space4 are unchanged. */
        const ndl = max(dot(N, vec3(-0.3932, 0.4423, 0.8061)), 0.0); // L: upper-left, toward camera
        const diffuse = float(0.6).add(ndl.mul(0.4)); // 0.6 (shaded) .. 1.0 (fully lit)
        const ndh = max(dot(N, vec3(-0.2069, 0.2327, 0.9503)), 0.0); // H = normalize(L + view)
        const spec = pow(ndh, 18.0).mul(0.4); // a soft white glint upper-left
        const cSphere = c.mul(diffuse).add(vec3(spec));
        const cShaded = withTex ? mix(c, cSphere, uSphere).add(emis!) : mix(c, cSphere, uSphere);
        out.assign(vec4(cShaded, a));
      });

      /* Stage 8 (§7): the highlight roll-off, in-fragment behind one selectable uniform —
         identity when the op uniform is 0, so 'none' is the stage-6 pipeline untouched. */
      return vec4(opts.tonemap.apply(out.xyz as unknown as Node<'vec3'>), out.w);
    })() as unknown as Node<'vec4'>;
  };
  material.colorNode = makeColorNode(false) as unknown as typeof material.colorNode;

  /** Opt-in billboard texturing. First non-off call: allocate the iTexLayer attribute, capture the
   *  shared array + build the avg uniform array, compile the textured variant ONCE. Later calls are
   *  live writes (mode uniform, layer attribute, avg vectors) with no recompile — unless the shared
   *  array instance or the layer count changes, which forces a fresh compile that recaptures them. */
  const setGlyphTexture = (params: GlyphTextureParams | null): void => {
    if (params === null || params.mode === 'off') {
      uGlyphTexMode.value = 0; // park it; inert if the textured variant was never compiled
      return;
    }
    const modeCode = params.mode === 'full' ? 2 : 1;
    let recompile = false;

    // per-node layer buffer + attribute (lazy — the default geometry carries no iTexLayer)
    if (!texLayers || texLayers.length !== n) texLayers = new Float32Array(n);
    texLayers.set(params.layers.subarray(0, n));
    if (!texLayerAttr) {
      texLayerAttr = new InstancedBufferAttribute(texLayers, 1);
      texLayerAttr.setUsage(DynamicDrawUsage);
      geo.setAttribute('iTexLayer', texLayerAttr);
      recompile = true; // a never-textured glyph gains the attribute + the textured variant
    } else {
      texLayerAttr.needsUpdate = true; // relayer (facet change): same buffer, new values
    }

    // per-layer average colours (mutated in place; uniformArray re-reads them each render)
    if (avgVecs.length !== params.layerCount) {
      avgVecs = Array.from({ length: params.layerCount }, () => new Vector3());
      uAvg = uniformArray(avgVecs, 'vec3');
      recompile = true; // the uniform-array node changed identity — recapture it
    }
    for (let k = 0; k < params.layerCount; k++) {
      avgVecs[k]!.set(params.avg[k * 3]!, params.avg[k * 3 + 1]!, params.avg[k * 3 + 2]!);
    }

    // the shared planet array (captured by the shader; a new instance forces a recompile)
    if (texArray !== params.array) {
      texArray = params.array;
      recompile = true;
    }

    uGlyphTexMode.value = modeCode;
    if (recompile || !textured) {
      material.colorNode = makeColorNode(true) as unknown as typeof material.colorNode;
      material.needsUpdate = true; // rebuild the node program once
      textured = true;
    }
  };

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
    uniforms: {
      halfH: uHalfH,
      tanHalf: uTanHalf,
      dRef: uDRef,
      fogNear: uFogNear,
      fogFar: uFogFar,
      flatPx: uFlatPx,
      bodyR: uBodyR,
      haloPx: uHaloPx,
      haloGain: uHaloGain,
      coreHot: uCoreHot,
      minPx: uMinPx,
      maxPx: uMaxPx,
      falloff: uFalloff,
      sphere: uSphere,
    },
    setGlyphTexture,
    dispose() {
      geo.dispose();
      material.dispose();
    },
  };
}
