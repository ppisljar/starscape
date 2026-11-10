/* The HDR bloom post-process — the "tier-3 post-sum pass reusing the tonemap curves" DESIGN.md
 * §5 anticipated, built at last as a helper the CONSUMER installs (the library owns no renderer,
 * no scene, no render target — §5). It is the signature nebula aesthetic: a million tiny additive
 * dots SUM in a half-float HDR buffer, the dense piles bloom into soft halos, and ONE tonemap
 * rolls the accumulated blaze off at the very end.
 *
 * Why this exists at all (the diagnosis DESIGN §5 and field/tonemap.ts record): the field rolls
 * each GLYPH off in its own fragment stage, which is the best a library with no render target can
 * do per draw call — but a per-FRAGMENT curve cannot shape a SUM it never sees. Zoomed out at
 * 10⁶, every glyph is a sub-pixel additive dot; with normal blending they average to grey mush,
 * with additive they pile past 1 and the hardware clamps the whole field to flat white. The only
 * cure is to tonemap the ACCUMULATED result: render additively into an HDR buffer, bloom it, and
 * curve the sum once — which needs a screen-space pass, which needs the renderer, which is the
 * consumer's. So this module is a helper the app drops into its loop in place of renderer.render.
 *
 * The chain (all linear until the last node):
 *
 *     scene ──pass──▶ HDR colour (HalfFloat) ──┬─────────────▶ add ──▶ tonemap ──▶ renderOutput ──▶ screen
 *                                              └─▶ bloom ─────▶ (sRGB only; NoToneMapping)
 *
 * Three seams handled deliberately:
 *   - NO DOUBLE TONEMAP. The post applies the filmic/agx/reinhard curve (field/tonemap.ts, the
 *     exact same TSL — two copies of a curve is two chances to disagree) as the FINAL colour node.
 *     The pipeline's own renderOutput then applies ONLY the sRGB transform, because it tonemaps
 *     iff renderer.toneMapping ≠ NoToneMapping and WebGPURenderer defaults to NoToneMapping. The
 *     consuming app must therefore also set the field's in-material roll-off to 'none' while the
 *     post is active (field.setTonemap('none')) so the sum is curved exactly once, post-sum.
 *   - RESIZE. pass and bloom both re-read renderer.getDrawingBufferSize every frame in their
 *     updateBefore, so resize is automatic; setSize() forwards once for an immediate (same-frame)
 *     effect instead of one frame late.
 *   - BLOOM IS ALREADY SUB-FULL-RES. BloomNode's bright-pass + blur mips run at resolutionScale
 *     0.5 by default (half-res), so the glow is cheap by construction; the only full-res added
 *     cost is the HDR scene target and the final composite. resolutionScale is exposed for perf.
 */

import { RenderPipeline } from 'three/webgpu';
import type { Camera, Object3D, Renderer } from 'three/webgpu';
import { pass, vec4 } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { createTonemap, OP_INDEX } from '../field/tonemap';
import type { TonemapOp } from '../field/tonemap';

/** Bloom's three knobs (UnrealBloom lineage): overall glow STRENGTH, the blur RADIUS (0..1,
 *  how far the halo reaches across the mip pyramid) and the luminance THRESHOLD below which a
 *  pixel does not seed the bloom (with additive HDR the dense cores are the ones over it). */
export interface BloomParams {
  strength: number;
  radius: number;
  threshold: number;
}

export interface PostProcessOpts {
  /** Initial bloom knobs. Defaults: strength 1.1, radius 0.8, threshold 0.0 (the nebula seed). */
  bloom?: Partial<BloomParams>;
  /** The post-sum roll-off operator (field/tonemap.ts). Default 'agx' — keeps hue through the
   *  highlights where the hot core blazes past white. */
  tonemap?: TonemapOp;
  /** Pre-curve exposure gain. Default 1.0. */
  exposure?: number;
  /** Bloom mip resolution scale (BloomNode default 0.5 = half-res — the standard perf lever).
   *  1 is full-res (sharper, costlier); below 0.5 is softer and cheaper. */
  resolutionScale?: number;
}

export interface PostProcess {
  /** Render the scene through the HDR → bloom → tonemap chain to the screen. Drop-in for
   *  renderer.render(scene, camera) in the consumer's loop. Call renderer.init() (awaited) once
   *  before the first render, exactly as for a plain WebGPURenderer. */
  render(): void;
  /** Live bloom knobs (uniform writes — no recompile). Only the named fields change. */
  setBloom(params: Partial<BloomParams>): void;
  /** The post-sum roll-off: operator + optional pre-curve exposure. 'none' bypasses the curve
   *  (the raw accumulated sum, sRGB-encoded) — rarely what you want with additive, but honest. */
  setTonemap(op: TonemapOp, exposure?: number): void;
  /** Bloom's resolution scale (1 full-res, 0.5 half-res default). The perf lever. */
  setResolutionScale(scale: number): void;
  /** Forward a viewport resize. CSS px (the renderer owns devicePixelRatio); pass/bloom also
   *  auto-track the renderer each frame, so this only makes the change land one frame sooner. */
  setSize(width: number, height: number): void;
  /** The live bloom knobs, read-only. */
  readonly bloom: BloomParams;
  /** The live roll-off state. */
  readonly tonemap: { op: TonemapOp; exposure: number };
  dispose(): void;
}

/** Build the HDR bloom post-process for a consumer's renderer/scene/camera. The returned
 *  `render()` replaces `renderer.render(scene, camera)` in the loop; everything else is a live
 *  uniform write. The renderer should keep toneMapping at NoToneMapping (the WebGPURenderer
 *  default) — the post owns the roll-off, and a renderer-level tonemap would curve the result a
 *  second time (see the double-tonemap seam above). */
export function createPostProcess(
  renderer: Renderer,
  scene: Object3D,
  camera: Camera,
  opts: PostProcessOpts = {},
): PostProcess {
  /* The HDR scene pass: HalfFloatType render target by default (PassNode's own default), which
     is the whole point — additive blending needs headroom past 1 to accumulate before the sum
     is ever curved. 'output' is the colour texture node. */
  const scenePass = pass(scene, camera);
  const scenePassColor = scenePass.getTextureNode('output');

  const params: BloomParams = {
    strength: opts.bloom?.strength ?? 1.1,
    radius: opts.bloom?.radius ?? 0.8,
    threshold: opts.bloom?.threshold ?? 0.0,
  };
  const bloomPass = bloom(scenePassColor, params.strength, params.radius, params.threshold);
  if (opts.resolutionScale !== undefined) bloomPass.setResolutionScale(opts.resolutionScale);

  /* The post-sum roll-off, its OWN rig (independent of the field's in-material rig): reusing
     field/tonemap.ts means the post chain and the per-glyph path share one source for the agx
     constants — the DESIGN §5 promise that the tier-3 pass "reuses the tonemap curves". */
  const tonemap = createTonemap();
  let op: TonemapOp = opts.tonemap ?? 'agx';
  tonemap.uniforms.op.value = OP_INDEX[op];
  tonemap.uniforms.exposure.value = opts.exposure ?? 1.0;

  /* The composite, still linear HDR: the scene colour PLUS its bloom, then the roll-off as the
     final colour node. Alpha is forced to 1 — the screen is opaque; only the summed RGB is the
     nebula. renderOutput (the pipeline, below) applies the sRGB transform after this. */
  const hdr = scenePassColor.add(bloomPass);
  const graded = vec4(tonemap.apply(hdr.xyz), 1.0);

  // RenderPipeline is the r183+ name for what used to be PostProcessing (a deprecated alias of
  // it now). Same API — a renderer + an output node graph, render()/dispose() — no warning.
  const post = new RenderPipeline(renderer, graded);
  // outputColorTransform stays true: renderOutput still needs to sRGB-encode. It tonemaps only
  // when renderer.toneMapping ≠ NoToneMapping, which the consumer keeps at the default — so the
  // one roll-off is ours, applied above, post-sum.
  post.outputColorTransform = true;

  const api: PostProcess = {
    render(): void {
      post.render();
    },

    setBloom(p: Partial<BloomParams>): void {
      if (p.strength !== undefined) {
        params.strength = p.strength;
        bloomPass.strength.value = p.strength;
      }
      if (p.radius !== undefined) {
        params.radius = p.radius;
        bloomPass.radius.value = p.radius;
      }
      if (p.threshold !== undefined) {
        params.threshold = p.threshold;
        bloomPass.threshold.value = p.threshold;
      }
    },

    setTonemap(nextOp: TonemapOp, exposure?: number): void {
      if (!(nextOp in OP_INDEX)) throw new Error(`setTonemap: "${nextOp}" names no operator`);
      op = nextOp;
      tonemap.uniforms.op.value = OP_INDEX[nextOp];
      if (exposure !== undefined) {
        if (!Number.isFinite(exposure) || exposure <= 0) {
          throw new Error(`setTonemap: exposure must be finite and > 0, got ${exposure}`);
        }
        tonemap.uniforms.exposure.value = exposure;
      }
    },

    setResolutionScale(scale: number): void {
      if (!Number.isFinite(scale) || scale <= 0) {
        throw new Error(`setResolutionScale: scale must be finite and > 0, got ${scale}`);
      }
      bloomPass.setResolutionScale(scale);
    },

    setSize(width: number, height: number): void {
      // pass/bloom re-read the renderer's drawing-buffer size every frame; forwarding the
      // device-pixel size here just avoids the one-frame lag. getDrawingBufferSize folds in
      // the renderer's pixel ratio, which is where devicePixelRatio lives (one pixel unit).
      const dpr = renderer.getPixelRatio();
      scenePass.setSize(Math.floor(width * dpr), Math.floor(height * dpr));
    },

    get bloom(): BloomParams {
      return { strength: params.strength, radius: params.radius, threshold: params.threshold };
    },

    get tonemap(): { op: TonemapOp; exposure: number } {
      return { op, exposure: tonemap.uniforms.exposure.value };
    },

    dispose(): void {
      post.dispose();
      scenePass.dispose();
    },
  };
  return api;
}
