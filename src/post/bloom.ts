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
import type { Camera, Node, Object3D, Renderer } from 'three/webgpu';
import { dot, float, max, mix, pass, pow, uniform, vec3, vec4 } from 'three/tsl';

type VecNode = Node<'vec4'>;
type FloatNode = Node<'float'>;
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { createTonemap, OP_INDEX } from '../field/tonemap';
import type { TonemapOp } from '../field/tonemap';

/* The sublinear highlight-compression step that runs on the accumulated HDR BEFORE the bloom
 * bright-pass. Additive blending makes a pixel's brightness scale LINEARLY with local point
 * density, so a spiral-galaxy core (thousands of overlapping dots) outruns the sparse arms (a
 * handful) by ~2000:1 — a single exposure + tone curve cannot serve both: push the floor down
 * far enough to stop the core clipping and the arms go black; lift the arms and the core is a
 * flat white disc. The cure is to make the field respond SUBLINEARLY to density: curve the
 * luminance with pow(L, g) where g < 1 crushes the bright end, then scale RGB by the luma ratio
 * so HUE is preserved (only the magnitude moves). Done PRE-bloom on purpose — the bloom
 * threshold then sees the LIFTED sparse regions (they bloom more) and the COMPRESSED core (it
 * blooms less and stops blowing out), which is the whole point. */

/** The exponent g at full compression (c = 1): g = mix(1, COMPRESSION_LOW, c), so c = 0 gives
 *  g = 1 (an exact linear pass-through — today's look) and c = 1 gives g = COMPRESSION_LOW.
 *  0.33 was tuned in the browser on a 1M synth spiral: strong enough that c = 1 reveals the
 *  core's internal structure instead of a flat disc, without flattening the whole field to a
 *  uniform grey. */
export const COMPRESSION_LOW = 0.33;

/** Luminance floor: guards the pow and the divide against a zero-luma (true black) pixel, and
 *  bounds how hard a near-black pixel is lifted. True black (RGB 0) stays black — 0 · ratio = 0
 *  — so the negative space between clusters is untouched; only non-zero-but-dim pixels lift. */
const COMPRESSION_EPS = 1e-4;

/** Build the pre-bloom compression node: hdr' = hdr · pow(L, g)/L with L the Rec.709 luma and
 *  g driven live by the compression uniform. A hue-preserving luminance remap — the RGB ratio
 *  carries the chroma through untouched. At c = 0 the ratio is exactly 1, so this is a true
 *  bypass (the regression off-switch the nebula look depends on). */
function compressHighlights(color: VecNode, uCompression: FloatNode): VecNode {
  const L = dot(color.xyz, vec3(0.2126, 0.7152, 0.0722)); // Rec.709 luma of the HDR rgb
  const Lsafe = max(L, float(COMPRESSION_EPS));
  const g = mix(float(1.0), float(COMPRESSION_LOW), uCompression); // 1 at c=0 → exact bypass
  const ratio = pow(Lsafe, g).div(Lsafe); // L'/L: scales rgb by the luma ratio, hue preserved
  return color.mul(ratio) as unknown as VecNode;
}

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
  /** Sublinear highlight compression on the accumulated HDR, applied PRE-bloom. [0,1]: 0 is
   *  off — an exact linear pass-through, today's look — and 1 is the strongest roll-off
   *  (exponent COMPRESSION_LOW). The cure for an additive core blowing out to a flat white
   *  disc while the sparse arms stay dim: it crushes the brightness dynamic range hue-first,
   *  so the core shows structure and the lifted arms bloom. Default 0 (nothing changes unless
   *  asked). */
  compression?: number;
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
  /** Live highlight-compression amount, clamped to [0,1]: 0 off (linear), 1 strongest. A
   *  uniform write (no recompile). Applied PRE-bloom — see compressHighlights above. */
  setCompression(c: number): void;
  /** Forward a viewport resize. CSS px (the renderer owns devicePixelRatio); pass/bloom also
   *  auto-track the renderer each frame, so this only makes the change land one frame sooner. */
  setSize(width: number, height: number): void;
  /** The live bloom knobs, read-only. */
  readonly bloom: BloomParams;
  /** The live highlight-compression amount [0,1], read-only. */
  readonly compression: number;
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

  /* The pre-bloom highlight compression (see compressHighlights above). Its one uniform is
     written live by setCompression; 0 is an exact pass-through, so with compression off the
     chain is byte-for-byte what it was before this node existed. */
  const clamp01 = (c: number): number => (c < 0 ? 0 : c > 1 ? 1 : c);
  let compression = clamp01(opts.compression ?? 0);
  const uCompression = uniform(compression) as unknown as FloatNode;
  const compressed = compressHighlights(scenePassColor as unknown as VecNode, uCompression);

  const params: BloomParams = {
    strength: opts.bloom?.strength ?? 1.1,
    radius: opts.bloom?.radius ?? 0.8,
    threshold: opts.bloom?.threshold ?? 0.0,
  };
  const bloomPass = bloom(compressed, params.strength, params.radius, params.threshold);
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
  const hdr = compressed.add(bloomPass);
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

    setCompression(c: number): void {
      if (!Number.isFinite(c)) throw new Error(`setCompression: c must be finite, got ${c}`);
      compression = clamp01(c);
      (uCompression as unknown as { value: number }).value = compression;
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

    get compression(): number {
      return compression;
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
