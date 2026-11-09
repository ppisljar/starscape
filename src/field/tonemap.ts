/* The highlight roll-off (DESIGN.md §7 theming; ported from vault-site-3d src/tonemap.js).
 *
 * Additive blending accumulates in the framebuffer and nothing bounds the sum, so wherever
 * nodes pile up — the core of a dense cluster — the total runs past 1 and the hardware clamps
 * it to flat white. Every bit of colour and structure inside that core is gone, and no amount
 * of per-glyph tuning brings it back: each fragment was well behaved, it is the SUM that blew
 * out. The predecessor answered with a half-float target and a fullscreen post pass; starscape
 * owns no renderer and no render target (the caller does — §5), so the same three operators are
 * applied IN the fragment stage of both layer materials, behind one selectable uniform. That
 * rolls each glyph's own highlights (the hot core, the stacked halo ALU) before they hit the
 * framebuffer; a caller that wants the predecessor's true post-sum roll-off can build it as a
 * layer (§5 tier 3) reusing these exact curves.
 *
 * The curves are the predecessor's, byte for byte: Narkowicz's ACES fit ("filmic"), the usual
 * "AgX minimal" chain (keeps hue through the highlights where ACES skews warm), and Reinhard
 * (the gentlest). Every constant lives here exactly once, shared by the CPU reference
 * functions (what the tests pin) and the TSL expression (what the GPU runs) — two copies of a
 * polynomial is two chances to disagree.
 *
 * `op = 0` ("none") is a true bypass: the input colour passes through untouched, exposure
 * included — the predecessor's rule that "off" must be the OLD behaviour, not a third one.
 */

import type { Node } from 'three/webgpu';
import { clamp, log2, mat3, max, select, uniform, vec3 } from 'three/tsl';

/** Operator names, in uniform-index order — index 0 MUST stay 'none' (the bypass). */
export const TONEMAP_OPS = ['none', 'filmic', 'agx', 'reinhard'] as const;
export type TonemapOp = (typeof TONEMAP_OPS)[number];

/** Operator name → the index the shader branches on (predecessor's OP_INDEX, same values —
 *  a name drifting from its index turns a selected operator into a silent no-op). */
export const OP_INDEX: Record<TonemapOp, number> = { none: 0, filmic: 1, agx: 2, reinhard: 3 };

/** The resting exposure (pre-curve gain). Exported once — the config schema's exposure row
 *  references it rather than restating the number (§10 spirit). */
export const EXPOSURE_DEFAULT = 1;

/* ---- the curve constants (single source for CPU and TSL) ------------------------------- */

/** AgX log2 window. */
export const AGX_LO = -12.47393;
export const AGX_HI = 4.026069;

/** AgX inset/outset matrices, column-major (the GLSL mat3 constructor order — column i is
 *  elements 3i..3i+2). Nine numbers each, verbatim from the predecessor. */
const AGX_FWD = [
  0.842479, 0.0423282, 0.0423757, 0.0784336, 0.8784686, 0.0784336, 0.0792237, 0.0791661,
  0.879143,
] as const;
const AGX_INV = [
  1.196879, -0.0528969, -0.0529716, -0.0980209, 1.1519031, -0.0980435, -0.0990297, -0.0989612,
  1.1510737,
] as const;

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/* ---- CPU reference curves (pure; what the tests pin, what a custom layer can reuse) ------ */

/** Narkowicz ACES fit, per channel. */
export function filmicCurve(x: number): number {
  return clamp01((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

/** Reinhard, per channel. */
export function reinhardCurve(x: number): number {
  return x / (1 + x);
}

/** The AgX sigmoid polynomial (operates on the log2-normalised value). */
export function agxPoly(x: number): number {
  const x2 = x * x;
  const x4 = x2 * x2;
  return (
    15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x -
    0.00232
  );
}

/** The full "AgX minimal" chain — needs all three channels (the matrices mix them). */
export function agxCurve(r: number, g: number, b: number): [number, number, number] {
  // v = FWD * max(v, 0)   (column-major: out_row = Σ col_j[row] * v_j)
  const r0 = Math.max(r, 0);
  const g0 = Math.max(g, 0);
  const b0 = Math.max(b, 0);
  let vr = AGX_FWD[0] * r0 + AGX_FWD[3] * g0 + AGX_FWD[6] * b0;
  let vg = AGX_FWD[1] * r0 + AGX_FWD[4] * g0 + AGX_FWD[7] * b0;
  let vb = AGX_FWD[2] * r0 + AGX_FWD[5] * g0 + AGX_FWD[8] * b0;
  // log2 window → [0, 1]
  const span = AGX_HI - AGX_LO;
  const win = (v: number): number =>
    (Math.min(AGX_HI, Math.max(AGX_LO, Math.log2(Math.max(v, 1e-10)))) - AGX_LO) / span;
  vr = agxPoly(win(vr));
  vg = agxPoly(win(vg));
  vb = agxPoly(win(vb));
  // clamp(INV * v, 0, 1)
  return [
    clamp01(AGX_INV[0] * vr + AGX_INV[3] * vg + AGX_INV[6] * vb),
    clamp01(AGX_INV[1] * vr + AGX_INV[4] * vg + AGX_INV[7] * vb),
    clamp01(AGX_INV[2] * vr + AGX_INV[5] * vg + AGX_INV[8] * vb),
  ];
}

/** CPU mirror of the whole fragment-stage application: rgb → op(rgb · exposure), identity at
 *  op 0 (exposure NOT applied — the bypass is the old pipeline, not a dimmer). */
export function tonemapRGB(
  op: number,
  r: number,
  g: number,
  b: number,
  exposure: number = 1,
): [number, number, number] {
  if (op === OP_INDEX.none) return [r, g, b];
  const er = r * exposure;
  const eg = g * exposure;
  const eb = b * exposure;
  if (op === OP_INDEX.filmic) return [filmicCurve(er), filmicCurve(eg), filmicCurve(eb)];
  if (op === OP_INDEX.agx) return agxCurve(er, eg, eb);
  if (op === OP_INDEX.reinhard) return [reinhardCurve(er), reinhardCurve(eg), reinhardCurve(eb)];
  return [r, g, b]; // an unknown index behaves as the shader's select chain does: like 'none'
}

/* ---- the TSL side ------------------------------------------------------------------------ */

export interface TonemapRig {
  /** Driven by the Field ("animation = uniforms"): op is an OP_INDEX value, exposure a
   *  pre-curve gain. Structural `.value` holders, same convention as GpuTransform. */
  uniforms: {
    op: { value: number };
    exposure: { value: number };
  };
  /** Fragment-stage application: curve(c · exposure) selected by the op uniform, or c
   *  untouched at op 0. Alpha is the caller's business — only RGB goes through the curve. */
  apply(color: Node<'vec3'>): Node<'vec3'>;
}

/** Build the shared tonemap rig — ONE op/exposure uniform pair spliced into every layer's
 *  fragment stage, so the two draw calls can never roll off differently. All operators are
 *  cheap ALU, so the select chain (both sides evaluated) costs less than a real branch. */
export function createTonemap(): TonemapRig {
  const uOp = uniform(OP_INDEX.none);
  const uExposure = uniform(EXPOSURE_DEFAULT);

  const apply = (color: Node<'vec3'>): Node<'vec3'> => {
    const x = color.mul(uExposure);

    const filmic = clamp(
      x.mul(x.mul(2.51).add(0.03)).div(x.mul(x.mul(2.43).add(0.59)).add(0.14)),
      0.0,
      1.0,
    );

    const reinhard = x.div(x.add(1.0));

    const fwd = mat3(
      AGX_FWD[0], AGX_FWD[1], AGX_FWD[2], AGX_FWD[3], AGX_FWD[4], AGX_FWD[5], AGX_FWD[6],
      AGX_FWD[7], AGX_FWD[8],
    );
    const inv = mat3(
      AGX_INV[0], AGX_INV[1], AGX_INV[2], AGX_INV[3], AGX_INV[4], AGX_INV[5], AGX_INV[6],
      AGX_INV[7], AGX_INV[8],
    );
    const v = clamp(log2(max(fwd.mul(max(x, vec3(0.0))), vec3(1e-10))), AGX_LO, AGX_HI)
      .sub(AGX_LO)
      .div(AGX_HI - AGX_LO);
    const v2 = v.mul(v);
    const v4 = v2.mul(v2);
    const poly = v4
      .mul(v2)
      .mul(15.5)
      .sub(v4.mul(v).mul(40.14))
      .add(v4.mul(31.96))
      .sub(v2.mul(v).mul(6.868))
      .add(v2.mul(0.4298))
      .add(v.mul(0.1191))
      .sub(0.00232);
    const agx = clamp(inv.mul(poly), 0.0, 1.0);

    return select(
      uOp.lessThan(0.5),
      color, // the bypass: untouched, exposure included
      select(uOp.lessThan(1.5), filmic, select(uOp.lessThan(2.5), agx, reinhard)),
    ) as unknown as Node<'vec3'>;
  };

  return {
    uniforms: {
      op: uOp as unknown as { value: number },
      exposure: uExposure as unknown as { value: number },
    },
    apply,
  };
}
