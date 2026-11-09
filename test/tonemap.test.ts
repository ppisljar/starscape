/* The highlight roll-off (src/field/tonemap.ts). Two things break silently here, both
 * inherited guards from the predecessor's test/tonemap.test.mjs:
 *
 *   - an operator name drifting from the index the shader branches on turns a selected
 *     operator into a no-op rather than an error (OP_INDEX pins);
 *   - the curves drifting from the predecessor's — every pinned value below was computed
 *     from vault-site-3d src/tonemap.js's GLSL formulas verbatim, so a port typo (a matrix
 *     element, a polynomial coefficient, the log2 window) fails loudly instead of just
 *     looking slightly wrong.
 *
 * The TSL side is exercised headless: three/tsl builds its node graph without a GPU, so
 * createTonemap().apply() is asserted to compose (the constants are shared with the CPU
 * reference by construction — one source, see tonemap.ts).
 */

import { describe, expect, it } from 'vitest';
import { vec3 } from 'three/tsl';
import {
  AGX_HI,
  AGX_LO,
  OP_INDEX,
  TONEMAP_OPS,
  agxCurve,
  createTonemap,
  filmicCurve,
  reinhardCurve,
  tonemapRGB,
} from '../src/field/tonemap';

describe('OP_INDEX (name → shader index, the drift guard)', () => {
  it('every operator offered exists in OP_INDEX', () => {
    for (const name of TONEMAP_OPS) expect(OP_INDEX[name]).toBeTypeOf('number');
  });

  it('the off position is index 0 and no two operators collide', () => {
    // the Field treats 0 as the bypass and the shader's select chain branches on < 0.5 —
    // both depend on 'none' being exactly 0 (predecessor pin).
    expect(OP_INDEX.none).toBe(0);
    const vals = Object.values(OP_INDEX);
    expect(new Set(vals).size).toBe(vals.length);
  });

  it('keeps the predecessor indices: filmic 1, agx 2, reinhard 3', () => {
    expect(OP_INDEX.filmic).toBe(1);
    expect(OP_INDEX.agx).toBe(2);
    expect(OP_INDEX.reinhard).toBe(3);
  });
});

describe('filmic (Narkowicz ACES fit) — predecessor curve pins', () => {
  it('pins exact values of the predecessor formula', () => {
    // (x·(2.51x+0.03)) / (x·(2.43x+0.59)+0.14), clamped to [0,1]
    expect(filmicCurve(0)).toBe(0);
    expect(filmicCurve(0.18)).toBeCloseTo(0.26689892, 8);
    expect(filmicCurve(0.5)).toBeCloseTo(0.61630695, 8);
    expect(filmicCurve(1)).toBeCloseTo(0.80379747, 8);
    expect(filmicCurve(4)).toBeCloseTo(0.97341711, 8);
  });

  it('is monotonic and bounded (bright cores keep their shape, never clip)', () => {
    let prev = -1;
    for (let x = 0; x <= 16; x += 0.125) {
      const y = filmicCurve(x);
      expect(y).toBeGreaterThanOrEqual(prev);
      expect(y).toBeLessThanOrEqual(1);
      prev = y;
    }
  });
});

describe('reinhard — predecessor curve pins', () => {
  it('pins x / (1 + x) exactly', () => {
    expect(reinhardCurve(0)).toBe(0);
    expect(reinhardCurve(0.18)).toBeCloseTo(0.15254237, 8);
    expect(reinhardCurve(1)).toBe(0.5);
    expect(reinhardCurve(4)).toBeCloseTo(0.8, 10);
  });
});

describe('agx (the "AgX minimal" chain) — predecessor curve pins', () => {
  const pin = (input: [number, number, number], want: [number, number, number]): void => {
    const got = agxCurve(input[0], input[1], input[2]);
    for (let c = 0; c < 3; c++) expect(got[c]).toBeCloseTo(want[c]!, 7);
  };

  it('pins the log2 window the chain normalises into', () => {
    expect(AGX_LO).toBe(-12.47393);
    expect(AGX_HI).toBe(4.026069);
  });

  it('pins greys through the full matrix→log→polynomial→matrix chain', () => {
    pin([0, 0, 0], [0, 0, 0]);
    pin([0.18, 0.18, 0.18], [0.49667618, 0.49674478, 0.49674915]);
    pin([1, 1, 1], [0.78674523, 0.78688464, 0.78689344]);
    pin([8, 8, 8], [0.97059331, 0.97079474, 0.9708074]); // past-white sums roll off, not clip
  });

  it('pins saturated colours (the matrices mix channels — a swapped element shows here)', () => {
    pin([1, 0, 0], [0.8609764, 0.23015864, 0.23032496]);
    pin([0.5, 0.25, 0.125], [0.68961132, 0.56004368, 0.44558317]);
  });

  it('keeps hue through the highlights (the reason agx exists next to filmic)', () => {
    // a hot orange: under agx the red channel must stay dominant, nothing at 1.0 yet
    const [r, g, b] = agxCurve(4, 2, 0.5);
    expect(r).toBeGreaterThan(g);
    expect(g).toBeGreaterThan(b);
    expect(r).toBeLessThan(1);
  });
});

describe('tonemapRGB (the CPU mirror of the fragment-stage application)', () => {
  it('op 0 is a TRUE bypass: untouched, exposure included', () => {
    // the predecessor rule: with the operator off the pipeline must be EXACTLY what it was —
    // "off" must not become a third behaviour (here: not even a dimmer).
    expect(tonemapRGB(OP_INDEX.none, 0.3, 0.6, 0.9, 2.5)).toEqual([0.3, 0.6, 0.9]);
  });

  it('applies exposure BEFORE the curve (predecessor: "exposure, before roll-off")', () => {
    const [r] = tonemapRGB(OP_INDEX.filmic, 0.5, 0.5, 0.5, 2);
    expect(r).toBeCloseTo(filmicCurve(1), 10);
  });

  it('routes each index to its curve', () => {
    expect(tonemapRGB(OP_INDEX.filmic, 1, 1, 1)[0]).toBeCloseTo(filmicCurve(1), 10);
    expect(tonemapRGB(OP_INDEX.reinhard, 1, 1, 1)[0]).toBe(0.5);
    expect(tonemapRGB(OP_INDEX.agx, 1, 1, 1)[0]).toBeCloseTo(agxCurve(1, 1, 1)[0], 10);
  });
});

describe('createTonemap (the TSL rig)', () => {
  it('boots off: op 0, exposure 1 — the bypass is the default', () => {
    const rig = createTonemap();
    expect(rig.uniforms.op.value).toBe(0);
    expect(rig.uniforms.exposure.value).toBe(1);
  });

  it('builds the select chain headless (TSL needs no GPU to compose)', () => {
    const rig = createTonemap();
    const out = rig.apply(vec3(0.5, 0.25, 0.125) as never);
    // a composed TSL node: chainable and typed vec3 — proves every operator expression
    // (matrices, log2 window, polynomial) constructed without throwing
    expect(out).toBeTruthy();
    expect(typeof (out as { mul: unknown }).mul).toBe('function');
  });

  it('uniform writes land on the shared values (one rig drives both layers)', () => {
    const rig = createTonemap();
    rig.uniforms.op.value = OP_INDEX.agx;
    rig.uniforms.exposure.value = 1.8;
    expect(rig.uniforms.op.value).toBe(2);
    expect(rig.uniforms.exposure.value).toBe(1.8);
  });
});
