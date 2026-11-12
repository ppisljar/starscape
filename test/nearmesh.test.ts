/* Headless tests for the near-mesh LOD's PURE texture helpers (no GPU / no three.js construction):
 * the deterministic planet-texture buffer and the per-layer average colour that drives the far
 * billboards' cheap 'tint' mode. The DataArrayTexture wrapper and the material are GPU-only, so
 * they are exercised in the browser, not here.
 */

import { describe, expect, it } from 'vitest';
import {
  generateNearTextures,
  nearTextureAverages,
  NEAR_TEXTURE_NAMES,
  TEX_W,
  TEX_H,
} from '../src/lod/nearmesh';

describe('generateNearTextures (seeded planet atlas)', () => {
  it('fills one RGBA layer per texture name at TEX_W×TEX_H', () => {
    const raw = generateNearTextures(1);
    expect(raw.length).toBe(TEX_W * TEX_H * NEAR_TEXTURE_NAMES.length * 4);
  });

  it('is deterministic from the seed (same seed → byte-identical)', () => {
    expect(generateNearTextures(7)).toEqual(generateNearTextures(7));
  });

  it('a different seed gives different bytes', () => {
    const a = generateNearTextures(1);
    const b = generateNearTextures(2);
    expect(a).not.toEqual(b);
  });
});

describe('nearTextureAverages (per-layer dominant colour for tint)', () => {
  const raw = generateNearTextures(1);
  const avg = nearTextureAverages(raw);

  it('returns one linear RGB triple per layer', () => {
    expect(avg.length).toBe(NEAR_TEXTURE_NAMES.length * 3);
  });

  it('every component is a finite value in [0, 1]', () => {
    for (const v of avg) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('is deterministic for a given buffer', () => {
    expect(nearTextureAverages(raw)).toEqual(avg);
  });

  it('distinguishes planets — earth (layer 0, blue-dominant) vs mars (layer 1, red-dominant)', () => {
    const earthB = avg[0 * 3 + 2]!; // earth's blue
    const earthR = avg[0 * 3]!; // earth's red
    const marsR = avg[1 * 3]!; // mars's red
    const marsB = avg[1 * 3 + 2]!; // mars's blue
    expect(earthB).toBeGreaterThan(earthR); // oceans → blue over red
    expect(marsR).toBeGreaterThan(marsB); // rust → red over blue
  });

  it('matches an independent linear-space average on a tiny hand-rolled buffer', () => {
    // one 1-layer buffer is awkward (TEX_W×TEX_H fixed), so verify the sRGB→linear decode on a
    // single solid-grey layer: byte 188 ≈ linear 0.5 (the 0.5^(1/2.4)·255 round-trip point).
    const solid = new Uint8Array(TEX_W * TEX_H * 1 * 4);
    for (let o = 0; o < solid.length; o += 4) {
      solid[o] = 188;
      solid[o + 1] = 188;
      solid[o + 2] = 188;
      solid[o + 3] = 255;
    }
    const a = nearTextureAverages(solid, 1);
    const srgbToLinear = (b: number): number => {
      const c = b / 255;
      return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    };
    const want = srgbToLinear(188);
    expect(a[0]).toBeCloseTo(want, 6);
    expect(a[1]).toBeCloseTo(want, 6);
    expect(a[2]).toBeCloseTo(want, 6);
  });
});
