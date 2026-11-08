/* Headless tests for the field's pure parts (no GPU, no three.js import): the rank-normalised
 * size channel, palette lookup buffers, link-budget draw-range maths, and the glyph px-size
 * function the shader/picker/ladder all share.
 */

import { describe, expect, it } from 'vitest';
import {
  clampGlyphPx,
  fieldExtent,
  glyphPx,
  linkDrawCount,
  linkDrawVertices,
  rankPercentiles,
  sizesFromColumn,
  uniformSizes,
} from '../src/field/channels';
import { facetPalette, hslToRgb, paintFacetColors, GOLDEN } from '../src/field/palette';
import {
  NODE_MAX_PX,
  NODE_MIN_PX,
  SIZE_MAX_WORLD,
  SIZE_MIN_WORLD,
  SIZE_UNIFORM_WORLD,
} from '../src/field/constants';

describe('rankPercentiles (the size channel core)', () => {
  it('maps the column to even percentiles in [0, 1]', () => {
    const pct = rankPercentiles(Float32Array.from([10, 30, 20, 40]), 4);
    const want = [0, 2 / 3, 1 / 3, 1];
    for (let i = 0; i < 4; i++) expect(pct[i]).toBeCloseTo(want[i]!, 6); // f32 storage
  });

  it('breaks ties by index, deterministically', () => {
    const pct = rankPercentiles(Float32Array.from([5, 5, 5]), 3);
    // equal values rank in index order: node 0 lowest, node 2 highest
    expect(Array.from(pct)).toEqual([0, 0.5, 1]);
  });

  it('rank-normalises, not linear-maps: an outlier cannot flatten the band', () => {
    // one huge value; linear mapping would pin the other four to ~0
    const pct = rankPercentiles(Float32Array.from([1, 2, 3, 4, 1e9]), 5);
    expect(pct[3]).toBeCloseTo(0.75); // still evenly spread
    expect(pct[0]).toBe(0);
    expect(pct[4]).toBe(1);
  });

  it('handles n=0 and n=1', () => {
    expect(rankPercentiles(new Float32Array(0), 0).length).toBe(0);
    expect(Array.from(rankPercentiles(Float32Array.from([7]), 1))).toEqual([1]);
  });

  it('is deterministic across calls (same input, identical output)', () => {
    const data = Uint32Array.from({ length: 1000 }, (_, i) => (i * 2654435761) % 97);
    const a = rankPercentiles(data, 1000);
    const b = rankPercentiles(data, 1000);
    expect(a).toEqual(b);
  });
});

describe('sizesFromColumn', () => {
  it('sweeps world radii from SIZE_MIN_WORLD to SIZE_MAX_WORLD', () => {
    const s = sizesFromColumn(Float32Array.from([1, 2, 3]), 3);
    expect(s[0]).toBeCloseTo(SIZE_MIN_WORLD);
    expect(s[2]).toBeCloseTo(SIZE_MAX_WORLD);
    expect(s[1]).toBeGreaterThan(SIZE_MIN_WORLD);
    expect(s[1]).toBeLessThan(SIZE_MAX_WORLD);
  });

  it('respects the curve exponent: same WHAT is big, different how aggressively', () => {
    const data = Float32Array.from([0, 1, 2, 3, 4]);
    const sqrt = sizesFromColumn(data, 5, 0.5);
    const linear = sizesFromColumn(data, 5, 1);
    // same endpoints…
    expect(sqrt[0]).toBeCloseTo(linear[0]!);
    expect(sqrt[4]).toBeCloseTo(linear[4]!);
    // …but the sqrt curve lifts the middle
    expect(sqrt[2]).toBeGreaterThan(linear[2]!);
  });

  it('uniformSizes gives every node the same radius', () => {
    const u = uniformSizes(4);
    for (const v of u) expect(v).toBeCloseTo(SIZE_UNIFORM_WORLD, 5); // f32 storage
  });
});

describe('glyph px-size (the one pixel unit, shared by shader/picker/ladder)', () => {
  const halfH = 400; // 800 CSS px viewport
  const tanHalf = Math.tan((55 * Math.PI) / 360);
  const dRef = 400;

  it('is the true projection at falloff = 1', () => {
    const size = 10;
    const depth = 1000;
    // r·halfH / (tanHalf·d), independent of dRef when falloff = 1
    const expected = (size * halfH) / (tanHalf * depth);
    expect(glyphPx(size, depth, halfH, tanHalf, dRef, 1)).toBeCloseTo(expected);
    expect(glyphPx(size, depth, halfH, tanHalf, 123, 1)).toBeCloseTo(expected);
  });

  it('pivots falloff about dRef: size AT the reference distance never changes', () => {
    const at = (falloff: number) => glyphPx(8, dRef, halfH, tanHalf, dRef, falloff);
    expect(at(0.6)).toBeCloseTo(at(1));
    expect(at(2)).toBeCloseTo(at(1));
  });

  it('shrinks with distance, monotonically', () => {
    const a = glyphPx(8, 200, halfH, tanHalf, dRef);
    const b = glyphPx(8, 400, halfH, tanHalf, dRef);
    const c = glyphPx(8, 800, halfH, tanHalf, dRef);
    expect(a).toBeGreaterThan(b);
    expect(b).toBeGreaterThan(c);
  });

  it('scales with CSS viewport height only — devicePixelRatio never enters', () => {
    // halving the CSS height halves the px; there is no dpr parameter to disagree about
    const full = glyphPx(8, 500, 400, tanHalf, dRef);
    const half = glyphPx(8, 500, 200, tanHalf, dRef);
    expect(full).toBeCloseTo(half * 2);
  });

  it('clamps to the exported CSS-px window', () => {
    expect(clampGlyphPx(0.01)).toBe(NODE_MIN_PX);
    expect(clampGlyphPx(1e6)).toBe(NODE_MAX_PX);
    expect(clampGlyphPx(17)).toBe(17);
  });

  it('guards the near plane (depth <= 0 never divides by zero)', () => {
    expect(Number.isFinite(glyphPx(8, 0, halfH, tanHalf, dRef))).toBe(true);
    expect(Number.isFinite(glyphPx(8, -5, halfH, tanHalf, dRef))).toBe(true);
  });
});

describe('link-budget draw-range maths', () => {
  it('clamps the budget into [0, total]', () => {
    expect(linkDrawCount(100, 40)).toBe(40);
    expect(linkDrawCount(100, 0)).toBe(0);
    expect(linkDrawCount(100, -5)).toBe(0);
    expect(linkDrawCount(100, 1e9)).toBe(100);
  });

  it('rounds fractional budgets', () => {
    expect(linkDrawCount(100, 49.6)).toBe(50);
  });

  it('draw-range vertices are two per link (non-indexed line list)', () => {
    expect(linkDrawVertices(0)).toBe(0);
    expect(linkDrawVertices(41635)).toBe(83270);
  });
});

describe('facet palette (flat RGB lookup buffer)', () => {
  it('builds 3k floats in [0, 1]', () => {
    const pal = facetPalette(7);
    expect(pal.length).toBe(21);
    for (const v of pal) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('is deterministic', () => {
    expect(facetPalette(12)).toEqual(facetPalette(12));
  });

  it('golden-angle hues keep adjacent values apart, however many there are', () => {
    const pal = facetPalette(24);
    for (let v = 0; v + 1 < 24; v++) {
      const d =
        Math.abs(pal[v * 3]! - pal[v * 3 + 3]!) +
        Math.abs(pal[v * 3 + 1]! - pal[v * 3 + 4]!) +
        Math.abs(pal[v * 3 + 2]! - pal[v * 3 + 5]!);
      expect(d).toBeGreaterThan(0.15); // neighbours never land on neighbouring colours
    }
  });

  it('hue steps by the golden ratio', () => {
    const pal = facetPalette(2);
    const second = hslToRgb(GOLDEN, 0.72, 0.58);
    expect(pal[3]).toBeCloseTo(second[0]);
    expect(pal[4]).toBeCloseTo(second[1]);
    expect(pal[5]).toBeCloseTo(second[2]);
  });

  it('hslToRgb pins the corners', () => {
    expect(hslToRgb(0, 1, 0.5)).toEqual([1, 0, 0]); // pure red
    const [r, g, b] = hslToRgb(1 / 3, 1, 0.5); // pure green
    expect(r).toBeCloseTo(0);
    expect(g).toBeCloseTo(1);
    expect(b).toBeCloseTo(0);
    expect(hslToRgb(0.5, 0, 0.25)).toEqual([0.25, 0.25, 0.25]); // s=0 is grey at l
  });

  it('paintFacetColors scatters palette rows by code', () => {
    const pal = Float32Array.from([1, 0, 0, 0, 1, 0]); // red, green
    const out = paintFacetColors(Uint8Array.from([1, 0, 1]), 3, pal);
    expect(Array.from(out)).toEqual([0, 1, 0, 1, 0, 0, 0, 1, 0]);
  });
});

describe('fieldExtent (fit the live extent, never a stored one)', () => {
  it('centre, radius and diagonal of a known box', () => {
    // two opposite corners of a 2×2×2 box about (1, 1, 1)
    const pos = Float32Array.from([0, 0, 0, 2, 2, 2]);
    const e = fieldExtent(pos, 2);
    expect(e.centre).toEqual([1, 1, 1]);
    expect(e.radius).toBeCloseTo(Math.sqrt(3));
    expect(e.diagonal).toBeCloseTo(2 * Math.sqrt(3));
  });

  it('empty field is a zero extent, not NaN', () => {
    const e = fieldExtent(new Float32Array(0), 0);
    expect(e.centre).toEqual([0, 0, 0]);
    expect(e.radius).toBe(0);
    expect(e.diagonal).toBe(0);
  });
});
