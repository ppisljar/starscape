/* bootQuality (src/field/quality.ts) — the stage-9 boot notch. The policy is the
 * predecessor's, ported (vault-site-3d main.js startingStep + its SCALE.md summary: full to
 * 10⁴, half links at 10⁵, a sixth + 1:1 device pixels at 10⁶), so these tests pin the ported
 * numbers rather than invent new ones. Pure function of (n, dpr): no clock, no feedback. */

import { describe, expect, it } from 'vitest';
import { bootQuality, RENDER_SCALE_FLOOR } from '../src/field/quality';
import { FLAT_PX, NODE_MIN_PX } from '../src/field/constants';
import { DEFAULTS, SESSION_KEYS } from '../src/config/schema';

describe('bootQuality', () => {
  it('draws everything up to 10⁴ — that decade is measured free', () => {
    for (const n of [1, 100, 10_000, 49_999]) {
      expect(bootQuality(n, 2)).toEqual({ linkFraction: 1, renderScale: 1, flatPx: FLAT_PX });
    }
  });

  it('halves the links at 10⁵ (the long cross-group tail goes first)', () => {
    const q = bootQuality(100_000, 2);
    expect(q.linkFraction).toBe(0.5);
    expect(q.renderScale).toBe(1); // native pixels still affordable at 10⁵
    expect(q.flatPx).toBeGreaterThan(NODE_MIN_PX); // the flat branch must actually fire
  });

  it('a sixth of the links from 3·10⁵, native pixels kept', () => {
    const q = bootQuality(300_000, 2);
    expect(q.linkFraction).toBeCloseTo(1 / 6, 12);
    expect(q.renderScale).toBe(1);
  });

  it('10⁶: 4% of the links, 1:1 device pixels, flat glyphs from 9 px', () => {
    // 0.04, not the predecessor's 8-16%: measured on this renderer (SCALE.md), a drawn link
    // costs ~100 ns/frame in a dense spinning arrangement, and the §9 30 fps floor at 10⁶
    // clears only below ~200 k drawn links.
    const q = bootQuality(1_000_000, 2);
    expect(q.linkFraction).toBeCloseTo(0.04, 12);
    expect(q.renderScale).toBe(0.5); // dpr 2 × 0.5 = 1:1 device pixels
    expect(q.flatPx).toBe(9);
  });

  it('1:1 pixels respect the schema rail on exotic ratios', () => {
    expect(bootQuality(1_000_000, 3).renderScale).toBe(RENDER_SCALE_FLOOR); // 1/3 clamps to 0.5
    expect(bootQuality(1_000_000, 1).renderScale).toBe(1);
    expect(bootQuality(1_000_000, 0).renderScale).toBe(1); // degenerate dpr: never upscale
  });

  it('the flat threshold clears the NODE_MIN_PX clamp floor at every budgeted notch', () => {
    // The px clamp pins a distant field at NODE_MIN_PX; a flat threshold at or under it can
    // never fire (the stage-9 finding: the shipped FLAT_PX of 2 sits under the 2.2 clamp).
    for (const n of [50_000, 100_000, 300_000, 1_000_000]) {
      expect(bootQuality(n, 2).flatPx).toBeGreaterThan(NODE_MIN_PX);
    }
  });

  it('monotone: a bigger corpus never gets a richer notch', () => {
    let prev = bootQuality(1, 2);
    for (const n of [10_000, 50_000, 100_000, 300_000, 500_000, 1_000_000, 4_000_000]) {
      const q = bootQuality(n, 2);
      expect(q.linkFraction).toBeLessThanOrEqual(prev.linkFraction);
      expect(q.renderScale).toBeLessThanOrEqual(prev.renderScale);
      expect(q.flatPx).toBeGreaterThanOrEqual(prev.flatPx);
      prev = q;
    }
  });

  it('writes only session-scoped config keys (re-derived each boot, never persisted)', () => {
    // The notch is applied through the config store; each of its three outputs must map to a
    // SESSION_KEYS row so a boot default can never be persisted into a look (schema.ts rule).
    for (const key of ['linkBudget', 'renderScale', 'flatPx']) {
      expect(SESSION_KEYS).toContain(key);
      expect(DEFAULTS[key]).toBeDefined();
    }
  });
});
