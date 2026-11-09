/* Headless tests for stage 7 part 2 — the PURE parts only (vitest runs in node, no DOM):
 *   - projectPoint: the overlay's clip-matrix projection, and its agreement with the
 *     ladder's own projectedPx (one pixel unit — shader, picker, ladder AND overlay);
 *   - cardDims / clampBox / coverFrac: the grow-in-pixels box, the edge slide, the declutter;
 *   - OVERLAY_CSS: type is never scaled (no transform:scale anywhere), injection is a string
 *     export (no external file fetch to get wrong);
 *   - makeSynthProvider: deterministic prose — pure function of (graph, salt), per §10.1;
 *   - the house-rules audit: no clock, no RNG in the new sources.
 */

import { describe, expect, it } from 'vitest';
import { buildGraph } from '../src/core/graph';
import { generateSynth } from '../src/synth/generate';
import { DEFAULT_FOV } from '../src/field/constants';
import { fovPxOf, projectedPx } from '../src/ladder/ladder';
import { LOCK_H, LOCK_W } from '../src/ladder/constants';
import {
  CARD_MIN_H,
  CARD_MIN_W,
  OVERLAY_CSS,
  cardDims,
  clampBox,
  coverFrac,
  projectPoint,
} from '../src/ladder/overlay';
import type { Dims, Point2, ProjectedPoint } from '../src/ladder/overlay';
import { makeSynthProvider } from '../src/ladder/provider';

/** A standard perspective clip matrix (camera at the origin looking down −z, so world space
 *  IS view space), column-major in the three.js Matrix4.elements layout. */
function perspective(fovDeg: number, aspect: number, near = 0.1, far = 10_000): number[] {
  const f = 1 / Math.tan((fovDeg * Math.PI) / 360);
  const e = new Array<number>(16).fill(0);
  e[0] = f / aspect;
  e[5] = f;
  e[10] = (far + near) / (near - far);
  e[11] = -1;
  e[14] = (2 * far * near) / (near - far);
  return e;
}

const P: ProjectedPoint = { x: 0, y: 0, depth: 0 };
const D: Dims = { w: 0, h: 0 };
const B: Point2 = { x: 0, y: 0 };

describe('projectPoint', () => {
  const W = 1280;
  const H = 800;
  const m = perspective(DEFAULT_FOV, W / H);

  it('projects a point on the view axis to the exact screen centre, depth = distance', () => {
    expect(projectPoint(m, 0, 0, -10, W, H, P)).toBe(true);
    expect(P.x).toBeCloseTo(W / 2, 9);
    expect(P.y).toBeCloseTo(H / 2, 9);
    expect(P.depth).toBeCloseTo(10, 9);
  });

  it('reports a point behind the camera and zeroes the coordinates', () => {
    expect(projectPoint(m, 0, 0, 10, W, H, P)).toBe(false);
    expect(P.x).toBe(0);
    expect(P.y).toBe(0);
    expect(P.depth).toBeLessThan(0);
  });

  it('screen y grows downward: a point above the axis lands in the upper half', () => {
    projectPoint(m, 0, 5, -20, W, H, P);
    expect(P.y).toBeLessThan(H / 2);
    projectPoint(m, 0, -5, -20, W, H, P);
    expect(P.y).toBeGreaterThan(H / 2);
  });

  it('agrees with the ladder projection: an offset r at distance d spans projectedPx/2 px', () => {
    // One pixel unit, everywhere (§5): the overlay's matrix path and the ladder's analytic
    // path must measure the same sphere identically, or labels drift off their glyphs.
    const F = fovPxOf(H, DEFAULT_FOV);
    for (const [r, d] of [
      [5, 120],
      [22, 37.5],
      [1.8, 2000],
    ] as const) {
      projectPoint(m, r, 0, -d, W, H, P);
      const span = P.x - W / 2; // px right of centre
      expect(span).toBeCloseTo(projectedPx(r, d, F) / 2, 6);
    }
  });
});

describe('cardDims — the box grows in pixels, then freezes', () => {
  it('is the mount box at growT 0 and the locked rect at 1', () => {
    cardDims(0, LOCK_W, LOCK_H, D);
    expect(D).toEqual({ w: CARD_MIN_W, h: CARD_MIN_H });
    cardDims(1, LOCK_W, LOCK_H, D);
    expect(D).toEqual({ w: LOCK_W, h: LOCK_H });
  });

  it('clamps outside [0, 1] — past the lock the box must NOT keep growing', () => {
    cardDims(1.7, LOCK_W, LOCK_H, D);
    expect(D).toEqual({ w: LOCK_W, h: LOCK_H });
    cardDims(-0.3, LOCK_W, LOCK_H, D);
    expect(D).toEqual({ w: CARD_MIN_W, h: CARD_MIN_H });
  });

  it('is linear and monotone between, honouring a custom lock size', () => {
    cardDims(0.5, 500, 254, D);
    expect(D.w).toBe(Math.round((CARD_MIN_W + 500) / 2));
    expect(D.h).toBe(Math.round((CARD_MIN_H + 254) / 2));
    let last = -1;
    for (let t = 0; t <= 1.0001; t += 0.05) {
      cardDims(t, LOCK_W, LOCK_H, D);
      expect(D.w).toBeGreaterThanOrEqual(last);
      last = D.w;
    }
  });
});

describe('clampBox — cards slide along the stage edge, never hang off it', () => {
  it('leaves an interior box alone', () => {
    clampBox(300, 200, 100, 50, 1280, 800, 8, B);
    expect(B).toEqual({ x: 300, y: 200 });
  });

  it('clamps all four edges to the margin', () => {
    clampBox(-40, -40, 100, 50, 1280, 800, 8, B);
    expect(B).toEqual({ x: 8, y: 8 });
    clampBox(2000, 2000, 100, 50, 1280, 800, 8, B);
    expect(B).toEqual({ x: 1280 - 100 - 8, y: 800 - 50 - 8 });
  });

  it('pins a box wider than the stage at the leading margin', () => {
    clampBox(-500, 10, 2000, 50, 1280, 800, 8, B);
    expect(B.x).toBe(8);
  });
});

describe('coverFrac — the card declutter test', () => {
  it('is 1 for identical boxes, 0 for disjoint ones', () => {
    expect(coverFrac(10, 10, 100, 50, 10, 10, 100, 50)).toBe(1);
    expect(coverFrac(10, 10, 100, 50, 500, 500, 100, 50)).toBe(0);
    expect(coverFrac(10, 10, 100, 50, 110, 10, 100, 50)).toBe(0); // exactly touching
  });

  it('measures the overlapped fraction of the FIRST box', () => {
    expect(coverFrac(0, 0, 100, 100, 50, 0, 100, 100)).toBeCloseTo(0.5, 12);
    expect(coverFrac(0, 0, 100, 100, 50, 50, 100, 100)).toBeCloseTo(0.25, 12);
    // asymmetric: a small box fully inside a big one is fully covered…
    expect(coverFrac(40, 40, 20, 20, 0, 0, 100, 100)).toBe(1);
    // …while the big one is only fractionally covered by the small
    expect(coverFrac(0, 0, 100, 100, 40, 40, 20, 20)).toBeCloseTo(0.04, 12);
  });
});

describe('the stylesheet', () => {
  it('never scales type: no transform scale anywhere in OVERLAY_CSS', () => {
    expect(OVERLAY_CSS).not.toMatch(/scale\s*\(/);
    expect(OVERLAY_CSS).not.toMatch(/zoom\s*:/);
  });

  it('styles the three pieces the overlay mounts: labels, cards, bodies', () => {
    for (const sel of ['.ss-label', '.ss-card', '.ss-body', '.ss-pin', '.ss-loading']) {
      expect(OVERLAY_CSS).toContain(sel);
    }
  });

  it('keeps only the open card pointer-interactive — the canvas owns every drag', () => {
    expect(OVERLAY_CSS).toMatch(/\.ss-card\{[^}]*pointer-events:none/);
    expect(OVERLAY_CSS).toMatch(/\.ss-card\.ss-open\{[^}]*pointer-events:auto/);
  });
});

describe('makeSynthProvider — deterministic prose from hash(i, salt)', () => {
  const graph = buildGraph(generateSynth({ n: 64, seed: 7 }));

  it('titles come from graph.labelOf, verbatim', () => {
    const p = makeSynthProvider(graph);
    for (const i of [0, 13, 63]) {
      expect(p.card(i).title).toBe(graph.labelOf(i));
    }
  });

  it('the summary is exactly two sentences', () => {
    const p = makeSynthProvider(graph);
    const s = p.card(5).summary!;
    expect(s.length).toBeGreaterThan(20);
    expect(s.match(/\./g)!.length).toBe(2);
    expect(s.endsWith('.')).toBe(true);
  });

  it('the body is a few paragraphs of HTML', async () => {
    const p = makeSynthProvider(graph);
    const b = (await p.body(9)) as string;
    expect(typeof b).toBe('string');
    const paras = b.match(/<p>/g)!.length;
    expect(paras).toBeGreaterThanOrEqual(3);
    expect(paras).toBeLessThanOrEqual(5);
    expect(b.match(/<\/p>/g)!.length).toBe(paras);
  });

  it('two instances agree exactly — a pure function of (graph, salt), no RNG, no clock', async () => {
    const a = makeSynthProvider(graph);
    const b = makeSynthProvider(graph);
    for (const i of [0, 7, 31, 63]) {
      expect(a.card(i)).toEqual(b.card(i));
      expect(await a.body(i)).toBe(await b.body(i));
    }
  });

  it('different nodes and different salts read differently', async () => {
    const p = makeSynthProvider(graph);
    expect(await p.body(1)).not.toBe(await p.body(2));
    expect(p.card(1).summary).not.toBe(p.card(2).summary);
    const other = makeSynthProvider(graph, 0x1234);
    expect(other.card(1).summary).not.toBe(p.card(1).summary);
  });
});

describe('house rules (§10.1)', () => {
  it('the new ladder sources contain no clock and no RNG', async () => {
    // Same audit as test/ladder.test.ts, extended to the stage-7 part-2 files. The one
    // Node API is imported with a non-literal specifier — the tsconfig carries no Node
    // types (browser-targeted library), see the ladder test for the full rationale.
    const fsModule = 'node:fs';
    const { readFileSync } = (await import(/* @vite-ignore */ fsModule)) as {
      readFileSync: (path: URL, encoding: 'utf8') => string;
    };
    const banned = /(Date\s*\.\s*now|performance\s*\.\s*now|Math\s*\.\s*random)\s*\(/;
    for (const f of ['provider.ts', 'overlay.ts', 'index.ts']) {
      const src = readFileSync(new URL(`../src/ladder/${f}`, import.meta.url), 'utf8');
      expect(src).not.toMatch(banned);
    }
  });
});
