/* Stage-6 morph/chain/spread arithmetic (DESIGN.md §4), headless: the retarget bake
 * (mix-then-swap continuity), chain attribute packing, centroid computation, the pivot
 * uniform fold, and the §4 exactness rules — spread back to 1.0 and a completed morph
 * round trip restore byte-identical coordinates, pinned by FNV equality against the
 * frozen source (CPU reference via frames.transformPoint).
 */

import { describe, expect, it } from 'vitest';
import '../src/place/engine'; // built-ins: cloud / byIndex / uniform / none
import '../src/place/shapes';
import '../src/place/orderings';
import '../src/place/densities';
import '../src/place/scatters';
import {
  CHAIN_CAP,
  bakeMix,
  centroidOf,
  fieldSpinPoint,
  flattenAxis,
  packChainIds,
  packPivotUniforms,
  spreadPoint,
} from '../src/field/morph';
import { createFrameSet } from '../src/anim/frames';
import { compileHierarchy } from '../src/place/hierarchy';
import { buildGraph } from '../src/core/graph';
import { generateSynth } from '../src/synth/generate';
import { fnv1a } from '../src/core/fnv';
import type { Graph } from '../src/core/types';

function smallGraph(n = 400, seed = 7): Graph {
  return buildGraph(generateSynth({ n, seed }), { orderByFacet: 'system' });
}

/* ---------------------------------------------------------------- the retarget bake */

describe('bakeMix (the retarget bake, §4)', () => {
  const a0 = new Float32Array([1, 2, 3, -4.5, 0.25, 1e-7]);
  const b0 = new Float32Array([2, -2, 3.5, 0.5, 0.25, 7e3]);

  it('t <= 0 leaves A untouched, bit for bit', () => {
    const a = a0.slice();
    bakeMix(a, b0, 0);
    expect(fnv1a(a)).toBe(fnv1a(a0));
    bakeMix(a, b0, -0.25);
    expect(fnv1a(a)).toBe(fnv1a(a0));
  });

  it('t >= 1 makes A a bit-exact copy of B — never a + (b−a)·1', () => {
    const a = a0.slice();
    bakeMix(a, b0, 1);
    expect(fnv1a(a)).toBe(fnv1a(b0));
    const a2 = a0.slice();
    bakeMix(a2, b0, 1.0000001);
    expect(fnv1a(a2)).toBe(fnv1a(b0));
  });

  it('mid-tween bake is continuous: mix(A′, anything, 0) equals the pre-bake blend', () => {
    const t = 0.37;
    const a = a0.slice();
    // the pre-bake blend, computed the same way the store computes it (f32 rounding)
    const expected = new Float32Array(a0.length);
    for (let i = 0; i < a0.length; i++) expected[i] = a0[i]! + (b0[i]! - a0[i]!) * t;
    bakeMix(a, b0, t);
    expect(fnv1a(a)).toBe(fnv1a(expected)); // A′ IS the blend snapshot
    // …and a fresh morph from A′ starts exactly there (t = 0 → A′ untouched)
    const a2 = a.slice();
    bakeMix(a2, new Float32Array(a0.length), 0);
    expect(fnv1a(a2)).toBe(fnv1a(a));
  });

  it('throws on length mismatch rather than blending garbage', () => {
    expect(() => bakeMix(new Float32Array(3), new Float32Array(6), 0.5)).toThrow(/length/);
  });

  it('a completed round trip restores the frozen source byte-exactly (FNV-pinned)', () => {
    const graph = smallGraph();
    const frozen = graph.positions!;
    const h = compileHierarchy(
      graph,
      [
        { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.2 } },
        { place: { shape: 'cloud' } },
      ],
      { spacing: 10 },
    );
    const a = new Float32Array(frozen);
    const b = new Float32Array(h.positions);
    bakeMix(a, b, 1); // morph frozen → hierarchy, completed
    expect(fnv1a(a)).toBe(fnv1a(h.positions));
    bakeMix(a, frozen, 1); // morph back, completed
    expect(fnv1a(a)).toBe(fnv1a(frozen)); // §4: byte-identical, not "close"
  });
});

/* ---------------------------------------------------------------- chain packing */

describe('packChainIds (the per-node chain attribute)', () => {
  // a 3-deep pivot spine: 0 (root, level 0) ← 1 (level 1) ← 2 (level 2)
  const parent = new Int32Array([-1, 0, 1]);
  const level = new Uint8Array([0, 1, 2]);
  const nodePivot = new Uint32Array([2, 1, 0]);

  it('keeps the leaf pivot when the chain depth covers it', () => {
    expect([...packChainIds(nodePivot, parent, level, 3, 4)]).toEqual([2, 1, 0]);
  });

  it('clamps to the deepest ancestor inside the cap — deeper levels stop spinning independently (§3.4)', () => {
    expect([...packChainIds(nodePivot, parent, level, 3, 2)]).toEqual([1, 1, 0]);
    expect([...packChainIds(nodePivot, parent, level, 3, 1)]).toEqual([0, 0, 0]);
  });

  it('chainDepth caps the same way cap does (fixed depth = min(chainDepth, cap))', () => {
    expect([...packChainIds(nodePivot, parent, level, 2, 4)]).toEqual([1, 1, 0]);
  });

  it('chainDepth 0 (no level declares spin) packs the world frame for every node', () => {
    expect([...packChainIds(nodePivot, parent, level, 0, 4)]).toEqual([-1, -1, -1]);
  });

  it('over a compiled hierarchy: ids are the node pivots when every grouping level spins', () => {
    const graph = smallGraph(300, 3);
    const h = compileHierarchy(
      graph,
      [
        { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.1 } },
        { place: { shape: 'cloud' } },
      ],
      { spacing: 10 },
    );
    expect(h.chainDepth).toBe(1);
    const ids = packChainIds(h.nodePivot, h.pivots.parent, h.pivots.level, h.chainDepth, CHAIN_CAP);
    for (let i = 0; i < graph.n; i++) {
      expect(ids[i]).toBe(h.nodePivot[i]); // leaf pivots are level 0 < depth 1
    }
    // determinism: same inputs, byte-identical packing
    const again = packChainIds(h.nodePivot, h.pivots.parent, h.pivots.level, h.chainDepth, CHAIN_CAP);
    expect(fnv1a(again)).toBe(fnv1a(ids));
  });
});

/* ---------------------------------------------------------------- centroid + flatten */

describe('centroidOf', () => {
  it('is the mean, f32-rounded (the value the GPU uniform holds)', () => {
    const p = new Float32Array([0, 0, 0, 2, 4, 6, 4, 8, 0]);
    expect(centroidOf(p, 3)).toEqual([2, 4, 2]);
  });

  it('n = 0 is the origin, not NaN', () => {
    expect(centroidOf(new Float32Array(0), 0)).toEqual([0, 0, 0]);
  });
});

describe('flattenAxis (the 2d projection target, §3.6)', () => {
  it('drops exactly one axis and leaves the others bit-identical; the input is untouched', () => {
    const p = new Float32Array([1, 2, 3, 4, 5, 6]);
    const before = fnv1a(p);
    const flat = flattenAxis(p, 2, 1, 2.5);
    expect(fnv1a(p)).toBe(before);
    expect([...flat]).toEqual([1, 2.5, 3, 4, 2.5, 6]);
  });
});

/* ---------------------------------------------------------------- spread + chain reference */

describe('spread invariance over the chain reference (frames.transformPoint)', () => {
  it('spread → k → 1 lands byte-exactly on the chain output; undisplaced frames pass the frozen coords through FNV-identical', () => {
    const graph = smallGraph(500, 11);
    const frozen = graph.positions!;
    const h = compileHierarchy(
      graph,
      [
        { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.4 } },
        { groupBy: 'system', place: { shape: 'cloud' }, spin: { rate: 0.9, axis: [1, 0, 0] } },
        { place: { shape: 'cloud' } },
      ],
      { spacing: 10 },
    );
    const n = graph.n;
    const chain = packChainIds(h.nodePivot, h.pivots.parent, h.pivots.level, h.chainDepth, CHAIN_CAP);
    const frames = createFrameSet(h.pivots);
    const c = centroidOf(h.positions, n);

    /* (a) never advanced: the chain is an exact identity and spread 1 copies through, so
       the pinned coordinates survive the whole pipeline FNV-identical — the §4 rule. */
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) frames.transformPoint(chain[i]!, h.positions, out, i * 3, i * 3);
    for (let i = 0; i < n; i++) spreadPoint(out, c, 1, out, i * 3, i * 3);
    expect(fnv1a(out)).toBe(fnv1a(h.positions));
    // and the same through the frozen layout with no frames at all (chain id −1)
    const out2 = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) frames.transformPoint(-1, frozen, out2, i * 3, i * 3);
    for (let i = 0; i < n; i++) spreadPoint(out2, c, 1, out2, i * 3, i * 3);
    expect(fnv1a(out2)).toBe(fnv1a(frozen));

    /* (b) displaced frames: spread to 2.4 moves every point off the chain output, and
       dragging it back to 1 restores the chain output byte-exactly. */
    frames.advance(0.7);
    frames.advance(0.3); // two ticks, so composed quats are genuinely non-trivial
    const chained = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      frames.transformPoint(chain[i]!, h.positions, chained, i * 3, i * 3);
    }
    expect(fnv1a(chained)).not.toBe(fnv1a(h.positions)); // something actually turned
    const spreadOut = new Float32Array(n * 3);
    let movedSomewhere = false;
    for (let i = 0; i < n; i++) {
      spreadPoint(chained, c, 2.4, spreadOut, i * 3, i * 3);
      if (spreadOut[i * 3]! !== chained[i * 3]!) movedSomewhere = true;
    }
    expect(movedSomewhere).toBe(true);
    const back = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) spreadPoint(chained, c, 1, back, i * 3, i * 3);
    expect(fnv1a(back)).toBe(fnv1a(chained)); // s = 1 is an EXACT identity, not a lerp
  });
});

describe('fieldSpinPoint (the field-global root-frame rotation, §3.5)', () => {
  it('identity quaternion is an EXACT pass-through (byte-stable, like spread at s = 1)', () => {
    const graph = smallGraph(300, 7);
    const n = graph.n;
    const c = centroidOf(graph.positions!, n);
    const out = new Float32Array(graph.positions!);
    for (let i = 0; i < n; i++) fieldSpinPoint(out, c, [0, 0, 0, 1], out, i * 3, i * 3);
    expect(fnv1a(out)).toBe(fnv1a(graph.positions!));
  });

  it('rotates about the centroid, preserving distance to it; a quarter-turn about y maps +x→−z', () => {
    const c: [number, number, number] = [2, 5, -1];
    // quaternion for +90° about [0,1,0]: (0, sin45, 0, cos45)
    const s = Math.SQRT1_2;
    const q: [number, number, number, number] = [0, s, 0, s];
    const p = [c[0] + 1, c[1], c[2]]; // one unit along +x from the centre
    const out = [0, 0, 0];
    fieldSpinPoint(p, c, q, out);
    // +x about +y by +90° → −z (right-handed)
    expect(out[0]).toBeCloseTo(c[0], 6);
    expect(out[1]).toBeCloseTo(c[1], 6);
    expect(out[2]).toBeCloseTo(c[2] - 1, 6);
    // distance to the centre is preserved (a rigid rotation)
    const d = Math.hypot(out[0]! - c[0], out[1]! - c[1], out[2]! - c[2]);
    expect(d).toBeCloseTo(1, 6);
  });

  it('round-trips: rotating by q then by its conjugate returns to the start', () => {
    const c: [number, number, number] = [0, 0, 0];
    const ang = 0.9;
    const h = ang * 0.5;
    const ax = 0.3, ay = 0.8, az = -0.5;
    const len = Math.hypot(ax, ay, az);
    const si = Math.sin(h) / len;
    const q: [number, number, number, number] = [ax * si, ay * si, az * si, Math.cos(h)];
    const qInv: [number, number, number, number] = [-q[0], -q[1], -q[2], q[3]];
    const p = [1.2, -3.4, 2.1];
    const mid = [0, 0, 0];
    const back = [0, 0, 0];
    fieldSpinPoint(p, c, q, mid);
    fieldSpinPoint(mid, c, qInv, back);
    for (let k = 0; k < 3; k++) expect(back[k]).toBeCloseTo(p[k]!, 5);
  });
});

/* ---------------------------------------------------------------- pivot uniform fold */

describe('packPivotUniforms (the shader-side one-apply fold)', () => {
  it('q ⊗ p + offset reproduces frames.transformPoint for every pivot', () => {
    const graph = smallGraph(300, 5);
    const h = compileHierarchy(
      graph,
      [
        { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0.5, axis: [0, 1, 0] } },
        { groupBy: 'system', place: { shape: 'cloud' }, spin: { rate: 1.1, axis: [1, 0, 2] } },
        { place: { shape: 'cloud' } },
      ],
      { spacing: 10 },
    );
    const frames = createFrameSet(h.pivots);
    frames.advance(0.42);
    const P = frames.count;
    const quat = new Float32Array((P + 1) * 4);
    const offs = new Float32Array((P + 1) * 3);
    packPivotUniforms(frames, h.pivots.world, quat, offs);

    // slot 0 is the identity
    expect([...quat.subarray(0, 4)]).toEqual([0, 0, 0, 1]);
    expect([...offs.subarray(0, 3)]).toEqual([0, 0, 0]);

    const ref = new Float64Array(3);
    const p = [3.25, -7.5, 11.0] as const;
    for (let piv = 0; piv < P; piv++) {
      frames.transformPoint(piv, p, ref);
      const s = piv + 1;
      const qx = quat[s * 4]!, qy = quat[s * 4 + 1]!, qz = quat[s * 4 + 2]!, qw = quat[s * 4 + 3]!;
      // v' = v + w·t + q×t, t = 2(q×v) — exactly the shader kernel
      const tx = 2 * (qy * p[2] - qz * p[1]);
      const ty = 2 * (qz * p[0] - qx * p[2]);
      const tz = 2 * (qx * p[1] - qy * p[0]);
      const x = p[0] + qw * tx + (qy * tz - qz * ty) + offs[s * 3]!;
      const y = p[1] + qw * ty + (qz * tx - qx * tz) + offs[s * 3 + 1]!;
      const z = p[2] + qw * tz + (qx * ty - qy * tx) + offs[s * 3 + 2]!;
      // the fold runs through f32 quat/offset where transformPoint stays f64 — agreement
      // is to f32 precision, which is all the GPU ever sees
      expect(Math.abs(x - ref[0]!)).toBeLessThan(1e-3);
      expect(Math.abs(y - ref[1]!)).toBeLessThan(1e-3);
      expect(Math.abs(z - ref[2]!)).toBeLessThan(1e-3);
    }
  });
});
