import { describe, expect, it } from 'vitest';
import { generateSynth } from '../src/synth/generate';
import { buildGraph } from '../src/core/graph';

/* FNV-1a over raw bytes — the cheap way to assert "byte-identical", which is the actual
 * determinism contract (DESIGN.md §1): not "statistically similar", the same bytes. */
function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

const bytesOf = (a: Float32Array | Uint32Array): Uint8Array =>
  new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

describe('generateSynth', () => {
  it('is deterministic: two runs produce byte-identical positions and links', () => {
    const spec = { n: 20000, seed: 7 };
    const a = generateSynth(spec);
    const b = generateSynth(spec);
    expect(a.positions).toBeDefined();
    expect(b.positions).toBeDefined();
    expect(fnv1a(bytesOf(a.positions!))).toBe(fnv1a(bytesOf(b.positions!)));
    expect(fnv1a(bytesOf(a.links))).toBe(fnv1a(bytesOf(b.links)));
    // and the facet/column streams too — determinism is not just geometry
    expect(fnv1a(a.facets!['type']!.data as Uint8Array)).toBe(fnv1a(b.facets!['type']!.data as Uint8Array));
    expect(fnv1a(bytesOf(a.columns!['words'] as Uint32Array))).toBe(fnv1a(bytesOf(b.columns!['words'] as Uint32Array)));
  });

  it('a different seed produces different bytes', () => {
    const a = generateSynth({ n: 5000, seed: 7 });
    const b = generateSynth({ n: 5000, seed: 8 });
    expect(fnv1a(bytesOf(a.positions!))).not.toBe(fnv1a(bytesOf(b.positions!)));
    expect(fnv1a(bytesOf(a.links))).not.toBe(fnv1a(bytesOf(b.links)));
  });

  it('intra-system link fraction lands within ±5% of the spec', () => {
    for (const want of [0.72, 0.4]) {
      const g = generateSynth({ n: 10000, seed: 7, intraLinkFraction: want });
      const sys = g.facets!['system']!.data;
      const m = g.links.length >> 1;
      expect(m).toBeGreaterThan(0);
      let intra = 0;
      for (let e = 0; e < m; e++) {
        if (sys[g.links[e * 2]!]! === sys[g.links[e * 2 + 1]!]!) intra++;
      }
      expect(Math.abs(intra / m - want)).toBeLessThanOrEqual(0.05);
    }
  });

  it('facets: per-node data of length n, every code inside its value table, hierarchy real', () => {
    const n = 10000;
    const g = generateSynth({ n, seed: 7 });
    const facets = g.facets!;
    for (const key of ['type', 'galaxy', 'system', 'era']) {
      const f = facets[key]!;
      expect(f.data.length).toBe(n);
      let max = 0;
      for (let i = 0; i < n; i++) if (f.data[i]! > max) max = f.data[i]!;
      expect(max).toBeLessThan(f.values.length);
      // value table integrity: non-empty, unique names
      expect(new Set(f.values).size).toBe(f.values.length);
      for (const v of f.values) expect(v.length).toBeGreaterThan(0);
    }
    expect(facets['type']!.values).toHaveLength(7);
    expect(facets['era']!.values).toHaveLength(5);
    expect(facets['galaxy']!.values).toHaveLength(12);
    expect(facets['system']!.values).toHaveLength(12 * 7);
    // the hierarchy is real: a node's system always nests inside its galaxy
    const gal = facets['galaxy']!.data;
    const sys = facets['system']!.data;
    for (let i = 0; i < n; i++) expect((sys[i]! / 7) | 0).toBe(gal[i]!);
    // type and era are skewed, not uniform, and every value is actually used
    const eraCounts = new Uint32Array(5);
    for (let i = 0; i < n; i++) eraCounts[facets['era']!.data[i]!] = eraCounts[facets['era']!.data[i]!]! + 1;
    for (let j = 0; j < 5; j++) expect(eraCounts[j]!).toBeGreaterThan(0);
    expect(eraCounts[0]!).toBeGreaterThan(eraCounts[4]!);
  });

  it('words column is heavy-tailed: max far above median, all values >= 1', () => {
    const n = 10000;
    const g = generateSynth({ n, seed: 7 });
    const words = g.columns!['words'] as Uint32Array;
    expect(words.length).toBe(n);
    let max = 0;
    const sorted = Uint32Array.from(words).sort();
    for (let i = 0; i < n; i++) {
      expect(words[i]!).toBeGreaterThanOrEqual(1);
      if (words[i]! > max) max = words[i]!;
    }
    const median = sorted[n >> 1]!;
    expect(median).toBeGreaterThan(50);
    expect(median).toBeLessThan(300);
    expect(max).toBeGreaterThan(median * 10); // the tail is the point
  });

  it('n=100 smoke: well-formed output, and buildGraph round-trips it', () => {
    const input = generateSynth({ n: 100, seed: 7 });
    expect(input.n).toBe(100);
    expect(input.positions!.length).toBe(300);
    for (let i = 0; i < 300; i++) expect(Number.isFinite(input.positions![i]!)).toBe(true);
    expect(input.links.length % 2).toBe(0);
    const m = input.links.length >> 1;
    expect(m).toBeGreaterThan(100); // ~4 links per node drawn, minus degenerate skips
    for (let e = 0; e < m; e++) {
      const a = input.links[e * 2]!;
      const b = input.links[e * 2 + 1]!;
      expect(a).toBeLessThan(100);
      expect(b).toBeLessThan(100);
      expect(a).not.toBe(b);
    }
    expect(typeof input.labels!(0)).toBe('string');
    expect(input.labels!(0).length).toBeGreaterThan(0);

    const graph = buildGraph(input);
    expect(graph.n).toBe(100);
    expect(graph.counts.nodes).toBe(100);
    expect(graph.counts.links).toBeGreaterThan(0);
    expect(graph.csr.off.length).toBe(101);
    expect(graph.csr.off[100]!).toBe(graph.counts.links * 2);
    expect(graph.facets['system']).toBeDefined();
    expect(graph.columns['words']).toBeDefined();
    expect(typeof graph.labelOf(42)).toBe('string');
  });

  it('n=0 and tiny n do not blow up', () => {
    const empty = generateSynth({ n: 0, seed: 1 });
    expect(empty.links.length).toBe(0);
    expect(empty.positions!.length).toBe(0);
    const one = generateSynth({ n: 1, seed: 1 });
    expect(one.links.length).toBe(0); // the only possible pair is a self-link
    expect(one.positions!.length).toBe(3);
  });

  it('rejects malformed specs', () => {
    expect(() => generateSynth({ n: -1, seed: 1 })).toThrow();
    expect(() => generateSynth({ n: 10.5, seed: 1 })).toThrow();
    expect(() => generateSynth({ n: 10, seed: 1, galaxies: 0 })).toThrow();
    expect(() => generateSynth({ n: 10, seed: 1, intraLinkFraction: 1.5 })).toThrow();
  });

  // Timing sanity, not a benchmark: 2×10⁵ nodes + 8×10⁵ link draws should take well under a
  // second on anything modern; the generous ceiling only catches accidental O(n²) regressions.
  const onCI = !!(globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.['CI'];
  it.skipIf(onCI)('generates n=200000 under a generous ceiling', () => {
    const t0 = performance.now();
    const g = generateSynth({ n: 200000, seed: 7 });
    const dt = performance.now() - t0;
    expect(g.positions!.length).toBe(600000);
    expect(g.links.length >> 1).toBeGreaterThan(500000);
    expect(dt).toBeLessThan(10000);
  });
});
