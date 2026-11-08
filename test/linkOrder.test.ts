import { describe, expect, it } from 'vitest';
import { orderLinks } from '../src/core/linkOrder';
import { hash } from '../src/core/hash';

const BITS = 15;
const SPAN = (1 << BITS) - 1;

/** FNV-1a (32-bit) over a buffer's bytes — the golden-test fingerprint. */
function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** A small positioned graph with two spatial groups, fully deterministic from hash(). */
function makeGraph(n: number, m: number, seed: number) {
  const pos = new Float32Array(n * 3);
  const group = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const g = i < n / 2 ? 0 : 1;
    group[i] = g;
    // two clouds 100 apart, each a 10-wide box, so inter-group links are visibly longer
    pos[i * 3] = g * 100 + hash(i, seed) * 10;
    pos[i * 3 + 1] = hash(i, seed + 1) * 10;
    pos[i * 3 + 2] = hash(i, seed + 2) * 10;
  }
  const links = new Uint32Array(m * 2);
  for (let e = 0; e < m; e++) {
    const a = Math.floor(hash(e, seed + 3) * n);
    let b = Math.floor(hash(e, seed + 4) * n);
    if (a === b) b = (b + 1) % n;
    links[e * 2] = a;
    links[e * 2 + 1] = b;
  }
  return { pos, group, links };
}

function quantKey(
  links: Uint32Array,
  e: number,
  group: ArrayLike<number>,
  pos: Float32Array,
  q: number,
): number {
  const a = links[e * 2]!;
  const b = links[e * 2 + 1]!;
  const inter = group[a] !== group[b] ? 1 : 0;
  const L = Math.hypot(
    pos[a * 3]! - pos[b * 3]!,
    pos[a * 3 + 1]! - pos[b * 3 + 1]!,
    pos[a * 3 + 2]! - pos[b * 3 + 2]!,
  );
  return (inter << BITS) | Math.min(SPAN >> 1, (L * q) | 0);
}

function lengthScale(links: Uint32Array, pos: Float32Array): number {
  let maxL = 0;
  const m = links.length >> 1;
  for (let e = 0; e < m; e++) {
    const a = links[e * 2]! * 3;
    const b = links[e * 2 + 1]! * 3;
    const L = Math.hypot(pos[a]! - pos[b]!, pos[a + 1]! - pos[b + 1]!, pos[a + 2]! - pos[b + 2]!);
    if (L > maxL) maxL = L;
  }
  return maxL > 0 ? SPAN / maxL : 0;
}

describe('orderLinks', () => {
  const { pos, group, links } = makeGraph(64, 300, 1);
  const out = orderLinks(64, links, group, pos);
  const m = out.length >> 1;
  const q = lengthScale(links, pos);

  it('keeps every link, reordered', () => {
    expect(out.length).toBe(links.length);
    const canon = (ls: Uint32Array) => {
      const keys: number[] = [];
      for (let e = 0; e < ls.length >> 1; e++) keys.push(ls[e * 2]! * 64 + ls[e * 2 + 1]!);
      return keys.sort((a, b) => a - b);
    };
    expect(canon(out)).toEqual(canon(links));
  });

  it('every intra-group link sorts before every inter-group one', () => {
    let firstInter = m; // index of the first inter link; none may follow as intra
    for (let e = 0; e < m; e++) {
      const inter = group[out[e * 2]!] !== group[out[e * 2 + 1]!];
      if (inter && firstInter === m) firstInter = e;
      if (!inter) expect(e).toBeLessThan(firstInter);
    }
    // the construction guarantees both classes are present
    expect(firstInter).toBeGreaterThan(0);
    expect(firstInter).toBeLessThan(m);
  });

  it('within each class, quantised lengths are non-decreasing', () => {
    for (let e = 1; e < m; e++) {
      expect(quantKey(out, e, group, pos, q)).toBeGreaterThanOrEqual(
        quantKey(out, e - 1, group, pos, q),
      );
    }
  });

  it('clamps the length bucket at SPAN>>1', () => {
    for (let e = 0; e < m; e++) {
      expect(quantKey(out, e, group, pos, q) & SPAN).toBeLessThanOrEqual(SPAN >> 1);
    }
  });

  it('returns the input untouched when m < 2', () => {
    const empty = new Uint32Array(0);
    expect(orderLinks(10, empty, null, pos)).toBe(empty);
    const one = Uint32Array.from([0, 1]);
    expect(orderLinks(10, one, null, pos)).toBe(one);
  });

  it('degenerate layout (all nodes at one point): maxL = 0, q = 0, still total and stable', () => {
    // every length is 0 so the quantised key is pure inter-ness; intra must still lead and
    // nothing may NaN (0 * 0 and 0/0 are one branch apart — the q = 0 guard is what this pins)
    const flat = new Float32Array(64 * 3); // all zeros
    const ordered = orderLinks(64, links, group, flat);
    expect(ordered.length).toBe(links.length);
    let firstInter = ordered.length >> 1;
    for (let e = 0; e < ordered.length >> 1; e++) {
      const inter = group[ordered[e * 2]!] !== group[ordered[e * 2 + 1]!];
      if (inter && firstInter === ordered.length >> 1) firstInter = e;
      if (!inter) expect(e).toBeLessThan(firstInter);
    }
  });

  it('is deterministic: two calls are byte-identical', () => {
    const again = orderLinks(64, links, group, pos);
    const a = new Uint8Array(again.buffer, again.byteOffset, again.byteLength);
    const b = new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
    expect(a.length).toBe(b.length);
    let identical = true;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { identical = false; break; }
    expect(identical).toBe(true);
  });

  it('works with group = null (orders by length alone)', () => {
    const byLen = orderLinks(64, links, null, pos);
    for (let e = 1; e < byLen.length >> 1; e++) {
      const k0 = quantKey(byLen, e - 1, new Uint8Array(64), pos, q);
      const k1 = quantKey(byLen, e, new Uint8Array(64), pos, q);
      expect(k1).toBeGreaterThanOrEqual(k0);
    }
  });

  it('GOLDEN: fixed input -> known FNV-1a of the output buffer', () => {
    const g = makeGraph(128, 1000, 42);
    const ordered = orderLinks(128, g.links, g.group, g.pos);
    const digest = fnv1a(new Uint8Array(ordered.buffer, ordered.byteOffset, ordered.byteLength));
    // Recorded from the first run; any change here is a change in placement output and must
    // be deliberate (DESIGN.md §10.9).
    expect(digest).toBe(2672568477);
  });
});
