import { describe, expect, it } from 'vitest';
import { hash, gauss } from '../src/core/hash';

const N = 10_000;

describe('hash', () => {
  it('is deterministic: same (i, salt) gives the same value', () => {
    for (let i = 0; i < 100; i++) {
      expect(hash(i, 7)).toBe(hash(i, 7));
      expect(hash(i * 131 + 5, 0xdeadbeef)).toBe(hash(i * 131 + 5, 0xdeadbeef));
    }
  });

  it('stays in [0, 1)', () => {
    for (let i = 0; i < N; i++) {
      const v = hash(i, 42);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('different salts decorrelate', () => {
    // Pearson correlation of two salted streams over the same indices — near zero.
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < N; i++) {
      const x = hash(i, 1);
      const y = hash(i, 2);
      sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    const cov = sxy / N - (sx / N) * (sy / N);
    const vx = sxx / N - (sx / N) ** 2;
    const vy = syy / N - (sy / N) ** 2;
    const corr = cov / Math.sqrt(vx * vy);
    expect(Math.abs(corr)).toBeLessThan(0.05);
  });

  it('mean over 10k draws is within 0.48..0.52', () => {
    let s = 0;
    for (let i = 0; i < N; i++) s += hash(i, 123);
    const mean = s / N;
    expect(mean).toBeGreaterThan(0.48);
    expect(mean).toBeLessThan(0.52);
  });
});

describe('gauss', () => {
  it('is deterministic and finite', () => {
    for (let i = 0; i < 100; i++) {
      const v = gauss(i, 9);
      expect(v).toBe(gauss(i, 9));
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('has mean ~0 and variance ~1 (loose)', () => {
    let s = 0, ss = 0;
    for (let i = 0; i < N; i++) {
      const v = gauss(i, 77);
      s += v; ss += v * v;
    }
    const mean = s / N;
    const variance = ss / N - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(variance).toBeGreaterThan(0.9);
    expect(variance).toBeLessThan(1.1);
  });
});
