import { describe, expect, it } from 'vitest';
import {
  createTweens,
  cubicInOut,
  linear,
  smoothstep,
  type Ease,
} from '../src/anim/tween';

describe('eases', () => {
  const eases: Array<[string, Ease]> = [
    ['linear', linear],
    ['cubicInOut', cubicInOut],
    ['smoothstep', smoothstep],
  ];

  it('hit both endpoints exactly', () => {
    for (const [, ease] of eases) {
      expect(ease(0)).toBe(0);
      expect(ease(1)).toBe(1);
    }
  });

  it('are monotone non-decreasing on [0, 1]', () => {
    for (const [name, ease] of eases) {
      let prev = ease(0);
      for (let k = 1; k <= 1000; k++) {
        const v = ease(k / 1000);
        expect(v, `${name} at ${k / 1000}`).toBeGreaterThanOrEqual(prev);
        prev = v;
      }
    }
  });

  it('cubicInOut matches the predecessor formula', () => {
    // process.js:148 — t < 0.5 ? 4t³ : 1 - (-2t+2)³/2
    expect(cubicInOut(0.25)).toBeCloseTo(4 * 0.25 ** 3, 15);
    expect(cubicInOut(0.75)).toBeCloseTo(1 - Math.pow(-2 * 0.75 + 2, 3) / 2, 15);
    expect(cubicInOut(0.5)).toBeCloseTo(0.5, 15);
  });
});

describe('createTweens', () => {
  it('get on an unseen key reads 0; set/get round-trip', () => {
    const tw = createTweens();
    expect(tw.get('spread')).toBe(0);
    tw.set('spread', 1.5);
    expect(tw.get('spread')).toBe(1.5);
    expect(tw.active()).toBe(false);
  });

  it('lands EXACTLY on the target at completion', () => {
    const tw = createTweens();
    tw.set('t', 0.1);
    tw.to('t', 0.7, { duration: 1, ease: cubicInOut });
    // Awkward dt steps that never sum to exactly 1 until they overshoot.
    for (let i = 0; i < 7; i++) tw.update(0.17);
    expect(tw.get('t')).toBe(0.7); // exact, not approximate
    expect(tw.active()).toBe(false);
  });

  it('a duration <= 0 snaps immediately', () => {
    const tw = createTweens();
    tw.to('dim', 2, { duration: 0 });
    expect(tw.get('dim')).toBe(2);
    expect(tw.active()).toBe(false);
  });

  it('progress is monotone for a monotone ease', () => {
    const tw = createTweens();
    tw.to('x', 1, { duration: 1, ease: smoothstep });
    let prev = tw.get('x');
    while (tw.active()) {
      tw.update(0.03);
      const v = tw.get('x');
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
    expect(tw.get('x')).toBe(1);
  });

  it('retarget continues from the CURRENT value — no snap', () => {
    const tw = createTweens();
    tw.to('x', 10, { duration: 1, ease: linear });
    tw.update(0.4);
    const mid = tw.get('x');
    expect(mid).toBeCloseTo(4, 12);

    tw.to('x', -5, { duration: 1, ease: linear });
    // Immediately after retarget, value is unchanged (no snap)…
    expect(tw.get('x')).toBe(mid);
    // …and the very next tiny step moves continuously from `mid` toward the new target.
    tw.update(0.001);
    const next = tw.get('x');
    expect(Math.abs(next - mid)).toBeLessThan(0.1);
    expect(next).toBeLessThan(mid); // heading toward -5 now

    for (let i = 0; i < 100; i++) tw.update(0.05);
    expect(tw.get('x')).toBe(-5);
  });

  it('update(0) is a strict no-op (value, progress, callbacks)', () => {
    const tw = createTweens();
    let calls = 0;
    tw.onChange('x', () => calls++);
    tw.to('x', 1, { duration: 1, ease: linear });
    tw.update(0.5);
    const atHalf = tw.get('x');
    const callsAtHalf = calls;

    for (let i = 0; i < 10; i++) tw.update(0);
    expect(tw.get('x')).toBe(atHalf); // value untouched
    expect(calls).toBe(callsAtHalf); // no callbacks fired
    expect(tw.active()).toBe(true); // still running, progress not consumed

    tw.update(0.5);
    expect(tw.get('x')).toBe(1); // 0.5 + 0.5 completes — the zeros contributed nothing
  });

  it('many keys tween independently', () => {
    const tw = createTweens();
    tw.to('a', 1, { duration: 1, ease: linear });
    tw.to('b', 100, { duration: 2, ease: linear });
    tw.set('c', 7);

    tw.update(0.5);
    expect(tw.get('a')).toBeCloseTo(0.5, 12);
    expect(tw.get('b')).toBeCloseTo(25, 12);
    expect(tw.get('c')).toBe(7);

    tw.update(0.5); // completes a, b keeps going
    expect(tw.get('a')).toBe(1);
    expect(tw.get('b')).toBeCloseTo(50, 12);
    expect(tw.active()).toBe(true);

    tw.update(1);
    expect(tw.get('b')).toBe(100);
    expect(tw.active()).toBe(false);
  });

  it('is deterministic: the same dt sequence yields identical values', () => {
    const run = () => {
      const tw = createTweens();
      const trace: number[] = [];
      tw.set('t', 0.25);
      tw.to('t', 1, { duration: 0.8, ease: cubicInOut });
      tw.to('s', -3, { duration: 1.3, ease: smoothstep });
      const dts = [0.016, 0.033, 0.016, 0.2, 0.016, 0.5, 0.016, 0.7];
      for (const dt of dts) {
        tw.update(dt);
        trace.push(tw.get('t'), tw.get('s'));
      }
      return trace;
    };
    const a = run();
    const b = run();
    expect(a).toEqual(b); // element-wise exact equality of numbers
  });

  it('onChange fires on set, on update steps, and on completion; unsubscribe works', () => {
    const tw = createTweens();
    const seen: number[] = [];
    const off = tw.onChange('x', (v) => seen.push(v));

    tw.set('x', 2);
    expect(seen).toEqual([2]);
    tw.set('x', 2); // unchanged — no event
    expect(seen).toEqual([2]);

    tw.to('x', 4, { duration: 1, ease: linear });
    tw.update(0.5);
    tw.update(0.5);
    expect(seen).toEqual([2, 3, 4]);
    expect(seen[seen.length - 1]).toBe(4); // completion delivered the exact endpoint

    off();
    tw.set('x', 99);
    expect(seen).toEqual([2, 3, 4]); // unsubscribed
  });

  it('active() reflects running tweens, and to() toward the current value does not run', () => {
    const tw = createTweens();
    expect(tw.active()).toBe(false);
    tw.set('x', 5);
    tw.to('x', 5, { duration: 1 }); // already there — nothing to animate
    expect(tw.active()).toBe(false);
    tw.to('x', 6, { duration: 1 });
    expect(tw.active()).toBe(true);
    tw.set('x', 0); // set cancels the tween
    expect(tw.active()).toBe(false);
    tw.update(1);
    expect(tw.get('x')).toBe(0); // cancelled tween never resumes
  });
});
