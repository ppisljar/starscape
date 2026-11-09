/* The tween system — DESIGN.md §4 "Animation = uniforms".
 *
 * One small engine drives every animated scalar in the scene: morph `t`, spread, dim, camera
 * flight parameters. Values are plain numbers destined for uniforms; a vector animates as
 * several keys. The engine never touches a vertex buffer, and it owns no clock — time enters
 * ONLY through `update(dt)` accumulation (§10.1: no Date.now / performance.now in anything
 * that produces geometry), so a fixed dt sequence replays byte-identically.
 *
 * Retargeting is interruption-free by construction: `to()` on a running key restarts the
 * tween FROM THE CURRENT VALUE, never from the old start or old target, so a mid-flight
 * change of destination bends the motion without a snap.
 */

/** An easing curve: maps normalized time u ∈ [0, 1] to progress, with f(0)=0 and f(1)=1. */
export type Ease = (u: number) => number;

export const linear: Ease = (u) => u;

/** The predecessor's ease (vault-site-3d process.js): symmetric cubic in/out. */
export const cubicInOut: Ease = (u) =>
  u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;

/** Hermite smoothstep, the shader's native ease — matching it on the CPU keeps CPU-side
 *  previews in step with GPU-side fades. */
export const smoothstep: Ease = (u) => u * u * (3 - 2 * u);

export interface TweenOptions {
  /** Seconds (same unit the caller feeds to `update(dt)`). <= 0 snaps immediately. */
  duration?: number;
  ease?: Ease;
}

export interface Tweens {
  /** Animate `key` toward `target`. If the key is mid-tween it retargets from its CURRENT
   *  value — no snap. A duration <= 0 sets the value immediately. */
  to(key: string, target: number, opts?: TweenOptions): void;
  /** Set `key` instantly, cancelling any running tween on it. */
  set(key: string, value: number): void;
  /** Current value of `key`; a key never set or tweened reads as 0. */
  get(key: string): number;
  /** Advance all running tweens by `dt` (seconds). `update(0)` is a strict no-op. */
  update(dt: number): void;
  /** True while any tween is still running. */
  active(): boolean;
  /** Observe value changes on one key (fires on set, every update step, and completion).
   *  Returns an unsubscribe function. */
  onChange(key: string, cb: (value: number) => void): () => void;
}

const DEFAULT_DURATION = 0.3;

interface Channel {
  value: number;
  // Tween state; meaningful only while running.
  running: boolean;
  start: number;
  target: number;
  elapsed: number;
  duration: number;
  ease: Ease;
  listeners: Array<(value: number) => void> | null;
}

function makeChannel(): Channel {
  return {
    value: 0,
    running: false,
    start: 0,
    target: 0,
    elapsed: 0,
    duration: 0,
    ease: cubicInOut,
    listeners: null,
  };
}

export function createTweens(): Tweens {
  const channels = new Map<string, Channel>();
  let runningCount = 0;

  function channel(key: string): Channel {
    let c = channels.get(key);
    if (c === undefined) {
      c = makeChannel();
      channels.set(key, c);
    }
    return c;
  }

  function emit(c: Channel): void {
    const ls = c.listeners;
    if (ls === null) return;
    for (let i = 0; i < ls.length; i++) {
      const cb = ls[i];
      if (cb !== undefined) cb(c.value);
    }
  }

  function stop(c: Channel): void {
    if (c.running) {
      c.running = false;
      runningCount--;
    }
  }

  function setValue(c: Channel, value: number): void {
    stop(c);
    if (c.value !== value) {
      c.value = value;
      emit(c);
    }
  }

  return {
    to(key, target, opts) {
      const c = channel(key);
      const duration = opts?.duration ?? DEFAULT_DURATION;
      if (!(duration > 0) || target === c.value) {
        setValue(c, target);
        return;
      }
      if (!c.running) runningCount++;
      c.running = true;
      c.start = c.value; // retarget from the CURRENT value — never a snap
      c.target = target;
      c.elapsed = 0;
      c.duration = duration;
      c.ease = opts?.ease ?? cubicInOut;
    },

    set(key, value) {
      setValue(channel(key), value);
    },

    get(key) {
      return channels.get(key)?.value ?? 0;
    },

    update(dt) {
      if (!(dt > 0) || runningCount === 0) return; // update(0) is a strict no-op
      for (const c of channels.values()) {
        if (!c.running) continue;
        c.elapsed += dt;
        if (c.elapsed >= c.duration) {
          // Land EXACTLY on the target — endpoint exactness is what lets `spread -> 1`
          // restore byte-identical coordinates downstream (§4).
          c.running = false;
          runningCount--;
          c.value = c.target;
        } else {
          const e = c.ease(c.elapsed / c.duration);
          c.value = c.start + (c.target - c.start) * e;
        }
        emit(c);
      }
    },

    active() {
      return runningCount > 0;
    },

    onChange(key, cb) {
      const c = channel(key);
      if (c.listeners === null) c.listeners = [];
      c.listeners.push(cb);
      return () => {
        const ls = c.listeners;
        if (ls === null) return;
        const at = ls.indexOf(cb);
        if (at >= 0) ls.splice(at, 1);
      };
    },
  };
}
