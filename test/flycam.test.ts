/* The flycam, driven headlessly — no renderer, no DOM, synthetic events on stub surfaces,
 * assertions on raw camera numbers. These invariants are ports of what the predecessor
 * (vault-site-3d test/flycam.test.mjs, README §"The controls") PINNED after shipping the
 * bugs, plus the two measurements its README quotes as the contract of drag-orbits:
 * the target moves 0.000 and aim·toTarget stays 1.0000.
 */
import { describe, expect, it } from 'vitest';
import { PerspectiveCamera, Vector3 } from 'three';
import { createFlycam, FLYCAM_DEFAULTS } from '../src/cameras/flycam';
import type { Flycam, FlycamConfig, FlycamOptions } from '../src/cameras/flycam';
import type { ControlSurface } from '../src/cameras/types';

/** The smallest surface a controller will accept: store listeners, fire synthetic events. */
class Surface implements ControlSurface {
  private m = new Map<string, Array<(ev: Event) => void>>();
  addEventListener(type: string, listener: (ev: Event) => void): void {
    this.m.set(type, [...(this.m.get(type) ?? []), listener]);
  }
  removeEventListener(type: string, listener: (ev: Event) => void): void {
    this.m.set(type, (this.m.get(type) ?? []).filter((l) => l !== listener));
  }
  setPointerCapture(): void {}
  releasePointerCapture(): void {}
  fire(type: string, ev: unknown): void {
    for (const fn of [...(this.m.get(type) ?? [])]) fn(ev as Event);
  }
  count(type: string): number { return (this.m.get(type) ?? []).length; }
}

const key = (code: string, extra: Record<string, unknown> = {}): unknown =>
  ({ code, key: code, target: null, preventDefault() {}, ...extra });

interface Rig { camera: PerspectiveCamera; cam: Flycam; dom: Surface; keys: Surface }

function rig(
  over: Partial<FlycamOptions> = {},
  cfgOver: Partial<FlycamConfig> = {},
  pos: [number, number, number] = [0, 0, 4000],
): Rig {
  const camera = new PerspectiveCamera(55, 1.6, 0.5, 1e6);
  camera.position.set(pos[0], pos[1], pos[2]);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  const dom = new Surface();
  const keys = new Surface();
  const cam = createFlycam(camera, { ...FLYCAM_DEFAULTS, ...cfgOver }, {
    roomAt: () => 900,
    softStop: () => 50,
    zoomRange: () => [10, 1e6] as const,
    keyboard: keys,
    reducedMotion: false,
    ...over,
  });
  cam.attach(dom);
  cam.setTarget(new Vector3(0, 0, 0), false);
  return { camera, cam, dom, keys };
}

const fwd = (c: PerspectiveCamera): Vector3 =>
  new Vector3(0, 0, -1).applyQuaternion(c.quaternion);
/** aim·toTarget — 1.0000 means the nose points exactly at the target. */
const aimDot = (c: PerspectiveCamera, t: Vector3): number =>
  fwd(c).dot(t.clone().sub(c.position).normalize());

/** Hold `code` for a simulated stretch and report the displacement in the ship's own axes. */
function press(code: string, seconds = 1.2): { f: number; r: number; u: number } {
  const { camera, cam, keys } = rig();
  const p0 = camera.position.clone();
  keys.fire('keydown', key(code));
  for (let t = 0; t < seconds / 0.016; t++) cam.update(0.016);
  keys.fire('keyup', key(code));
  const d = camera.position.clone().sub(p0);
  return {
    f: d.dot(fwd(camera)),
    r: d.dot(new Vector3(1, 0, 0).applyQuaternion(camera.quaternion)),
    u: d.dot(new Vector3(0, 1, 0).applyQuaternion(camera.quaternion)),
  };
}
/** Which axis dominated, and in which direction. */
function axis(m: { f: number; r: number; u: number }): string {
  const rows: Array<[string, number]> = [['forward', m.f], ['right', m.r], ['up', m.u]];
  rows.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const top = rows[0]!;
  return (top[1] > 0 ? '' : '-') + top[0];
}

describe('flycam keymap', () => {
  it('W and S go UP and DOWN, not forward and back', () => {
    expect(axis(press('KeyW'))).toBe('up');
    expect(axis(press('KeyS'))).toBe('-up');
  });

  it('the right-hand modifiers are the throttle', () => {
    expect(axis(press('ShiftRight'))).toBe('forward');
    expect(axis(press('AltRight'))).toBe('-forward');
    expect(axis(press('ControlRight'))).toBe('-forward');
  });

  it('A and D still strafe, R and F still lift', () => {
    expect(axis(press('KeyD'))).toBe('right');
    expect(axis(press('KeyA'))).toBe('-right');
    expect(axis(press('KeyR'))).toBe('up');
    expect(axis(press('KeyF'))).toBe('-up');
  });
});

describe('flycam speed', () => {
  /** Fly forward 90 frames (optionally holding a modifier), coast 120 more; distance covered. */
  const fly = (mod: string | null, roomAt: (p: readonly [number, number, number]) => number)
  : number => {
    const { camera, cam, keys } = rig({ roomAt }, {}, [0, 0, 9000]);
    const p0 = camera.position.clone();
    if (mod) keys.fire('keydown', key(mod, { key: 'Shift' }));
    keys.fire('keydown', key('ShiftRight', { key: 'Shift' }));
    for (let t = 0; t < 90; t++) cam.update(0.016);
    keys.fire('keyup', key('ShiftRight', { key: 'Shift' }));
    if (mod) keys.fire('keyup', key(mod, { key: 'Shift' }));
    for (let t = 0; t < 120; t++) cam.update(0.016);
    return camera.position.distanceTo(p0);
  };
  // room shrinks as you close on the origin, which is what flying at a cluster feels like
  const closing = (p: readonly [number, number, number]): number =>
    Math.max(60, Math.hypot(p[0], p[1], p[2]) - 200);
  const flat = (): number => 900;

  it('LEFT shift holds speed through an approach; plain flight brakes', () => {
    const plain = fly(null, closing);
    const held = fly('ShiftLeft', closing);
    expect(held).toBeGreaterThan(plain * 1.3);
  });

  it('with nothing to brake for, holding is NOT a boost', () => {
    const pf = fly(null, flat);
    const hf = fly('ShiftLeft', flat);
    expect(hf).toBeLessThan(pf * 1.15);
  });

  it('plain flight is deterministic: two identical runs land identically', () => {
    expect(Math.abs(fly(null, flat) - fly(null, flat))).toBeLessThan(1e-9);
  });

  it('releasing left shift re-latches from wherever you then are', () => {
    // The latch has to clear on keyup, or a speed grabbed far out would follow you forever.
    const { camera, cam, keys } = rig(
      { roomAt: (p) => Math.max(60, Math.hypot(p[0], p[1], p[2]) - 200) }, {}, [0, 0, 9000]);
    const step = (codes: string[], n: number): number => {
      for (const c of codes) keys.fire('keydown', key(c, { key: 'Shift' }));
      let d = 0;
      for (let t = 0; t < n; t++) {
        const p = camera.position.clone();
        cam.update(0.016);
        d += camera.position.distanceTo(p);
      }
      for (const c of codes) keys.fire('keyup', key(c, { key: 'Shift' }));
      return d;
    };
    step(['ShiftLeft', 'ShiftRight'], 90);          // close in while holding
    const near = camera.position.length();
    const after = step(['ShiftRight'], 30);          // now fly WITHOUT the hold
    const expected = Math.max(60, near - 200) * FLYCAM_DEFAULTS.speedK * 30 * 0.016;
    expect(after).toBeLessThan(expected * 2.5);
  });
});

describe('flycam proximity damping', () => {
  it('damps toward a floor, never gates to zero: step > 0 whenever input is held', () => {
    // Deep inside a dense region — room tiny, well under the stop radius — the old gate
    // multiplied the step by exactly nothing and W went dead. The clamp floor (0.16) is the
    // invariant: every held frame moves the ship.
    const { camera, cam, keys } = rig(
      { roomAt: (p) => Math.max(1, Math.hypot(p[0], p[1], p[2]) - 100) }, {}, [0, 0, 300]);
    keys.fire('keydown', key('ShiftRight', { key: 'Shift' }));
    let sawBlocked = false;
    const prev = camera.position.clone();
    for (let t = 0; t < 400; t++) {
      const telem = cam.update(0.016);
      if (t > 10) {
        expect(camera.position.distanceTo(prev)).toBeGreaterThan(0);
      }
      prev.copy(camera.position);
      if (telem.blocked) sawBlocked = true;
    }
    // ...and the telemetry admitted the damp was pinning the step (DESIGN.md §10.7).
    expect(sawBlocked).toBe(true);
  });

  it('a constant tiny room (no gradient ahead) never freezes the ship either', () => {
    const { camera, cam, keys } = rig({ roomAt: () => 5 });
    keys.fire('keydown', key('ShiftRight', { key: 'Shift' }));
    const prev = camera.position.clone();
    for (let t = 0; t < 60; t++) {
      cam.update(0.016);
      if (t > 10) expect(camera.position.distanceTo(prev)).toBeGreaterThan(0);
      prev.copy(camera.position);
    }
  });

  it('closing on a cluster is slower than open space — the damp exists', () => {
    const run = (roomAt: (p: readonly [number, number, number]) => number): number => {
      const { camera, cam, keys } = rig({ roomAt }, {}, [0, 0, 300]);
      const p0 = camera.position.clone();
      keys.fire('keydown', key('ShiftRight', { key: 'Shift' }));
      for (let t = 0; t < 90; t++) cam.update(0.016);
      return camera.position.distanceTo(p0);
    };
    const damped = run((p) => Math.max(1, Math.hypot(p[0], p[1], p[2]) - 100));
    const open = run(() => 900);
    expect(damped).toBeLessThan(open * 0.5);
    expect(damped).toBeGreaterThan(0);
  });
});

describe('flycam drag = orbit the universe', () => {
  it('rotates the rig rigidly: target drift 0.000, aim·toTarget 1.0000, radius kept', () => {
    const { camera, cam, dom, keys } = rig();
    const t0 = cam.target.clone();
    const r0 = cam.radius;
    dom.fire('pointerdown', { button: 0, clientX: 200, clientY: 200, pointerId: 1 });
    for (let i = 1; i <= 25; i++) {
      keys.fire('pointermove', { clientX: 200 + i * 7, clientY: 200 + i * 4 });
    }
    // the predecessor measured these on the live viewer; here they are exact assertions
    expect(cam.target.distanceTo(t0)).toBe(0);                       // 0.000
    expect(aimDot(camera, cam.target)).toBeCloseTo(1, 4);            // 1.0000
    expect(Math.abs(cam.radius - r0) / r0).toBeLessThan(1e-6);
    expect(cam.consumeClick({ button: 0, clientX: 375, clientY: 300 })).toBe(false);
  });

  it('a press that never clears DRAG_SLOP stays a click; a latched drag never becomes one', () => {
    const { cam, dom, keys } = rig();
    // jittery press: 1–2 px wobble around the press point must remain a click
    dom.fire('pointerdown', { button: 0, clientX: 100, clientY: 100 });
    keys.fire('pointermove', { clientX: 101, clientY: 100 });
    keys.fire('pointermove', { clientX: 100, clientY: 101 });
    keys.fire('pointermove', { clientX: 99, clientY: 100 });
    expect(cam.consumeClick({ button: 0, clientX: 99, clientY: 100 })).toBe(true);
    // a long drag that returns to its start must NOT read as a click — isDrag latches
    dom.fire('pointerdown', { button: 0, clientX: 100, clientY: 100 });
    keys.fire('pointermove', { clientX: 180, clientY: 100 });
    keys.fire('pointermove', { clientX: 100, clientY: 100 });
    expect(cam.consumeClick({ button: 0, clientX: 100, clientY: 100 })).toBe(false);
  });
});

describe('flycam turning and flying', () => {
  it('arrows swing the NOSE: position still, target follows round at the same radius', () => {
    const { camera, cam, keys } = rig();
    const p0 = camera.position.clone();
    const t0 = cam.target.clone();
    keys.fire('keydown', key('ArrowLeft'));
    for (let t = 0; t < 60; t++) cam.update(0.016);
    keys.fire('keyup', key('ArrowLeft'));
    expect(camera.position.distanceTo(p0)).toBe(0);
    expect(cam.target.distanceTo(t0)).toBeGreaterThan(1);
    expect(Math.abs(cam.radius - 4000) / 4000).toBeLessThan(1e-6);
    expect(aimDot(camera, cam.target)).toBeCloseTo(1, 4);
  });

  it('flying moves camera AND target together, so the orbit radius survives', () => {
    const { cam, keys } = rig();
    const r0 = cam.radius;
    const t0 = cam.target.clone();
    keys.fire('keydown', key('KeyW'));
    for (let t = 0; t < 75; t++) cam.update(0.016);
    keys.fire('keyup', key('KeyW'));
    expect(cam.target.distanceTo(t0)).toBeGreaterThan(1);      // the target came along
    expect(Math.abs(cam.radius - r0) / r0).toBeLessThan(1e-6); // at the same radius
  });

  it('a release is honoured even if focus moved into a text box', () => {
    // the old bug: gating keyup on focus swallowed the release and the ship flew on forever
    const { camera, cam, keys } = rig();
    keys.fire('keydown', key('ShiftRight', { key: 'Shift' }));
    for (let t = 0; t < 30; t++) cam.update(0.016);
    const editor = { closest: () => ({}) };
    keys.fire('keyup', key('ShiftRight', { key: 'Shift', target: editor }));
    for (let t = 0; t < 200; t++) cam.update(0.016);
    const settled = camera.position.clone();
    for (let t = 0; t < 120; t++) cam.update(0.016);
    expect(camera.position.distanceTo(settled)).toBeLessThan(1);
  });

  it('a key pressed WHILE typing is ignored (but blur clears everything held)', () => {
    const { camera, cam, keys } = rig();
    const editor = { closest: () => ({}) };
    keys.fire('keydown', key('KeyW', { target: editor }));
    for (let t = 0; t < 30; t++) cam.update(0.016);
    expect(camera.position.distanceTo(new Vector3(0, 0, 4000))).toBeLessThan(1e-9);

    keys.fire('keydown', key('KeyW'));
    for (let t = 0; t < 10; t++) cam.update(0.016);
    keys.fire('blur', {});
    for (let t = 0; t < 200; t++) cam.update(0.016);
    const settled = camera.position.clone();
    for (let t = 0; t < 60; t++) cam.update(0.016);
    expect(camera.position.distanceTo(settled)).toBeLessThan(1);
  });
});

describe('flycam dolly and zoom', () => {
  it('the wheel dollies the radius on a log curve: one notch = one ratio', () => {
    const { cam, dom } = rig();
    const r0 = cam.radius;
    dom.fire('wheel', { deltaY: 100, target: null, preventDefault() {} });
    expect(cam.radius / r0).toBeCloseTo(1.14, 6);
    dom.fire('wheel', { deltaY: -100, target: null, preventDefault() {} });
    expect(cam.radius / r0).toBeCloseTo(1, 6);
    expect(cam.target.length()).toBe(0);       // dolly never moves the target
  });

  it('even slider travel means even ratio, and zoomOf inverts setZoom', () => {
    const { cam } = rig();
    cam.setZoom(0.25); const r1 = cam.radius;
    cam.setZoom(0.5); const r2 = cam.radius;
    cam.setZoom(0.75); const r3 = cam.radius;
    expect(r1 / r2).toBeCloseTo(r2 / r3, 9);
    cam.setZoom(0.37);
    expect(cam.zoomOf()).toBeCloseTo(0.37, 9);
  });

  it('dolly clamps to the zoom range', () => {
    const { cam } = rig();
    cam.dolly(1);
    expect(cam.radius).toBeCloseTo(10, 9);
    cam.dolly(1e9);
    expect(cam.radius).toBeCloseTo(1e6, 3);
  });
});

describe('flycam goTo / home / dispose', () => {
  it('goTo is a dt-driven tween: no clock, so two runs replay identically', () => {
    const dest = new Vector3(100, 50, -200);
    const run = (): Vector3 => {
      const { camera, cam } = rig();
      cam.goTo(dest, 300, 900);
      for (let t = 0; t < 61; t++) cam.update(0.015);   // 61 × 15 ms ≥ 900 ms
      expect(cam.tweening).toBe(false);
      return camera.position.clone();
    };
    const a = run(), b = run();
    expect(a.x).toBe(b.x); expect(a.y).toBe(b.y); expect(a.z).toBe(b.z);
    expect(a.distanceTo(dest)).toBeCloseTo(300, 6);
  });

  it('goTo lands aimed at the destination, target on it; input interrupts the tween', () => {
    const dest = new Vector3(-500, 120, 80);
    const { camera, cam, keys } = rig();
    cam.goTo(dest, 250, 600);
    for (let t = 0; t < 50; t++) cam.update(0.016);
    expect(cam.target.distanceTo(dest)).toBeLessThan(1e-6);
    expect(aimDot(camera, cam.target)).toBeCloseTo(1, 4);

    cam.goTo(new Vector3(0, 0, 0), 500, 900);
    expect(cam.tweening).toBe(true);
    keys.fire('keydown', key('KeyW'));
    expect(cam.tweening).toBe(false);          // a held key cancels the flight
    keys.fire('keyup', key('KeyW'));
  });

  it('home frames the live extent: target at centre, whole radius in view', () => {
    const { camera, cam } = rig();
    cam.home({ center: [100, -40, 7], radius: 500 });
    expect(cam.target.distanceTo(new Vector3(100, -40, 7))).toBe(0);
    expect(aimDot(camera, cam.target)).toBeCloseTo(1, 4);
    const expected = (500 / Math.tan((55 * Math.PI) / 360)) * 1.1;
    expect(cam.radius).toBeCloseTo(expected, 6);
  });

  it('dispose removes every listener it attached', () => {
    const { camera, cam, dom, keys } = rig();
    cam.dispose();
    expect(dom.count('pointerdown') + dom.count('wheel')).toBe(0);
    expect(keys.count('keydown') + keys.count('keyup') + keys.count('blur')
      + keys.count('pointermove')).toBe(0);
    keys.fire('keydown', key('KeyW'));
    for (let t = 0; t < 30; t++) cam.update(0.016);
    expect(camera.position.distanceTo(new Vector3(0, 0, 4000))).toBeLessThan(1e-9);
  });
});
