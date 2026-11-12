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
  it('W and S go FORWARD and BACK, not up and down', () => {
    expect(axis(press('KeyW'))).toBe('forward');
    expect(axis(press('KeyS'))).toBe('-forward');
  });

  it('X and C move UP and DOWN; the old right-hand throttle keys are inert', () => {
    expect(axis(press('KeyX'))).toBe('up');
    expect(axis(press('KeyC'))).toBe('-up');
    // ShiftRight/AltRight/ControlRight used to be the forward/back throttle; forward is W/S now,
    // so those keys move nothing at all.
    const still = (code: string): number => {
      const m = press(code);
      return Math.hypot(m.f, m.r, m.u);
    };
    expect(still('ShiftRight')).toBeLessThan(1e-9);
    expect(still('AltRight')).toBeLessThan(1e-9);
    expect(still('ControlRight')).toBeLessThan(1e-9);
  });

  it('A and D still strafe, X and C lift', () => {
    expect(axis(press('KeyD'))).toBe('right');
    expect(axis(press('KeyA'))).toBe('-right');
    expect(axis(press('KeyX'))).toBe('up');
    expect(axis(press('KeyC'))).toBe('-up');
  });
});

describe('flycam speed', () => {
  /** Fly forward (W) 90 frames (optionally with the speed-lock toggled on), coast 120 more;
   *  distance covered. `mod` is toggled on before the run and off after — CapsLock semantics. */
  const fly = (mod: string | null, roomAt: (p: readonly [number, number, number]) => number)
  : number => {
    const { camera, cam, keys } = rig({ roomAt }, {}, [0, 0, 9000]);
    const p0 = camera.position.clone();
    if (mod) keys.fire('keydown', key(mod));       // CapsLock keydown = lock ON
    keys.fire('keydown', key('KeyW'));
    for (let t = 0; t < 90; t++) cam.update(0.016);
    keys.fire('keyup', key('KeyW'));
    if (mod) keys.fire('keyup', key(mod));          // CapsLock keyup = lock OFF
    for (let t = 0; t < 120; t++) cam.update(0.016);
    return camera.position.distanceTo(p0);
  };
  // room shrinks as you close on the origin, which is what flying at a cluster feels like
  const closing = (p: readonly [number, number, number]): number =>
    Math.max(60, Math.hypot(p[0], p[1], p[2]) - 200);
  const flat = (): number => 900;

  it('CapsLock locks speed through an approach; plain flight brakes', () => {
    const plain = fly(null, closing);
    const held = fly('CapsLock', closing);
    expect(held).toBeGreaterThan(plain * 1.3);
  });

  it('with nothing to brake for, locking is NOT a boost', () => {
    const pf = fly(null, flat);
    const hf = fly('CapsLock', flat);
    expect(hf).toBeLessThan(pf * 1.15);
  });

  it('plain flight is deterministic: two identical runs land identically', () => {
    expect(Math.abs(fly(null, flat) - fly(null, flat))).toBeLessThan(1e-9);
  });

  it('releasing the CapsLock lock re-latches from wherever you then are', () => {
    // The latch has to clear when the lock toggles off, or a speed grabbed far out would follow
    // you forever.
    const { camera, cam, keys } = rig(
      { roomAt: (p) => Math.max(60, Math.hypot(p[0], p[1], p[2]) - 200) }, {}, [0, 0, 9000]);
    const step = (codes: string[], n: number): number => {
      for (const c of codes) keys.fire('keydown', key(c));
      let d = 0;
      for (let t = 0; t < n; t++) {
        const p = camera.position.clone();
        cam.update(0.016);
        d += camera.position.distanceTo(p);
      }
      for (const c of codes) keys.fire('keyup', key(c));
      return d;
    };
    step(['CapsLock', 'KeyW'], 90);          // close in while the speed is locked
    const near = camera.position.length();
    const after = step(['KeyW'], 30);          // now fly WITHOUT the lock
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
    keys.fire('keydown', key('KeyW'));
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
    keys.fire('keydown', key('KeyW'));
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
      keys.fire('keydown', key('KeyW'));
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

describe('flycam terrain collision (surfaceAt — the planet floor)', () => {
  const R = 1000;
  const PEAK = 500; // the ridge rises to R + PEAK toward world +X; the -X hemisphere stays the sphere
  /** Terrain floor: flat sphere (R) on the −X side, smoothly rising to a tall ridge/plateau toward +X.
   *  `dx,dy,dz` arrive as a UNIT direction (as the flycam passes them), so dx IS the x-component. */
  const terrain = (dx: number, dy: number, dz: number): number => {
    const len = Math.hypot(dx, dy, dz) || 1;
    const ax = Math.max(0, dx / len);
    const t = Math.min(1, ax / 0.4);
    return R + PEAK * (t * t * (3 - 2 * t)); // smoothstep ramp to the plateau by ax = 0.4
  };
  const withTerrain = (): FlycamOptions['surfaceAt'] =>
    () => ({ center: [0, 0, 0] as const, radius: R, terrain });
  const sphereOnly = (): FlycamOptions['surfaceAt'] =>
    () => ({ center: [0, 0, 0] as const, radius: R });

  it('skimming HORIZONTALLY into a ridge rides up and over it — never through', () => {
    // Camera just above the flat surface at the +Z point, looking at the planet centre, so world +X is
    // the ship's RIGHT axis: KeyD strafes TANGENTIALLY, straight at the ridge. The old backstop sampled
    // the floor under the camera's OLD (pre-move) direction, so a tangential step — which barely changes
    // the radius but sweeps the radial direction into much taller terrain — was never caught and the
    // camera flew clean through every peak. The floor must follow the camera's ACTUAL direction.
    const { camera, cam, keys } = rig({ surfaceAt: withTerrain() }, {}, [0, 0, R + 60]);
    keys.fire('keydown', key('KeyD'));
    let minClear = Infinity; // min (cameraRadius − terrainSurface) over the whole flight
    let maxDir = 0;          // how far into the ridge (x-component of the radial dir) we reached
    for (let t = 0; t < 400; t++) {
      const tel = cam.update(0.016);
      const rad = camera.position.length();
      const dir = camera.position.clone().normalize();
      const surf = terrain(dir.x, dir.y, dir.z);
      minClear = Math.min(minClear, rad - surf);
      maxDir = Math.max(maxDir, dir.x);
      expect(tel.altitude).toBeGreaterThan(0); // telemetry never reports sinking below the surface
    }
    expect(maxDir).toBeGreaterThan(0.35);            // genuinely skimmed deep into the tall ridge
    expect(minClear).toBeGreaterThan(-1e-6);         // and NEVER dipped below the terrain surface
    expect(camera.position.length()).toBeGreaterThan(R + PEAK * 0.6); // it rode UP the ridge, not through
  });

  it('descending straight onto a peak stops ON the peak, not at the base sphere', () => {
    // Over the plateau (dir ≈ +X, terrain = R + PEAK). Looking at the centre ⇒ KeyW is radial-IN (a
    // vertical descent). The camera must settle on the terrain (≈ R + PEAK + hover), never sink to R.
    const { camera, cam, keys } = rig({ surfaceAt: withTerrain() }, {}, [R + 600, 0, 0]);
    const start = camera.position.length();
    keys.fire('keydown', key('KeyW'));
    let minRad = Infinity;
    for (let t = 0; t < 600; t++) {
      cam.update(0.016);
      minRad = Math.min(minRad, camera.position.length());
    }
    const end = camera.position.length();
    expect(end).toBeLessThan(start);            // it descended
    expect(minRad).toBeGreaterThan(R + PEAK - 1e-3); // but never crossed below the peak surface (≈1500)
    expect(end).toBeLessThan(R + PEAK + 120);   // and settled just above it, not out in space
  });

  it('stage-A sphere floor (no terrain sampler) still stops a radial descent at radius+hover', () => {
    const { camera, cam, keys } = rig({ surfaceAt: sphereOnly() }, {}, [0, 0, R + 600]);
    keys.fire('keydown', key('KeyW')); // toward the centre ⇒ descend onto the bare sphere
    let minRad = Infinity;
    for (let t = 0; t < 600; t++) {
      cam.update(0.016);
      minRad = Math.min(minRad, camera.position.length());
    }
    expect(minRad).toBeGreaterThan(R - 1e-3);   // never below the sphere
    expect(minRad).toBeLessThan(R + 80);        // and it did settle down onto it (hover ≈ 40)
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
    keys.fire('keydown', key('KeyW'));
    for (let t = 0; t < 30; t++) cam.update(0.016);
    const editor = { closest: () => ({}) };
    keys.fire('keyup', key('KeyW', { target: editor }));
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
