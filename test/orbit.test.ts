/* The simple orbit/dolly camera, driven headlessly through the same stub surface as the
 * flycam tests. It shares the flycam's one load-bearing idea — a drag rotates the rig
 * RIGIDLY about the target — so it inherits the same pinned measurements: the target moves
 * 0.000 and aim·toTarget stays 1.0000.
 */
import { describe, expect, it } from 'vitest';
import { PerspectiveCamera, Vector3 } from 'three';
import { createOrbitCam, ORBIT_DEFAULTS } from '../src/cameras/orbit';
import type { OrbitCam } from '../src/cameras/orbit';
import type { ControlSurface } from '../src/cameras/types';

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

interface Rig { camera: PerspectiveCamera; cam: OrbitCam; dom: Surface }

function rig(): Rig {
  const camera = new PerspectiveCamera(55, 1.6, 0.5, 1e6);
  camera.position.set(0, 0, 4000);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  const dom = new Surface();
  const cam = createOrbitCam(camera, { ...ORBIT_DEFAULTS }, {
    zoomRange: () => [10, 1e6] as const,
  });
  cam.attach(dom);
  cam.setTarget(new Vector3(0, 0, 0), false);
  return { camera, cam, dom };
}

const aimDot = (c: PerspectiveCamera, t: Vector3): number =>
  new Vector3(0, 0, -1).applyQuaternion(c.quaternion)
    .dot(t.clone().sub(c.position).normalize());

describe('orbit camera', () => {
  it('a drag orbits the rig rigidly: target drift 0.000, aim·toTarget 1.0000', () => {
    const { camera, cam, dom } = rig();
    const t0 = cam.target.clone();
    const r0 = cam.radius;
    const p0 = camera.position.clone();
    dom.fire('pointerdown', { button: 0, clientX: 300, clientY: 300, pointerId: 7 });
    for (let i = 1; i <= 30; i++) {
      dom.fire('pointermove', { clientX: 300 + i * 6, clientY: 300 - i * 2 });
    }
    dom.fire('pointerup', { button: 0, clientX: 480, clientY: 240, pointerId: 7 });
    expect(camera.position.distanceTo(p0)).toBeGreaterThan(1);   // the camera moved…
    expect(cam.target.distanceTo(t0)).toBe(0);                   // …the target did not: 0.000
    expect(aimDot(camera, cam.target)).toBeCloseTo(1, 4);        // 1.0000
    expect(Math.abs(cam.radius - r0) / r0).toBeLessThan(1e-6);   // rigid = radius intact
  });

  it('click vs drag is displacement from the press point, latched', () => {
    const { cam, dom } = rig();
    dom.fire('pointerdown', { button: 0, clientX: 50, clientY: 50 });
    dom.fire('pointermove', { clientX: 51, clientY: 51 });       // jitter under DRAG_SLOP
    expect(cam.consumeClick({ button: 0, clientX: 51, clientY: 51 })).toBe(true);

    dom.fire('pointerdown', { button: 0, clientX: 50, clientY: 50 });
    dom.fire('pointermove', { clientX: 150, clientY: 50 });
    dom.fire('pointermove', { clientX: 50, clientY: 50 });       // back where it began
    expect(cam.consumeClick({ button: 0, clientX: 50, clientY: 50 })).toBe(false);
  });

  it('the wheel dollies the radius by one ratio per notch, target untouched', () => {
    const { cam, dom } = rig();
    const r0 = cam.radius;
    dom.fire('wheel', { deltaY: 120, preventDefault() {} });
    expect(cam.radius / r0).toBeCloseTo(ORBIT_DEFAULTS.wheelK, 6);
    dom.fire('wheel', { deltaY: -120, preventDefault() {} });
    expect(cam.radius / r0).toBeCloseTo(1, 6);
    expect(cam.target.length()).toBe(0);
  });

  it('dolly clamps to the zoom range; the zoom slider runs a log curve', () => {
    const { cam } = rig();
    cam.dolly(1);
    expect(cam.radius).toBeCloseTo(10, 9);
    cam.dolly(1e9);
    expect(cam.radius).toBeCloseTo(1e6, 3);
    cam.setZoom(0.25); const r1 = cam.radius;
    cam.setZoom(0.5); const r2 = cam.radius;
    cam.setZoom(0.75); const r3 = cam.radius;
    expect(r1 / r2).toBeCloseTo(r2 / r3, 9);   // even travel, even ratio
    cam.setZoom(0.61);
    expect(cam.zoomOf()).toBeCloseTo(0.61, 9);
  });

  it('home frames the live extent and levels against world up', () => {
    const { camera, cam } = rig();
    cam.home({ center: [-250, 80, 10], radius: 900 });
    expect(cam.target.distanceTo(new Vector3(-250, 80, 10))).toBe(0);
    expect(aimDot(camera, cam.target)).toBeCloseTo(1, 4);
    const expected = (900 / Math.tan((55 * Math.PI) / 360)) * 1.1;
    expect(cam.radius).toBeCloseTo(expected, 6);
  });

  it('update does nothing without input, and reports the live radius', () => {
    const { camera, cam } = rig();
    const p0 = camera.position.clone();
    const telem = cam.update(0.016);
    expect(camera.position.distanceTo(p0)).toBe(0);
    expect(telem.radius).toBeCloseTo(4000, 9);
    expect(telem.speed).toBe(0);
    expect(telem.flying).toBe(false);
  });

  it('setTarget with keepRadius re-dollies onto the new pivot', () => {
    const { cam } = rig();
    const r0 = cam.radius;
    cam.setTarget(new Vector3(500, 0, 0), true);
    expect(cam.target.distanceTo(new Vector3(500, 0, 0))).toBe(0);
    expect(Math.abs(cam.radius - r0) / r0).toBeLessThan(1e-9);
  });

  it('dispose removes every listener; a re-attach detaches the old surface first', () => {
    const { camera, cam, dom } = rig();
    const other = new Surface();
    cam.attach(other);                       // idempotent re-attach
    expect(dom.count('pointerdown') + dom.count('pointermove')
      + dom.count('pointerup') + dom.count('wheel')).toBe(0);
    cam.dispose();
    expect(other.count('pointerdown') + other.count('pointermove')
      + other.count('pointerup') + other.count('wheel')).toBe(0);
    const p0 = camera.position.clone();
    other.fire('pointerdown', { button: 0, clientX: 0, clientY: 0 });
    other.fire('pointermove', { clientX: 400, clientY: 0 });
    expect(camera.position.distanceTo(p0)).toBe(0);
  });
});
