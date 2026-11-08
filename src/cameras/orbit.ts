/* The simple camera: orbit + dolly, nothing else. For consumers who want a graph viewer, not
 * a game — embedders, screenshot rigs, docs pages. Same `CameraController` contract as the
 * flycam, so the host can swap one for the other without touching its loop.
 *
 * It shares the flycam's one non-obvious idea: a drag rotates the RIG rigidly about the
 * target — position and orientation by the same quaternion — so the aim and the roll survive
 * for free, with no spherical-coordinate bookkeeping, no gimbal pole at the zenith, and no
 * accumulated-angle state to desynchronise. The target never moves during a drag (exactly
 * 0.000 drift, pinned in test/orbit.test.ts) and the wheel dollies the radius on the same log
 * curve the flycam uses, so even travel means even ratio.
 */

import { MathUtils, Matrix4, Quaternion, Vector3 } from 'three';
import { DEFAULT_FOV } from '../field/constants';
import { DRAG_SLOP } from './types';
import type {
  CameraController, CameraTelemetry, ControlSurface, Extent,
  PointerEventLike, RigCamera, WheelEventLike,
} from './types';

export interface OrbitConfig {
  /** Drag sensitivity, rad per CSS px. */
  mouseSens: number;
  /** Wheel dolly ratio per notch. */
  wheelK: number;
}

export const ORBIT_DEFAULTS: Readonly<OrbitConfig> = { mouseSens: 0.0052, wheelK: 1.14 };

export interface OrbitOptions {
  /** [closest, farthest] dolly radius, world units. */
  zoomRange: () => readonly [number, number];
  /** Fired on every user input (drag, wheel). */
  onInput?: () => void;
}

export type OrbitCam = CameraController;

export function createOrbitCam(
  camera: RigCamera, cfg: OrbitConfig, opts: OrbitOptions,
): OrbitCam {
  const { zoomRange } = opts;
  const onInput = opts.onInput ?? ((): void => {});

  const WORLD_UP = new Vector3(0, 1, 0);
  const target = new Vector3();
  const R = new Vector3(), U = new Vector3();
  const Q = new Quaternion(), OFF = new Vector3(), C = new Vector3();
  const M = new Matrix4();
  const telem: CameraTelemetry =
    { speed: 0, room: Infinity, flying: false, radius: 0, blocked: false, held: 0 };

  const radius = (): number => camera.position.distanceTo(target);

  /** Rigid rig rotation about the target — same math as the flycam, same reason: a rigid
   *  rotation cannot change where the nose points relative to the target. */
  function orbit(ax: number, ay: number): void {
    U.set(0, 1, 0).applyQuaternion(camera.quaternion);
    OFF.subVectors(camera.position, target);
    Q.setFromAxisAngle(U, ax);
    OFF.applyQuaternion(Q); camera.quaternion.premultiply(Q);
    R.set(1, 0, 0).applyQuaternion(camera.quaternion);
    Q.setFromAxisAngle(R, ay);
    OFF.applyQuaternion(Q); camera.quaternion.premultiply(Q);
    camera.quaternion.normalize();
    camera.position.copy(target).add(OFF);
  }

  function dolly(r: number): void {
    const [lo, hi] = zoomRange();
    const want = MathUtils.clamp(r, lo, hi);
    OFF.subVectors(camera.position, target);
    if (OFF.length() < 1e-5) OFF.set(0, 0, 1);
    OFF.normalize().multiplyScalar(want);
    camera.position.copy(target).add(OFF);
  }
  const zoomOf = (): number => {
    const [lo, hi] = zoomRange();
    return MathUtils.clamp(Math.log(radius() / hi) / Math.log(lo / hi), 0, 1);
  };
  const setZoom = (t: number): void => {
    const [lo, hi] = zoomRange();
    dolly(hi * Math.pow(lo / hi, MathUtils.clamp(t, 0, 1)));
  };

  function home(extent: Extent): void {
    C.set(extent.center[0], extent.center[1], extent.center[2]);
    const fov = typeof camera.fov === 'number' ? camera.fov : DEFAULT_FOV;
    const [lo, hi] = zoomRange();
    const dist = MathUtils.clamp(
      (extent.radius / Math.tan((fov * Math.PI) / 360)) * 1.1, lo, hi);
    target.copy(C);
    OFF.subVectors(camera.position, C);
    if (OFF.lengthSq() < 1e-10) OFF.set(0, 0, 1); else OFF.normalize();
    camera.position.copy(C).addScaledVector(OFF, dist);
    M.lookAt(camera.position, C, WORLD_UP);
    camera.quaternion.setFromRotationMatrix(M);
  }

  // ----------------------------------------------------------------- pointer
  // Click-vs-drag by DISPLACEMENT from the press point, latched — see DRAG_SLOP in types.ts.
  let dragging = false, isDrag = false, downX = 0, downY = 0, lastX = 0, lastY = 0, btn = -1;
  const pd = (ev: PointerEventLike): void => {
    if (ev.button !== 0 && ev.button !== 2) return;
    dragging = true; isDrag = false; btn = ev.button;
    downX = lastX = ev.clientX; downY = lastY = ev.clientY;
    if (ev.pointerId !== undefined) domSurface?.setPointerCapture?.(ev.pointerId);
  };
  const pm = (ev: PointerEventLike): void => {
    if (!dragging) return;
    const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
    lastX = ev.clientX; lastY = ev.clientY;
    if (!isDrag) {
      if (Math.hypot(ev.clientX - downX, ev.clientY - downY) < DRAG_SLOP) return;
      isDrag = true;
    }
    orbit(dx * cfg.mouseSens, dy * cfg.mouseSens);
    onInput();
  };
  const pu = (ev: PointerEventLike): boolean => {
    if (ev.pointerId !== undefined) domSurface?.releasePointerCapture?.(ev.pointerId);
    const wasClick = dragging && !isDrag && btn === 0;
    dragging = false; isDrag = false; btn = -1;
    return wasClick;
  };
  const wheel = (ev: WheelEventLike): void => {
    ev.preventDefault?.();
    dolly(radius() * (ev.deltaY > 0 ? cfg.wheelK : 1 / cfg.wheelK));
    onInput();
  };

  function update(dt: number): CameraTelemetry {
    void dt;   // no inertia, no drift — the simple camera moves only under input
    telem.radius = radius();
    return telem;
  }

  // ---------------------------------------------------------------- DOM wiring
  let domSurface: ControlSurface | null = null;
  const pdH = (ev: Event): void => pd(ev as unknown as PointerEventLike);
  const pmH = (ev: Event): void => pm(ev as unknown as PointerEventLike);
  const puH = (ev: Event): void => { pu(ev as unknown as PointerEventLike); };
  const wheelH = (ev: Event): void => wheel(ev as unknown as WheelEventLike);

  function detach(): void {
    if (!domSurface) return;
    domSurface.removeEventListener('pointerdown', pdH);
    domSurface.removeEventListener('pointermove', pmH);
    domSurface.removeEventListener('pointerup', puH);
    domSurface.removeEventListener('wheel', wheelH);
    domSurface = null;
  }
  function attach(dom: ControlSurface): void {
    detach();
    domSurface = dom;
    // All pointer events on the surface itself: pointer capture keeps a drag flowing to it
    // even after the pointer leaves, so no window-level listener is needed here.
    dom.addEventListener('pointerdown', pdH);
    dom.addEventListener('pointermove', pmH);
    dom.addEventListener('pointerup', puH);
    dom.addEventListener('wheel', wheelH, { passive: false });
  }

  return {
    attach, update, home, dispose: detach,
    target,
    get radius(): number { return radius(); },
    orbit, dolly, zoomOf, setZoom,
    setTarget(v: Vector3, keepRadius = true): void {
      const r = keepRadius ? radius() : 0;
      target.copy(v);
      if (keepRadius && r > 1e-5) dolly(r);
    },
    consumeClick: pu,
  };
}
