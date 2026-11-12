/* A ship camera that also orbits — the two-verb model, ported from the predecessor's
 * flycam.js (vault-site-3d) with its logic intact. The distinction is the whole design, and
 * it is worth restating because the two feel completely different even though both are
 * "rotation":
 *
 *   TURNING (arrow keys, Q/E) swings the ship's NOSE. The target moves around the camera at a
 *     fixed radius; the camera stays put and re-aims. This reads as *you* turning.
 *
 *   ORBITING (mouse drag) swings the CAMERA around the target. The rig — position and
 *     orientation together — rotates rigidly about the target, so the aim is preserved and so
 *     is roll. This reads as *the universe* turning. Measured in the predecessor and pinned in
 *     test/flycam.test.ts: the target moves 0.000 and aim·toTarget stays 1.0000.
 *
 *   MOVING (WASD, X/C) adds the same delta to BOTH camera and target, which slides the whole
 *     rig through the scene and leaves the orbit radius untouched. So flying somewhere does
 *     not quietly change what dragging will pivot around.
 *
 *   DOLLYING (wheel, zoom slider) changes only the radius. The target does not move, which is
 *     what makes a zoom slider and the wheel agree by construction rather than by bookkeeping.
 *
 * Speed scales with how much room is ahead (the injected `roomAt`) rather than with a notional
 * orbit radius, so you slow down automatically entering a dense region and open out crossing
 * the void between them — and by the time a page's card is open you are already crawling,
 * which is the approach damping the zoom ladder needs, for free. Proximity DAMPS toward a
 * floor, it never GATES to zero (also pinned in the tests).
 *
 * Determinism (DESIGN.md §10.1): no clock anywhere — tweens advance on the `dt` handed to
 * `update()`, so a scripted flight replays identically and tests need no fake timers.
 */

import { MathUtils, Matrix4, Quaternion, Vector3 } from 'three';
import { DEFAULT_FOV } from '../field/constants';
import { DRAG_SLOP, HOME_MARGIN } from './types';
import type {
  CameraController, CameraTelemetry, ControlSurface, Extent,
  KeyEventLike, PointerEventLike, RigCamera, WheelEventLike,
} from './types';

/** Fine-motion gain while left Alt is held. */
const FINE = 0.28;
/** Wheel dolly ratio per notch; left Shift raises it to the 2.4th power. */
const WHEEL_K = 1.14;

/* ---- planet-surface collision + altitude speed-scale (DESIGN: near-mesh terrain, stage A) ------
 * Consulted ONLY through the opt-in `surfaceAt` callback, and only when it returns a promoted
 * planet near the camera. Absent or returning null ⇒ none of this runs and the translation step is
 * byte-identical to the node-only path — the whole feature is off with the near-mesh LOD off. */
/** Hover clearance above a planet sphere the camera may never cross: max(radius × FRAC, MIN). The
 *  absolute floor keeps a tiny planet clear of the adaptive near plane. */
const HOVER_FRAC = 0.04;
const HOVER_MIN = 0.02;
/** Altitude-scaled speed: translation scales by altitude/(radius × SPEED_ALT_SCALE), clamped to
 *  [SPEED_FLOOR, 1] — you crawl at the surface (a controllable skim) but never fully freeze, and
 *  far from any planet the scale is 1 (unchanged). */
const SPEED_ALT_SCALE = 2.0;
const SPEED_FLOOR = 0.12;
/** Soft-landing band: within SURF_BRAKE_FRAC × radius of the floor the DESCENDING component of the
 *  step eases to zero (the same soft-stop shape the node field uses), so you settle onto the
 *  surface instead of hitting a wall. */
const SURF_BRAKE_FRAC = 0.6;

const EDIT = 'input,textarea,select,[contenteditable=""],[contenteditable="true"]';

type MoveAxis = 'f' | 'r' | 'u';
type TurnAxis = 'yaw' | 'pitch' | 'roll';

/* W/S drive forward and back; X/C lift and drop.
 * Flying a graph is not flying a plane: you spend most of the time rising above a cluster to
 * see it and dropping into it to read, and far less time holding a straight line. W/S run the
 * throttle on the home row while X/C sit right below them, so the left hand can hold a climb
 * and a cruise at once without leaving wasd. PageUp/PageDown stay bound to the same vertical as
 * X/C — they cost nothing and they are what a returning user will reach for. */
const MOVE: Readonly<Record<string, readonly [MoveAxis, 1 | -1]>> = {
  KeyW: ['f', 1], KeyS: ['f', -1],
  KeyA: ['r', -1], KeyD: ['r', 1],
  KeyX: ['u', 1], KeyC: ['u', -1],
  PageUp: ['u', 1], PageDown: ['u', -1],
};
/* CapsLock LOCKS the speed; left Alt is the fine-motion gain; left Shift is the coarse-wheel
 * modifier (its only remaining job — see the wheel handler). CapsLock is a toggle, not a key
 * you hold: on macOS it fires keydown when the lock turns ON and keyup when it turns OFF, so the
 * add-on-keydown / delete-on-keyup tracking below makes `hold` mean simply "caps is on" — tap
 * once to lock the speed, tap again to release it, no finger pinned down.
 * It HOLDS the speed, it does not multiply it: cruising speed is proportional to room, with a
 * proximity damp on top, so "faster" would scale the cruise AND the crawl — you would still
 * decelerate, just from a higher number. Locking latches the speed you had when you toggled it
 * on and bypasses both, so approaching a cluster no longer brakes for you.
 * Only the roles differ per entry; what the rig reads is membership in `mods`, which is why
 * ShiftLeft must stay listed here for the wheel to keep seeing it. */
const MODS: Readonly<Record<string, 'hold' | 'fine' | 'coarse'>> = {
  CapsLock: 'hold', AltLeft: 'fine', ShiftLeft: 'coarse',
};
const TURN: Readonly<Record<string, readonly [TurnAxis, 1 | -1]>> = {
  ArrowLeft: ['yaw', 1], ArrowRight: ['yaw', -1],
  ArrowUp: ['pitch', 1], ArrowDown: ['pitch', -1],
  KeyQ: ['roll', -1], KeyE: ['roll', 1],
};
const BY_KEY: Readonly<Record<string, string>> = {
  arrowup: 'ArrowUp', arrowdown: 'ArrowDown', arrowleft: 'ArrowLeft', arrowright: 'ArrowRight',
  pageup: 'PageUp', pagedown: 'PageDown', w: 'KeyW', a: 'KeyA', s: 'KeyS', d: 'KeyD',
  q: 'KeyQ', e: 'KeyE', x: 'KeyX', c: 'KeyC',
};

const closestEdit = (t: unknown): boolean => {
  const el = t as { closest?: (sel: string) => unknown } | null | undefined;
  return !!(el && typeof el.closest === 'function' && el.closest(EDIT));
};
const inEditor = (ev: { target?: unknown }): boolean => {
  if (closestEdit(ev.target)) return true;
  if (typeof document === 'undefined') return false;
  return closestEdit(document.activeElement);
};
/* `ev.key` for left Alt is just "Alt" with no side, so the BY_KEY fallback cannot tell AltLeft
 * from AltRight — the speed modifiers are matched on `ev.code` above and nowhere else. */
const codeOf = (ev: KeyEventLike): string | null => {
  if (MOVE[ev.code] || TURN[ev.code]) return ev.code;
  return BY_KEY[(ev.key ?? '').toLowerCase()] ?? null;
};
const ease = (t: number): number =>
  (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** The flycam's tunables. Defaults are the predecessor's schema defaults, argued over on a
 *  real corpus; read live every frame so a panel can mutate the object. */
export interface FlycamConfig {
  /** Cruise speed = room ahead × this (per second). */
  speedK: number;
  /** Drag sensitivity, rad per CSS px. */
  mouseSens: number;
  /** Ship turn rate, rad/s (roll runs at 1.26×). */
  turnRate: number;
  /** Idle world-up drift, rad/s. 0.047 ≈ 133 s per orbit — "alive", not "moving". */
  spinRate: number;
  /** Throttle ease time constant, seconds. */
  easeLin: number;
}

export const FLYCAM_DEFAULTS: Readonly<FlycamConfig> = {
  speedK: 2.9, mouseSens: 0.0052, turnRate: 1.15, spinRate: 0.047, easeLin: 0.13,
};

export interface FlycamOptions {
  /** Free room (distance to the nearest obstacle, world units) at a position. Cruise speed is
   *  proportional to it; `Infinity` is fine (it clamps to the zoom range's far end). */
  roomAt: (pos: readonly [number, number, number]) => number;
  /** The radius at which approach should bottom out — the damp floor's scale. */
  softStop: () => number;
  /** [closest, farthest] dolly radius, world units. */
  zoomRange: () => readonly [number, number];
  /** OPT-IN planet-surface floor (near-mesh LOD, terrain stage A). The nearest promoted planet's
   *  LIVE centre + world radius to `pos`, or null when none is near (LOD off / nothing promoted).
   *  When present the flycam treats that sphere as solid ground: translation decelerates smoothly
   *  into it, cannot descend below `radius + hover`, and scales down with altitude so you can skim
   *  and orbit it controllably. Omitted, or returning null, ⇒ the flycam behaves exactly as without
   *  it (no collision, no speed change). The centre is read every frame, so the floor holds as the
   *  body spins / morphs. */
  surfaceAt?: (
    pos: readonly [number, number, number],
  ) => {
    center: readonly [number, number, number];
    radius: number;
    /** OPT-IN terrain floor (near-mesh terrain stage B): the effective surface radius along a UNIT
     *  direction from the centre (sphere radius + displaced terrain height). When present the camera
     *  lands on the relief — skimming ridges, settling in valleys — instead of the bounding sphere.
     *  Absent ⇒ the stage-A sphere floor, byte-identical. Sampled every frame at the camera's radial
     *  direction, so you rise and fall with the terrain as you fly over it. */
    terrain?: (dx: number, dy: number, dz: number) => number;
  } | null;
  /** Fired on every user input (key, drag, wheel) — hosts use it to wake auto-behaviours. */
  onInput?: () => void;
  /** Where key events live. Defaults to `globalThis`; tests hand in a stub surface. */
  keyboard?: ControlSurface;
  /** Collapse easing and suppress drift. Defaults to the prefers-reduced-motion media query
   *  where one exists, false headless. */
  reducedMotion?: boolean;
}

interface Tween {
  elapsed: number; ms: number;
  p0: Vector3; p1: Vector3;
  q0: Quaternion; q1: Quaternion;
  g0: Vector3; g1: Vector3;
}

/** Everything the flycam offers beyond the plain controller contract. */
export interface Flycam extends CameraController {
  /** The slow turntable about WORLD up (not the ship's — or the drift would change direction
   *  every time you pitch). */
  orbitWorld(ax: number): void;
  /** Fly to `dist` away from `to`, aimed at it, over `ms` of update-time. */
  goTo(to: Vector3, dist: number, ms?: number): void;
  /** Re-level the horizon in place (420 ms tween, position untouched). */
  level(): void;
  /** Scale the rig about a pivot — used when a spread change rescales the field, so your
   *  position relative to what you were looking at survives. */
  rescale(pivot: Vector3, k: number): void;
  /** Idle drift on/off. */
  spin: boolean;
  /** Ease the drift off as you close on a page — drift is pleasant at a distance and
   *  nauseating up close. */
  spinScale: number;
  /** True while the user is actively flying or dragging — hosts suppress their own geometry
   *  rotations on the same condition, so everything stops together. */
  readonly busy: boolean;
  /** True while a goTo/level tween is running. */
  readonly tweening: boolean;
  /** Cancel any running tween. */
  cancel(): void;
}

export function createFlycam(camera: RigCamera, cfg: FlycamConfig, opts: FlycamOptions): Flycam {
  const { roomAt, softStop, zoomRange } = opts;
  const surfaceAt = opts.surfaceAt;
  const onInput = opts.onInput ?? ((): void => {});
  const reduced = opts.reducedMotion ??
    (typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
  const tauLin = (): number => (reduced ? 0.001 : cfg.easeLin);
  const tauAng = (): number => (reduced ? 0.001 : cfg.easeLin * 0.77);

  const held = new Set<string>();
  const mods = new Set<string>();
  const vel: Record<MoveAxis | TurnAxis, number> =
    { f: 0, r: 0, u: 0, yaw: 0, pitch: 0, roll: 0 };
  const want: Record<MoveAxis | TurnAxis, number> =
    { f: 0, r: 0, u: 0, yaw: 0, pitch: 0, roll: 0 };
  let hold = false, fine = false, blocked = false;
  let tween: Tween | null = null;
  let heldSpeed = 0;        // the speed latched when the CapsLock speed-lock turned on
  let spin = false, spinScale = 1, busyNow = false;
  const WORLD_UP = new Vector3(0, 1, 0);

  const target = new Vector3();   // what dragging pivots around, and zoom measures to
  const F = new Vector3(), R = new Vector3(), U = new Vector3();
  const Q = new Quaternion(), TMP = new Vector3(), OFF = new Vector3();
  const AFTER = new Vector3(), M = new Matrix4();
  const surfC = new Vector3(), surfN = new Vector3();   // planet centre + radial unit (surfaceAt)
  const posT: [number, number, number] = [0, 0, 0];
  const telem: CameraTelemetry =
    { speed: 0, room: 0, flying: false, radius: 0, blocked: false, altitude: Infinity, held: 0 };

  const axes = (): void => {
    F.set(0, 0, -1).applyQuaternion(camera.quaternion);
    R.set(1, 0, 0).applyQuaternion(camera.quaternion);
    U.set(0, 1, 0).applyQuaternion(camera.quaternion);
  };
  const radius = (): number => camera.position.distanceTo(target);
  const roomAtV = (v: Vector3): number => {
    posT[0] = v.x; posT[1] = v.y; posT[2] = v.z;
    return roomAt(posT);
  };

  // ---------------------------------------------------------------- keyboard
  const onKey = (ev: KeyEventLike, down: boolean): void => {
    // Modifier state first, and before any early exit: a release must always be honoured or
    // the ship stays stuck at speed after you tab away.
    const mod = MODS[ev.code];
    if (mod) {
      if (down) mods.add(ev.code); else mods.delete(ev.code);
      hold = mods.has('CapsLock');
      if (!hold) heldSpeed = 0;        // re-latch next time, from wherever you then are
      fine = mods.has('AltLeft');
    }
    const c = codeOf(ev);
    if (!c) return;
    // A key-UP is always honoured, even while typing. Gating both directions on focus means
    // pressing W over the canvas and then clicking into the search box swallows the release,
    // and the ship flies on forever with nothing held.
    if (!down) { held.delete(c); return; }
    if (inEditor(ev)) return;
    held.add(c);
    tween = null; onInput();
    ev.preventDefault?.();
  };

  const blur = (): void => {
    held.clear(); mods.clear(); hold = false; fine = false; heldSpeed = 0;
  };

  // ------------------------------------------------- drag = orbit the universe
  /* Click or drag, and the distinction has to be DISPLACEMENT from the press point — not the
   * path the pointer walked while pressed (see DRAG_SLOP in types.ts for the history).
   * `isDrag` latches, because the mirror-image mistake is just as wrong: a long drag that
   * happens to end near where it began has a displacement of nearly zero, and must not become
   * a click on release. */
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
    tween = null;
    orbit(dx * cfg.mouseSens, dy * cfg.mouseSens);
    onInput();
  };
  const pu = (ev: PointerEventLike): boolean => {
    if (ev.pointerId !== undefined) domSurface?.releasePointerCapture?.(ev.pointerId);
    const wasClick = dragging && !isDrag && btn === 0;
    dragging = false; isDrag = false; btn = -1;
    return wasClick;
  };

  /** The slow turntable. Rotates about WORLD up rather than the ship's own up, which is what
   *  makes it a turntable and not a barrel roll: the axis has to stay put while you look
   *  around, or the drift changes direction every time you pitch. */
  function orbitWorld(ax: number): void {
    Q.setFromAxisAngle(WORLD_UP, ax);
    OFF.subVectors(camera.position, target).applyQuaternion(Q);
    camera.quaternion.premultiply(Q).normalize();
    camera.position.copy(target).add(OFF);
  }

  /** Rotate the rig rigidly about the target: position AND orientation by the same quaternion.
   *  Doing both is what keeps the camera aimed and the roll intact without any extra
   *  bookkeeping — a rigid rotation cannot change where the nose points relative to the
   *  target. */
  function orbit(ax: number, ay: number): void {
    axes();
    OFF.subVectors(camera.position, target);
    Q.setFromAxisAngle(U, ax);
    OFF.applyQuaternion(Q); camera.quaternion.premultiply(Q);
    axes();
    Q.setFromAxisAngle(R, ay);
    OFF.applyQuaternion(Q); camera.quaternion.premultiply(Q);
    camera.quaternion.normalize();
    camera.position.copy(target).add(OFF);
  }

  // ---------------------------------------------------------------- dolly
  function dolly(r: number): void {
    const [lo, hi] = zoomRange();
    const want2 = MathUtils.clamp(r, lo, hi);
    OFF.subVectors(camera.position, target);
    const cur = OFF.length();
    if (cur < 1e-5) OFF.set(0, 0, 1);
    OFF.normalize().multiplyScalar(want2);
    camera.position.copy(target).add(OFF);
  }
  /** Zoom as a 0..1 slider position on a LOG curve: even travel means even ratio, which is the
   *  only mapping that does not crawl at the wide end and leap at the close end. */
  const zoomOf = (): number => {
    const [lo, hi] = zoomRange();
    return MathUtils.clamp(Math.log(radius() / hi) / Math.log(lo / hi), 0, 1);
  };
  const setZoom = (t: number): void => {
    const [lo, hi] = zoomRange();
    dolly(hi * Math.pow(lo / hi, MathUtils.clamp(t, 0, 1)));
  };

  const wheel = (ev: WheelEventLike): void => {
    if (inEditor(ev)) return;
    ev.preventDefault?.();
    tween = null;
    const k = ev.deltaY > 0 ? WHEEL_K : 1 / WHEEL_K;
    // left Shift held while wheeling raises the dolly ratio — a coarse zoom, distinct from the
    // CapsLock speed-lock; this one reads ShiftLeft directly off `ev.code` and nothing else
    dolly(radius() * Math.pow(k, mods.has('ShiftLeft') ? 2.4 : 1));
    onInput();
  };

  // ---------------------------------------------------------------- goto / home
  function goTo(to: Vector3, dist: number, ms = 900): void {
    const t1 = new Vector3().copy(to);
    const dir = new Vector3().subVectors(camera.position, t1);
    if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
    dir.normalize();
    const p1 = new Vector3().copy(t1).addScaledVector(dir, dist);
    axes();
    const m = new Matrix4().lookAt(p1, t1, U);
    tween = {
      elapsed: 0, ms,
      p0: camera.position.clone(), p1,
      q0: camera.quaternion.clone(), q1: new Quaternion().setFromRotationMatrix(m),
      g0: target.clone(), g1: t1,
    };
  }

  function home(extent: Extent): void {
    tween = null;
    const c = AFTER.set(extent.center[0], extent.center[1], extent.center[2]);
    const fov = typeof camera.fov === 'number' ? camera.fov : DEFAULT_FOV;
    const [lo, hi] = zoomRange();
    // Frame the bounding sphere vertically, with a small margin so the rim is not on the edge.
    const dist = MathUtils.clamp(
      (extent.radius / Math.tan((fov * Math.PI) / 360)) * HOME_MARGIN, lo, hi);
    target.copy(c);
    OFF.subVectors(camera.position, c);
    if (OFF.lengthSq() < 1e-10) OFF.set(0, 0, 1); else OFF.normalize();
    camera.position.copy(c).addScaledVector(OFF, dist);
    M.lookAt(camera.position, c, WORLD_UP);
    camera.quaternion.setFromRotationMatrix(M);
  }

  // ---------------------------------------------------------------- per-frame
  function update(dt: number): CameraTelemetry {
    if (tween) {
      // dt-driven, never the wall clock: a tween is geometry in motion and must replay
      // byte-identically from the same inputs (DESIGN.md §10.1).
      tween.elapsed += dt * 1000;
      const k = Math.min(1, tween.elapsed / tween.ms);
      const e = ease(k);
      camera.position.lerpVectors(tween.p0, tween.p1, e);
      camera.quaternion.slerpQuaternions(tween.q0, tween.q1, e);
      target.lerpVectors(tween.g0, tween.g1, e);
      if (k >= 1) tween = null;
      telem.speed = 0; telem.room = roomAtV(camera.position); telem.flying = true;
      telem.radius = radius(); telem.blocked = false; telem.altitude = Infinity;
      telem.held = held.size;
      return telem;
    }

    want.f = 0; want.r = 0; want.u = 0; want.yaw = 0; want.pitch = 0; want.roll = 0;
    for (const c of held) {
      const mv = MOVE[c]; if (mv) want[mv[0]] += mv[1];
      const tn = TURN[c]; if (tn) want[tn[0]] += tn[1];
    }
    const gain = fine ? FINE : 1;
    const room = roomAtV(camera.position);
    const stop = softStop();
    const [, hi] = zoomRange();
    const cruise = MathUtils.clamp(
      (Number.isFinite(room) ? room : hi) * cfg.speedK, stop * 1.2, hi) * gain;
    /* Latch on the first frame of the hold rather than when the key arrives, so the speed is
     * one the rig was actually flying at — and then keep it EXACTLY, not as a floor. A floor
     * re-reads the cruise every frame, so flying out of a cluster into open space accelerates
     * you without bound, which is the opposite of holding a speed. */
    if (hold && heldSpeed <= 0) heldSpeed = cruise;
    const speed = hold ? heldSpeed : cruise;

    const aLin = 1 - Math.exp(-dt / tauLin());
    const aAng = 1 - Math.exp(-dt / tauAng());
    vel.f += (want.f - vel.f) * aLin;
    vel.r += (want.r - vel.r) * aLin;
    vel.u += (want.u - vel.u) * aLin;
    vel.yaw += (want.yaw - vel.yaw) * aAng;
    vel.pitch += (want.pitch - vel.pitch) * aAng;
    vel.roll += (want.roll - vel.roll) * aAng;

    // TURNING: the nose swings, the target follows it round at the current radius.
    const rad = radius();
    let turned = false;
    axes();
    if (Math.abs(vel.yaw) > 1e-4) {
      Q.setFromAxisAngle(U, vel.yaw * cfg.turnRate * gain * dt);
      camera.quaternion.premultiply(Q); axes(); turned = true;
    }
    if (Math.abs(vel.pitch) > 1e-4) {
      Q.setFromAxisAngle(R, vel.pitch * cfg.turnRate * gain * dt);
      camera.quaternion.premultiply(Q); axes(); turned = true;
    }
    if (Math.abs(vel.roll) > 1e-4) {
      Q.setFromAxisAngle(F, -vel.roll * cfg.turnRate * 1.26 * gain * dt);
      camera.quaternion.premultiply(Q); axes();
    }
    if (turned) {
      camera.quaternion.normalize();
      target.copy(camera.position).addScaledVector(F, rad);
    }

    // The drift. Suppressed while you are actually doing something — it is ambience, and
    // ambience that fights your input is just a bug with a nice name. It resumes the moment
    // you stop, with no re-arm delay.
    const busy = dragging || held.size > 0;
    if (spin && !busy && !reduced) orbitWorld(cfg.spinRate * spinScale * dt);
    busyNow = busy;

    // MOVING: camera and target together, so the orbit radius survives the flight.
    const moving = Math.abs(vel.f) + Math.abs(vel.r) + Math.abs(vel.u);
    let altitude = Infinity;   // height above the nearest planet surface (telemetry)
    let effSpeed = speed;      // the speed actually applied (altitude-scaled near a planet)
    if (moving > 1e-4) {
      TMP.set(0, 0, 0)
        .addScaledVector(F, vel.f).addScaledVector(R, vel.r).addScaledVector(U, vel.u)
        .multiplyScalar(speed * dt);
      blocked = false;

      /* PLANET SURFACE (opt-in, stage A) — applied BEFORE the node soft-stop, so a promoted planet
       * is an additional, stronger floor layered on top of the field's node braking (which is never
       * removed). `surfaceAt` null (LOD off, or no planet near) ⇒ this whole block is skipped and
       * the step is byte-identical to the node-only path below. The planet's LIVE centre is read
       * each frame, so the floor tracks it as the body spins / morphs. */
      posT[0] = camera.position.x; posT[1] = camera.position.y; posT[2] = camera.position.z;
      const surf = surfaceAt ? surfaceAt(posT) : null;
      let floor = 0;
      if (surf) {
        surfC.set(surf.center[0], surf.center[1], surf.center[2]);
        const rSurf = surf.radius;
        surfN.subVectors(camera.position, surfC);
        const dist = surfN.length();
        if (dist > 1e-6) surfN.multiplyScalar(1 / dist); else surfN.set(0, 1, 0); // radial-out unit
        // TERRAIN floor (stage B): the displaced surface radius under the camera, else the sphere
        // (stage A). Sampled at the live radial direction, so the floor rises and falls with the
        // relief as you skim over it. The speed-scale / brake bands keep using the sphere radius as
        // their SCALE (the planet's size), so only the floor itself follows the terrain.
        const rFloor = surf.terrain ? surf.terrain(surfN.x, surfN.y, surfN.z) : rSurf;
        const hover = Math.max(rSurf * HOVER_FRAC, HOVER_MIN);
        floor = rFloor + hover;
        altitude = dist - rFloor; // height above the true (terrain) surface
        // Holding (the CapsLock speed-lock) bypasses the smooth brake + speed-scale, exactly as it
        // bypasses the node damp; the hard backstop (c) below still runs, so you cannot tunnel.
        if (!hold) {
          // (a) altitude-scaled speed — crawl near the ground, unchanged (scale 1) far away.
          const altScale = MathUtils.clamp(
            altitude / Math.max(rSurf * SPEED_ALT_SCALE, 1e-6), SPEED_FLOOR, 1);
          TMP.multiplyScalar(altScale);
          effSpeed = speed * altScale;
          // (b) soft-landing — damp only the DESCENDING (toward-centre) part of the step as the
          //     floor nears; tangential + climbing motion is untouched, so you keep full control to
          //     skim and orbit just above the surface.
          const vrad = TMP.dot(surfN); // >0 climbing away, <0 descending into the planet
          if (vrad < 0) {
            const band = Math.max(rSurf * SURF_BRAKE_FRAC, hover);
            const kk = MathUtils.clamp((dist - floor) / band, 0, 1); // 1 high, 0 at the floor
            TMP.addScaledVector(surfN, (kk - 1) * vrad); // scale the radial-in component by kk
            if (kk <= 0.2) blocked = true;
          }
        }
      }

      /* NODE soft-stop (the field behaviour, UNCHANGED). Proximity DAMPS, it never GATES: a gate
       * that evaluates to exactly zero strands the ship, because `room` is the distance to the
       * nearest page ANYWHERE, not the nearest page ahead — so inside a dense region you are
       * routinely within the stop radius of something off to the side and the step would multiply
       * by nothing. The clamp floor (0.16) is the whole fix. */
      AFTER.copy(TMP).add(camera.position);
      const ahead = roomAtV(AFTER);
      if (!hold && Number.isFinite(ahead) && ahead < room && ahead < stop * 2.5) {
        const k = MathUtils.clamp(ahead / (stop * 2.5), 0.16, 1);
        TMP.multiplyScalar(k);
        if (k <= 0.2) blocked = true;
      }
      camera.position.add(TMP);
      target.add(TMP);

      // (c) hard floor backstop: never let the camera cross below radius+hover. Catches overshoot
      //     (a large dt) and the hold case where the smooth damp is bypassed. The target moves by
      //     the same delta, so the orbit radius and the aim survive the correction.
      if (surf) {
        surfN.subVectors(camera.position, surfC);
        const dNew = surfN.length();
        if (dNew > 1e-6 && dNew < floor) {
          surfN.multiplyScalar((floor - dNew) / dNew); // outward delta back onto the floor
          camera.position.add(surfN);
          target.add(surfN);
          blocked = true;
        }
      }
    }
    telem.speed = moving > 1e-4 ? effSpeed : 0; telem.room = room;
    telem.flying = moving > 1e-4; telem.radius = radius();
    telem.blocked = blocked; telem.altitude = altitude; telem.held = held.size;
    return telem;
  }

  // ---------------------------------------------------------------- DOM wiring
  let domSurface: ControlSurface | null = null;
  let keySurface: ControlSurface | null = null;
  const kdH = (ev: Event): void => onKey(ev as unknown as KeyEventLike, true);
  const kuH = (ev: Event): void => onKey(ev as unknown as KeyEventLike, false);
  const blurH = (): void => blur();
  const pdH = (ev: Event): void => pd(ev as unknown as PointerEventLike);
  const pmH = (ev: Event): void => pm(ev as unknown as PointerEventLike);
  const wheelH = (ev: Event): void => wheel(ev as unknown as WheelEventLike);

  function detach(): void {
    if (keySurface) {
      keySurface.removeEventListener('keydown', kdH);
      keySurface.removeEventListener('keyup', kuH);
      keySurface.removeEventListener('blur', blurH);
      keySurface.removeEventListener('pointermove', pmH);
      keySurface = null;
    }
    if (domSurface) {
      domSurface.removeEventListener('pointerdown', pdH);
      domSurface.removeEventListener('wheel', wheelH);
      domSurface = null;
    }
  }
  function attach(dom: ControlSurface): void {
    detach();
    domSurface = dom;
    // Keys live on the window (or an injected stand-in): focus sits anywhere, the ship must
    // still hear the release. pointermove too, so a drag survives leaving the canvas.
    keySurface = opts.keyboard ?? (globalThis as unknown as ControlSurface);
    keySurface.addEventListener('keydown', kdH);
    keySurface.addEventListener('keyup', kuH);
    keySurface.addEventListener('blur', blurH);
    keySurface.addEventListener('pointermove', pmH);
    dom.addEventListener('pointerdown', pdH);
    dom.addEventListener('wheel', wheelH, { passive: false });
  }

  return {
    attach, update, home, dispose: detach,
    target,
    get radius(): number { return radius(); },
    orbit, orbitWorld, dolly, zoomOf, setZoom,
    goTo,
    setTarget(v: Vector3, keepRadius = true): void {
      const r = keepRadius ? radius() : 0;
      target.copy(v);
      if (keepRadius && r > 1e-5) dolly(r);
    },
    consumeClick: pu,
    level(): void {
      axes();
      const m = new Matrix4().lookAt(
        camera.position, TMP.copy(camera.position).add(F), new Vector3(0, 1, 0));
      tween = {
        elapsed: 0, ms: 420,
        p0: camera.position.clone(), p1: camera.position.clone(),
        q0: camera.quaternion.clone(), q1: new Quaternion().setFromRotationMatrix(m),
        g0: target.clone(), g1: target.clone(),
      };
    },
    rescale(pivot: Vector3, k: number): void {
      camera.position.sub(pivot).multiplyScalar(k).add(pivot);
      target.sub(pivot).multiplyScalar(k).add(pivot);
      tween = null;
    },
    get spin(): boolean { return spin; },
    set spin(v: boolean) { spin = !!v; },
    get spinScale(): number { return spinScale; },
    set spinScale(v: number) { spinScale = v; },
    get busy(): boolean { return busyNow; },
    get tweening(): boolean { return !!tween; },
    cancel(): void { tween = null; },
  };
}
