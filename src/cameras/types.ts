/* Camera controller contracts (DESIGN.md §6). Controllers are pluggable: the library talks to
 * every camera through `CameraController`, and each concrete camera (flycam, orbit) adds its
 * own verbs on top. Nothing in here touches a renderer — a controller owns only a rig
 * (position + quaternion + orbit target) and the DOM surface it is attached to, which is what
 * lets the tests drive one headlessly and assert on raw numbers.
 */

import type { Quaternion, Vector3 } from 'three';

/** The rig a controller drives — structurally the subset of a three.js camera it touches, so
 *  tests (and non-three consumers) can hand in anything with a position and a quaternion.
 *  `fov` (vertical, degrees) is read by `home()` when present; a PerspectiveCamera satisfies
 *  this as-is. */
export interface RigCamera {
  position: Vector3;
  quaternion: Quaternion;
  fov?: number;
}

/** The live extent to frame — always measured from current positions, never a stored one
 *  (DESIGN.md §10.8). */
export interface Extent {
  center: readonly [number, number, number];
  radius: number;
}

/** What `update()` reports every frame. Telemetry must show hidden states (DESIGN.md §10.7):
 *  the proximity damp (`blocked`), the room measurement the speed came from, the live radius.
 *  The SAME object is returned every call (mutated in place) — read it, do not keep it. */
export interface CameraTelemetry {
  /** World units/s actually flown this frame (0 when idle or tweening). */
  speed: number;
  /** `roomAt(camera.position)` — how much free space the speed scaling saw. */
  room: number;
  /** True while flying under key input or inside a goTo/level tween. */
  flying: boolean;
  /** Camera-to-target distance. */
  radius: number;
  /** True when the proximity damp has the step pinned near its floor. */
  blocked: boolean;
  /** Number of movement/turn keys currently held. */
  held: number;
}

/** The DOM surface a controller wires itself to — the subset of an EventTarget/HTMLElement it
 *  actually uses. Method syntax keeps parameters bivariant, so both a real canvas and a
 *  ten-line test stub satisfy it. */
export interface ControlSurface {
  addEventListener(
    type: string,
    listener: (ev: Event) => void,
    options?: boolean | { passive?: boolean },
  ): void;
  removeEventListener(type: string, listener: (ev: Event) => void): void;
  setPointerCapture?(pointerId: number): void;
  releasePointerCapture?(pointerId: number): void;
}

/** Structural event shapes — what the handlers actually read, so synthetic events in tests are
 *  plain object literals, no DOM required. */
export interface KeyEventLike {
  code: string;
  key?: string;
  target?: unknown;
  preventDefault?(): void;
}
export interface PointerEventLike {
  button: number;
  clientX: number;
  clientY: number;
  pointerId?: number;
}
export interface WheelEventLike {
  deltaY: number;
  target?: unknown;
  preventDefault?(): void;
}

/** Pixels of DISPLACEMENT from the press point (CSS px, like every pixel constant in this
 *  library) under which a pointer gesture is still a click. Displacement, not path walked: a
 *  real press emits a handful of 1–2 px jitter moves, and summing them discarded genuine
 *  clicks as drags in the predecessor. 5 is forgiving enough for a trackpad tap and tight
 *  enough that a deliberate drag never reads as a click. Shared by every camera, exported
 *  once. */
export const DRAG_SLOP = 5;

/** What every camera controller provides. Concrete controllers extend this with their own
 *  verbs (the flycam's goTo/level/rescale, …); consumers that only need "a camera" code
 *  against this. */
export interface CameraController {
  /** Wire the controller to a DOM surface. Idempotent: re-attaching detaches first. A
   *  controller is fully drivable without ever attaching (tests, scripted flights). */
  attach(dom: ControlSurface): void;
  /** Advance the rig by `dt` seconds. Returns the (reused) telemetry object. */
  update(dt: number): CameraTelemetry;
  /** Frame the LIVE extent: target to its centre, camera retreated far enough to see all of
   *  it, levelled against world up. Cancels any tween. */
  home(extent: Extent): void;
  /** Remove every listener `attach` added. */
  dispose(): void;

  /** The orbit target: what dragging pivots around and zoom measures to. Live reference. */
  readonly target: Vector3;
  /** Camera-to-target distance, world units. */
  readonly radius: number;
  /** Rotate the rig rigidly about the target (position AND orientation by one quaternion):
   *  `ax` about the camera's up, `ay` about its right, radians. */
  orbit(ax: number, ay: number): void;
  /** Set the camera-to-target distance (clamped to `zoomRange()`); the target stays put. */
  dolly(r: number): void;
  /** Radius as a 0..1 slider position on a log curve (0 = closest). */
  zoomOf(): number;
  /** Inverse of `zoomOf`: even slider travel is even radius RATIO. */
  setZoom(t: number): void;
  /** Move the orbit target; `keepRadius` re-dollies so the camera keeps its distance. */
  setTarget(v: Vector3, keepRadius?: boolean): void;
  /** Resolve the gesture on pointer-up: true when it stayed a click (never exceeded
   *  DRAG_SLOP displacement with the primary button). The host calls this from its own
   *  pointerup and uses the answer to decide select-vs-ignore. */
  consumeClick(ev: PointerEventLike): boolean;
}
