/* The stage-4 demo: the two-draw-call field under real cameras, with picking, hover and
 * selection — the full interactive loop, consuming the library exactly as an external user
 * would (DESIGN.md §8). `?n=100000` overrides the corpus size; `?cam=orbit` swaps the default
 * flycam for the simple orbit camera; `?blend=additive` opts into the glow look.
 *
 * House rules honoured: no wall clock anywhere — rAF deltas drive the cameras and geometry
 * stays deterministic (§10.1); the boot path has a rejection handler that surfaces the error
 * into the UI (§10.5); `window.__starscape` is the debug handle and the overlay shows hidden
 * states — room, damping, budgets (§10.7); the camera homes to the LIVE extent (§10.8). FPS
 * is measured the SCALE.md way: median of 90 requestAnimationFrame intervals.
 *
 * Interaction model (DESIGN.md §6, ported from vault-site-3d src/main.js):
 *   hover    pointermove only marks the pick due; the ray is cast at most once per frame from
 *            the latest cursor position, and the field repaints only the hovered DEGREE.
 *   select   resolved on pointerup through the camera's consumeClick (displacement from the
 *            press point, so a drag never selects); independent of hover; dims the rest of
 *            the field. Escape clears. The pick coordinate comes from the pointerup EVENT,
 *            not from the last pointermove — the predecessor's stale-coordinate bug.
 */

import { Color, PerspectiveCamera, Scene, Vector3, WebGPURenderer } from 'three/webgpu';
import {
  buildGraph,
  buildGrid,
  createField,
  createFlycam,
  createOrbitCam,
  nearestDistance,
  pickRay,
  DEFAULT_FOV,
  D_REF_FRACTION,
  FALLOFF,
  FLYCAM_DEFAULTS,
  NODE_MAX_PX,
  NODE_MIN_PX,
  NODE_SCALE,
  ORBIT_DEFAULTS,
} from '../src/index';
import type { CameraController, Flycam } from '../src/index';
import { generateSynth } from '../src/synth/generate';

const statsEl = document.getElementById('stats')!;
const errorEl = document.getElementById('error')!;

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const n = Math.max(1, Number(params.get('n') ?? 10000) | 0);
  const seed = Number(params.get('seed') ?? 42) | 0;
  const camMode = params.get('cam') === 'orbit' ? 'orbit' : 'fly';

  statsEl.textContent = `generating synth corpus n=${n}…`;
  const t0 = performance.now();
  const input = generateSynth({ n, seed });
  const graph = buildGraph(input, { orderByFacet: 'system' });
  const tGraph = performance.now();

  /* ---- renderer. The ONE place devicePixelRatio is handled; everything the field reasons
     about stays in CSS px. WebGPURenderer falls back to WebGL2 automatically when WebGPU is
     unavailable. */
  const renderer = new WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  await renderer.init();
  document.body.appendChild(renderer.domElement);
  const canvas = renderer.domElement;

  const field = createField(graph, {
    heightCss: window.innerHeight,
    fov: DEFAULT_FOV,
    blend: params.get('blend') === 'additive' ? 'additive' : 'normal',
  });

  const scene = new Scene();
  scene.background = new Color(field.background);
  for (const o of field.objects) scene.add(o);

  /* ---- picking: ONE grid per field (the predecessor ran two for a while and clicked
     boot-time positions all session). The layout is frozen at this stage, so the grid never
     goes stale and drift stays 0 — rebuild-on-morph arrives with stage 6. */
  const positions = graph.positions!;
  const grid = buildGrid(positions, n);
  const { centre, radius, diagonal } = field.extent;
  const dRef = diagonal * D_REF_FRACTION; // the same falloff pivot the node shader uses

  /* ---- cameras. Flycam by default (?cam=orbit for the simple one). roomAt feeds the
     nearest-node distance from the pick grid, so cruise speed scales with free space and
     proximity damps toward the soft stop instead of gating to zero. */
  const camera = new PerspectiveCamera(
    DEFAULT_FOV,
    window.innerWidth / window.innerHeight,
    Math.max(0.05, radius * 0.001),
    (radius + diagonal) * 6,
  );

  const roomPos = { x: 0, y: 0, z: 0 }; // one reusable probe — no per-frame allocation
  const roomAt = (p: readonly [number, number, number]): number => {
    roomPos.x = p[0];
    roomPos.y = p[1];
    roomPos.z = p[2];
    return nearestDistance(grid, roomPos);
  };
  const softStop = (): number => radius * 0.02;
  const zoomRange = (): readonly [number, number] => [radius * 0.01, diagonal * 1.5];

  let fly: Flycam | null = null;
  let cam: CameraController;
  if (camMode === 'fly') {
    fly = createFlycam(camera, { ...FLYCAM_DEFAULTS }, { roomAt, softStop, zoomRange });
    fly.spin = true; // the idle turntable — ambience, suppressed while you fly or drag
    cam = fly;
  } else {
    cam = createOrbitCam(camera, { ...ORBIT_DEFAULTS }, { zoomRange });
  }
  cam.attach(canvas);
  cam.home({ center: centre, radius }); // the LIVE extent, never a stored one (§10.8)

  /* ---- hover: cast at most once per frame, from the latest cursor position. The repaint is
     O(degree) inside the field — never the whole colour buffer. */
  let W = window.innerWidth;
  let H = window.innerHeight;
  let px = 0;
  let py = 0;
  let pickDue = false;

  const RAY = new Vector3();
  function castAt(sx: number, sy: number): number {
    RAY.set((sx / W) * 2 - 1, -(sy / H) * 2 + 1, 0.5)
      .unproject(camera)
      .sub(camera.position)
      .normalize();
    return pickRay(
      grid,
      camera.position,
      RAY,
      {
        // The shader's own sizing, constant for constant, so the hit disc IS the drawn disc
        // (one CSS-pixel unit shared by shader, picker and ladder — DESIGN.md §5).
        tanHalf: Math.tan((DEFAULT_FOV * Math.PI) / 360),
        halfH: H / 2,
        sizeScale: NODE_SCALE,
        minPx: NODE_MIN_PX,
        maxPx: NODE_MAX_PX,
        falloff: FALLOFF,
        dRef,
        sizeOf: (i: number): number => field.sizeOf(i),
        // The LIVE boost: a hovered/selected glyph draws up to 1.75× larger (radius and
        // clamp rails both), and its hit disc must be that bigger disc, not the resting one.
        boostOf: (i: number): number => field.boostOf(i),
        drift: 0, // frozen layout; a stale-grid bound arrives with the stage-6 morph
      },
    );
  }

  function hover(i: number): void {
    if (field.setHover(i)) updateOverlay();
  }

  canvas.addEventListener('pointermove', (ev: PointerEvent): void => {
    px = ev.clientX;
    py = ev.clientY;
    pickDue = true;
  });
  canvas.addEventListener('pointerleave', (): void => {
    pickDue = false;
    hover(-1);
  });

  /* ---- select. CAPTURE phase: the orbit camera resolves its own gesture from a bubble-phase
     pointerup on the canvas, so the host must read consumeClick before that listener resets
     the gesture state. The flycam leaves pointerup entirely to the host. */
  window.addEventListener(
    'pointerup',
    (ev: PointerEvent): void => {
      const wasClick = cam.consumeClick(ev);
      if (!wasClick || ev.target !== canvas) return;
      const i = castAt(ev.clientX, ev.clientY); // THIS event's coordinate, never the last move's
      field.setSelect(i);
      if (i >= 0) {
        // The node becomes the orbit pivot — the camera stays put (keepRadius false), so a
        // click re-aims what dragging turns around without jumping the view.
        cam.setTarget(new Vector3(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!), false);
      }
      updateOverlay();
    },
    true,
  );

  window.addEventListener('keydown', (ev: KeyboardEvent): void => {
    if (ev.key === 'Escape') {
      if (field.setSelect(-1)) updateOverlay();
    }
  });

  window.addEventListener('resize', () => {
    W = window.innerWidth;
    H = window.innerHeight;
    renderer.setSize(W, H);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    field.resize(H); // CSS px — the shared pixel unit
  });

  /* ---- live stats: rAF-interval median over 90 frames (SCALE.md's measurement), draw calls
     from renderer.info, the field's budget report, the camera's telemetry (hidden states —
     room, damping — belong on screen, §10.7), and the hover/selection readout. */
  const intervals = new Float64Array(90);
  let filled = 0;
  let cursor = 0;
  let prevT = -1;
  let frame = 0;
  let drawCalls = 0;
  let room = Infinity;
  let speed = 0;
  let blocked = false;

  const fpsMedian = (): number => {
    const k = filled;
    if (k === 0) return 0;
    const copy = Array.from(intervals.subarray(0, k)).sort((a, b) => a - b);
    const ms = copy[k >> 1]!;
    return ms > 0 ? 1000 / ms : 0;
  };

  const backendName =
    (renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend === true
      ? 'webgpu'
      : 'webgl2';
  const fmt = (x: number): string => x.toLocaleString('en-US');
  function updateOverlay(): void {
    const s = field.stats();
    const hov = field.hover;
    const sel = field.select;
    statsEl.textContent =
      `nodes ${fmt(s.nodes)} · links ${fmt(s.links.drawn)} / ${fmt(s.links.total)} drawn · ` +
      `draw calls ${drawCalls} · fps ${fpsMedian().toFixed(1)} (median of ${filled} rAF intervals) · ` +
      `graph ${(tGraph - t0).toFixed(0)} ms · ${backendName} · cam ${camMode}\n` +
      `room ${Number.isFinite(room) ? room.toFixed(0) : '∞'} · speed ${speed.toFixed(0)}` +
      `${blocked ? ' (damped)' : ''} · ` +
      `hover: ${hov >= 0 ? graph.labelOf(hov) : '—'} · ` +
      `selected: ${sel >= 0 ? graph.labelOf(sel) : '—'}`;
  }

  const loop = (t: number): void => {
    let dt = 0;
    if (prevT >= 0) {
      const dtMs = t - prevT;
      intervals[cursor] = dtMs;
      cursor = (cursor + 1) % intervals.length;
      if (filled < intervals.length) filled++;
      // clamp the step so a backgrounded tab cannot slingshot the cameras
      dt = Math.min(dtMs, 100) * 0.001;
    }
    prevT = t;

    const telem = cam.update(dt);
    room = telem.room;
    speed = telem.speed;
    blocked = telem.blocked;

    // Hover resolves against the latest cursor position, once per frame at most — and not
    // mid-tween, when the ray would sweep nodes nobody is pointing at.
    if (pickDue && !(fly?.tweening ?? false)) {
      pickDue = false;
      hover(castAt(px, py));
    }

    renderer.render(scene, camera);
    drawCalls = renderer.info.render.drawCalls;
    if (frame % 30 === 0) updateOverlay();
    frame++;
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  // The debug handle every demo carries (§10.7). castAt is exposed so a headless browser
  // check can resolve "what is under this pixel" without synthesising pointer events.
  (window as unknown as Record<string, unknown>)['__starscape'] = {
    renderer,
    scene,
    camera,
    field,
    graph,
    grid,
    cam,
    castAt,
  };
}

main().catch((err: unknown) => {
  // §10.5: a boot failure must surface in the UI, never an idle page with an empty console.
  errorEl.style.display = 'block';
  errorEl.textContent = `boot failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`;
  statsEl.textContent = 'boot failed';
});
