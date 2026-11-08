/* The stage-3 demo: a synthetic corpus rendered as the two-draw-call field, on a dark ground,
 * under a slow deterministic orbit. Consumes the library exactly as an external user would
 * (DESIGN.md §8). `?n=100000` overrides the corpus size — synth is fast.
 *
 * House rules honoured: the orbit angle integrates per-frame dt (never Date.now / wall clock,
 * §10.1 — rAF deltas drive animation, geometry stays deterministic); the boot path has a
 * rejection handler that surfaces the error into the UI (§10.5); `window.__starscape` is the
 * debug handle (§10.7); the camera fits the LIVE extent (§10.8). FPS is measured the
 * SCALE.md way: median of 90 requestAnimationFrame intervals, never renderer.render timing.
 */

import { Color, PerspectiveCamera, Scene, WebGPURenderer } from 'three/webgpu';
import { buildGraph, createField, DEFAULT_FOV } from '../src/index';
import { generateSynth } from '../src/synth/generate';

const statsEl = document.getElementById('stats')!;
const errorEl = document.getElementById('error')!;

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const n = Math.max(1, Number(params.get('n') ?? 10000) | 0);
  const seed = Number(params.get('seed') ?? 42) | 0;

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

  // ?blend=additive opts into the glow look (stacks to white at high density until the
  // tonemap work lands); the default is the predecessor's shipped 'normal'.
  const field = createField(graph, {
    heightCss: window.innerHeight,
    fov: DEFAULT_FOV,
    blend: params.get('blend') === 'additive' ? 'additive' : 'normal',
  });

  const scene = new Scene();
  scene.background = new Color(field.background);
  for (const o of field.objects) scene.add(o);

  /* ---- camera, fitted to the live extent, orbiting deterministically. */
  const { centre, radius, diagonal } = field.extent;
  const dist = Math.max(1, radius * 1.9);
  const camera = new PerspectiveCamera(
    DEFAULT_FOV,
    window.innerWidth / window.innerHeight,
    Math.max(0.1, dist * 0.002),
    (dist + diagonal) * 3,
  );

  let angle = 0; // integrated from per-frame dt — deterministic given the frame sequence
  const ORBIT_RATE = 0.04; // rad/s, slow — stage 4 replaces this with real cameras
  const placeCamera = (): void => {
    camera.position.set(
      centre[0] + Math.sin(angle) * dist,
      centre[1] + dist * 0.3,
      centre[2] + Math.cos(angle) * dist,
    );
    camera.lookAt(centre[0], centre[1], centre[2]);
  };
  placeCamera();

  window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    field.resize(window.innerHeight); // CSS px — the shared pixel unit
  });

  /* ---- live stats: rAF-interval median over 90 frames (SCALE.md's measurement), draw calls
     from renderer.info, the field's own budget report. */
  const intervals = new Float64Array(90);
  let filled = 0;
  let cursor = 0;
  let prevT = -1;
  let frame = 0;
  let drawCalls = 0;

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
  const updateOverlay = (): void => {
    const s = field.stats();
    statsEl.textContent =
      `nodes ${fmt(s.nodes)} · links ${fmt(s.links.drawn)} / ${fmt(s.links.total)} drawn · ` +
      `draw calls ${drawCalls} · fps ${fpsMedian().toFixed(1)} (median of ${filled} rAF intervals) · ` +
      `graph ${(tGraph - t0).toFixed(0)} ms · ${backendName}`;
  };

  const loop = (t: number): void => {
    if (prevT >= 0) {
      const dtMs = t - prevT;
      intervals[cursor] = dtMs;
      cursor = (cursor + 1) % intervals.length;
      if (filled < intervals.length) filled++;
      // clamp the integration step so a backgrounded tab cannot slingshot the orbit
      angle += Math.min(dtMs, 100) * 0.001 * ORBIT_RATE;
      placeCamera();
    }
    prevT = t;
    renderer.render(scene, camera);
    drawCalls = renderer.info.render.drawCalls;
    if (frame % 30 === 0) updateOverlay();
    frame++;
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  // The debug handle every demo carries (§10.7).
  (window as unknown as Record<string, unknown>)['__starscape'] = {
    renderer,
    scene,
    camera,
    field,
    graph,
  };
}

main().catch((err: unknown) => {
  // §10.5: a boot failure must surface in the UI, never an idle page with an empty console.
  errorEl.style.display = 'block';
  errorEl.textContent = `boot failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`;
  statsEl.textContent = 'boot failed';
});
