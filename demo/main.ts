/* The stage-6 demo: the two-draw-call field under real cameras with the full §4 transform
 * pipeline live — GPU morph between arrangements, frame chains spinning per level, spread,
 * 2d/3d projection — driven from the ARRANGE panel, consuming the library exactly as an
 * external user would (DESIGN.md §8). `?n=100000` overrides the corpus size; `?cam=orbit`
 * swaps the flycam; `?blend=additive` opts into the glow look; `?pull=texture` forces the
 * DataTexture pull path even on WebGPU (the ./stores seam, testable from the URL).
 *
 * House rules honoured: no wall clock in anything that produces geometry — rAF deltas
 * drive cameras, tweens and frames (§10.1); the boot path surfaces rejections in the UI
 * (§10.5); `window.__starscape` is the debug handle and the overlay shows hidden states —
 * room, damping, budgets, morph t, inside-frame, jsFrameMs (§10.7); cameras fly to the
 * LIVE target extent, never a stored one (§10.8).
 *
 * Picking under motion (DESIGN §6): the grid is rebuilt on SETTLE (morph completed, spin
 * stopped, camera tween done) — never per frame. While anything is in flight the hover
 * pick is suppressed; the drawn positions and the grid would disagree, and a wrong answer
 * is worse than none. Selection targets read the same settled view snapshot.
 */

import { Color, PerspectiveCamera, Scene, Vector3, WebGPURenderer } from 'three/webgpu';
import {
  attachToAnchors,
  buildGraph,
  buildGrid,
  compileHierarchy,
  createField,
  createFlycam,
  createOrbitCam,
  createTweens,
  fnv1a,
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
import type { CameraController, CompiledHierarchy, Flycam, LevelSpec } from '../src/index';
import { generateSynth } from '../src/synth/generate';

const statsEl = document.getElementById('stats')!;
const errorEl = document.getElementById('error')!;

/** The four non-frozen arrangements: the 2-level type>system hierarchy with this leaf. */
const LEAVES: Record<string, LevelSpec['place']> = {
  galaxies: { preset: 'spiralGalaxy' },
  ring: { shape: 'ring', scatter: 'gaussian', params: { thickness: 0.06 } },
  globe: { shape: 'shell' },
  random: { shape: 'cloud' },
};

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
     unavailable — and the position-store pull mode follows the backend that actually booted
     (the ./stores seam): storage buffers on WebGPU, DataTexture texelFetch on WebGL2. */
  const renderer = new WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  await renderer.init();
  document.body.appendChild(renderer.domElement);
  const canvas = renderer.domElement;
  const backendName =
    (renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend === true
      ? 'webgpu'
      : 'webgl2';
  const pullMode =
    params.get('pull') === 'texture' ? 'texture' : backendName === 'webgpu' ? 'storage' : 'texture';

  const field = createField(graph, {
    heightCss: window.innerHeight,
    fov: DEFAULT_FOV,
    blend: params.get('blend') === 'additive' ? 'additive' : 'normal',
    pull: pullMode,
  });

  const scene = new Scene();
  scene.background = new Color(field.background);
  for (const o of field.objects) scene.add(o);

  const positions = graph.positions!;
  const fnvFrozen = fnv1a(positions); // the §4 round-trip pin: basePositions must return here
  const { centre, radius, diagonal } = field.extent;
  const dRef = diagonal * D_REF_FRACTION; // the same falloff pivot the node shader uses

  /* ---- picking: ONE grid per field, over the SETTLED view snapshot (rebuilt on settle,
     never per frame — see the header). */
  const viewBuf = new Float32Array(n * 3);
  viewBuf.set(positions);
  let grid = buildGrid(viewBuf, n);
  const rebuildGrid = (): void => {
    field.viewPositions(viewBuf);
    grid = buildGrid(viewBuf, n);
  };

  /* ---- cameras. Flycam by default (?cam=orbit for the simple one). */
  const camera = new PerspectiveCamera(
    DEFAULT_FOV,
    window.innerWidth / window.innerHeight,
    Math.max(0.05, radius * 0.001),
    (radius + diagonal) * 20, // headroom for spread×3 arrangements
  );
  let liveFov = DEFAULT_FOV;

  const roomPos = { x: 0, y: 0, z: 0 }; // one reusable probe — no per-frame allocation
  const roomAt = (p: readonly [number, number, number]): number => {
    roomPos.x = p[0];
    roomPos.y = p[1];
    roomPos.z = p[2];
    return nearestDistance(grid, roomPos);
  };
  const softStop = (): number => radius * 0.02;
  const zoomRange = (): readonly [number, number] => [radius * 0.01, diagonal * 4];

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

  /* ---- demo-side tweens: camera FOV for the 2d/3d flip (field tweens live inside the
     field; the camera is the demo's). Driven by the same rAF dt. */
  const demoTweens = createTweens();
  demoTweens.set('fov', DEFAULT_FOV);
  demoTweens.onChange('fov', (v) => {
    liveFov = v;
    camera.fov = v;
    camera.updateProjectionMatrix();
    field.setFov(v);
  });

  /** Fly the camera to frame the current TARGET extent — the cameras-module home() framing
   *  (radius / tan(fov/2) with the same 1.1 margin), tweened via goTo on the flycam. */
  const flyHome = (fovDeg: number): void => {
    const e = field.targetExtent();
    const c = new Vector3(e.centre[0], e.centre[1], e.centre[2]);
    const dist = (Math.max(e.radius, radius * 0.05) / Math.tan((fovDeg * Math.PI) / 360)) * 1.1;
    if (fly) fly.goTo(c, dist, 900);
    else cam.home({ center: e.centre, radius: e.radius });
  };

  /* ---- the ARRANGE panel ------------------------------------------------------------- */
  const ui = {
    arrangement: document.getElementById('arrangement') as HTMLSelectElement,
    spin0: document.getElementById('spin0') as HTMLInputElement,
    rate0: document.getElementById('rate0') as HTMLInputElement,
    spin1: document.getElementById('spin1') as HTMLInputElement,
    rate1: document.getElementById('rate1') as HTMLInputElement,
    spread: document.getElementById('spread') as HTMLInputElement,
    spreadVal: document.getElementById('spreadVal') as HTMLSpanElement,
    proj: document.getElementById('proj') as HTMLSelectElement,
    pull: document.getElementById('pull') as HTMLInputElement,
  };

  let hierarchy: CompiledHierarchy | null = null;
  const spacing = radius / Math.cbrt(Math.max(n, 1)); // arrangements fill the frozen volume

  /** Lazily build the 'pull' facet: every node BFS-attached (attachToAnchors) to the
   *  top-degree anchor of its nearest `type` group — forcing the anchors onto a shape
   *  blooms the rest around them (§3.4 pull). Deterministic: anchors are argmax degree,
   *  ties to the lowest id. */
  let pullReady = false;
  const buildPullFacet = (): void => {
    if (pullReady) return;
    const typeFacet = graph.facets['type']!;
    const deg = graph.columns['degree']!.data;
    const nv = typeFacet.values.length;
    const best = new Int32Array(nv).fill(-1);
    for (let i = 0; i < n; i++) {
      const v = typeFacet.data[i]!;
      const b = best[v]!;
      if (b < 0 || deg[i]! > deg[b]!) best[v] = i;
    }
    const anchors: number[] = [];
    for (let v = 0; v < nv; v++) if (best[v]! >= 0) anchors.push(best[v]!);
    anchors.sort((a, b) => a - b);
    const att = attachToAnchors(graph, Uint32Array.from(anchors));
    const ordinal = new Map<number, number>();
    for (let k = 0; k < anchors.length; k++) ordinal.set(anchors[k]!, k);
    const data = new Uint16Array(n);
    for (let i = 0; i < n; i++) data[i] = ordinal.get(att.anchorOf[i]!)!;
    graph.facets['pull'] = {
      key: 'pull',
      values: anchors.map((a) => graph.labelOf(a)),
      data,
    };
    pullReady = true;
  };

  /** Write the live spin toggles into the pivot table — the FrameSet reads `rate` fresh
   *  every advance (frames.ts), so no recompile: a parked frame resumes from its angle. */
  const applySpins = (): void => {
    if (!hierarchy) return;
    const pv = hierarchy.pivots;
    const on = [ui.spin0.checked, ui.spin1.checked];
    const rates = [Number(ui.rate0.value) || 0, Number(ui.rate1.value) || 0];
    for (let p = 0; p < pv.count; p++) {
      const lv = pv.level[p]!;
      pv.rate[p] = lv <= 1 && on[lv] ? rates[lv]! : 0;
    }
  };

  const applyArrangement = (): void => {
    const name = ui.arrangement.value;
    if (name === 'frozen') {
      hierarchy = null;
      field.setHierarchy(null);
    } else {
      const leaf = LEAVES[name]!;
      let levels: LevelSpec[];
      if (ui.pull.checked) {
        buildPullFacet();
        levels = [
          { groupBy: 'pull', place: { shape: 'shell' }, spin: { rate: 0 } },
          { place: leaf },
        ];
      } else {
        levels = [
          { groupBy: 'type', place: { shape: 'shell' }, spin: { rate: 0 } },
          { groupBy: 'system', place: { shape: 'cloud' }, spin: { rate: 0 } },
          { place: leaf },
        ];
      }
      hierarchy = compileHierarchy(graph, levels, { spacing });
      field.setHierarchy(hierarchy);
      applySpins();
    }
    flyHome(liveFov);
    updateOverlay();
  };

  const applyProjection = (): void => {
    const mode = ui.proj.value === '2d' ? '2d' : '3d';
    field.setProjection(mode);
    const fovTarget = mode === '2d' ? 30 : DEFAULT_FOV; // narrow lens flattens the read
    demoTweens.to('fov', fovTarget, { duration: 0.9 });
    flyHome(fovTarget);
  };

  ui.arrangement.addEventListener('change', applyArrangement);
  ui.pull.addEventListener('change', applyArrangement);
  ui.spin0.addEventListener('change', applySpins);
  ui.spin1.addEventListener('change', applySpins);
  ui.rate0.addEventListener('change', applySpins);
  ui.rate1.addEventListener('change', applySpins);
  ui.proj.addEventListener('change', applyProjection);
  ui.spread.addEventListener('input', () => {
    const v = Number(ui.spread.value);
    ui.spreadVal.textContent = v.toFixed(2);
    field.setSpread(v, 0.15); // a short tween — lands EXACTLY on the slider value
  });

  /* ---- hover: cast at most once per frame, from the latest cursor position, and only
     while the view is settled (the grid is only valid then — see the header). */
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
    return pickRay(grid, camera.position, RAY, {
      // The shader's own sizing, constant for constant, so the hit disc IS the drawn disc
      // (one CSS-pixel unit shared by shader, picker and ladder — DESIGN.md §5).
      tanHalf: Math.tan((liveFov * Math.PI) / 360),
      halfH: H / 2,
      sizeScale: NODE_SCALE,
      minPx: NODE_MIN_PX,
      maxPx: NODE_MAX_PX,
      falloff: FALLOFF,
      dRef,
      sizeOf: (i: number): number => field.sizeOf(i),
      boostOf: (i: number): number => field.boostOf(i),
      drift: 0, // the grid is rebuilt on settle; picks are suppressed in flight
    });
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
      if (!settledNow) return; // the grid is stale mid-flight; a wrong select is worse than none
      const i = castAt(ev.clientX, ev.clientY); // THIS event's coordinate, never the last move's
      field.setSelect(i);
      if (i >= 0) {
        // The node becomes the orbit pivot — the camera stays put (keepRadius false), so a
        // click re-aims what dragging turns around without jumping the view. viewBuf is the
        // settled view snapshot — where the node is DRAWN, not its frozen coordinate.
        cam.setTarget(new Vector3(viewBuf[i * 3]!, viewBuf[i * 3 + 1]!, viewBuf[i * 3 + 2]!), false);
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

  /* ---- live stats: rAF-interval median over 90 frames (SCALE.md's measurement) PLUS
     jsFrameMs — the per-frame CPU around the update calls (cameras, tweens, frames, pick),
     median of 90, NEVER renderer.render (gl.finish does not block in Chrome). Telemetry
     shows hidden states (§10.7): room, damping, budgets, morph t, inside-frame. */
  const intervals = new Float64Array(90);
  const jsMs = new Float64Array(90);
  let filled = 0;
  let cursor = 0;
  let jsFilled = 0;
  let jsCursor = 0;
  let prevT = -1;
  let frame = 0;
  let drawCalls = 0;
  let room = Infinity;
  let speed = 0;
  let blocked = false;
  let settledNow = true;
  let wasSettled = true;

  const medianOf = (buf: Float64Array, k: number): number => {
    if (k === 0) return 0;
    const copy = Array.from(buf.subarray(0, k)).sort((a, b) => a - b);
    return copy[k >> 1]!;
  };
  const fpsMedian = (): number => {
    const ms = medianOf(intervals, filled);
    return ms > 0 ? 1000 / ms : 0;
  };
  const jsFrameMs = (): number => medianOf(jsMs, jsFilled);

  /** The frame the camera is inside (§3.5 telemetry): frameOfPoint over the camera
   *  position mapped back through the inverse spread (the pivot spheres live unspread). */
  const insideFrame = (): string => {
    const f = field.frames;
    if (!f) return 'world';
    const s = field.spread;
    const c = field.centroid();
    roomPos.x = c[0] + (camera.position.x - c[0]) / s;
    roomPos.y = c[1] + (camera.position.y - c[1]) / s;
    roomPos.z = c[2] + (camera.position.z - c[2]) / s;
    const piv = f.frameOfPoint([roomPos.x, roomPos.y, roomPos.z]);
    if (piv < 0) return 'world';
    const lv = hierarchy ? hierarchy.pivots.level[piv]! : 0;
    return `p${piv}/L${lv}`;
  };

  const fmt = (x: number): string => x.toLocaleString('en-US');
  function updateOverlay(): void {
    const s = field.stats();
    const hov = field.hover;
    const sel = field.select;
    statsEl.textContent =
      `nodes ${fmt(s.nodes)} · links ${fmt(s.links.drawn)} / ${fmt(s.links.total)} drawn · ` +
      `draw calls ${drawCalls} · fps ${fpsMedian().toFixed(1)} (median of ${filled}) · ` +
      `jsFrameMs ${jsFrameMs().toFixed(2)} · graph ${(tGraph - t0).toFixed(0)} ms · ` +
      `${backendName}/${pullMode} · cam ${camMode}\n` +
      `inside:${insideFrame()} · morph t ${s.morph.t.toFixed(2)} · spread ${s.spread.toFixed(2)} · ` +
      `${s.projection} · pivots ${fmt(s.pivots)}${settledNow ? '' : ' · animating'}\n` +
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

    const c0 = performance.now(); // jsFrameMs window opens: every per-frame CPU update
    const telem = cam.update(dt);
    room = telem.room;
    speed = telem.speed;
    blocked = telem.blocked;

    demoTweens.update(dt);
    field.update(dt); // tweens + frames: O(pivots), zero when idle (§4)

    settledNow = field.settled() && !demoTweens.active() && !(fly?.tweening ?? false);
    if (settledNow && !wasSettled) {
      rebuildGrid(); // on settle — never per frame (DESIGN §6)
      updateOverlay();
    }
    wasSettled = settledNow;

    // Hover resolves against the latest cursor position, once per frame at most — and only
    // while settled (grid validity; also not mid-tween, when the ray sweeps nodes nobody
    // is pointing at).
    if (pickDue && settledNow) {
      pickDue = false;
      hover(castAt(px, py));
    }
    const c1 = performance.now(); // jsFrameMs window closes BEFORE render
    jsMs[jsCursor] = c1 - c0;
    jsCursor = (jsCursor + 1) % jsMs.length;
    if (jsFilled < jsMs.length) jsFilled++;

    renderer.render(scene, camera);
    drawCalls = renderer.info.render.drawCalls;
    if (frame % 30 === 0) updateOverlay();
    frame++;
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  // The debug handle every demo carries (§10.7). The setters mirror the panel so a headless
  // browser check can drive the ARRANGE flow and verify the §4 round-trip FNV equality
  // programmatically; fnvBase() hashes the CPU base positions (mix with exact endpoints).
  (window as unknown as Record<string, unknown>)['__starscape'] = {
    renderer,
    scene,
    camera,
    field,
    graph,
    cam,
    castAt,
    fnvFrozen,
    fnvBase: (): number => fnv1a(field.basePositions()),
    settled: (): boolean => settledNow,
    fpsMedian,
    jsFrameMs,
    setArrangement: (name: string): void => {
      ui.arrangement.value = name;
      applyArrangement();
    },
    setProjection: (mode: '2d' | '3d'): void => {
      ui.proj.value = mode;
      applyProjection();
    },
    setSpread: (v: number): void => {
      ui.spread.value = String(v);
      ui.spreadVal.textContent = v.toFixed(2);
      field.setSpread(v, 0.15);
    },
    setSpin: (level: 0 | 1, on: boolean, rate?: number): void => {
      (level === 0 ? ui.spin0 : ui.spin1).checked = on;
      if (rate !== undefined) (level === 0 ? ui.rate0 : ui.rate1).value = String(rate);
      applySpins();
    },
    setPull: (on: boolean): void => {
      ui.pull.checked = on;
      applyArrangement();
    },
    // §9: "viewpoint asserted (never trusted)". The idle turntable keeps the camera
    // orbiting, so an fps A/B measured over it compares two different fills — baseline
    // drift dwarfs small per-feature costs. A measurement harness parks it with
    // setTurntable(false) (and restores it after) so both windows share one viewpoint.
    setTurntable: (on: boolean): void => {
      if (fly) fly.spin = on;
    },
  };

  updateOverlay();
}

main().catch((err: unknown) => {
  // §10.5: a boot failure must surface in the UI, never an idle page with an empty console.
  errorEl.style.display = 'block';
  errorEl.textContent = `boot failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`;
  statsEl.textContent = 'boot failed';
});
