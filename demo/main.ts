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
 * Picking and the ladder under motion (DESIGN §6): the grid is rebuilt on SETTLE — never
 * per frame — and, while spins keep the field from ever settling, lazily on a drift
 * budget (the predecessor's ensureGrid: the error is TRACKED, walks widen by it, and the
 * O(n) rebuild is rate-limited off the steady frame clock). A MORPH suppresses picking
 * and the candidate pass outright (base positions are mid-flight; the dwell hysteresis
 * absorbs the 0.9 s gap), but a SPINNING field stays fully alive: every resident node the
 * candidate walk visits is re-transformed through the live frame chain (O(1) per node,
 * frames.transformPoint — §4's "per-frame CPU is O(pivots + active), never O(n)"), so
 * labels, cards, open and select all work with all spins on — §9's 10⁶ acceptance row is
 * measured in exactly that state.
 */

import { Color, Matrix4, PerspectiveCamera, Scene, Vector3, WebGPURenderer } from 'three/webgpu';
import type { BufferAttribute, InstancedBufferAttribute, LineSegments, Mesh } from 'three/webgpu';
import {
  attachToAnchors,
  bootQuality,
  buildGraph,
  buildGrid,
  cellMaxOf,
  compileHierarchy,
  createConfigStore,
  createField,
  createFlycam,
  createLadder,
  createOrbitCam,
  createOverlay,
  createPanel,
  createTweens,
  facetPalette,
  fnv1a,
  fovPxOf,
  injectPanelStyles,
  injectStyles,
  localStorageAdapter,
  makeSynthProvider,
  nearestDistance,
  paintFacetColors,
  paletteBuffer,
  pickRay,
  projectedPx,
  sizesFromColumn,
  standoffOf,
  uniformSizes,
  BG_LIGHT,
  BOOST_HOVER,
  BOOST_SELECT,
  DEFAULT_FOV,
  D_REF_FRACTION,
  FALLOFF,
  FLYCAM_DEFAULTS,
  HOME_MARGIN,
  NODE_MAX_PX,
  NODE_MIN_PX,
  NODE_SCALE,
  ORBIT_DEFAULTS,
  PALETTES,
  UNIFORM_RGB,
  VISUAL_PRESETS,
} from '../src/index';
import type {
  CameraController,
  CompiledHierarchy,
  ConfigValue,
  Flycam,
  FlycamConfig,
  LadderCandidates,
  LadderView,
  LevelSpec,
  PaletteName,
  RampName,
  TonemapOp,
} from '../src/index';
import { generateSynth } from '../src/synth/generate';

const statsEl = document.getElementById('stats')!;
const errorEl = document.getElementById('error')!;

/** The four non-frozen arrangements: the 2-level type>galaxy hierarchy with this leaf.
 *  `galaxy`, not `system`: since the synth generator's constant-density rescale the system
 *  facet carries ~n^⅔ values (10,000 at 10⁶ — 7×10,000 pivots would overflow the pivot
 *  store AND break §3.5's "rotation work is one quaternion per pivot, hundreds"), while
 *  galaxies grow as ~n^⅓ (100 at 10⁶ → ~700 pivots). */
const LEAVES: Record<string, LevelSpec['place']> = {
  galaxies: { preset: 'spiralGalaxy' },
  ring: { shape: 'ring', scatter: 'gaussian', params: { thickness: 0.06 } },
  globe: { shape: 'shell' },
  random: { shape: 'cloud' },
};

/** The candidate pass's frustum margins, in NDC (the predecessor's ±1.25 / ±1.3). ONE copy:
 *  the reach slack in gatherCandidates is DERIVED from these, so widening the gate widens the
 *  grid-walk sphere with it — two literal copies meant a gate change silently under-sized the
 *  walk and labels popped in late at the frustum edge (§10: constants exported once). */
const NDC_MARGIN_X = 1.25;
const NDC_MARGIN_Y = 1.3;

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

  /* ---- stage 8: the config store. Loaded BEFORE the field exists, because the boot-routed
     keys (blend, size channel, the focus alphas) are read at createField(). Only touched keys
     were persisted (store rules), the presets are the six theme looks, and the load warnings
     are returned — surfaced on the debug handle, never console.warn'd by the library. */
  const store = createConfigStore({ storage: localStorageAdapter(), presets: VISUAL_PRESETS });
  const configWarnings = store.load();
  /* Stage 9: the boot quality notch — the corpus picks the starting budgets, every load,
     like a game detecting the hardware once at startup (quality.ts: the ported predecessor
     policy, no feedback loop). Written INTO the session-scoped config keys, so the panel
     shows what is actually drawn and the user overrides by dragging the same knobs; the
     store never persists them (schema SESSION_KEYS), so a budget tuned for 10⁶ pages can
     never silently survive into a 10³ session. */
  const quality = bootQuality(n, window.devicePixelRatio || 1);
  store.set('linkBudget', quality.linkFraction);
  store.set('renderScale', quality.renderScale);
  store.set('flatPx', quality.flatPx);
  const num = (k: string): number => store.get(k) as number;
  const str = (k: string): string => store.get(k) as string;
  const bool = (k: string): boolean => store.get(k) === true;

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
    blend: str('blend') === 'additive' ? 'additive' : 'normal',
    sizeColumn: str('sizeChannel') === 'uniform' ? 'uniform' : 'degree',
    pull: pullMode,
    /* BOOT-routed keys (see the routing table below): the focus alphas are baked into the
       focus state at creation — a live change persists and lands on the next reload. */
    focus: {
      linkBase: num('linkBase'),
      dimOthers: num('dimOthers'),
      linkHover: num('linkHover'),
      linkFocus: num('linkFocus'),
      linkDim: num('linkDim'),
    },
  });

  const scene = new Scene();
  scene.background = new Color(field.background);
  for (const o of field.objects) scene.add(o);

  /* The per-instance channel buffers, reached through the PUBLIC three.js objects
     (field.objects is [links, nodes] — field.ts). The iSize/iColor arrays ARE the field's own
     channel buffers (field.sizeOf reads the same Float32Array), so a demo rewrite keeps the
     shader, the picker and the ladder agreeing about every node — the one-pixel-unit rule. */
  const linkLines = field.objects[0] as LineSegments;
  const nodeMesh = field.objects[1] as Mesh;
  const sizeAttr = nodeMesh.geometry.getAttribute('iSize') as InstancedBufferAttribute;
  const colorAttr = nodeMesh.geometry.getAttribute('iColor') as InstancedBufferAttribute;
  const linkColorAttr = linkLines.geometry.getAttribute('lColor') as BufferAttribute;

  const positions = graph.positions!;
  const fnvFrozen = fnv1a(positions); // the §4 round-trip pin: basePositions must return here
  const { centre, radius, diagonal } = field.extent;
  const dRef = diagonal * D_REF_FRACTION; // the same falloff pivot the node shader uses

  /* ---- picking: ONE grid per field, over the view snapshot — rebuilt on settle, on a
     morph landing, and lazily under spin once the tracked drift passes a third of a cell
     (see the header). */
  const viewBuf = new Float32Array(n * 3);
  viewBuf.set(positions);
  let grid = buildGrid(viewBuf, n);
  /* Drift bookkeeping, stage 9: `driftTotal` is MONOTONE (never reset) so a structure pinned
     to an older grid — the far candidate sweep below survives rebuilds — can widen by exactly
     the drift since ITS grid was built; `gridDrift` (drift since the CURRENT grid) derives
     from it. `lastRebuildMs` is telemetry: the rebuild is the one O(n) cost left on the
     spinning path, and §10.7 says hidden costs are shown, not implied. */
  let driftTotal = 0;
  let gridDriftBase = 0;
  let gridDrift = 0; // ≡ driftTotal − gridDriftBase, kept unrolled for the hot paths
  let gridCooldown = 0; // seconds until the next in-spin rebuild (the port's 250 ms limiter)
  let lastRebuildMs = 0;
  /* Per-cell max of the size channel — the far sweep's cell-level pruning bound (pick/grid
     cellMaxOf). Follows the grid (rebuilds) AND the channel (applySizes): both events. */
  let gridCellMax = cellMaxOf(grid, (i) => field.sizeOf(i));
  let roomValid = false; // the roomAt cache below — invalidated by every rebuild
  const rebuildGrid = (): void => {
    const r0 = performance.now();
    field.viewPositions(viewBuf);
    grid = buildGrid(viewBuf, n);
    // Reuse the cellMax slab UNLESS the far sweep has it pinned (sweepCellMax below): a
    // sweep mid-generation keeps pruning against ITS grid, and rewriting its bound array in
    // place would silently mismatch bounds to cells. The guard costs one allocation per
    // rebuild only while a sweep is actually in flight.
    gridCellMax = cellMaxOf(
      grid,
      (i) => field.sizeOf(i),
      gridCellMax === sweepCellMax ? undefined : gridCellMax,
    );
    gridDriftBase = driftTotal;
    gridDrift = 0;
    roomValid = false;
    lastRebuildMs = performance.now() - r0;
  };

  /* The CPU mirror of the shader's BASE endpoints (mix(A,B,t), parked), refreshed when a
     morph lands. refreshViewOf() composes the live frame chain + spread on top of it for
     ONE node — exactly the vertex-shader arithmetic — so the ladder, the overlay and the
     orbit target can follow a SPINNING field with O(active) work, never O(n) (§4; the
     predecessor recomposed all n on the CPU every spinning frame, the flaw this design
     removes). Valid whenever no morph is rewriting the base. */
  const baseBuf = new Float32Array(n * 3);
  baseBuf.set(positions);
  let morphing = false; // a retarget tween is rewriting the base endpoints right now
  let fieldSettledNow = true; // field.settled() this frame: frames parked, no field tween
  /* Worst-case node speed under the live spins, world units/s before spread — drives the
     drift bound. Stage 9: from PIVOT RADII (|rate| × lever, summed down the chain), not
     0.5 × diagonal: the old bound over-reported drift ~5× at 10⁶ (systems are small), and
     every walk pays the widening. Set by applySpins. */
  let spinDriftRate = 0;
  const spreadC: [number, number, number] = [0, 0, 0]; // live spread centre, synced per frame
  let spreadK = 1;
  const refreshViewOf = (i: number): void => {
    const o = i * 3;
    const f = field.frames;
    if (f !== null) {
      f.transformPoint(field.chainIds[i]!, baseBuf, viewBuf, o, o);
    } else {
      viewBuf[o] = baseBuf[o]!;
      viewBuf[o + 1] = baseBuf[o + 1]!;
      viewBuf[o + 2] = baseBuf[o + 2]!;
    }
    if (spreadK !== 1) {
      viewBuf[o] = spreadC[0] + (viewBuf[o]! - spreadC[0]) * spreadK;
      viewBuf[o + 1] = spreadC[1] + (viewBuf[o + 1]! - spreadC[1]) * spreadK;
      viewBuf[o + 2] = spreadC[2] + (viewBuf[o + 2]! - spreadC[2]) * spreadK;
    }
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
  /* roomAt, stage 9: CACHED by displacement. The flycam probes twice per frame (current
     position + step destination) and nearestDistance costs ~0.35 ms per probe at 10⁶ from
     the home framing — 0.7 ms of a 1 ms §9 frame-CPU budget for a number that cannot have
     changed while the camera is parked. The cache returns a safe LOWER bound instead: the
     nearest node can have approached by at most (probe displacement + node drift since the
     walk), so cached − slack is correct for the flycam's damping whatever happened — and the
     real walk is re-paid only once the slack passes half a cell. Parked on the home framing
     the probe is free; flying fast re-pays it, which SCALE.md reports. Rebuilds invalidate
     (roomValid, rebuildGrid above). */
  let roomX = 0;
  let roomY = 0;
  let roomZ = 0;
  let roomVal = Infinity;
  let roomDriftAt = 0;
  const roomAt = (p: readonly [number, number, number]): number => {
    if (roomValid) {
      const dx = p[0] - roomX;
      const dy = p[1] - roomY;
      const dz = p[2] - roomZ;
      const slack = Math.sqrt(dx * dx + dy * dy + dz * dz) + (driftTotal - roomDriftAt);
      if (slack < grid.cellMax * 0.5) {
        return roomVal === Infinity ? Infinity : Math.max(roomVal - slack, 0);
      }
    }
    roomPos.x = p[0];
    roomPos.y = p[1];
    roomPos.z = p[2];
    const d = nearestDistance(grid, roomPos, { drift: gridDrift });
    roomX = p[0];
    roomY = p[1];
    roomZ = p[2];
    roomVal = d;
    roomDriftAt = driftTotal;
    roomValid = true;
    return d;
  };
  const softStop = (): number => radius * 0.02;
  const zoomRange = (): readonly [number, number] => [radius * 0.01, diagonal * 4];

  /* The flycam reads its config object LIVE every frame (flycam.ts: "so a panel can mutate
     the object") — the flight group of the routing table writes into this one reference. */
  const flyCfg: FlycamConfig = { ...FLYCAM_DEFAULTS };
  let fly: Flycam | null = null;
  let cam: CameraController;
  if (camMode === 'fly') {
    fly = createFlycam(camera, flyCfg, { roomAt, softStop, zoomRange });
    fly.spin = bool('cameraDrift'); // the idle turntable is a config knob now (motion group)
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
   *  (radius / tan(fov/2) × HOME_MARGIN, the cameras' own exported constant), tweened via
   *  goTo on the flycam. */
  const flyHome = (fovDeg: number): void => {
    const e = field.targetExtent();
    const c = new Vector3(e.centre[0], e.centre[1], e.centre[2]);
    const dist =
      (Math.max(e.radius, radius * 0.05) / Math.tan((fovDeg * Math.PI) / 360)) * HOME_MARGIN;
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
   *  every advance (frames.ts), so no recompile: a parked frame resumes from its angle.
   *  The config's `levelSpin` knob OVERRIDES when > 0: every pivot at every level turns at
   *  that one rate (the schema ships one knob; per-level rates stay the ARRANGE panel's).
   *  In the frozen layout no frames exist, so the knob idles — exactly the schema's note. */
  const applySpins = (): void => {
    if (!hierarchy) {
      spinDriftRate = 0;
      return;
    }
    const override = num('levelSpin');
    const pv = hierarchy.pivots;
    const on = [ui.spin0.checked, ui.spin1.checked];
    const rates = [Number(ui.rate0.value) || 0, Number(ui.rate1.value) || 0];
    /* ONE pass sets the live rates AND accumulates the drift bound. A node riding pivot p
       moves at ≤ |rate_p| × (lever about p) ≤ |rate_p| × radius_p, and the levers of a
       chain ADD — so Σ over levels of max(|rate| × radius) bounds every node's speed.
       ×1.5 margin: scatter may place a member a little outside its group sphere. (Per-level
       maxima, summed; the demo's hierarchies carry spin on at most L0/L1 + the override.) */
    let d0 = 0;
    let d1 = 0;
    let dRest = 0;
    for (let p = 0; p < pv.count; p++) {
      const lv = pv.level[p]!;
      const base = lv <= 1 && on[lv] ? rates[lv]! : 0;
      const r = override > 0 ? override : base;
      pv.rate[p] = r;
      const speed = Math.abs(r) * pv.radius[p]!;
      if (lv === 0) {
        if (speed > d0) d0 = speed;
      } else if (lv === 1) {
        if (speed > d1) d1 = speed;
      } else if (speed > dRest) {
        dRest = speed;
      }
    }
    spinDriftRate = (d0 + d1 + dRest) * 1.5;
  };

  const applyArrangement = (): void => {
    const name = ui.arrangement.value;
    morphing = true; // every branch below retargets — the base is in flight until t = 1
    if (name === 'frozen') {
      hierarchy = null;
      field.setHierarchy(null);
      applySpins(); // spinRateSum → 0: no frames exist in the frozen layout
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
          { groupBy: 'galaxy', place: { shape: 'cloud' }, spin: { rate: 0 } },
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
    if (mode !== field.projection) morphing = true; // setProjection no-ops on a same-mode call
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
  // spread moved into the tuning panel's `motion` group (stage 8) — one knob, one owner

  /* ---- stage 7: the zoom ladder + DOM overlay (DESIGN §6) ----------------------------
     The machine is pure (src/ladder/ladder.ts) — the demo's job is the candidate pass, the
     cursor, the pin (= the existing select state) and the per-frame overlay call. */
  injectStyles();
  const provider = makeSynthProvider(graph);
  const ladderRoot = document.getElementById('ladder')!;
  // the pin affordance reuses the existing select-clear, exactly like Escape below
  const onUnpin = (): void => {
    if (field.setSelect(-1)) updateOverlay();
  };
  /** Every ladder tunable, from the config (ladder group of the routing table). The threshold
   *  and band keys are ALSO mutated live on ladder.config; this builder serves the rebuild
   *  keys — namedCap/cards size fixed buffers and lockW/lockH freeze the overlay rect at
   *  creation, so those four recreate the machine + overlay (an event, never per frame). */
  const ladderOptsFromCfg = (): Parameters<typeof createLadder>[0] => ({
    namedPx: num('namedPx'),
    earn: num('labelEarn'),
    keep: num('labelKeep'),
    dwellMs: num('labelDwell'),
    namedCap: num('labelCap'),
    cardSpacings: num('tierCard'),
    lockSpacings: num('tierLock'),
    openSpacings: num('tierOpen'),
    cards: num('maxCards'),
    lockW: num('lockW'),
    lockH: num('lockH'),
    cardRadiusSpacings: num('cardRadius'),
  });
  let ladder = createLadder(ladderOptsFromCfg());
  let ladderOverlay = createOverlay({
    container: ladderRoot,
    provider,
    lockW: num('lockW'),
    lockH: num('lockH'),
    onUnpin,
  });
  const rebuildLadder = (): void => {
    ladderOverlay.dispose(); // unmounts every live label/card — the old rule's leftovers
    ladder = createLadder(ladderOptsFromCfg());
    ladderOverlay = createOverlay({
      container: ladderRoot,
      provider,
      lockW: num('lockW'),
      lockH: num('lockH'),
      onUnpin,
    });
  };

  // pointer state the ladder consumes (written by the pointermove wiring below)
  let pointerActive = false;
  let cursorMoved = false;

  // the one clip matrix (projection × worldInverse) the candidate pass and the overlay share
  const camInv = new Matrix4();
  const clipM = new Matrix4();

  // the largest size-channel radius in the corpus — the candidate reach bound. Maintained by
  // applySizes() (the nodes group rebinds the channel live). A loop, never a spread (§10.3).
  let maxSizeWorld = 0;
  for (let i = 0; i < n; i++) {
    const s = field.sizeOf(i);
    if (s > maxSizeWorld) maxSizeWorld = s;
  }
  const maxBoost = Math.max(BOOST_HOVER, BOOST_SELECT);

  /* THE CANDIDATE RECIPE, stage-9 edition — bounded per frame HOWEVER big the corpus.
     The stage-7 version walked every grid cell in reach each frame; at the §9 home framing
     reach covers the whole field, so it was an O(n) sweep per frame — measured 67 ms at 10⁶,
     all by itself two orders over the §9 frame-CPU budget. The fix splits the pass by what
     actually needs per-frame freshness:

     NEAR PASS (every frame, exact): cells within the CARD distance (dCard × frustum slack +
     drift) of the camera. Cards, locks and the open page key off live DEPTH — a stale
     distance would lag the open tier exactly where the ladder lives — and near the camera
     this is a handful of cells whatever n is. Per node: view-forward depth (clip w ≡ −z_view,
     what the shader sizes by), the px gate, the frustum margin test (±1.25/±1.3 NDC), the
     cursor pull. While spinning, each resident is re-formed through the live frame chain
     first (refreshViewOf — written back into viewBuf, so overlay and click target agree).

     FAR SWEEP (incremental, SWEEP_CELLS_PER_FRAME cells a frame): the named band only needs
     the best ~`keep` candidates by projected size, and label admission already carries dwell
     hysteresis — so membership may lag a few hundred ms, exactly the staleness class of the
     morph gap the stage-7 header accepted. One GENERATION sweeps the full reach box (the
     stage-7 reach formula) keeping a bounded top-K-by-px selection; cells are pruned with the
     per-cell max of the size channel (pick/grid cellMaxOf): a cell whose LARGEST node cannot
     project over the admission floor at the cell's nearest possible depth is skipped whole.
     The sweep pins ITS grid + cellMax + drift base, so a mid-generation rebuild (spins) never
     restarts it — the widening just grows by the drift since the PINNED grid was built. On
     completion the selection becomes the published FAR SET: node ids only, re-projected
     FRESH every frame (≤ FAR_CAP nodes), so rank, cursor pull and distances are live — only
     membership is a generation stale. A morph clears it (every position moved).

     REFINEMENT vs stage 7 (deliberate, documented in SCALE.md): far admission is by RESTING
     projected size — the cursor pull and the hover boost promote candidates within the
     published set but no longer recruit from outside the top-K×2 band. At the framings where
     that band is even oversubscribed (10⁵⁺ zoomed out, tens of thousands passing the px
     gate), individual sub-band glyphs are pixels apart and unpointable anyway.

     The pinned node is appended even when both passes missed it — a pin outranks the ladder.
     Columns are fixed scratch, grown geometrically, never allocated per frame (§10). */
  let candCap = 0;
  let candNode = new Uint32Array(0);
  let candPx = new Float64Array(0);
  let candDist = new Float64Array(0);
  let candCur = new Float64Array(0);
  const ladderCandidates: LadderCandidates = {
    count: 0,
    node: candNode,
    projectedPx: candPx,
    distance: candDist,
    cursorPx: candCur,
  };
  const ladderView: LadderView = {
    candidates: ladderCandidates,
    spacing,
    pinned: -1,
    cursorMoved: false,
  };
  const growCand = (min: number): void => {
    candCap = Math.max(256, candCap * 2, min);
    const nn = new Uint32Array(candCap);
    nn.set(candNode);
    candNode = nn;
    const np = new Float64Array(candCap);
    np.set(candPx);
    candPx = np;
    const nd = new Float64Array(candCap);
    nd.set(candDist);
    candDist = nd;
    const nc = new Float64Array(candCap);
    nc.set(candCur);
    candCur = nc;
    ladderCandidates.node = candNode;
    ladderCandidates.projectedPx = candPx;
    ladderCandidates.distance = candDist;
    ladderCandidates.cursorPx = candCur;
  };

  /* ---- far-sweep state (see THE CANDIDATE RECIPE above). All fixed scratch. ---- */
  /** Hard ceiling on the published far set (and its selection heap). */
  const FAR_CAP = 256;
  /** Cells examined per frame — the sweep's whole per-frame budget is this many cheap cell
   *  tests plus the members of the few that survive the pruning. 4096 ≈ a sixth of the 10⁵
   *  grid (6-frame generations) and 1/61 of the 10⁶ grid (~1.5 s generations): measured, the
   *  number where the slice stays under ~0.3 ms at 10⁵ (the §9 0.5 ms row leaves no room for
   *  more) while label-set refresh stays comfortably inside the dwell hysteresis. */
  const SWEEP_CELLS_PER_FRAME = 4096;
  const heapNode = new Uint32Array(FAR_CAP);
  const heapPx = new Float64Array(FAR_CAP);
  let heapK = 0;
  let heapLimit = FAR_CAP;
  const farNode = new Uint32Array(FAR_CAP);
  let farCount = 0;
  let sweepActive = false;
  let sweepGrid = grid; // the generation's PINNED grid (survives rebuilds mid-sweep)
  let sweepCellMax = gridCellMax;
  let sweepDriftBase = 0; // driftTotal when the pinned grid was built
  let sweepReach = 0;
  let sweepLoX = 0;
  let sweepHiX = -1;
  let sweepLoY = 0;
  let sweepHiY = -1;
  let sweepLoZ = 0;
  let sweepHiZ = -1;
  let sweepX = 0;
  let sweepY = 0;
  let sweepZ = 0;
  let sweepCells = 0; // telemetry: cells in this generation's box / done so far
  let sweepDone = 0;
  let sweepGen = 0;
  /** Which nodes THIS frame's near pass admitted — O(1) dedupe for the far merge. */
  const nearStamp = new Uint32Array(n);
  let nearEpoch = 0;

  const heapSift = (j: number): void => {
    // min-heap on px: root holds the admission floor once the selection is full
    for (;;) {
      const l = j * 2 + 1;
      if (l >= heapK) return;
      const r = l + 1;
      const c = r < heapK && heapPx[r]! < heapPx[l]! ? r : l;
      if (heapPx[c]! >= heapPx[j]!) return;
      const tn = heapNode[j]!;
      const tp = heapPx[j]!;
      heapNode[j] = heapNode[c]!;
      heapPx[j] = heapPx[c]!;
      heapNode[c] = tn;
      heapPx[c] = tp;
      j = c;
    }
  };
  const heapPush = (node: number, px: number): void => {
    if (heapK < heapLimit) {
      let j = heapK++;
      heapNode[j] = node;
      heapPx[j] = px;
      while (j > 0) {
        const p = (j - 1) >> 1;
        if (heapPx[p]! <= heapPx[j]!) break;
        const tn = heapNode[j]!;
        const tp = heapPx[j]!;
        heapNode[j] = heapNode[p]!;
        heapPx[j] = heapPx[p]!;
        heapNode[p] = tn;
        heapPx[p] = tp;
        j = p;
      }
    } else if (px > heapPx[0]!) {
      heapNode[0] = node;
      heapPx[0] = px;
      heapSift(0);
    }
  };

  /** Drop the far set and any sweep in flight — a morph moved every position, so a stale
   *  membership would name the wrong places; the next gather starts a fresh generation. */
  const resetSweep = (): void => {
    sweepActive = false;
    heapK = 0;
    farCount = 0;
  };

  /** Advance the far sweep by one budget slice (see THE CANDIDATE RECIPE). */
  function farSlice(fovPx: number, slack: number, dCard: number): void {
    if (!sweepActive) {
      // a new generation pins the CURRENT grid and derives its box from the live camera
      sweepGrid = grid;
      sweepCellMax = gridCellMax;
      sweepDriftBase = gridDriftBase;
      heapLimit = Math.min(FAR_CAP, Math.max(ladder.config.keep, ladder.config.earn) * 2);
      heapK = 0;
      sweepReach =
        Math.max(
          standoffOf(maxSizeWorld * NODE_SCALE * maxBoost, ladder.config.namedPx, fovPx),
          dCard,
        ) *
          slack +
        gridDrift;
      const g = sweepGrid;
      sweepLoX = Math.max(0, Math.floor((camera.position.x - sweepReach - g.minX) / g.cellX));
      sweepHiX = Math.min(g.side - 1, Math.floor((camera.position.x + sweepReach - g.minX) / g.cellX));
      sweepLoY = Math.max(0, Math.floor((camera.position.y - sweepReach - g.minY) / g.cellY));
      sweepHiY = Math.min(g.side - 1, Math.floor((camera.position.y + sweepReach - g.minY) / g.cellY));
      sweepLoZ = Math.max(0, Math.floor((camera.position.z - sweepReach - g.minZ) / g.cellZ));
      sweepHiZ = Math.min(g.side - 1, Math.floor((camera.position.z + sweepReach - g.minZ) / g.cellZ));
      sweepX = sweepLoX;
      sweepY = sweepLoY;
      sweepZ = sweepLoZ;
      sweepCells =
        Math.max(0, sweepHiX - sweepLoX + 1) *
        Math.max(0, sweepHiY - sweepLoY + 1) *
        Math.max(0, sweepHiZ - sweepLoZ + 1);
      sweepDone = 0;
      sweepActive = sweepCells > 0;
      if (!sweepActive) {
        farCount = 0;
        return;
      }
    }
    const g = sweepGrid;
    const cm = sweepCellMax;
    const live = !fieldSettledNow;
    const driftW = driftTotal - sweepDriftBase; // nodes may sit this far from their old cells
    const reachW = sweepReach + driftW;
    const r2 = reachW * reachW;
    const cx = camera.position.x;
    const cy = camera.position.y;
    const cz = camera.position.z;
    const m = clipM.elements;
    const namedPx = ladder.config.namedPx;
    /* Best-case px factor: diameter × NODE_SCALE × fovPx over the cell's minimum VIEW DEPTH.
       Depth (clip w) is linear over the cell box, so its exact minimum is one corner picked
       by coefficient sign — the same number the per-node gate divides by, with no Euclidean
       slack inflating the bound 1.6× (measured: the slack form passed ~3× the cells). */
    const pxNum = 2 * NODE_SCALE * fovPx;
    const m3 = m[3]!;
    const m7 = m[7]!;
    const m11 = m[11]!;
    const m15 = m[15]!;
    let budget = SWEEP_CELLS_PER_FRAME;
    while (budget > 0) {
      const z0 = g.minZ + sweepZ * g.cellZ;
      const y0 = g.minY + sweepY * g.cellY;
      const x0 = g.minX + sweepX * g.cellX;
      const dz0 = cz < z0 ? z0 - cz : cz > z0 + g.cellZ ? cz - (z0 + g.cellZ) : 0;
      const dy0 = cy < y0 ? y0 - cy : cy > y0 + g.cellY ? cy - (y0 + g.cellY) : 0;
      const dx0 = cx < x0 ? x0 - cx : cx > x0 + g.cellX ? cx - (x0 + g.cellX) : 0;
      const d2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
      budget--;
      sweepDone++;
      if (d2 <= r2) {
        const k = (sweepZ * g.side + sweepY) * g.side + sweepX;
        // the admission floor: the px gate, raised to the selection's own minimum once full
        const floor = heapK >= heapLimit && heapPx[0]! > namedPx ? heapPx[0]! : namedPx;
        const depthMin =
          (m3 >= 0 ? m3 * x0 : m3 * (x0 + g.cellX)) +
          (m7 >= 0 ? m7 * y0 : m7 * (y0 + g.cellY)) +
          (m11 >= 0 ? m11 * z0 : m11 * (z0 + g.cellZ)) +
          m15;
        const dMin = Math.max(depthMin - driftW, 1e-3);
        if ((cm[k]! * pxNum) / dMin >= floor) {
          const lo = g.start[k]!;
          const hi = g.start[k + 1]!;
          for (let q = lo; q < hi; q++) {
            const i = g.order[q]!;
            if (live) refreshViewOf(i);
            const o = i * 3;
            const wx = viewBuf[o]!;
            const wy = viewBuf[o + 1]!;
            const wz = viewBuf[o + 2]!;
            const cw = m[3]! * wx + m[7]! * wy + m[11]! * wz + m[15]!;
            if (cw <= 0) continue;
            // RESTING size (boost excluded — see the recipe's refinement note)
            const px = projectedPx(field.sizeOf(i) * NODE_SCALE, cw, fovPx);
            if (px < namedPx || (heapK >= heapLimit && px <= heapPx[0]!)) continue;
            const ndx = (m[0]! * wx + m[4]! * wy + m[8]! * wz + m[12]!) / cw;
            const ndy = (m[1]! * wx + m[5]! * wy + m[9]! * wz + m[13]!) / cw;
            if (ndx < -NDC_MARGIN_X || ndx > NDC_MARGIN_X || ndy < -NDC_MARGIN_Y || ndy > NDC_MARGIN_Y) {
              continue;
            }
            heapPush(i, px);
          }
        }
      }
      // advance the cursor; past the box end the generation lands
      if (++sweepX > sweepHiX) {
        sweepX = sweepLoX;
        if (++sweepY > sweepHiY) {
          sweepY = sweepLoY;
          if (++sweepZ > sweepHiZ) {
            farNode.set(heapNode.subarray(0, heapK));
            farCount = heapK;
            sweepActive = false;
            sweepGen++;
            return;
          }
        }
      }
    }
  }

  function gatherCandidates(): number {
    nearEpoch++;
    const fovPx = fovPxOf(H, liveFov);
    const live = !fieldSettledNow; // spins/spread in flight: re-transform every resident
    const tanHalf = Math.tan((liveFov * Math.PI) / 360);
    const tV = tanHalf * NDC_MARGIN_Y;
    const tH = tanHalf * (W / H) * NDC_MARGIN_X;
    const slack = Math.sqrt(1 + tV * tV + tH * tH); // Euclidean ≤ depth × slack inside the margin
    const dCard = ladder.config.cardSpacings * spacing * field.spread;
    // the NEAR pass reaches the card distance only — the far sweep owns the rest (the recipe)
    const reach = dCard * slack + gridDrift;
    const r2 = reach * reach;
    const cx = camera.position.x;
    const cy = camera.position.y;
    const cz = camera.position.z;
    const m = clipM.elements;
    const pinnedNode = field.select;
    const g = grid;
    const loX = Math.max(0, Math.floor((cx - reach - g.minX) / g.cellX));
    const hiX = Math.min(g.side - 1, Math.floor((cx + reach - g.minX) / g.cellX));
    const loY = Math.max(0, Math.floor((cy - reach - g.minY) / g.cellY));
    const hiY = Math.min(g.side - 1, Math.floor((cy + reach - g.minY) / g.cellY));
    const loZ = Math.max(0, Math.floor((cz - reach - g.minZ) / g.cellZ));
    const hiZ = Math.min(g.side - 1, Math.floor((cz + reach - g.minZ) / g.cellZ));
    let k = 0;
    let sawPinned = false;
    for (let iz = loZ; iz <= hiZ; iz++) {
      const z0 = g.minZ + iz * g.cellZ;
      const dz0 = cz < z0 ? z0 - cz : cz > z0 + g.cellZ ? cz - (z0 + g.cellZ) : 0;
      for (let iy = loY; iy <= hiY; iy++) {
        const y0 = g.minY + iy * g.cellY;
        const dy0 = cy < y0 ? y0 - cy : cy > y0 + g.cellY ? cy - (y0 + g.cellY) : 0;
        if (dy0 * dy0 + dz0 * dz0 > r2) continue;
        for (let ix = loX; ix <= hiX; ix++) {
          const x0 = g.minX + ix * g.cellX;
          const dx0 = cx < x0 ? x0 - cx : cx > x0 + g.cellX ? cx - (x0 + g.cellX) : 0;
          if (dx0 * dx0 + dy0 * dy0 + dz0 * dz0 > r2) continue; // cell box outside the sphere
          const cell = (iz * g.side + iy) * g.side + ix;
          const lo = g.start[cell]!;
          const hi = g.start[cell + 1]!;
          for (let q = lo; q < hi; q++) {
            const i = g.order[q]!;
            const o = i * 3;
            if (live) refreshViewOf(i); // follow the spin BEFORE gating — and the overlay
            //                             and click target read this same refreshed slot
            const wx = viewBuf[o]!;
            const wy = viewBuf[o + 1]!;
            const wz = viewBuf[o + 2]!;
            const isPin = i === pinnedNode;
            // view-forward depth: clip w ≡ −z_view for a standard perspective matrix —
            // the same divisor the node shader sizes by (one pixel unit, §5)
            const cw = m[3]! * wx + m[7]! * wy + m[11]! * wz + m[15]!;
            if (cw <= 0 && !isPin) continue; // behind the nose
            let dist = cw;
            if (cw <= 0) {
              // a pinned node behind the camera: any positive stand-in — the pin path
              // bypasses every distance gate, this only keeps the column sane
              const dx = wx - cx;
              const dy = wy - cy;
              const dz = wz - cz;
              dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            }
            const ppx = projectedPx(field.sizeOf(i) * NODE_SCALE * field.boostOf(i), dist, fovPx);
            if (ppx < ladder.config.namedPx && !isPin) continue; // a mote and nothing more
            let cur = Infinity;
            if (cw > 0) {
              const ndx = (m[0]! * wx + m[4]! * wy + m[8]! * wz + m[12]!) / cw;
              const ndy = (m[1]! * wx + m[5]! * wy + m[9]! * wz + m[13]!) / cw;
              if (
                (ndx < -NDC_MARGIN_X || ndx > NDC_MARGIN_X ||
                  ndy < -NDC_MARGIN_Y || ndy > NDC_MARGIN_Y) &&
                !isPin
              ) {
                continue;
              }
              if (pointerActive) {
                const sx = (ndx * 0.5 + 0.5) * W - px;
                const sy = (-ndy * 0.5 + 0.5) * H - py;
                cur = Math.hypot(sx, sy);
              }
            }
            if (k >= candCap) growCand(k + 1);
            candNode[k] = i;
            candPx[k] = ppx;
            candDist[k] = dist;
            candCur[k] = cur;
            nearStamp[i] = nearEpoch; // the far merge below must not re-admit this node
            if (isPin) sawPinned = true;
            k++;
          }
        }
      }
    }

    /* ---- the far sweep: advance one slice, then merge the published set. Membership is
       ≤ a generation stale; EVERYTHING ELSE is fresh — each far node is re-projected here
       every frame (≤ FAR_CAP of them), so rank, cursor pull and distance are live. */
    farSlice(fovPx, slack, dCard);
    for (let f = 0; f < farCount; f++) {
      const i = farNode[f]!;
      if (nearStamp[i] === nearEpoch || i === pinnedNode) continue; // pin appended below
      if (live) refreshViewOf(i);
      const o = i * 3;
      const wx = viewBuf[o]!;
      const wy = viewBuf[o + 1]!;
      const wz = viewBuf[o + 2]!;
      const cw = m[3]! * wx + m[7]! * wy + m[11]! * wz + m[15]!;
      if (cw <= 0) continue; // drifted behind the nose since admission
      const ppx = projectedPx(field.sizeOf(i) * NODE_SCALE * field.boostOf(i), cw, fovPx);
      if (ppx < ladder.config.namedPx) continue; // fell under the gate — re-earns next sweep
      const ndx = (m[0]! * wx + m[4]! * wy + m[8]! * wz + m[12]!) / cw;
      const ndy = (m[1]! * wx + m[5]! * wy + m[9]! * wz + m[13]!) / cw;
      if (ndx < -NDC_MARGIN_X || ndx > NDC_MARGIN_X || ndy < -NDC_MARGIN_Y || ndy > NDC_MARGIN_Y) {
        continue;
      }
      let cur = Infinity;
      if (pointerActive) {
        const sx = (ndx * 0.5 + 0.5) * W - px;
        const sy = (-ndy * 0.5 + 0.5) * H - py;
        cur = Math.hypot(sx, sy);
      }
      if (k >= candCap) growCand(k + 1);
      candNode[k] = i;
      candPx[k] = ppx;
      candDist[k] = cw;
      candCur[k] = cur;
      nearStamp[i] = nearEpoch;
      k++;
    }

    if (pinnedNode >= 0 && !sawPinned) {
      // a pin outranks the walk: report its true depth so its card tier is honest
      if (live) refreshViewOf(pinnedNode);
      const o = pinnedNode * 3;
      const wx = viewBuf[o]!;
      const wy = viewBuf[o + 1]!;
      const wz = viewBuf[o + 2]!;
      const cw = m[3]! * wx + m[7]! * wy + m[11]! * wz + m[15]!;
      const dx = wx - cx;
      const dy = wy - cy;
      const dz = wz - cz;
      if (k >= candCap) growCand(k + 1);
      candNode[k] = pinnedNode;
      candPx[k] = 0; // the ladder's pin path bypasses the px gate anyway
      candDist[k] = cw > 0 ? cw : Math.sqrt(dx * dx + dy * dy + dz * dz);
      candCur[k] = Infinity;
      k++;
    }
    return k;
  }

  // the overlay's per-frame view: stable references, fields refreshed in the loop
  const overlayView = {
    matrix: clipM.elements,
    positions: viewBuf,
    width: window.innerWidth,
    height: window.innerHeight,
    pinned: -1,
  };
  let lastCand = 0;

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
      // widened while spinning: the grid's positions lag the drawn ones by at most this
      // (the port's gridDrift trade — a wider walk beats a click that does nothing)
      drift: gridDrift,
    });
  }

  function hover(i: number): void {
    if (field.setHover(i)) updateOverlay();
  }

  canvas.addEventListener('pointermove', (ev: PointerEvent): void => {
    px = ev.clientX;
    py = ev.clientY;
    pickDue = true;
    pointerActive = true; // attention follows the cursor (ladder scoring)
    cursorMoved = true; // and a real move is what clears a retreat()'s suspension
  });
  canvas.addEventListener('pointerleave', (): void => {
    pickDue = false;
    pointerActive = false;
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
      if (morphing) return; // base positions are mid-retarget; a wrong select is worse than none
      const i = castAt(ev.clientX, ev.clientY); // THIS event's coordinate, never the last move's
      field.setSelect(i);
      if (i >= 0) {
        // The node becomes the orbit pivot — the camera stays put (keepRadius false), so a
        // click re-aims what dragging turns around without jumping the view. viewBuf holds
        // where the node is DRAWN — refreshed through the live frame chain while spinning.
        if (!fieldSettledNow) refreshViewOf(i);
        cam.setTarget(new Vector3(viewBuf[i * 3]!, viewBuf[i * 3 + 1]!, viewBuf[i * 3 + 2]!), false);
      }
      updateOverlay();
    },
    true,
  );

  window.addEventListener('keydown', (ev: KeyboardEvent): void => {
    if (ev.key === 'Escape') {
      // Escape = close + retreat + unpin (the port's deselect-from-close-up, main.js
      // ~189): close the open slot, suspend auto-open, clear the selection (the pin is
      // the select state) — and fly the CAMERA back past the card threshold so the open
      // page shrinks into the point it came from. Without the flight the locked card
      // keeps filling the screen, and the next cursor move (which clears the suspension
      // — that is user intent) would reopen the very page Escape just closed.
      const was = ladder.state.open >= 0 ? ladder.state.open : field.select;
      ladder.retreat();
      if (field.setSelect(-1)) updateOverlay();
      if (fly && was >= 0) {
        if (!fieldSettledNow && !morphing) refreshViewOf(was);
        const o = was * 3;
        const p = new Vector3(viewBuf[o]!, viewBuf[o + 1]!, viewBuf[o + 2]!);
        const dCard = ladder.config.cardSpacings * spacing * field.spread;
        if (camera.position.distanceTo(p) < dCard * 1.05) fly.goTo(p, dCard * 1.7, 800);
      }
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

  /* =========================================================================================
   * THE ROUTING TABLE (stage 8) — the ONE place a config key meets its subsystem.
   *
   * The panel knows no key's meaning (it renders schema rows); the store knows no subsystem
   * (it validates and persists); every key's effect is decided HERE, once. Three kinds:
   *
   *   LIVE — takes effect the tick you drag. Routes: a field setter (blend, tonemap,
   *     exposure, link budget, colour column, spread), a channel-buffer rewrite through the
   *     public objects (node sizes, facet/uniform colours — the iSize/iColor arrays are the
   *     field's own buffers, so picker and ladder follow for free), a ladder.config write
   *     (thresholds and the label band are read fresh every update), a ladder REBUILD for
   *     the four keys that size fixed buffers or freeze the overlay rect (labelCap,
   *     maxCards, lockW, lockH — an event, never per frame), a flycam-config write (read
   *     live every frame by design), the pivot-table rate write (levelSpin), the scene
   *     ground, or the renderer's pixel ratio (renderScale).
   *
   *   BOOT — read once at createField(): the five focus alphas (linkBase, linkFocus,
   *     linkHover, linkDim, dimOthers) are baked into the focus state. The knob validates,
   *     persists and exports like any other; a change lands on the next reload.
   *
   *   NONE (yet) — the schema ships the knob (§7) but this library version bakes the
   *     constant into the TSL material at build time (field/nodes.ts, links.ts): nodeMinPx,
   *     nodeMaxPx, falloff, bodyR, haloPx, haloGain, coreHot, lenSoft, lenFloor.
   *     The §5 shader-hook tier is the seam they will route through; listed so the gap is
   *     explicit, not silent (§10.4's spirit). flatPx graduated to LIVE at stage 9
   *     (field.setFlatPx — the boot quality notch's LOD lever).
   * ======================================================================================= */

  const linkTotal = field.stats().links.total;
  const colorFacetKey = Object.keys(graph.facets)[0]!; // the field's own default facet
  /* The uniform hue and the light ground are LIBRARY data now (theme/palettes UNIFORM_RGB,
     field/constants BG_LIGHT): the shipped 'blueprint' and 'paper' presets lean on them, so
     a host inventing its own values made the same named look render differently per app. */

  /* -- nodes: size channel × contrast curve × scale, folded into the iSize buffer. The
     shader's radius is iSize × NODE_SCALE (baked), so writing channel × (nodeScale /
     NODE_SCALE) draws channel × nodeScale exactly — and field.sizeOf (same array) keeps the
     picker, the candidate walk and the ladder on the drawn disc. The channel buffer is
     cached so dragging `nodeScale` rescales without re-ranking the column. */
  const sizeBuf = sizeAttr.array as Float32Array;
  let sizeChannelKey = '';
  let sizeCurveVal = NaN;
  let baseSizes: Float32Array = new Float32Array(0);
  const applySizes = (): void => {
    const channel = str('sizeChannel');
    const curve = num('sizeCurve');
    if (channel !== sizeChannelKey || curve !== sizeCurveVal) {
      baseSizes =
        channel === 'uniform'
          ? uniformSizes(n)
          : sizesFromColumn(graph.columns['degree']!.data, n, curve);
      sizeChannelKey = channel;
      sizeCurveVal = curve;
    }
    const k = num('nodeScale') / NODE_SCALE;
    maxSizeWorld = 0;
    for (let i = 0; i < n; i++) {
      const s = baseSizes[i]! * k;
      sizeBuf[i] = s;
      if (s > maxSizeWorld) maxSizeWorld = s;
    }
    sizeAttr.needsUpdate = true;
    // the far sweep prunes cells by their size maximum — the bound follows the channel
    // (an event, never per frame; same pinned-slab guard as rebuildGrid)
    gridCellMax = cellMaxOf(
      grid,
      (i) => field.sizeOf(i),
      gridCellMax === sweepCellMax ? undefined : gridCellMax,
    );
  };

  /* -- links take their endpoints' colours (field.ts rule) — after a demo-side node repaint
     the RGB lanes are rewritten the same way. RGB ONLY: every alpha slot belongs to the
     focus state, and a sweep is ONE whole-buffer range (never a bare version bump — the
     field's own upload discipline). */
  const rewriteLinkRGB = (): void => {
    const colors = colorAttr.array as Float32Array;
    const lc = linkColorAttr.array as Float32Array;
    const ls = graph.links;
    const m = ls.length >> 1;
    for (let e = 0; e < m; e++) {
      const a = ls[e * 2]! * 3;
      const b = ls[e * 2 + 1]! * 3;
      const o = e * 8;
      lc[o] = colors[a]!;
      lc[o + 1] = colors[a + 1]!;
      lc[o + 2] = colors[a + 2]!;
      lc[o + 4] = colors[b]!;
      lc[o + 5] = colors[b + 1]!;
      lc[o + 6] = colors[b + 2]!;
    }
    linkColorAttr.clearUpdateRanges();
    linkColorAttr.addUpdateRange(0, lc.length);
    linkColorAttr.needsUpdate = true;
  };

  /* -- colour: three modes, one writer. 'column' goes through the library (setColorColumn
     rank-normalises degree through the ramp and recolours the links itself); 'facet' binds
     the demo's facet through the chosen palette (golden = the generator, names = theme
     data); 'uniform' is one flat hue. */
  const applyColors = (): void => {
    const mode = str('colorMode');
    if (mode === 'column') {
      field.setColorColumn('degree', str('ramp') as RampName);
      return;
    }
    const colors = colorAttr.array as Float32Array;
    if (mode === 'uniform') {
      for (let i = 0; i < n; i++) {
        colors[i * 3] = UNIFORM_RGB[0];
        colors[i * 3 + 1] = UNIFORM_RGB[1];
        colors[i * 3 + 2] = UNIFORM_RGB[2];
      }
    } else {
      const facet = graph.facets[colorFacetKey]!;
      const name = str('palette');
      const pal =
        name === 'golden'
          ? facetPalette(facet.values.length)
          : paletteBuffer(PALETTES[name as PaletteName], facet.values.length);
      paintFacetColors(facet.data, n, pal, colors);
    }
    colorAttr.needsUpdate = true;
    rewriteLinkRGB();
  };

  /* -- the ground. The scene owns the background (the field only ships its default), and the
     chrome follows the ground — a light field under dark-ground labels is not a theme, it is
     unreadable (the predecessor's paintGround). Blending is its OWN key: the §5 rule
     "blending follows the ground" is the presets' job, enforced by the theme tests. */
  const applyGround = (): void => {
    const light = str('background') === 'light';
    scene.background = new Color(light ? BG_LIGHT : field.background);
    document.body.classList.toggle('light', light);
  };

  /* -- the ladder's live keys: thresholds and the label band are read fresh every
     ladder.update, so mutating the resolved config routes them with no rebuild. The keep
     band re-applies createLadder's clamp (keep ≥ earn — hysteresis cannot be negative). */
  const applyLadderCfg = (): void => {
    const c = ladder.config;
    c.namedPx = num('namedPx');
    c.earn = num('labelEarn');
    c.keep = Math.max(num('labelKeep'), c.earn);
    c.dwellMs = num('labelDwell');
    c.cardSpacings = num('tierCard');
    c.lockSpacings = num('tierLock');
    c.openSpacings = num('tierOpen');
    c.cardRadiusSpacings = num('cardRadius');
  };

  /* -- flight: the flycam reads flyCfg live every frame; the orbit cam has its own defaults
     and no knobs here (?cam=orbit is a diagnostics mode). */
  const applyFlight = (): void => {
    flyCfg.speedK = num('speedK');
    flyCfg.mouseSens = num('mouseSens');
    flyCfg.turnRate = num('turnRate');
    flyCfg.easeLin = num('easeLin');
    flyCfg.spinRate = num('driftRate');
    if (fly) fly.spin = bool('cameraDrift');
  };

  const applyBlend = (): void => {
    field.setBlend(str('blend') === 'additive' ? 'additive' : 'normal');
  };
  const applyTonemap = (): void => {
    field.setTonemap(str('tonemap') as TonemapOp, num('exposure'));
  };
  const applyBudget = (): void => {
    field.setLinkBudget(Math.round(num('linkBudget') * linkTotal));
  };
  const applySpread = (): void => {
    field.setSpread(num('spread'), 0.15); // a short tween — lands EXACTLY on the slider value
  };
  const applyRenderScale = (): void => {
    renderer.setPixelRatio(window.devicePixelRatio * num('renderScale'));
    renderer.setSize(W, H);
  };
  const applyFlatPx = (): void => {
    field.setFlatPx(num('flatPx'));
  };

  /** key → route. One entry per LIVE key; the boot and none lists close the enumeration so
   *  every schema key is accounted for here, by name. */
  const ROUTE: Record<string, () => void> = {
    // ladder (live config writes)
    tierCard: applyLadderCfg,
    tierLock: applyLadderCfg,
    tierOpen: applyLadderCfg,
    cardRadius: applyLadderCfg,
    // ladder (rebuild: fixed buffers / frozen overlay rect)
    maxCards: rebuildLadder,
    lockW: rebuildLadder,
    lockH: rebuildLadder,
    labelCap: rebuildLadder,
    // the label band
    namedPx: applyLadderCfg,
    labelEarn: applyLadderCfg,
    labelKeep: applyLadderCfg,
    labelDwell: applyLadderCfg,
    // nodes
    sizeChannel: applySizes,
    nodeScale: applySizes,
    sizeCurve: applySizes,
    // colour and ground
    colorMode: applyColors,
    palette: applyColors,
    ramp: applyColors,
    background: applyGround,
    blend: applyBlend,
    tonemap: applyTonemap,
    exposure: applyTonemap,
    // links
    linkBudget: applyBudget,
    // motion
    spread: applySpread,
    cameraDrift: applyFlight,
    driftRate: applyFlight,
    levelSpin: applySpins,
    // flight
    speedK: applyFlight,
    mouseSens: applyFlight,
    turnRate: applyFlight,
    easeLin: applyFlight,
    // perf
    renderScale: applyRenderScale,
    flatPx: applyFlatPx,
  };
  /** Read once at createField (focus state) — persisted, applied on the next reload. */
  const BOOT_KEYS = ['linkBase', 'linkFocus', 'linkHover', 'linkDim', 'dimOthers'] as const;
  /** Baked into the TSL materials — no seam in this library version (see the header).
   *  flatPx left this list at stage 9: it is a live uniform now (field.setFlatPx), the boot
   *  quality notch's LOD lever. */
  const UNROUTED_KEYS = [
    'nodeMinPx',
    'nodeMaxPx',
    'falloff',
    'bodyR',
    'haloPx',
    'haloGain',
    'coreHot',
    'lenSoft',
    'lenFloor',
  ] as const;

  const routeKey = (key: string): void => {
    const route = ROUTE[key];
    if (route) route();
    updateOverlay();
  };
  /** The '*' path: a preset, reset or import may have moved every key — re-apply each route
   *  exactly once (the table maps many keys onto few appliers, so dedupe by function). */
  const applyAll = (): void => {
    rebuildLadder();
    const seen = new Set<() => void>([rebuildLadder]);
    for (const key of Object.keys(ROUTE)) {
      const route = ROUTE[key]!;
      if (seen.has(route)) continue;
      seen.add(route);
      route();
    }
    updateOverlay();
  };

  /* ---- the tuning panel (stage 8): generated from the schema, live on drag, ~ toggles.
     onChange routes ONE key; onBulk (preset pick / reset / paste) re-applies the table. */
  injectPanelStyles();
  const panel = createPanel({
    store,
    onChange: routeKey,
    onBulk: (): void => applyAll(),
    inertKeys: UNROUTED_KEYS, // greyed in the panel: a live-looking dead knob reads as a bug
  });
  document.body.appendChild(panel.el);
  document.getElementById('tune')?.addEventListener('click', () => panel.toggle());
  // the boot applyAll() runs below, after the stats state it reports into exists

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
  let bootMs = 0; // navigationStart → first rendered frame (§9 boot; marked in the loop)
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

  /* jsFrameMs by SEGMENT (measurement rig, not shipped behaviour): rolling 90-frame
   * medians per phase of the update window, so "what costs the frame" is read off the
   * handle instead of guessed. Four extra performance.now() calls per frame — noise
   * next to the 0.1 ms clock resolution they share with jsFrameMs itself. */
  const SEGS = ['cam', 'field', 'grid', 'pick', 'ladder'] as const;
  const segBuf = new Float64Array(SEGS.length * 90);
  let segFilled = 0;
  let segCursor = 0;
  const segMark = new Float64Array(SEGS.length + 1);
  const jsBreakdown = (): Record<string, number> => {
    const out: Record<string, number> = {};
    const scratch = new Float64Array(90);
    for (let s = 0; s < SEGS.length; s++) {
      for (let k = 0; k < segFilled; k++) scratch[k] = segBuf[k * SEGS.length + s]!;
      out[SEGS[s]!] = medianOf(scratch, segFilled);
    }
    return out;
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
      `${s.projection} · pivots ${fmt(s.pivots)}` +
      // The two boot-notch levers that drop RESOLUTION/DETAIL rather than content. §10.4's
      // letter doesn't require a stats row (nothing is culled), but §10.7 does require the
      // telemetry to show hidden states — a reader must be able to tell the field is
      // rendering at half resolution without digging for __starscape.quality.
      ` · render dpr ${renderer.getPixelRatio()} (scale ${num('renderScale')}) · ` +
      `flat < ${num('flatPx')} px${settledNow ? '' : ' · animating'}\n` +
      `room ${Number.isFinite(room) ? room.toFixed(0) : '∞'} · speed ${speed.toFixed(0)}` +
      `${blocked ? ' (damped)' : ''} · ` +
      `hover: ${hov >= 0 ? graph.labelOf(hov) : '—'} · ` +
      `selected: ${sel >= 0 ? graph.labelOf(sel) : '—'}\n` +
      // the ladder's budgets, reported not implied (§10.4) — named/cards/open counts, and
      // the far sweep's generation state (a budget that lags content must say so too)
      `ladder: cand ${lastCand} · far ${farCount}/${heapLimit}` +
      `${sweepActive ? ` (sweep ${Math.round((100 * sweepDone) / Math.max(sweepCells, 1))}%)` : ''} · ` +
      `named ${ladder.state.namedCount}/${ladder.config.namedCap} · ` +
      `cards ${ladder.state.cards.length}/${ladder.config.cards} · ` +
      `open ${ladder.state.open >= 0 ? graph.labelOf(ladder.state.open) : '—'}` +
      `${ladder.state.suspended ? ' · suspended' : ''}` +
      `${field.select >= 0 ? ' · pinned' : ''}` +
      ` · look ${store.preset}`;
  }

  applyAll(); // boot: the loaded config (defaults ⊕ persisted diff) reaches every subsystem

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
    segMark[0] = c0;
    const telem = cam.update(dt);
    room = telem.room;
    speed = telem.speed;
    blocked = telem.blocked;
    segMark[1] = performance.now();

    demoTweens.update(dt);
    field.update(dt); // tweens + frames: O(pivots), zero when idle (§4)
    segMark[2] = performance.now();

    fieldSettledNow = field.settled();
    settledNow = fieldSettledNow && !demoTweens.active() && !(fly?.tweening ?? false);
    if (settledNow && !wasSettled) {
      rebuildGrid(); // on settle — never per frame (DESIGN §6)
      updateOverlay();
    }
    wasSettled = settledNow;
    if (morphing && field.morphT >= 1) {
      // the retarget landed: the base endpoints are parked — refresh the CPU mirror, and
      // rebuild the index NOW if live spins will keep the settle edge from ever firing
      morphing = false;
      field.basePositions(baseBuf);
      resetSweep(); // every position moved — a stale far set would name the wrong places
      if (!fieldSettledNow) {
        rebuildGrid();
        gridCooldown = 0.25;
      }
    }
    /* Grid upkeep under spin (the port's ensureGrid, moved on-loop): rotation walks the
       drawn positions away from the snapshot the grid indexes, so the error is TRACKED —
       candidate reach and pickRay widen by gridDrift — and the index rebuilt lazily once
       the drift passes a full cell, rate-limited to once a second so the O(n) pass stays
       off the steady frame clock. Stage 9 widened both (from cell/3 every 250 ms): the
       rebuild measures ~68 ms at 10⁶, and at 4 Hz that was a visible hitch train — the
       walks' drift-widening keeps every consumer CORRECT at any staleness, so the only
       price of the longer leash is a few hundred extra near-cells visited per frame. */
    if (!morphing && (field.frames?.activeThisTick() ?? false)) {
      const dd = spinDriftRate * dt * field.spread;
      driftTotal += dd;
      gridDrift += dd;
    }
    gridCooldown = Math.max(0, gridCooldown - dt);
    if (!morphing && gridCooldown === 0 && gridDrift > grid.cellMax) {
      rebuildGrid();
      gridCooldown = 1.0;
    }
    if (!fieldSettledNow && !morphing) {
      // sync the spread mirror refreshViewOf composes with (one read per frame, not per node)
      const c = field.centroid();
      spreadC[0] = c[0];
      spreadC[1] = c[1];
      spreadC[2] = c[2];
      spreadK = field.spread;
    }
    segMark[3] = performance.now();

    // Hover resolves against the latest cursor position, once per frame at most. A morph
    // suppresses it (positions and index genuinely disagree), and so does a camera tween
    // (the ray would sweep nodes nobody is pointing at); a SPINNING field picks fine —
    // the ladder's nodes are written back fresh and the walk widens by the drift.
    if (pickDue && !morphing && !(fly?.tweening ?? false)) {
      pickDue = false;
      hover(castAt(px, py));
    }
    segMark[4] = performance.now();

    /* ---- the ladder tick (stage 7). Candidates are gathered every frame the BASE is
       parked — including during spins and spread tweens, where gatherCandidates re-forms
       each resident through the live frame chain (refreshViewOf): the ladder must live
       exactly when you fly toward a rotating page, and §9's 10⁶ acceptance row is "all
       spins on". Only a MORPH skips the pass (mid-retarget the positions and the index
       disagree) — the dwell hysteresis absorbs that 0.9 s gap. The clip matrix is rebuilt
       here, not read from the renderer — the camera moved this frame. */
    camera.updateMatrixWorld();
    camInv.copy(camera.matrixWorld).invert();
    clipM.multiplyMatrices(camera.projectionMatrix, camInv);
    ladderView.pinned = field.select; // the pin IS the select state
    ladderView.cursorMoved = cursorMoved;
    ladderView.spacing = spacing * field.spread; // the tier thresholds are spacing-denominated
    //   and candidate depths live in SPREAD world units — the unit must follow the slider
    //   (the port recomputed sp = spacing × spreadK on every config change)
    cursorMoved = false;
    lastCand = morphing ? 0 : gatherCandidates();
    ladderCandidates.count = lastCand;
    const ladderState = ladder.update(dt, ladderView);
    if (!fieldSettledNow && !morphing) {
      // glue every mounted label/card to the spot the shader draws THIS frame — including
      // ones riding out their drop grace, which the candidate walk no longer visits
      for (let j = 0; j < ladderState.namedCount; j++) refreshViewOf(ladderState.named[j]!);
      for (const slot of ladderState.cards) refreshViewOf(slot.node);
    }
    overlayView.width = W;
    overlayView.height = H;
    overlayView.pinned = field.select;
    ladderOverlay.update(ladderState, overlayView);

    const c1 = performance.now(); // jsFrameMs window closes BEFORE render
    jsMs[jsCursor] = c1 - c0;
    jsCursor = (jsCursor + 1) % jsMs.length;
    if (jsFilled < jsMs.length) jsFilled++;
    segMark[5] = c1;
    for (let s = 0; s < SEGS.length; s++) {
      segBuf[segCursor * SEGS.length + s] = segMark[s + 1]! - segMark[s]!;
    }
    segCursor = (segCursor + 1) % 90;
    if (segFilled < 90) segFilled++;

    renderer.render(scene, camera);
    drawCalls = renderer.info.render.drawCalls;
    if (frame === 0) {
      /* §9 boot = navigationStart → first frame. performance.now() is relative to
       * timeOrigin (navigationStart), so this one read IS the boot time; the mark makes
       * it visible in the Performance panel too. First render submitted = first frame
       * (SCALE.md methodology: never time renderer.render itself — this is a timestamp,
       * not a duration around the call). */
      bootMs = performance.now();
      performance.mark('starscape:first-frame');
    }
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
    /* ---- the SCALE.md measurement rig (§9: assert the viewpoint, median of 90 rAF
       intervals, never renderer.render). All three are demo-side instrumentation, not
       library API. */
    bootMs: (): number => bootMs,
    jsBreakdown,
    /** Far-sweep + grid-upkeep telemetry (§10.7: hidden costs shown, not implied). */
    sweep: (): Record<string, number | boolean> => ({
      gen: sweepGen,
      active: sweepActive,
      done: sweepDone,
      cells: sweepCells,
      far: farCount,
      lastRebuildMs,
      gridDrift,
    }),
    quality,
    /** Camera distance to the live TARGET extent centre over the home-framing distance
     *  (flyHome's own formula) — must read ~1.0 before any §9 number is recorded
     *  (SCALE.md: every wrong number was a moved viewpoint). */
    homeRatio: (): number => {
      const e = field.targetExtent();
      const d = Math.hypot(
        camera.position.x - e.centre[0],
        camera.position.y - e.centre[1],
        camera.position.z - e.centre[2],
      );
      const home =
        (Math.max(e.radius, radius * 0.05) / Math.tan((liveFov * Math.PI) / 360)) * HOME_MARGIN;
      return d / home;
    },
    /** A FRESH median-of-`frames` rAF-interval window (the rolling overlay buffer keeps
     *  old state; a measurement wants only the state being measured). Resolves with the
     *  medians plus the context that makes the number honest: spins, dpr, budgets. */
    measure: (
      frames = 90,
    ): Promise<{
      msMedian: number;
      msP90: number;
      msWorst: number;
      fps: number;
      jsMedian: number;
      drawCalls: number;
      dpr: number;
      homeRatio: number;
      spinning: boolean;
      linksDrawn: number;
      linksTotal: number;
    }> =>
      new Promise((resolve) => {
        const buf = new Float64Array(frames);
        let k = -1; // first callback only primes prev
        let prev = 0;
        const tick = (t: number): void => {
          if (k >= 0) buf[k] = t - prev;
          prev = t;
          k++;
          if (k < frames) {
            requestAnimationFrame(tick);
            return;
          }
          const sorted = Array.from(buf).sort((a, b) => a - b);
          const ms = sorted[frames >> 1]!;
          const p90 = sorted[Math.min(frames - 1, Math.floor(frames * 0.9))]!;
          const worst = sorted[frames - 1]!;
          const s = field.stats();
          const handle = (window as unknown as Record<string, Record<string, () => number>>)[
            '__starscape'
          ]!;
          resolve({
            msMedian: ms,
            msP90: p90, // stalls (grid rebuilds under spin) live here, not in the median
            msWorst: worst,
            fps: ms > 0 ? 1000 / ms : 0,
            jsMedian: jsFrameMs(), // the rolling buffer — full of steady state by now
            drawCalls,
            dpr: renderer.getPixelRatio(),
            homeRatio: handle['homeRatio']!(),
            spinning: field.frames?.activeThisTick() ?? false,
            linksDrawn: s.links.drawn,
            linksTotal: s.links.total,
          });
        };
        requestAnimationFrame(tick);
      }),
    setArrangement: (name: string): void => {
      ui.arrangement.value = name;
      applyArrangement();
    },
    setProjection: (mode: '2d' | '3d'): void => {
      ui.proj.value = mode;
      applyProjection();
    },
    setSpread: (v: number): void => {
      // spread is a config key now — the panel, the store and the field stay in agreement
      if (store.set('spread', v)) {
        panel.markCustom();
        store.save();
        routeKey('spread');
        panel.sync();
      }
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
    // stage 7: the ladder, the overlay and a tier-walking helper. flyToNode(i, s) stands
    // off s page-spacings from node i's DRAWN position — s 2.4 approaches past the card
    // threshold (2.2), 1.5 grows the box, 1.0 locks it, 0.4 opens the body (0.45), all the
    // ladder's own units (spread included, like the thresholds), so a headless check can
    // walk the tiers deterministically.
    // live getters — the stage-8 rebuild keys (labelCap/maxCards/lockW/lockH) swap both
    get ladder() {
      return ladder;
    },
    get overlay() {
      return ladderOverlay;
    },
    provider,
    /* stage 8: the config store, the panel, the routing table's '*' path, and a setter that
       walks the full user path (validate → touch → persist → route → panel sync) so a
       headless check drives config exactly as a slider would. */
    store,
    panel,
    configWarnings,
    applyAll,
    // the routing table's full enumeration, inspectable: live keys route, boot keys land on
    // the next reload (createField focus opts), unrouted keys await the §5 shader-hook seam
    routes: {
      live: Object.keys(ROUTE),
      boot: BOOT_KEYS,
      unrouted: UNROUTED_KEYS,
    },
    setConfig: (key: string, value: ConfigValue): boolean => {
      if (!store.set(key, value)) return false;
      panel.markCustom();
      store.save();
      routeKey(key);
      panel.sync();
      return true;
    },
    applyLook: (name: string): boolean => {
      if (!store.applyLook(name)) return false;
      store.save();
      panel.sync();
      applyAll();
      return true;
    },
    ladderState: (): string =>
      JSON.stringify({
        cand: lastCand,
        named: ladder.state.namedCount,
        cards: ladder.state.cards.map((c) => ({ node: c.node, tier: c.tier, growT: c.growT })),
        open: ladder.state.open,
        suspended: ladder.state.suspended,
        pinned: field.select,
      }),
    flyToNode: (i: number, spacings = 0.4, ms = 900): void => {
      if (!fly) return;
      cursorMoved = true; // a deliberate goto is user intent — it clears a retreat's
      //                     suspension, exactly as the port's flyTo re-enabled auto-open
      if (!fieldSettledNow && !morphing) refreshViewOf(i);
      const t = new Vector3(viewBuf[i * 3]!, viewBuf[i * 3 + 1]!, viewBuf[i * 3 + 2]!);
      fly.goTo(t, Math.max(spacings * spacing * field.spread, 0.01), ms);
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
