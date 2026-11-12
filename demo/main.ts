/* The starscape PLAYGROUND (DESIGN.md §8) — the library consumed exactly as an external user
 * would, on a FULLY SYNTHETIC corpus generated client-side from (n, seed) alone: no external
 * dataset, no fetch, nothing but the library and three. It is the public GitHub-Pages demo and
 * the dev demo in one artifact (vite.config.ts: `root: 'demo'`).
 *
 * It exposes the library's FULL control surface:
 *   - a DATA/size selector (1k / 10k / 100k / 1M synthetic nodes — the universe scales with n so
 *     density stays constant), seeded by ?seed, state in the query string so it survives reloads;
 *   - a per-level HIERARCHY editor (the custom arrange mode): per level a clustering facet, a
 *     shape/placement-preset, ordering, density + scatter each with their ParamDesc sub-params,
 *     a spread (the library radiusScale) and a spin (rate + axis) — every option list enumerated
 *     at runtime from the library's own registries (shapeKeys / presetKeys / densityKeys /
 *     scatterKeys / orderingKeys) and the live graph's facets, nothing hardcoded;
 *   - colour controls (colour-by-facet with a facet picker, colour ORDER value-vs-spatial,
 *     palette + ramp), shape-by-facet, the full node-glyph + link style knobs;
 *   - a GLOW (HDR bloom) panel (strength / radius / threshold / compression), a view-distance /
 *     far-plane control, a close-orbit camera mode (double-click a node to fly up and circle it),
 *     and a card-attention centre-gate — all on top of the schema tuning panel + visual presets.
 *
 * Structure and the non-obvious rules follow the library's own stage-9 demo — grid rebuilt on
 * settle and lazily under drift, hover suppressed mid-morph, refreshViewOf composing the live
 * frame chain for exactly the nodes the ladder touches. gatherCandidates is the simple exhaustive
 * loop (every node, every frame): at 10⁶ it costs a few ms/frame, visible honestly in jsFrameMs.
 * House rules kept: boot failures surface in the UI (§10.5), `window.__starscape` is the debug
 * handle and the stats line shows hidden states (§10.7), the camera fits the LIVE extent (§10.8),
 * budgets are reported (§10.4), and nothing producing geometry reads the wall clock (§10.1). */

import {
  CanvasTexture,
  Color,
  Matrix4,
  PerspectiveCamera,
  Quaternion,
  Scene,
  Vector3,
  WebGPURenderer,
} from 'three/webgpu';
import type {
  BufferAttribute,
  InstancedBufferAttribute,
  LineSegments,
  Mesh,
  Texture,
} from 'three/webgpu';
import {
  bootQuality,
  buildGrid,
  compileHierarchy,
  createConfigStore,
  createField,
  createFlycam,
  createLadder,
  createNearMesh,
  createOrbitCam,
  createOverlay,
  createPanel,
  createPostProcess,
  createTweens,
  cubicInOut,
  densityKeys,
  densityParams,
  facetPalette,
  fovPxOf,
  getPreset,
  getShape,
  hash,
  injectPanelStyles,
  injectStyles,
  localStorageAdapter,
  generateNearTextures,
  makeSynthProvider,
  namedPaletteBuffer,
  nearestDistance,
  nearTextureAverages,
  wrapNearTextures,
  NEAR_TEXTURE_NAMES,
  neighbours,
  orderingKeys,
  paintFacetColors,
  paletteBuffer,
  pickRay,
  presetKeys,
  projectedPx,
  scatterKeys,
  scatterParams,
  shapeKeys,
  shapeParams,
  sizesFromColumn,
  uniformSizes,
  BG_BLACK,
  BG_LIGHT,
  DEFAULT_FOV,
  D_REF_FRACTION,
  FALLOFF,
  FLYCAM_DEFAULTS,
  HOME_MARGIN,
  MAX_LEVELS,
  NODE_MAX_PX,
  NODE_MAX_PX_OFF,
  NODE_MIN_PX,
  NODE_SCALE,
  ORBIT_DEFAULTS,
  PALETTES,
  PIVOT_CAPACITY,
  SCHEMA,
  UNIFORM_RGB,
  VISUAL_PRESETS,
} from '../src/index';
import type {
  CameraController,
  CompiledHierarchy,
  ConfigValue,
  Flycam,
  FlycamConfig,
  GlyphTextureMode,
  ImageEvent,
  ImageTier,
  LadderCandidates,
  LadderView,
  LevelSpec,
  NearMesh,
  NodeSource,
  PaletteName,
  ParamDesc,
  PlacementSpec,
  PostProcess,
  RampName,
  RandomPlace,
  SchemaRow,
  TonemapOp,
} from '../src/index';
import { loadSynth } from './synthdata';
import type { SynthData } from './synthdata';

const statsEl = document.getElementById('stats')!;
const errorEl = document.getElementById('error')!;

/** Candidate-pass frustum margins, NDC (the ladder names things slightly off-screen so
 *  labels do not pop at the edge) — the predecessor's ±1.25 / ±1.3. */
const NDC_MARGIN_X = 1.25;
const NDC_MARGIN_Y = 1.3;

/** The dataset schema: the shipped rows with two option lists widened from the live corpus —
 *  exactly the extension point schema.ts documents ("a panel extends this list from the live
 *  graph"). The size-channel options are the columns the ACTIVE dataset actually carries
 *  (vault: the pre-baked sizes.bin 'size' + meta3's words/rank; synth: degree/words), never a
 *  hardcoded vault list. 'vault' palette = the dataset's layer colours by name through
 *  namedPaletteBuffer (synth supplies the library's own vaultLayer stops). */
function vaultSchema(columns: readonly string[], facets: readonly string[]): SchemaRow[] {
  const sizeOptions = ['size', 'degree', 'words', 'rank'].filter((c) => columns.includes(c));
  sizeOptions.push('uniform');
  const sizeDefault = sizeOptions.includes('size') ? 'size' : 'degree';
  // colour/shape facets are WIDENED from the live graph (the schema ships only the universals):
  // colorFacet picks which facet colours the nodes in 'facet' mode; shapeFacet binds the node
  // glyph to a facet ('none' = all discs). Defaults preserve today's look (colour by 'type').
  const colorDefault = facets.includes('type') ? 'type' : (facets[0] ?? 'type');
  /* The 'colour order' control (F1): in colorMode='facet', 'value' keeps today's categorical
     palette by value index; 'spatial' colours each facet-value cluster by its centroid's position
     along a 3D Hilbert curve through the chosen ramp (library Field.setColorFacetSpatial), so
     spatially-adjacent clusters share hue. Not in the shipped library schema — it is this
     consumer's knob over the library API — so it is inserted next to 'colour facet' here. */
  const colorOrderRow: SchemaRow = {
    group: 'colour',
    key: 'colorOrder',
    label: 'colour order',
    kind: 'select',
    options: ['value', 'spatial'],
    default: 'value',
    unit: 'none',
  };
  /* CARD ATTENTION (this consumer's knob over the library ladder's `attentionGateRadiusPx`): where
     a card is ALLOWED to open. 'cursor' (default — today's behaviour) feeds the ladder the live
     cursor→node distance as cursorPx and leaves the gate at Infinity, so the cursor only RE-RANKS
     (boosts) and a card can open on any close-enough node. 'center' feeds the SCREEN-CENTRE→node
     distance as cursorPx AND sets the gate to `centerSize`, so a card opens ONLY on nodes inside
     the centre region (labels still appear off-centre; a pinned node still opens anywhere). The
     library ladder never learns which anchor is in play — the site decides by what it passes. */
  const cardAttentionRow: SchemaRow = {
    group: 'ladder',
    key: 'cardAttention',
    label: 'card attention',
    kind: 'select',
    options: ['cursor', 'center'],
    default: 'cursor',
    unit: 'none',
  };
  /* CENTER SIZE = the centre-region radius in CSS px: how much of the screen counts as "centre"
     when card attention is 'center'. Inert (greyed) in cursor mode. The canvas centre and this
     radius are both CSS px, consistent with how the cursor distance is already measured, so no
     devicePixelRatio factor enters. */
  const centerSizeRow: SchemaRow = {
    group: 'ladder',
    key: 'centerSize',
    label: 'center size',
    kind: 'range',
    min: 40,
    max: 600,
    step: 10,
    default: 160,
    unit: 'px',
  };
  const out = SCHEMA.map((row): SchemaRow => {
    if (row.key === 'sizeChannel' && row.kind === 'select') {
      return { ...row, options: sizeOptions, default: sizeDefault };
    }
    if (row.key === 'colorFacet' && row.kind === 'select') {
      return { ...row, options: facets.length ? facets : row.options, default: colorDefault };
    }
    if (row.key === 'shapeFacet' && row.kind === 'select') {
      return { ...row, options: ['none', ...facets], default: 'none' };
    }
    if (row.key === 'palette' && row.kind === 'select') {
      return { ...row, options: ['vault', ...row.options], default: 'vault' };
    }
    return row;
  });
  // insert 'colour order' right after 'colour facet' so the panel keeps the colour group contiguous
  const at = out.findIndex((r) => r.key === 'colorFacet');
  out.splice(at < 0 ? out.length : at + 1, 0, colorOrderRow);
  // append the two card-attention rows to the END of the ladder group (after its last shipped row,
  // 'lockH') so the panel keeps the ladder group contiguous and the shipped order unchanged.
  const ladderEnd = out.findIndex((r) => r.key === 'lockH');
  out.splice(ladderEnd < 0 ? out.length : ladderEnd + 1, 0, cardAttentionRow, centerSizeRow);
  return out;
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const camMode = params.get('cam') === 'orbit' ? 'orbit' : 'fly';
  const boot = (msg: string): void => {
    statsEl.textContent = msg;
  };

  /* ---- the dataset: a synthetic corpus of ?n nodes at ?seed (default 10000 @ 42). State lives
     in the query string so a chosen size survives reloads; the DATA select below writes it and
     reloads. The universe scales with n inside generateSynth, so density stays constant across
     sizes — 1k and 1M look like the same field at different zooms. */
  const synthN = Math.max(1, Math.min(2_000_000, Number(params.get('n') ?? 10000) | 0));
  const synthSeed = Number(params.get('seed') ?? 42) | 0;
  // optional image-tier LRU budget override (MB), for demonstrating cache eviction under pressure;
  // omitted ⇒ the library default (~64 MB). e.g. ?imgbudget=16 forces the high tiers to evict.
  // (guard the absent param: Number(null) === 0, which would wrongly force a zero-byte budget.)
  const imgBudgetRaw = params.get('imgbudget');
  const imgBudgetMB = imgBudgetRaw !== null ? Number(imgBudgetRaw) : NaN;

  const t0 = performance.now();
  // generateSynth(10⁶) is synchronous and takes seconds: paint the message FIRST (rAF runs
  // before paint, the macrotask after it — one yielded frame between message and work).
  boot(`generating ${synthN.toLocaleString('en-US')} synthetic nodes…`);
  await new Promise<void>((resolve) => {
    requestAnimationFrame(() => setTimeout(resolve, 0));
  });
  const V: SynthData = loadSynth(synthN, synthSeed, boot);
  const graph = V.graph;
  const n = graph.n;
  const tGraph = performance.now();

  /* ---- config store, loaded before the field exists (boot-routed keys). Keys are
     app-versioned so this viewer and the library demo never trade localStorage blobs — and
     mode-scoped, so a size channel tuned for the vault ('size') can never leak into a synth
     session whose graph has no such column, or vice versa. */
  const schema = vaultSchema(Object.keys(graph.columns), Object.keys(graph.facets));
  const store = createConfigStore({
    schema,
    storage: localStorageAdapter(),
    presets: VISUAL_PRESETS,
    configKey: 'starscape-playground:cfg:v1',
    looksKey: 'starscape-playground:looks:v1',
  });
  const configWarnings = store.load();
  const quality = bootQuality(n, window.devicePixelRatio || 1);
  store.set('linkBudget', quality.linkFraction);
  store.set('renderScale', quality.renderScale);
  store.set('flatPx', quality.flatPx);
  const num = (k: string): number => store.get(k) as number;
  const str = (k: string): string => store.get(k) as string;
  const bool = (k: string): boolean => store.get(k) === true;

  /* ---- SCENE PRESETS: one top-level pick sets the WHOLE scene at once — the hierarchy AND the
     look AND glow AND view — so a single choice reproduces a complete curated scene. SPACE4_LOOK
     is the config diff for the shipped 'space4' galaxy-nebula (its rows/glow/view live in
     applyScene below). A fresh visitor opens on space4 (a far better first impression than the
     neutral frozen field), so its look is pre-seeded into the store HERE — before createField —
     so the BOOT-routed focus alphas and the size channel it needs are baked at creation, not only
     after a reload. Every space4 value is valid in the synth schema (words column, system facet,
     filmic tone curve — all present in generateSynth's corpus). */
  const SCENE_KEY = 'starscape-playground:scene:v1';
  const SPACE4_LOOK: Record<string, ConfigValue> = {
    tierCard: 0.5, tierLock: 0.5, tierOpen: 0.4, maxCards: 1, lockW: 750, lockH: 580,
    namedPx: 8, labelEarn: 60, labelKeep: 90, labelCap: 90, labelDwell: 2000,
    sizeChannel: 'words', nodeScale: 4, sizeCurve: 3, nodeMinPx: 1,
    bodyR: 1, haloPx: 5.5, haloGain: 0.6, coreHot: 0.5,
    colorFacet: 'system', colorOrder: 'spatial', background: 'black', blend: 'additive',
    tonemap: 'filmic', exposure: 0.1, compression: 0.45,
    linkBase: 0.26, linkFocus: 0.84, linkDim: 0.148, dimOthers: 0.43, lenSoft: 0.17, lenFloor: 0,
  };
  const SPACE4_GLOW = { on: true, strength: 0.25, radius: 0.04, threshold: 0 };
  const SPACE4_VIEW = { distance: 20, projection: '3d' as const };
  // fresh visitor → open on space4: bake its look into the store before createField reads it.
  if (localStorage.getItem(SCENE_KEY) === null) {
    for (const [k, v] of Object.entries(SPACE4_LOOK)) store.set(k, v);
  }

  /* ---- renderer: the ONE place devicePixelRatio lives; the field reasons in CSS px. */
  boot('starting the renderer');
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

  const field = createField(graph, {
    heightCss: window.innerHeight,
    fov: DEFAULT_FOV,
    blend: str('blend') === 'additive' ? 'additive' : 'normal',
    sizeColumn: str('sizeChannel') === 'uniform' ? 'uniform' : str('sizeChannel'),
    colorFacet: 'type',
    pull: backendName === 'webgpu' ? 'storage' : 'texture',
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

  /* Per-instance channel buffers through the PUBLIC objects (field.objects = [links, nodes]):
     the iSize/iColor arrays ARE the field's own channel buffers, so rewriting them keeps the
     shader, the picker and the ladder agreeing about every node. */
  const linkLines = field.objects[0] as LineSegments;
  const nodeMesh = field.objects[1] as Mesh;
  const sizeAttr = nodeMesh.geometry.getAttribute('iSize') as InstancedBufferAttribute;
  const colorAttr = nodeMesh.geometry.getAttribute('iColor') as InstancedBufferAttribute;
  const linkColorAttr = linkLines.geometry.getAttribute('lColor') as BufferAttribute;

  const positions = graph.positions!;
  const { centre, radius, diagonal } = field.extent;
  const dRef = diagonal * D_REF_FRACTION;
  /** The ladder's unit: the layout build's own measured page spacing (meta3.spacing), the
   *  predecessor's convention — "a card appears two pages away" keeps meaning across looks.
   *  A synth corpus measures nothing (spacing 0): derive it the way the library demo does,
   *  radius/∛n over the LIVE extent. */
  const spacing = V.spacing > 0 ? V.spacing : radius / Math.cbrt(Math.max(n, 1));
  /** Arrangement spacing: fill the frozen volume (the demo's rule). */
  const arrangeSpacing = radius / Math.cbrt(Math.max(n, 1));

  /* ---- picking: one grid over the view snapshot — rebuilt on settle / morph landing /
     lazily under spin drift, never per frame. */
  const viewBuf = new Float32Array(n * 3);
  viewBuf.set(positions);
  let grid = buildGrid(viewBuf, n);
  let gridDrift = 0;
  let gridCooldown = 0;
  const rebuildGrid = (): void => {
    field.viewPositions(viewBuf);
    grid = buildGrid(viewBuf, n);
    gridDrift = 0;
  };

  /* CPU mirror of the shader's BASE endpoints; refreshViewOf composes the live frame chain +
     spread for ONE node — the vertex-shader arithmetic — so ladder/overlay/select follow a
     SPINNING field with O(active) work (§4). */
  const baseBuf = new Float32Array(n * 3);
  baseBuf.set(positions);
  let morphing = false;
  let fieldSettledNow = true;
  let spinDriftRate = 0;
  const spreadC: [number, number, number] = [0, 0, 0];
  let spreadK = 1;
  /* The field-global root-frame rotation, mirrored for the CPU view transform (the shader's
     last stage): a quaternion xyzw about spreadC (the centroid — the same centre spread uses),
     refreshed in the loop while the field is live, gated by fieldDisplaced like spreadK gates
     spread. Keeps picking / ladder / selection on the rotating field under a top-row spin. */
  const fieldQ: [number, number, number, number] = [0, 0, 0, 1];
  let fieldDisplaced = false;
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
    if (fieldDisplaced) {
      // v' = c + q ⊗ (v − c), t = 2 q×v — the stores.ts / morph.fieldSpinPoint kernel
      const qx = fieldQ[0], qy = fieldQ[1], qz = fieldQ[2], qw = fieldQ[3];
      const x = viewBuf[o]! - spreadC[0];
      const y = viewBuf[o + 1]! - spreadC[1];
      const z = viewBuf[o + 2]! - spreadC[2];
      const tx = 2 * (qy * z - qz * y);
      const ty = 2 * (qz * x - qx * z);
      const tz = 2 * (qx * y - qy * x);
      viewBuf[o] = spreadC[0] + x + qw * tx + (qy * tz - qz * ty);
      viewBuf[o + 1] = spreadC[1] + y + qw * ty + (qz * tx - qx * tz);
      viewBuf[o + 2] = spreadC[2] + z + qw * tz + (qx * ty - qy * tx);
    }
  };

  /* ---- cameras: flycam by default, ?cam=orbit for diagnostics. ------------------------- */
  const camera = new PerspectiveCamera(
    DEFAULT_FOV,
    window.innerWidth / window.innerHeight,
    Math.max(0.05, radius * 0.001),
    (radius + diagonal) * 20,
  );
  let liveFov = DEFAULT_FOV;

  /* Close-orbit (double-click) state, hoisted here because the flycam's injected zoomRange /
     softStop / roomAt closures read it and run at BOOT (cam.home → zoomRange), long before the
     ladder section where close-orbit is otherwise wired. -1 = not orbiting. While set, a single
     node is the orbit PIVOT (dragging circles it) and its reading card is held suspended. With
     the node px cap gone globally (library default uncapped), close-orbit no longer touches node
     SIZE — growth is global, this is only a re-target, so the damp and dolly behave identically
     whether or not you are orbiting (Fix 3: one consistent behavior). */
  let closeOrbit = -1;

  /** Leave close-orbit (any exit: new select, Escape, wiki-navigate, or the leash in the loop).
   *  A function declaration so every earlier handler can call it. */
  function leaveCloseOrbit(): void {
    if (closeOrbit < 0) return;
    closeOrbit = -1;
  }

  const roomPos = { x: 0, y: 0, z: 0 };
  const roomAt = (p: readonly [number, number, number]): number => {
    roomPos.x = p[0];
    roomPos.y = p[1];
    roomPos.z = p[2];
    return nearestDistance(grid, roomPos, { drift: gridDrift });
  };
  /* The nearest promoted-planet SURFACE, fed to the flycam as a solid floor + altitude speed-scale
     (near-mesh terrain stage A). nearMesh is created LAZILY when the 3D-nodes toggle is switched on
     (declared far below); this closure late-binds it, so with the LOD off (default) it returns null
     and the flycam is byte-identical to today. nearestSurface is O(promoted) and already null while
     the layer is disabled. */
  const surfaceQ = { x: 0, y: 0, z: 0 };
  const surfaceAt = (
    p: readonly [number, number, number],
  ): {
    center: readonly [number, number, number];
    radius: number;
    terrain?: (dx: number, dy: number, dz: number) => number;
  } | null => {
    if (!nearMesh) return null;
    surfaceQ.x = p[0];
    surfaceQ.y = p[1];
    surfaceQ.z = p[2];
    return nearMesh.nearestSurface(surfaceQ);
  };
  /* softStop is the proximity-damp SCALE (flycam.ts): approaching the nearest node, the W/S step
     eases toward a floor once the room AHEAD falls under ~2.5× this — a SMOOTH soft-landing ramp
     (the damp never gates to zero, so you can still push through) that brings you to a gentle
     creep right at the node and lets you release-to-hover rather than blowing through its centre.
     Scaled to NODE SPACING (× the live spread), NOT the whole-field radius, so the brake zone is
     "about one node away" at every zoom and every spread — the lever that fixes BOTH the old
     "block" (a field-radius scale made the ramp too abrupt) and the "zoom past" (close-orbit used
     to switch the damp off). ~0.35 spacing: the ramp spans ahead ∈ [0.4·stop, 2.5·stop] ≈ 0.7 of
     a spacing, so you decelerate over roughly one node-gap — gentle, not a crawl-to-halt, not off.
     This works WITHOUT selecting: any nearby node slows you, because roomAt measures the nearest
     node's distance regardless of selection. */
  const SOFT_STOP_SPACINGS = 0.35;
  const softStop = (): number => Math.max(spacing * field.spread * SOFT_STOP_SPACINGS, radius * 0.004);
  /* Min dolly radius (zoomRange lo) is relative to NODE SPACING (× the live spread): that is what
     lets the flycam get right up to a SINGLE node and circle it. Lowered GLOBALLY (not just in
     close-orbit) to ~0.01 of a node-spacing — nose-to-node — so a plain wheel-in on any target
     fills the view with NO select/orbit step, with a small absolute floor that still clears the
     adaptive near plane (render loop), which shrinks to < radius so nothing clips. The far end is
     unchanged. One floor for every mode (Fix 3). */
  const zoomRange = (): readonly [number, number] => [
    Math.max(spacing * field.spread * 0.01, 0.012),
    diagonal * 4,
  ];

  const flyCfg: FlycamConfig = { ...FLYCAM_DEFAULTS };
  let fly: Flycam | null = null;
  let cam: CameraController;
  if (camMode === 'fly') {
    fly = createFlycam(camera, flyCfg, { roomAt, softStop, zoomRange, surfaceAt });
    fly.spin = bool('cameraDrift');
    cam = fly;
  } else {
    cam = createOrbitCam(camera, { ...ORBIT_DEFAULTS }, { zoomRange });
  }
  cam.attach(canvas);
  cam.home({ center: centre, radius }); // the LIVE extent, never meta3.home (§10.8)

  /* ---- the HDR bloom post-process (DESIGN §5 tier-3): render additively into a half-float
     buffer → bloom → ONE post-sum tonemap → screen. Installed as the consumer helper it is —
     the loop calls post.render() instead of renderer.render() when GLOW is on. Default OFF, so
     nothing regresses; the 'nebula' look turns it on. Created eagerly (cheap — the render
     targets allocate lazily on the first post.render), tuned live via the glow controls.

     THE DOUBLE-TONEMAP SEAM: when glow is on the field's IN-MATERIAL roll-off is set to 'none'
     and the post applies the curve once over the accumulated sum; when glow is off the field
     keeps its own roll-off and the post is never rendered. applyTonemap (below) owns that seam. */
  const GLOW_KEY = 'starscape-playground:glow:v1';
  interface GlowState { on: boolean; strength: number; radius: number; threshold: number; }
  // Defaults tuned on 1M synth against the reference nebula: a moderate strength, a TIGHT radius
  // (a wide radius weights the coarse bloom mips and washes the whole screen grey — the nebula
  // wants local glow), and a threshold that blooms only the dense cores (so the sparse field and
  // the negative space between clusters stay black while the hubs blaze).
  const glow: GlowState = { on: false, strength: 1.3, radius: 0.4, threshold: 1.0 };
  try {
    const raw = localStorage.getItem(GLOW_KEY);
    if (raw !== null) {
      const s = JSON.parse(raw) as Partial<GlowState>;
      if (typeof s.on === 'boolean') glow.on = s.on;
      const fin = (v: unknown, d: number): number =>
        typeof v === 'number' && Number.isFinite(v) ? v : d;
      glow.strength = Math.min(3, Math.max(0, fin(s.strength, glow.strength)));
      glow.radius = Math.min(1, Math.max(0, fin(s.radius, glow.radius)));
      glow.threshold = Math.min(2, Math.max(0, fin(s.threshold, glow.threshold)));
    }
  } catch {
    /* storage denied/corrupt: the defaults (glow off) are a safe session */
  }
  const post: PostProcess = createPostProcess(renderer, scene, camera, {
    bloom: { strength: glow.strength, radius: glow.radius, threshold: glow.threshold },
    tonemap: (str('tonemap') as TonemapOp) ?? 'reinhard',
    exposure: num('exposure'),
    // Full-res bloom (resolutionScale 1): BloomNode defaults to 0.5× (half-res), whose coarse
    // mip pyramid smears bright energy screen-wide into a grey haze over the negative space.
    // Full-res keeps the glow LOCAL to the clusters — the nebula's black space needs it, and at
    // 1M it still holds vsync (the per-frame cost is the candidate loop, not the bloom).
    resolutionScale: 1.0,
  });
  const saveGlow = (): void => {
    try {
      localStorage.setItem(GLOW_KEY, JSON.stringify(glow));
    } catch {
      /* storage full/blocked — the control still works this session */
    }
  };

  const appTweens = createTweens();
  appTweens.set('fov', DEFAULT_FOV);
  appTweens.onChange('fov', (v) => {
    liveFov = v;
    camera.fov = v;
    camera.updateProjectionMatrix();
    field.setFov(v);
    nearMesh?.setView(window.innerHeight, v); // projected-size promotion tracks the live fov
  });

  /* A pose tween the flycam does not provide: goTo/home deliberately RETREAT ALONG THE
     CURRENT APPROACH (cameras/flycam.ts), which is right everywhere except the 2d flatten —
     a camera that was inside the field sees the flattened plane edge-on, i.e. a line. The
     predecessor's 2d button "brings the camera round"; this is that move, app-side: tween
     position AND orientation to a stated pose (top-down over the plane, north up). */
  const pose = {
    active: false,
    t: 0,
    ms: 900,
    p0: new Vector3(),
    p1: new Vector3(),
    q0: new Quaternion(),
    q1: new Quaternion(),
    g: new Vector3(),
  };
  const POSE_M = new Matrix4();
  const flyPoseTo = (p1: Vector3, look: Vector3, ms = 900): void => {
    if (!fly) {
      cam.home({ center: [look.x, look.y, look.z], radius });
      return;
    }
    fly.cancel();
    pose.p0.copy(camera.position);
    pose.q0.copy(camera.quaternion);
    // looking straight down −y: use −z as up (north-up map view; world-up is degenerate)
    POSE_M.lookAt(p1, look, new Vector3(0, 0, -1));
    pose.q1.setFromRotationMatrix(POSE_M);
    pose.p1.copy(p1);
    pose.g.copy(look);
    pose.t = 0;
    pose.ms = ms;
    pose.active = true;
  };

  const flyHome = (fovDeg: number): void => {
    const e = field.targetExtent();
    const c = new Vector3(e.centre[0], e.centre[1], e.centre[2]);
    const dist =
      (Math.max(e.radius, radius * 0.05) / Math.tan((fovDeg * Math.PI) / 360)) * HOME_MARGIN;
    if (field.projection === '2d') {
      // top-down over the flattened plane (flatAxis 1 = y), or the home is an edge-on line
      flyPoseTo(new Vector3(c.x, c.y + dist, c.z), c, 900);
    } else if (fly) {
      pose.active = false;
      fly.goTo(c, dist, 900);
    } else {
      cam.home({ center: e.centre, radius: e.radius });
    }
  };

  /* ---- the rail ------------------------------------------------------------------------- */
  const ui = {
    scene: document.getElementById('scene') as HTMLSelectElement,
    data: document.getElementById('data') as HTMLSelectElement,
    mode: document.getElementById('mode') as HTMLSelectElement,
    levelsbox: document.getElementById('levelsbox') as HTMLDivElement,
    hpreset: document.getElementById('hpreset') as HTMLSelectElement,
    levels: document.getElementById('levels') as HTMLDivElement,
    addlevel: document.getElementById('addlevel') as HTMLButtonElement,
    levelsreset: document.getElementById('levelsreset') as HTMLAnchorElement,
    pivotwarn: document.getElementById('pivotwarn') as HTMLDivElement,
    proj: document.getElementById('proj') as HTMLSelectElement,
    look: document.getElementById('look') as HTMLSelectElement,
    glow: document.getElementById('glow') as HTMLInputElement,
    glowStrength: document.getElementById('glowStrength') as HTMLInputElement,
    glowStrengthv: document.getElementById('glowStrengthv') as HTMLSpanElement,
    glowRadius: document.getElementById('glowRadius') as HTMLInputElement,
    glowRadiusv: document.getElementById('glowRadiusv') as HTMLSpanElement,
    glowThreshold: document.getElementById('glowThreshold') as HTMLInputElement,
    glowThresholdv: document.getElementById('glowThresholdv') as HTMLSpanElement,
    glowCompression: document.getElementById('glowCompression') as HTMLInputElement,
    glowCompressionv: document.getElementById('glowCompressionv') as HTMLSpanElement,
    viewdist: document.getElementById('viewdist') as HTMLInputElement,
    viewdistv: document.getElementById('viewdistv') as HTMLSpanElement,
    links: document.getElementById('links') as HTMLInputElement,
    near: document.getElementById('near') as HTMLInputElement,
    nearbox: document.getElementById('nearbox') as HTMLDivElement,
    nearShape: document.getElementById('nearShape') as HTMLSelectElement,
    nearTex: document.getElementById('nearTex') as HTMLSelectElement,
    glyphTex: document.getElementById('glyphTex') as HTMLSelectElement,
    surfaceDetail: document.getElementById('surfaceDetail') as HTMLSelectElement,
    nearTerrain: document.getElementById('nearTerrain') as HTMLInputElement,
    nearPromote: document.getElementById('nearPromote') as HTMLInputElement,
    nearPromotev: document.getElementById('nearPromotev') as HTMLSpanElement,
    nearCount: document.getElementById('nearCount') as HTMLInputElement,
    nearCountv: document.getElementById('nearCountv') as HTMLSpanElement,
    nearImg: document.getElementById('nearImg') as HTMLInputElement,
    home: document.getElementById('home') as HTMLButtonElement,
    tune: document.getElementById('tune') as HTMLButtonElement,
    search: document.getElementById('search') as HTMLInputElement,
    hits: document.getElementById('hits') as HTMLDivElement,
    regionsBox: document.getElementById('regions') as HTMLDivElement,
    regionList: document.getElementById('regionlist') as HTMLDivElement,
    regionLabels: document.getElementById('regionlabels') as HTMLDivElement,
    selpanel: document.getElementById('selpanel') as HTMLDivElement,
  };

  /* ---- the SCENE selector's "you've diverged" bookkeeping. Any manual edit to the hierarchy,
     the look, the glow or the view flips the selector to the 'custom' sentinel and persists it, so
     the control honestly reflects that the scene no longer matches a shipped preset. Guarded by
     applyingScene so the programmatic apply (applyScene, below) never trips it. */
  let currentScene = 'space4';
  let applyingScene = false;
  const markSceneCustom = (): void => {
    if (applyingScene || currentScene === 'custom') return;
    currentScene = 'custom';
    ui.scene.value = 'custom';
    try {
      localStorage.setItem(SCENE_KEY, 'custom');
    } catch {
      /* storage denied: the session still works, the choice just will not persist */
    }
  };

  /* ---- the DATA select: reflect the booted dataset, and write the choice into the query
     string + reload on change (state survives reloads — the predecessor's CORPUS picker). */
  {
    const want = String(synthN);
    if (!Array.from(ui.data.options).some((o) => o.value === want)) {
      // a hand-typed ?n= that is not one of the shipped sizes still shows truthfully
      const opt = document.createElement('option');
      opt.value = want;
      opt.textContent = `${synthN.toLocaleString('en-US')} nodes`;
      ui.data.appendChild(opt);
    }
    ui.data.value = want;
  }
  ui.data.addEventListener('change', () => {
    const q = new URLSearchParams(location.search);
    q.set('n', ui.data.value);
    const qs = q.toString();
    location.href = location.pathname + (qs ? `?${qs}` : '');
  });

  let hierarchy: CompiledHierarchy | null = null;
  /** True when the live hierarchy's level 0 groups by 'region' — the only arrangements in
   *  which a region HAS a place (centre + radius) to label and fly to. Frozen counts too:
   *  meta3's measured region centres are the frozen layout's own. */
  let regionHierarchy = false;

  /** Live spin rates into the pivot table (the FrameSet reads `rate` fresh every advance)
   *  PLUS the field-global root-frame rate, and the drift bound from both.
   *
   *  THE SHIFTED MAPPING — the exact parallel of the (already-shifted) spread mapping
   *  (rootRadiusScale ← row0, level-k radiusScale ← row k+1). A row's spin now rotates its
   *  groups about their PARENT, so the CHILD row drives the parent frame: pivot level `lv`
   *  reads row `lv+1`. Row 0 has no pivot above it, so it drives the FIELD-GLOBAL rotation
   *  (its top-level groups orbit the centroid together — the new root frame). The leaf row is
   *  the deepest child and now turns the innermost groupBy groups' contents (it is live, not
   *  grayed). The field rotation does NOT touch the pivot store, so row 0 spins the field even
   *  when the pivots overflow (no FrameSet installs and per-pivot rates are inert then). */
  /** The three canonical spin axes the per-level axis select offers, and the label of a
   *  (possibly persisted/arbitrary) axis — its dominant component. */
  const AXES: Record<string, [number, number, number]> = {
    X: [1, 0, 0],
    Y: [0, 1, 0],
    Z: [0, 0, 1],
  };
  const axisLabel = (a: [number, number, number]): string => {
    const ax = Math.abs(a[0]);
    const ay = Math.abs(a[1]);
    const az = Math.abs(a[2]);
    return ax >= ay && ax >= az ? 'X' : az >= ay ? 'Z' : 'Y';
  };
  /** Unit axis (the FrameSet requires unit length; the field spin normalises too). Falls back
   *  to canonical up for a degenerate zero axis. */
  const axisUnit = (a: [number, number, number]): [number, number, number] => {
    const len = Math.hypot(a[0], a[1], a[2]);
    return len > 1e-9 ? [a[0] / len, a[1] / len, a[2] / len] : [0, 1, 0];
  };

  const applySpins = (): void => {
    if (!hierarchy) {
      spinDriftRate = 0;
      field.setFieldSpin(0);
      return;
    }
    const override = num('levelSpin');
    const pv = hierarchy.pivots;
    const rowRate = (k: number): number => (rows[k]?.spin ? rows[k]!.spinRate || 0 : 0);
    // the axis DRIVING a pivot/field frame follows the same shifted mapping as the rate: the
    // field frame is row 0's axis, a pivot at level `lv` is driven by row `lv + 1`.
    const rowAxis = (k: number): [number, number, number] =>
      axisUnit(rows[k]?.spinAxis ?? [0, 1, 0]);
    const fieldRate = override > 0 ? override : rowRate(0); // row 0 → the whole field turns
    field.setFieldSpin(fieldRate, rowAxis(0));
    if (pivotOverflow) {
      // pivot frames are inert (no FrameSet), but the field rotation still lives — drift from it
      spinDriftRate = Math.abs(fieldRate) * radius * 1.5;
      return;
    }
    const maxSpeed = rows.map(() => 0);
    for (let p = 0; p < pv.count; p++) {
      const lv = pv.level[p]!;
      const r = override > 0 ? override : rowRate(lv + 1); // the child row drives this frame
      pv.rate[p] = r;
      // the FrameSet re-reads `axis` each advance (it must stay unit) — write the driving
      // row's axis live, parallel to the rate, so an axis change tilts without a recompile.
      const [axx, axy, axz] = rowAxis(lv + 1);
      pv.axis[p * 3] = axx;
      pv.axis[p * 3 + 1] = axy;
      pv.axis[p * 3 + 2] = axz;
      const speed = Math.abs(r) * pv.radius[p]!;
      if (speed > (maxSpeed[lv] ?? 0)) maxSpeed[lv] = speed;
    }
    // drift bound: each level's fastest node motion (the frame spins its own contents about
    // its own centre) PLUS the field rotation sweeping the whole field about the centroid
    // (up to ~the extent radius out). The render loop scales this by the live spread.
    let sum = Math.abs(fieldRate) * radius;
    for (const s of maxSpeed) sum += s;
    spinDriftRate = sum * 1.5;
  };

  /* ======================================================================================
   * THE LEVELS EDITOR — the custom arrange mode. Every option list is enumerated AT
   * RUNTIME from the library's own registries (shapeKeys / presetKeys / densityKeys /
   * scatterKeys) and from the ACTIVE graph's facets — nothing hardcoded, so a registered
   * extension or a new dataset facet shows up here for free. The editor is the source of
   * truth: the "preset hierarchy" select only POPULATES it with shipped starting points
   * (the old fixed arrangements, now just default level configs). Per level: clustering
   * (groupBy), arrangement (shape or placement preset), distribution (density + its main
   * param + the axis it warps when the shape has more than one), scatter (+ thickness
   * param), spread (the library's per-level radiusScale) and spin (live rates through the
   * FrameSet plumbing — applySpins above). The config persists per dataset mode.
   * ====================================================================================== */

  /** A per-level placement recipe. The four knobs (shape/ordering/density/scatter) plus a
   *  numeric-override map per knob (empty = every declared param at its library default, so
   *  an untouched row compiles byte-identically to the shipped behaviour). `ordering` +
   *  `orderFacet`/`orderColumn` feed the library's `ordering` + `stringParams`; `spinAxis`
   *  feeds the spin frame's axis. */
  interface LevelRow {
    groupBy: string | null; // null = the leaf row (always last)
    place: string; // shape key, 'preset:<name>', or a per-group sentinel
    ordering: string; // ordering key ('byIndex' default = today's implicit behaviour)
    orderFacet: string | null; // byFacet's facet key (null until chosen)
    orderColumn: string | null; // byColumn's numeric-column key
    density: string;
    densityAxis: number;
    scatter: string;
    shapeP: Record<string, number>; // shape param overrides (key → value); empty = defaults
    densP: Record<string, number>; // density param overrides
    scatP: Record<string, number>; // scatter param overrides
    spread: number; // the level's radiusScale (1 = the shipped count^⅓ policy)
    spin: boolean;
    spinRate: number;
    spinAxis: [number, number, number]; // rotation axis (unit; default canonical up)
  }

  /** The shape KEY whose params a row exposes: a bare shape is itself; a preset exposes its
   *  bundled shape; a per-group (random / by-facet) row has no single shape (null). */
  const shapeKeyOf = (place: string): string | null =>
    isPerGroupRow(place)
      ? null
      : place.startsWith('preset:')
        ? getPreset(place.slice('preset:'.length)).shape
        : place;

  /** The declared shape params for a row's place ([] for per-group rows). */
  const rowShapeParams = (place: string): ParamDesc[] => {
    const k = shapeKeyOf(place);
    return k === null ? [] : shapeParams(k);
  };

  /** Emit into `dst` the map entries the library should receive: a param is sent only when
   *  it differs from its declared default (omit-at-default — so an untouched control is
   *  byte-identical to omission, which matters for e.g. ball.radialPower's cbrt and
   *  dendrite.tiers' auto), UNLESS its key is forced (mixture component means, which are
   *  presence flags the library reads verbatim). */
  const emitParams = (
    descs: ParamDesc[],
    map: Record<string, number>,
    dst: Record<string, number>,
    force?: Set<string>,
  ): void => {
    for (const d of descs) {
      const v = map[d.key];
      if (v === undefined || !Number.isFinite(v)) continue;
      if (force?.has(d.key) || v !== d.default) dst[d.key] = v;
    }
  };

  /** Editor depth: 1..6 level rows, never past the compiler's own MAX_LEVELS. */
  const UI_MAX_LEVELS = Math.min(6, MAX_LEVELS);

  /** The level-editor sentinel for a per-group RANDOM arrangement — mapped to the library's
   *  `{ random: true }` place at compile time (compileLevels/buildPlace). Not a shape key, so it
   *  never reaches getShape; placeSpecOf/dim lookups guard on it. */
  const RANDOM_PLACE = '__random__';
  const isRandomRow = (place: string): boolean => place === RANDOM_PLACE;

  /** The level-editor sentinel for a per-group BY-FACET arrangement (F2) — mapped to the library's
   *  `{ random: true, by: 'facet' }` place at compile time (buildPlace). Each filled group draws
   *  its shape from the shape set by its OWN groupBy facet-value index (shapes[value mod n]), so
   *  all clusters sharing a facet value get the SAME shape — meaningful, not scattered like the
   *  random form. Like RANDOM_PLACE it is not a shape key, so placeSpecOf/dim/getShape guard on it. */
  const BY_FACET_PLACE = '__byfacet__';
  const isByFacetRow = (place: string): boolean => place === BY_FACET_PLACE;
  /** Either per-group form (random or by-facet): both map to the library `{ random: true }` family
   *  and share the same editor guards — no single fixed shape, so density axis is pinned to 0. */
  const isPerGroupRow = (place: string): boolean => isRandomRow(place) || isByFacetRow(place);

  const placeSpecOf = (place: string): PlacementSpec =>
    place.startsWith('preset:') ? getPreset(place.slice('preset:'.length)) : { shape: place };

  /** The mixture density's component MEANS are presence flags, not `?? default` fallbacks —
   *  the engine reads `m<i>` verbatim to switch component i on. buildPlace forces these keys
   *  through emit-at-default so a mean that happens to equal its seed still reaches the
   *  engine (and component 0's `m0` is injected when absent — a mixture needs one). */
  const MIX_MEANS = new Set(['m0', 'm1', 'm2', 'm3']);

  /** Pick the declared keys present in `src` into a fresh override map (preset seeding). */
  const seedMap = (descs: ParamDesc[], src: Record<string, number>): Record<string, number> => {
    const m: Record<string, number> = {};
    for (const d of descs) {
      const v = src[d.key];
      if (v !== undefined) m[d.key] = v;
    }
    return m;
  };

  /** Choosing a place POPULATES the row's knobs: a preset seeds density/scatter/ordering/axis
   *  AND each knob's override map from its own bundle (still editable — edits spread the frozen
   *  preset into a fresh spec at compile time); a bare shape keeps the density/scatter knobs,
   *  resets its shape-param map (the new shape's params differ) and clamps the density axis to
   *  its dim. The per-group sentinels (RANDOM / BY-FACET) keep density/scatter (applied to every
   *  group) but have no single shape, so shape params are cleared and the density axis pinned
   *  to 0 — the per-group shapes vary in dimension, and axis 0 is the one they all have. */
  const setPlace = (row: LevelRow, place: string): void => {
    row.place = place;
    row.shapeP = {};
    if (isPerGroupRow(place)) {
      row.densityAxis = 0;
      return;
    }
    if (place.startsWith('preset:')) {
      const p = getPreset(place.slice('preset:'.length));
      const pp = p.params ?? {};
      row.ordering = p.ordering ?? 'byIndex';
      row.density = p.density ?? 'uniform';
      row.densityAxis = p.densityAxis ?? 0;
      row.scatter = p.scatter ?? 'none';
      row.shapeP = seedMap(shapeParams(p.shape), pp);
      row.densP = seedMap(densityParams(row.density), pp);
      row.scatP = seedMap(scatterParams(row.scatter), pp);
      return;
    }
    if (row.densityAxis >= getShape(place).dim) row.densityAxis = 0;
  };

  const mkRow = (groupBy: string | null, place: string, over: Partial<LevelRow> = {}): LevelRow => {
    const row: LevelRow = {
      groupBy,
      place: 'cloud',
      ordering: 'byIndex',
      orderFacet: null,
      orderColumn: null,
      density: 'uniform',
      densityAxis: 0,
      scatter: 'none',
      shapeP: {},
      densP: {},
      scatP: {},
      spread: 1,
      spin: false,
      spinRate: 0.3,
      spinAxis: [0, 1, 0],
    };
    setPlace(row, place);
    return Object.assign(row, over);
  };

  /** The shipped default hierarchies — the old fixed arrangements as starting points that
   *  POPULATE the editor. Facet keys are the ACTIVE dataset's own (vault: region/type,
   *  synth: galaxy/type), exactly as the old hardcoded levels were. */
  const HPRESETS: Record<string, { label: string; rows: () => LevelRow[] }> = {
    'regions-2': {
      label: `${V.regionFacet} shells (2 levels)`,
      rows: () => [mkRow(V.regionFacet, 'shell', { spinRate: 0.15 }), mkRow(null, 'cloud')],
    },
    galaxies: {
      label: 'galaxies (spiralGalaxy)',
      rows: () => [
        mkRow(V.regionFacet, 'shell', { spinRate: 0.15 }),
        mkRow(V.groupFacet, 'cloud', { spinRate: 0.5 }),
        mkRow(null, 'preset:spiralGalaxy'),
      ],
    },
    ring: {
      label: `ring (${V.regionFacet}s)`,
      rows: () => [
        mkRow(V.regionFacet, 'shell', { spinRate: 0.15 }),
        mkRow(V.groupFacet, 'cloud', { spinRate: 0.5 }),
        mkRow(null, 'ring', { scatter: 'gaussian', scatP: { thickness: 0.06 } }),
      ],
    },
    globe: {
      label: 'globe (shell)',
      rows: () => [
        mkRow(V.regionFacet, 'shell', { spinRate: 0.15 }),
        mkRow(V.groupFacet, 'cloud', { spinRate: 0.5 }),
        mkRow(null, 'shell'),
      ],
    },
    random: {
      label: 'random (one cloud)',
      rows: () => [mkRow(null, 'cloud', { spinRate: 0.1 })],
    },
    'random-shapes': {
      // one group per top facet value, each laying its pages out with a DIFFERENT shape
      // (library `{ random: true }`): a ring here, a globe there, a box elsewhere — all
      // deterministic by group. A coarse top facet gives ~8–20 groups, so the variety reads.
      label: `random shapes (${V.regionFacet}s)`,
      rows: () => [mkRow(V.regionFacet, 'shell', { spinRate: 0.1 }), mkRow(null, RANDOM_PLACE)],
    },
  };
  const DEFAULT_HPRESET = 'regions-2';

  let mode: 'frozen' | 'custom' = 'frozen';
  let rows: LevelRow[] = HPRESETS[DEFAULT_HPRESET]!.rows();
  /** True when the live hierarchy's pivot count would overflow the field's pivot store —
   *  the geometry still compiles and morphs, but no FrameSet installs and spins stay off
   *  (the inline warning says so) instead of the store throwing mid-boot. */
  let pivotOverflow = false;

  /* ---- persistence: mode-scoped, like the config store's keys. */
  const LEVELS_KEY = 'starscape-playground:levels:v1';
  const saveLevels = (): void => {
    try {
      localStorage.setItem(LEVELS_KEY, JSON.stringify({ mode, preset: ui.hpreset.value, rows }));
    } catch {
      /* storage denied/full: the session still works, it just will not persist */
    }
    markSceneCustom(); // any hierarchy edit diverges from the active scene preset (guarded)
  };
  const loadLevels = (): { mode: 'frozen' | 'custom'; preset: string; rows: LevelRow[] } | null => {
    try {
      const raw = localStorage.getItem(LEVELS_KEY);
      if (raw === null) return null;
      const s = JSON.parse(raw) as { mode?: unknown; preset?: unknown; rows?: unknown };
      const list = s.rows;
      if (!Array.isArray(list) || list.length < 1 || list.length > UI_MAX_LEVELS) return null;
      const shapes = new Set(shapeKeys());
      const presets = new Set(presetKeys().map((p) => `preset:${p}`));
      const dens = new Set(densityKeys());
      const scats = new Set(scatterKeys());
      const orders = new Set(orderingKeys());
      const facets = new Set(Object.keys(graph.facets));
      const cols = new Set(Object.keys(graph.columns));
      const fin = (v: unknown, d: number): number =>
        typeof v === 'number' && Number.isFinite(v) ? v : d;
      /** Keep only the declared keys of a persisted override map, dropping unknown/missing/
       *  non-finite entries — the guard the plan asks for (fall back to declared defaults). */
      const sanMap = (v: unknown, descs: ParamDesc[]): Record<string, number> => {
        const m: Record<string, number> = {};
        if (v !== null && typeof v === 'object') {
          const o = v as Record<string, unknown>;
          for (const d of descs) {
            const x = o[d.key];
            if (typeof x === 'number' && Number.isFinite(x)) m[d.key] = x;
          }
        }
        return m;
      };
      const out: LevelRow[] = [];
      for (let k = 0; k < list.length; k++) {
        const r = list[k] as Partial<LevelRow>;
        const isLeaf = k === list.length - 1;
        if (isLeaf ? r.groupBy != null : typeof r.groupBy !== 'string' || !facets.has(r.groupBy))
          return null;
        if (
          typeof r.place !== 'string' ||
          !(shapes.has(r.place) || presets.has(r.place) || isPerGroupRow(r.place))
        )
          return null;
        if (typeof r.density !== 'string' || !dens.has(r.density)) return null;
        if (typeof r.scatter !== 'string' || !scats.has(r.scatter)) return null;
        const place = r.place;
        const ax = r.spinAxis;
        const spinAxis: [number, number, number] =
          Array.isArray(ax) &&
          ax.length === 3 &&
          ax.every((c) => typeof c === 'number' && Number.isFinite(c)) &&
          (ax[0] !== 0 || ax[1] !== 0 || ax[2] !== 0)
            ? [ax[0] as number, ax[1] as number, ax[2] as number]
            : [0, 1, 0];
        out.push({
          groupBy: isLeaf ? null : (r.groupBy as string),
          place,
          ordering: typeof r.ordering === 'string' && orders.has(r.ordering) ? r.ordering : 'byIndex',
          orderFacet:
            typeof r.orderFacet === 'string' && facets.has(r.orderFacet) ? r.orderFacet : null,
          orderColumn:
            typeof r.orderColumn === 'string' && cols.has(r.orderColumn) ? r.orderColumn : null,
          density: r.density,
          densityAxis: Math.min(Math.max(fin(r.densityAxis, 0) | 0, 0), 2),
          scatter: r.scatter,
          shapeP: sanMap(r.shapeP, rowShapeParams(place)),
          densP: sanMap(r.densP, densityParams(r.density)),
          scatP: sanMap(r.scatP, scatterParams(r.scatter)),
          spread: Math.min(10, Math.max(0.25, fin(r.spread, 1))),
          spin: r.spin === true,
          spinRate: fin(r.spinRate, 0.3),
          spinAxis,
        });
      }
      return {
        mode: s.mode === 'custom' ? 'custom' : 'frozen',
        preset: typeof s.preset === 'string' ? s.preset : '',
        rows: out,
      };
    } catch {
      return null;
    }
  };

  /* ---- compile the editor's rows into the library's LevelSpec list. A RANDOM row maps to the
     library's per-group `{ random: true }` place: the density/scatter/params still apply to every
     group (via `base`), only the SHAPE varies per group. densityAxis is left at 0 — the per-group
     shapes vary in dimension, so axis 0 is the one they all have. */
  /** The ordering's `stringParams` (the facet/column KEY byFacet/byColumn read) — undefined
   *  for the keyless orderings, so nothing is sent and the default byIndex path is unchanged. */
  const orderStringParams = (row: LevelRow): Record<string, string> | undefined => {
    if (row.ordering === 'byFacet' && row.orderFacet) return { facet: row.orderFacet };
    if (row.ordering === 'byColumn' && row.orderColumn) return { column: row.orderColumn };
    return undefined;
  };

  const buildPlace = (row: LevelRow): PlacementSpec | RandomPlace => {
    const densForce = row.density === 'mixture' ? MIX_MEANS : undefined;
    const sp = orderStringParams(row);

    if (isPerGroupRow(row.place)) {
      const base: Omit<PlacementSpec, 'shape'> = { density: row.density, scatter: row.scatter };
      if (row.ordering !== 'byIndex') base.ordering = row.ordering;
      if (sp) base.stringParams = sp;
      const params: Record<string, number> = {};
      emitParams(densityParams(row.density), row.densP, params, densForce);
      emitParams(scatterParams(row.scatter), row.scatP, params);
      if (row.density === 'mixture' && params['m0'] === undefined) params['m0'] = 0;
      base.params = params;
      // BY-FACET keys the per-group shape to each group's own facet value; RANDOM hashes it.
      return isByFacetRow(row.place) ? { random: true, by: 'facet', base } : { random: true, base };
    }

    const isPreset = row.place.startsWith('preset:');
    const base = placeSpecOf(row.place); // {shape} for a bare shape, or the preset's frozen spec
    const spec: PlacementSpec = { ...base };
    spec.ordering = row.ordering;
    if (sp) spec.stringParams = sp;
    else delete spec.stringParams;
    spec.density = row.density;
    spec.scatter = row.scatter;
    const dim = getShape(spec.shape).dim;
    spec.densityAxis = Math.min(Math.max(row.densityAxis | 0, 0), dim - 1) as 0 | 1 | 2;

    // A PRESET starts from its own param bundle and the override maps (seeded to the preset's
    // values) win key-by-key, so an untouched preset row reproduces it exactly; a BARE shape
    // starts empty and emits only overrides that differ from the library default (omit-at-
    // default — the untouched row is byte-identical to the shipped behaviour).
    const params: Record<string, number> = isPreset ? { ...(base.params ?? {}) } : {};
    if (isPreset) {
      const over = (descs: ParamDesc[], map: Record<string, number>, force?: Set<string>): void => {
        for (const d of descs) {
          const v = map[d.key];
          if (v === undefined || !Number.isFinite(v)) continue;
          if (force?.has(d.key) || v !== d.default || params[d.key] !== undefined) params[d.key] = v;
        }
      };
      over(shapeParams(spec.shape), row.shapeP);
      over(densityParams(row.density), row.densP, densForce);
      over(scatterParams(row.scatter), row.scatP);
    } else {
      emitParams(shapeParams(spec.shape), row.shapeP, params);
      emitParams(densityParams(row.density), row.densP, params, densForce);
      emitParams(scatterParams(row.scatter), row.scatP, params);
    }
    if (row.density === 'mixture' && params['m0'] === undefined) params['m0'] = 0;
    spec.params = params;
    return spec;
  };

  const compileLevels = (): CompiledHierarchy => {
    const L = rows.length;
    /* A row's SPREAD controls how far apart ITS OWN groups sit — which is the radius of
       their PARENT, one level up. So each level's radiusScale is driven by the NEXT row
       down (the children seated inside it), and row 0's groups (the outermost) are seated
       in the root radius, which is opts.rootRadiusScale. The leaf row therefore gets a live
       spread (it governs how far pages spread inside their deepest group), and the leaf
       LEVEL never carries a radiusScale (it has no pivot of its own). Leaf-only (L===1) is
       the exception: the single whole-field group's own radius IS the page spread, so row
       0 drives that level's radiusScale directly. */
    const levels: LevelSpec[] = rows.map((row, k) => {
      const isLeaf = k === L - 1;
      const lv: LevelSpec = { place: buildPlace(row) };
      if (!isLeaf) lv.groupBy = row.groupBy ?? undefined;
      if (!isLeaf) lv.spin = { rate: 0 }; // an idle frame so live rate writes have a target (§3.5)
      if (L === 1) {
        lv.spin = { rate: 0 };
        if (row.spread !== 1) lv.radiusScale = row.spread; // leaf-only: scales the whole field
      } else {
        const childRow = rows[k + 1]; // the children seated inside this level
        if (childRow && childRow.spread !== 1) lv.radiusScale = childRow.spread;
      }
      return lv;
    });
    // row 0's spread always scales the root radius — the sphere its groups sit in. For a
    // multi-level hierarchy that spreads the top-level groups apart; for a leaf-only field it
    // grows the root in step with the single group's own radiusScale, so the two stay equal
    // and the parent-fit guard never clamps it (without this, leaf-only spread > 1 was eaten
    // by the clamp and had no effect).
    const rootRadiusScale = Math.min(10, Math.max(0.25, rows[0]?.spread ?? 1));
    return compileHierarchy(graph, levels, { spacing: arrangeSpacing, rootRadiusScale });
  };

  /* ---- the two arrange paths. */
  const applyFrozen = (): void => {
    morphing = true;
    hierarchy = null;
    pivotOverflow = false;
    field.setHierarchy(null);
    regionHierarchy = true; // meta3's region centres ARE the frozen layout's
    applySpins();
    computeRegionViews();
    updateLevelStats();
    flyHome(liveFov);
    updateStats();
  };

  const applyCustom = (flyAfter: boolean): void => {
    try {
      const h = compileLevels();
      // The pivot STORE (not the compiler) holds PIVOT_CAPACITY slots, and only spinning
      // frames reach it: past the cap, install the same geometry with chainDepth 0 so no
      // FrameSet ever syncs — the arrangement still works, spins stay off, and the inline
      // warning says why (the synth `system` facet at 10⁶ is the known case).
      pivotOverflow = h.pivots.count + 1 > PIVOT_CAPACITY;
      morphing = true;
      hierarchy = h;
      field.setHierarchy(pivotOverflow ? { ...h, chainDepth: 0 } : h);
      regionHierarchy = rows.length > 1 && rows[0]!.groupBy === V.regionFacet;
      applySpins();
      computeRegionViews();
      updateLevelStats();
      if (flyAfter) flyHome(liveFov);
    } catch (err) {
      // a compile refusal (the library's loud guard rails) lands in the warning box —
      // the previous arrangement stays on screen, nothing breaks
      ui.pivotwarn.textContent = err instanceof Error ? err.message : String(err);
      ui.pivotwarn.classList.add('on', 'err');
    }
    updateStats();
  };

  let levelTimer = 0;
  /** Debounced recompile (~150 ms): slider drags fire per pixel and the compile is O(n). */
  const scheduleCustom = (): void => {
    if (mode !== 'custom') return;
    window.clearTimeout(levelTimer);
    levelTimer = window.setTimeout(() => applyCustom(false), 150);
  };
  const applyCustomNow = (flyAfter: boolean): void => {
    window.clearTimeout(levelTimer);
    applyCustom(flyAfter);
  };

  /* ---- the editor DOM: one card per level, rebuilt on structural change only (number
     and range inputs keep focus through their own drags). */
  interface RowEls {
    badge: HTMLSpanElement;
    spreadRow: HTMLElement;
    spinRow: HTMLElement;
  }
  let rowEls: RowEls[] = [];
  const fmtI = (x: number): string => x.toLocaleString('en-US');

  /** Group counts + guard rails, refreshed after every compile: each grouping row shows
   *  its level's group count (stats.groupsPerLevel), the leaf shows the page count; the
   *  spread/spin rows grey out where they have no pivot to act on. */
  const updateLevelStats = (): void => {
    const h = hierarchy;
    for (let k = 0; k < rows.length; k++) {
      const els = rowEls[k];
      if (!els) continue;
      const isLeaf = k === rows.length - 1;
      els.badge.textContent = !h
        ? ''
        : isLeaf
          ? `${fmtI(n)} nodes`
          : `${fmtI(h.stats.groupsPerLevel[k] ?? 0)} groups`;
      // Spin is live on EVERY row now (the shifted mapping): the leaf row turns the innermost
      // groupBy groups' contents, and row 0 drives the FIELD-GLOBAL rotation. Under pivot
      // overflow the per-pivot frames are inert — but row 0's field spin needs no pivots, so
      // only rows ≥ 1 gray out; row 0 stays live and spins the whole field.
      els.spinRow.classList.toggle('off', pivotOverflow && k > 0);
      // spread is always live: a row's slider spreads ITS groups within the parent, and the
      // leaf row spreads pages within their deepest group — so it is meaningful on every row.
      els.spreadRow.classList.remove('off');
    }
    if (h !== null && pivotOverflow) {
      ui.pivotwarn.textContent =
        `${fmtI(h.pivots.count)} pivots exceed the spin store's capacity ` +
        `(${fmtI(PIVOT_CAPACITY)}). The arrangement works, but spins are off for this ` +
        `config — group by a coarser facet to spin it.`;
      ui.pivotwarn.classList.add('on');
      ui.pivotwarn.classList.remove('err');
    } else {
      ui.pivotwarn.classList.remove('on', 'err');
      ui.pivotwarn.textContent = '';
    }
  };

  /** Any manual edit: the editor is now the source of truth (the preset select drops to
   *  its CUSTOM marker), persists, optionally rebuilds the DOM, recompiles debounced. */
  const edited = (structural = false): void => {
    ui.hpreset.value = '';
    saveLevels();
    if (structural) renderLevels();
    scheduleCustom();
  };

  const buildRowEl = (row: LevelRow, k: number): HTMLDivElement => {
    const isLeaf = k === rows.length - 1;
    const el = document.createElement('div');
    el.className = isLeaf ? 'lv leaf' : 'lv';

    /* header: level tag · clustering facet (the ACTIVE dataset's own) · count · remove */
    const head = document.createElement('div');
    head.className = 'lvh';
    const tag = document.createElement('b');
    tag.textContent = `L${k}`;
    head.appendChild(tag);
    const g = document.createElement('select');
    g.title = 'clustering: the facet this level groups by';
    if (isLeaf) {
      const o = document.createElement('option');
      o.textContent = '— nodes —';
      g.appendChild(o);
      g.disabled = true;
    } else {
      for (const f of Object.keys(graph.facets)) {
        const o = document.createElement('option');
        o.value = f;
        o.textContent = f;
        g.appendChild(o);
      }
      g.value = row.groupBy ?? '';
      g.addEventListener('change', () => {
        row.groupBy = g.value;
        edited();
      });
    }
    head.appendChild(g);
    const badge = document.createElement('span');
    badge.className = 'lv-n';
    head.appendChild(badge);
    if (!isLeaf) {
      const x = document.createElement('button');
      x.type = 'button';
      x.className = 'lv-x';
      x.textContent = '×';
      x.title = 'remove this level';
      x.addEventListener('click', () => {
        rows.splice(k, 1);
        edited(true);
      });
      head.appendChild(x);
    }
    el.appendChild(head);

    const lrow = (label: string): HTMLDivElement => {
      const r = document.createElement('div');
      r.className = 'lvrow';
      const i = document.createElement('i');
      i.textContent = label;
      r.appendChild(i);
      el.appendChild(r);
      return r;
    };
    const opt = (sel: HTMLSelectElement | HTMLOptGroupElement, value: string, label: string): void => {
      const o = document.createElement('option');
      o.value = value;
      o.textContent = label;
      sel.appendChild(o);
    };

    /* A compact subrow of one number input per declared param of a knob, writing into `map`.
       Each input shows the stored override or the declared default; an untouched input never
       enters `map`, so buildPlace omits it (byte-identical to the shipped behaviour). */
    const numField = (d: ParamDesc, map: Record<string, number>): HTMLLabelElement => {
      const pp = document.createElement('label');
      pp.className = 'pp';
      pp.title = `${d.key}  ·  default ${+d.default.toFixed(4)}  ·  ${d.min}–${d.max}`;
      const sp = document.createElement('span');
      sp.textContent = d.label;
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = String(d.step);
      inp.min = String(d.min);
      inp.max = String(d.max);
      inp.value = String(map[d.key] ?? d.default);
      inp.addEventListener('input', () => {
        const v = Number(inp.value);
        if (inp.value.trim() !== '' && Number.isFinite(v)) map[d.key] = v;
        else delete map[d.key];
        edited();
      });
      pp.appendChild(sp);
      pp.appendChild(inp);
      return pp;
    };
    const paramPack = (label: string, descs: ParamDesc[], map: Record<string, number>): void => {
      if (descs.length === 0) return;
      const r = lrow(label);
      const wrap = document.createElement('div');
      wrap.className = 'pp-wrap';
      for (const d of descs) wrap.appendChild(numField(d, map));
      r.appendChild(wrap);
    };
    /* mixture needs presence-based components: m0 is always on (the compiler injects it),
       m1..m3 opt in via a checkbox (their mean's PRESENCE is the toggle the engine reads). */
    const mixturePack = (map: Record<string, number>): void => {
      const by = new Map(densityParams('mixture').map((d) => [d.key, d] as const));
      const r = lrow('mix·c');
      const wrap = document.createElement('div');
      wrap.className = 'pp-wrap';
      for (const key of ['m0', 's0', 'w0']) wrap.appendChild(numField(by.get(key)!, map));
      for (let i = 1; i <= 3; i++) {
        const on = map[`m${i}`] !== undefined;
        const cb = document.createElement('label');
        cb.className = 'pp pp-cb';
        cb.title = `component ${i}: toggle its gaussian on/off`;
        const chk = document.createElement('input');
        chk.type = 'checkbox';
        chk.checked = on;
        const lab = document.createElement('span');
        lab.textContent = `c${i}`;
        chk.addEventListener('change', () => {
          if (chk.checked) map[`m${i}`] = by.get(`m${i}`)!.default;
          else {
            delete map[`m${i}`];
            delete map[`s${i}`];
            delete map[`w${i}`];
          }
          edited(true);
        });
        cb.appendChild(lab);
        cb.appendChild(chk);
        wrap.appendChild(cb);
        if (on) for (const key of [`m${i}`, `s${i}`, `w${i}`]) wrap.appendChild(numField(by.get(key)!, map));
      }
      r.appendChild(wrap);
    };

    /* arrangement: ONE select — the registry's shapes, its placement presets, then the
       per-group RANDOM entry (maps to the library's `{ random: true }` place: each group in
       this level draws a different shape, deterministically by group). */
    const srow = lrow('shape');
    const s = document.createElement('select');
    const ogS = document.createElement('optgroup');
    ogS.label = 'shapes';
    for (const key of shapeKeys()) opt(ogS, key, key);
    const ogP = document.createElement('optgroup');
    ogP.label = 'presets';
    for (const name of presetKeys()) opt(ogP, `preset:${name}`, name);
    const ogR = document.createElement('optgroup');
    ogR.label = 'special';
    opt(ogR, RANDOM_PLACE, 'random (per group)');
    opt(ogR, BY_FACET_PLACE, 'by cluster facet');
    s.appendChild(ogS);
    s.appendChild(ogP);
    s.appendChild(ogR);
    s.value = row.place;
    s.addEventListener('change', () => {
      setPlace(row, s.value); // seeds density/scatter/ordering/params — row re-renders to show it
      edited(true);
    });
    srow.appendChild(s);
    // the row's shape (bare or the preset's bundled shape) exposes its own params here.
    paramPack('shape·p', rowShapeParams(row.place), row.shapeP);

    /* ordering: WHO sits where along the shape's primary axis + the facet/column sub-picker
       the keyed orderings (byFacet/byColumn) read. byIndex (the default) changes nothing;
       byFacet/byColumn are graph-backed, so they resolve on the leaf (page) level. */
    const orow = lrow('order');
    const os = document.createElement('select');
    os.title = 'ordering: who sits where along the shape’s primary axis (facet/column act on the leaf)';
    for (const key of orderingKeys()) opt(os, key, key);
    os.value = row.ordering;
    os.addEventListener('change', () => {
      row.ordering = os.value;
      if (row.ordering === 'byFacet' && !row.orderFacet)
        row.orderFacet = Object.keys(graph.facets)[0] ?? null;
      if (row.ordering === 'byColumn' && !row.orderColumn)
        row.orderColumn = Object.keys(graph.columns)[0] ?? null;
      edited(true);
    });
    orow.appendChild(os);
    if (row.ordering === 'byFacet') {
      const fs = document.createElement('select');
      fs.title = 'facet whose values order the members (one contiguous sector each)';
      for (const f of Object.keys(graph.facets)) opt(fs, f, f);
      fs.value = row.orderFacet ?? Object.keys(graph.facets)[0] ?? '';
      fs.addEventListener('change', () => {
        row.orderFacet = fs.value;
        edited();
      });
      orow.appendChild(fs);
    } else if (row.ordering === 'byColumn') {
      const cs = document.createElement('select');
      cs.title = 'numeric column whose gradient orders the members';
      for (const col of Object.keys(graph.columns)) opt(cs, col, col);
      cs.value = row.orderColumn ?? Object.keys(graph.columns)[0] ?? '';
      cs.addEventListener('change', () => {
        row.orderColumn = cs.value;
        edited();
      });
      orow.appendChild(cs);
    }

    /* distribution: density + the axis it warps (dim > 1 shapes), then its param pack. */
    const drow = lrow('dens');
    const d = document.createElement('select');
    for (const key of densityKeys()) opt(d, key, key);
    d.value = row.density;
    d.addEventListener('change', () => {
      row.density = d.value;
      row.densP = {}; // the new density's params differ — start at its declared defaults
      edited(true);
    });
    drow.appendChild(d);
    // A per-group row (random / by-facet) has no single shape (so no fixed dim): skip the
    // density-axis select — the density still warps axis 0 on every group, which every shape has.
    const dim = isPerGroupRow(row.place) ? 1 : getShape(placeSpecOf(row.place).shape).dim;
    if (dim > 1) {
      const ax = document.createElement('select');
      ax.title = 'densityAxis: which shape axis the density warps';
      ax.style.flex = '0 0 52px';
      for (let a = 0; a < dim; a++) opt(ax, String(a), `ax${a}`);
      ax.value = String(Math.min(row.densityAxis, dim - 1));
      ax.addEventListener('change', () => {
        row.densityAxis = Number(ax.value);
        edited();
      });
      drow.appendChild(ax);
    }
    if (row.density === 'mixture') mixturePack(row.densP);
    else paramPack('dens·p', densityParams(row.density), row.densP);

    /* scatter: flesh off the manifold + its param pack. */
    const crow = lrow('scat');
    const c = document.createElement('select');
    for (const key of scatterKeys()) opt(c, key, key);
    c.value = row.scatter;
    c.addEventListener('change', () => {
      row.scatter = c.value;
      row.scatP = {};
      edited(true);
    });
    crow.appendChild(c);
    paramPack('scat·p', scatterParams(row.scatter), row.scatP);

    /* spread: how far apart THIS level's own groups sit (seated in their parent's radius) —
       so the galaxy row spreads the galaxies apart, the type row spreads types within each
       galaxy, and the leaf row spreads pages within their deepest group. Drives the parent's
       radius (root radius for level 0), not this level's own — see compileLevels. */
    const prow = lrow('spread');
    const range = document.createElement('input');
    range.type = 'range';
    range.min = '0.25';
    range.max = '10';
    range.step = '0.05';
    range.value = String(row.spread);
    range.title = 'spread: how far apart this level’s groups sit';
    const rv = document.createElement('span');
    rv.className = 'lv-rv';
    rv.textContent = row.spread.toFixed(2);
    range.addEventListener('input', () => {
      row.spread = Number(range.value);
      rv.textContent = row.spread.toFixed(2);
      edited();
    });
    prow.appendChild(range);
    prow.appendChild(rv);

    /* spin: live rate straight into the pivot table — no recompile needed. */
    const wrow = lrow('spin');
    const on = document.createElement('input');
    on.type = 'checkbox';
    on.checked = row.spin;
    on.addEventListener('change', () => {
      row.spin = on.checked;
      ui.hpreset.value = '';
      saveLevels();
      applySpins();
      updateStats();
    });
    const rate = document.createElement('input');
    rate.type = 'number';
    rate.step = '0.05';
    rate.value = String(row.spinRate);
    rate.title = 'spin rate (rad per time unit)';
    rate.addEventListener('input', () => {
      row.spinRate = Number(rate.value) || 0;
      saveLevels();
      applySpins();
    });
    // spin axis: X / Y / Z (the compiler normalises it; default Y = canonical up). Live like
    // the rate — applySpins writes the pivot/field axis straight into the table, no recompile.
    const axisSel = document.createElement('select');
    axisSel.title = 'spin axis (X / Y / Z)';
    axisSel.style.flex = '0 0 46px';
    for (const lbl of ['X', 'Y', 'Z']) opt(axisSel, lbl, lbl);
    axisSel.value = axisLabel(row.spinAxis);
    axisSel.addEventListener('change', () => {
      row.spinAxis = AXES[axisSel.value] ?? [0, 1, 0];
      ui.hpreset.value = '';
      saveLevels();
      applySpins();
    });
    wrow.appendChild(on);
    wrow.appendChild(rate);
    wrow.appendChild(axisSel);

    rowEls[k] = { badge, spreadRow: prow, spinRow: wrow };
    return el;
  };

  const renderLevels = (): void => {
    ui.levels.innerHTML = '';
    rowEls = [];
    rows.forEach((row, k) => ui.levels.appendChild(buildRowEl(row, k)));
    ui.addlevel.disabled = rows.length >= UI_MAX_LEVELS;
    updateLevelStats();
  };

  /* ---- the rail handlers (also driven by the debug handle). */
  const onModeChange = (): void => {
    mode = ui.mode.value === 'custom' ? 'custom' : 'frozen';
    ui.levelsbox.style.display = mode === 'custom' ? '' : 'none';
    saveLevels();
    if (mode === 'custom') {
      renderLevels();
      applyCustomNow(true);
    } else {
      applyFrozen();
    }
  };
  const onPresetChange = (): void => {
    const hp = HPRESETS[ui.hpreset.value];
    if (hp === undefined) return;
    rows = hp.rows();
    saveLevels();
    renderLevels();
    applyCustomNow(true);
  };
  const onAddLevel = (): void => {
    if (rows.length >= UI_MAX_LEVELS) return;
    const facets = Object.keys(graph.facets);
    const used = new Set(rows.map((r) => r.groupBy));
    const pick = facets.find((f) => !used.has(f)) ?? facets[0] ?? V.groupFacet;
    rows.splice(rows.length - 1, 0, mkRow(pick, 'shell'));
    edited(true);
  };
  const onLevelsReset = (ev?: Event): void => {
    ev?.preventDefault();
    localStorage.removeItem(LEVELS_KEY);
    rows = HPRESETS[DEFAULT_HPRESET]!.rows();
    ui.hpreset.value = DEFAULT_HPRESET;
    saveLevels();
    renderLevels();
    if (mode === 'custom') applyCustomNow(true);
  };

  /* the preset-hierarchy select's options (plus the CUSTOM marker it drops to on edit) */
  {
    const custom = document.createElement('option');
    custom.value = '';
    custom.textContent = '· custom ·';
    custom.disabled = true;
    ui.hpreset.appendChild(custom);
    for (const [name, hp] of Object.entries(HPRESETS)) {
      const o = document.createElement('option');
      o.value = name;
      o.textContent = hp.label;
      ui.hpreset.appendChild(o);
    }
  }

  const applyProjection = (): void => {
    const mode = ui.proj.value === '2d' ? '2d' : '3d';
    if (mode !== field.projection) morphing = true;
    field.setProjection(mode);
    const fovTarget = mode === '2d' ? 30 : DEFAULT_FOV;
    appTweens.to('fov', fovTarget, { duration: 0.9 });
    flyHome(fovTarget);
  };

  /* ---- view distance: how far you can see. Pushes the field's depth haze out (setViewDistance)
     AND the camera's far clip plane past it, so distant galaxies stay drawn instead of fading or
     clipping — important once the spread sliders blow the field well past its original extent. */
  const baseCameraFar = (radius + diagonal) * 20;
  const VIEWDIST_KEY = 'starscape-playground:viewdist';
  const applyViewDistance = (persist = true): void => {
    const mult = Math.min(20, Math.max(0.5, Number(ui.viewdist.value) || 1));
    const fog = field.setViewDistance(mult);
    camera.far = Math.max(baseCameraFar, fog.far * 2);
    camera.updateProjectionMatrix();
    ui.viewdistv.textContent = `${mult.toFixed(1)}×`;
    if (persist) {
      try {
        localStorage.setItem(VIEWDIST_KEY, String(mult));
      } catch {
        /* storage full or blocked — the control still works this session */
      }
    }
  };
  {
    const saved = Number(localStorage.getItem(VIEWDIST_KEY));
    if (Number.isFinite(saved) && saved >= 0.5 && saved <= 20) ui.viewdist.value = String(saved);
    applyViewDistance(false);
  }
  ui.viewdist.addEventListener('input', () => {
    applyViewDistance();
    markSceneCustom();
  });

  ui.mode.addEventListener('change', onModeChange);
  ui.hpreset.addEventListener('change', onPresetChange);
  ui.addlevel.addEventListener('click', onAddLevel);
  ui.levelsreset.addEventListener('click', onLevelsReset);
  ui.proj.addEventListener('change', () => {
    applyProjection();
    markSceneCustom();
  });
  ui.home.addEventListener('click', () => flyHome(liveFov));

  /* ---- the keys/help legend: a button in the rail header toggles the overlay; Escape and a
     click anywhere outside close it (it is a reference card, not a mode). */
  const legendEl = document.getElementById('legend')!;
  const keysBtn = document.getElementById('keys') as HTMLButtonElement;
  const toggleLegend = (show?: boolean): void => {
    const on = show ?? legendEl.hasAttribute('hidden');
    if (on) legendEl.removeAttribute('hidden');
    else legendEl.setAttribute('hidden', '');
    keysBtn.classList.toggle('on', on);
  };
  keysBtn.addEventListener('click', () => toggleLegend());
  legendEl.addEventListener('click', () => toggleLegend(false));

  /* ---- the GLOW controls: on/off + the three bloom knobs. Live uniform writes (no recompile);
     the on/off switches the render path. Initialised from the persisted glow state below. */
  ui.glow.addEventListener('change', () => {
    setGlow(ui.glow.checked);
    markSceneCustom();
  });
  ui.glowStrength.addEventListener('input', () => {
    glow.strength = Number(ui.glowStrength.value);
    applyGlowParams();
    saveGlow();
    markSceneCustom();
  });
  ui.glowRadius.addEventListener('input', () => {
    glow.radius = Number(ui.glowRadius.value);
    applyGlowParams();
    saveGlow();
    markSceneCustom();
  });
  ui.glowThreshold.addEventListener('input', () => {
    glow.threshold = Number(ui.glowThreshold.value);
    applyGlowParams();
    saveGlow();
    markSceneCustom();
  });
  /* Compression is a CONFIG key (store-backed, persisted + preset-aware), so its slider drives
     the store like the tuning panel does — marks the look custom, saves, routes (applyCompression
     reflects it back + writes the post uniform) and syncs the tuning panel row. */
  ui.glowCompression.addEventListener('input', () => {
    if (store.set('compression', Number(ui.glowCompression.value))) {
      panel.markCustom();
      store.save();
      routeKey('compression');
      panel.sync();
      syncLook();
      markSceneCustom();
    }
  });

  /* ---- the zoom ladder + overlay, with the REAL wiki provider --------------------------- */
  injectStyles();
  /* closeOrbit (declared up by the camera block) is the node we flew into a tight orbit of
     (double-click), or -1. While set, its pinned reading card is suppressed so it stays a glyph
     you circle; it clears on a new select, on Escape, on a wiki-link navigate, or when the camera
     leaves its leash (handled in the loop). leaveCloseOrbit() centralises that teardown. */
  // the library's deterministic prose provider: a synthetic page body is a pure function of the
  // node id and the same salt everywhere, so an opened card is stable across reloads (§10.1).
  const provider = makeSynthProvider(graph);
  const ladderRoot = document.getElementById('ladder')!;
  const onUnpin = (): void => {
    paintSelected(-1);
    if (field.setSelect(-1)) updateStats();
  };
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
    // the per-tier master switches (Fix 1): off fully suppresses that tier, below any threshold.
    showLabels: bool('showLabels'),
    showCards: bool('showCards'),
    showOpen: bool('showOpen'),
    // card attention: 'center' gates cards to the centre region (radius centerSize); 'cursor'
    // leaves the gate off (Infinity) so the cursor only re-ranks. See gatherCandidates for the
    // matching cursorPx anchor (cursor→node vs centre→node).
    attentionGateRadiusPx:
      str('cardAttention') === 'center' ? num('centerSize') : Number.POSITIVE_INFINITY,
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
    ladderOverlay.dispose();
    ladder = createLadder(ladderOptsFromCfg());
    ladderOverlay = createOverlay({
      container: ladderRoot,
      provider,
      lockW: num('lockW'),
      lockH: num('lockH'),
      onUnpin,
    });
  };

  let pointerActive = false;
  let cursorMoved = false;

  const camInv = new Matrix4();
  const clipM = new Matrix4();

  /* THE CANDIDATE PASS — exhaustive by design at this corpus size (see the header): every
     node, every frame: depth, px gate, frustum margin, cursor pull. The pinned node bypasses
     every gate (a pin outranks the ladder). Columns are fixed scratch. */
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

  function gatherCandidates(): number {
    const fovPx = fovPxOf(H, liveFov);
    const live = !fieldSettledNow;
    const m = clipM.elements;
    // the card-attention anchor: in 'center' mode cursorPx is the CSS-px distance from the CANVAS
    // CENTRE to each node (available for every on-screen node, no pointer needed); in 'cursor' mode
    // it stays the live cursor→node distance (only when the pointer is over the canvas). W/H are CSS
    // px and the node screen position is derived from the same clip matrix, so the centre (W/2, H/2)
    // is in CSS px too — consistent with the cursor path, no devicePixelRatio factor enters.
    const centerMode = str('cardAttention') === 'center';
    // In close orbit the orbited node is deliberately NOT a candidate: it draws no label and no
    // card at any distance, so it stays a bare highlighted glyph you circle (its identity lives
    // in the side panel). Its pin is suppressed too — nothing forces its card open.
    const pinnedNode = closeOrbit >= 0 ? -1 : field.select;
    const cx = camera.position.x;
    const cy = camera.position.y;
    const cz = camera.position.z;
    let k = 0;
    let sawPinned = false;
    for (let i = 0; i < n; i++) {
      if (i === closeOrbit) continue;
      if (live) refreshViewOf(i);
      const o = i * 3;
      const wx = viewBuf[o]!;
      const wy = viewBuf[o + 1]!;
      const wz = viewBuf[o + 2]!;
      const isPin = i === pinnedNode;
      // view-forward depth: clip w ≡ −z_view — the same divisor the node shader sizes by
      const cw = m[3]! * wx + m[7]! * wy + m[11]! * wz + m[15]!;
      if (cw <= 0 && !isPin) continue;
      let dist = cw;
      if (cw <= 0) {
        const dx = wx - cx;
        const dy = wy - cy;
        const dz = wz - cz;
        dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      const ppx = projectedPx(field.sizeOf(i) * NODE_SCALE * field.boostOf(i), dist, fovPx);
      if (ppx < ladder.config.namedPx && !isPin) continue;
      let cur = Infinity;
      if (cw > 0) {
        const ndx = (m[0]! * wx + m[4]! * wy + m[8]! * wz + m[12]!) / cw;
        const ndy = (m[1]! * wx + m[5]! * wy + m[9]! * wz + m[13]!) / cw;
        if (
          (ndx < -NDC_MARGIN_X || ndx > NDC_MARGIN_X || ndy < -NDC_MARGIN_Y || ndy > NDC_MARGIN_Y) &&
          !isPin
        ) {
          continue;
        }
        if (centerMode) {
          // distance from the canvas centre (W/2, H/2) to the node, CSS px:
          //   nodeX − W/2 = (ndx·0.5+0.5)·W − W/2 = ndx·0.5·W, likewise nodeY − H/2 = −ndy·0.5·H.
          const sx = ndx * 0.5 * W;
          const sy = ndy * 0.5 * H;
          cur = Math.hypot(sx, sy);
        } else if (pointerActive) {
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
      if (isPin) sawPinned = true;
      k++;
    }
    if (pinnedNode >= 0 && !sawPinned) {
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
      candPx[k] = 0;
      candDist[k] = cw > 0 ? cw : Math.sqrt(dx * dx + dy * dy + dz * dz);
      candCur[k] = Infinity;
      k++;
    }
    return k;
  }

  const overlayView = {
    matrix: clipM.elements,
    positions: viewBuf,
    width: window.innerWidth,
    height: window.innerHeight,
    pinned: -1,
  };
  let lastCand = 0;

  /* ---- hover: once per frame at most, suppressed mid-morph and mid-flight. -------------- */
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
      tanHalf: Math.tan((liveFov * Math.PI) / 360),
      halfH: H / 2,
      sizeScale: NODE_SCALE,
      minPx: NODE_MIN_PX,
      maxPx: NODE_MAX_PX,
      falloff: FALLOFF,
      dRef,
      sizeOf: (i: number): number => field.sizeOf(i),
      boostOf: (i: number): number => field.boostOf(i),
      drift: gridDrift,
    });
  }

  function hover(i: number): void {
    if (field.setHover(i)) updateStats();
  }

  canvas.addEventListener('pointermove', (ev: PointerEvent): void => {
    px = ev.clientX;
    py = ev.clientY;
    pickDue = true;
    pointerActive = true;
    cursorMoved = true;
  });
  canvas.addEventListener('pointerleave', (): void => {
    pickDue = false;
    pointerActive = false;
    hover(-1);
  });

  /* ======================================================================================
   * SELECTION MODEL (COMPARISON.md F1): a click is an APPROACH. The baseline's rule, ported:
   * selecting a page exists to show what it connects to, so the camera flies in and frames
   * the whole neighbourhood — stand-off from the neighbours' own spread, clamped so a leaf
   * page does not put you inside it and a 400-link hub does not fling you across the vault.
   * ====================================================================================== */

  function viewVecOf(i: number): Vector3 {
    if (!fieldSettledNow && !morphing) refreshViewOf(i);
    return new Vector3(viewBuf[i * 3]!, viewBuf[i * 3 + 1]!, viewBuf[i * 3 + 2]!);
  }

  function focusDistance(i: number): number {
    const p = viewVecOf(i);
    let far = 0;
    const nb = neighbours(graph, i);
    for (let k = 0; k < nb.length; k++) {
      const d = viewVecOf(nb[k]!).distanceTo(p);
      if (d > far) far = d;
    }
    const sp = spacing * field.spread;
    const vfov = (liveFov * Math.PI) / 360;
    const hfov = Math.atan(Math.tan(vfov) * camera.aspect);
    const need = (far / Math.sin(Math.min(vfov, hfov))) * 0.62;
    return Math.min(
      Math.max(need, ladder.config.openSpacings * sp * 1.6),
      ladder.config.cardSpacings * sp * 9,
    );
  }

  /** Deselecting from close up RETREATS past the card threshold (the demo's rule: a closed
   *  page must shrink into the point it came from, or the next cursor move reopens it). */
  function retreatFrom(was: number): void {
    if (!fly || was < 0) return;
    const p = viewVecOf(was);
    const dCard = ladder.config.cardSpacings * spacing * field.spread;
    if (camera.position.distanceTo(p) < dCard * 1.05) fly.goTo(p, dCard * 1.7, 800);
  }

  function selectNode(i: number): void {
    const was = field.select;
    leaveCloseOrbit(); // a plain (single-click) select leaves close-orbit mode: pin + read as usual
    field.setSelect(i);
    paintSelected(i);
    if (i >= 0) {
      const t = viewVecOf(i);
      // the node becomes the orbit target, so drift turns the neighbourhood around it
      cam.setTarget(t, false);
      if (fly) fly.goTo(t, focusDistance(i), 950);
    } else if (was >= 0) {
      retreatFrom(was);
    }
    updateStats();
  }

  /** DOUBLE-CLICK: fly into a TIGHT orbit of a SINGLE node — close enough that the node fills a
   *  good part of the view, not the whole neighbourhood (that is single-click's focusDistance).
   *  The node becomes the orbit pivot (cam.setTarget), so dragging circles it and Space/turntable
   *  auto-orbits it; the wheel then dollies even closer (the lowered zoomRange floor). It stays
   *  SELECTED (highlight + side panel) but its reading card is kept unpinned/suspended — see the
   *  closeOrbit handling in the render loop — so the node you are circling is never obscured. */
  function flyCloseOrbit(i: number): void {
    if (!fly || i < 0) return;
    closeOrbit = i;
    // No node-clamp lift any more: the px cap is OFF globally (library default uncapped), so the
    // node already grows without bound as you wheel in. Double-click only re-targets the orbit
    // onto this node (so dragging circles it) and suspends its card — growth is no longer tied to
    // orbit (Fix 3).
    field.setSelect(i);
    paintSelected(i);
    ladder.retreat(); // drop any card open right now; the loop holds it suspended while orbiting
    const t = viewVecOf(i);
    cam.setTarget(t, false);
    // Land close (the dolly floor is now nose-to-node, so the wheel takes it the rest of the way
    // to a full-frame fill). zoomRange() already reflects closeOrbit — floor is tiny here.
    const tightRadius = Math.max(spacing * field.spread * 0.9, zoomRange()[0] * 1.2);
    fly.goTo(t, tightRadius, 700);
    updateStats();
  }

  /* ---- select (capture phase — see the demo's note on orbit's bubble-phase pointerup). -- */
  window.addEventListener(
    'pointerup',
    (ev: PointerEvent): void => {
      const wasClick = cam.consumeClick(ev);
      if (!wasClick || ev.target !== canvas) return;
      if (morphing) return;
      selectNode(castAt(ev.clientX, ev.clientY));
    },
    true,
  );

  /* DOUBLE-CLICK a node → fly into a TIGHT orbit of just that node (flyCloseOrbit). This is a
     window-level listener (capture) that resolves the node with castAt from the pointer, not from
     the DOM target, so it fires over the canvas AND over a reading card sitting on the node —
     "read a page, then double-click it to circle it". The single-click above still fires first
     (select + neighbourhood framing); the double-click REFINES it, redirecting the in-flight goTo
     to the close orbit and dropping the card. Clicks on the control UI (panel, side panel, config,
     stats) and on wiki-links are left alone. */
  window.addEventListener(
    'dblclick',
    (ev: MouseEvent): void => {
      if (morphing) return;
      const t = ev.target as HTMLElement | null;
      if (
        t &&
        (t.closest('#panel') ||
          t.closest('.ss-cfg') ||
          t.closest('#selpanel') ||
          t.closest('#stats') ||
          t.closest('#error') ||
          t.closest('a'))
      ) {
        return;
      }
      const i = castAt(ev.clientX, ev.clientY);
      if (i >= 0) flyCloseOrbit(i);
    },
    true,
  );

  window.addEventListener('keydown', (ev: KeyboardEvent): void => {
    const typing = (): boolean => {
      const tag = (document.activeElement as HTMLElement | null)?.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
    };
    if (ev.key === 'Escape') {
      const cm = document.getElementById('codemodal');
      if (cm && !cm.hasAttribute('hidden')) {
        cm.setAttribute('hidden', ''); // Escape closes the export-code modal first
        return;
      }
      if (!legendEl.hasAttribute('hidden')) {
        toggleLegend(false); // then the keys legend
        return;
      }
      // close + retreat + unpin (retreatFrom carries the demo's shrink-back rule)
      leaveCloseOrbit(); // Escape also leaves close-orbit mode (and restores the node clamp)
      const was = ladder.state.open >= 0 ? ladder.state.open : field.select;
      ladder.retreat();
      paintSelected(-1);
      if (field.setSelect(-1)) updateStats();
      retreatFrom(was);
    }
    // '/' focuses the node search, as its placeholder promises (never while typing)
    if (ev.key === '/' && !typing()) {
      ev.preventDefault();
      ui.search.focus();
    }
    // '~' (backquote) toggles the full schema tuning panel, as the button title promises
    if ((ev.key === '~' || ev.key === '`') && !typing()) {
      ev.preventDefault();
      panel.toggle();
    }
  });

  /* ======================================================================================
   * THE SELECTED-PAGE PANEL (COMPARISON.md F5): what this page connects to, as clickable
   * chips — the panel the open body's wiki-links complement, not replace.
   * ====================================================================================== */

  const escHtml = (s: unknown): string =>
    String(s ?? '').replace(
      /[&<>"]/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!,
    );
  const layerCss = (li: number): string => {
    const c = V.layerColorsByName[V.layerNames[li] ?? ''] ?? [0.6, 0.62, 0.66];
    return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
  };

  function paintSelected(i: number): void {
    const p = ui.selpanel;
    if (i < 0) {
      p.classList.remove('on');
      p.innerHTML = '';
      return;
    }
    const li = typeFacet.data[i]!;
    const deg = graph.columns['degree']!.data[i] ?? 0;
    const regionName = V.regions[graph.facets[V.regionFacet]!.data[i]!]?.name ?? '—';
    const nb = neighbours(graph, i);
    const chips: string[] = [];
    const cap = Math.min(nb.length, 24);
    for (let k = 0; k < cap; k++) {
      const j = nb[k]!;
      chips.push(
        `<button class="chip" data-i="${j}"><i style="background:${layerCss(
          typeFacet.data[j]!,
        )}"></i>${escHtml(graph.labelOf(j))}</button>`,
      );
    }
    p.innerHTML =
      `<div class="s-h"><i style="background:${layerCss(li)}"></i>${escHtml(graph.labelOf(i))}</div>` +
      `<div class="s-m">${escHtml(V.layerNames[li])} · ${deg} links · region “${escHtml(regionName)}”</div>` +
      (V.summaries[i] ? `<div class="s-s">${escHtml(V.summaries[i])}</div>` : '') +
      `<div class="s-n">${chips.join('')}</div>`;
    p.classList.add('on');
    p.querySelectorAll<HTMLButtonElement>('.chip').forEach((b) => {
      b.addEventListener('click', (ev) => {
        ev.stopPropagation();
        selectNode(Number(b.dataset['i']));
      });
    });
  }

  /* ======================================================================================
   * PAGE SEARCH (COMPARISON.md F2): the baseline's "find a page /" — substring match over
   * the real labels, hits sorted by degree, a click selects and flies the neighbourhood in.
   * ====================================================================================== */

  // graph.labelOf either way (vault labels are an array behind it, synth labels a closure —
  // never 10⁶ eager lowered strings; the scan early-exits at 12 hits).
  ui.search.addEventListener('input', () => {
    const v = ui.search.value.trim().toLowerCase();
    ui.hits.innerHTML = '';
    if (v.length < 2) {
      ui.hits.classList.remove('on');
      return;
    }
    const deg = graph.columns['degree']!.data;
    const out: number[] = [];
    for (let i = 0; i < n && out.length < 12; i++) {
      if (graph.labelOf(i).toLowerCase().includes(v)) out.push(i);
    }
    out.sort((a, b) => (deg[b] ?? 0) - (deg[a] ?? 0));
    for (const i of out) {
      const b = document.createElement('button');
      b.innerHTML =
        `<i class="dot" style="background:${layerCss(typeFacet.data[i]!)}"></i>` +
        `<span>${escHtml(graph.labelOf(i))}</span><b>${deg[i] ?? 0}</b>`;
      b.addEventListener('click', () => {
        ui.hits.classList.remove('on');
        ui.hits.innerHTML = '';
        ui.search.value = '';
        ui.search.blur();
        selectNode(i);
      });
      ui.hits.appendChild(b);
    }
    ui.hits.classList.toggle('on', out.length > 0);
  });
  ui.search.addEventListener('keydown', (ev) => {
    ev.stopPropagation(); // Escape here clears the search, never closes an open page
    if (ev.key === 'Escape') {
      ui.search.value = '';
      ui.hits.classList.remove('on');
      ui.hits.innerHTML = '';
      ui.search.blur();
    }
    if (ev.key === 'Enter') {
      (ui.hits.querySelector('button') as HTMLButtonElement | null)?.click();
    }
  });

  /* ======================================================================================
   * REGIONS (COMPARISON.md F3): named labels over the field — visible from outside a
   * region, gone once you are inside and its pages name themselves — plus the fly-to list.
   * Region centres: meta3's own measured centres in the frozen layout; the level-0 pivots
   * when a hierarchy groups by region; NOTHING in ringGlobal/force, where a region has no
   * place (the labels and the list hide rather than point at fiction).
   * ====================================================================================== */

  interface RegionView {
    x: number;
    y: number;
    z: number;
    r: number;
  }
  let regionViews: RegionView[] = [];
  let regionsValid = false;

  const regionEls = V.regions.map((r) => {
    const el = document.createElement('div');
    el.className = 'rl';
    el.innerHTML = `<b>${escHtml(r.name)}</b><i>${r.count} nodes</i>`;
    ui.regionLabels.appendChild(el);
    return el;
  });

  function computeRegionViews(): void {
    if (hierarchy === null) {
      regionViews = V.regions.map((r) => ({
        x: r.centre[0],
        y: r.centre[1],
        z: r.centre[2],
        r: r.radius,
      }));
      regionsValid = true;
    } else if (regionHierarchy) {
      // level-0 pivot per region value, found by walking one member node's pivot chain
      const pv = hierarchy.pivots;
      const regData = graph.facets[V.regionFacet]!.data;
      const views: (RegionView | null)[] = V.regions.map(() => null);
      let missing = views.length;
      for (let i = 0; i < n && missing > 0; i++) {
        const v = regData[i]!;
        if (views[v]) continue;
        let p = hierarchy.nodePivot[i]!;
        while (pv.level[p]! > 0) p = pv.parent[p]!;
        views[v] = {
          x: pv.world[p * 3]!,
          y: pv.world[p * 3 + 1]!,
          z: pv.world[p * 3 + 2]!,
          r: pv.radius[p]!,
        };
        missing--;
      }
      regionViews = views.map((w) => w ?? { x: 0, y: 0, z: 0, r: 0 });
      regionsValid = true;
    } else {
      regionsValid = false;
    }
    ui.regionsBox.style.display = regionsValid ? '' : 'none';
    if (!regionsValid) for (const el of regionEls) el.style.opacity = '0';
  }

  function flyToRegion(k: number): void {
    if (!regionsValid || !fly) return;
    const rv = regionViews[k]!;
    const C = field.centroid();
    const s = field.spread;
    const t = new Vector3(
      C[0] + (rv.x - C[0]) * s,
      C[1] + (rv.y - C[1]) * s,
      C[2] + (rv.z - C[2]) * s,
    );
    fly.goTo(t, Math.max(rv.r * s * 2.3, spacing * s * 6), 1100);
  }

  V.regions.forEach((r, k) => {
    const b = document.createElement('button');
    b.className = 'reg';
    b.innerHTML = `<i style="background:${layerCss(r.layer)}"></i><span>${escHtml(r.name)}</span><b>${r.count}</b>`;
    b.addEventListener('click', () => flyToRegion(k));
    ui.regionList.appendChild(b);
  });
  computeRegionViews();

  /** Projected-span fades, the baseline's constants: in at REGION_IN px of projected region
   *  diameter, gone by REGION_OUT (by then its pages label themselves); overlapping labels
   *  cull first-come. Hidden mid-morph — centres describe the TARGET arrangement. */
  const REGION_IN = 55;
  const REGION_OUT = 1500;
  const regionBoxes: { x: number; y: number; w: number; h: number }[] = [];
  function updateRegionLabels(): void {
    if (!regionsValid || morphing) {
      if (regionsValid) for (const el of regionEls) el.style.opacity = '0';
      return;
    }
    const m = clipM.elements;
    const C = field.centroid();
    const s = field.spread;
    const tanHalf = Math.tan((liveFov * Math.PI) / 360);
    regionBoxes.length = 0;
    for (let k = 0; k < regionEls.length; k++) {
      const el = regionEls[k]!;
      const rv = regionViews[k]!;
      const wx = C[0] + (rv.x - C[0]) * s;
      const wy = C[1] + (rv.y - C[1]) * s;
      const wz = C[2] + (rv.z - C[2]) * s;
      const cw = m[3]! * wx + m[7]! * wy + m[11]! * wz + m[15]!;
      if (cw <= 0) {
        el.style.opacity = '0';
        continue;
      }
      const ndx = (m[0]! * wx + m[4]! * wy + m[8]! * wz + m[12]!) / cw;
      const ndy = (m[1]! * wx + m[5]! * wy + m[9]! * wz + m[13]!) / cw;
      if (ndx < -1.1 || ndx > 1.1 || ndy < -1.15 || ndy > 1.15) {
        el.style.opacity = '0';
        continue;
      }
      const span = (2 * rv.r * s * (H / 2)) / (tanHalf * Math.max(cw, 1));
      const RG_FULL = REGION_IN * 2.7;
      const a = Math.min(
        (span - REGION_IN) / Math.max(RG_FULL - REGION_IN, 1),
        Math.max(0, 1 - (span - REGION_OUT) / REGION_OUT),
      );
      if (a <= 0.02) {
        el.style.opacity = '0';
        continue;
      }
      const x = (ndx * 0.5 + 0.5) * W;
      const y = (-ndy * 0.5 + 0.5) * H;
      const ew = el.offsetWidth || 120;
      const eh = el.offsetHeight || 26;
      const bx = x - ew / 2;
      const by = y - eh / 2;
      let hit = false;
      for (const bb of regionBoxes) {
        if (bx < bb.x + bb.w && bx + ew > bb.x && by < bb.y + bb.h && by + eh > bb.y) {
          hit = true;
          break;
        }
      }
      if (hit) {
        el.style.opacity = '0';
        continue;
      }
      regionBoxes.push({ x: bx, y: by, w: ew, h: eh });
      el.style.transform = `translate3d(${Math.round(bx)}px,${Math.round(by)}px,0)`;
      el.style.opacity = String(Math.min(1, a));
    }
  }

  window.addEventListener('resize', () => {
    W = window.innerWidth;
    H = window.innerHeight;
    renderer.setSize(W, H);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    field.resize(H);
    post.setSize(W, H); // the HDR/bloom targets track the renderer; this lands it same-frame
    nearMesh?.setView(H, liveFov); // the near-mesh reasons in CSS px too (projected-size promotion)
  });

  /* ======================================================================================
   * 3D NODES — the OPT-IN near-mesh LOD (library createNearMesh). Off by default: NOTHING is
   * instantiated until the toggle is first switched on, so the default playground and the 10⁶
   * path pay exactly zero. When on, nodes large on screen become genuine shaded / textured 3D
   * meshes (fly up to one, or double-click to orbit it) while the field stays billboards — the
   * two coexist, billboards fading as the mesh takes over. It cooperates with close-orbit for
   * free: flying into a node makes it big on screen, which is exactly what promotes it.
   * ====================================================================================== */
  let nearMesh: NearMesh | null = null;
  {
    // populate the facet pickers from the live graph ('none' = default sphere / flat, no texture).
    const facets = Object.keys(graph.facets);
    for (const sel of [ui.nearShape, ui.nearTex]) {
      const none = document.createElement('option');
      none.value = 'none';
      none.textContent = 'none';
      sel.appendChild(none);
      for (const f of facets) {
        const o = document.createElement('option');
        o.value = f;
        o.textContent = f;
        sel.appendChild(o);
      }
      sel.value = 'none';
    }
    // default shape-by 'type' (7 values → 7 shapes) so the first toggle-on already shows variety.
    if (facets.includes('type')) ui.nearShape.value = 'type';
  }
  const nearShapeFacet = (): string | null =>
    ui.nearShape.value === 'none' ? null : ui.nearShape.value;
  const nearTexFacet = (): string | null => (ui.nearTex.value === 'none' ? null : ui.nearTex.value);
  const glyphMode = (): GlyphTextureMode =>
    ui.glyphTex.value === 'tint' ? 'tint' : ui.glyphTex.value === 'full' ? 'full' : 'off';

  /* ---- the SHARED planet texture array (DESIGN §12 "mote → mesh"): built ONCE, from the same
     seed the LOD defaults to, and handed to BOTH the near-mesh LOD (createNearMesh `textures`) AND
     the FAR billboards (field.setGlyphTexture) so the two are byte-identical — a node's billboard
     and the mesh it promotes to sample the same layer of the same array. Generated lazily, only
     when something actually needs a texture (the LOD's texture-by facet is set, or the far-texture
     mode is tint/full), so the default playground still pays zero. The per-layer averages (linear
     dominant colour) drive the cheap 'tint' mode. */
  const TEX_SEED = 1; // must match createNearMesh's default textureSeed so contents are identical
  let sharedRaw: Uint8Array | null = null;
  let sharedArray: ReturnType<typeof wrapNearTextures> | null = null;
  let sharedAvg: Float32Array | null = null;
  const ensureSharedTex = (): { array: ReturnType<typeof wrapNearTextures>; avg: Float32Array } => {
    if (!sharedRaw) sharedRaw = generateNearTextures(TEX_SEED);
    if (!sharedArray) sharedArray = wrapNearTextures(sharedRaw);
    if (!sharedAvg) sharedAvg = nearTextureAverages(sharedRaw);
    return { array: sharedArray, avg: sharedAvg };
  };
  const needShared = (): boolean => nearTexFacet() !== null || glyphMode() !== 'off';

  /** The per-node texture LAYER buffer for the billboards — node i's layer is its 'texture by'
   *  facet value mod the texture count, the SAME map the LOD uses internally (`texCodes[i] %
   *  texCount`), so the faded billboard and the promoted mesh always agree. No facet → layer 0. */
  const glyphLayers = (): Float32Array => {
    const out = new Float32Array(n);
    const facet = nearTexFacet();
    const codes = facet && graph.facets[facet] ? graph.facets[facet]!.data : null;
    if (codes) {
      const texCount = NEAR_TEXTURE_NAMES.length;
      for (let i = 0; i < n; i++) out[i] = codes[i]! % texCount;
    }
    return out;
  };

  /** Push the current far-texture mode (+ layers from the live 'texture by' facet) to the field.
   *  Off parks it (null); tint/full ensure the shared array and hand it over. Called on either
   *  select's change. */
  const applyGlyphTexture = (): void => {
    const mode = glyphMode();
    if (mode === 'off') {
      field.setGlyphTexture(null);
      return;
    }
    const shared = ensureSharedTex();
    field.setGlyphTexture({
      array: shared.array,
      layerCount: NEAR_TEXTURE_NAMES.length,
      layers: glyphLayers(),
      avg: shared.avg,
      mode,
    });
  };

  /* ---- IMAGE-PLANET showcase (library per-node texture overrides + progressive tiers) ----------
   * Pin ONE node to a real IMAGE texture instead of a procedural planet. The library serves it in
   * resolution TIERS: it loads the sharpest tier that fits the node's on-screen px (async, never
   * blocking), shows the best already-loaded tier meanwhile, swaps the material's map when a sharper
   * tier resolves, and releases high tiers (LRU, ~64 MB) when you fly away. Procedural neighbours are
   * untouched — they stay in the shared instanced pool (the millions-scale path).
   *
   * Here each tier is generated IN-MEMORY on a 2:1 equirect canvas (ocean + hashed continents + a big
   * resolution LABEL), resolving after a short delay to exercise async + progressive + swap + cache
   * with NO external photo. The label makes the low→high swap legible; the console logs load/swap/evict.
   *
   * SWAP IN A REAL EARTH IN ONE LINE: drop a same-origin assets/earth-4k.jpg (CSP-clean) and use
   * realEarthTiers() below instead of demoEarthTiers() — one TextureLoader().loadAsync per tier. */
  const drawEarthTier = (px: number): HTMLCanvasElement => {
    const w = px;
    const h = Math.max(1, px >> 1); // 2:1 equirectangular (sphere uv maps full [0,1]²)
    const cv = document.createElement('canvas');
    cv.width = w;
    cv.height = h;
    const ctx = cv.getContext('2d')!;
    const oc = ctx.createLinearGradient(0, 0, 0, h); // ocean
    oc.addColorStop(0, '#0a2f63');
    oc.addColorStop(0.5, '#0e63a6');
    oc.addColorStop(1, '#0a2f63');
    ctx.fillStyle = oc;
    ctx.fillRect(0, 0, w, h);
    // hashed "continents" sampled at the tier's resolution: a higher tier resolves finer coastlines
    // (visibly sharper). Deterministic via the lib `hash` — field geometry is untouched.
    const cells = 7;
    const land = (u: number, v: number): number => {
      const x = u * cells;
      const y = v * cells;
      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const xf = x - x0;
      const yf = y - y0;
      const sx = xf * xf * (3 - 2 * xf);
      const sy = yf * yf * (3 - 2 * yf);
      const hv = (ix: number, iy: number): number =>
        hash((((ix % cells) + cells) % cells) + iy * 131, 20260);
      const a = hv(x0, y0);
      const b = hv(x0 + 1, y0);
      const c = hv(x0, y0 + 1);
      const d = hv(x0 + 1, y0 + 1);
      return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
    };
    const img = ctx.getImageData(0, 0, w, h);
    const data = img.data;
    for (let py = 0; py < h; py++) {
      const v = (py + 0.5) / h;
      const lat = Math.abs(v - 0.5) * 2;
      for (let pxi = 0; pxi < w; pxi++) {
        const l = land((pxi + 0.5) / w, v);
        const o = (py * w + pxi) * 4;
        if (l > 0.56) {
          const t = Math.min(1, (l - 0.56) / 0.3);
          data[o] = 70 + t * 90;
          data[o + 1] = 120 + t * 40;
          data[o + 2] = 55 + t * 30;
        }
        if (lat > 0.86) {
          const t = Math.min(1, (lat - 0.86) / 0.1); // ice caps
          data[o] = data[o]! + (235 - data[o]!) * t;
          data[o + 1] = data[o + 1]! + (240 - data[o + 1]!) * t;
          data[o + 2] = data[o + 2]! + (245 - data[o + 2]!) * t;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'; // graticule
    ctx.lineWidth = Math.max(1, px / 1024);
    for (let gx = 1; gx < 8; gx++) {
      const x = (gx / 8) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let gy = 1; gy < 4; gy++) {
      const y = (gy / 4) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.fillStyle = 'rgba(255,255,255,0.85)'; // a legible resolution label (which tier is showing)
    ctx.font = `bold ${Math.round(h * 0.22)}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${px}px`, w * 0.5, h * 0.5);
    return cv;
  };
  const makeEarthTier = (px: number, delayMs: number): ImageTier => ({
    px, // doubles as the on-screen threshold: loaded when the node grows past ~px on screen
    load: () =>
      new Promise<Texture>((resolve) => {
        window.setTimeout(() => resolve(new CanvasTexture(drawEarthTier(px))), delayMs);
      }),
  });
  const demoEarthTiers = (): ImageTier[] => [
    makeEarthTier(160, 120), // base: loads first, soft
    makeEarthTier(420, 170),
    makeEarthTier(900, 240),
    makeEarthTier(1600, 340), // sharpest: only when the planet nearly fills the view
  ];
  // DROP-IN REAL EARTH (same-origin, CSP-clean): drop assets/earth-4k.jpg next to the demo and swap
  // demoEarthTiers() → realEarthTiers() in imageOverrides() below. One TextureLoader call per tier:
  //   const realEarthTiers = (): ImageTier[] => [
  //     { px: 1024, load: () => new TextureLoader().loadAsync('assets/earth-2k.jpg') },
  //     { px: 4096, load: () => new TextureLoader().loadAsync('assets/earth-4k.jpg') },
  //   ];

  let pinnedImageNode = -1; // the node pinned to an image planet (-1 = none)
  const imageOverrides = (): Record<number, NodeSource> | undefined =>
    pinnedImageNode >= 0
      ? { [pinnedImageNode]: { kind: 'image', tiers: demoEarthTiers() } }
      : undefined;
  const onImageEvent = (e: ImageEvent): void => {
    const mb = (e.cacheBytes / (1024 * 1024)).toFixed(1);
    console.debug(`[near-image] ${e.type} node ${e.node} tier@${e.px}px · cache ${mb} MB`);
  };

  /* ---- 3D-NODES as a first-class part of the SCENE (mirrors GLOW/VIEWDIST/LEVELS exactly). The
     group is demo-side UI state governed by the top-level scene preset: a scene pick sets the whole
     near config (applyScene → applyNear3d), editing any near control diverges the scene to 'custom'
     (onNearEdit → markSceneCustom), and the group persists under its OWN key — the parallel of
     LEVELS_KEY/GLOW_KEY/VIEWDIST_KEY — restored early (before the scene boot) so a preset boot can
     override it and a 'custom' boot keeps it. So it survives a reload AND a dataset-size switch.
     The pinned node id is stored raw: synth ids are stable for a given size/seed, so it re-pins
     after a reload, and is dropped on a size switch when the id no longer exists (guard in apply). */
  const NEAR3D_KEY = 'starscape-playground:near3d:v1';
  interface Near3dState {
    near: boolean;
    shape: string;
    tex: string;
    glyphTex: string;
    surfaceDetail: string;
    terrain: boolean;
    promote: number;
    count: number;
    pinned: number; // the pinned image-planet node id (-1 = none)
  }
  /* The canonical OFF default — the shipped first-run state (index.html values + the init block's
     shape-by 'type'). Both built-in scenes (space4, default) use it: picking either turns the LOD
     off, exactly as a clean scene should. A future scene is free to ship its own near spec here. */
  const NEAR3D_DEFAULT: Near3dState = {
    near: false,
    shape: Object.keys(graph.facets).includes('type') ? 'type' : 'none',
    tex: 'none',
    glyphTex: 'off',
    surfaceDetail: 'baked',
    terrain: false,
    promote: 18,
    count: 128,
    pinned: -1,
  };
  const nearSpecOf = (_scene: string): Near3dState => NEAR3D_DEFAULT; // both presets ⇒ off
  const saveNear3d = (): void => {
    try {
      const state: Near3dState = {
        near: ui.near.checked,
        shape: ui.nearShape.value,
        tex: ui.nearTex.value,
        glyphTex: ui.glyphTex.value,
        surfaceDetail: ui.surfaceDetail.value,
        terrain: ui.nearTerrain.checked,
        promote: Number(ui.nearPromote.value),
        count: Number(ui.nearCount.value),
        pinned: pinnedImageNode,
      };
      localStorage.setItem(NEAR3D_KEY, JSON.stringify(state));
    } catch {
      /* storage denied/full — the controls still work this session, they just will not persist */
    }
  };
  /* The single reaction every near-control edit fires: persist the group under its own key AND
     diverge the scene to 'custom' — the same two-step the look/glow/view/hierarchy edits run
     (own save + markSceneCustom). markSceneCustom is guarded by applyingScene, so the programmatic
     applyNear3d (from applyScene) never trips it. */
  const onNearEdit = (): void => {
    saveNear3d();
    markSceneCustom();
  };

  const ensureNear = (): NearMesh => {
    if (nearMesh) return nearMesh;
    nearMesh = createNearMesh(graph, field, {
      count: Number(ui.nearCount.value) | 0,
      promotePx: Number(ui.nearPromote.value),
      shapeFacet: nearShapeFacet(),
      textureFacet: nearTexFacet(),
      proceduralDetail: ui.surfaceDetail.value === 'procedural',
      terrain: ui.nearTerrain.checked,
      // share the ONE array with the billboards whenever either side needs textures (so the LOD
      // and the far billboards sample the identical instance, not two deterministic copies).
      textures: needShared() ? ensureSharedTex().array : undefined,
      textureSeed: TEX_SEED,
      heightCss: window.innerHeight,
      fov: liveFov,
      overrides: imageOverrides(), // undefined unless a node is pinned ⇒ zero image machinery
      imageCacheBudget:
        Number.isFinite(imgBudgetMB) && imgBudgetMB > 0 ? imgBudgetMB * 1024 * 1024 : undefined,
      onImageEvent,
    });
    for (const o of nearMesh.objects) scene.add(o);
    return nearMesh;
  };
  const disposeNear = (): void => {
    if (!nearMesh) return;
    for (const o of nearMesh.objects) scene.remove(o);
    nearMesh.dispose();
    nearMesh = null;
  };
  ui.near.addEventListener('change', () => {
    const on = ui.near.checked;
    ui.nearbox.classList.toggle('on', on);
    if (on) ensureNear().setEnabled(true);
    else nearMesh?.setEnabled(false);
    updateStats();
    onNearEdit();
  });
  ui.nearShape.addEventListener('change', () => {
    nearMesh?.setShapeFacet(nearShapeFacet());
    onNearEdit();
  });
  ui.nearTex.addEventListener('change', () => {
    nearMesh?.setTextureFacet(nearTexFacet());
    // the far billboards key their planet off the SAME facet, so a facet change relayers them too
    if (glyphMode() !== 'off') applyGlyphTexture();
    onNearEdit();
  });
  // FAR TEXTURE: off (default) leaves the billboards solid facet colour, byte-identical to today;
  // tint/full texture them to match the near planets, killing the promote-time colour→texture pop.
  ui.glyphTex.addEventListener('change', () => {
    applyGlyphTexture();
    updateStats();
    onNearEdit();
  });
  // SURFACE DETAIL: baked = today's shader (zero cost); procedural = the hard-edge planets (earth,
  // ice, lava, moon) resolve infinite close-up detail in the fragment shader, seamlessly blended from
  // baked, with their poles guarded against the equirect pinch.
  ui.surfaceDetail.addEventListener('change', () => {
    nearMesh?.setProceduralDetail(ui.surfaceDetail.value === 'procedural');
    onNearEdit();
  });
  /* DISPLACED TERRAIN prereqs: terrain needs a texture facet (so nodes have a planet layer — earth is
   * layer 0) AND procedural surface detail (the terrain reuses the earth colour, seamless at the swap).
   * Turning terrain on forces both; the terrain subsystem is created lazily on the first close approach. */
  const ensureTerrainPrereqs = (): void => {
    if (ui.nearTex.value === 'none') {
      const facets = Object.keys(graph.facets);
      const pick = facets.includes('type') ? 'type' : (facets[0] ?? 'none');
      if (pick !== 'none') {
        ui.nearTex.value = pick;
        nearMesh?.setTextureFacet(pick);
        if (glyphMode() !== 'off') applyGlyphTexture();
      }
    }
    if (ui.surfaceDetail.value !== 'procedural') {
      ui.surfaceDetail.value = 'procedural';
      nearMesh?.setProceduralDetail(true);
    }
  };
  ui.nearTerrain.addEventListener('change', () => {
    const on = ui.nearTerrain.checked;
    if (on) ensureTerrainPrereqs();
    nearMesh?.setTerrain(on);
    updateStats();
    // ensureTerrainPrereqs may have forced the tex facet + surface detail — save the group after.
    onNearEdit();
  });
  ui.nearPromote.addEventListener('input', () => {
    ui.nearPromotev.textContent = `${ui.nearPromote.value} px`;
    nearMesh?.setPromotePx(Number(ui.nearPromote.value));
    onNearEdit();
  });
  ui.nearCount.addEventListener('input', () => {
    ui.nearCountv.textContent = ui.nearCount.value;
  });
  ui.nearCount.addEventListener('change', () => {
    // count sizes the per-pool buffers, so a change re-creates the layer (preserving on/off + pickers).
    onNearEdit();
    if (!nearMesh) return;
    const wasOn = ui.near.checked;
    disposeNear();
    if (wasOn) ensureNear().setEnabled(true);
  });

  /* PIN AS IMAGE PLANET: pin the selected node (or a fixed showcase node if none is selected) to a
   *  tiered image source via the library's setOverrides. Turns 3D nodes on if needed and flies onto
   *  the node so it promotes and progresses through the tiers. Off clears the override (the node
   *  returns to its procedural planet). */
  const pinImageNode = (i: number): void => {
    pinnedImageNode = i;
    ui.nearImg.checked = i >= 0;
    if (i >= 0 && !ui.near.checked) {
      ui.near.checked = true;
      ui.nearbox.classList.add('on');
      ensureNear().setEnabled(true);
    }
    // setOverrides drives it live when the layer exists; otherwise the next ensureNear picks it up.
    nearMesh?.setOverrides(imageOverrides() ?? null);
    if (i >= 0) {
      field.setSelect(i);
      const handle = (window as unknown as Record<string, Record<string, (a: number, b?: number) => void>>)[
        '__starscape'
      ];
      handle?.['flyToNode']?.(i, 0.35);
    }
    updateStats();
    // pinImageNode is the single mutation point for the pin (and it may auto-enable near) — persist
    // + diverge the scene to 'custom', like every other near edit.
    onNearEdit();
  };
  ui.nearImg.addEventListener('change', () => {
    if (ui.nearImg.checked) pinImageNode(field.select >= 0 ? field.select : Math.floor(n / 2));
    else pinImageNode(-1);
  });

  /* ---- applyNear3d: route a WHOLE near-mesh spec through the controls + subsystems WITHOUT firing
     change events, so it never trips the onNearEdit save/divergence path. The ONE apply used by both
     the scene preset (applyScene) and the boot restore. ORDER matters — ensureNear bakes count +
     facets + terrain + pin at creation, so we (1) set every control VALUE, (2) reconcile terrain's
     prereqs, (3) push the far-texture (field-side), then (4) rebuild or tear down the mesh so the
     LIVE effect matches the spec: near on ⇒ a fresh mesh adopts the new count/facets/pin, near off
     ⇒ fully disposed (back to zero cost, the point of a clean scene). Effects apply with no user
     action. The pinned id is best-effort — kept only if that id still exists at this size/seed. */
  const applyNear3d = (spec: Near3dState): void => {
    const setSelect = (el: HTMLSelectElement, v: string): void => {
      if (Array.from(el.options).some((o) => o.value === v)) el.value = v;
    };
    const setRange = (el: HTMLInputElement, v: number): void => {
      if (!Number.isFinite(v)) return;
      el.value = String(Math.min(Number(el.max), Math.max(Number(el.min), v)));
    };
    // (1) control VALUES — selects (option still-present guarded) and ranges (clamped to the slider).
    setSelect(ui.nearShape, spec.shape);
    setSelect(ui.nearTex, spec.tex);
    setSelect(ui.glyphTex, spec.glyphTex);
    setSelect(ui.surfaceDetail, spec.surfaceDetail);
    setRange(ui.nearPromote, spec.promote);
    setRange(ui.nearCount, spec.count);
    ui.nearPromotev.textContent = `${ui.nearPromote.value} px`;
    ui.nearCountv.textContent = ui.nearCount.value;
    ui.nearTerrain.checked = spec.terrain === true;
    pinnedImageNode = spec.pinned >= 0 && spec.pinned < n ? spec.pinned : -1;
    ui.nearImg.checked = pinnedImageNode >= 0;
    // (2) terrain prereqs: reconcile the ui (tex facet + procedural detail) BEFORE the mesh reads it.
    if (ui.nearTerrain.checked) ensureTerrainPrereqs();
    // (3) far texture is field-side (independent of the mesh): off ⇒ null (zero cost), tint/full push.
    applyGlyphTexture();
    // (4) (re)build or tear down the mesh so the live state matches — a fresh build adopts the spec.
    const on = spec.near === true;
    ui.near.checked = on;
    ui.nearbox.classList.toggle('on', on);
    disposeNear(); // drop any stale mesh (null-safe); off ⇒ stays torn down
    if (on) ensureNear().setEnabled(true);
    updateStats();
  };

  /* ---- 3D-NODES boot restore: re-apply the group's OWN persisted key (the parallel of
     loadLevels / GLOW_KEY / VIEWDIST_KEY), so it survives a reload and a dataset-size switch. Runs
     BEFORE the scene boot: a preset boot then overrides it via applyScene, a 'custom' boot keeps it.
     No blob ⇒ leave the first-run defaults untouched (applyScene(space4) sets the group for a fresh
     visitor). Missing/corrupt fields fall back onto NEAR3D_DEFAULT. */
  const restoreNear3d = (): void => {
    let s: Partial<Near3dState> | null = null;
    try {
      const raw = localStorage.getItem(NEAR3D_KEY);
      if (raw !== null) s = JSON.parse(raw) as Partial<Near3dState>;
    } catch {
      /* storage denied/corrupt: the defaults (near off) are a safe session */
    }
    if (!s) return; // first visit / no stored group → today's defaults, unchanged
    const d = NEAR3D_DEFAULT;
    const str = (v: unknown, dv: string): string => (typeof v === 'string' ? v : dv);
    const numv = (v: unknown, dv: number): number =>
      typeof v === 'number' && Number.isFinite(v) ? v : dv;
    applyNear3d({
      near: s.near === true,
      shape: str(s.shape, d.shape),
      tex: str(s.tex, d.tex),
      glyphTex: str(s.glyphTex, d.glyphTex),
      surfaceDetail: str(s.surfaceDetail, d.surfaceDetail),
      terrain: s.terrain === true,
      promote: numv(s.promote, d.promote),
      count: numv(s.count, d.count),
      pinned: numv(s.pinned, -1),
    });
  };

  /* ======================================================================================
   * THE ROUTING TABLE — one place a config key meets its subsystem (the demo's pattern).
   * ====================================================================================== */

  const linkTotal = field.stats().links.total;
  const typeFacet = graph.facets['type']!;

  /* -- nodes: channel × curve × scale into the iSize buffer (field.sizeOf reads the same
     array, so picker and ladder stay on the drawn disc). */
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
          : sizesFromColumn(graph.columns[channel]!.data, n, curve);
      sizeChannelKey = channel;
      sizeCurveVal = curve;
    }
    const k = num('nodeScale') / NODE_SCALE;
    for (let i = 0; i < n; i++) sizeBuf[i] = baseSizes[i]! * k;
    sizeAttr.needsUpdate = true;
  };

  /* -- links take their endpoints' colours; RGB lanes only, one whole-buffer range. */
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

  /* -- colour: 'column' through the library; 'facet' through a palette, where 'vault' is the
     data's own layer palette — meta3.layerColors through namedPaletteBuffer; 'uniform' flat. */
  const applyColors = (): void => {
    const mode = str('colorMode');
    if (mode === 'column') {
      field.setColorColumn('degree', str('ramp') as RampName);
      return;
    }
    if (mode === 'density') {
      // LOCAL SPATIAL DENSITY through the library (field/density.ts): the field computes a
      // per-node neighbourhood count off the CURRENT target geometry and ranks it through the
      // ramp — crowded hot, sparse cool, white cores. It recolours itself on every arrange change
      // (setHierarchy/setTarget), so the look follows a morph without this being re-run; the link
      // RGB is rewritten inside the library, no site-side rewriteLinkRGB needed.
      field.setColorByDensity(str('ramp') as RampName);
      return;
    }
    if (mode === 'facet' && str('colorOrder') === 'spatial') {
      // SPATIAL facet order (F1): the library projects the chosen facet's value centroids onto their
      // dominant spatial axis and colours each cluster by that projection through the ramp, so
      // spatially-adjacent clusters share hue (a smooth, seam-free gradient). Position-dependent, so
      // it recolours itself on every arrange change (setHierarchy/setTarget) — and the link RGB is
      // rewritten inside the library, no site-side rewriteLinkRGB needed. The palette control is
      // inert here (ramp drives it); colorInertKeys() greys it to say so.
      const facetKey = str('colorFacet');
      field.setColorFacetSpatial(graph.facets[facetKey] ? facetKey : 'type', str('ramp') as RampName);
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
      // WHICH facet colours the field is the user's choice (the 'colour facet' picker, widened
      // from the live graph) — default 'type', today's behaviour. The palette is built from the
      // CHOSEN facet's values and painted by its data, exactly the field's own facet-colour path.
      const facetKey = str('colorFacet');
      const facet = graph.facets[facetKey] ?? typeFacet;
      const name = str('palette');
      const pal =
        name === 'vault'
          ? // the vault's named layer colours key on the 'type' facet's value names; for any
            // other facet those names don't match, so fall back to the golden generator there.
            facet === typeFacet
            ? namedPaletteBuffer(facet.values, V.layerColorsByName, [0.6, 0.62, 0.66])
            : facetPalette(facet.values.length)
          : name === 'golden'
            ? facetPalette(facet.values.length)
            : paletteBuffer(PALETTES[name as PaletteName], facet.values.length);
      paintFacetColors(facet.data, n, pal, colors);
    }
    colorAttr.needsUpdate = true;
    rewriteLinkRGB();
  };

  /* -- which of the 'palette' / 'colour ramp' controls has NO live effect in the current colour
     mode, so the panel can GREY the dead one (Fix 2). Exactly one of the two ever drives the
     colour: the categorical PALETTE drives only facet + value order; the RAMP drives column,
     density, and facet + spatial order; 'uniform' is a flat hue and uses neither. Greying the
     inapplicable control stops the "I changed the palette and nothing happened" confusion — the
     value still saves and rides looks, it just can't move the field from here. Recomputed live
     whenever colorMode / colorOrder change (routeKey / onBulk below). */
  const colorInertKeys = (): string[] => {
    const mode = str('colorMode');
    if (mode === 'uniform') return ['palette', 'ramp']; // flat hue — neither drives
    if (mode === 'facet') return str('colorOrder') === 'spatial' ? ['palette'] : ['ramp'];
    return ['palette']; // column | density — the ramp drives
  };

  /* -- 'center size' is meaningful ONLY when card attention is 'center'; in cursor mode it sizes a
     region nothing is gated to, so grey it (the same inert-row convention as the colour controls:
     the value still saves and rides looks, it just has no live effect here). */
  const ladderInertKeys = (): string[] => (str('cardAttention') === 'center' ? [] : ['centerSize']);

  /* The full inert set the panel paints. setInert REPLACES the set each call, so every caller must
     pass the union — colour greying AND ladder greying — or flipping one would clear the other. */
  const inertKeys = (): string[] => [...colorInertKeys(), ...ladderInertKeys()];

  /* -- shape: bind the node glyph to a facet (library setShapeFacet). 'none' → all discs. The
     glyphs only read as distinct at a decent node size / lower density, so this pairs with the
     'node size' control. An independent channel from colour — the two do not interact. */
  const applyShapeFacet = (): void => {
    const key = str('shapeFacet');
    field.setShapeFacet(key === 'none' || !graph.facets[key] ? null : key);
  };

  const applyGround = (): void => {
    const bg = str('background');
    const light = bg === 'light';
    // 'black' is pure #000 (the nebula look's ground); 'dark' is the field's BG_DARK near-black.
    const color = light ? BG_LIGHT : bg === 'black' ? BG_BLACK : field.background;
    scene.background = new Color(color);
    document.body.classList.toggle('light', light);
  };

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
    // master switches: the ladder reads these live every tick, so flipping one drops its tier the
    // same frame (and emits the exits the overlay unmounts on) — no ladder rebuild needed.
    c.showLabels = bool('showLabels');
    c.showCards = bool('showCards');
    c.showOpen = bool('showOpen');
    // the attention gate is read live too: 'center' gates cards to a centreSize-px region, 'cursor'
    // leaves it off. gatherCandidates independently switches the cursorPx anchor to match.
    c.attentionGateRadiusPx =
      str('cardAttention') === 'center' ? num('centerSize') : Number.POSITIVE_INFINITY;
  };

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
  /* The tonemap seam: with GLOW on the roll-off happens ONCE, post-sum, in the bloom chain, so
     the field's in-material curve is bypassed ('none') to avoid tonemapping twice; with glow off
     the field owns the curve exactly as before. One function, both paths — nothing can drift. */
  const applyTonemap = (): void => {
    const op = str('tonemap') as TonemapOp;
    const exposure = num('exposure');
    if (glow.on) {
      post.setTonemap(op, exposure);
      field.setTonemap('none');
    } else {
      field.setTonemap(op, exposure);
    }
  };
  /* Live bloom knobs → post.setBloom + the readout spans. */
  const applyGlowParams = (): void => {
    post.setBloom({ strength: glow.strength, radius: glow.radius, threshold: glow.threshold });
    ui.glowStrengthv.textContent = glow.strength.toFixed(2);
    ui.glowRadiusv.textContent = glow.radius.toFixed(2);
    ui.glowThresholdv.textContent = glow.threshold.toFixed(2);
  };
  /* Highlight compression (PRE-bloom, post/bloom.ts). Unlike strength/radius/threshold it is a
     CONFIG key (the nebula preset ships a moderate default, so the core shows structure out of
     the box, and the tuning panel's "highlight compression" row edits the same value) — so the
     store is the source of truth and this mirrors it into the glow-box slider. Only the post
     chain reads it; with glow off it has no visible effect. */
  const applyCompression = (): void => {
    const c = num('compression');
    post.setCompression(c);
    ui.glowCompression.value = String(c);
    ui.glowCompressionv.textContent = c.toFixed(2);
  };
  /* Turn the bloom on/off: reroute the tonemap (applyTonemap reads glow.on), reflect the UI,
     persist. The render loop reads glow.on to pick post.render() vs renderer.render(). */
  const setGlow = (on: boolean, persist = true): void => {
    glow.on = on;
    ui.glow.checked = on;
    document.getElementById('glowbox')?.classList.toggle('on', on);
    applyGlowParams();
    applyTonemap();
    if (persist) saveGlow();
    updateStats();
  };
  /* -- links on/off (COMPARISON.md F5) composes with the budget: off is a budget of zero —
     reported in the stats line like every drop (§10.4), never a hidden object. */
  const applyBudget = (): void => {
    field.setLinkBudget(ui.links.checked ? Math.round(num('linkBudget') * linkTotal) : 0);
  };
  ui.links.addEventListener('change', () => {
    applyBudget();
    updateStats();
  });
  const applySpread = (): void => {
    field.setSpread(num('spread'), 0.15);
  };
  const applyRenderScale = (): void => {
    renderer.setPixelRatio(window.devicePixelRatio * num('renderScale'));
    renderer.setSize(W, H);
    post.setSize(W, H); // the pixel ratio just changed — resize the HDR/bloom targets with it
  };
  const applyFlatPx = (): void => {
    field.setFlatPx(num('flatPx'));
  };
  /* -- node GLYPH style: the seven knobs that were compile-time shader constants until the
     library's stage 10, now live GLOBAL uniforms (one Field.setNodeStyle write, no recompile).
     All seven are pushed together — cheap scalar uniforms, so splitting them per-key would buy
     nothing. `nodeMinPx`/`nodeMaxPx` are the px clamp rails (library key `minPx`/`maxPx`). */
  /* The 'size cap' slider ships at its TOP detent = OFF (schema default === offAt): map that to
     the library's uncapped sentinel so the node grows without bound and fills the view on any
     approach — no select, no "pushing". Dragging the slider DOWN passes a finite px cap through
     verbatim (a user re-imposing a cap). offAt is read from the live schema row so the threshold
     can never drift from the slider's own top. */
  const nodeCapOffAt =
    ((schema.find((r) => r.key === 'nodeMaxPx') as { offAt?: number } | undefined)?.offAt) ??
    Infinity;
  const applyNodeStyle = (): void => {
    const cap = num('nodeMaxPx');
    field.setNodeStyle({
      bodyR: num('bodyR'),
      haloPx: num('haloPx'),
      haloGain: num('haloGain'),
      coreHot: num('coreHot'),
      minPx: num('nodeMinPx'),
      maxPx: cap >= nodeCapOffAt ? NODE_MAX_PX_OFF : cap,
      falloff: num('falloff'),
      sphere: num('sphere'),
    });
  };
  /* -- link style: the long-edge fade. `lenSoft` is a FRACTION of the field diagonal (the
     library multiplies by its diagonal), `lenFloor` the fraction the longest edge keeps. */
  const applyLinkStyle = (): void => {
    field.setLinkStyle({ lenSoft: num('lenSoft'), lenFloor: num('lenFloor') });
  };

  const ROUTE: Record<string, () => void> = {
    tierCard: applyLadderCfg,
    tierLock: applyLadderCfg,
    tierOpen: applyLadderCfg,
    cardRadius: applyLadderCfg,
    maxCards: rebuildLadder,
    lockW: rebuildLadder,
    lockH: rebuildLadder,
    labelCap: rebuildLadder,
    namedPx: applyLadderCfg,
    labelEarn: applyLadderCfg,
    labelKeep: applyLadderCfg,
    labelDwell: applyLadderCfg,
    showLabels: applyLadderCfg,
    showCards: applyLadderCfg,
    showOpen: applyLadderCfg,
    cardAttention: applyLadderCfg,
    centerSize: applyLadderCfg,
    sizeChannel: applySizes,
    nodeScale: applySizes,
    sizeCurve: applySizes,
    shapeFacet: applyShapeFacet,
    bodyR: applyNodeStyle,
    haloPx: applyNodeStyle,
    haloGain: applyNodeStyle,
    coreHot: applyNodeStyle,
    nodeMinPx: applyNodeStyle,
    nodeMaxPx: applyNodeStyle,
    falloff: applyNodeStyle,
    sphere: applyNodeStyle,
    lenSoft: applyLinkStyle,
    lenFloor: applyLinkStyle,
    colorMode: applyColors,
    colorFacet: applyColors,
    colorOrder: applyColors,
    palette: applyColors,
    ramp: applyColors,
    background: applyGround,
    blend: applyBlend,
    tonemap: applyTonemap,
    exposure: applyTonemap,
    compression: applyCompression,
    linkBudget: applyBudget,
    spread: applySpread,
    cameraDrift: applyFlight,
    driftRate: applyFlight,
    levelSpin: applySpins,
    speedK: applyFlight,
    mouseSens: applyFlight,
    turnRate: applyFlight,
    easeLin: applyFlight,
    renderScale: applyRenderScale,
    flatPx: applyFlatPx,
  };
  const BOOT_KEYS = ['linkBase', 'linkFocus', 'linkHover', 'linkDim', 'dimOthers'] as const;
  /* Nothing is inert any more: the node glyph knobs (bodyR/haloPx/haloGain/coreHot/
     nodeMinPx/nodeMaxPx/falloff) and the link-length knobs (lenSoft/lenFloor) were the
     library's last compile-time shader constants, and stage 10 turned them into live GLOBAL
     uniforms — all nine now route to Field.setNodeStyle/setLinkStyle above, so the panel
     renders every row normally rather than graying it with a "no live effect" tooltip. */
  const UNROUTED_KEYS = [] as const;

  const routeKey = (key: string): void => {
    const route = ROUTE[key];
    if (route) route();
    // colorMode / colorOrder decide which of palette / ramp is live, and cardAttention decides
    // whether 'center size' is live, so re-grey the dead row the instant any of them moves — the
    // panel repaints only the affected rows, no rebuild.
    if (key === 'colorMode' || key === 'colorOrder' || key === 'cardAttention')
      panel.setInert(inertKeys());
    updateStats();
  };
  const applyAll = (): void => {
    rebuildLadder();
    const seen = new Set<() => void>([rebuildLadder]);
    for (const key of Object.keys(ROUTE)) {
      const route = ROUTE[key]!;
      if (seen.has(route)) continue;
      seen.add(route);
      route();
    }
    updateStats();
  };

  /* ---- the tuning panel + the rail's look select (both drive the same store). ----------- */
  injectPanelStyles();
  const panel = createPanel({
    store,
    schema,
    onChange: (key: string): void => {
      routeKey(key);
      markSceneCustom(); // a tuning-panel edit diverges from the active scene preset (guarded)
    },
    onBulk: (): void => {
      applyAll();
      syncLook();
      // a preset / reset / paste may have moved colorMode|colorOrder|cardAttention — re-grey all
      panel.setInert(inertKeys());
      markSceneCustom();
    },
    inertKeys: UNROUTED_KEYS,
  });
  document.body.appendChild(panel.el);
  panel.setInert(inertKeys()); // seed palette/ramp AND center-size greying from the booted config
  ui.tune.addEventListener('click', () => panel.toggle());

  /* ======================================================================================
   * EXPORT AS CODE — serialize the LIVE scene (the config store + the level-editor rows) into a
   * runnable TypeScript snippet a visitor can paste into their own app. Everything is read fresh
   * on open, so the code changes when the settings change. Pure demo-side serializer (no library
   * change): it reuses buildPlace (the compiler's own level→spec step) so a level's exported place
   * is byte-identical to what the field is drawing, and mirrors applySpins' shifted spin mapping.
   * ====================================================================================== */
  const r4 = (x: number): number => Math.round(x * 1e4) / 1e4;
  const defOf = (key: string): ConfigValue | undefined => schema.find((row) => row.key === key)?.default;
  /** Drop a place's default-valued knobs so an untouched level reads as `{ shape: 'shell' }`. */
  const stripPlaceDefaults = (p: Record<string, unknown>): Record<string, unknown> => {
    const o: Record<string, unknown> = { ...p };
    if (o['ordering'] === 'byIndex') delete o['ordering'];
    if (o['density'] === 'uniform') delete o['density'];
    if (o['scatter'] === 'none') delete o['scatter'];
    if (o['densityAxis'] === 0) delete o['densityAxis'];
    if (o['params'] && typeof o['params'] === 'object' && Object.keys(o['params'] as object).length === 0) {
      delete o['params'];
    }
    if (o['base'] && typeof o['base'] === 'object') {
      o['base'] = stripPlaceDefaults(o['base'] as Record<string, unknown>);
    }
    return o;
  };

  const generateCode = (): string => {
    const ss = ['generateSynth', 'buildGraph', 'createField']; // starscape imports (grown below)
    const body: string[] = [];

    // 1. data — the synthetic corpus the playground runs on, reproducible from (n, seed).
    body.push('// 1. data — swap generateSynth for your own buildGraph({ n, positions, links, columns, facets }) input');
    body.push(`const input = generateSynth({ n: ${synthN}, seed: ${synthSeed} });`);
    body.push('const graph = buildGraph(input);');
    body.push('');

    // 2. renderer + camera + field (two draw calls, WebGL2 fallback) — README's compose pattern.
    body.push('// 2. renderer, camera, field');
    body.push('const renderer = new WebGPURenderer({ antialias: true });');
    body.push('await renderer.init();');
    body.push('document.body.appendChild(renderer.domElement);');
    body.push(
      `const camera = new PerspectiveCamera(${DEFAULT_FOV}, window.innerWidth / window.innerHeight, 0.1, 1e7);`,
    );
    const fieldOpts: string[] = [`heightCss: window.innerHeight`, `fov: ${DEFAULT_FOV}`];
    const sizeCh = str('sizeChannel');
    if (sizeCh !== 'degree') fieldOpts.push(`sizeColumn: '${sizeCh}'`);
    if (str('blend') === 'additive') fieldOpts.push(`blend: 'additive'`);
    body.push(`const field = createField(graph, { ${fieldOpts.join(', ')} });`);
    body.push('const scene = new Scene();');
    body.push('scene.background = new Color(field.background);');
    body.push('for (const o of field.objects) scene.add(o);');
    body.push('');

    // 3. arrangement — the per-level hierarchy the custom editor compiles (custom mode only).
    if (mode === 'custom' && rows.length > 0) {
      ss.push('compileHierarchy');
      const L = rows.length;
      const override = num('levelSpin');
      const rowRate = (k: number): number => {
        const row = rows[k];
        const base = row && row.spin ? row.spinRate || 0 : 0;
        return override > 0 ? override : base;
      };
      const rowAxisArr = (k: number): [number, number, number] => {
        const a = axisUnit(rows[k]?.spinAxis ?? [0, 1, 0]);
        return [r4(a[0]), r4(a[1]), r4(a[2])];
      };
      const levels = rows.map((row, k): Record<string, unknown> => {
        const isLeaf = k === L - 1;
        const o: Record<string, unknown> = {};
        if (!isLeaf) o['groupBy'] = row.groupBy;
        o['place'] = stripPlaceDefaults(buildPlace(row) as unknown as Record<string, unknown>);
        if (L === 1) {
          if (row.spread !== 1) o['radiusScale'] = r4(row.spread);
        } else if (!isLeaf) {
          const cs = rows[k + 1]?.spread;
          if (cs !== undefined && cs !== 1) o['radiusScale'] = r4(cs);
        }
        // the shifted spin mapping (applySpins): a grouping level k's pivots are driven by row k+1
        if (!isLeaf) {
          const rt = rowRate(k + 1);
          if (rt !== 0) o['spin'] = { rate: r4(rt), axis: rowAxisArr(k + 1) };
        }
        return o;
      });
      const spacing = r4(arrangeSpacing);
      const rootRadiusScale = r4(Math.min(10, Math.max(0.25, rows[0]?.spread ?? 1)));
      body.push('// 3. arrangement — the per-level hierarchy (shapes, ordering, density, scatter, spin)');
      body.push(`const levels = ${JSON.stringify(levels, null, 2)};`);
      body.push(
        `const hierarchy = compileHierarchy(graph, levels, { spacing: ${spacing}, rootRadiusScale: ${rootRadiusScale} });`,
      );
      body.push('field.setHierarchy(hierarchy);');
      // row 0 drives the FIELD-GLOBAL rotation (no pivot sits above it)
      const fieldRate = rowRate(0);
      if (fieldRate !== 0) {
        const a = rowAxisArr(0);
        body.push(`field.setFieldSpin(${r4(fieldRate)}, [${a[0]}, ${a[1]}, ${a[2]}]);`);
      }
      body.push('');
    } else {
      body.push('// 3. arrangement — frozen: the generated layout as-is (no hierarchy compiled)');
      body.push('');
    }

    // 4. visuals — only what differs from the library defaults (keep it clean).
    const vis: string[] = [];
    // colour
    const cmode = str('colorMode');
    const ramp = str('ramp');
    const facet = str('colorFacet');
    if (cmode === 'column') {
      vis.push(`field.setColorColumn('degree', '${ramp}');`);
    } else if (cmode === 'density') {
      vis.push(`field.setColorByDensity('${ramp}');`);
    } else if (cmode === 'facet' && str('colorOrder') === 'spatial') {
      vis.push(`field.setColorFacetSpatial('${facet}', '${ramp}');`);
    } else if (cmode === 'facet' && facet !== 'type') {
      // value-order facet colour: the library binds the facet; the playground additionally paints a
      // custom palette (palette = '${str('palette')}') into the iColor buffer — see demo/main.ts applyColors
      vis.push(`field.setColorFacet('${facet}');`);
    }
    // shape-by-facet
    const shapeFacet = str('shapeFacet');
    if (shapeFacet !== 'none') vis.push(`field.setShapeFacet('${shapeFacet}');`);
    // node glyph style — only the keys the user moved off their default
    const nodeStyle: string[] = [];
    const pushIf = (cfgKey: string, libKey: string): void => {
      const v = num(cfgKey);
      if (v !== defOf(cfgKey)) nodeStyle.push(`${libKey}: ${r4(v)}`);
    };
    pushIf('bodyR', 'bodyR');
    pushIf('haloPx', 'haloPx');
    pushIf('haloGain', 'haloGain');
    pushIf('coreHot', 'coreHot');
    pushIf('nodeMinPx', 'minPx');
    pushIf('falloff', 'falloff');
    if (num('nodeMaxPx') !== defOf('nodeMaxPx')) nodeStyle.push(`maxPx: ${r4(num('nodeMaxPx'))}`); // a finite px cap
    if (nodeStyle.length > 0) vis.push(`field.setNodeStyle({ ${nodeStyle.join(', ')} });`);
    // link style
    const linkStyle: string[] = [];
    if (num('lenSoft') !== defOf('lenSoft')) linkStyle.push(`lenSoft: ${r4(num('lenSoft'))}`);
    if (num('lenFloor') !== defOf('lenFloor')) linkStyle.push(`lenFloor: ${r4(num('lenFloor'))}`);
    if (linkStyle.length > 0) vis.push(`field.setLinkStyle({ ${linkStyle.join(', ')} });`);
    // spread
    if (num('spread') !== 1) vis.push(`field.setSpread(${r4(num('spread'))});`);
    // view distance
    const vd = Math.min(20, Math.max(0.5, Number(ui.viewdist.value) || 1));
    if (vd !== 1) vis.push(`field.setViewDistance(${r4(vd)});`);
    // projection
    if (field.projection === '2d') vis.push(`field.setProjection('2d');`);
    // tone curve (when glow is OFF the field owns the roll-off; with glow it moves to the post chain)
    if (!glow.on && str('tonemap') !== 'none') {
      vis.push(`field.setTonemap('${str('tonemap')}', ${r4(num('exposure'))});`);
    }
    if (vis.length > 0) {
      body.push('// 4. visuals (only what differs from the library defaults)');
      body.push(...vis);
      body.push('');
    }

    // 5. glow — the HDR bloom post-process (only when enabled).
    if (glow.on) {
      ss.push('createPostProcess');
      body.push('// 5. glow — HDR bloom: render additively, bloom the sum, tonemap once post-sum');
      const tm = str('tonemap');
      body.push(
        `const post = createPostProcess(renderer, scene, camera, {\n` +
          `  bloom: { strength: ${r4(glow.strength)}, radius: ${r4(glow.radius)}, threshold: ${r4(glow.threshold)} },\n` +
          `  tonemap: '${tm === 'none' ? 'reinhard' : tm}', exposure: ${r4(num('exposure'))}, resolutionScale: 1,\n` +
          `});`,
      );
      if (num('compression') !== 0) body.push(`post.setCompression(${r4(num('compression'))});`);
      body.push('// render loop: call post.render() instead of renderer.render(scene, camera)');
      body.push('');
    }

    body.push('// camera + zoom-ladder wiring (createFlycam + createLadder + createOverlay): see demo/main.ts');

    const importThree = `import { Color, PerspectiveCamera, Scene, WebGPURenderer } from 'three/webgpu';`;
    const importStar = `import {\n  ${ss.join(',\n  ')},\n} from 'starscape';`;
    return `${importStar}\n${importThree}\n\n${body.join('\n')}\n`;
  };

  /* ---- the export-as-code modal: reuse the legend overlay conventions (Esc + click-outside to
     close), a scrollable <pre>, and a Copy button that writes to the clipboard and flips to a
     confirmation for ~1.5 s (falling back to selecting the text when clipboard is unavailable). */
  const codeModal = document.getElementById('codemodal')!;
  const codeOut = document.getElementById('codeout')!;
  const codeCopy = document.getElementById('codecopy') as HTMLButtonElement;
  const openCode = (): void => {
    codeOut.textContent = generateCode();
    codeModal.removeAttribute('hidden');
  };
  const closeCode = (): void => codeModal.setAttribute('hidden', '');
  document.getElementById('code')!.addEventListener('click', openCode);
  // backdrop click closes; clicks inside the card (selecting code) do not
  codeModal.addEventListener('click', (ev) => {
    if (ev.target === codeModal) closeCode();
  });
  let copyTimer = 0;
  codeCopy.addEventListener('click', () => {
    const text = codeOut.textContent ?? '';
    const flip = (): void => {
      codeCopy.textContent = 'copied ✓';
      codeCopy.classList.add('ok');
      window.clearTimeout(copyTimer);
      copyTimer = window.setTimeout(() => {
        codeCopy.textContent = 'Copy';
        codeCopy.classList.remove('ok');
      }, 1500);
    };
    // clipboard unavailable / denied / hung: select the <pre> so the user can Cmd/Ctrl-C.
    const selectFallback = (): void => {
      const range = document.createRange();
      range.selectNodeContents(codeOut);
      const sel = window.getSelection();
      sel?.removeAllRanges();
      sel?.addRange(range);
      flip();
    };
    // resolve exactly once: success flips; reject or a stalled write (no user gesture / blurred
    // document — some embedded contexts never settle writeText) falls back within 500 ms.
    let done = false;
    const win = (): void => {
      if (done) return;
      done = true;
      flip();
    };
    const fail = (): void => {
      if (done) return;
      done = true;
      selectFallback();
    };
    const clip = navigator.clipboard;
    if (clip && typeof clip.writeText === 'function') {
      clip.writeText(text).then(win, fail);
      window.setTimeout(fail, 500);
    } else {
      fail();
    }
  });

  for (const name of Object.keys(VISUAL_PRESETS)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = VISUAL_PRESETS[name as keyof typeof VISUAL_PRESETS].label;
    ui.look.appendChild(opt);
  }
  const syncLook = (): void => {
    const names = Object.keys(VISUAL_PRESETS);
    ui.look.value = names.includes(store.preset) ? store.preset : names[0]!;
  };
  syncLook();
  ui.look.addEventListener('change', () => {
    if (!store.applyLook(ui.look.value)) return;
    store.save();
    panel.sync();
    // the 'nebula' look turns the glow ON (its halos are the bloom's input); every other look
    // turns it off, so picking any other look is a clean return to the un-bloomed field.
    setGlow(ui.look.value === 'nebula');
    applyAll();
    markSceneCustom(); // picking a look diverges from the active scene preset
  });

  /* ======================================================================================
   * SCENE PRESETS — one pick drives the WHOLE scene (hierarchy + look + glow + view), reusing
   * the playground's existing apply paths. applyScene is guarded by applyingScene so the cascade
   * of applies it fires never flips the selector to 'custom'. 'space4' is the curated galaxy-
   * nebula; 'default' returns to the neutral frozen field. The chosen scene persists (SCENE_KEY)
   * and is re-applied on every boot — so it survives a dataset-size switch (which reloads).
   * ====================================================================================== */
  const buildSpace4Rows = (): LevelRow[] => [
    // L0 — systems laid out as spiral galaxies, slow turn
    mkRow('system', 'preset:spiralGalaxy', {
      ordering: 'byHash',
      density: 'exponential',
      densityAxis: 0,
      scatter: 'gaussian',
      shapeP: { arms: 6, swirl: 5, bar: 0.03, radialPower: 0.35 },
      densP: { rate: 1.25 },
      scatP: { thickness: 0.01 },
      spread: 1,
      spin: true,
      spinRate: 0.02,
      spinAxis: [0, 1, 0],
    }),
    // L1 — a shape per type cluster (by-cluster-facet), power-law packed
    mkRow('type', BY_FACET_PLACE, {
      ordering: 'byIndex',
      density: 'powerLaw',
      scatter: 'gaussian',
      shapeP: {},
      densP: { alpha: 1e8 },
      scatP: { thickness: 0.5 },
      spread: 1,
      spin: true,
      spinRate: 0.1,
      spinAxis: [0, 1, 0],
    }),
    // L2 (leaf) — nodes on a thin disc
    mkRow(null, 'disc', {
      ordering: 'byIndex',
      density: 'exponential',
      densityAxis: 1,
      scatter: 'gaussian',
      shapeP: {},
      densP: {},
      scatP: { thickness: 0.01 },
      spread: 1,
      spin: true,
      spinRate: 0.3,
      spinAxis: [0, 1, 0],
    }),
  ];
  const applyScene = (name: string): void => {
    if (name !== 'space4' && name !== 'default') return;
    applyingScene = true;
    // 1. LOOK (config): reset to the neutral 'field' baseline, then overlay the scene's diff.
    store.applyPreset('field');
    if (name === 'space4') {
      for (const [k, v] of Object.entries(SPACE4_LOOK)) store.set(k, v);
      panel.markCustom();
    }
    // 2. GLOW: set the bloom params + on/off (setGlow reroutes the tonemap seam, reflects the UI,
    //    persists, and toggles the glow-box rows).
    const g = name === 'space4' ? SPACE4_GLOW : { on: false, strength: 1.3, radius: 0.4, threshold: 1 };
    glow.strength = g.strength;
    glow.radius = g.radius;
    glow.threshold = g.threshold;
    ui.glowStrength.value = String(g.strength);
    ui.glowRadius.value = String(g.radius);
    ui.glowThreshold.value = String(g.threshold);
    setGlow(g.on);
    // 3. HIERARCHY / mode
    if (name === 'space4') {
      mode = 'custom';
      ui.mode.value = 'custom';
      ui.levelsbox.style.display = '';
      rows = buildSpace4Rows();
      ui.hpreset.value = '';
      renderLevels();
      applyCustomNow(false);
    } else {
      mode = 'frozen';
      ui.mode.value = 'frozen';
      ui.levelsbox.style.display = 'none';
      applyFrozen();
    }
    saveLevels();
    // 4. VIEW: distance + projection
    const view = name === 'space4' ? SPACE4_VIEW : { distance: 1, projection: '3d' as const };
    ui.viewdist.value = String(view.distance);
    applyViewDistance();
    ui.proj.value = view.projection;
    applyProjection();
    // 4b. 3D NODES: the scene governs the near-mesh group too (both presets ⇒ off). Direct-apply
    //     under the applyingScene guard (no events, so no spurious divergence), then persist its own
    //     key — the exact parallel of saveLevels above, so NEAR3D_KEY stays consistent with the scene.
    applyNear3d(nearSpecOf(name));
    saveNear3d();
    // 5. route the config through every subsystem + sync the panels
    applyAll();
    panel.sync();
    panel.setInert(inertKeys());
    syncLook();
    store.save();
    // 6. selector state + persistence
    currentScene = name;
    ui.scene.value = name;
    try {
      localStorage.setItem(SCENE_KEY, name);
    } catch {
      /* storage denied: the session still works, the choice just will not persist */
    }
    applyingScene = false;
    updateStats();
  };
  ui.scene.addEventListener('change', () => {
    const v = ui.scene.value;
    if (v === 'space4' || v === 'default') applyScene(v);
  });

  /* ---- the stats overlay: rAF-interval median (SCALE.md's measurement) + jsFrameMs, and
     every hidden state §10.7 wants visible. */
  const intervals = new Float64Array(90);
  const jsMs = new Float64Array(90);
  let filled = 0;
  let cursor = 0;
  let jsFilled = 0;
  let jsCursor = 0;
  let prevT = -1;
  let frame = 0;
  let bootMs = 0;
  let drawCalls = 0;
  let room = Infinity;
  let speed = 0;
  let blocked = false;
  let altitude = Infinity; // height above the nearest promoted-planet surface (near-mesh floor)
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

  const insideFrame = (): string => {
    const f = field.frames;
    if (!f) return 'world';
    const s = field.spread;
    const c = field.centroid();
    const p: [number, number, number] = [
      c[0] + (camera.position.x - c[0]) / s,
      c[1] + (camera.position.y - c[1]) / s,
      c[2] + (camera.position.z - c[2]) / s,
    ];
    const piv = f.frameOfPoint(p);
    if (piv < 0) return 'world';
    const lv = hierarchy ? hierarchy.pivots.level[piv]! : 0;
    return `p${piv}/L${lv}`;
  };

  const fmt = (x: number): string => x.toLocaleString('en-US');
  function updateStats(): void {
    const s = field.stats();
    const hov = field.hover;
    const sel = field.select;
    statsEl.textContent =
      `nodes ${fmt(s.nodes)} · links ${fmt(s.links.drawn)} / ${fmt(s.links.total)} drawn · ` +
      `regions ${V.regions.length} · draw calls ${drawCalls} · ` +
      `fps ${fpsMedian().toFixed(1)} (median of ${filled}) · jsFrameMs ${jsFrameMs().toFixed(2)} · ` +
      `boot ${bootMs.toFixed(0)} ms (graph ${(tGraph - t0).toFixed(0)}) · ${backendName}\n` +
      `inside:${insideFrame()} · morph t ${s.morph.t.toFixed(2)} · spread ${s.spread.toFixed(2)} · ` +
      `${s.projection} · pivots ${fmt(s.pivots)} · field-spin ${s.fieldSpin.toFixed(2)} · ` +
      `render dpr ${renderer.getPixelRatio()} ` +
      `(scale ${num('renderScale')}) · flat < ${num('flatPx')} px${settledNow ? '' : ' · animating'}\n` +
      `room ${Number.isFinite(room) ? room.toFixed(0) : '∞'} · speed ${speed.toFixed(0)}` +
      `${blocked ? ' (damped)' : ''} · ` +
      `hover: ${hov >= 0 ? graph.labelOf(hov) : '—'} · ` +
      `selected: ${sel >= 0 ? graph.labelOf(sel) : '—'}\n` +
      `ladder: cand ${lastCand} · named ${ladder.state.namedCount}/${ladder.config.namedCap} · ` +
      `cards ${ladder.state.cards.length}/${ladder.config.cards} · ` +
      `open ${ladder.state.open >= 0 ? graph.labelOf(ladder.state.open) : '—'}` +
      `${ladder.state.suspended ? ' · suspended' : ''}` +
      `${field.select >= 0 ? ' · pinned' : ''}` +
      ` · look ${store.preset}` +
      `${nearMesh && ui.near.checked ? ` · 3D ${nearMesh.promoted} meshes / ${nearMesh.drawCalls} dc` : ''}` +
      `${
        nearMesh && ui.near.checked && nearMesh.terrainActive
          ? ` · terrain L${nearMesh.terrainSubdiv} ${fmt(nearMesh.terrainTriangles)} tris` +
            `${Number.isFinite(altitude) ? ` · alt ${altitude.toFixed(2)}` : ''}`
          : ''
      }` +
      `${
        nearMesh && ui.near.checked && (nearMesh.imageMeshes > 0 || pinnedImageNode >= 0)
          ? ` · img ${nearMesh.imageMeshes}${
              nearMesh.imageInfo[0]
                ? ` t${nearMesh.imageInfo[0].tier} @${Math.round(nearMesh.imageInfo[0].px)}px`
                : ''
            } · cache ${(nearMesh.imageCacheBytes / 1048576).toFixed(1)}MB`
          : ''
      }`;
  }

  /* GLOW boot: the 'nebula' look implies the glow on (the signature look); any other persisted
     look restores the saved glow pref. Reflect the knobs in the rail, then let applyAll route the
     tonemap through the right path (applyTonemap reads glow.on). */
  glow.on = store.preset === 'nebula' ? true : glow.on;
  ui.glow.checked = glow.on;
  document.getElementById('glowbox')?.classList.toggle('on', glow.on);
  ui.glowStrength.value = String(glow.strength);
  ui.glowRadius.value = String(glow.radius);
  ui.glowThreshold.value = String(glow.threshold);
  applyGlowParams();

  applyAll(); // boot: defaults ⊕ persisted diff reach every subsystem

  /* ---- restore the persisted arrange mode + level config (mode-scoped, like the config
     store), or the shipped defaults; a stored custom mode morphs to it right away. */
  {
    const stored = loadLevels();
    if (stored !== null) {
      mode = stored.mode;
      rows = stored.rows;
      ui.hpreset.value = stored.preset;
    } else {
      ui.hpreset.value = DEFAULT_HPRESET;
    }
    ui.mode.value = mode;
    ui.levelsbox.style.display = mode === 'custom' ? '' : 'none';
    renderLevels();
    if (mode === 'custom') applyCustomNow(false);
  }

  /* ---- 3D-NODES boot: restore the group's own persisted key, exactly like the levels block above
     restores LEVELS_KEY. Runs BEFORE the scene boot so a preset boot overrides it (applyScene) and a
     'custom' boot keeps it — the single rule the whole scene system follows. */
  restoreNear3d();

  /* ---- SCENE boot: a fresh visitor (no stored scene) opens directly on space4; a returning
     visitor re-applies whichever scene they chose, so it survives reloads AND dataset-size
     switches (which reload the page). 'custom' keeps the state restored above (levels + 3D-nodes)
     and just reflects it in the selector. */
  {
    const storedScene = localStorage.getItem(SCENE_KEY);
    if (storedScene === null || storedScene === 'space4') {
      applyScene('space4');
    } else if (storedScene === 'default') {
      applyScene('default');
    } else {
      currentScene = 'custom';
      ui.scene.value = 'custom';
    }
  }

  const loop = (t: number): void => {
    let dt = 0;
    if (prevT >= 0) {
      const dtMs = t - prevT;
      intervals[cursor] = dtMs;
      cursor = (cursor + 1) % intervals.length;
      if (filled < intervals.length) filled++;
      dt = Math.min(dtMs, 100) * 0.001;
    }
    prevT = t;

    const c0 = performance.now();
    const telem = cam.update(dt);
    room = telem.room;
    speed = telem.speed;
    blocked = telem.blocked;
    altitude = telem.altitude ?? Infinity;
    if (pose.active) {
      // the app-side pose tween (see flyPoseTo): wins over the idle flycam this frame
      pose.t += dt * 1000;
      const k = cubicInOut(Math.min(1, pose.t / pose.ms));
      camera.position.lerpVectors(pose.p0, pose.p1, k);
      camera.quaternion.slerpQuaternions(pose.q0, pose.q1, k);
      if (pose.t >= pose.ms) {
        pose.active = false;
        cam.setTarget(pose.g, false); // dragging now pivots around the framed centre
      }
    }

    appTweens.update(dt);
    field.update(dt);

    fieldSettledNow = field.settled();
    settledNow =
      fieldSettledNow && !appTweens.active() && !(fly?.tweening ?? false) && !pose.active;
    if (settledNow && !wasSettled) {
      rebuildGrid();
      updateStats();
    }
    wasSettled = settledNow;
    if (morphing && field.morphT >= 1) {
      morphing = false;
      field.basePositions(baseBuf);
      if (!fieldSettledNow) {
        rebuildGrid();
        gridCooldown = 0.25;
      }
    }
    // grid upkeep under spin: tracked drift widens every walk; rebuild lazily, rate-limited
    if (!morphing && (field.frames?.activeThisTick() ?? false)) {
      gridDrift += spinDriftRate * dt * field.spread;
    }
    gridCooldown = Math.max(0, gridCooldown - dt);
    if (!morphing && gridCooldown === 0 && gridDrift > grid.cellMax) {
      rebuildGrid();
      gridCooldown = 1.0;
    }
    if (!fieldSettledNow && !morphing) {
      const c = field.centroid();
      spreadC[0] = c[0];
      spreadC[1] = c[1];
      spreadC[2] = c[2];
      spreadK = field.spread;
      field.fieldSpinQuat(fieldQ); // the field-global rotation, mirrored in refreshViewOf
      fieldDisplaced = field.fieldSpinDisplaced();
    }

    if (pickDue && !morphing && !(fly?.tweening ?? false) && !pose.active) {
      pickDue = false;
      hover(castAt(px, py));
    }

    /* ADAPTIVE NEAR PLANE (O(1)): track the camera's distance to its orbit target so a node you
       close on does not clip, WITHOUT a tiny global near that would z-fight the huge far plane.
       Far away, cam.radius is large and `want` clamps to nearFar (= the original radius*0.001),
       so depth precision at range is byte-for-byte what it was. Closing on a node shrinks `want`
       toward the 0.02 floor. updateProjectionMatrix only when it moves meaningfully (>8%), to
       avoid per-frame matrix churn. */
    {
      const nearFar = Math.max(0.05, radius * 0.001);
      // Keep the near plane BELOW the orbit radius so a node you close right up on never clips:
      // normally the 0.02 floor holds (byte-identical at range), but once the radius shrinks under
      // ~0.04 (close-orbit wheel-in) the floor follows it down to radius·0.5 — always < radius, so
      // the billboard (at the target's depth) stays in front of near. It climbs back on fly-out.
      const want = Math.min(nearFar, Math.max(cam.radius * 0.02, Math.min(0.02, cam.radius * 0.5)));
      if (Math.abs(camera.near - want) > want * 0.08) {
        camera.near = want;
        camera.updateProjectionMatrix();
      }
    }

    /* CLOSE ORBIT (double-click): while flying around a single node, suppress its PINNED reading
       card (the one a pin forces fully open on top of everything) so the node stays a glyph you
       circle — the selection highlight + side panel stay. Holding the ladder suspended
       (cursorMoved forced false) also keeps a card from auto-opening if you wheel right in. The
       mode ends — and the node deselects — once you leave its card distance (any direction). */
    if (closeOrbit >= 0 && !(fly?.tweening ?? false)) {
      // Armed only once the fly-in has LANDED (not while the goTo tween is still closing the
      // distance — the camera starts beyond the leash and flies in, which must not self-cancel).
      const o = closeOrbit * 3;
      if (!fieldSettledNow && !morphing) refreshViewOf(closeOrbit);
      const dx = camera.position.x - viewBuf[o]!;
      const dy = camera.position.y - viewBuf[o + 1]!;
      const dz = camera.position.z - viewBuf[o + 2]!;
      // Generous leash: getting VERY CLOSE (wheel-in shrinks the radius) never trips it — the
      // leash is an upper bound, and dolly/orbit keep the camera within a node-radius of the
      // target. It ends the mode only when you deliberately fly well AWAY from the node.
      const leash = spacing * field.spread * 4.5;
      if (dx * dx + dy * dy + dz * dz > leash * leash) {
        leaveCloseOrbit();
        paintSelected(-1);
        if (field.setSelect(-1)) updateStats();
      }
    }

    camera.updateMatrixWorld();
    camInv.copy(camera.matrixWorld).invert();
    clipM.multiplyMatrices(camera.projectionMatrix, camInv);
    ladderView.pinned = closeOrbit >= 0 ? -1 : field.select;
    if (closeOrbit >= 0) cursorMoved = false; // keep the ladder suspended while orbiting
    ladderView.cursorMoved = cursorMoved;
    ladderView.spacing = spacing * field.spread;
    cursorMoved = false;
    lastCand = morphing ? 0 : gatherCandidates();
    ladderCandidates.count = lastCand;
    const ladderState = ladder.update(dt, ladderView);
    if (!fieldSettledNow && !morphing) {
      for (let j = 0; j < ladderState.namedCount; j++) refreshViewOf(ladderState.named[j]!);
      for (const slot of ladderState.cards) refreshViewOf(slot.node);
    }
    overlayView.width = W;
    overlayView.height = H;
    overlayView.pinned = closeOrbit >= 0 ? -1 : field.select;
    ladderOverlay.update(ladderState, overlayView);
    updateRegionLabels();

    // the OPT-IN near-mesh LOD (null unless the 3D-nodes toggle was switched on): promote the
    // nodes large on screen to real meshes and fade their billboards. O(local); its cost shows
    // honestly in jsFrameMs (it sits before the measurement close).
    if (nearMesh) nearMesh.update(camera.position, dt);

    const c1 = performance.now();
    jsMs[jsCursor] = c1 - c0;
    jsCursor = (jsCursor + 1) % jsMs.length;
    if (jsFilled < jsMs.length) jsFilled++;

    // GLOW on → the HDR bloom chain (scene → half-float → bloom → post-sum tonemap → screen);
    // off → the plain direct render, byte-for-byte the pre-bloom path (zero added cost).
    if (glow.on) post.render();
    else renderer.render(scene, camera);
    drawCalls = renderer.info.render.drawCalls;
    if (frame === 0) {
      bootMs = performance.now(); // navigationStart → first frame (a timestamp, not a
      performance.mark('starscape-playground:first-frame'); // duration around render — SCALE.md)
    }
    if (frame % 30 === 0) updateStats();
    frame++;
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  /* ---- the debug handle (§10.7) + the SCALE.md measurement rig. ------------------------- */
  (window as unknown as Record<string, unknown>)['__starscape'] = {
    renderer,
    scene,
    camera,
    field,
    graph,
    synth: V,
    cam,
    castAt,
    store,
    panel,
    provider,
    configWarnings,
    // the routing table's enumeration, inspectable: live keys route this frame, boot keys land
    // on the next reload (createField focus opts), unrouted keys have no live seam this version.
    routes: { live: Object.keys(ROUTE), boot: BOOT_KEYS, unrouted: UNROUTED_KEYS },
    settled: (): boolean => settledNow,
    // live flycam telemetry (surface floor + altitude speed-scale observability, stage A)
    get nearMesh() {
      return nearMesh;
    },
    camInfo: (): { speed: number; altitude: number; room: number; blocked: boolean; radius: number } => ({
      speed,
      altitude,
      room,
      blocked,
      radius: cam.radius,
    }),
    fpsMedian,
    jsFrameMs,
    bootMs: (): number => bootMs,
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
    measure: (
      frames = 90,
    ): Promise<{
      msMedian: number;
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
        let k = -1;
        let prev = 0;
        const tick = (tt: number): void => {
          if (k >= 0) buf[k] = tt - prev;
          prev = tt;
          k++;
          if (k < frames) {
            requestAnimationFrame(tick);
            return;
          }
          const sorted = Array.from(buf).sort((a, b) => a - b);
          const ms = sorted[frames >> 1]!;
          const s = field.stats();
          const handle = (window as unknown as Record<string, Record<string, () => number>>)[
            '__starscape'
          ]!;
          resolve({
            msMedian: ms,
            fps: ms > 0 ? 1000 / ms : 0,
            jsMedian: jsFrameMs(),
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
    setMode: (m: 'frozen' | 'custom'): void => {
      ui.mode.value = m;
      onModeChange();
    },
    applyHierarchyPreset: (name: string): boolean => {
      if (HPRESETS[name] === undefined) return false;
      if (mode !== 'custom') {
        ui.mode.value = 'custom';
        onModeChange();
      }
      ui.hpreset.value = name;
      onPresetChange();
      return true;
    },
    get levelRows() {
      return rows;
    },
    setLevelRows: (rs: LevelRow[]): void => {
      rows = rs;
      ui.hpreset.value = '';
      saveLevels();
      renderLevels();
      if (mode === 'custom') applyCustomNow(false);
    },
    get hierarchy() {
      return hierarchy;
    },
    get pivotOverflow() {
      return pivotOverflow;
    },
    setProjection: (mode2d: '2d' | '3d'): void => {
      ui.proj.value = mode2d;
      applyProjection();
    },
    setSpin: (level: number, on: boolean, rate?: number): void => {
      const r = rows[level];
      if (r === undefined) return;
      r.spin = on;
      if (rate !== undefined) r.spinRate = rate;
      saveLevels();
      renderLevels();
      applySpins();
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
      setGlow(name === 'nebula');
      applyAll();
      syncLook();
      return true;
    },
    setGlow: (on: boolean): void => setGlow(on),
    setBloom: (p: { strength?: number; radius?: number; threshold?: number }): void => {
      if (p.strength !== undefined) {
        glow.strength = p.strength;
        ui.glowStrength.value = String(p.strength);
      }
      if (p.radius !== undefined) {
        glow.radius = p.radius;
        ui.glowRadius.value = String(p.radius);
      }
      if (p.threshold !== undefined) {
        glow.threshold = p.threshold;
        ui.glowThreshold.value = String(p.threshold);
      }
      applyGlowParams();
      saveGlow();
    },
    get glow() {
      return { on: glow.on, ...post.bloom, tonemap: post.tonemap };
    },
    post,
    get ladder() {
      return ladder;
    },
    get overlay() {
      return ladderOverlay;
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
      cursorMoved = true;
      if (!fieldSettledNow && !morphing) refreshViewOf(i);
      const tv = new Vector3(viewBuf[i * 3]!, viewBuf[i * 3 + 1]!, viewBuf[i * 3 + 2]!);
      fly.goTo(tv, Math.max(spacings * spacing * field.spread, 0.01), ms);
    },
    flyToPage: (id: string, spacings = 0.4): boolean => {
      const i = V.byId.get(id);
      if (i === undefined) return false;
      field.setSelect(i);
      paintSelected(i);
      const handle = (window as unknown as Record<string, Record<string, (a: number, b?: number) => void>>)[
        '__starscape'
      ]!;
      handle['flyToNode']!(i, spacings);
      return true;
    },
    setTurntable: (on: boolean): void => {
      if (fly) fly.spin = on;
    },
    // terrain stage B: toggle the displaced-terrain feature (forces its prereqs — a texture facet +
    // procedural detail — and turns 3D nodes on, so one call is enough to see it).
    setTerrain: (on: boolean): void => {
      if (on) {
        ensureTerrainPrereqs(); // sets the texture facet + procedural detail (read by a fresh ensureNear)
        if (!ui.near.checked) {
          ui.near.checked = true;
          ui.nearbox.classList.add('on');
        }
        ensureNear().setEnabled(true);
        ui.nearTerrain.checked = true;
        nearMesh?.setTerrain(true);
      } else {
        ui.nearTerrain.checked = false;
        nearMesh?.setTerrain(false);
      }
      updateStats();
      onNearEdit(); // the debug toggle changes the same scene group — persist + diverge to custom
    },
    terrainState: (): string =>
      JSON.stringify({
        active: nearMesh?.terrainActive ?? false,
        subdiv: nearMesh?.terrainSubdiv ?? 0,
        triangles: nearMesh?.terrainTriangles ?? 0,
        promoted: nearMesh?.promoted ?? 0,
        drawCalls: nearMesh?.drawCalls ?? 0,
        altitude: Number.isFinite(altitude) ? Number(altitude.toFixed(3)) : null,
        blocked,
        fps: Number(fpsMedian().toFixed(1)),
      }),
    pinImage: (i: number): void => pinImageNode(i),
    nearImageState: (): string =>
      JSON.stringify({
        pinned: pinnedImageNode,
        meshes: nearMesh?.imageMeshes ?? 0,
        drawCalls: nearMesh?.drawCalls ?? 0,
        cacheBytes: nearMesh?.imageCacheBytes ?? 0,
        info: nearMesh ? nearMesh.imageInfo.map((x) => ({ ...x })) : [],
      }),
  };

  updateStats();
}

main().catch((err: unknown) => {
  // §10.5: a boot failure surfaces in the UI, never an idle page with an empty console.
  errorEl.style.display = 'block';
  errorEl.textContent = `boot failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`;
  statsEl.textContent = 'boot failed';
});
