/* createField — the field renderer (DESIGN.md §5, stage 3; §4 transform pipeline, stage 6).
 *
 * Two primary draw calls whatever the corpus size: one instanced quad for all nodes, one
 * batched line list for all links. The returned objects attach to any three.js scene; the
 * caller owns renderer, scene and camera (the library never fetches, never owns the page).
 *
 * Stage 6 adds the §4 pipeline, evaluated per vertex on the GPU:
 *
 *     p = spread( chain( mix(A, B, t) ) )
 *
 *   - MORPH: A and B are two full position sets in a shared store (storage buffers on
 *     WebGPU, DataTextures on the WebGL2 fallback — ./stores owns the seam); `t` is one
 *     uniform driven by the tween system. `setTarget`/`setHierarchy` retarget with the
 *     standard bake: current = mix(A,B,t_now) is written into A (CPU, once, on the event),
 *     the store uploads ONCE, t resets to 0 and tweens to 1. No snap, by construction.
 *   - FRAME CHAINS: each node carries one chain slot (morph.ts packChainIds — the chain
 *     itself is composed per PIVOT by the FrameSet, frames.ts rule 1), and the shader
 *     applies the composed transform as one quaternion+offset read from the pivot store.
 *     Per-frame CPU is O(pivots): advance + one small texture upload. CPU rewrites of
 *     node/link buffers during spin DO NOT EXIST in this design — the predecessor flaw
 *     (SCALE.md "Rotation re-uploads the link buffer") is structurally closed.
 *   - SPREAD: a uniform scale about the field centroid, in-shader. Spread back to 1.0 is
 *     byte-exact on every CPU reference path (morph.ts spreadPoint). The centre itself
 *     rides the morph tween (old target's centroid → new target's), so a retarget under
 *     spread ≠ 1 keeps "no snap" true — an instant centre swap would displace everything
 *     by (1−spread)·Δcentroid.
 *   - PROJECTION (§3.6): '2d' is a second morph to a flattened target (one axis dropped
 *     over the live arrangement, the 3D target kept pristine), tweened the same way.
 *     Dimensionality is a destination, not a branch.
 *
 * No per-node JS objects anywhere (§10.2): every channel is a typed array written in place.
 */

import { AdditiveBlending, NormalBlending, type Object3D } from 'three/webgpu';
import type { Graph } from '../core/types';
import { createTweens } from '../anim/tween';
import { createFrameSet } from '../anim/frames';
import type { FrameSet } from '../anim/frames';
import type { CompiledHierarchy } from '../place/hierarchy';
import {
  fieldExtent,
  linkDrawCount,
  sizesFromColumn,
  uniformSizes,
} from './channels';
import { facetPalette, paintFacetColors } from './palette';
import { localDensity } from './density';
import { createTonemap, OP_INDEX } from './tonemap';
import type { TonemapOp } from './tonemap';
import { DEFAULT_RAMP, rampColors, rampColorsTied } from '../theme/palettes';
import type { RampLike } from '../theme/palettes';
import {
  BG_DARK,
  DEFAULT_FOV,
  D_REF_FRACTION,
  FOG_FAR,
  FOG_NEAR,
  LINK_BASE_ALPHA,
  LINK_LEN_SOFT,
} from './constants';
import { buildNodeLayer } from './nodes';
import type { NodeStyle } from './nodes';
import { buildLinkLayer } from './links';
import type { LinkStyle } from './links';
import { createFocus } from './focus';
import type { FocusOpts } from './focus';
import { CHAIN_CAP, bakeMix, centroidOf, flattenAxis, packChainIds, spreadPoint } from './morph';
import { createGpuTransform, createPivotStore, createPositionStore } from './stores';

/** Default morph/projection tween duration, seconds. */
export const MORPH_DURATION = 0.9;

export interface FieldOpts {
  /** Numeric column driving node size (rank-normalised). 'uniform' for equal dots.
   *  Default 'degree'. */
  sizeColumn?: string;
  /** Facet driving node colour. Default: the graph's first facet (uniform grey without any). */
  colorFacet?: string;
  /** 'additive' (glows that stack — the dark-ground preset look) or 'normal' (solid glyphs,
   *  correct on any ground). Blending is a property of the GROUND, not a per-material choice —
   *  both layers follow this one setting. Default 'normal', the predecessor's shipped default
   *  even on its dark ground: untonemapped additive piles to white at 10⁴⁺ density, and the
   *  tonemap curves that make the glow look viable arrive with theming (stage 8). */
  blend?: 'additive' | 'normal';
  /** Vertical field of view in degrees — must match the camera's. Default 55. */
  fov?: number;
  /** Viewport height in CSS px (one pixel unit; devicePixelRatio is the renderer's business).
   *  Keep it current via resize(). Default 800. */
  heightCss?: number;
  /** Initial link budget, in links. Default: all of them. */
  linkBudget?: number;
  /** Focus levels (hover/selection alphas, dim level) — see focus.ts for each default. */
  focus?: FocusOpts;
  /** Vertex-pull backend for the shared position store (stage 6): 'storage' only when the
   *  renderer actually booted the WebGPU backend; 'texture' (the default) works on both
   *  backends. See ./stores for the seam. */
  pull?: 'storage' | 'texture';
  /** Pivot-store slot capacity (frame-chain pivots + 1). Default stores.PIVOT_CAPACITY. */
  pivotCapacity?: number;
  /** Which axis the '2d' projection drops. Default 1 (y — flatten onto the ground plane). */
  flatAxis?: 0 | 1 | 2;
  /** Default morph/projection tween duration, seconds. Default MORPH_DURATION. */
  morphDuration?: number;
  /** Initial GLYPH style (boot path, so a saved look applies on load): the solid-body
   *  fraction, halo width/strength, core heat, px clamp rails and the distance falloff. Each
   *  field falls back to its constants.ts default; all are live afterwards via setNodeStyle. */
  nodeStyle?: NodeStyle;
  /** Initial LINK style (boot path): the long-edge fade length (fraction of the diagonal) and
   *  the floor the longest edges keep. Defaults from constants.ts; live via setLinkStyle. */
  linkStyle?: LinkStyle;
}

export interface FieldStats {
  nodes: number;
  /** The budget report (§10.4): silent truncation lies, so what is drawn vs what exists is
   *  always available. */
  links: { drawn: number; total: number };
  sizeColumn: string;
  colorFacet: string | null;
  /** The numeric column colour is bound to (setColorColumn), or null when a facet or the density
   *  channel drives it — at most one of colorFacet / colorColumn is non-null (both null when
   *  density colours the field, where colorDensity is true instead). */
  colorColumn: string | null;
  /** True when LOCAL SPATIAL DENSITY colours the field (setColorByDensity) — colorFacet and
   *  colorColumn are both null then. The density is recomputed from the target geometry on every
   *  retarget, so this look follows an arrangement change without the caller re-binding it. */
  colorDensity: boolean;
  /** The live highlight roll-off: operator name + pre-curve exposure (stage 8, ./tonemap). */
  tonemap: { op: TonemapOp; exposure: number };
  /** Morph progress (§4): t and whether any tween (morph or spread) is still running. */
  morph: { t: number; active: boolean };
  projection: '2d' | '3d';
  spread: number;
  /** Frame-chain pivots currently installed (0 = no hierarchy). */
  pivots: number;
}

export interface Field {
  /** Scene-attachable objects — add every one to your scene (links first keeps the draw
   *  order explicit, though renderOrder already enforces it). */
  objects: Object3D[];
  /** Draw only the k most important links (the ordered buffer makes "first k" = "best k").
   *  Returns the count actually drawn. */
  setLinkBudget(k: number): number;
  /** Re-bind node colour to a facet; links take their endpoints' colours with it. */
  setColorFacet(key: string): void;
  /** Re-bind node colour to a numeric column through a ramp (stage 8, §7): rank-normalised
   *  percentile → ramp colour — the same rank machinery the size channel uses, for the same
   *  reason (a linear map leaves a heavy-tailed column on the ramp's floor). Links take
   *  their endpoints' colours with it. Default ramp DEFAULT_RAMP (theme/palettes). */
  setColorColumn(key: string, ramp?: RampLike): void;
  /** Re-bind node colour to LOCAL SPATIAL DENSITY (field/density.ts): a per-node count of how
   *  crowded its neighbourhood is in the CURRENT target geometry, rank-normalised through the ramp
   *  (tie-collapsed, so equally-sparse nodes share one cool colour) — crowded regions hot, sparse
   *  ones cool, the nebula look's defining signal. Unlike setColorColumn this binds to GEOMETRY,
   *  not a data column, so it is recomputed automatically whenever the arrangement changes
   *  (setHierarchy / setTarget / setProjection). Links take their endpoints' colours with it.
   *  Default ramp 'nebula'. */
  setColorByDensity(ramp?: RampLike): void;
  /** Select the highlight roll-off operator for BOTH layers ('none' bypasses — the raw
   *  pipeline, exposure included) and optionally the pre-curve exposure (stage 8, §7). */
  setTonemap(op: TonemapOp, exposure?: number): void;
  /** Re-bind node size to a numeric column ('uniform' for equal dots). */
  setSizeColumn(key: string): void;
  /** One blending setting for both layers (see FieldOpts.blend). */
  setBlend(mode: 'additive' | 'normal'): void;
  /** Report the viewport's CSS-pixel height after a resize (one pixel unit, everywhere). */
  resize(heightCss: number): void;
  /** Report a camera fov change (degrees). */
  setFov(fov: number): void;
  /** Scale how far the field fades out — the depth haze that decides how far you can see.
   *  The node/link fog near+far planes become their creation defaults (derived from the
   *  field diagonal) × `mult`; 1 is the default, larger sees further. Returns the resulting
   *  near/far world distances so the caller can push its own camera far-plane past `far`
   *  (the camera is the consumer's — the field only owns the haze). Live uniform write, no
   *  recompile. */
  setViewDistance(mult: number): { near: number; far: number };
  /** Flat-glyph threshold, CSS px (stage 9 LOD): glyphs projected under this draw the cheap
   *  no-halo branch. The px clamp pins a distant field at NODE_MIN_PX, so the boot quality
   *  notch (quality.ts) must lift this OVER the clamp floor before the branch fires at all.
   *  A simplification, not a cull — nothing is dropped, so no stats row (§10.4). */
  setFlatPx(px: number): void;
  /** Node GLYPH style — the seven knobs that were compile-time constants until stage 10, all
   *  GLOBAL (one value for the whole field) and all live (a uniform `.value` write, no
   *  recompile, like setFov/setFlatPx): the solid-body fraction (`bodyR`), halo width in CSS
   *  px (`haloPx`) and strength (`haloGain`), how hot the core burns (`coreHot`), the
   *  projected-radius clamp rails in CSS px (`minPx`/`maxPx`) and the perspective falloff
   *  (`falloff`, 1 = true projection). Pass only the fields you are changing; omitted (or
   *  non-finite) fields are left untouched. Note the px clamp is also baked into the picker's
   *  hit disc (pick/grid.ts constants) — moving `minPx`/`maxPx`/`falloff` far from their
   *  defaults drifts the clickable disc from the drawn one. */
  setNodeStyle(style: NodeStyle): void;
  /** Link style — the long-edge fade. `lenSoft` is the fade length as a FRACTION of the field
   *  diagonal (the field multiplies by its diagonal internally, matching LINK_LEN_SOFT);
   *  `lenFloor` (0..1) is the fraction of itself the longest edge keeps. Both live uniform
   *  writes. Pass only the fields you are changing. */
  setLinkStyle(style: LinkStyle): void;
  /** Hover node i (-1 clears): lights it, its CSR neighbourhood and its incident links.
   *  Repaints only the hovered degree — never the whole colour buffer (DESIGN.md §6).
   *  Returns true when the state actually changed. */
  setHover(i: number): boolean;
  /** Select node i (-1 clears). Independent of hover: dims everything outside the node's
   *  neighbourhood to the focus `dimOthers` level. Full sweep, on an event — never per
   *  frame. Returns true when the state actually changed. */
  setSelect(i: number): boolean;
  /** Hovered node index, -1 for none. */
  readonly hover: number;
  /** Selected node index, -1 for none. */
  readonly select: number;
  /** Node i's size-channel world radius — what the picker must feed pickRay's `sizeOf` so
   *  the hit disc is the drawn disc (one pixel unit, shared constants). Boost excluded:
   *  pair it with boostOf, which the shader applies to the radius AND the clamp rails. */
  sizeOf(i: number): number;
  /** Node i's live hover/selection boost (1 resting) — feed pickRay's `boostOf`, or the
   *  emphasised node's hit disc stays the resting disc while its drawn disc grows. */
  boostOf(i: number): number;
  stats(): FieldStats;
  /** The live extent the field was built from — fit the camera to this, never to a stored
   *  number (§10.8). After retargets, frame the TARGET: see targetExtent(). */
  extent: { centre: [number, number, number]; radius: number; diagonal: number };
  /** The background colour the default theme assumes (0x05070d on 'additive'). */
  background: number;

  /* ---- stage 6: the §4 transform pipeline -------------------------------------------- */

  /** Advance the animation clock by `dt` seconds: tweens (morph t, spread) and the frame
   *  set. O(pivots) — the ONLY per-frame CPU this field performs, and zero when idle.
   *  Returns true when anything moved (positions on screen changed this tick). */
  update(dt: number): boolean;
  /** Morph to a raw position set (3n). The standard retarget bake: the current blend is
   *  written into A, the store uploads once, t resets and tweens to 1. Any installed
   *  frame chains stay as they are — use setHierarchy to swap frames with the target. */
  setTarget(positions: Float32Array, duration?: number): void;
  /** Morph to a compiled hierarchy: its positions become the target; when any level
   *  declares `spin` (chainDepth > 0) its pivots become the frame chains (fresh
   *  FrameSet, identity until spun — declare `spin: { rate: 0 }` for a frame that
   *  exists but idles, the hierarchy.ts escape hatch; with NO spin declared no FrameSet
   *  installs, so a later rate write cannot spin a per-tick upload loop with zero
   *  visual effect). `null` restores the graph's frozen layout with no frames. The
   *  round trip is byte-exact on basePositions(). */
  setHierarchy(h: CompiledHierarchy | null, duration?: number): void;
  /** Projection (§3.6): '2d' morphs to the live target flattened along flatAxis; '3d'
   *  morphs back to the pristine 3D target. Byte-exact round trip on basePositions(). */
  setProjection(mode: '2d' | '3d', duration?: number): void;
  /** Spread: uniform scale about the field centroid, in-shader. duration 0 (default)
   *  sets it immediately; a duration tweens it. */
  setSpread(k: number, duration?: number): void;
  /** True when nothing is animating: no tween running, no frame moved last tick. The
   *  moment to rebuild a pick grid (DESIGN §6: on morph completion, never per frame). */
  settled(): boolean;
  /** Current morph progress t ∈ [0, 1]. */
  readonly morphT: number;
  /** Current spread factor. */
  readonly spread: number;
  /** Current projection mode. */
  readonly projection: '2d' | '3d';
  /** The live FrameSet (null when no hierarchy is installed) — for telemetry
   *  (frameOfPoint), camera rides (rideDelta) and CPU reference transforms. */
  readonly frames: FrameSet | null;
  /** Per-node chain pivot ids (−1 = world) — frames.transformPoint(chainIds[i], …) is the
   *  CPU reference for what the shader does to node i. Read-only by contract. */
  readonly chainIds: Int32Array;
  /** The current BASE positions — mix(A, B, t) with exact endpoints (§4: a completed
   *  round trip returns byte-identical coordinates, FNV-comparable). Allocates unless
   *  `out` (3n) is given. */
  basePositions(out?: Float32Array): Float32Array;
  /** The current VIEW positions — base → chain → spread, the CPU mirror of the vertex
   *  shader (via frames.transformPoint + morph.spreadPoint). O(n): call on events (grid
   *  rebuilds), never per frame. */
  viewPositions(out?: Float32Array): Float32Array;
  /** The extent of the current TARGET (B), spread included — what a camera should frame
   *  after a retarget (§10.8: the live extent, never a stored one). */
  targetExtent(): { centre: [number, number, number]; radius: number; diagonal: number };
  /** The spread centre the shader is using NOW (what its spread scales about): the
   *  current target's centroid once a morph lands, and mix(previous, next) while one
   *  runs — the centre MIGRATES with the tween so a retarget under spread ≠ 1 cannot
   *  snap (§4). Needed to map a world point back to unspread space
   *  (p′ = c + (p − c)/spread) for frameOfPoint telemetry. */
  centroid(): [number, number, number];

  dispose(): void;
}

/** Build the two-draw-call field for a compiled Graph. The graph must carry a frozen layout —
 *  it is the boot arrangement and what `setHierarchy(null)` restores. */
export function createField(graph: Graph, opts: FieldOpts = {}): Field {
  const n = graph.n;
  const positions = graph.positions;
  if (!positions) {
    throw new Error(
      'createField: graph has no positions — a frozen layout is required (compileHierarchy can make one)',
    );
  }

  const fov = opts.fov ?? DEFAULT_FOV;
  const heightCss = opts.heightCss ?? 800;
  const extent = fieldExtent(positions, n);
  const diagonal = extent.diagonal || 1;
  const morphDuration = opts.morphDuration ?? MORPH_DURATION;
  const flatAxis = opts.flatAxis ?? 1;

  /* ---- channels ---------------------------------------------------------------------- */
  let sizeColumn = opts.sizeColumn ?? (graph.columns['degree'] ? 'degree' : 'uniform');
  const firstFacet = Object.keys(graph.facets)[0] ?? null;
  let colorFacet = opts.colorFacet ?? firstFacet;
  if (colorFacet !== null && !graph.facets[colorFacet]) {
    throw new Error(`createField: colorFacet "${colorFacet}" names no facet`);
  }

  let colorColumn: string | null = null; // set by setColorColumn; exclusive with colorFacet
  /* Local-density colour state (setColorByDensity). densityRamp non-null ⟺ the density channel
     drives the field (colorFacet and colorColumn both null then); densScratch is the per-node
     count buffer, reused across recomputes so a retarget recolour allocates nothing node-shaped
     (§10.2). The density is a function of the TARGET geometry (baseB) alone — rotation preserves
     distances and spread is a uniform scale the grid's own extent cancels, so neither changes the
     rank — which is why one recompute per retarget suffices, never per frame (§4). */
  let densityRamp: RampLike | null = null;
  let densScratch: Float32Array | null = null;
  const colors = new Float32Array(n * 3);
  const shapes = new Float32Array(n); // all discs at stage 3; shapeBy binding arrives later
  const sizes = computeSizes(graph, sizeColumn, n);
  paintColors(graph, colorFacet, n, colors);

  /* Stage 8: the shared highlight roll-off rig — ONE op/exposure uniform pair spliced into
     both layers' fragment stages (./tonemap). Off (op 0) by default: the bypass is the
     stage-6 pipeline untouched. */
  const tonemap = createTonemap();
  let tonemapOp: TonemapOp = 'none';

  /* ---- stage 6 state: the §4 pipeline's CPU mirrors ----------------------------------- *
   * baseA/baseB are the endpoint sets A and B; target3d is the PRISTINE 3D target the '2d'
   * flatten derives from (stages compose, nothing mutates in place — §4). All retargets
   * are events; nothing here is touched per frame. */
  const baseA = new Float32Array(positions);
  const baseB = new Float32Array(positions);
  const target3d = new Float32Array(positions);
  let chainIds: Int32Array = new Int32Array(n).fill(-1);
  let frames: FrameSet | null = null;
  let world0: Float32Array | null = null;
  let pivotCount = 0;
  let projection: '2d' | '3d' = '3d';

  /* The spread centre rides the morph (§4 continuity): the shader's world is
   * c + (chain(base) − c)·spread, so swapping c in one tick while spread ≠ 1 would
   * displace every node by (1−s)·(c_new − c_old) — a snap the bake cannot prevent.
   * Instead cA holds the centroid in effect when a retarget fires (the live value — the
   * continuity anchor) and cB the new target's centroid; the live `centroid` is
   * mix(cA, cB, t) with exact endpoints, driven by the same morph tween. */
  const cA = centroidOf(baseB, n);
  const cB: [number, number, number] = [cA[0], cA[1], cA[2]];
  const centroid: [number, number, number] = [cA[0], cA[1], cA[2]];

  const tweens = createTweens();
  tweens.set('morph', 0);
  tweens.set('spread', 1);

  const posStore = createPositionStore(n, opts.pull ?? 'texture');
  const pivStore = createPivotStore(opts.pivotCapacity);
  const transform = createGpuTransform(posStore, pivStore);
  /** centroid ← mix(cA, cB, v), exact at both endpoints, f32-rounded between (the GPU
   *  uniform is f32 — the CPU mirror must agree with the shader on the centre). */
  const mixCentroid = (v: number): void => {
    if (v <= 0) {
      centroid[0] = cA[0];
      centroid[1] = cA[1];
      centroid[2] = cA[2];
    } else if (v >= 1) {
      centroid[0] = cB[0];
      centroid[1] = cB[1];
      centroid[2] = cB[2];
    } else {
      centroid[0] = Math.fround(cA[0] + (cB[0] - cA[0]) * v);
      centroid[1] = Math.fround(cA[1] + (cB[1] - cA[1]) * v);
      centroid[2] = Math.fround(cA[2] + (cB[2] - cA[2]) * v);
    }
  };
  tweens.onChange('morph', (v) => {
    transform.uniforms.morphT.value = v;
    mixCentroid(v);
    syncCentroid();
  });
  tweens.onChange('spread', (v) => {
    transform.uniforms.spread.value = v;
  });
  const syncCentroid = (): void => {
    transform.uniforms.centroid.value.set(centroid[0], centroid[1], centroid[2]);
  };
  syncCentroid();
  pivStore.sync(null, null);
  posStore.upload(baseA, baseB, chainIds, n);

  /* The retarget machinery. bakeBase folds the running morph into A (chain NOT included —
     the chain stays live in the shader unless the frames themselves are being replaced,
     in which case the parked chain displacement is baked in so the new identity frames
     start exactly where the old ones left the field). */
  const bakeBase = (): void => {
    bakeMix(baseA, baseB, tweens.get('morph'));
  };
  const bakeChainIntoA = (): void => {
    if (frames !== null && frames.displaced()) {
      for (let i = 0; i < n; i++) {
        frames.transformPoint(chainIds[i]!, baseA, baseA, i * 3, i * 3);
      }
    }
  };
  const finishRetarget = (duration: number): void => {
    if (projection === '2d') {
      const c3 = centroidOf(target3d, n);
      flattenAxis(target3d, n, flatAxis, c3[flatAxis], baseB);
    } else {
      baseB.set(target3d);
    }
    // No snap, by construction — spread included: the spread centre HOLDS at its live
    // value through the event (cA ← current centroid) and migrates to the new target's
    // centroid (cB) with the morph tween. Writing cB to the uniform here instead would
    // displace every node by (1−spread)·(cB − cA) in one tick whenever spread ≠ 1.
    cA[0] = centroid[0];
    cA[1] = centroid[1];
    cA[2] = centroid[2];
    const c = centroidOf(baseB, n);
    cB[0] = c[0];
    cB[1] = c[1];
    cB[2] = c[2];
    pivStore.sync(frames, world0); // O(pivots)
    posStore.upload(baseA, baseB, chainIds, n); // the ONE upload of the retarget
    tweens.set('morph', 0); // emits → centroid ← cA (already cA if t was 0: no emit needed)
    tweens.to('morph', 1, { duration });
    // The density channel depends on the TARGET geometry, which just changed: recolour from the
    // new baseB (O(n) density, an event cost — never per frame). Colours snap to the destination's
    // density while the positions morph into it; the brief staleness is the accepted §4 seam.
    if (densityRamp !== null) recolorDensity();
  };

  /* ---- layers ------------------------------------------------------------------------ */
  const link = buildLinkLayer(graph.links, transform, {
    fogNear: diagonal * FOG_NEAR * 0.8,
    fogFar: diagonal * FOG_FAR * 0.9,
    // lenSoft is a FRACTION of the diagonal (constant or boot override) → world length here.
    lenSoft: diagonal * (opts.linkStyle?.lenSoft ?? LINK_LEN_SOFT),
    lenFloor: opts.linkStyle?.lenFloor,
    tonemap,
  });

  /* The focus state (hover/select — focus.ts). It owns the iAlpha/iBoost node channels and
     every link ALPHA; link RGB stays the facet colour, written below. The writer targets the
     two alpha slots of edge e directly, which is what makes a hover repaint O(degree): the
     RGB lanes are untouched, so nothing recolours.

     THE GPU TRANSFER MUST BE O(degree) TOO. A bare `needsUpdate = true` with no update
     ranges re-uploads the whole attribute — at 10⁶ nodes that is ~100 MB of link colours
     pushed across the bus on every pointermove frame, which is the predecessor's hover cost
     this rewrite exists to remove (DESIGN.md §6: hover lights the neighbourhood WITHOUT
     repainting the field). So while a hover repaint runs, the writer mirrors each touched
     edge into a BufferAttribute update range (slots 3..7: both alphas; the RGB lanes between
     them re-upload their own current bytes, which costs less than two ranges per edge).

     ONE DISCIPLINE, EVERYWHERE: after boot, every upload of the focus-owned attributes is
     expressed as ranges — full sweeps (selection, recolour) as ONE range covering the whole
     buffer, never as a bare version bump. Mixing the two is order-dependent: a select
     followed by a hover in the same frame would leave small ranges pending, and the renderer
     would upload only those, silently dropping the sweep. With ranges always present, any
     same-frame interleaving uploads the union. Both backends consume and then clear
     `updateRanges` on upload. */
  const lc = link.colors;
  /* Edge-upload tracking while a hover repaint runs. 'edges' mirrors every touched edge
     into its own update range (small footprints); 'span' folds them into one min..max
     window — the past-RANGE_CAP fallback, which keeps even a hub hover's transfer
     bounded by the touched slots' extent instead of the whole buffer (§6: hover lights
     the neighbourhood WITHOUT repainting the field — at 10⁶ links a full re-upload is
     ~32 MB of bus traffic per pointermove, the exact cost this discipline exists to
     remove). eMin/eMax accumulate in BOTH modes: they also answer "was any edge touched
     at all", so a hover between isolated nodes never version-bumps the attribute with
     zero ranges pending (which would upload everything). */
  let edgeTrack: 'off' | 'edges' | 'span' = 'off';
  let eMin = 0;
  let eMax = -1;
  const focus = createFocus(
    n,
    graph.csr,
    graph.links,
    (e: number, alpha: number): void => {
      lc[e * 8 + 3] = alpha;
      lc[e * 8 + 7] = alpha;
      if (edgeTrack !== 'off') {
        if (edgeTrack === 'edges') link.colorAttr.addUpdateRange(e * 8 + 3, 5);
        if (e < eMin) eMin = e;
        if (e > eMax) eMax = e;
      }
    },
    { linkBase: LINK_BASE_ALPHA, ...opts.focus },
  );

  /** One range per touched buffer slot stays cheap for real degrees; past this many touched
   *  edges the per-edge ranges would cost thousands of tiny writeBuffer calls, so the
   *  repaint coalesces into ONE min..max span per buffer instead. */
  const RANGE_CAP = 1024;

  const csr = graph.csr;
  const degreeOf = (i: number): number => (i < 0 ? 0 : csr.off[i + 1]! - csr.off[i]!);
  /** Boost update ranges for node i's hover footprint: itself plus its CSR neighbours. */
  const addBoostRanges = (i: number): void => {
    if (i < 0) return;
    node.boostAttr.addUpdateRange(i, 1);
    for (let k = csr.off[i]!; k < csr.off[i + 1]!; k++) {
      node.boostAttr.addUpdateRange(csr.nbr[k]!, 1);
    }
  };
  /** min..max node slot a hover move prev→next touches: both nodes, their CSR
   *  neighbourhoods, and the selected node (setHover re-asserts its boost). The span
   *  form of addBoostRanges, for the past-RANGE_CAP fallback. Writes [lo, hi] into
   *  `boostLoHi`; hi < lo means nothing touched (cannot happen when the hover changed). */
  const boostLoHi: [number, number] = [0, -1];
  const boostSpan = (prev: number, next: number): void => {
    let lo = n;
    let hi = -1;
    for (const j of [prev, next]) {
      if (j < 0) continue;
      if (j < lo) lo = j;
      if (j > hi) hi = j;
      for (let k = csr.off[j]!; k < csr.off[j + 1]!; k++) {
        const nb = csr.nbr[k]!;
        if (nb < lo) lo = nb;
        if (nb > hi) hi = nb;
      }
    }
    const sel = focus.select;
    if (sel >= 0) {
      if (sel < lo) lo = sel;
      if (sel > hi) hi = sel;
    }
    boostLoHi[0] = lo;
    boostLoHi[1] = hi;
  };
  const fullRange = (attr: { clearUpdateRanges(): void; addUpdateRange(s: number, c: number): void }, len: number): void => {
    attr.clearUpdateRanges();
    if (len > 0) attr.addUpdateRange(0, len);
  };

  const node = buildNodeLayer(n, transform, colors, sizes, shapes, focus.nodeAlpha, focus.nodeBoost, {
    fov,
    heightCss,
    dRef: diagonal * D_REF_FRACTION,
    fogNear: diagonal * FOG_NEAR,
    fogFar: diagonal * FOG_FAR,
    tonemap,
    style: opts.nodeStyle,
  });

  const totalLinks = graph.links.length >> 1;
  let drawn = linkDrawCount(totalLinks, opts.linkBudget ?? totalLinks);
  link.setDrawCount(drawn);

  /* Link colours: an edge takes the colour of the nodes it joins, so recolouring the field
     recolours its links for free and the two can never disagree. All m links are painted
     (the buffer is static; painting only the draw range would hand a later budget-grow
     stale zeroes — the exact bug the predecessor had to re-derive its prefix to avoid).
     RGB only: every alpha slot belongs to the focus state, so a recolour cannot flatten a
     live hover or selection. */
  const writeLinkColors = (): void => {
    const ls = graph.links;
    for (let e = 0; e < totalLinks; e++) {
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
    fullRange(link.colorAttr, lc.length); // a sweep is ONE range — see the focus comment
    link.colorAttr.needsUpdate = true;
  };
  writeLinkColors();
  focus.repaint(); // seed every alpha slot (resting link alpha, nodes lit)

  /** Recompute the local-density colouring from the current TARGET geometry (baseB) and repaint
   *  nodes + links. Called by setColorByDensity and, while the density channel is active, by every
   *  retarget (finishRetarget). O(n); an event cost, never per frame. */
  const recolorDensity = (): void => {
    densScratch = localDensity(baseB, n, densScratch ?? undefined);
    rampColorsTied(densScratch, n, densityRamp ?? 'nebula', colors);
    node.colorAttr.needsUpdate = true;
    writeLinkColors();
  };

  /** mix(A, B, t) with exact endpoints — shared by basePositions and viewPositions
   *  (no `this`: the Field object must survive destructuring). */
  const basePositionsInto = (out?: Float32Array): Float32Array => {
    const dst = out ?? new Float32Array(n * 3);
    dst.set(baseA);
    bakeMix(dst, baseB, tweens.get('morph')); // exact at both endpoints (morph.ts)
    return dst;
  };

  const setBlend = (mode: 'additive' | 'normal'): void => {
    const blending = mode === 'additive' ? AdditiveBlending : NormalBlending;
    for (const mat of [node.material, link.material]) {
      if (mat.blending !== blending) {
        mat.blending = blending;
        mat.needsUpdate = true;
      }
    }
  };
  setBlend(opts.blend ?? 'normal');

  return {
    objects: [link.lines, node.mesh],
    extent,
    background: BG_DARK,

    setLinkBudget(k: number): number {
      drawn = linkDrawCount(totalLinks, k);
      link.setDrawCount(drawn);
      return drawn;
    },

    setColorFacet(key: string): void {
      if (!graph.facets[key]) throw new Error(`setColorFacet: "${key}" names no facet`);
      colorFacet = key;
      colorColumn = null;
      densityRamp = null;
      paintColors(graph, colorFacet, n, colors);
      node.colorAttr.needsUpdate = true;
      writeLinkColors();
    },

    setColorColumn(key: string, ramp: RampLike = DEFAULT_RAMP): void {
      const col = graph.columns[key];
      if (!col) throw new Error(`setColorColumn: "${key}" names no numeric column`);
      colorColumn = key;
      colorFacet = null;
      densityRamp = null;
      rampColors(col.data, n, ramp, colors); // rank-normalised — the sizeOf rank machinery
      node.colorAttr.needsUpdate = true;
      writeLinkColors();
    },

    setColorByDensity(ramp: RampLike = 'nebula'): void {
      colorColumn = null;
      colorFacet = null;
      densityRamp = ramp;
      recolorDensity(); // from the current baseB; recomputed on every retarget hereafter
    },

    setTonemap(op: TonemapOp, exposure?: number): void {
      if (!(op in OP_INDEX)) throw new Error(`setTonemap: "${op}" names no operator`);
      tonemapOp = op;
      tonemap.uniforms.op.value = OP_INDEX[op];
      if (exposure !== undefined) {
        if (!Number.isFinite(exposure) || exposure <= 0) {
          throw new Error(`setTonemap: exposure must be finite and > 0, got ${exposure}`);
        }
        tonemap.uniforms.exposure.value = exposure;
      }
    },

    setSizeColumn(key: string): void {
      if (key !== 'uniform' && !graph.columns[key]) {
        throw new Error(`setSizeColumn: "${key}" names no numeric column`);
      }
      sizeColumn = key;
      node.sizes.set(computeSizes(graph, key, n));
      node.sizeAttr.needsUpdate = true;
    },

    setBlend,

    setHover(i: number): boolean {
      const prev = focus.hover;
      // Hover repaints O(degree) on the CPU; the ranges keep the GPU transfer bounded
      // too. Past RANGE_CAP touched edges, per-edge ranges give way to ONE coalesced
      // min..max span per buffer — proportional to the hover footprint's extent, never
      // the corpus (a full-buffer fallback here would re-pay the predecessor's hover
      // cost on every hub under the cursor). Always ranges, never a bare version bump,
      // so same-frame interleavings with sweeps stay a union.
      const small = degreeOf(prev) + degreeOf(i) <= RANGE_CAP;
      edgeTrack = small ? 'edges' : 'span';
      eMin = totalLinks;
      eMax = -1;
      const changed = focus.setHover(i);
      edgeTrack = 'off';
      if (changed) {
        if (small) {
          addBoostRanges(prev);
          addBoostRanges(i);
          if (focus.select >= 0) node.boostAttr.addUpdateRange(focus.select, 1);
        } else {
          if (eMax >= eMin) {
            // slots eMin*8+3 .. eMax*8+7 inclusive — both alpha lanes of every touched edge
            link.colorAttr.addUpdateRange(eMin * 8 + 3, (eMax - eMin) * 8 + 5);
          }
          boostSpan(prev, i);
          if (boostLoHi[1] >= boostLoHi[0]) {
            node.boostAttr.addUpdateRange(boostLoHi[0], boostLoHi[1] - boostLoHi[0] + 1);
          }
        }
        node.boostAttr.needsUpdate = true;
        if (eMax >= 0) link.colorAttr.needsUpdate = true;
      }
      return changed;
    },

    setSelect(i: number): boolean {
      const changed = focus.setSelect(i);
      if (changed) {
        // A selection is a full sweep on an event — upload it as ONE whole-buffer range
        // (never a bare version bump: see the focus comment for the ordering hazard).
        fullRange(node.alphaAttr, n);
        fullRange(node.boostAttr, n);
        fullRange(link.colorAttr, lc.length);
        node.alphaAttr.needsUpdate = true;
        node.boostAttr.needsUpdate = true;
        link.colorAttr.needsUpdate = true;
      }
      return changed;
    },

    get hover(): number {
      return focus.hover;
    },

    get select(): number {
      return focus.select;
    },

    sizeOf(i: number): number {
      return sizes[i]!;
    },

    boostOf(i: number): number {
      return focus.nodeBoost[i]!;
    },

    resize(h: number): void {
      // CSS pixels, deliberately NOT drawing-buffer pixels — the ladder lays out DOM in CSS
      // px, and clamping in device px would put the two a devicePixelRatio apart.
      node.uniforms.halfH.value = h / 2;
    },

    setFov(f: number): void {
      node.uniforms.tanHalf.value = Math.tan((f * Math.PI) / 360);
    },

    setViewDistance(mult: number): { near: number; far: number } {
      const m = Number.isFinite(mult) && mult > 0 ? mult : 1;
      // nodes fade over [diagonal·FOG_NEAR, diagonal·FOG_FAR]; links fade a touch nearer
      // (×0.8/×0.9, as at creation) so edges recede before the dots they connect.
      const near = diagonal * FOG_NEAR * m;
      const far = diagonal * FOG_FAR * m;
      node.uniforms.fogNear.value = near;
      node.uniforms.fogFar.value = far;
      link.uniforms.fogNear.value = diagonal * FOG_NEAR * 0.8 * m;
      link.uniforms.fogFar.value = diagonal * FOG_FAR * 0.9 * m;
      return { near, far };
    },

    setFlatPx(px: number): void {
      if (!Number.isFinite(px) || px < 0) {
        throw new Error(`setFlatPx: threshold must be finite and >= 0, got ${px}`);
      }
      node.uniforms.flatPx.value = px;
    },

    setNodeStyle(style: NodeStyle): void {
      // Each knob is a live GLOBAL uniform (§10): write the ones supplied, skip the rest. A
      // non-finite slip would blank the whole field (one bad uniform poisons every glyph), so
      // guard each — a defensive no-op is cheaper than a black screen.
      const u = node.uniforms;
      const set = (slot: { value: number }, v: number | undefined): void => {
        if (v !== undefined && Number.isFinite(v)) slot.value = v;
      };
      set(u.bodyR, style.bodyR);
      set(u.haloPx, style.haloPx);
      set(u.haloGain, style.haloGain);
      set(u.coreHot, style.coreHot);
      set(u.minPx, style.minPx);
      set(u.maxPx, style.maxPx);
      set(u.falloff, style.falloff);
    },

    setLinkStyle(style: LinkStyle): void {
      // lenSoft arrives as a FRACTION of the diagonal (matching LINK_LEN_SOFT and the config
      // key) → world length, the same conversion as at build. lenFloor is a bare 0..1.
      if (style.lenSoft !== undefined && Number.isFinite(style.lenSoft)) {
        link.uniforms.lenSoft.value = diagonal * style.lenSoft;
      }
      if (style.lenFloor !== undefined && Number.isFinite(style.lenFloor)) {
        link.uniforms.lenFloor.value = style.lenFloor;
      }
    },

    stats(): FieldStats {
      return {
        nodes: n,
        links: { drawn, total: totalLinks },
        sizeColumn,
        colorFacet,
        colorColumn,
        colorDensity: densityRamp !== null,
        tonemap: { op: tonemapOp, exposure: tonemap.uniforms.exposure.value },
        morph: { t: tweens.get('morph'), active: tweens.active() },
        projection,
        spread: tweens.get('spread'),
        pivots: pivotCount,
      };
    },

    /* ---- stage 6 ----------------------------------------------------------------------- */

    update(dt: number): boolean {
      const wasTweening = tweens.active();
      tweens.update(dt);
      let framesMoved = false;
      if (frames !== null) {
        framesMoved = frames.advance(dt); // O(pivots), free when idle (frames.ts rule 2)
        if (framesMoved) pivStore.sync(frames, world0); // the only per-frame upload
      }
      return framesMoved || wasTweening || tweens.active();
    },

    setTarget(target: Float32Array, duration: number = morphDuration): void {
      if (target.length !== n * 3) {
        throw new Error(`setTarget: expected ${n * 3} floats, got ${target.length}`);
      }
      bakeBase(); // chain stays live — frames are not being replaced
      target3d.set(target);
      finishRetarget(duration);
    },

    setHierarchy(h: CompiledHierarchy | null, duration: number = morphDuration): void {
      if (h !== null && h.positions.length !== n * 3) {
        throw new Error(
          `setHierarchy: hierarchy places ${h.positions.length / 3} nodes, graph has ${n}`,
        );
      }
      bakeBase();
      bakeChainIntoA(); // the OLD frames' parked displacement, folded in before they die
      if (h !== null) {
        target3d.set(h.positions);
        chainIds = packChainIds(h.nodePivot, h.pivots.parent, h.pivots.level, h.chainDepth, CHAIN_CAP);
        /* Frames install only when some level DECLARED spin (chainDepth > 0). With none
           declared, packChainIds parked every node at −1 — a FrameSet could never move a
           node, but its live `rate` channel would still accept writes (frames.ts: "a UI
           spin toggle writes there"), costing an O(pivots) pivot-store upload every tick
           and pinning settled() false for ZERO visual effect. Gating here makes that
           trap unrepresentable: a level that wants its frames live later declares
           `spin: { rate: 0 }` — the "frame exists but idle" form (hierarchy.ts SpinSpec). */
        if (h.chainDepth > 0) {
          frames = createFrameSet(h.pivots);
          world0 = h.pivots.world;
          pivotCount = h.pivots.count;
        } else {
          frames = null;
          world0 = null;
          pivotCount = 0;
        }
      } else {
        target3d.set(positions); // the frozen boot layout, byte-for-byte
        chainIds = new Int32Array(n).fill(-1);
        frames = null;
        world0 = null;
        pivotCount = 0;
      }
      finishRetarget(duration);
    },

    setProjection(mode: '2d' | '3d', duration: number = morphDuration): void {
      if (mode === projection) return;
      bakeBase(); // frames stay live; only the target changes shape
      projection = mode;
      finishRetarget(duration);
    },

    setSpread(k: number, duration = 0): void {
      if (!Number.isFinite(k) || k <= 0) {
        throw new Error(`setSpread: spread must be finite and > 0, got ${k}`);
      }
      if (duration > 0) tweens.to('spread', k, { duration });
      else tweens.set('spread', k);
    },

    settled(): boolean {
      return !tweens.active() && !(frames?.activeThisTick() ?? false);
    },

    get morphT(): number {
      return tweens.get('morph');
    },

    get spread(): number {
      return tweens.get('spread');
    },

    get projection(): '2d' | '3d' {
      return projection;
    },

    get frames(): FrameSet | null {
      return frames;
    },

    get chainIds(): Int32Array {
      return chainIds;
    },

    basePositions(out?: Float32Array): Float32Array {
      return basePositionsInto(out);
    },

    viewPositions(out?: Float32Array): Float32Array {
      const dst = basePositionsInto(out);
      if (frames !== null && frames.displaced()) {
        for (let i = 0; i < n; i++) {
          frames.transformPoint(chainIds[i]!, dst, dst, i * 3, i * 3);
        }
      }
      const s = tweens.get('spread');
      if (s !== 1) {
        for (let i = 0; i < n; i++) spreadPoint(dst, centroid, s, dst, i * 3, i * 3);
      }
      return dst;
    },

    centroid(): [number, number, number] {
      return [centroid[0], centroid[1], centroid[2]];
    },

    targetExtent(): { centre: [number, number, number]; radius: number; diagonal: number } {
      const e = fieldExtent(baseB, n);
      const s = tweens.get('spread');
      if (s === 1) return e;
      // cB, not the live centroid: when the morph lands, the spread scales about the
      // TARGET's centroid — that is the extent a camera framing the destination wants.
      return {
        centre: [
          cB[0] + (e.centre[0] - cB[0]) * s,
          cB[1] + (e.centre[1] - cB[1]) * s,
          cB[2] + (e.centre[2] - cB[2]) * s,
        ],
        radius: e.radius * s,
        diagonal: e.diagonal * s,
      };
    },

    dispose(): void {
      node.dispose();
      link.dispose();
      posStore.dispose();
      pivStore.dispose();
    },
  };
}

function computeSizes(graph: Graph, key: string, n: number): Float32Array {
  if (key === 'uniform') return uniformSizes(n);
  const col = graph.columns[key];
  if (!col) throw new Error(`createField: sizeColumn "${key}" names no numeric column`);
  return sizesFromColumn(col.data, n);
}

function paintColors(
  graph: Graph,
  facetKey: string | null,
  n: number,
  out: Float32Array,
): void {
  if (facetKey === null) {
    // no facets at all: a neutral mid-grey, visible on either ground
    for (let i = 0; i < n * 3; i++) out[i] = 0.62;
    return;
  }
  const facet = graph.facets[facetKey]!;
  paintFacetColors(facet.data, n, facetPalette(facet.values.length), out);
}
