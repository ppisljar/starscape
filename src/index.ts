/** starscape — graph visualization for very large graphs. Further modules arrive in later
 *  stages; exported here: the stage-2 core data model (DESIGN.md §3.1), the stage-3 field
 *  renderer (§5 — two draw calls, TSL materials), the stage-4 interaction base (§6 —
 *  cameras, picking, hover/selection focus), and the stage-5 placement engine (§3.3–3.4 —
 *  the four knobs, their libraries, presets, the hierarchy compiler and the BFS pull).
 *  The synth corpus generator (`src/synth/`) IS part of this surface (below, with the
 *  ladder's makeSynthProvider): consumers need the same deterministic 10²…10⁶ corpora the
 *  demo and the tests run on to exercise an integration at sizes no real dataset reaches. */
export const VERSION = '0.0.0';

export { hash, gauss } from './core/hash';
export { buildCSR } from './core/csr';
export type { CSR } from './core/csr';
export { orderLinks } from './core/linkOrder';
export { buildGraph, neighbours, incident } from './core/graph';
export type { Facet, Graph, GraphInput, NumericColumn } from './core/types';

export { createField, MORPH_DURATION } from './field/field';
export type { Field, FieldOpts, FieldStats } from './field/field';
export { NODE_SHAPES, SHAPE_COUNT } from './field/nodes';
export type { GlyphTextureMode, GlyphTextureParams, NodeStyle, NodeShape } from './field/nodes';
export type { LinkStyle } from './field/links';
export {
  CHAIN_CAP,
  bakeMix,
  centroidOf,
  flattenAxis,
  packChainIds,
  packPivotUniforms,
  spreadPoint,
} from './field/morph';
export {
  createGpuTransform,
  createPivotStore,
  createPositionStore,
  PIVOT_CAPACITY,
  POS_TEX_WIDTH,
} from './field/stores';
export type { GpuTransform, PivotStore, PositionStore } from './field/stores';
export { createTweens, cubicInOut, linear, smoothstep as smoothstepEase } from './anim/tween';
export type { Ease, TweenOptions, Tweens } from './anim/tween';
export { createFrameSet, ANGLE_EPS, RIDE_DELTA_SIZE } from './anim/frames';
export type { FrameSet, NumArray } from './anim/frames';
export { fnv1a } from './core/fnv';
export {
  clampGlyphPx,
  fieldExtent,
  glyphPx,
  linkDrawCount,
  linkDrawVertices,
  rankPercentiles,
  rankPercentilesTied,
  sizesFromColumn,
  uniformSizes,
} from './field/channels';
export { localDensity, DENSITY_TARGET_PER_CELL } from './field/density';
export type { DensityOpts } from './field/density';
export { spatialFacetColors, principalAxis3 } from './field/spatial';
export { bootQuality, RENDER_SCALE_FLOOR } from './field/quality';
export type { BootQuality } from './field/quality';
export { facetPalette, hslToRgb, paintFacetColors, GOLDEN } from './field/palette';
export {
  createFocus,
  BOOST_HOVER,
  BOOST_NEAR,
  BOOST_SELECT,
  DIM_OTHERS,
  FOCUS_REST,
  LINK_DIM_ALPHA,
  LINK_FOCUS_ALPHA,
  LINK_HOVER_ALPHA,
} from './field/focus';
export type { Focus, FocusOpts } from './field/focus';
export * from './field/constants';

export { createFlycam, FLYCAM_DEFAULTS } from './cameras/flycam';
export type { Flycam, FlycamConfig, FlycamOptions } from './cameras/flycam';
export { createOrbitCam, ORBIT_DEFAULTS } from './cameras/orbit';
export type { OrbitCam, OrbitConfig, OrbitOptions } from './cameras/orbit';
export { DRAG_SLOP, HOME_MARGIN } from './cameras/types';
export type {
  CameraController,
  CameraTelemetry,
  ControlSurface,
  Extent,
  KeyEventLike,
  PointerEventLike,
  RigCamera,
  WheelEventLike,
} from './cameras/types';

export {
  buildGrid,
  cellMaxOf,
  nearestDistance,
  pickRay,
  pxToWorld,
  PICK_DEFAULTS,
  TARGET_PER_CELL,
} from './pick/grid';
export type { Grid, NearestOptions, PickOptions, Vec3 } from './pick/grid';

/* ---- stage 5: placement (§3.3) and the hierarchy compiler (§3.4) --------------------
 * The knob libraries register themselves on import; re-exporting their names here is
 * what makes `{ shape: 'ring', ordering: 'byFacet', density: 'gaussian' }` resolve the
 * moment the package is imported. `scatters` exports no names, so it is imported for its
 * registration side effect alone. */
import './place/scatters';

export { placeGroup } from './place/engine';
export type {
  Density,
  Ordering,
  ParamDesc,
  Params,
  PlaceInput,
  PlacementSpec,
  Scatter,
  Shape,
} from './place/types';
export {
  densityKeys,
  densityParams,
  getDensity,
  getOrdering,
  getScatter,
  getShape,
  orderingKeys,
  registerDensity,
  registerOrdering,
  registerScatter,
  registerShape,
  scatterKeys,
  scatterParams,
  shapeKeys,
  shapeParams,
} from './place/registry';
export { getPreset, preset, presetKeys, registerPreset } from './place/presets';
export { dendriteTiers } from './place/shapes';
export { byAdjacency, byColumn, byFacet, byHash } from './place/orderings';
export { beta, edges, exponential, gaussian, mixture, powerLaw, triangular } from './place/densities';
export { probitApprox } from './place/stat';

export { compileHierarchy, MAX_LEVELS, RANDOM_SHAPES } from './place/hierarchy';
export type {
  CompiledHierarchy,
  HierarchyOpts,
  HierarchyStats,
  LevelSpec,
  PivotTable,
  RandomPlace,
  SpinSpec,
} from './place/hierarchy';
export { attachToAnchors, HOP_MAX } from './place/pull';
export type { AnchorAttachment } from './place/pull';

/* ---- stage 7: the zoom ladder (§6) — the pure machine, its constants, the content
 * contract (ContentProvider + makeSynthProvider) and the DOM overlay. One barrel
 * (src/ladder/index.ts, the §8 `/ladder` subpath), re-exported whole: the ladder names are
 * namespaced by convention (NAMED_*, CARD_*, LOCK_*, OVERLAY_*, ss-* classes) and collide
 * with nothing above. */
export * from './ladder/index';

/* ---- stage 8: config schema + panel + palettes + presets/looks (§7) ----------------------
 * The panel barrel (src/panel/index.ts, the §8 `/panel` subpath) carries the panel AND the
 * config engine it renders (schema rows, DEFAULTS, the persisting store) — re-exported whole.
 * One renaming happens there: the schema's palette OPTION LIST ships as PALETTE_OPTIONS, so
 * the theme module's PALETTES (the categorical colour DATA) keeps the canonical name below. */
export * from './panel/index';

export {
  DEFAULT_RAMP,
  PALETTES,
  PALETTE_NAMES,
  RAMPS,
  RAMP_NAMES,
  UNIFORM_RGB,
  VAULT_LAYER_NAMES,
  facetColors,
  namedPaletteBuffer,
  paletteBuffer,
  rampAt,
  rampColors,
  rampColorsTied,
} from './theme/palettes';
export type { Palette, PaletteName, RGB, RampLike, RampName, RampStops } from './theme/palettes';
export { VISUAL_PRESETS, VISUAL_PRESET_NAMES } from './theme/presets';
export type { VisualPreset, VisualPresetName } from './theme/presets';
export { createTonemap, tonemapRGB, EXPOSURE_DEFAULT, OP_INDEX, TONEMAP_OPS } from './field/tonemap';
export type { TonemapOp, TonemapRig } from './field/tonemap';

/* ---- the near-mesh real-3D LOD (DESIGN.md §5 tier-3 custom layer; §12 "mote → mesh"): an
 * OPT-IN subsystem that draws genuine shaded, optionally textured 3D meshes for the few nodes
 * large on screen (projected-size promotion), fading their billboards — the CLOSE look to the
 * billboards' FAR look. Zero cost unless the consumer calls createNearMesh. */
export {
  createNearMesh,
  generateNearTextures,
  nearTextureAverages,
  pickImageTier,
  wrapNearTextures,
  IMAGE_CACHE_BUDGET,
  IMAGE_MESH_CAP,
  NEAR_SHAPE_NAMES,
  NEAR_TEXTURE_NAMES,
  PROCEDURAL_LAYERS,
  TEX_W,
  TEX_H,
} from './lod/nearmesh';
export type {
  ImageEvent,
  ImageTier,
  NearMesh,
  NearMeshOpts,
  NearShapeName,
  NearTextureName,
  NodeSource,
} from './lod/nearmesh';

/* ---- TRUE displaced planetary terrain (near-mesh terrain stage B): the DEEPEST close look — when
 * you descend onto a solid planet (prototype: Earth) the near-mesh swaps its textured sphere for a
 * higher-detail icosphere displaced by the planet's own height field (real mountains + silhouettes you
 * fly over and land on), resolving in seamlessly. Driven BY `createNearMesh` ({ terrain: true }); these
 * are the building blocks + the pluggable DEM-ready height source. */
export {
  createTerrain,
  earthTerrainSource,
  TERRAIN_ALT_ON,
  TERRAIN_ALT_FULL,
  TERRAIN_HEIGHT_SCALE,
} from './lod/terrain';
export type {
  Terrain,
  TerrainColorNode,
  TerrainHeightSource,
  TerrainOpts,
  TerrainUpdate,
} from './lod/terrain';

/* ---- the HDR bloom post-process (DESIGN.md §5 "tier-3 post-sum pass reusing the tonemap
 * curves"): a helper the CONSUMER installs — scene → additive HDR buffer → bloom → one post-sum
 * tonemap → screen — replacing renderer.render in the loop with post.render(). The library owns
 * no renderer or render target (§5), so this is deliberately a helper, not a field method. */
export { createPostProcess } from './post/bloom';
export type { BloomParams, PostProcess, PostProcessOpts } from './post/bloom';

/* ---- the synth corpus generator (src/synth/) --------------------------------------------
 * Deterministic 10²…10⁶-node corpora from (n, seed) alone — the demo's and the tests' fuel,
 * exported so an external consumer can exercise its own integration at sizes no real dataset
 * reaches (pairs with the ladder's makeSynthProvider above for page content). */
export { generateSynth, synthDefaults } from './synth/generate';
export type { SynthSpec } from './synth/generate';
