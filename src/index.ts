/** starscape — graph visualization for very large graphs. Further modules arrive in later
 *  stages; exported here: the stage-2 core data model (DESIGN.md §3.1), the stage-3 field
 *  renderer (§5 — two draw calls, TSL materials), the stage-4 interaction base (§6 —
 *  cameras, picking, hover/selection focus), and the stage-5 placement engine (§3.3–3.4 —
 *  the four knobs, their libraries, presets, the hierarchy compiler and the BFS pull).
 *  The synth corpus generator (`src/synth/`) is test/demo fuel, deliberately not part of
 *  this surface. */
export const VERSION = '0.0.0';

export { hash, gauss } from './core/hash';
export { buildCSR } from './core/csr';
export type { CSR } from './core/csr';
export { orderLinks } from './core/linkOrder';
export { buildGraph, neighbours, incident } from './core/graph';
export type { Facet, Graph, GraphInput, NumericColumn } from './core/types';

export { createField, MORPH_DURATION } from './field/field';
export type { Field, FieldOpts, FieldStats } from './field/field';
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
  sizesFromColumn,
  uniformSizes,
} from './field/channels';
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
export { DRAG_SLOP } from './cameras/types';
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
  Params,
  PlaceInput,
  PlacementSpec,
  Scatter,
  Shape,
} from './place/types';
export {
  densityKeys,
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
  shapeKeys,
} from './place/registry';
export { getPreset, preset, presetKeys, registerPreset } from './place/presets';
export { dendriteTiers } from './place/shapes';
export { byAdjacency, byColumn, byFacet, byHash } from './place/orderings';
export { exponential, gaussian, mixture, powerLaw } from './place/densities';
export { probitApprox } from './place/stat';

export { compileHierarchy, MAX_LEVELS } from './place/hierarchy';
export type {
  CompiledHierarchy,
  HierarchyOpts,
  HierarchyStats,
  LevelSpec,
  PivotTable,
  SpinSpec,
} from './place/hierarchy';
export { attachToAnchors, HOP_MAX } from './place/pull';
export type { AnchorAttachment } from './place/pull';
