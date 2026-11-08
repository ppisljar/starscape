/** starscape — graph visualization for very large graphs. Further modules arrive in later
 *  stages; exported here: the stage-2 core data model (DESIGN.md §3.1), the stage-3 field
 *  renderer (§5 — two draw calls, TSL materials), and the stage-4 interaction base (§6 —
 *  cameras, picking, hover/selection focus). The synth corpus generator (`src/synth/`) is
 *  test/demo fuel, deliberately not part of this surface. */
export const VERSION = '0.0.0';

export { hash, gauss } from './core/hash';
export { buildCSR } from './core/csr';
export type { CSR } from './core/csr';
export { orderLinks } from './core/linkOrder';
export { buildGraph, neighbours, incident } from './core/graph';
export type { Facet, Graph, GraphInput, NumericColumn } from './core/types';

export { createField } from './field/field';
export type { Field, FieldOpts, FieldStats } from './field/field';
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
