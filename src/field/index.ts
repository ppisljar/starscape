/* The field module (roadmap stage 3 + the stage-6 §4 transform pipeline): the two-draw-call
 * renderer, its pure parts, and the GPU morph/frame-chain/spread machinery. */
export { createField, MORPH_DURATION } from './field';
export type { Field, FieldOpts, FieldStats } from './field';
export {
  CHAIN_CAP,
  bakeMix,
  centroidOf,
  flattenAxis,
  packChainIds,
  packPivotUniforms,
  spreadPoint,
} from './morph';
export {
  createGpuTransform,
  createPivotStore,
  createPositionStore,
  PIVOT_CAPACITY,
  POS_TEX_WIDTH,
} from './stores';
export type { GpuTransform, PivotStore, PositionStore } from './stores';
export {
  clampGlyphPx,
  fieldExtent,
  glyphPx,
  linkDrawCount,
  linkDrawVertices,
  rankPercentiles,
  sizesFromColumn,
  uniformSizes,
} from './channels';
export { bootQuality, RENDER_SCALE_FLOOR } from './quality';
export type { BootQuality } from './quality';
export { facetPalette, hslToRgb, paintFacetColors, GOLDEN } from './palette';
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
} from './focus';
export type { Focus, FocusOpts } from './focus';
export * from './constants';
