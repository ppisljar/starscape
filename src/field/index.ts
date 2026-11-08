/* The field module (roadmap stage 3): the two-draw-call static renderer and its pure parts. */
export { createField } from './field';
export type { Field, FieldOpts, FieldStats } from './field';
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
