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
export * from './constants';
