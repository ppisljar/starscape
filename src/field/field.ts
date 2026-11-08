/* createField — the static field renderer (DESIGN.md §5, roadmap stage 3).
 *
 * Two primary draw calls whatever the corpus size: one instanced quad for all nodes, one
 * batched line list for all links. The returned objects attach to any three.js scene; the
 * caller owns renderer, scene and camera (the library never fetches, never owns the page).
 *
 * No per-node JS objects anywhere (§10.2): every channel is a typed array written in place.
 */

import { AdditiveBlending, NormalBlending, type Object3D } from 'three/webgpu';
import type { Graph } from '../core/types';
import {
  fieldExtent,
  linkDrawCount,
  sizesFromColumn,
  uniformSizes,
} from './channels';
import { facetPalette, paintFacetColors } from './palette';
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
import { buildLinkLayer } from './links';

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
}

export interface FieldStats {
  nodes: number;
  /** The budget report (§10.4): silent truncation lies, so what is drawn vs what exists is
   *  always available. */
  links: { drawn: number; total: number };
  sizeColumn: string;
  colorFacet: string | null;
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
  /** Re-bind node size to a numeric column ('uniform' for equal dots). */
  setSizeColumn(key: string): void;
  /** One blending setting for both layers (see FieldOpts.blend). */
  setBlend(mode: 'additive' | 'normal'): void;
  /** Report the viewport's CSS-pixel height after a resize (one pixel unit, everywhere). */
  resize(heightCss: number): void;
  /** Report a camera fov change (degrees). */
  setFov(fov: number): void;
  stats(): FieldStats;
  /** The live extent the field was built from — fit the camera to this, never to a stored
   *  number (§10.8). */
  extent: { centre: [number, number, number]; radius: number; diagonal: number };
  /** The background colour the default theme assumes (0x05070d on 'additive'). */
  background: number;
  dispose(): void;
}

/** Build the two-draw-call field for a compiled Graph. The graph must carry a frozen layout —
 *  the placement engine that could synthesise one arrives at stage 5. */
export function createField(graph: Graph, opts: FieldOpts = {}): Field {
  const n = graph.n;
  const positions = graph.positions;
  if (!positions) {
    throw new Error(
      'createField: graph has no positions — a frozen layout is required until the placement engine (stage 5) arrives',
    );
  }

  const fov = opts.fov ?? DEFAULT_FOV;
  const heightCss = opts.heightCss ?? 800;
  const extent = fieldExtent(positions, n);
  const diagonal = extent.diagonal || 1;

  /* ---- channels ---------------------------------------------------------------------- */
  let sizeColumn = opts.sizeColumn ?? (graph.columns['degree'] ? 'degree' : 'uniform');
  const firstFacet = Object.keys(graph.facets)[0] ?? null;
  let colorFacet = opts.colorFacet ?? firstFacet;
  if (colorFacet !== null && !graph.facets[colorFacet]) {
    throw new Error(`createField: colorFacet "${colorFacet}" names no facet`);
  }

  const colors = new Float32Array(n * 3);
  const shapes = new Float32Array(n); // all discs at stage 3; shapeBy binding arrives later
  const sizes = computeSizes(graph, sizeColumn, n);
  paintColors(graph, colorFacet, n, colors);

  /* ---- layers ------------------------------------------------------------------------ */
  const node = buildNodeLayer(n, positions, colors, sizes, shapes, {
    fov,
    heightCss,
    dRef: diagonal * D_REF_FRACTION,
    fogNear: diagonal * FOG_NEAR,
    fogFar: diagonal * FOG_FAR,
  });
  const link = buildLinkLayer(graph.links, positions, {
    fogNear: diagonal * FOG_NEAR * 0.8,
    fogFar: diagonal * FOG_FAR * 0.9,
    lenSoft: diagonal * LINK_LEN_SOFT,
  });

  const totalLinks = graph.links.length >> 1;
  let drawn = linkDrawCount(totalLinks, opts.linkBudget ?? totalLinks);
  link.setDrawCount(drawn);

  /* Link colours: an edge takes the colour of the nodes it joins, so recolouring the field
     recolours its links for free and the two can never disagree. All m links are painted
     (the buffer is static; painting only the draw range would hand a later budget-grow
     stale zeroes — the exact bug the predecessor had to re-derive its prefix to avoid). */
  const writeLinkColors = (): void => {
    const lc = link.colors;
    const ls = graph.links;
    for (let e = 0; e < totalLinks; e++) {
      const a = ls[e * 2]! * 3;
      const b = ls[e * 2 + 1]! * 3;
      const o = e * 8;
      lc[o] = colors[a]!;
      lc[o + 1] = colors[a + 1]!;
      lc[o + 2] = colors[a + 2]!;
      lc[o + 3] = LINK_BASE_ALPHA;
      lc[o + 4] = colors[b]!;
      lc[o + 5] = colors[b + 1]!;
      lc[o + 6] = colors[b + 2]!;
      lc[o + 7] = LINK_BASE_ALPHA;
    }
    link.colorAttr.needsUpdate = true;
  };
  writeLinkColors();

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
      paintColors(graph, colorFacet, n, colors);
      node.colorAttr.needsUpdate = true;
      writeLinkColors();
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

    resize(h: number): void {
      // CSS pixels, deliberately NOT drawing-buffer pixels — the ladder lays out DOM in CSS
      // px, and clamping in device px would put the two a devicePixelRatio apart.
      node.uniforms.halfH.value = h / 2;
    },

    setFov(f: number): void {
      node.uniforms.tanHalf.value = Math.tan((f * Math.PI) / 360);
    },

    stats(): FieldStats {
      return {
        nodes: n,
        links: { drawn, total: totalLinks },
        sizeColumn,
        colorFacet,
      };
    },

    dispose(): void {
      node.dispose();
      link.dispose();
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
