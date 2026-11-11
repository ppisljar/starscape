/* The playground's data source: a fully SYNTHETIC corpus, generated client-side from (n, seed)
 * alone — no vault, no SecondBrain, no fetch of any external file. It is the library's own
 * deterministic corpus generator (`generateSynth` — the demo's and the tests' fuel, 10²…10⁶
 * nodes from a seed) shaped into the small context object the playground's UI reads: the compiled
 * graph plus which facets play the "region" and "group" roles, the measured galaxy regions, and
 * the layer colours by name.
 *
 * Everything is a pure function of (n, seed): two loads of the same (n, seed) are byte-identical
 * (house rule §10.1 — no Math.random / Date.now in anything producing geometry). The seed is a URL
 * knob (?seed=…, default 42) so a reported number is reproducible on any machine.
 *
 * The facet mapping follows the library demo:
 *   regions ← the `galaxy` facet (centre/radius/count measured from the generated positions)
 *   groups  ← the `type` facet (bounded pivot count at every n; `system` carries ~n^⅔ values and
 *             would overflow the field's pivot store at 10⁶)
 *   type    ← the synth `type` facet, whose values mirror the library's vaultLayer palette names,
 *             so the 'vault' palette option colours them by name out of the box.
 */

import { buildGraph, generateSynth, PALETTES, VAULT_LAYER_NAMES } from '../src/index';
import type { Graph } from '../src/index';

/** A measured cluster of the `galaxy` facet — centre, radius and member count, the synthetic
 *  equivalent of the vault's pre-baked regions. */
export interface Region {
  id: number;
  name: string;
  count: number;
  centre: [number, number, number];
  radius: number;
  layer: number;
}

/** The small context the playground UI reads around the compiled graph — which facets play the
 *  region/group roles, the measured galaxy regions, the layer colours by name. No page content,
 *  no ids: the ladder's open tier uses the library's deterministic `makeSynthProvider`. */
export interface SynthData {
  graph: Graph;
  n: number;
  /** The facet the "regions" concept binds to (the synth corpus: 'galaxy'). */
  regionFacet: string;
  /** The facet hierarchies group by inside a region (the synth corpus: 'type'). */
  groupFacet: string;
  summaries: string[];
  layerNames: string[];
  /** name → RGB (the library's vaultLayer stops) for namedPaletteBuffer. */
  layerColorsByName: Record<string, [number, number, number]>;
  regions: Region[];
  /** 0 = "derive live": main.ts falls back to radius/∛n once the field's extent exists. */
  spacing: number;
  byId: Map<string, number>;
}

/** Generate the synthetic corpus and measure its galaxy regions. Synchronous; generateSynth(10⁶)
 *  takes a few seconds, so callers paint a progress message and yield a frame first. */
export function loadSynth(n: number, seed: number, onStep: (msg: string) => void = () => {}): SynthData {
  onStep(`generating ${n.toLocaleString('en-US')} nodes…`);
  const input = generateSynth({ n, seed });

  onStep('building the graph');
  // 'system' = the demo's link-importance rule: intra-system links before the chords, so the
  // link budget drops the long jumps first.
  const graph = buildGraph(input, { orderByFacet: 'system' });

  /* ---- regions from the galaxy facet: centroid, max-distance radius, count, dominant type.
     Three O(n) passes over typed arrays. */
  onStep('measuring regions');
  const positions = graph.positions!;
  const galaxy = graph.facets['galaxy']!;
  const type = graph.facets['type']!;
  const g = galaxy.values.length;
  const t = type.values.length;
  const count = new Uint32Array(g);
  const sum = new Float64Array(g * 3);
  const typeCount = new Uint32Array(g * t);
  for (let i = 0; i < n; i++) {
    const k = galaxy.data[i]!;
    count[k] = count[k]! + 1;
    sum[k * 3] = sum[k * 3]! + positions[i * 3]!;
    sum[k * 3 + 1] = sum[k * 3 + 1]! + positions[i * 3 + 1]!;
    sum[k * 3 + 2] = sum[k * 3 + 2]! + positions[i * 3 + 2]!;
    typeCount[k * t + type.data[i]!] = typeCount[k * t + type.data[i]!]! + 1;
  }
  const centre = new Float64Array(g * 3);
  for (let k = 0; k < g; k++) {
    const c = count[k]! || 1;
    centre[k * 3] = sum[k * 3]! / c;
    centre[k * 3 + 1] = sum[k * 3 + 1]! / c;
    centre[k * 3 + 2] = sum[k * 3 + 2]! / c;
  }
  const r2 = new Float64Array(g);
  for (let i = 0; i < n; i++) {
    const k = galaxy.data[i]!;
    const dx = positions[i * 3]! - centre[k * 3]!;
    const dy = positions[i * 3 + 1]! - centre[k * 3 + 1]!;
    const dz = positions[i * 3 + 2]! - centre[k * 3 + 2]!;
    const d = dx * dx + dy * dy + dz * dz;
    if (d > r2[k]!) r2[k] = d;
  }
  const regions: Region[] = [];
  for (let k = 0; k < g; k++) {
    let lead = 0;
    for (let j = 1; j < t; j++) if (typeCount[k * t + j]! > typeCount[k * t + lead]!) lead = j;
    regions.push({
      id: k,
      name: galaxy.values[k]!,
      count: count[k]!,
      centre: [centre[k * 3]!, centre[k * 3 + 1]!, centre[k * 3 + 2]!],
      radius: Math.sqrt(r2[k]!),
      layer: lead,
    });
  }

  /* ---- layer colours by NAME from the library's own vaultLayer palette (the synth type values
     mirror the layer names; any name without a stop takes the grey fallback). */
  const layerColorsByName: Record<string, [number, number, number]> = {};
  const pal = PALETTES.vaultLayer;
  for (let j = 0; j < VAULT_LAYER_NAMES.length && j < pal.length; j++) {
    const c = pal[j]!;
    layerColorsByName[VAULT_LAYER_NAMES[j]!] = [c[0]!, c[1]!, c[2]!];
  }

  return {
    graph,
    n,
    regionFacet: 'galaxy',
    groupFacet: 'type',
    summaries: [], // main.ts reads graph.labelOf; no eager 10⁶ strings (§10.2)
    layerNames: type.values,
    layerColorsByName,
    regions,
    spacing: 0, // derive live: radius/∛n once the field's extent exists
    byId: new Map(),
  };
}
