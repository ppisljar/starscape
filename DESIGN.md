# starscape — design

A graph visualization library for very large graphs (10⁶+ nodes) rendered as a 3D field you can
fly through like a game — down to reading individual pages in place — while staying a small,
data-agnostic, heavily customizable library.

This document is the contract. Code that contradicts it is wrong, or the document is — either
way, fix the disagreement before building on it.

## 1. Vision and goals

- **Scale**: a million nodes and their links, *flyable* (not merely rendered) on an ordinary
  laptop GPU. The predecessor measured ~20 fps at 10⁶ with known CPU bottlenecks still in place;
  removing them (see §5) is what this design is shaped around.
- **Game-like**: 6-DOF flight, nested rotating reference frames you can ride, levels of detail
  that resolve from mote → label → card → full content as you approach.
- **Hierarchy without limits**: group nodes on arbitrarily many levels; every level picks its own
  arrangement, distribution, spacing and rotation. The predecessor hardcoded two levels
  (galaxy/system); starscape takes a list.
- **Customizable at three depths**: themes (pure data), shader hooks (inject logic), and custom
  layers (bring your own objects, read our buffers).
- **Data-agnostic**: the core consumes typed arrays and callbacks. It never fetches, never parses
  a file format, never knows what a "wiki page" is. Format adapters live in the consuming app.
- **Deterministic**: every placement is a pure function of (data, spec). No RNG, no clock. The
  same inputs produce byte-identical buffers, forever, so screenshots are reproducible and tests
  can be golden.

Non-goals, for now: computing graph layouts in the browser (frozen positions are a first-class
*input*; a layout plugin interface can come later), 2D-canvas/SVG fallback renderers, React
bindings.

## 2. Lineage

starscape is a design-first rebuild of `SecondBrain/tools/vault-site-3d`, which proved the
architecture: two draw calls at every scale, typed arrays + CSR (no JS object per node),
importance-ordered link buffer with a draw-range budget, frozen deterministic layouts, a
distance-driven zoom ladder, nested rotating frames. Its `README.md` and `SCALE.md` are required
reading; its hard-won rules are restated in §10 so they survive the rewrite.

What gets ported nearly as-is (translated to TypeScript, logic preserved): CSR construction and
counting-sort link ordering, the glyph/halo shader *logic*, the uniform-grid picker, the flycam,
the tonemap curves, the zoom-ladder thresholds and hysteresis.

What gets redesigned: the grouping/arrangement system (§4), the transform pipeline (GPU morphing
replaces CPU `recompose()`, §5), the config system (schema engine stays, the ad-hoc schema
contents do not), and the materials (TSL instead of raw GLSL, §6).

## 3. Core model

### 3.1 Graph: columns, never objects

```ts
interface Graph {
  n: number;
  positions?: Float32Array;            // 3n, optional frozen layout
  links: Uint32Array;                  // 2m node-index pairs
  columns: Record<string, NumericColumn>;      // degree, words, rank, …
  facets: Record<string, Facet>;       // categorical: { values: string[], data: Uint8/16/32Array }
  labels?: LabelSource;                // i => string, lazy; never 10⁶ eager strings
}
```

`buildGraph()` validates, dedups directed links, computes degree if absent, builds CSR
(offsets + neighbours + edge indices) and the importance-ordered link buffer. ~8 bytes per link,
not the ~130 of object-shaped graph libraries. Any reduction over a node-length array is a loop
— spreading a typed array into a call blows the stack at 10⁶ (measured in the predecessor).

### 3.2 The spec: a scene grammar

The library is driven by one declarative spec; imperative methods exist but are sugar that
patches the spec. Compiling spec + graph yields GPU buffers; changing the spec diffs against the
previous compilation and animates what moved.

```ts
const scene = createStarscape(canvas, graph, {
  hierarchy: [ /* §4 — any number of levels */ ],
  projection: { mode: '3d' },          // or '2d', or a transition
  theme: { palette, ramps, background, blend, tonemap, sizeBy, colorBy },
  ladder: { contentProvider, budgets }, // optional module
  camera: { type: 'hybrid', home: 'auto' },
});
```

### 3.3 Placement: four orthogonal knobs

The predecessor's "arrangement" conflated several decisions. starscape splits placement into
four independently pluggable pieces. Placing the members of one group means:

```
param = density.warp( ordering.assign(member) )    // where along the shape
pos   = shape.at(param) + scatter.offset(member)   // the shape, plus flesh on it
```

1. **Shape** — a parametric manifold: a map from parameter space U to 3D positions, with a
   natural orientation frame and a size policy. Dimension of U is the shape's own business:
   - curves (1D): `ring`, `line`, `helix`, `spiralArm`, `dendrite` (tree-structured parameter)
   - surfaces (2D): `shell` (sphere), `disc`, `torus`
   - volumes (3D): `ball`, `box`, `cloud`
2. **Ordering** — which member gets which parameter coordinate: `byIndex`, `byFacet(key)`
   (contiguous sectors), `byColumn(key)` (rank gradient, e.g. degree around a ring),
   `byAdjacency` (linked nodes adjacent — seriation), `byHash` (deterministic shuffle).
3. **Density** — a distribution over parameter space, applied as an inverse-CDF warp of the
   ordered quantiles: `uniform`, `gaussian(center, sigma)`, `exponential(rate)`,
   `powerLaw(alpha)`, `mixture([...])`, or a custom CDF. Ring + gaussian = most members bunched
   at one point of the ring, thinning toward the antipode — exactly the intuitive reading.
4. **Scatter** — displacement off the manifold, from a deterministic `hash(index, salt)`:
   `none`, `gaussian(thickness)` (a ring becomes a tube, a shell becomes a fuzzy shell),
   `discFalloff(h)` (galaxy-disc thickness that thins with radius), `noiseField(scale, amp)`.

Named looks like "spiral galaxy", "barred galaxy", "elliptical", "globular cluster" are
**placement presets** — bundles of the four knobs (multi-arm spiral curves + exponential radial
density + disc scatter + swirl) — registered in the same registry users extend. Primitives for
composing, presets for reaching.

### 3.4 Hierarchy: recursive levels

```ts
hierarchy: [
  { groupBy: 'type',      place: { shape: 'shell' },                 spin: { rate: 0.1 } },
  { groupBy: 'region',    place: preset('spiralGalaxy'),             spin: { rate: 0.4 } },
  { groupBy: 'community', place: { shape: 'ring', density: gaussian(0, 0.3) } },
  { place: { shape: 'cloud', scatter: gaussian(1) } },   // leaf level: the nodes
]
```

Level k partitions each group from level k−1 by its `groupBy` facet (or a function of node
index) and places the *child group centres* on that level's shape; the final level places
individual nodes. Group radius defaults to `spacing × count^⅓` with a floor (sublinear so one
giant group cannot swallow the field; floored so a four-member group is still a place you can
fly to). Depth is unbounded by design; the shader supports a fixed maximum frame-chain depth
(start at 4, a constant to raise), beyond which deeper levels simply stop spinning
independently — placement itself has no depth limit.

An optional `pull` setting reattaches non-grouped nodes to their nearest forced anchor by
multi-source BFS over links (the predecessor's "pull the rest in"), so forcing a subset onto a
shape blooms the rest around it.

### 3.5 Reference frames

Each level's `spin` defines a rotating frame about its group pivots. Frames nest; a camera
"riding" the frame it is inside applies that frame's *per-tick* rotation to itself —
incremental, never by cancelling accumulated angle, so crossing a frame boundary is seamless.
Rotation work is one quaternion per **pivot** per frame (hundreds), never per node; nodes follow
in the vertex shader (§5). "Is anything turning" tests the per-tick delta, not the accumulated
angle, so an idle scene costs nothing.

### 3.6 Projection and 2D

`projection` is a pipeline stage, not a mode switch: `3d` (perspective), `2d` (positions
flattened to a plane — either by dropping an axis or by supplying an alternate 2D position set),
and any transition between the two is the standard position morph (§5) plus a camera/FOV tween.
Dimensionality is a destination, not a branch in the code.

## 4. Transform pipeline

```
base positions (input or placement)        [CPU, once per spec change]
  → hierarchy placement compile  → endpoint buffer B, pivot chain per node
  → projection target            → (optional) endpoint buffer B′
  → GPU per frame: p = chain of frame quaternions ( mix(A, B, t) ) × spread
```

Three rules carried over, hardened:

- **Stages compose; nothing mutates in place.** Every stage reads only its own input. Dragging
  spread back to 1.0 restores byte-exact coordinates.
- **Per-frame CPU is O(pivots), never O(nodes).** The predecessor's one design flaw was
  `recompose()` rewriting node and link buffers on the CPU every frame while anything span —
  the dominant per-frame cost at 10⁵⁺. In starscape, per-node work lives in the vertex shader:
  position endpoints are attributes, `t` is a uniform, frame quaternions are a uniform
  array/texture indexed by a per-node pivot-chain attribute. Links read node positions the same
  way (vertex pulling from a position texture/storage buffer), so no link rewrite exists at all.
- **Animation = uniforms.** One small tween system (easing, duration, interrupt-by-retarget)
  drives morph `t`, spread, dim, camera flights. It never touches a vertex buffer.

## 5. Rendering

- **Backend: three.js `WebGPURenderer` with materials in TSL** (Three Shading Language). TSL
  compiles to WGSL on WebGPU and falls back automatically to GLSL/WebGL2, so one material
  codebase covers both; WebGPU also opens compute shaders for future in-browser layout/LOD work.
  three.js is a **peer dependency**.
- **Two primary draw calls**: one instanced quad for all nodes (glyph + halo in the fragment
  stage, flat cheap branch under ~2 px), one batched line list for links. Overlay layers (labels,
  regions, cards) add their own few.
- **LOD / budgets**: link budget = `setDrawRange` over the importance-ordered buffer (intra-group
  before inter, short before long — the dropped links are the ones saying the least and costing
  the most fill); node glyph simplification under 2 px; `renderScale` as the blunt lever; boot
  quality chosen from corpus size (no feedback loop — the predecessor's adaptive governor
  oscillated). Post-v1: chunked frustum culling for the 10⁶ node-fill floor.
- **Custom rendering, three tiers**:
  1. **Theme** (data): palettes, ramps, size/color channel bindings, blend, tonemap, background.
  2. **Shader hooks** (TSL functions): glyph shape, per-node color, halo — user-supplied nodes
     spliced into the material graph.
  3. **Layers** (objects): register a render layer that reads the shared position/attribute
     buffers. This is how near-LOD "planet" meshes, trails, or replay overlays (the ingest
     conveyor / agent spiders of the predecessor) are built without the core knowing about them.
- **One pixel unit**: shader, picker and ladder all reason in CSS pixels; constants exported
  from one module. **Blending is a property of the ground**: additive on dark, normal on light,
  both layers follow one setting.

## 6. Interaction

- **Cameras**: orbit, 6-DOF fly (speed scaled by free room ahead; proximity *damps* toward a
  floor, never gates to zero), and the hybrid drag-turns-the-universe scheme. Pluggable.
- **Picking**: uniform spatial grid over current positions; cone pick, coverage before depth.
  Rebuilt on morph completion, not per frame.
- **Zoom ladder** (optional module): mote → named → card → lock → open, a pure function of
  camera distance via one unclamped projected-size number. Glyph tier keys off the node's own
  apparent size; card tier keys off a fixed world radius. Label admission by rank band with
  dwell hysteresis. Attention follows the cursor. A pinned (selected) page outranks every
  budget. Content arrives through callbacks — `getCard(i)`, `getBody(i)` — the library renders
  frames and layout, the app supplies meaning.
- **Selection/hover** are independent states; hover lights the neighbourhood via CSR without
  repainting the field.

## 7. Theming and config

- Palettes/ramps ship as data (including colorblind-safe sets); facet → categorical palette,
  numeric column → ramp. Many palettes, trivially extensible.
- The **schema-driven config engine** survives: every tunable is a schema row (group, key,
  label, range, default); the optional `panel` module renders it; presets and user "looks" are
  **diffs from defaults**. Distances in the schema are expressed in page-spacings, not world
  units, so numbers keep meaning across corpora. The *contents* of the schema are designed
  fresh per module — nothing is copied from the predecessor wholesale.

## 8. Packaging

- Single npm package, subpath exports: `starscape` (core), `/cameras`, `/pick`, `/ladder`,
  `/panel`, `/layers`, `/placements`. Tree-shakeable ESM, `.d.ts` from `tsc`. Plain `starscape`
  is taken on npm — final published name (scoped `@…/starscape` or a variant) is decided at
  publish time; nothing in the code depends on it.
- TypeScript strict. Vite for the demo apps, Vitest for tests, ESLint. `demo/` consumes the
  library exactly as an external user would.
- A TS port of the synthetic corpus generator lives in `test/` — scale work needs 10⁴–10⁶
  corpora on tap, with multiple independent facets (one facet cannot tell a working channel
  from a coincidence).

## 9. Performance budgets (acceptance, not aspiration)

Measured the predecessor's way: median of 90 `requestAnimationFrame` intervals at the home
framing, viewpoint asserted (never trusted — camera position is part of the measurement). Never
time `renderer.render()` directly; `gl.finish()` does not block in Chrome.

| corpus | boot | frame CPU | fps target |
|---:|---:|---:|---|
| 10⁴ | < 0.5 s | ~0 | vsync |
| 10⁵ | < 1 s | < 0.5 ms | vsync with budgets at defaults |
| 10⁶ | < 3 s | < 1 ms | **flyable (≥ 30)** with budgets engaged, all spins on |

The 10⁶ row is the one the predecessor fails today (fill-bound at ~20 fps, and CPU-bound the
moment anything rotates); §5's GPU-side transform chain and the link/node budgets are the
design response, and this table is how we know they worked.

## 10. House rules (inherited, non-negotiable)

1. No `Math.random()`, no `Date.now()` in anything that produces geometry. `hash(i, salt)`.
2. No per-node JS objects, no per-node closures, no per-node strings held eagerly.
3. Reductions over node-length arrays are loops.
4. Every budget that drops content is reported (`linkStats`-style) — silent truncation lies.
5. Boot paths have rejection handlers that surface the error into the UI.
6. Dev servers and loaders use `no-store` — a stale .bin is indistinguishable from a bug.
7. `window.__starscape` debug handle in demos; telemetry shows hidden states (damping, budgets).
8. Fit the camera to the *live* extent, never a stored one.
9. Golden tests for placement: fixed spec + fixed corpus → byte-identical buffers.

## 11. Roadmap — one commit per stage

| stage | deliverable | proves |
|---|---|---|
| 0 | this document | the design |
| 1 | toolchain: tsconfig strict, vite, vitest, eslint, CI check script | repo hygiene |
| 2 | core data: `buildGraph`, CSR, link ordering, synth generator (TS) + tests | data model |
| 3 | static field: TSL mote + link materials, two draw calls, theme basics; demo on synth 10⁴ | renderer + TSL bet |
| 4 | cameras + picking + hover/select | interaction base |
| 5 | placement engine: shapes/ordering/density/scatter + hierarchy compiler, CPU reference + golden tests | §3.3–3.4 |
| 6 | GPU morph + frame chains + tween system + 2D/3D projection | §4, the big unlock |
| 7 | zoom ladder + labels + contentProvider | reading in place |
| 8 | config schema + panel + palettes + presets/looks | customization |
| 9 | perf pass at 10⁵/10⁶ against §9 table; budgets, LOD; SCALE.md-style writeup | the scale claim |
| 10 | SecondBrain integration: vault adapters + wiki content provider, `vault-site-3d` becomes a consumer | the real page |

Stage 3 is deliberately early and small: it is the TSL/WebGPU risk retired before anything
depends on it. If TSL proves unworkable for the glyph shader, the fallback is raw
GLSL/WebGL2 with the material behind an interface — decided there, not later.

## 12. Open questions

- Label rendering at scale: DOM labels cap out in the hundreds; an SDF glyph atlas is the
  likely stage-7 answer, sized by what stage 9 measures.
- Pivot count ceiling: quaternion uniform array vs. data texture — decided by the WebGPU
  storage-buffer limits actually available (stage 6).
- In-browser layout as a compute-shader plugin (post-v1): the frozen-layout philosophy stays
  the default either way.
- VR/XR: nothing in the design precludes it (three.js carries it); explicitly out of scope
  until after stage 10.
