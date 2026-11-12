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
- **The Milky Way test**: using the library one can build a game that shows the full Milky Way
  with all its real clusterings — spiral arms, star clusters, solar systems, planetary systems —
  and fly it continuously from the galactic rim down to a single planet, close enough that Earth
  shows oceans and continents. Not v1 scope; it is the stress test every API decision is checked
  against (§12).
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
interface Graph {                      // the COMPILED graph; buildGraph's input takes
  n: number;                           // `positions?` and `labels?`
  positions: Float32Array | null;      // 3n frozen layout, or null when none was given
  links: Uint32Array;                  // 2m node-index pairs
  columns: Record<string, NumericColumn>;      // degree, words, rank, …
  facets: Record<string, Facet>;       // categorical: { values: string[], data: Uint8/16/32Array }
  labelOf: (i: number) => string;      // lazy, called on demand; never 10⁶ eager strings
}
```

`buildGraph()` validates, dedups directed links, computes degree if absent, builds CSR
(offsets + neighbours + edge indices) and the importance-ordered link buffer. ~8 bytes per link,
not the ~130 of object-shaped graph libraries. Any reduction over a node-length array is a loop
— spreading a typed array into a call blows the stack at 10⁶ (measured in the predecessor).

A `Graph` is one eager *source*, good to ~10⁷. Chunked and procedural sources (§12) present the
same columnar shape per chunk, so nothing downstream may assume "all nodes exist".

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
design response, and this table is how we know they worked. *(Stage 9, 2026-11-09: measured
green on all three rows — 10⁶ at 31–39 fps / ≤ 0.3 ms frame CPU with all spins on and the
boot budgets engaged. The numbers, the levers and the methodology warnings live in
`SCALE.md`.)*

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

Post-v1, the Milky Way test (§12) drives the next arc: chunked sources, floating origin, and
the mote → mesh handoff demo.

## 12. The Milky Way test

The goal in §1, unpacked into what it demands of the design — none of it v1 scope, all of it
shaping v1 decisions:

- **Dynamic range.** Galactic disc to coastline is ~16 orders of magnitude, far past Float32's
  ~7 significant digits. The answer is structural, and the hierarchy already is it: a position
  lives in its *frame's* local space (planet in system, system in cluster, cluster in galaxy),
  so no stored coordinate ever spans the full range. Rendering is camera-relative — the
  floating origin is subtracted at *pivot* granularity on the CPU, never per node — with a
  logarithmic depth buffer. The rule for every API: world space is a chain of local spaces; a
  global Float32 coordinate does not exist.
- **Population.** 10¹¹ stars do not fit in one typed array. The eager `Graph` stays the v1
  source (§3.1); the game needs a chunked **source** behind the same columnar shape — a region
  materializes as you approach it, and because placement is deterministic (§3.3), a chunk can
  be *generated* on demand rather than stored: a procedural universe is `hash(i, salt)` all the
  way down. Budgets, picking and the ladder therefore operate on what is *resident*, never on
  "all nodes" — a constraint cheap to honor from stage 2 and expensive to retrofit.
- **Mote → mesh.** The zoom ladder's top tier is content-agnostic: `getBody(i)` may return an
  HTML body *or a custom layer object* (§5 tier 3) — a textured, shaded sphere with its own
  surface LOD is the app's business. The library's job is the handoff: sprite and mesh
  co-located in the same frame chain, cross-faded with no pop, so approaching Earth is one
  continuous act from mote to oceans.
- **Orbits are frames.** Planets around stars, stars around the core: the nested spin frames of
  §3.5, with per-level rates and axes. "Ride it" is how you stand on a turning body and watch
  its moons move.

Acceptance scenario, after stage 10: one uninterrupted flight — galactic rim → a spiral arm →
an open cluster → a solar system → a planet showing coasts on its day side — with telemetry
reporting frame-chain depth and resident-set size the whole way down.

## 13. Open questions

- Label rendering at scale: DOM labels cap out in the hundreds; an SDF glyph atlas is the
  likely stage-7 answer, sized by what stage 9 measures.
- Pivot count ceiling: quaternion uniform array vs. data texture — decided by the WebGPU
  storage-buffer limits actually available (stage 6).
- In-browser layout as a compute-shader plugin (post-v1): the frozen-layout philosophy stays
  the default either way.
- Chunked-source protocol (§12): what a chunk promises (counts, extents, facet dictionaries),
  and when it may be evicted — sketched during stage 2 so the API never assumes full residency,
  decided post-v1.
- VR/XR: nothing in the design precludes it (three.js carries it); explicitly out of scope
  until after stage 10.

## 14. Near-mesh LOD — the close look (the first mote → mesh)

The field draws every node as a billboard: the FAR look, two draw calls, 10⁶ flyable, unchanged.
`createNearMesh(graph, field, opts)` (`src/lod/nearmesh.ts`, the `/layers` kind of §5 tier-3) adds
the CLOSE look — the first concrete realization of §12's "mote → mesh". For the handful of nodes
that have grown LARGE ON SCREEN it draws a genuine shaded, optionally textured 3D object (a planet
you orbit), co-located with and cross-faded against its billboard so approaching a node is one
continuous act from mote to oceans. It is a consumer-installed layer, exactly like the bloom
helper: the core owns no part of it and the library stays self-contained.

Three properties are load-bearing, and each is tested:

- **Opt-in ⇒ zero cost when off.** A consumer that never calls `createNearMesh` instantiates
  nothing — no objects, no buffers, no textures, no per-frame work. The 10⁶ path is byte-for-byte
  today's (measured: 1M at the home framing, LOD off, renders at vsync with the same 3 draw calls
  and the subsystem provably never constructed). The only per-frame addition a consumer who
  *declines* the feature pays is the absence of the call.
- **Promotion is by PROJECTED SIZE, never a fixed nearest-K.** A node becomes a mesh iff its
  projected glyph radius (the same `glyphPx` the billboards and the zoom ladder already use)
  exceeds `promotePx`; `count` is only a performance cap (the largest-on-screen win). So a small
  graph seen whole stays all-billboards — every node is tiny, nothing promotes — and only flying
  close, which makes a node and its neighbours big on screen, promotes them. The pick grid
  (§6) finds LOCAL candidates cheaply (a per-cell max-size prune skips whole cells whose biggest
  node cannot reach the threshold); the px test decides each survivor. Per-frame cost is O(local).
- **Default sphere ⇒ one draw call; textures are procedural and opt-in.** With no shapeFacet every
  promoted node is a sphere, so a single instanced pool (one draw call) covers the whole near set;
  alternate shapes (10 built in: sphere, box, the four gem polyhedra, cylinder, cone, torus,
  capsule) cost one pool each, created lazily only when a node of that shape actually promotes.
  The material is self-lit (a fixed key light + hemisphere ambient baked into the TSL, no scene
  lights). When `textureFacet` is null the shader never samples and NO texture is generated (flat
  tinted shading); when set, the 10 planet textures (earth, mars, moon, jupiter, neptune, venus,
  ice, lava, desert, star) are generated procedurally at runtime from the lib's own `hash` into one
  DataArrayTexture — no external image (the Pages CSP blocks them), deterministic forever (§10.1).

Live positions come from `frames.transformPoint` + spread + field-spin per promoted node (O(K)), so
a mesh tracks the spinning/morphing field exactly and sits precisely where its billboard was; that
billboard is faded (`iAlpha` → 0) while the node is a mesh and restored when it leaves the near set,
so the two never double-draw. The grid buckets a view snapshot rebuilt on settle / throttled under
motion — never per frame.

**Per-node texture-source overrides (tiered images + progressive LOD).** `opts.overrides` (node id →
`NodeSource`, live via `setOverrides`) pins a specific node to a different source than the facet
default: a forced built-in planet (`{ kind: 'procedural', planet }` — still an instanced instance,
just a fixed texture layer, zero extra draw calls) or a real image (`{ kind: 'image', tiers }`, e.g.
a Blue-Marble Earth). An image node can't share the pools' single material (each needs its own
texture), so it renders as its own unit-sphere `Mesh` with a self-lit textured material (the same
fixed key light + hemisphere ambient; opaque, `depthWrite`/`depthTest` so it occludes) — a small
managed set, cap `IMAGE_MESH_CAP` (16; overrides are rare). It is positioned/scaled/gated and its
billboard faded exactly like an instanced node, so draw calls grow only by the number of VISIBLE
image nodes (bounded, few). Each `ImageTier` is `{ px, load }`: `px` is the tier's resolution and
doubles as its on-screen threshold; `load` is a consumer-supplied async loader (`pickImageTier` picks
the sharpest tier whose `px` ≤ the node's on-screen px). Tiers load once (tracked in-flight), never
blocking the frame — the best already-loaded tier shows meanwhile and the material's map swaps when a
sharper tier resolves. A per-node working set of `{base, shown, desired}` plus an LRU budget
(`imageCacheBudget`, ~64 MB; `.dispose()` on eviction) bounds GPU memory and drops a node's high tiers
back toward the base as it shrinks or flies away. Still OPT-IN: with no `overrides`, no image meshes,
loaders, or cache exist — byte-identical to today, zero added cost.

## 15. Planetary terrain — the ground (surface collision + true-geometry relief)

The near-mesh (§14) resolves a node from mote → billboard → textured sphere → procedurally-detailed
sphere. The last band is the **ground**: descend toward a solid planet and its surface becomes real
displaced geometry — mountains and valleys with silhouettes you fly over and land on — continuous and
seamless with the textured sphere.

**Two pluggable fields over the sphere.** A planet is colour + height, each either computed (noise) or
sampled (tiles): `{ colour: procedural | image-tiers, height: procedural | image-tiers }`. The terrain
renderer only asks *"elevation at surface point P, at detail D?"* — it never cares whether the answer is
noise or a sampled DEM, so a procedural planet's mountains and the real Earth's mountains run the
identical displacement / tessellation / collision path. `TerrainHeightSource { height(cpu), node(tsl) }`
is the seam; swapping `earthTerrainSource` for a tiered-DEM source leaves everything else untouched.

**Displacement.** A unit icosphere is displaced along its normal in the vertex shader by
`height(dir)·amp·radius` (so amplitude ramps per-frame with no geometry rebuild); the surface normal is
recomputed analytically from two finite-difference height taps so relief is lit, collapsing to the
smooth sphere normal at `amp = 0`. The same procedural colour as the instanced sphere is draped on top,
so only the lighting bends as relief appears. Height reuses the planet's OWN land-mask noise, so relief
registers to the coastlines (ocean flat, land rises) + a ridged fbm for ranges.

**Seamless handoff.** Amplitude ramps `heightScale·(1 − smoothstep(lo, hi, altitude/radius))` → 0 at the
swap altitude (sub-pixel) up to full near the ground; nearmesh suppresses that one node's instanced
sphere so the two never double-draw. Discrete icosphere subdivisions swap by altitude (geometry LOD only
where you're close). It is the geometry resolving in, exactly as the colour detail (§14) sharpens — no
switch to hide.

**Collision is the terrain.** `nearestSurface()` returns a `terrain(dir)` sampler for the active planet;
the flycam (§6) floors the camera at `terrain(surfN) + hover` and scales speed by the altitude above it,
so you land ON ridges and rise/fall with the relief, not at the bounding sphere. Absent the sampler it is
the stage-A sphere floor.

Still OPT-IN and bounded: terrain exists only when the LOD is on, the node is textured + detailed, and
you are close to a solid planet — one planet at a time, created lazily, nothing allocated otherwise. The
full-generality target (adaptive CDLOD tessellation, every solid planet, real tiered DEMs, erosion) is
the §12 Milky Way test's last mile; this is its first true-ground implementation.
