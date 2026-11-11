# starscape

A graph visualization library for very large graphs (10⁶+ nodes) rendered as a 3D field you fly
through like a game — nested rotating hierarchies of galaxies and systems, levels of detail that
resolve from a mote to a readable page as you approach, deterministic placement, and two draw
calls at every scale.

**[▶ Playground](https://ppisljar.github.io/starscape/playground/)** · **[Docs](https://ppisljar.github.io/starscape/docs/)** · **[API reference](https://ppisljar.github.io/starscape/api/)**

The core is implemented and measured: the field renderer, the N-level placement engine, cameras,
the distance-driven zoom ladder, the schema-driven tuning panel, HDR bloom, and a deterministic
synthetic-corpus generator. [`DESIGN.md`](DESIGN.md) is the contract everything follows;
[`SCALE.md`](SCALE.md) has the measured numbers (10⁴ at 120 fps, 10⁶ flyable at 40–67 fps, three
draw calls at every scale). The proven predecessor this rebuilds lives in
`SecondBrain/tools/vault-site-3d`.

- TypeScript, three.js `WebGPURenderer` + TSL materials (automatic WebGL2 fallback)
- Columnar data model: typed arrays + CSR, never an object per node
- N-level grouping: every level picks its shape, ordering, density, scatter and spin
- GPU-side morphing for arrangement switches, 2D↔3D transposition and animation
- Customizable at three depths: themes (data), shader hooks (TSL), custom layers (objects)
- Data-agnostic core: the app supplies adapters and content callbacks

## Using it

`three` is a **peer dependency** — the consuming app installs and owns it (so there is one
copy, and `instanceof` checks never split across two). `package.json` `exports` points at the
built `dist/`, so build the library before an external app consumes it:

```sh
npm install          # in starscape/
npm run build        # emits dist/ (js + d.ts + sourcemaps) — the product
```

Then, from an app, depend on the built package and compose the pieces you need:

```ts
import {
  buildGraph, createField, createFlycam, createLadder, createOverlay,
  buildGrid, pickRay, createConfigStore, createPanel, compileHierarchy,
} from 'starscape';
import { PerspectiveCamera, Scene, WebGPURenderer } from 'three/webgpu';

// 1. data: your adapter maps a file format to typed arrays + callbacks (the core never fetches)
const graph = buildGraph({ n, positions, links, columns, facets, labels });

// 2. the field: two draw calls, TSL materials, a WebGL2 fallback when WebGPU is absent
const field = createField(graph, { heightCss: innerHeight, fov: 55 });
const scene = new Scene();
for (const o of field.objects) scene.add(o);

// 3. interaction: a camera (createFlycam / createOrbitCam), a pick grid (buildGrid + pickRay),
//    the zoom ladder with YOUR content provider (createLadder + createOverlay), and the
//    schema-driven tuning panel (createConfigStore + createPanel). See demo/main.ts.
```

A complete, real-data consumer — vault adapters, a wiki-page content provider, arrangements,
presets, the stats overlay — lives in `SecondBrain/tools/vault-site-starscape`, which depends
on this package exactly the way an external app would (`"starscape": "file:…"`). The library's
own `demo/` consumes `src/` directly for development.

Roadmap and stage-by-stage commits: see `DESIGN.md` §11.
