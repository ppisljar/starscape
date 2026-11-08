# starscape

A graph visualization library for very large graphs (10⁶+ nodes) rendered as a 3D field you fly
through like a game — nested rotating hierarchies of galaxies and systems, levels of detail that
resolve from a mote to a readable page as you approach, deterministic placement, and two draw
calls at every scale.

**Status: design stage.** Read [`DESIGN.md`](DESIGN.md) — it is the contract everything else
follows. The proven predecessor this rebuilds (and its measurements) lives in
`SecondBrain/tools/vault-site-3d`.

- TypeScript, three.js `WebGPURenderer` + TSL materials (automatic WebGL2 fallback)
- Columnar data model: typed arrays + CSR, never an object per node
- N-level grouping: every level picks its shape, ordering, density, scatter and spin
- GPU-side morphing for arrangement switches, 2D↔3D transposition and animation
- Customizable at three depths: themes (data), shader hooks (TSL), custom layers (objects)
- Data-agnostic core: the app supplies adapters and content callbacks

Roadmap and stage-by-stage commits: see `DESIGN.md` §11.
