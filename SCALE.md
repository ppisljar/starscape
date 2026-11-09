# Where starscape actually stands

Measured, not estimated — 2026-11-09, MacBook (Chrome, WebGPU backend with storage-buffer
vertex pulling, `devicePixelRatio` 2, 1512×768 CSS px), the in-page synthetic corpora
(`?n=<count>`, seed 42), served by `npm run dev -- --port 5199` with a WARM vite module cache
(the first hit after an edit pays ~7 s of on-the-fly TS transform that is vite's, not the
library's — reload once before trusting a boot number).

Every row is the median of 90 `requestAnimationFrame` intervals at the ASSERTED home framing
(`__starscape.homeRatio()` ≈ 1.0 before every number — the predecessor's SCALE.md records that
every wrong number it ever produced was a moved viewpoint), taken by `__starscape.measure(90)`.
Boot is navigationStart → first rendered frame (the `starscape:first-frame` mark the demo
plants). `jsFrameMs` is the per-frame CPU of the whole update loop (cameras, tweens, frames,
grid upkeep, pick, ladder), median of 90, measured AROUND the updates and never around
`renderer.render` — `gl.finish()` does not block in Chrome and WebGPU submits asynchronously,
so timing the render call returns nothing truthful.

Budgets are the boot notch's own defaults (`src/field/quality.ts`, applied through the
session-scoped config keys) — nothing below was hand-tuned per run.

## The table (§9: acceptance, not aspiration)

| nodes | links drawn / total | boot | ms/frame | fps | jsFrameMs | draw calls | state |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10⁴ | 39,022 / 39,022 | 0.31–0.38 s | 8.2 | 120–122 | 0.2–0.3 | 3 | frozen, dpr 2 |
| 10⁴ | 39,022 / 39,022 | — | 8.1–8.3 | 120–123 | 0.3 | 3 | galaxies, **all spins on** |
| 10⁵ | 199,516 / 399,032 | 0.52–0.67 s | 8.5–8.6 | 116–118 | 0.3 | 3 | frozen, dpr 2, half links |
| 10⁵ | 199,516 / 399,032 | — | 32.7–33.0 | 30.3 | 0.1 | 3 | galaxies, all spins on |
| 10⁶ | 159,963 / 3,999,065 | 1.76–1.82 s | 16.2–17.4 | 57–62 | 0.1 | 3 | frozen, dpr 1, 4% links, flat ≥ 9 px |
| 10⁶ | 159,963 / 3,999,065 | — | 25.8–31.9 | **31–39** | 0.1–0.3 | 3 | galaxies, **all spins on** |

Every band above is min–max over at least two 90-frame windows, the 10⁶ rows over at least
two independent warm reloads (the stage-9 gate re-measured every row; an earlier draft of the
10⁶ frozen row said 21.6 ms / 46 fps from a mid-pass build and did not reproduce — the
shipped code measures 16.2–17.4 ms over four reloads. Quote the band, never a draft).

Draw calls are 2 + the renderer's background pass (one fullscreen triangle — `info.render`
shows 20,001 triangles at 10⁴ nodes). The §9 table holds on every row:

- **10⁴ — vsync, boot < 0.5 s** ✓. 8.2 ms is the 120 Hz frame, with or without spins. The
  0.2–0.3 ms of frame CPU is the ladder rig (DOM for ~20 live labels + the candidate sweep),
  not the field — the `field` segment of the breakdown reads 0.0.
- **10⁵ — vsync with budgets at defaults, boot < 1 s, CPU < 0.5 ms** ✓ (8.5 ms, 0.3 ms). The
  galaxies+spins row is NOT vsync (33 ms): the spiral look draws longer lines than the frozen
  gaussian blobs and the half-link budget is ~200 k segments of fill at dpr 2. Links off it is
  8.3 ms again — the spins themselves cost nothing measurable.
- **10⁶ — flyable ≥ 30 fps with budgets engaged and ALL spins on, boot < 3 s, CPU < 1 ms** ✓
  (every 90-frame window medianed 31–39 fps — the spinning fill breathes, see below — at
  0.1–0.3 ms frame CPU, 1.76–1.82 s boot of which ~0.7–0.9 s is synth generation + buildGraph).
  p90 is 34–41 ms; the worst frame in a 90-frame window is 57–59 ms — that spike is the
  once-per-second pick-grid rebuild under spin (below), never the median.

## What costs the frame, by subtraction

At 10⁶, galaxies, all spins on, the boot-notch defaults (dpr 1, 4% links, flat ≥ 9 px). One
phase of the breathing fill — the full band is 25.8–31.9 ms; the A/B rows compare within the
phase, which is the only honest way to read them:

| | ms/frame | fps |
|---|---:|---:|
| as shipped | 29.3 | 34 |
| links off | 14.7 | 68 |
| render scale back to dpr 2 | 52.0 | 19 |
| flat-glyph branch disabled (flatPx 2, under the clamp — never fires) | 30.5 | 33 |

- **Links are still the bill, and they are only half fill now.** In the dense spiral look a
  drawn link costs ~100 ns/frame — 666 k links measured 83 ms, 320 k measured 42, 160 k
  measured 27, linear — but dropping resolution 1 → 0.6 only bought 1.25×. Line fill scales
  linearly with resolution (a line's fragment count is its pixel LENGTH), and the other half
  is vertex work: every link vertex pulls both endpoints from the position store (morph mix ×
  2 + pivot texels), which is the price of never re-uploading a link buffer. That trade is the
  design (§4) and it wins overall — rotation is free — but it moves the link budget's knee:
  the 10⁶ notch had to drop to 4% (the predecessor shipped 8–16%) to clear 30 fps in the
  heaviest arrangement. The ordered buffer means those 160 k are the most informative links
  (intra-group, short), and the overlay reports drawn/total (§10.4).
- **The arrangement is part of the cost.** The same budgets measure ~17 ms frozen and
  26–32 ms as spinning galaxies at 10⁶ — and at 10⁵, 8.5 ms frozen vs 33 ms galaxies. Spiral
  discs spread systems into long on-screen lines; gaussian blobs keep them short. A fill
  number quoted without its arrangement is as wrong as one without its camera.
- **Spins are free.** 10⁵ galaxies, links off: 8.3 ms vsync with both levels turning. At 10⁶
  the `field` segment (frames.advance + pivot-store sync, 595 pivots) medians 0.0–0.1 ms —
  §4's O(pivots) claim, confirmed. The sync-only-when-moved path needed no fixing: `advance`
  already tests the per-tick delta.
- **Flat glyphs buy ~1 ms at dpr 1** (they bought more for the predecessor at dpr 2, where
  node fill was double). Kept: the branch costs nothing and the halo it skips is unreadable
  under 9 px anyway.

## The two CPU sinks stage 9 actually removed

The §9 CPU budgets were not lost to rendering at all. At 10⁶ the baseline frame spent
**67.6 ms in JavaScript** — the whole fps story at every scale above 10⁴ until it was fixed:

1. **The ladder's candidate pass was O(n) per frame.** At the home framing its reach sphere
   covers the whole field, so the stage-7 grid walk visited every node, every frame: 1 ms at
   10⁴, 3 ms at 10⁵, 67 ms at 10⁶ — and ~30,000 nodes genuinely pass the "could carry a
   label" px gate at 10⁶, so no admission floor fixes it alone. The rewrite (demo `main.ts`,
   THE CANDIDATE RECIPE): an exact NEAR pass every frame (cells within the card distance —
   cards and the open tier need live depth), plus an INCREMENTAL far sweep, 4,096 cells per
   frame, that keeps a bounded top-K-by-px selection and prunes whole cells with the per-cell
   size maximum (`cellMaxOf`) against the cell's exact minimum view depth. Published far
   nodes (≤ 64 shipped: `heapLimit = min(FAR_CAP 256, 2 × max(keep 32, earn 20))` — 256 is
   the scratch arrays' capacity, not the published band, which saturates at exactly 64/64)
   are re-projected fresh every frame; only MEMBERSHIP is a generation stale (~6 frames at
   10⁵; 61 frames at 10⁶ zoomed out — ~1 s frozen, ~2 s under spin), which the label dwell
   hysteresis absorbs by design. One semantic refinement, deliberate: far admission is by resting projected size,
   so the cursor pull promotes within the published band rather than recruiting from below it
   — at the only framings where the band oversubscribes, glyphs are overlapping pixels and
   individual pointing is already meaningless. Cost now: ≤ 0.3 ms at every scale.
2. **`roomAt` cost 0.7 ms parked.** The flycam probes the nearest-node distance twice per
   frame (position + step destination); each probe walks the grid (~0.35 ms at 10⁶). Cached
   by displacement: the nearest node can approach by at most (probe movement + node drift
   since the walk), so `cached − slack` is a safe lower bound until the slack passes half a
   cell. A parked camera pays nothing; fast flight re-pays the real walk.

With both fixed, 10⁶ frame CPU medians 0.1 ms frozen and 0.1–0.3 ms spinning. The breakdown
(`jsBreakdown()`): ladder 0–0.3 — the sweep slice, plus 20 label DOM writes in the frozen
look (the galaxies look publishes no labels at this framing, see Honest limits) — field
0–0.1 (frames + pivot sync), everything else under the 0.1 ms clock. At these sizes every
segment sits at the clock's own granularity; read the caveat under "Measuring it yourself"
before quoting one.

## The one stall left, reported not hidden

Under spin the pick grid's bucketing goes stale; walks widen by the tracked drift bound and
stay CORRECT at any staleness, but the bound must eventually be reset by an O(n) rebuild
(viewPositions through the frame chains + counting sort + cellMaxOf): **37–68 ms at 10⁶**.
Stage 9 stretched its leash — drift tolerance a full cell (was a third), cooldown 1 s (was
250 ms), and the drift RATE now comes from pivot radii (Σ |rate| × lever per level, ×1.5)
instead of 0.5 × diagonal, which over-reported ~5× — so the stall lands about once a second,
in the p90/worst columns, never the median. Slicing the rebuild like the far sweep is the
known next step if that hitch ever matters more than it does.

## What was changed, and what each change buys

| change | where | what it buys |
|---|---|---|
| boot quality notch | `src/field/quality.ts` (+ demo boot: session config keys) | the corpus picks the budgets at boot, no feedback loop, user overrides via the same knobs; 10⁶ drops from 2.7 s/frame naive to 26–32 ms |
| 10⁶ notch at 4% links + dpr 1 + flat ≥ 9 px | same | the measured knee: ≥ 30 fps in the heaviest spinning arrangement; 8–16% (the ported numbers) measured 42–83 ms there |
| flatPx a live uniform, routed | `field/nodes.ts`, `Field.setFlatPx`, demo route | the stage-3 flat branch NEVER FIRED before: FLAT_PX (2) sits under the NODE_MIN_PX clamp (2.2), so no glyph ever projected below it — found only because stage 9 went looking; ~1 ms at 10⁶ dpr 1 once lifted |
| near pass + incremental far sweep + `cellMaxOf` pruning | demo `main.ts`, `src/pick/grid.ts` | frame CPU 67.6 → ≤ 0.3 ms at 10⁶; the ladder stays alive while spinning (§6) |
| render-quality telemetry in the overlay | demo `main.ts` | the two notch levers that drop resolution/detail rather than content (`renderScale`, `flatPx`) are shown, not hidden (§10.7) — links drawn/total already reported the one that drops content |
| `roomAt` displacement cache | demo `main.ts` | 0.7 → ~0 ms parked at 10⁶ |
| drift bound from pivot radii; rebuild leash 1 cell / 1 s | demo `main.ts` | rebuild stalls 4×/s → ~1×/s; pick walks widen less under spin |
| boot mark + `measure()` + `homeRatio()` + segment medians | demo `main.ts` | the §9 methodology is one call, viewpoint asserted, not eyeballed |

## Measuring it yourself — read this first

The predecessor's warning stands verbatim: **every wrong number is a moved viewpoint** until
proven otherwise. Its three traps (home button ≠ boot framing; flying in gets cheaper;
localStorage keeping a toggle you forgot) all still apply, plus three new ones found here:

- **The arrangement is part of the measurement.** Frozen and galaxies differ 1.5–4× in frame
  time at identical budgets. Say which one a number came from.
- **A spinning field's fill breathes.** Galaxy discs turn edge-on and face-on; successive
  90-frame medians at 10⁶ galaxies varied 25.8–31.9 ms with nothing changed. Compare A/B
  within one phase, or expect ±10%.
- **The dev server's first hit is vite, not boot.** A cold module graph adds seconds of
  transform time; reload once and measure the second load. (The predecessor served static
  files and never saw this.)
- `__starscape.measure(90)` returns median, p90 and worst, plus the spin state, dpr,
  homeRatio and the link budget it saw — quote the whole object, the context is the number.
- **Sub-millisecond jsFrameMs is clock-granularity data.** `performance.now()` ticks at
  0.1 ms here, so a 0.1–0.3 ms median is honest to about ±0.1–0.2 ms and two sessions in the
  same state can read 0.1 and 0.3 with nothing changed (measured). Quote bands, compare them
  against the §9 budgets — never a 0.1 against a 0.2.
- The session config keys (`linkBudget`, `renderScale`, `flatPx`) are re-derived every boot
  and never persisted, so a measurement session cannot leak budgets into the next one. Keys
  you touch in the tuning panel DO persist — reset before measuring.

## Honest limits

- One machine, one GPU, one window size, one backend (WebGPU; the WebGL2/DataTexture fallback
  path is untimed here). The curve's shape transfers; the milliseconds do not.
- Home framing only — the worst case for fill, the right one to budget against. Flying into a
  galaxy culls most of the field and runs faster; flying fast re-pays the roomAt walk.
- The synthetic corpus is MORE clustered than a real vault (~72% intra-system links), which
  flatters the link budget's "intra-group first" ordering. A flatter graph would spend the
  same budget on longer, less local links.
- The 10⁵ galaxies look misses vsync (30 fps at dpr 2 with half links). The §9 10⁵ row is the
  boot state and holds; if the dense look at 10⁵ must be vsync too, the levers are the same
  two (links, dpr) and the notch table is one line to change.
- Label membership at 10⁶ zoomed far out refreshes once per far-sweep generation (61 frames:
  ~1 s frozen, ~2 s in the spinning galaxies look). Flying toward anything shrinks the sweep
  box and the refresh follows the frame rate again. Mounted labels track per frame regardless
  (they ride `refreshViewOf`).
- **The galaxies look publishes NO labels at the 10⁴–10⁶ home framings, by design.** The
  spiral arrangement spreads the field wider than the frozen synth layout (extent radius
  2,787 vs 1,677 at 10⁶), so at its refit home framing the largest glyph projects ~6 px —
  under the namedPx 13 admission gate: a mote field, named 0 / far 0 / cand 0 in the
  overlay. The frozen home framing publishes named 20/34 · far 64/64 the whole time at every
  scale (verified at 10⁴/10⁵/10⁶ over repeated sweep generations), and flying in restores
  the full ladder in any arrangement. A ladder number quoted without its arrangement is as
  wrong as a fill number quoted without one.
