import { defineConfig } from 'vite';

/* Builds the `demo/` into the static playground that ships on GitHub Pages
 * (https://ppisljar.github.io/starscape/playground/). The demo IS the library consumed as an
 * external user would (DESIGN.md §8), on deterministic synth corpora — so the public playground
 * and the dev demo are one artifact, never two to keep in sync.
 *
 * `base` is the Pages sub-path for a `build`, but `/` under `vite dev` so `npm run dev` serves at
 * the root locally. Output lands in the assembled site tree the Pages workflow uploads. */
export default defineConfig(({ command }) => ({
  root: 'demo',
  base: command === 'build' ? '/starscape/playground/' : '/',
  build: {
    outDir: '../site-dist/playground',
    emptyOutDir: true,
    target: 'es2022',
    chunkSizeWarningLimit: 4096,
  },
}));
