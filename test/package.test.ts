/* The package's consumable surface (DESIGN.md §8: "package.json exports + dist/ are the
 * product"). Stage 10 found the bug this test pins: package.json shipped with NO "exports",
 * "main" or "types" at all, so an external `file:`/npm consumer could not even resolve
 * `import 'starscape'` — every green check ran against src/, never against what a user gets.
 *
 * check.sh runs tests BEFORE build, so nothing here may require dist/ to exist. Instead the
 * test exploits the build's own contract (tsconfig.build.json: rootDir src → outDir dist,
 * declaration: true): a dist/<p>.js entry is real iff src/<p>.ts exists. That stays true
 * whatever tsc emits, so the test can verify every export target without building.
 */
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')) as {
  main?: string;
  module?: string;
  types?: string;
  files?: string[];
  exports?: Record<string, unknown>;
  peerDependencies?: Record<string, string>;
};

/** dist/x/y.js | dist/x/y.d.ts → the src/x/y.ts the build compiles it from. */
const srcOf = (distPath: string): string =>
  distPath.replace(/^\.\/dist\//, 'src/').replace(/\.d\.ts$/, '.ts').replace(/\.js$/, '.ts');

describe('package.json — the consumable surface (§8)', () => {
  it('declares an exports map with a root entry', () => {
    expect(pkg.exports, 'package.json must carry "exports" — without it no consumer resolves the package').toBeDefined();
    expect(pkg.exports!['.'], 'the root export "." must exist').toBeDefined();
  });

  it('every export condition points into dist/ at a path the build will emit', () => {
    for (const [subpath, entry] of Object.entries(pkg.exports!)) {
      if (subpath === './package.json') {
        expect(entry).toBe('./package.json');
        continue;
      }
      expect(typeof entry, `export "${subpath}" must be a conditions object`).toBe('object');
      const cond = entry as Record<string, string>;
      expect(cond['types'], `export "${subpath}" must carry a "types" condition`).toMatch(/^\.\/dist\/.*\.d\.ts$/);
      expect(cond['default'], `export "${subpath}" must carry a "default" condition`).toMatch(/^\.\/dist\/.*\.js$/);
      // "types" must come first: Node and TS pick the FIRST matching condition.
      expect(Object.keys(cond)[0]).toBe('types');
      for (const target of Object.values(cond)) {
        const src = srcOf(target);
        expect(existsSync(join(root, src)), `export "${subpath}" → ${target}: no ${src} for the build to compile`).toBe(true);
      }
    }
  });

  it('main/module/types agree with the root export (older resolvers read these)', () => {
    const rootExport = pkg.exports!['.'] as Record<string, string>;
    expect(pkg.main).toBe(rootExport['default']);
    expect(pkg.module).toBe(rootExport['default']);
    expect(pkg.types).toBe(rootExport['types']);
  });

  it('ships dist/ and declares three as a peer', () => {
    expect(pkg.files).toContain('dist');
    expect(pkg.peerDependencies?.['three']).toBeDefined();
  });
});
