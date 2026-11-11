/* The predecessor's config existed for one rule: `no-undef`. A refactor deleted
 * `writeLinkPositions()`, every module still parsed, the tests (which never imported three.js)
 * still passed, and the viewer threw on its first animated frame. Scope-aware analysis catches
 * that class of bug in under a second.
 *
 * In TypeScript that job belongs to the compiler — `tsc --noEmit` is scope-aware across the
 * whole project, so classic `no-undef` is redundant (and typescript-eslint tells you to turn it
 * off on TS files). What lint adds on top of the type check:
 *   - the typescript-eslint recommended set (unsafe-ish patterns tsc tolerates), and
 *   - `no-unused-vars` as a *warning*, exactly as the predecessor had it: a variable left
 *     behind by a refactor is worth knowing about and not worth blocking on. `_`-prefixed
 *     names are the deliberate-discard convention.
 */
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist/', 'site-dist/', 'tmp/', 'node_modules/', 'coverage/'] },
  ...tseslint.configs.recommended,
  {
    files: ['**/*.ts'],
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'warn',
        { args: 'none', varsIgnorePattern: '^_' },
      ],
    },
  },
);
