/* Minimal Node built-in declarations for tests that read the repo from disk
 * (package.test.ts). Deliberately NOT @types/node: the library is browser-only
 * ("lib": ["ES2022", "DOM"]), and installing the full Node ambient types would leak
 * `process`, `Buffer` etc. into src/'s type space, where using them must stay an error.
 * Only what the tests call is declared; extend it here when a test needs more. */

declare module 'node:fs' {
  export function readFileSync(path: string, encoding: 'utf8'): string;
  export function existsSync(path: string): boolean;
}

declare module 'node:path' {
  export function join(...parts: string[]): string;
  export function resolve(...parts: string[]): string;
  export function dirname(p: string): string;
}

declare module 'node:url' {
  export function fileURLToPath(url: string | URL): string;
}
