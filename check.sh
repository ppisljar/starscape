#!/bin/sh
# Everything that can be verified without a browser. Run it before claiming anything works.
set -e
cd "$(dirname "$0")"

echo "— typecheck (tsc --noEmit: the scope-aware pass that catches a deleted function)"
npm run --silent typecheck

echo "— lint (eslint flat config, typescript-eslint recommended)"
npm run --silent lint

echo "— tests (vitest, node environment)"
npm run --silent test

echo "— build (tsc -p tsconfig.build.json → dist/ js + d.ts + sourcemaps)"
npm run --silent build

echo "all green"
