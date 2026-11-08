import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
    // Tests arrive with stage 2 (core data). Until then the gate must still run green.
    passWithNoTests: true,
  },
});
