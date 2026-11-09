/* The zoom-ladder module (DESIGN.md §6, stage 7): the pure state machine (./ladder), its
 * numbers (./constants), the content contract + synthetic provider (./provider), and the DOM
 * overlay that renders what the machine decides (./overlay). This barrel is the `/ladder`
 * subpath export of §8; the root index re-exports it so plain `starscape` works too. */

export * from './constants';
export * from './ladder';
export * from './provider';
export * from './overlay';
export * from './types';
