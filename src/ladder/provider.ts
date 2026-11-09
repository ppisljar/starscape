/* The ladder's content contract (DESIGN.md §6, §12). The library renders frames and layout;
 * the app supplies meaning — content arrives through callbacks, the core never fetches and
 * never knows what a "wiki page" is. Two calls, two costs:
 *
 *   - `card(i)` is SYNCHRONOUS and must be cheap: it runs on card mount, inside the frame
 *     that crossed the card threshold. Title and a short summary only.
 *   - `body(i)` is ASYNC and may be arbitrarily slow (a fetch, a markdown render): the
 *     overlay shows a loading state and guards against stale resolutions, so a provider
 *     never has to worry about the camera outrunning its network.
 *
 * `body` may resolve to an HTML string or a ready element. The element form is the §12
 * mote→mesh seam in miniature: what fills the locked rect is the app's business — the
 * library's job is only the handoff.
 *
 * `makeSynthProvider` generates deterministic prose from hash(i, salt) (§10.1: no RNG, no
 * clock — the same node reads the same page forever, so screenshots and tests are
 * reproducible). It is demo/test fuel for corpora that have no real bodies, exported because
 * every consumer bootstrapping an integration needs exactly this stub.
 */

import { hash } from '../core/hash';
import type { Graph } from '../core/types';

/** What the card tier shows the moment it mounts: a title, and optionally one short summary
 *  that earns its space as the box grows (the predecessor's `.c-sum`). */
export interface CardContent {
  title: string;
  summary?: string;
}

/** The app side of the ladder: WHEN is the ladder's decision, WHAT is the provider's. */
export interface ContentProvider {
  /** Synchronous, cheap — called on card mount (and label mount, for the title). */
  card(i: number): CardContent;
  /** The open tier's body: an HTML string (innerHTML'd into the card body) or a ready
   *  element (adopted as the body's only child). Called when the open tier fires; the
   *  overlay handles loading state and stale responses. */
  body(i: number): Promise<string | HTMLElement>;
}

/* ---- the synthetic provider ---------------------------------------------------------- */

/* Word banks for the generated prose. The vocabulary is the library's own, so an open card
 * in the demo reads as documentation-flavoured filler rather than lorem ipsum — and the
 * banks are module constants: nothing here holds per-node strings eagerly (§10.2), every
 * sentence is built on demand from hash draws. */
const ADJS = [
  'frozen', 'unclamped', 'deterministic', 'nested', 'importance-ordered', 'projected',
  'rotating', 'columnar', 'measured', 'rank-banded', 'settled', 'instanced', 'sparse',
] as const;
const NOUNS = [
  'field', 'ladder', 'frame chain', 'pivot', 'budget', 'glyph', 'halo', 'corpus',
  'spacing', 'quaternion', 'buffer', 'facet', 'ramp', 'tier', 'cursor', 'dwell',
] as const;
const VERBS = [
  'resolves', 'grows in pixels', 'locks', 'drifts', 'settles', 'spins about its pivot',
  'blooms', 'fades with depth', 'ranks by score', 'anchors the view', 'mounts',
  'holds through the wobble', 'yields the slot',
] as const;
const TAILS = [
  'at every scale', 'under the lock', 'without a clock', 'in rank order',
  'past the threshold', 'frame for frame', 'by construction', 'on approach',
  'in CSS pixels', 'with no per-node state', 'before the camera arrives',
] as const;

/** One deterministic draw stream per (i, salt): draw k is hash(i, salt + k·FNV_PRIME). */
function sentenceAt(i: number, salt: number, k: number): string {
  const d = (j: number): number => hash(i, (salt + (k * 8 + j) * 0x01000193) | 0);
  const adj = ADJS[(d(0) * ADJS.length) | 0]!;
  const noun = NOUNS[(d(1) * NOUNS.length) | 0]!;
  const verb = VERBS[(d(2) * VERBS.length) | 0]!;
  const tail = TAILS[(d(3) * TAILS.length) | 0]!;
  return `The ${adj} ${noun} ${verb} ${tail}.`;
}

/** The default salt — any fixed value works; exported so a test can prove the stream is a
 *  pure function of (graph, salt). */
export const SYNTH_PROVIDER_SALT = 0x5eed;

/**
 * Deterministic prose for a corpus with no real bodies: title from `graph.labelOf`, a
 * two-sentence summary, and a few paragraphs of body — all pure functions of (i, salt).
 * The demo and the tests use it; a real app replaces it with its own provider.
 */
export function makeSynthProvider(graph: Graph, salt: number = SYNTH_PROVIDER_SALT): ContentProvider {
  return {
    card(i: number): CardContent {
      return {
        title: graph.labelOf(i),
        summary: `${sentenceAt(i, salt, 0)} ${sentenceAt(i, salt, 1)}`,
      };
    },
    body(i: number): Promise<string> {
      // 3–5 paragraphs of 2–4 sentences each; draw indices disjoint from the summary's.
      const nPara = 3 + ((hash(i, (salt + 0x9e37) | 0) * 3) | 0);
      const parts: string[] = [];
      let k = 2; // sentences 0 and 1 are the summary's
      for (let p = 0; p < nPara; p++) {
        const nSent = 2 + ((hash(i, (salt + 0x85eb + p) | 0) * 3) | 0);
        let para = '';
        for (let s = 0; s < nSent; s++) para += (s > 0 ? ' ' : '') + sentenceAt(i, salt, k++);
        parts.push(`<p>${para}</p>`);
      }
      return Promise.resolve(parts.join('\n'));
    },
  };
}
