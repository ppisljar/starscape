/* The ordering library (DESIGN.md §3.3 knob 2): WHO sits WHERE along the shape's primary
 * axis. Importing this module registers `byFacet`, `byColumn`, `byAdjacency` and `byHash`;
 * `byIndex` is the engine's built-in (src/place/engine.ts) and is deliberately not
 * duplicated here.
 *
 * Every ordering honours the contract in ./types: the result is a PERMUTATION of the
 * equal-spaced quantiles (rank + 0.5) / count — an ordering decides seating, never spacing
 * (spacing is the density knob's job, and the inverse-CDF warp is only meaningful over
 * equal-spaced inputs).
 *
 * Determinism discipline (§10): every tie anywhere breaks by member id — never by position
 * in the `members` buffer, which depends on how the caller assembled the group (iteration
 * order is the classic determinism leak). Randomness is `hash(member, salt)`; no RNG, no
 * clock. Ranking is argsort over index arrays reading typed columns — no per-member
 * objects, no closures over node objects.
 *
 * The facet/column KEY arrives via `PlacementSpec.stringParams` (`stringParams.facet`,
 * `stringParams.column`) — the stage-5 plumbing decision documented on PlacementSpec:
 * numeric `params` stay pure numbers, and the engine hands the string bag to orderings
 * only.
 */

import { hash } from '../core/hash';
import type { Graph } from '../core/types';
import { registerOrdering } from './registry';
import type { Ordering, PlaceInput } from './types';

/* ------------------------------------------------------------------ shared helpers */

/** The graph, or a descriptive throw — an ordering that reads columns/facets/CSR must
 *  never guess on `null` (PlaceInput contract). */
function requireGraph(inp: PlaceInput, who: string): Graph {
  if (inp.graph === null) {
    throw new Error(
      `ordering "${who}" needs a graph (columns/facets/CSR) but PlaceInput.graph is null — ` +
        `at levels without a graph use byIndex or byHash`,
    );
  }
  return inp.graph;
}

/** The named string param, or a descriptive throw — a byFacet without a facet key is a
 *  spec typo, not a default. */
function requireStringParam(
  stringParams: Readonly<Record<string, string>> | undefined,
  name: string,
  who: string,
): string {
  const v = stringParams?.[name];
  if (v === undefined) {
    throw new Error(`ordering "${who}" needs stringParams.${name} (the ${name} key) in the spec`);
  }
  return v;
}

/** Member ids must index the graph's columns/facets/CSR; an out-of-range id would read
 *  `undefined` into arithmetic and surface as NaN geometry three modules later. */
function checkMemberBounds(members: Uint32Array, count: number, n: number, who: string): void {
  for (let k = 0; k < count; k++) {
    if (members[k]! >= n) {
      throw new RangeError(
        `ordering "${who}": member id ${members[k]!} out of range for graph with n=${n}`,
      );
    }
  }
}

/**
 * Positions `0 … count-1` sorted so `members[idx[r]]` ascends with `r` — the canonical
 * "by member id" iteration every tie-break below uses. The common case (members already
 * ascending) is detected in one pass and skips the sort, same as the engine's byIndex.
 */
function ascendingIdOrder(members: Uint32Array, count: number): Uint32Array {
  const idx = new Uint32Array(count);
  for (let k = 0; k < count; k++) idx[k] = k;
  let sorted = true;
  for (let k = 1; k < count; k++) {
    if (members[k]! < members[k - 1]!) {
      sorted = false;
      break;
    }
  }
  if (!sorted) idx.sort((x, y) => members[x]! - members[y]!);
  return idx;
}

/* ------------------------------------------------------------------ byFacet */

/** `byFacet`: contiguous sectors in facet-value order — all members sharing a facet value
 *  occupy one contiguous run of quantiles, runs ordered by the facet's value INDEX (the
 *  dictionary order `facet.values` was compiled with), members inside a run ordered by
 *  ascending member id. On a ring this is the pie chart: one sector per category.
 *
 *  Counting sort over the facet's small-int column: O(count + |values|), no comparators.
 *  Reads `stringParams.facet`; throws on a null graph, a missing key, or an unknown facet
 *  (naming the known ones). */
const byFacet: Ordering = {
  key: 'byFacet',
  assign(inp, _params, stringParams): Float32Array {
    const g = requireGraph(inp, 'byFacet');
    const key = requireStringParam(stringParams, 'facet', 'byFacet');
    const facet = g.facets[key];
    if (facet === undefined) {
      const known = Object.keys(g.facets).sort().join(', ');
      throw new Error(`byFacet: unknown facet "${key}" — known: ${known || '(none)'}`);
    }
    const count = inp.count;
    const members = inp.members;
    checkMemberBounds(members, count, g.n, 'byFacet');
    const data = facet.data;
    const nValues = facet.values.length;

    // Count members per facet value, then exclusive-prefix-sum into write cursors.
    const cursor = new Uint32Array(nValues);
    for (let k = 0; k < count; k++) cursor[data[members[k]!]!]!++;
    let acc = 0;
    for (let v = 0; v < nValues; v++) {
      const c = cursor[v]!;
      cursor[v] = acc;
      acc += c;
    }

    // Scan members in ascending-id order so ranks inside a sector ascend by id — stable
    // against the caller's assembly order of `members`.
    const idx = ascendingIdOrder(members, count);
    const q = new Float32Array(count);
    for (let r = 0; r < count; r++) {
      const k = idx[r]!;
      const v = data[members[k]!]!;
      const rank = cursor[v]!;
      cursor[v] = rank + 1;
      q[k] = (rank + 0.5) / count;
    }
    return q;
  },
};

/* ------------------------------------------------------------------ byColumn */

/** `byColumn`: rank gradient by a numeric column — the member with the smallest value
 *  sits at quantile ≈ 0, the largest at ≈ 1 (ascending; put high values first by warping
 *  with a decreasing look or by negating the column at build time). NaNs are PARTITIONED
 *  to the top end — every NaN member seats after every real-valued one, NaNs among
 *  themselves and value ties both breaking by member id. The partition is load-bearing,
 *  not taste: NaN compares false both ways, so a comparator that let NaN pairs fall
 *  through to the id tie-break while numbers compare by value is INTRANSITIVE on mixed
 *  columns, and ECMA-262 leaves sort order with an inconsistent comparator
 *  implementation-defined — a determinism leak across JS engines (§10). With the
 *  partition the comparator is a consistent total order for any column contents.
 *
 *  Argsort of an index array over the typed column; no per-member objects are created.
 *  Reads `stringParams.column`; throws on a null graph, a missing key, or an unknown
 *  column (naming the known ones). */
const byColumn: Ordering = {
  key: 'byColumn',
  assign(inp, _params, stringParams): Float32Array {
    const g = requireGraph(inp, 'byColumn');
    const key = requireStringParam(stringParams, 'column', 'byColumn');
    const column = g.columns[key];
    if (column === undefined) {
      const known = Object.keys(g.columns).sort().join(', ');
      throw new Error(`byColumn: unknown column "${key}" — known: ${known || '(none)'}`);
    }
    const count = inp.count;
    const members = inp.members;
    checkMemberBounds(members, count, g.n, 'byColumn');
    const vals = column.data;

    const idx = new Uint32Array(count);
    for (let k = 0; k < count; k++) idx[k] = k;
    idx.sort((x, y) => {
      const a = vals[members[x]!]!;
      const b = vals[members[y]!]!;
      const an = a !== a; // NaN, allocation-free
      const bn = b !== b;
      if (an !== bn) return an ? 1 : -1; // NaNs after every real value
      if (a < b) return -1;
      if (a > b) return 1;
      return members[x]! - members[y]!; // value ties and NaN-vs-NaN by member id
    });

    const q = new Float32Array(count);
    for (let r = 0; r < count; r++) q[idx[r]!] = (r + 0.5) / count;
    return q;
  },
};

/* ------------------------------------------------------------------ byAdjacency */

/* Scratch for the group-membership test, generation-marked so it never needs clearing
 * (the predecessor's `S.mark[i] = gen` trick from arrange.js): `mark[i] === adjGen` means
 * node i is in the CURRENT group, and `pos[i]` is then its position k in `members`. Grown
 * monotonically to the largest graph seen; module state is safe because placement is
 * synchronous and single-threaded. */
let adjMark = new Int32Array(0);
let adjPos = new Uint32Array(0);
let adjGen = 0;

/** `byAdjacency`: seriation so linked members sit near each other along the axis.
 *
 *  THE HEURISTIC, honestly: a single breadth-first search over the graph's CSR,
 *  restricted to the group's members —
 *    - start at the member with the highest full-graph degree (CSR slot count), ties to
 *      the lowest member id — a hub is what the nodes around it are about (the same
 *      reasoning as the predecessor's anchor cap in arrange.js);
 *    - neighbours are visited in CSR slot order, which buildGraph made
 *      importance-ordered, so stronger links seat closer — deterministic, not tuned;
 *    - members the BFS never reaches (other components, or only linked through
 *      non-members) are APPENDED IN ASCENDING MEMBER ID ORDER after the reached ones.
 *      They get no seriation of their own — a deliberate simplicity, not an oversight.
 *  BFS visit order is a crude seriator — it keeps linked nodes within a few ranks of each
 *  other but is nothing like optimal (no Cuthill–McKee degree sorting, no spectral
 *  ordering). It is O(members + their CSR slots), allocation-light, and fully
 *  deterministic, which is the §10 bar it has to clear.
 *
 *  Assumes member ids are distinct within the group (the PlaceInput contract — members
 *  partition nodes). Throws on a null graph. */
const byAdjacency: Ordering = {
  key: 'byAdjacency',
  assign(inp): Float32Array {
    const g = requireGraph(inp, 'byAdjacency');
    const count = inp.count;
    const members = inp.members;
    checkMemberBounds(members, count, g.n, 'byAdjacency');
    const off = g.csr.off;
    const nbr = g.csr.nbr;

    if (adjMark.length < g.n) {
      adjMark = new Int32Array(g.n);
      adjPos = new Uint32Array(g.n);
      adjGen = 0; // fresh zero-filled mark array: all previous generations are invalid anyway
    }
    const gen = ++adjGen;
    for (let k = 0; k < count; k++) {
      const m = members[k]!;
      adjMark[m] = gen;
      adjPos[m] = k;
    }

    // Start: highest degree, ties to lowest id.
    let startK = 0;
    let bestDeg = -1;
    let bestId = 0;
    for (let k = 0; k < count; k++) {
      const m = members[k]!;
      const d = off[m + 1]! - off[m]!;
      if (d > bestDeg || (d === bestDeg && m < bestId)) {
        bestDeg = d;
        bestId = m;
        startK = k;
      }
    }

    // BFS; `order[r]` is the position k of the member ranked r.
    const order = new Uint32Array(count);
    const visited = new Uint8Array(count);
    let tail = 0;
    order[tail++] = startK;
    visited[startK] = 1;
    for (let head = 0; head < tail; head++) {
      const m = members[order[head]!]!;
      const e1 = off[m + 1]!;
      for (let e = off[m]!; e < e1; e++) {
        const w = nbr[e]!;
        if (adjMark[w] === gen) {
          const kw = adjPos[w]!;
          if (visited[kw] === 0) {
            visited[kw] = 1;
            order[tail++] = kw;
          }
        }
      }
    }

    // Unreached members: appended by ascending member id (see the heuristic note above).
    if (tail < count) {
      const idx = ascendingIdOrder(members, count);
      for (let r = 0; r < count; r++) {
        const k = idx[r]!;
        if (visited[k] === 0) order[tail++] = k;
      }
    }

    const q = new Float32Array(count);
    for (let r = 0; r < count; r++) q[order[r]!] = (r + 0.5) / count;
    return q;
  },
};

/* ------------------------------------------------------------------ byHash */

/* Decorrelates byHash's stream from everything else drawing on the group salt: the engine
 * reserves salt+100+axis for parameter axes, scatters add their own small constants — a
 * large odd constant keeps this stream out of all their neighbourhoods. */
const HASH_ORDER_SALT = 0x5bf03635;

/** `byHash`: a deterministic shuffle — sort key `hash(member, salt')` per member, where
 *  `salt'` folds the group salt with `params.shuffle` (default 0; bump it to deal a
 *  different hand without touching anything else). Hash collisions — and the degenerate
 *  all-equal case — break by member id. Same group, same salt, same shuffle: the same
 *  permutation, forever (§10.1); sibling groups decorrelate through their salts. */
const byHash: Ordering = {
  key: 'byHash',
  assign(inp, params): Float32Array {
    const count = inp.count;
    const members = inp.members;
    const salt = (inp.salt + HASH_ORDER_SALT + (params['shuffle'] ?? 0)) | 0;

    const keys = new Float64Array(count);
    for (let k = 0; k < count; k++) keys[k] = hash(members[k]!, salt);

    const idx = new Uint32Array(count);
    for (let k = 0; k < count; k++) idx[k] = k;
    idx.sort((x, y) => {
      const a = keys[x]!;
      const b = keys[y]!;
      if (a < b) return -1;
      if (a > b) return 1;
      return members[x]! - members[y]!;
    });

    const q = new Float32Array(count);
    for (let r = 0; r < count; r++) q[idx[r]!] = (r + 0.5) / count;
    return q;
  },
};

registerOrdering(byFacet);
registerOrdering(byColumn);
registerOrdering(byAdjacency);
registerOrdering(byHash);

export { byFacet, byColumn, byAdjacency, byHash };
