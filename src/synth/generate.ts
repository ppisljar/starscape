/* Synthetic corpora on tap, 10² … 10⁶ nodes — the library's test fuel and demo data.
 *
 * Ported in spirit from the predecessor's `AI/tools/synth-build.py`, which earned its keep in
 * SCALE.md: what made those corpora *useful* was (a) hierarchical clustering — galaxies of
 * systems, ~72% of links inside a system — so budgets and link ordering have real structure to
 * bite on, and (b) several INDEPENDENT categorical facets. With one facet every visual channel
 * is driven by the same thing and you cannot tell a working channel from a coincidence; with
 * four, a channel that looks structured when it should not is a bug you can see.
 *
 * Everything is a pure function of (spec.n, spec.seed) through hash/gauss streams — no
 * Math.random, no clock (DESIGN.md §10.1) — so two calls with the same spec are byte-identical
 * on every machine, forever. Golden tests and cross-run screenshots depend on that.
 *
 * Scale rule (§10.2–3): only typed arrays are allocated per node; the only strings live in the
 * facet value dictionaries, and labels are a single lazy closure. generateSynth(10⁶) is a few
 * linear passes and completes in seconds.
 *
 * THE UNIVERSE SCALES WITH n (defaults only — explicit SynthSpec counts still override).
 * The first cut used a fixed ~2000-unit field, 12 fixed galaxies and fixed sigmas, so node
 * density grew LINEARLY with n: 10⁶ nodes was not a bigger universe, just a denser one.
 * Defaults now target constant spatial density, derived bottom-up the way
 * src/place/hierarchy.ts sizes groups (radius = spacing × count^⅓, floored):
 *
 *   galaxies         = clamp(round(n^⅓), 8, 256)
 *   systemsPerGalaxy = clamp(round(n^⅓), 4, 256)
 *   nodesPerSystem   = n / (galaxies × systemsPerGalaxy)
 *   systemRadius     = max(2·SPACING, SPACING · nodesPerSystem^⅓)
 *   galaxyRadius     = max(2·systemRadius, PACK · systemRadius · systemsPerGalaxy^⅓)
 *   universeRadius   = max(2·galaxyRadius, PACK · galaxyRadius · galaxies^⅓)
 *
 * A balanced three-level tree, each level ~n^⅓: 10⁶ ≈ 100×100×100, 10⁴ ≈ 22×22×21. When no
 * clamp or floor binds, the per-level cube roots telescope: universeRadius =
 * SPACING · PACK² · n^⅓ exactly, independent of how rounding split the levels — so the
 * field's diagonal grows as n^⅓ and n/diagonal³ holds constant across 10³…10⁶ (both pinned
 * in test/synth.test.ts). Sigmas are FRACTIONS of their level's radius, never absolute
 * units, so cluster texture is scale-free too.
 */

import { gauss, hash } from '../core/hash';
import type { GraphInput } from '../core/types';

export interface SynthSpec {
  n: number;
  seed: number;
  /** How many top-level clusters ("galaxies"). Default ~n^⅓ clamped to [8, 256] — see
   *  `synthDefaults` and the module header: counts grow with n so density does not. */
  galaxies?: number;
  /** Sub-clusters per galaxy ("systems"). Default ~n^⅓ clamped to [4, 256]. */
  systemsPerGalaxy?: number;
  /** Fraction of link draws made inside one system. Default 0.72 — the predecessor's measured
   *  mix, which SCALE.md notes is *more* clustered than a real vault (the honest direction for
   *  a load test: a flatter mix overdraws more, not less). */
  intraLinkFraction?: number;
}

/* The type-like facet mirrors the real vault's seven layers so a synthetic run and a real one
 * are visually comparable; `era` is the deliberately-coarse independent facet (few values, good
 * for testing a channel that must show large blocks). */
const TYPE_VALUES = ['source', 'note', 'person', 'system', 'synthesis', 'project', 'other'];
const ERA_VALUES = ['2023', '2024', '2025', '2026', '2027'];

/* Scale constants (module header: the constant-density derivation). SPACING is the
 * characteristic node spacing — the vault's ~192-unit page spacing — the ONE density
 * constant everything else multiplies; PACK keeps sibling clusters from overlapping badly
 * when a level's children are packed into their parent. Sigmas are fractions of their
 * level's radius (never absolute units, which is what made the old field densify). */
const SPACING = 192;
const PACK = 1.6;
const GALAXIES_MIN = 8;
const GALAXIES_MAX = 256;
const SYSTEMS_MIN = 4;
const SYSTEMS_MAX = 256;
const NODE_SIGMA_FRAC = 0.5; // of systemRadius — ~95% of a system's nodes inside its radius
const SYSTEM_SIGMA_FRAC = 0.4; // of galaxyRadius
const LINKS_PER_NODE = 4;
const GOLDEN_ANGLE = 2.399963229728653;

function clampInt(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Default cluster counts for an n-node corpus: a balanced three-level tree, each level
 *  ~n^⅓, clamped (module header). Exported so tests and consumers derive the structure
 *  from the same formula the generator uses instead of hardcoding yesterday's defaults. */
export function synthDefaults(n: number): { galaxies: number; systemsPerGalaxy: number } {
  const side = Math.round(Math.cbrt(Math.max(n, 1)));
  return {
    galaxies: clampInt(side, GALAXIES_MIN, GALAXIES_MAX),
    systemsPerGalaxy: clampInt(side, SYSTEMS_MIN, SYSTEMS_MAX),
  };
}

/* Independent hash streams, all derived from spec.seed. Each constant below is a stream id;
 * mixing it with the seed through two odd multipliers gives decorrelated salts (same trick as
 * hash() itself — different salts are independent streams by construction). */
const S_TYPE = 1;
const S_GALAXY = 2;
const S_SYSTEM = 3;
const S_ERA = 4;
const S_POS_X = 5;
const S_POS_Y = 6;
const S_POS_Z = 7;
const S_SYS_X = 8;
const S_SYS_Y = 9;
const S_SYS_Z = 10;
const S_WORDS = 11;
const S_LINK_MODE = 12;
const S_LINK_A = 13;
const S_LINK_B = 14;

function streamSalt(seed: number, stream: number): number {
  return (Math.imul(seed | 0, 0x85ebca6b) ^ Math.imul(stream, 0x9e3779b1)) | 0;
}

/** Cumulative weights for a skewed categorical: w_j ∝ 1/(j+1)^0.7. Skewed, not uniform — a
 *  real corpus is 63% one type, and a uniform test hides every bug that only shows when one
 *  group dominates (synth-build.py's reasoning, kept verbatim in spirit). */
function skewedCDF(k: number): Float64Array {
  const cum = new Float64Array(k);
  let total = 0;
  for (let j = 0; j < k; j++) total += 1 / Math.pow(j + 1, 0.7);
  let acc = 0;
  for (let j = 0; j < k; j++) {
    acc += 1 / Math.pow(j + 1, 0.7) / total;
    cum[j] = acc;
  }
  if (k > 0) cum[k - 1] = 1; // float dust must not leave a reachable r above the last bucket
  return cum;
}

/** Index of the first cumulative bucket at or above r. k is small (≤ a few hundred) — a
 *  linear scan beats anything cleverer and allocates nothing. */
function pick(cum: Float64Array, r: number): number {
  const k = cum.length;
  for (let j = 0; j < k - 1; j++) if (r <= cum[j]!) return j;
  return k - 1;
}

/** Smallest unsigned typed array that can index `k` categories. */
function facetStore(n: number, k: number): Uint8Array | Uint16Array | Uint32Array {
  if (k <= 0x100) return new Uint8Array(n);
  if (k <= 0x10000) return new Uint16Array(n);
  return new Uint32Array(n);
}

/** Generate a synthetic corpus: fully deterministic from (spec.n, spec.seed). */
export function generateSynth(spec: SynthSpec): GraphInput {
  const n = spec.n;
  const seed = spec.seed | 0;
  const defaults = synthDefaults(n);
  const galaxies = spec.galaxies ?? defaults.galaxies;
  const perGalaxy = spec.systemsPerGalaxy ?? defaults.systemsPerGalaxy;
  const intra = spec.intraLinkFraction ?? 0.72;
  if (!Number.isInteger(n) || n < 0) throw new Error(`generateSynth: n must be a non-negative integer, got ${n}`);
  if (!Number.isInteger(galaxies) || galaxies < 1) throw new Error(`generateSynth: galaxies must be >= 1, got ${galaxies}`);
  if (!Number.isInteger(perGalaxy) || perGalaxy < 1) throw new Error(`generateSynth: systemsPerGalaxy must be >= 1, got ${perGalaxy}`);
  if (!(intra >= 0 && intra <= 1)) throw new Error(`generateSynth: intraLinkFraction must be in [0,1], got ${intra}`);

  const systems = galaxies * perGalaxy;

  /* ---- radii, bottom-up (the hierarchy radius policy — module header). Derived from the
   * ACTUAL counts, so an explicit galaxies/systemsPerGalaxy override keeps constant density
   * too: fewer clusters simply means bigger ones in a correspondingly bigger universe. */
  const nodesPerSystem = Math.max(1, n / systems);
  const systemRadius = Math.max(2 * SPACING, SPACING * Math.cbrt(nodesPerSystem));
  const galaxyRadius = Math.max(2 * systemRadius, PACK * systemRadius * Math.cbrt(perGalaxy));
  const universeRadius = Math.max(2 * galaxyRadius, PACK * galaxyRadius * Math.cbrt(galaxies));
  const nodeSigma = NODE_SIGMA_FRAC * systemRadius;
  const systemSigma = SYSTEM_SIGMA_FRAC * galaxyRadius;

  /* ---- facets. `type` and `era` are independent draws (their own streams); `galaxy` is its
   * own draw; `system` nests inside galaxy by construction (same node can never be in system
   * 12 of galaxy 3 and galaxy 5 — the hierarchy must be real for intra-links to mean
   * anything). Independence of type/era from the geometry-driving facets is the point. */
  const typeData = facetStore(n, TYPE_VALUES.length);
  const galaxyData = facetStore(n, galaxies);
  const systemData = facetStore(n, systems);
  const eraData = facetStore(n, ERA_VALUES.length);

  const cumType = skewedCDF(TYPE_VALUES.length);
  const cumGalaxy = skewedCDF(galaxies);
  const cumSystem = skewedCDF(perGalaxy);
  const cumEra = skewedCDF(ERA_VALUES.length);

  const saltType = streamSalt(seed, S_TYPE);
  const saltGalaxy = streamSalt(seed, S_GALAXY);
  const saltSystem = streamSalt(seed, S_SYSTEM);
  const saltEra = streamSalt(seed, S_ERA);
  for (let i = 0; i < n; i++) {
    const g = pick(cumGalaxy, hash(i, saltGalaxy));
    typeData[i] = pick(cumType, hash(i, saltType));
    galaxyData[i] = g;
    systemData[i] = g * perGalaxy + pick(cumSystem, hash(i, saltSystem));
    eraData[i] = pick(cumEra, hash(i, saltEra));
  }

  /* ---- centres. Galaxy centres on a Fibonacci sphere — m centres get m directions that are
   * GUARANTEED spread (a hash only gets that on average, and on a small m it routinely does
   * not — ported from arrange.js's spreadNormal rationale). System centres gaussian about
   * their galaxy, nodes gaussian about their system: nested clusters in a universe whose
   * radius grows as n^⅓ (module header), never a fixed-size field. */
  const galaxyCentre = new Float64Array(galaxies * 3);
  for (let g = 0; g < galaxies; g++) {
    const y = galaxies > 1 ? 1 - 2 * ((g + 0.5) / galaxies) : 0;
    const rr = Math.sqrt(Math.max(0, 1 - y * y));
    const th = g * GOLDEN_ANGLE;
    galaxyCentre[g * 3] = Math.cos(th) * rr * universeRadius;
    galaxyCentre[g * 3 + 1] = y * universeRadius;
    galaxyCentre[g * 3 + 2] = Math.sin(th) * rr * universeRadius;
  }
  const systemCentre = new Float64Array(systems * 3);
  const saltSysX = streamSalt(seed, S_SYS_X);
  const saltSysY = streamSalt(seed, S_SYS_Y);
  const saltSysZ = streamSalt(seed, S_SYS_Z);
  for (let s = 0; s < systems; s++) {
    const g = (s / perGalaxy) | 0;
    systemCentre[s * 3] = galaxyCentre[g * 3]! + gauss(s, saltSysX) * systemSigma;
    systemCentre[s * 3 + 1] = galaxyCentre[g * 3 + 1]! + gauss(s, saltSysY) * systemSigma;
    systemCentre[s * 3 + 2] = galaxyCentre[g * 3 + 2]! + gauss(s, saltSysZ) * systemSigma;
  }

  /* ---- positions: one pass, three gaussian draws per node about its system centre. */
  const positions = new Float32Array(n * 3);
  const saltX = streamSalt(seed, S_POS_X);
  const saltY = streamSalt(seed, S_POS_Y);
  const saltZ = streamSalt(seed, S_POS_Z);
  for (let i = 0; i < n; i++) {
    const s3 = systemData[i]! * 3;
    const o = i * 3;
    positions[o] = systemCentre[s3]! + gauss(i, saltX) * nodeSigma;
    positions[o + 1] = systemCentre[s3 + 1]! + gauss(i, saltY) * nodeSigma;
    positions[o + 2] = systemCentre[s3 + 2]! + gauss(i, saltZ) * nodeSigma;
  }

  /* ---- words: a heavy-tailed numeric column (log-normal — median ~120, a long right tail
   * into the tens of thousands), so size/ramp channels see the distribution real corpora
   * have, not a flattering uniform one. */
  const words = new Uint32Array(n);
  const saltWords = streamSalt(seed, S_WORDS);
  for (let i = 0; i < n; i++) {
    const w = Math.round(120 * Math.exp(gauss(i, saltWords) * 1.1));
    words[i] = w < 1 ? 1 : w > 200000 ? 200000 : w;
  }

  /* ---- system membership buckets, CSR-style counting sort: count, prefix-sum, scatter.
   * O(n), typed arrays only — never an array-of-arrays with one JS array per system. */
  const sysCount = new Uint32Array(systems);
  for (let i = 0; i < n; i++) sysCount[systemData[i]!] = sysCount[systemData[i]!]! + 1;
  const sysOff = new Uint32Array(systems + 1);
  for (let s = 0; s < systems; s++) sysOff[s + 1] = sysOff[s]! + sysCount[s]!;
  const bySystem = new Uint32Array(n);
  const cursor = sysOff.slice(0, systems);
  for (let i = 0; i < n; i++) {
    const s = systemData[i]!;
    bySystem[cursor[s]!] = i;
    cursor[s] = cursor[s]! + 1;
  }

  /* ---- links. Each draw either stays inside one system (intraLinkFraction of them) or jumps
   * anywhere — the structure a community detector should find, and the structure the
   * importance-ordered link budget (intra before inter) needs to be testable at all.
   * Intra draws anchor on a uniformly random node and pair it within its own system, which
   * weights systems by their real population. Degenerate draws (self-pair, one-member system)
   * are skipped — a deterministic function of the stream, so the skip pattern itself is
   * reproducible. */
  const target = n * LINKS_PER_NODE;
  const raw = new Uint32Array(target * 2);
  const saltMode = streamSalt(seed, S_LINK_MODE);
  const saltA = streamSalt(seed, S_LINK_A);
  const saltB = streamSalt(seed, S_LINK_B);
  let written = 0;
  for (let e = 0; e < target; e++) {
    const a = (hash(e, saltA) * n) | 0;
    let b: number;
    if (hash(e, saltMode) < intra) {
      const s = systemData[a]!;
      const count = sysCount[s]!;
      if (count < 2) continue;
      b = bySystem[sysOff[s]! + ((hash(e, saltB) * count) | 0)]!;
    } else {
      b = (hash(e, saltB) * n) | 0;
    }
    if (a === b) continue;
    raw[written * 2] = a;
    raw[written * 2 + 1] = b;
    written++;
  }
  const links = raw.slice(0, written * 2);

  return {
    n,
    positions,
    links,
    columns: { words },
    facets: {
      type: { values: TYPE_VALUES.slice(), data: typeData },
      galaxy: { values: nameList('galaxy', galaxies), data: galaxyData },
      system: { values: nameList('system', systems), data: systemData },
      era: { values: ERA_VALUES.slice(), data: eraData },
    },
    // One closure for the whole corpus, called lazily — never 10⁶ eager strings.
    labels: (i: number) => `${TYPE_VALUES[typeData[i]!]} ${i}`,
  };
}

/** `prefix-00` … `prefix-NN`: the facet value dictionary — the one place strings are allowed. */
function nameList(prefix: string, k: number): string[] {
  const out = new Array<string>(k);
  for (let j = 0; j < k; j++) out[j] = `${prefix}-${String(j).padStart(2, '0')}`;
  return out;
}
