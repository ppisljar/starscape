/* Link importance order.
 *
 * The field is fill-rate bound and links are most of the bill: in the predecessor, at 10^5
 * pages they were 55 of the 73 ms in a frame, scaling with pixel count — fragment shading, not
 * geometry. The fix is drawing fewer of them, and "fewer" is only acceptable if the dropped
 * ones are the ones saying the least:
 *
 *   1. intra-group before inter-group. Position already says which group a node is in, so a
 *      chord between two groups re-states what the layout shows; an edge inside a group is the
 *      only thing that says which members belong together.
 *   2. shorter before longer. Long edges are faded by the shader toward invisibility AND cover
 *      the most pixels — cutting them removes cost exactly where it removes nothing to look at.
 *
 * With the buffer in this order, the budget is just `setDrawRange(0, k * 2)`: keep the best k
 * with no per-frame work and no second buffer.
 *
 * Counting sort, not Array.sort: at 4.2M edges a JS comparator costs seconds of boot time,
 * while a 65,536-bucket histogram is two linear passes and ~50 ms. The key packs the
 * intra/inter bit above a 15-bit quantised length, so one pass orders by both.
 */

/** Reorder `links` (2m node-index pairs) by descending worth. `group` is a categorical
 *  assignment per node (the predecessor's "community"), or null to rank by length alone.
 *  `pos` is the 3n frozen layout the lengths are measured in. Pure and deterministic;
 *  returns the input array untouched when there is nothing to order (m < 2).
 *
 *  NOTE: order links BEFORE building the CSR — the CSR stores edge indices, so permuting
 *  afterwards would point every `csr.edge[k]` at somebody else's link. */
export function orderLinks(
  n: number,
  links: Uint32Array,
  group: ArrayLike<number> | null,
  pos: Float32Array,
): Uint32Array {
  void n; // part of the shared signature; the ordering needs only links/group/pos
  const m = links.length >> 1;
  if (m < 2) return links;
  const BITS = 15;
  const SPAN = (1 << BITS) - 1;

  // one pass for the length scale, so the quantisation uses the range the data actually has
  let maxL = 0;
  const len = new Float32Array(m);
  for (let e = 0; e < m; e++) {
    const a = links[e * 2]! * 3;
    const b = links[e * 2 + 1]! * 3;
    const L = Math.hypot(pos[a]! - pos[b]!, pos[a + 1]! - pos[b + 1]!, pos[a + 2]! - pos[b + 2]!);
    len[e] = L;
    if (L > maxL) maxL = L;
  }
  const q = maxL > 0 ? SPAN / maxL : 0;

  const key = new Uint16Array(m);
  for (let e = 0; e < m; e++) {
    const a = links[e * 2]!;
    const b = links[e * 2 + 1]!;
    const inter = group && group[a] !== group[b] ? 1 : 0;
    // 15 bits of length under 1 bit of inter-ness: intra sorts entirely before inter, and
    // within each half, short before long. The SPAN>>1 clamp folds the long tail into one
    // bucket — past half the max length, finer ranking buys nothing.
    key[e] = (inter << BITS) | Math.min(SPAN >> 1, (len[e]! * q) | 0);
  }

  const B = 1 << 16;
  const count = new Uint32Array(B + 1);
  for (let e = 0; e < m; e++) {
    const k = key[e]! + 1;
    count[k] = count[k]! + 1;
  }
  for (let i = 0; i < B; i++) count[i + 1]! += count[i]!;
  const out = new Uint32Array(m * 2);
  for (let e = 0; e < m; e++) {
    const k = key[e]!;
    const slot = count[k]!;
    count[k] = slot + 1;
    out[slot * 2] = links[e * 2]!;
    out[slot * 2 + 1] = links[e * 2 + 1]!;
  }
  return out;
}
