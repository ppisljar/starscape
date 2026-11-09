/* FNV-1a (32-bit) over a typed array's bytes — the golden-test fingerprint (DESIGN.md
 * §10.9: fixed spec + fixed corpus → byte-identical buffers, pinned by FNV goldens).
 *
 * THE one shared copy, importable by tests and tools alike; do not re-inline it. Hashing
 * raw bytes means the digest of a Float32Array is platform-endian — every platform this
 * library targets is little-endian, and the goldens are pinned to that.
 */

/** FNV-1a over the bytes of any typed array view (hashes `byteLength` bytes starting at
 *  `byteOffset` — exactly the view, not its whole backing buffer). */
export function fnv1a(view: ArrayBufferView): number {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
