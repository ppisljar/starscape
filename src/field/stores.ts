/* The stage-6 GPU stores (DESIGN.md §4): the shared node-position store both layers pull
 * from, the pivot store the frame chains live in, and the one transform expression —
 *
 *     world(i) = spread( chain( mix(A[i], B[i], t) ) )
 *
 * — that the node material evaluates at `instanceIndex` and the link material at the two
 * node ids baked into its vertices (vertex pulling). This is what kills the predecessor's
 * design flaw (SCALE.md "Rotation re-uploads the link buffer"): there is NO per-frame CPU
 * rewrite of node or link buffers, ever. Per frame, the only upload is the pivot store —
 * O(pivots), hundreds of floats.
 *
 * THE CODE SEAM (task: storage buffers on WebGPU; DataTexture pull when the fallback
 * cannot): `createPositionStore(n, mode)` hides the backend —
 *   - 'storage': two read-only storage buffers (vec4 per node), WebGPU only. The WebGL2
 *     fallback in three r187 routes StorageBufferNode reads through an emulated PBO path
 *     that is gated and compute-oriented, not something to bet the render path on.
 *   - 'texture': two RGBA32F DataTextures read with textureLoad (texelFetch) in the
 *     vertex stage — core WebGL2, and equally valid on WebGPU.
 * Both expose the same two fetches; the material code upstream cannot tell them apart.
 * The caller picks 'storage' only when the renderer actually booted the WebGPU backend.
 *
 * Texel/element layout, both modes: slot i = (A.x, A.y, A.z, chainSlot) and
 * (B.x, B.y, B.z, 0). The chain slot rides in A.w so a node costs the links exactly two
 * fetches for position + chain together. chainSlot = pivot id + 1; 0 is the identity.
 *
 * THE PIVOT STORE IS ALWAYS A DATATEXTURE — the "uniform array vs texture" threshold
 * (DESIGN §13) decided from the real limits: WebGL2 guarantees only 256 vec4 vertex
 * uniforms IN TOTAL (GLES3 MAX_VERTEX_UNIFORM_VECTORS minimum), two vec4 slots per pivot
 * would cap a safe uniform-array path near ~100 pivots, and the demo's two-level
 * type>system hierarchy alone compiles ~600. So the threshold is zero: one texture path,
 * always exercised, instead of a uniform path that never safely fits plus a texture twin
 * (two code paths is two chances to disagree). The per-tick re-upload is O(pivots) in
 * transfer as well as in CPU: the texture image is sized to the INSTALLED pivot count
 * (1024-slot row-pair granularity — ~600 pivots is one 32 KB pair), never to capacity,
 * and re-sent only on ticks where something turned.
 *
 * House rules: no per-node JS objects (typed slabs only), loops not spreads, loud errors
 * on capacity overflow (§10.5 spirit — silence would render garbage geometry).
 */

import { DataTexture, FloatType, NearestFilter, RGBAFormat, StorageBufferAttribute, Vector3 } from 'three/webgpu';
import type { Node } from 'three/webgpu';
import { cross, int, ivec2, mix, storage, textureLoad, uniform } from 'three/tsl';
import type { FrameSet } from '../anim/frames';
import { packPivotUniforms } from './morph';

/* ---- position store ------------------------------------------------------------------ */

/** Texture-mode width (texels). 2048 is the WebGL2 MAX_TEXTURE_SIZE floor, so the layout
 *  works on the weakest conformant GL; height = ceil(n / 2048) caps n at ~4.2 M nodes per
 *  store — past the §9 scale target, and loud below. Power of two so the shader's
 *  index→texel split is a mask and a shift. */
export const POS_TEX_WIDTH = 2048;
const POS_TEX_SHIFT = 11;

export interface PositionStore {
  readonly mode: 'storage' | 'texture';
  /** TSL: endpoint set A of node `index` — xyz the position, w the chain slot. */
  fetchA(index: Node<'int'>): Node<'vec4'>;
  /** TSL: endpoint set B of node `index` (xyz; w unused). */
  fetchB(index: Node<'int'>): Node<'vec4'>;
  /** Re-upload both endpoint sets + chain ids (id −1 → slot 0). O(n), and called ONLY on
   *  retarget events (setTarget / setHierarchy / setProjection) — never per frame. */
  upload(baseA: Float32Array, baseB: Float32Array, chainIds: Int32Array, n: number): void;
  dispose(): void;
}

/** Build the shared position store for `n` nodes. See the header for what `mode` means;
 *  'texture' is the safe default, 'storage' is for a renderer that actually runs WebGPU. */
export function createPositionStore(n: number, mode: 'storage' | 'texture'): PositionStore {
  if (mode === 'storage') {
    const a = new Float32Array(n * 4);
    const b = new Float32Array(n * 4);
    const attrA = new StorageBufferAttribute(a, 4);
    const attrB = new StorageBufferAttribute(b, 4);
    const nodeA = storage(attrA, 'vec4', n).toReadOnly();
    const nodeB = storage(attrB, 'vec4', n).toReadOnly();
    return {
      mode,
      fetchA: (index) => nodeA.element(index) as unknown as Node<'vec4'>,
      fetchB: (index) => nodeB.element(index) as unknown as Node<'vec4'>,
      upload(baseA, baseB, chainIds, count) {
        packSlots(a, b, baseA, baseB, chainIds, count);
        attrA.needsUpdate = true;
        attrB.needsUpdate = true;
      },
      dispose() {
        /* The attributes belong to no geometry, so no geometry.dispose() can ever reach
           them — BufferAttribute.dispose() exists for exactly this (the renderer tracks
           the 'dispose' event and destroys the GPU buffer). Without it, every
           createField/dispose cycle would strand 32n bytes of storage buffers until the
           whole renderer died (~32 MB per cycle at the §9 10⁶ target). */
        attrA.dispose();
        attrB.dispose();
      },
    };
  }

  const W = POS_TEX_WIDTH;
  const H = Math.max(1, Math.ceil(n / W));
  if (H > W) {
    throw new Error(
      `createPositionStore: n = ${n} exceeds the ${W}×${W} texture layout (~${W * W} nodes) — ` +
        'raise POS_TEX_WIDTH deliberately rather than overflowing it silently',
    );
  }
  const texA = makeDataTexture(W, H);
  const texB = makeDataTexture(W, H);
  const fetch =
    (tex: DataTexture) =>
    (index: Node<'int'>): Node<'vec4'> =>
      textureLoad(
        tex,
        ivec2(index.bitAnd(int(W - 1)), index.shiftRight(int(POS_TEX_SHIFT))),
        0,
      ) as unknown as Node<'vec4'>;
  return {
    mode,
    fetchA: fetch(texA),
    fetchB: fetch(texB),
    upload(baseA, baseB, chainIds, count) {
      packSlots(
        texA.image.data as Float32Array,
        texB.image.data as Float32Array,
        baseA,
        baseB,
        chainIds,
        count,
      );
      texA.needsUpdate = true;
      texB.needsUpdate = true;
    },
    dispose() {
      texA.dispose();
      texB.dispose();
    },
  };
}

function makeDataTexture(w: number, h: number): DataTexture {
  const tex = new DataTexture(new Float32Array(w * h * 4), w, h, RGBAFormat, FloatType);
  tex.minFilter = NearestFilter; // texelFetch path; float32 textures are not filterable
  tex.magFilter = NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/** The one packing loop both modes share: 3n xyz + n chain ids → two vec4 slabs. */
function packSlots(
  a4: Float32Array,
  b4: Float32Array,
  baseA: Float32Array,
  baseB: Float32Array,
  chainIds: Int32Array,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    const d = i * 4;
    const s = i * 3;
    a4[d] = baseA[s]!;
    a4[d + 1] = baseA[s + 1]!;
    a4[d + 2] = baseA[s + 2]!;
    a4[d + 3] = chainIds[i]! + 1; // slot 0 = identity; exact in f32 to 2^24 pivots
    b4[d] = baseB[s]!;
    b4[d + 1] = baseB[s + 1]!;
    b4[d + 2] = baseB[s + 2]!;
    b4[d + 3] = 0;
  }
}

/* ---- pivot store ----------------------------------------------------------------------- */

/** Default pivot-slot capacity (identity slot included) — the loud OVERFLOW ceiling,
 *  not an allocation: the texture is sized to the installed hierarchy (see
 *  createPivotStore), so raising this costs nothing per tick. The demo's two-level
 *  hierarchy compiles ~600 pivots — an order of magnitude of headroom before the loud
 *  error asks for more. */
export const PIVOT_CAPACITY = 4096;
const PIV_TEX_WIDTH = 1024;
const PIV_TEX_SHIFT = 10;

export interface PivotStore {
  readonly capacity: number;
  /** TSL: apply the composed chain transform of chain slot `slot` (float; 0 = identity
   *  texel, so no branch) to the base-space position `p` — the one-quaternion-apply form
   *  of frames.transformPoint (frames.ts rule 1): `q ⊗ p + offset`. */
  apply(p: Node<'vec3'>, slot: Node<'float'>): Node<'vec3'>;
  /** Refold the FrameSet into (quat, offset) texels and re-upload — O(pivots). Called on
   *  hierarchy install and on every tick where `frames.advance` reported motion; `null`
   *  resets to the identity-only store. `world0` is the pivots' compile-time world
   *  centres (PivotTable.world). */
  sync(frames: FrameSet | null, world0: Float32Array | null): void;
  dispose(): void;
}

export function createPivotStore(capacity: number = PIVOT_CAPACITY): PivotStore {
  const W = PIV_TEX_WIDTH;

  /* The texture is sized to the INSTALLED hierarchy (row-pair granularity: one 1024-slot
     row pair per started W slots), NOT to `capacity`: needsUpdate re-sends the whole
     image, and neither the WebGPU backend nor the WebGL2 fallback consumes
     Texture.updateRanges — so a capacity-sized image would make the per-tick transfer
     O(capacity) however few pivots are live (a raised pivotCapacity would then be paid
     on EVERY spinning tick, not on install). Resizes happen on install, an event:
     dispose() drops the old GPU texture and the next use re-creates it at the new size;
     the Texture OBJECT (which the TSL `apply` closure captured) lives on. `capacity`
     stays the loud ceiling. */
  let pairRows = 1;
  const tex = makeDataTexture(W, pairRows * 2); // per slot: quat texel row above, offset row below
  let data = tex.image.data as Float32Array;
  data[3] = 1; // slot 0: identity quaternion (0,0,0,1), zero offset — pre-seeded
  const ensureRows = (rows: number): void => {
    if (rows === pairRows) return;
    pairRows = rows;
    tex.dispose(); // free the GPU-side texture; the object stays bound in the material
    data = new Float32Array(W * rows * 2 * 4);
    data[3] = 1;
    tex.image = { data, width: W, height: rows * 2 };
    tex.needsUpdate = true;
  };

  /* packPivotUniforms writes slot-contiguous quat/offset slabs; the scatter into the
     row-pair texel layout is this store's own business. Allocated once. */
  const quat = new Float32Array(capacity * 4);
  const offs = new Float32Array(capacity * 3);

  return {
    capacity,
    apply(p, slot) {
      const s = int(slot);
      const x = s.bitAnd(int(W - 1));
      const y = s.shiftRight(int(PIV_TEX_SHIFT)).shiftLeft(int(1));
      const q = textureLoad(tex, ivec2(x, y), 0);
      const o = textureLoad(tex, ivec2(x, y.add(int(1))), 0);
      // v' = v + w·t + q×t with t = 2(q×v) — identity slot passes p through EXACTLY
      // (t = 0, so no arithmetic can move it), which is what keeps an unframed field
      // byte-stable through this stage.
      const t2 = cross(q.xyz, p).mul(2.0);
      const rotated = p.add(t2.mul(q.w)).add(cross(q.xyz, t2));
      return rotated.add(o.xyz) as unknown as Node<'vec3'>;
    },
    sync(frames, world0) {
      let slots = 1;
      if (frames !== null && world0 !== null) {
        if (frames.count + 1 > capacity) {
          throw new Error(
            `PivotStore: ${frames.count} pivots exceed capacity ${capacity} — pass a larger ` +
              'capacity to createField (pivotCapacity) rather than truncating the chain silently',
          );
        }
        packPivotUniforms(frames, world0, quat, offs);
        slots = frames.count + 1;
      } else {
        quat[0] = 0;
        quat[1] = 0;
        quat[2] = 0;
        quat[3] = 1;
        offs[0] = 0;
        offs[1] = 0;
        offs[2] = 0;
      }
      ensureRows(Math.max(1, Math.ceil(slots / W))); // size to the live need, on an event
      for (let s = 0; s < slots; s++) {
        const x = s & (W - 1);
        const yPair = s >> PIV_TEX_SHIFT;
        const qi = (yPair * 2 * W + x) * 4;
        const oi = ((yPair * 2 + 1) * W + x) * 4;
        data[qi] = quat[s * 4]!;
        data[qi + 1] = quat[s * 4 + 1]!;
        data[qi + 2] = quat[s * 4 + 2]!;
        data[qi + 3] = quat[s * 4 + 3]!;
        data[oi] = offs[s * 3]!;
        data[oi + 1] = offs[s * 3 + 1]!;
        data[oi + 2] = offs[s * 3 + 2]!;
        data[oi + 3] = 0;
      }
      tex.needsUpdate = true;
    },
    dispose() {
      tex.dispose();
    },
  };
}

/* ---- the composed transform ------------------------------------------------------------ */

/** The §4 pipeline as one TSL expression factory, shared by the node and link materials
 *  so the two layers can never disagree about where a node is. The uniforms are exposed
 *  structurally (`.value` holders) — the Field drives them from the tween system
 *  (“animation = uniforms”); nothing here is per-frame CPU. */
export interface GpuTransform {
  uniforms: {
    /** Morph progress t ∈ [0, 1] — mix(A, B, t). */
    morphT: { value: number };
    /** Uniform scale about the centroid (1 = identity). */
    spread: { value: number };
    /** The spread centre — the field centroid of the CURRENT target (§4). */
    centroid: { value: Vector3 };
  };
  /** Base-space position of node `index`: mix(A, B, t). Links use it for length. */
  baseOf(index: Node<'int'>): Node<'vec3'>;
  /** Chain slot of node `index` (A.w). */
  chainOf(index: Node<'int'>): Node<'float'>;
  /** Fully transformed world position: spread(chain(mix(A, B, t))). */
  worldOf(index: Node<'int'>): Node<'vec3'>;
}

export function createGpuTransform(positions: PositionStore, pivots: PivotStore): GpuTransform {
  const uMorphT = uniform(0);
  const uSpread = uniform(1);
  const uCentroid = uniform(new Vector3(0, 0, 0));

  const baseOf = (index: Node<'int'>): Node<'vec3'> =>
    mix(positions.fetchA(index).xyz, positions.fetchB(index).xyz, uMorphT) as unknown as Node<'vec3'>;

  return {
    uniforms: {
      morphT: uMorphT as unknown as { value: number },
      spread: uSpread as unknown as { value: number },
      centroid: uCentroid as unknown as { value: Vector3 },
    },
    baseOf,
    chainOf: (index) => positions.fetchA(index).w as unknown as Node<'float'>,
    worldOf(index) {
      // ONE fetch of A reused for both xyz and the chain slot (same node object, so the
      // builder emits a single texelFetch/element read, not two).
      const a = positions.fetchA(index);
      const base = mix(a.xyz, positions.fetchB(index).xyz, uMorphT) as unknown as Node<'vec3'>;
      const chained = pivots.apply(base, a.w as unknown as Node<'float'>);
      return uCentroid.add(chained.sub(uCentroid).mul(uSpread)) as unknown as Node<'vec3'>;
    },
  };
}
