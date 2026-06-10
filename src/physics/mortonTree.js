/**
 * Morton-order Barnes-Hut machinery.
 *
 * Particles are kept sorted along a 30-bit Z-curve (10 bits/axis over a cubic
 * bounding box). Sorted order means every octree node covers a CONTIGUOUS
 * range of the particle array, so:
 *   - the tree builds top-down in O(N · depth) with no insertion churn
 *   - leaves hold particle buckets as [first, count) ranges
 *   - the GPU kernel resolves near-field leaves by iterating their members
 *     with coherent, sequential memory access
 *
 * Flat node layout — 16 words (64 bytes) per node:
 *   f32[0..2] : centre of mass (x, y, z)
 *   f32[3]    : total mass
 *   f32[4]    : node size (full width)
 *   i32[5]    : isLeaf  (1 = leaf, 0 = internal) — REAL int, not a float:
 *               f32 bit-pattern smuggling gets denormal-flushed on some GPUs
 *   i32[6]    : leaf: first particle index (sorted order); internal: unused
 *   i32[7]    : leaf: particle count; internal: unused
 *   i32[8..15]: children[0..7], −1 = empty slot
 */

const WORDS = 16;
const BYTES = 64;
const BUCKET = 16;       // max particles per leaf (forced larger at MAX_LEVEL)
const MAX_LEVEL = 10;    // Morton resolution: 10 bits per axis

/** Spread the low 10 bits of v so there are 2 zero bits between each. */
function part1by2(v) {
  v &= 0x3ff;
  v = (v | (v << 16)) & 0x30000ff;
  v = (v | (v << 8))  & 0x300f00f;
  v = (v | (v << 4))  & 0x30c30c3;
  v = (v | (v << 2))  & 0x9249249;
  return v;
}

/**
 * Compute Morton codes over a CUBIC bounding box (uniform scale keeps the
 * opening criterion's cube-size geometry honest).
 * @returns {number} root cube size
 */
export function computeMortonCodes(pos, N, stride, codes) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = pos[i * stride], y = pos[i * stride + 1], z = pos[i * stride + 2];
    if (!Number.isFinite(x + y + z)) continue;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 1.0001 + 1e-9;
  const s = 1024 / size;
  for (let i = 0; i < N; i++) {
    const x = Math.min((((pos[i * stride]     - minX) * s) | 0), 1023);
    const y = Math.min((((pos[i * stride + 1] - minY) * s) | 0), 1023);
    const z = Math.min((((pos[i * stride + 2] - minZ) * s) | 0), 1023);
    codes[i] = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
  }
  return size;
}

/**
 * Radix-sort particle indices by Morton code: 3 passes × 10 bits.
 * @returns {Uint32Array} the sorted index array (one of indices/scratch)
 */
export function radixSortIndices(codes, indices, scratch, counts, N) {
  for (let i = 0; i < N; i++) indices[i] = i;
  let src = indices, dst = scratch;
  for (let pass = 0; pass < 3; pass++) {
    const shift = pass * 10;
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[(codes[src[i]] >>> shift) & 1023]++;
    let sum = 0;
    for (let b = 0; b < 1024; b++) { const c = counts[b]; counts[b] = sum; sum += c; }
    for (let i = 0; i < N; i++) {
      const v = src[i];
      dst[counts[(codes[v] >>> shift) & 1023]++] = v;
    }
    const t = src; src = dst; dst = t;
  }
  return src;
}

export class MortonTree {
  constructor(maxParticles) {
    this.maxNodes = maxParticles * 2 + 4096;
    this._buf = new ArrayBuffer(this.maxNodes * BYTES);
    this.f32 = new Float32Array(this._buf);
    this.i32 = new Int32Array(this._buf);
    this.numNodes = 0;
  }

  /**
   * Build from Morton-SORTED particle data — BREADTH-FIRST, so each tree
   * level occupies a contiguous node-index range (`levelRanges`). The GPU
   * refreshes COMs level-by-level every physics step; the level ranges are
   * what make that a handful of small dispatches.
   *
   * @param {Uint32Array}  codes   sorted Morton codes
   * @param {Float32Array} pos     sorted positions, stride-3
   * @param {Float32Array} masses  sorted masses
   * @param {number}       N
   * @param {number}       rootSize  cube width from computeMortonCodes
   */
  build(codes, pos, masses, N, rootSize) {
    const f = this.f32, i32 = this.i32;
    this.numNodes = 0;
    this.levelRanges = [];   // [start, count] per depth level

    let queue = [{ lo: 0, hi: N, parentSlot: -1 }];
    let depth = 0;
    let size = rootSize;

    while (queue.length) {
      const next = [];
      const levelStart = this.numNodes;
      for (const item of queue) {
        const idx = this.numNodes++;
        if (item.parentSlot >= 0) i32[item.parentSlot] = idx;
        const b = idx * WORDS;
        const count = item.hi - item.lo;
        f[b + 4] = size;

        const makeLeaf = count <= BUCKET || depth >= MAX_LEVEL
          || this.numNodes + 8 > this.maxNodes;

        if (makeLeaf) {
          let m = 0, cx = 0, cy = 0, cz = 0;
          for (let p = item.lo; p < item.hi; p++) {
            const pm = masses[p];
            m += pm;
            cx += pos[p * 3] * pm; cy += pos[p * 3 + 1] * pm; cz += pos[p * 3 + 2] * pm;
          }
          const inv = m > 0 ? 1 / m : 0;
          f[b] = cx * inv; f[b + 1] = cy * inv; f[b + 2] = cz * inv; f[b + 3] = m;
          i32[b + 5] = 1;
          i32[b + 6] = item.lo; i32[b + 7] = count;
          for (let c = 0; c < 8; c++) i32[b + 8 + c] = -1;
        } else {
          i32[b + 5] = 0;
          i32[b + 6] = 0; i32[b + 7] = 0;
          const shift = 27 - 3 * depth;
          let start = item.lo;
          for (let k = 0; k < 8; k++) {
            let end = start;
            while (end < item.hi && ((codes[end] >>> shift) & 7) === k) end++;
            if (end > start) {
              next.push({ lo: start, hi: end, parentSlot: b + 8 + k });
            } else {
              i32[b + 8 + k] = -1;
            }
            start = end;
          }
        }
      }
      this.levelRanges.push([levelStart, this.numNodes - levelStart]);
      queue = next;
      depth++;
      size *= 0.5;
    }

    // Bottom-up COM pass for internal nodes. BFS guarantees parent < child,
    // so a reverse index sweep sees children before parents.
    for (let n = this.numNodes - 1; n >= 0; n--) {
      const b = n * WORDS;
      if (i32[b + 5] === 1) continue;
      let m = 0, cx = 0, cy = 0, cz = 0;
      for (let c = 0; c < 8; c++) {
        const ch = i32[b + 8 + c];
        if (ch >= 0) {
          const cb = ch * WORDS;
          const cm = f[cb + 3];
          m += cm;
          cx += f[cb] * cm; cy += f[cb + 1] * cm; cz += f[cb + 2] * cm;
        }
      }
      const inv = m > 0 ? 1 / m : 0;
      f[b] = cx * inv; f[b + 1] = cy * inv; f[b + 2] = cz * inv; f[b + 3] = m;
    }
  }

  get byteLength() { return this.numNodes * BYTES; }
}

/**
 * One-shot helper for bootstrap: Morton-sort full 8-float-stride particle
 * data on the CPU and build the matching tree.
 * @returns {{ sorted: Float32Array, tree: MortonTree, perm: Uint32Array, centralIdx: number }}
 */
export function sortAndBuild(particleData, N) {
  const codes = new Uint32Array(N);
  const indices = new Uint32Array(N);
  const scratch = new Uint32Array(N);
  const counts = new Uint32Array(1024);

  const rootSize = computeMortonCodes(particleData, N, 8, codes);
  const perm = radixSortIndices(codes, indices, scratch, counts, N);

  const sorted = new Float32Array(N * 8);
  const sortedCodes = new Uint32Array(N);
  const sortedPos = new Float32Array(N * 3);
  const sortedMass = new Float32Array(N);
  let centralIdx = 0;
  for (let i = 0; i < N; i++) {
    const s = perm[i];
    if (s === 0) centralIdx = i;
    sorted.set(particleData.subarray(s * 8, s * 8 + 8), i * 8);
    sortedCodes[i] = codes[s];
    sortedPos[i * 3] = particleData[s * 8];
    sortedPos[i * 3 + 1] = particleData[s * 8 + 1];
    sortedPos[i * 3 + 2] = particleData[s * 8 + 2];
    sortedMass[i] = particleData[s * 8 + 3];
  }

  const tree = new MortonTree(N);
  tree.build(sortedCodes, sortedPos, sortedMass, N, rootSize);
  return { sorted, tree, perm: new Uint32Array(perm), centralIdx, masses: sortedMass };
}
