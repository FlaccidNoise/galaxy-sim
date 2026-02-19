/**
 * CPU-side Barnes-Hut octree.
 *
 * Builds from particle positions each frame, then the flattened buffer is
 * uploaded to the GPU via NBodyGPU.uploadBHTree().
 *
 * Flat node layout — 16 words (64 bytes) per node:
 *   f32[0..2] : centre of mass (x, y, z)
 *   f32[3]    : total mass
 *   f32[4]    : node size (full width of octant)
 *   f32[5]    : isLeaf  (1.0 = leaf, 0.0 = internal)
 *   f32[6..7] : padding
 *   i32[8..15]: children[0..7], −1 = empty slot
 *
 * Root node is always at index 0.
 */

const WORDS = 16;   // words per node
const BYTES = 64;   // bytes per node

export class BHTree {
  constructor(maxNodes) {
    this.maxNodes = maxNodes;
    this._buf = new ArrayBuffer(maxNodes * BYTES);
    this.f32  = new Float32Array(this._buf);
    this.i32  = new Int32Array(this._buf);

    // Split-centre arrays (geometric centre for octant assignment).
    // Kept separate from the COM stored in the GPU buffer.
    this._scx = new Float32Array(maxNodes);
    this._scy = new Float32Array(maxNodes);
    this._scz = new Float32Array(maxNodes);

    this.numNodes = 0;
  }

  /**
   * Build the octree from a stride-3 positions array (x,y,z per particle).
   * @param {Float32Array} positions
   * @param {number}       N
   * @param {number}       centralMass  Mass of particle 0 (default 1.0; all others = 1.0)
   */
  build(positions, N, centralMass = 1.0) {
    this.numNodes = 0;
    const f = this.f32, i32 = this.i32;

    // ── bounding box ─────────────────────────────────────────────────
    let minX=Infinity,  minY=Infinity,  minZ=Infinity;
    let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
    for (let i = 0; i < N; i++) {
      const x=positions[i*3], y=positions[i*3+1], z=positions[i*3+2];
      if (x < minX) minX=x; if (x > maxX) maxX=x;
      if (y < minY) minY=y; if (y > maxY) maxY=y;
      if (z < minZ) minZ=z; if (z > maxZ) maxZ=z;
    }
    const cx   = (minX + maxX) * 0.5;
    const cy   = (minY + maxY) * 0.5;
    const cz   = (minZ + maxZ) * 0.5;
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 1.05 + 1e-6;

    // ── allocate root (index 0) ───────────────────────────────────────
    this.numNodes = 1;
    f[0]=0; f[1]=0; f[2]=0; f[3]=0;        // COM=0, mass=0
    f[4]=size; f[5]=1.0; f[6]=0; f[7]=0;   // size, isLeaf=true
    for (let c = 0; c < 8; c++) i32[8+c] = -1;
    this._scx[0]=cx; this._scy[0]=cy; this._scz[0]=cz;

    // ── insert all particles ──────────────────────────────────────────
    for (let p = 0; p < N; p++) {
      const m = (p === 0) ? centralMass : 1.0;
      this._insert(0, positions[p*3], positions[p*3+1], positions[p*3+2], m, 0);
    }
  }

  _insert(idx, px, py, pz, mass, depth) {
    // Safety valve: coincident particles would recurse forever.
    if (depth > 60) return;

    const f = this.f32, i32 = this.i32;
    const b = idx * WORDS;
    const oldMass = f[b+3];
    const newMass = oldMass + mass;

    if (oldMass === 0.0) {
      // Empty leaf — place particle directly.
      f[b]=px; f[b+1]=py; f[b+2]=pz; f[b+3]=mass;
      f[b+5] = 1.0;  // isLeaf
      return;
    }

    // Save old COM *before* updating (for leaves, old COM == stored particle pos,
    // and old mass == that particle's mass — used when re-inserting after subdivision).
    const oldCX = f[b], oldCY = f[b+1], oldCZ = f[b+2];

    // Update COM and mass incrementally (mass-weighted).
    f[b]   = (oldCX * oldMass + px * mass) / newMass;
    f[b+1] = (oldCY * oldMass + py * mass) / newMass;
    f[b+2] = (oldCZ * oldMass + pz * mass) / newMass;
    f[b+3] = newMass;

    if (f[b+5] > 0.5) {
      // Was a single-particle leaf — convert to internal and re-insert old particle.
      // oldMass here equals exactly the old particle's mass (leaf invariant).
      f[b+5] = 0.0;
      this._insertIntoChild(idx, oldCX, oldCY, oldCZ, oldMass, depth);
    }

    // Insert new particle into appropriate child.
    this._insertIntoChild(idx, px, py, pz, mass, depth);
  }

  _insertIntoChild(parentIdx, px, py, pz, mass, depth) {
    const f = this.f32, i32 = this.i32;
    const pb  = parentIdx * WORDS;
    const cx  = this._scx[parentIdx];
    const cy  = this._scy[parentIdx];
    const cz  = this._scz[parentIdx];
    const oct = ((px > cx) ? 1 : 0) | ((py > cy) ? 2 : 0) | ((pz > cz) ? 4 : 0);

    let childIdx = i32[pb + 8 + oct];

    if (childIdx === -1) {
      // Allocate a new leaf child.
      childIdx = this.numNodes++;
      i32[pb + 8 + oct] = childIdx;

      const cb        = childIdx * WORDS;
      const childSize = f[pb+4] * 0.5;
      const q         = childSize * 0.5;

      f[cb]=0; f[cb+1]=0; f[cb+2]=0; f[cb+3]=0;
      f[cb+4]=childSize; f[cb+5]=1.0; f[cb+6]=0; f[cb+7]=0;
      for (let c = 0; c < 8; c++) i32[cb+8+c] = -1;

      this._scx[childIdx] = cx + ((oct & 1) ? q : -q);
      this._scy[childIdx] = cy + ((oct & 2) ? q : -q);
      this._scz[childIdx] = cz + ((oct & 4) ? q : -q);
    }

    this._insert(childIdx, px, py, pz, mass, depth + 1);
  }

  /** Byte length of the valid portion of the buffer (0..numNodes-1). */
  get byteLength() { return this.numNodes * BYTES; }
}
