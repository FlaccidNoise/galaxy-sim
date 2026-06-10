/**
 * Barnes-Hut tree-build worker.
 *
 * Each build: Morton codes → radix sort → gather sorted snapshot → top-down
 * bucketed tree. Returns the tree plus the permutation; the main thread
 * applies the permutation on the GPU, then uploads the tree, so GPU particle
 * order and the tree's leaf ranges stay consistent.
 *
 * The worker's `masses` array always tracks CURRENT GPU ordering (it is
 * re-gathered through every permutation it emits). `centralIdx` follows the
 * central-mass particle through the reorderings.
 *
 * Protocol:
 *   main → worker: { type: 'config', N, masses (current GPU order), centralIdx }
 *   main → worker: { type: 'centralMass', value }
 *   main → worker: { type: 'build', positions: Float32Array (vec4, transferred) }
 *   worker → main: { type: 'tree', tree: ArrayBuffer, numNodes, perm: Uint32Array,
 *                    centralIdx, positions: ArrayBuffer (returned for reuse) }
 */
import { MortonTree, computeMortonCodes, radixSortIndices } from './mortonTree.js';

let tree = null;
let masses = null;
let N = 0;
let centralIdx = 0;

// Scratch (allocated on config)
let codes, indices, scratch, counts, sortedPos, sortedMass, sortedCodes;

self.onmessage = (e) => {
  const msg = e.data;

  if (msg.type === 'config') {
    N = msg.N;
    masses = msg.masses;
    centralIdx = msg.centralIdx ?? 0;
    tree = new MortonTree(N);
    codes = new Uint32Array(N);
    indices = new Uint32Array(N);
    scratch = new Uint32Array(N);
    counts = new Uint32Array(1024);
    sortedPos = new Float32Array(N * 3);
    sortedMass = new Float32Array(N);
    sortedCodes = new Uint32Array(N);
    return;
  }

  if (msg.type === 'centralMass') {
    if (masses) masses[centralIdx] = msg.value;
    return;
  }

  if (msg.type === 'build') {
    const pos = msg.positions;   // vec4 stride, current GPU order

    const rootSize = computeMortonCodes(pos, N, 4, codes);
    const perm = radixSortIndices(codes, indices, scratch, counts, N);

    let newCentral = centralIdx;
    for (let i = 0; i < N; i++) {
      const s = perm[i];
      if (s === centralIdx) newCentral = i;
      sortedPos[i * 3]     = pos[s * 4];
      sortedPos[i * 3 + 1] = pos[s * 4 + 1];
      sortedPos[i * 3 + 2] = pos[s * 4 + 2];
      sortedMass[i] = masses[s];
      sortedCodes[i] = codes[s];
    }
    centralIdx = newCentral;
    const t = masses; masses = sortedMass; sortedMass = t;

    tree.build(sortedCodes, sortedPos, masses, N, rootSize);

    const out = tree._buf.slice(0, tree.byteLength);
    const permCopy = new Uint32Array(perm);
    self.postMessage(
      { type: 'tree', tree: out, numNodes: tree.numNodes, perm: permCopy,
        levelRanges: tree.levelRanges, centralIdx, positions: pos.buffer },
      [out, permCopy.buffer, pos.buffer],
    );
  }
};
