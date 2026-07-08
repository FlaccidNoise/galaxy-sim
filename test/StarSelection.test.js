import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickCandidates, remapIndex } from '../src/ui/StarSelection.js';

// Identity viewProj: clip == position, w = 1 → NDC == xy
const I = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

test('pickCandidates finds the nearest projected star', () => {
  const pos = new Float32Array([
    0, 0, 0, 1,        // → screen (50, 50) in a 100×100 viewport
    0.5, 0, 0, 1,      // → (75, 50)
    -0.9, 0.9, 0, 1,   // → (5, 5)
  ]);
  const hits = pickCandidates(pos, 3, I, 75, 50, 100, 100);
  assert.equal(hits[0].idx, 1);

  const all = pickCandidates(pos, 3, I, 50, 50, 100, 100, 200, 4);
  assert.equal(all.length, 3);
  assert.equal(all[0].idx, 0);
});

test('pickCandidates skips behind-camera stars and respects maxPx', () => {
  const m = new Float32Array(I);
  m[3] = 1;   // clip.w = x + 1 → x = −2 gives w = −1 (behind camera)
  const pos = new Float32Array([
    -2, 0, 0, 1,
    0, 0, 0, 1,
  ]);
  const hits = pickCandidates(pos, 2, m, 50, 50, 100, 100);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].idx, 1);
  assert.equal(pickCandidates(pos, 2, m, 99, 99, 100, 100).length, 0);
});

test('pickCandidates caps results at k', () => {
  const pos = new Float32Array(40);       // 10 particles all at NDC origin
  for (let i = 0; i < 10; i++) pos[i * 4 + 3] = 1;
  assert.equal(pickCandidates(pos, 10, I, 50, 50, 100, 100, 15, 4).length, 4);
});

test('remapIndex inverts a gather permutation', () => {
  const perm = new Uint32Array([2, 0, 1]);   // out[0]=in[2], out[1]=in[0], out[2]=in[1]
  assert.equal(remapIndex(perm, 2), 0);
  assert.equal(remapIndex(perm, 0), 1);
  assert.equal(remapIndex(perm, 1), 2);
  assert.equal(remapIndex(perm, 7), -1);
});
