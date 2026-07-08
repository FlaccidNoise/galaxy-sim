// Star picking: pure projection scoring + a tiny stellar-record readback.

/**
 * Project every particle in the CPU position snapshot (vec4/particle) and
 * return the k best candidates within maxPx of the click, sorted best-first.
 * O(N), click-time only. viewProj is column-major.
 */
export function pickCandidates(positions, N, viewProj, clickX, clickY, width, height, maxPx = 15, k = 4) {
  const m = viewProj;
  const best = [];
  for (let i = 0; i < N; i++) {
    const x = positions[i * 4], y = positions[i * 4 + 1], z = positions[i * 4 + 2];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (cw <= 0) continue;   // behind camera
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const sx = (cx / cw * 0.5 + 0.5) * width;
    const sy = (-cy / cw * 0.5 + 0.5) * height;
    const d = Math.hypot(sx - clickX, sy - clickY);
    if (d > maxPx) continue;
    let j = best.length;
    while (j > 0 && best[j - 1].distPx > d) j--;
    best.splice(j, 0, { idx: i, distPx: d });
    if (best.length > k) best.pop();
  }
  return best;
}

/** Read the 16-byte stellar records for the given particle indices. */
export async function readStellarRecords(device, stellarBuffer, indices) {
  const staging = device.createBuffer({
    size: Math.max(indices.length * 16, 16),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  indices.forEach((idx, i) => enc.copyBufferToBuffer(stellarBuffer, idx * 16, staging, i * 16, 16));
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}
