/** Request a WebGPU device with the adapter's full buffer limits. */
export async function requestGPUDevice() {
  if (!navigator.gpu) throw new Error('WebGPU not supported in this browser. Use Chrome 113+.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
}

export class NBodyGPU {
  constructor(device, particleData, stellarData) {
    this.device = device;
    this.N = particleData.length / 8;
    this.currentBuffer = 0;  // particle ping-pong index (flips every step)
    this.stellarCurrent = 0; // stellar ping-pong index (flips on permutation only)
    this.ready = false;
    this.initPromise = this._init(particleData, stellarData);
  }

  async _init(particleData, stellarData) {
    const device = this.device;
    const byteSize = particleData.byteLength;

    // Two ping-pong storage buffers
    this.buffers = [
      device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),
      device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),
    ];

    // Upload initial conditions to both buffers
    device.queue.writeBuffer(this.buffers[0], 0, particleData);
    device.queue.writeBuffer(this.buffers[1], 0, particleData);

    // Params uniform — 96 bytes, layout matches struct Params in the shader
    this.paramsBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Per-particle specific energy (written by the force kernel, packed into
    // the position readback for the drift diagnostic)
    this.energyBuffer = device.createBuffer({
      size: this.N * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // BH tree buffer — sized generously (worker pool is N*4+1024 nodes).
    this.maxBHNodes = this.N * 4 + 1024;
    this.bhBuffer = device.createBuffer({
      size: this.maxBHNodes * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Compacted positions (vec4/particle) + readback staging for tree rebuilds
    this.posCompactBuffer = device.createBuffer({
      size: this.N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.posStagingBuffer = device.createBuffer({
      size: this.N * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const url = new URL('./shaders/nbody_bh.wgsl', import.meta.url);
    const code = await (await fetch(url)).text();
    const shader = device.createShaderModule({ code, label: 'nbody_bh.wgsl' });
    const info = await shader.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length) {
      throw new Error('nbody_bh.wgsl: ' + errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }

    // Stellar state ping-pong: [stellarMass, birthTimeMyr, seed, 0] per star.
    // Read by the render shader; only rewritten when particles are reordered.
    this.stellarBuffers = [0, 1].map(() => device.createBuffer({
      size: this.N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    device.queue.writeBuffer(this.stellarBuffers[0], 0, stellarData);
    device.queue.writeBuffer(this.stellarBuffers[1], 0, stellarData);

    // Morton permutation buffer (u32 per particle)
    this.permBuffer = device.createBuffer({
      size: this.N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Bind group layout: particlesIn, particlesOut, params, bhNodes, posOut,
    // perm, energyOut
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });

    this.pipeline = device.createComputePipeline({
      layout, compute: { module: shader, entryPoint: 'main' },
    });
    this.compactPipeline = device.createComputePipeline({
      layout, compute: { module: shader, entryPoint: 'compact_positions' },
    });
    this.permPipeline = device.createComputePipeline({
      layout, compute: { module: shader, entryPoint: 'apply_permutation' },
    });

    // Stellar gather: own layout (stellarIn, stellarOut, params, perm)
    this.stellarPermLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.stellarPermPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.stellarPermLayout] }),
      compute: { module: shader, entryPoint: 'apply_permutation_stellar' },
    });
    this.stellarPermBindGroups = [0, 1].map(i => device.createBindGroup({
      layout: this.stellarPermLayout,
      entries: [
        { binding: 0, resource: { buffer: this.stellarBuffers[i] } },
        { binding: 1, resource: { buffer: this.stellarBuffers[i ^ 1] } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
        { binding: 3, resource: { buffer: this.permBuffer } },
      ],
    }));

    // Star formation / recycling: in-place mutation of particles + stellar.
    // Bind groups for all 4 parity combinations (particle × stellar).
    this.sfLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.sfPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.sfLayout] }),
      compute: { module: shader, entryPoint: 'star_formation' },
    });
    this.sfBindGroups = [0, 1].map(p => [0, 1].map(s => device.createBindGroup({
      layout: this.sfLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers[p] } },
        { binding: 1, resource: { buffer: this.stellarBuffers[s] } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    })));

    // Per-step COM refresh: separate layout (bhBuffer is read_write here;
    // it must not also be bound read-only in the same dispatch scope).
    // Level params use one dynamic-offset uniform (256-byte slots).
    this.refreshUniform = device.createBuffer({
      size: 16 * 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.refreshLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.refreshPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.refreshLayout] }),
      compute: { module: shader, entryPoint: 'refresh_level' },
    });
    this.refreshBindGroups = [0, 1].map(i => device.createBindGroup({
      layout: this.refreshLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers[i] } },
        { binding: 1, resource: { buffer: this.bhBuffer } },
        { binding: 2, resource: { buffer: this.refreshUniform, size: 16 } },
      ],
    }));
    this.levelRanges = null;

    // Two bind groups (one per ping-pong direction: A→B and B→A)
    this.bindGroups = [
      this._makeBindGroup(0, 1),
      this._makeBindGroup(1, 0),
    ];

    this.ready = true;
  }

  _makeBindGroup(inIdx, outIdx) {
    return this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers[inIdx] } },
        { binding: 1, resource: { buffer: this.buffers[outIdx] } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
        { binding: 3, resource: { buffer: this.bhBuffer } },
        { binding: 4, resource: { buffer: this.posCompactBuffer } },
        { binding: 5, resource: { buffer: this.permBuffer } },
        { binding: 6, resource: { buffer: this.energyBuffer } },
      ],
    });
  }

  /**
   * Reorder particles along the worker-supplied Morton permutation.
   * Gathers src→dst into the ping-pong partner, then flips, so the next
   * step (and the freshly built tree, which uses sorted indices) line up.
   */
  applyPermutation(commandEncoder, perm) {
    if (!this.ready) return;
    this.device.queue.writeBuffer(this.permBuffer, 0, perm);
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.permPipeline);
    pass.setBindGroup(0, this.bindGroups[this.currentBuffer]);
    pass.dispatchWorkgroups(Math.ceil(this.N / 256));
    pass.setPipeline(this.stellarPermPipeline);
    pass.setBindGroup(0, this.stellarPermBindGroups[this.stellarCurrent]);
    pass.dispatchWorkgroups(Math.ceil(this.N / 256));
    pass.end();
    this.currentBuffer ^= 1;
    this.stellarCurrent ^= 1;
  }

  /** Stellar-state buffer matching the current particle ordering */
  get stellarBuffer() {
    return this.stellarBuffers[this.stellarCurrent];
  }

  updateCentralMass(mass, idx = 0) {
    // Update the central particle's mass in both ping-pong buffers
    // (mass is the 4th float of the 8-float particle record)
    const buf = new Float32Array([mass]);
    this.device.queue.writeBuffer(this.buffers[0], idx * 32 + 12, buf);
    this.device.queue.writeBuffer(this.buffers[1], idx * 32 + 12, buf);
  }

  updateParams({ dt, G, epsilon, theta = 0.5, haloMass = 0, haloA = 1,
                 gasDrag = 0, time = 0, agingRate = 25, sfDensity = 1e9,
                 sfEff = 0, sfDt = 0, haloCenters = [] }) {
    const arr = new Float32Array(24);
    const u32 = new Uint32Array(arr.buffer);
    arr[0] = dt;
    arr[1] = G;
    arr[2] = epsilon * epsilon;
    u32[3] = this.N;
    arr[4] = theta;
    arr[5] = haloMass;
    arr[6] = haloA;
    arr[7] = gasDrag;
    arr[8] = time;
    arr[9] = agingRate;
    arr[10] = sfDensity;
    arr[11] = sfEff;
    arr[12] = sfDt;
    u32[13] = Math.min(haloCenters.length, 2);
    if (haloCenters[0]) arr.set(haloCenters[0], 16);
    if (haloCenters[1]) arr.set(haloCenters[1], 20);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, arr.buffer);
  }

  /** Star formation + stellar recycling pass — call once per frame, after steps. */
  encodeStarFormation(commandEncoder) {
    if (!this.ready) return;
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.sfPipeline);
    pass.setBindGroup(0, this.sfBindGroups[this.currentBuffer][this.stellarCurrent]);
    pass.dispatchWorkgroups(Math.ceil(this.N / 256));
    pass.end();
  }

  /** Upload a freshly-built BH tree (raw flat buffer from the worker). */
  uploadBHTree(treeBuffer, numNodes, levelRanges) {
    if (!this.ready) return;
    const bytes = Math.min(numNodes * 64, this.maxBHNodes * 64);
    this.device.queue.writeBuffer(this.bhBuffer, 0, treeBuffer, 0, bytes);
    if (levelRanges) {
      this.levelRanges = levelRanges;
      const u = new Uint32Array(16 * 64);   // 16 slots × 256 B
      for (let i = 0; i < levelRanges.length && i < 16; i++) {
        u[i * 64] = levelRanges[i][0];
        u[i * 64 + 1] = levelRanges[i][1];
      }
      this.device.queue.writeBuffer(this.refreshUniform, 0, u);
    }
  }

  /**
   * Re-sum node COMs/masses from CURRENT particle positions, deepest level
   * first. Keeps tree forces a smooth function of the live state between
   * asynchronous topology rebuilds. Call before each step().
   */
  encodeRefresh(commandEncoder) {
    if (!this.ready || !this.levelRanges) return;
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.refreshPipeline);
    for (let lvl = this.levelRanges.length - 1; lvl >= 0; lvl--) {
      const count = this.levelRanges[lvl][1];
      if (count === 0) continue;
      pass.setBindGroup(0, this.refreshBindGroups[this.currentBuffer], [lvl * 256]);
      pass.dispatchWorkgroups(Math.ceil(count / 64));
    }
    pass.end();
  }

  step(commandEncoder) {
    if (!this.ready) return;
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroups[this.currentBuffer]);
    pass.dispatchWorkgroups(Math.ceil(this.N / 256));
    pass.end();
    this.currentBuffer ^= 1; // flip ping-pong
  }

  /**
   * Encode: compact current positions → copy to staging (for the tree worker).
   * Caller must only invoke when no map is pending on the staging buffer.
   */
  encodePositionReadback(commandEncoder) {
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.compactPipeline);
    // currentBuffer points at the bind group whose IN buffer holds latest state
    pass.setBindGroup(0, this.bindGroups[this.currentBuffer]);
    pass.dispatchWorkgroups(Math.ceil(this.N / 256));
    pass.end();
    commandEncoder.copyBufferToBuffer(this.posCompactBuffer, 0, this.posStagingBuffer, 0, this.N * 16);
  }

  /** Map the staging buffer and copy positions into `out` (vec4 stride). */
  async readPositions(out) {
    await this.posStagingBuffer.mapAsync(GPUMapMode.READ);
    out.set(new Float32Array(this.posStagingBuffer.getMappedRange()));
    this.posStagingBuffer.unmap();
  }

  /** Buffer containing current (post-step) positions — for rendering */
  get positionBuffer() {
    // After step(), currentBuffer flipped to point at the OUTPUT buffer
    return this.buffers[this.currentBuffer];
  }

  destroy() {
    this.buffers?.forEach(b => b.destroy());
    this.stellarBuffers?.forEach(b => b.destroy());
    this.paramsBuffer?.destroy();
    this.bhBuffer?.destroy();
    this.posCompactBuffer?.destroy();
    this.posStagingBuffer?.destroy();
    this.permBuffer?.destroy();
    this.refreshUniform?.destroy();
    this.energyBuffer?.destroy();
  }
}
