/** Request a WebGPU device. Throws if WebGPU is unavailable. */
export async function requestGPUDevice() {
  if (!navigator.gpu) throw new Error('WebGPU not supported in this browser. Use Chrome 113+.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found.');
  return adapter.requestDevice();
}

export class NBodyGPU {
  constructor(device, particleData) {
    this.device = device;
    this.N = particleData.length / 8;
    this.currentBuffer = 0; // ping-pong index
    this.ready = false;
    this._init(particleData);
  }

  async _init(particleData) {
    const device = this.device;
    const byteSize = particleData.byteLength;

    // Two ping-pong storage buffers
    this.buffers = [
      device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX,
      }),
      device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX,
      }),
    ];

    // Upload initial conditions to both buffers
    device.queue.writeBuffer(this.buffers[0], 0, particleData);
    device.queue.writeBuffer(this.buffers[1], 0, particleData);

    // Params uniform: [dt, G, epsilon², N, theta, pad, pad, pad] — 32 bytes
    this.paramsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // BH tree buffer: preallocated for up to N*2+1000 nodes (64 bytes each).
    // Contents are written each frame via uploadBHTree().
    this.maxBHNodes = this.N * 2 + 1000;
    this.bhBuffer = device.createBuffer({
      size: this.maxBHNodes * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Load BH shader
    const resp = await fetch('./src/physics/shaders/nbody_bh.wgsl');
    const code = await resp.text();
    const shader = device.createShaderModule({ code });

    // Bind group layout: particlesIn, particlesOut, params, bhNodes
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    // Compute pipeline
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shader, entryPoint: 'main' },
    });

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
      ],
    });
  }

  updateCentralMass(mass) {
    // Update particle 0's mass in both ping-pong buffers (byte offset 12 = 3 floats in)
    const buf = new Float32Array([mass]);
    this.device.queue.writeBuffer(this.buffers[0], 12, buf);
    this.device.queue.writeBuffer(this.buffers[1], 12, buf);
  }

  updateParams(dt, G, epsilon, N, theta = 0.5) {
    // Pack as [dt, G, epsilon², N_as_u32, theta, pad, pad, pad]
    const arr = new Float32Array(8);
    arr[0] = dt;
    arr[1] = G;
    arr[2] = epsilon * epsilon;
    new Uint32Array(arr.buffer)[3] = N;
    arr[4] = theta;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, arr.buffer);
  }

  /**
   * Upload a freshly-built BHTree to the GPU buffer.
   * Must be called before the next sim.step() to take effect.
   */
  uploadBHTree(bhTree) {
    if (!this.ready) return;
    const bytes = bhTree.byteLength;
    if (bytes > this.maxBHNodes * 64) {
      console.warn(`BHTree: ${bhTree.numNodes} nodes exceeds preallocated max ${this.maxBHNodes}; tree truncated`);
      return;
    }
    // Write only the used portion — avoids uploading the entire preallocated buffer.
    this.device.queue.writeBuffer(this.bhBuffer, 0, bhTree.f32.buffer, 0, bytes);
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

  /** Buffer containing current (post-step) positions — for rendering/readback */
  get positionBuffer() {
    // After step(), currentBuffer flipped to point at the OUTPUT buffer
    return this.buffers[this.currentBuffer];
  }
}
