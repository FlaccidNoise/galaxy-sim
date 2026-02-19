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
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
      }),
      device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
      }),
    ];

    // Upload initial conditions to both buffers
    device.queue.writeBuffer(this.buffers[0], 0, particleData);
    device.queue.writeBuffer(this.buffers[1], 0, particleData);

    // Uniform buffer for simulation params
    this.paramsBuffer = device.createBuffer({
      size: 16, // [dt: f32, G: f32, epsilon2: f32, N: u32] — 16 bytes total
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Load WGSL shader
    const wgslResp = await fetch('./src/physics/shaders/nbody.wgsl');
    const wgslCode = await wgslResp.text();
    const shaderModule = device.createShaderModule({ code: wgslCode });

    // Bind group layout
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    // Compute pipeline
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' },
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
      ],
    });
  }

  updateParams(dt, G, epsilon, N) {
    // Pack as [dt, G, epsilon², N_as_u32_reinterpreted]
    const arr = new Float32Array(4);
    arr[0] = dt;
    arr[1] = G;
    arr[2] = epsilon * epsilon;
    // Write N as u32 into the 4th slot (same bytes, different type view)
    new Uint32Array(arr.buffer)[3] = N;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, arr.buffer);
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
