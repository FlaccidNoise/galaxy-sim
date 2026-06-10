/**
 * Raw WebGPU renderer — replaces Three.js entirely.
 *
 * Frame graph:
 *   1. scene pass (rgba16float HDR): procedural starfield, then additive
 *      particle sprites vertex-pulled from the physics storage buffer
 *   2. bloom downsample mip chain (13-tap, Karis-averaged first pass)
 *   3. bloom upsample chain (tent filter, additive)
 *   4. composite to swapchain: scene + bloom → exposure → AgX → sRGB
 *
 * Particle positions never touch the CPU.
 */
export class RendererGPU {
  constructor(device, canvas) {
    this.device = device;
    this.canvas = canvas;

    // Display tuning (driven by Controls)
    this.exposure = 1.0;        // linear, set via 2^EV
    this.bloomStrength = 0.35;
    this.starIntensity = 1.0;
    this.sizeScale = 1.0;
    this.colorMode = 0;         // 0 spectral, 1 velocity, 2 mass
    this.speedScale = 50.0;
    this.centralIdx = 0;        // tracks the central-mass particle through sorts
    this.agingRate = 25.0;      // Myr of stellar aging per simulation time unit
    this.dustStrength = 0.45;
    this.hiiStrength = 0.6;

    this._sceneBindGroups = new Map(); // pipeline → particleBuffer → stellarBuffer → bind group
  }

  async init() {
    const device = this.device;
    this.context = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });

    const url = new URL('./shaders/render.wgsl', import.meta.url);
    const code = await (await fetch(url)).text();
    this.module = device.createShaderModule({ code, label: 'render.wgsl' });
    const info = await this.module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length) {
      throw new Error('render.wgsl: ' + errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }

    // Uniforms
    this.uniformBuffer = device.createBuffer({
      size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.compositeParams = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });

    this._createPipelines();
    this._createTargets();

    window.addEventListener('resize', () => this._createTargets());
    return this;
  }

  _createPipelines() {
    const device = this.device, module = this.module;
    const HDR = 'rgba16float';

    this.bgPipeline = device.createRenderPipeline({
      label: 'background',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_background', targets: [{ format: HDR }] },
      primitive: { topology: 'triangle-list' },
    });

    this.particlePipeline = device.createRenderPipeline({
      label: 'particles',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_particle' },
      fragment: {
        module, entryPoint: 'fs_particle',
        targets: [{
          format: HDR,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Dust: multiplicative extinction (dst *= 1 − srcRGB), drawn after stars
    this.dustPipeline = device.createRenderPipeline({
      label: 'dust',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_dust' },
      fragment: {
        module, entryPoint: 'fs_dust',
        targets: [{
          format: HDR,
          blend: {
            color: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // H-II nebulae: additive pink emission on top of the dust
    this.hiiPipeline = device.createRenderPipeline({
      label: 'hii',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_hii' },
      fragment: {
        module, entryPoint: 'fs_hii',
        targets: [{
          format: HDR,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const fsPipeline = (entryPoint, format, blend) => device.createRenderPipeline({
      label: entryPoint,
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint, targets: [{ format, blend }] },
      primitive: { topology: 'triangle-list' },
    });

    this.downFirstPipeline = fsPipeline('fs_downsample_first', HDR);
    this.downPipeline = fsPipeline('fs_downsample', HDR);
    this.upPipeline = fsPipeline('fs_upsample', HDR, {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    });
    this.compositePipeline = fsPipeline('fs_composite', this.format);
  }

  _createTargets() {
    const dpr = devicePixelRatio || 1;
    const w = Math.max(Math.floor(this.canvas.clientWidth * dpr), 8);
    const h = Math.max(Math.floor(this.canvas.clientHeight * dpr), 8);
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;

    this.sceneTex?.destroy();
    this.bloomTex?.destroy();

    this.sceneTex = this.device.createTexture({
      label: 'sceneHDR',
      size: [w, h],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sceneView = this.sceneTex.createView();

    // Bloom chain starts at half res, down to ~8px
    const bw = Math.max(w >> 1, 1), bh = Math.max(h >> 1, 1);
    this.bloomLevels = Math.max(Math.floor(Math.log2(Math.min(bw, bh))) - 2, 1);
    this.bloomTex = this.device.createTexture({
      label: 'bloom',
      size: [bw, bh],
      format: 'rgba16float',
      mipLevelCount: this.bloomLevels,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bloomViews = [];
    for (let i = 0; i < this.bloomLevels; i++) {
      this.bloomViews.push(this.bloomTex.createView({ baseMipLevel: i, mipLevelCount: 1 }));
    }

    this._buildStaticBindGroups();
    this._sceneBindGroups.clear();
  }

  _buildStaticBindGroups() {
    const device = this.device;

    this.bgBindGroup = device.createBindGroup({
      layout: this.bgPipeline.getBindGroupLayout(0),
      entries: [{ binding: 1, resource: { buffer: this.uniformBuffer } }],
    });

    // Bloom downsample bind groups: [0] samples scene, [i] samples mip i-1
    this.downBindGroups = [
      device.createBindGroup({
        layout: this.downFirstPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sceneView },
          { binding: 1, resource: this.sampler },
        ],
      }),
    ];
    for (let i = 1; i < this.bloomLevels; i++) {
      this.downBindGroups.push(device.createBindGroup({
        layout: this.downPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.bloomViews[i - 1] },
          { binding: 1, resource: this.sampler },
        ],
      }));
    }

    // Upsample bind groups: render to mip i sampling mip i+1
    this.upBindGroups = [];
    for (let i = 0; i < this.bloomLevels - 1; i++) {
      this.upBindGroups.push(device.createBindGroup({
        layout: this.upPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.bloomViews[i + 1] },
          { binding: 1, resource: this.sampler },
        ],
      }));
    }

    this.compositeBindGroup = device.createBindGroup({
      layout: this.compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sceneView },
        { binding: 1, resource: this.bloomViews[0] },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.compositeParams } },
      ],
    });
  }

  _sceneBindGroup(pipeline, buffer, stellarBuffer) {
    let byBuffer = this._sceneBindGroups.get(pipeline);
    if (!byBuffer) { byBuffer = new Map(); this._sceneBindGroups.set(pipeline, byBuffer); }
    let inner = byBuffer.get(buffer);
    if (!inner) { inner = new Map(); byBuffer.set(buffer, inner); }
    let bg = inner.get(stellarBuffer);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.uniformBuffer } },
          { binding: 2, resource: { buffer: stellarBuffer } },
        ],
      });
      inner.set(stellarBuffer, bg);
    }
    return bg;
  }

  _writeUniforms(camera, time) {
    const u = new Float32Array(44);
    u.set(camera.viewProj, 0);
    u.set(camera.right, 16);  u[19] = Math.tan(camera.fovY / 2);
    u.set(camera.up, 20);     u[23] = camera.aspect;
    u.set(camera.forward, 24); u[27] = time;
    u.set(camera.position, 28); u[31] = this.starIntensity;
    u[32] = this.width; u[33] = this.height;
    u[34] = this.sizeScale;
    new Uint32Array(u.buffer)[35] = this.colorMode;
    u[36] = this.speedScale;
    new Uint32Array(u.buffer)[37] = this.centralIdx;
    u[38] = this.agingRate;
    u[39] = this.dustStrength;
    u[40] = this.hiiStrength;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u);

    this.device.queue.writeBuffer(this.compositeParams, 0,
      new Float32Array([this.exposure, this.bloomStrength, 0, 0]));
  }

  /** Drop cached bind groups (call when the sim's buffers are replaced). */
  clearSceneCache() {
    this._sceneBindGroups.clear();
  }

  /** Encode the full frame into the given command encoder. */
  encode(encoder, camera, particleBuffer, stellarBuffer, particleCount, time) {
    this._writeUniforms(camera, time);

    // 1. Scene: background + particles into HDR target
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.sceneView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear', storeOp: 'store',
        }],
      });
      pass.setPipeline(this.bgPipeline);
      pass.setBindGroup(0, this.bgBindGroup);
      pass.draw(3);

      pass.setPipeline(this.particlePipeline);
      pass.setBindGroup(0, this._sceneBindGroup(this.particlePipeline, particleBuffer, stellarBuffer));
      pass.draw(6, particleCount);

      if (this.dustStrength > 0) {
        pass.setPipeline(this.dustPipeline);
        pass.setBindGroup(0, this._sceneBindGroup(this.dustPipeline, particleBuffer, stellarBuffer));
        pass.draw(6, particleCount);
      }
      if (this.hiiStrength > 0) {
        pass.setPipeline(this.hiiPipeline);
        pass.setBindGroup(0, this._sceneBindGroup(this.hiiPipeline, particleBuffer, stellarBuffer));
        pass.draw(6, particleCount);
      }
      pass.end();
    }

    // 2. Bloom downsample chain
    for (let i = 0; i < this.bloomLevels; i++) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomViews[i],
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear', storeOp: 'store',
        }],
      });
      pass.setPipeline(i === 0 ? this.downFirstPipeline : this.downPipeline);
      pass.setBindGroup(0, this.downBindGroups[i]);
      pass.draw(3);
      pass.end();
    }

    // 3. Bloom upsample chain (additive into existing mip content)
    for (let i = this.bloomLevels - 2; i >= 0; i--) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomViews[i],
          loadOp: 'load', storeOp: 'store',
        }],
      });
      pass.setPipeline(this.upPipeline);
      pass.setBindGroup(0, this.upBindGroups[i]);
      pass.draw(3);
      pass.end();
    }

    // 4. Composite to swapchain
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear', storeOp: 'store',
        }],
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, this.compositeBindGroup);
      pass.draw(3);
      pass.end();
    }
  }
}
