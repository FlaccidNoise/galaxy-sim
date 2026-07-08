// src/render/SystemRenderer.js
// Renders the selected star's solar system: one pass into the galaxy's HDR
// target (loadOp: load) with its own depth texture. Local coordinate space:
// star at origin, ~1 unit = 1 AU. All body motion is analytic (vertex shader).

import { makeViewProj } from '../gpu/Camera.js';
import { KIND, mulberry32, orbitRate } from '../system/SystemGen.js';

export class SystemRenderer {
  constructor(device) {
    this.device = device;
    this.system = null;
    this.numBodies = 0;
    this.numRings = 0;
    this.numBelt = 0;
    this.numOrbits = 0;
    this.width = 1;
    this.height = 1;
  }

  async init() {
    const device = this.device;
    const url = new URL('./shaders/system.wgsl', import.meta.url);
    const code = await (await fetch(url)).text();
    this.module = device.createShaderModule({ code, label: 'system.wgsl' });
    const info = await this.module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length) {
      throw new Error('system.wgsl: ' + errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }

    this.uniformBuffer = device.createBuffer({
      size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });

    const HDR = 'rgba16float';
    // Premultiplied alpha: crossfades over the galaxy backdrop
    const premul = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    this.bodyPipeline = device.createRenderPipeline({
      label: 'sys-bodies',
      layout: this.pipelineLayout,
      vertex: { module: this.module, entryPoint: 'vs_body' },
      fragment: { module: this.module, entryPoint: 'fs_body', targets: [{ format: HDR, blend: premul }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    return this;
  }

  /** Pack GPU buffers for a generated system (null clears). */
  setSystem(system) {
    this.system = system;
    for (const b of ['bodyBuffer', 'ringBuffer', 'beltBuffer', 'orbitBuffer']) {
      this[b]?.destroy();
      this[b] = null;
    }
    this.bindGroup = null;
    this.numBodies = 0;
    this.numRings = 0;
    this.numBelt = 0;
    this.numOrbits = 0;
    if (!system) return;

    // Bodies: sun first, then planets, then moons (parent orbit inlined so
    // the shader computes moon centers without indirection).
    const bodies = [[0, 0, 0, system.star.radius, 0, 0, 0, KIND.SUN, 0.5, 0, 0, 0]];
    const rings = [];
    const orbits = [];
    for (const p of system.planets) {
      bodies.push([p.a, p.phase, p.rate, p.radius, 0, 0, 0, p.kind, p.seed, p.p1, p.p2, 0]);
      orbits.push([p.a, 0, 0, 0]);
      if (p.rings) rings.push([p.a, p.phase, p.rate, p.rings.seed, p.rings.inner, p.rings.outer, 0, 0]);
      for (const m of p.moons) {
        bodies.push([m.a, m.phase, m.rate, m.radius, p.a, p.phase, p.rate, KIND.ICY, p.seed + 0.31, 0.5, 0.5, 0]);
      }
    }
    this.numBodies = bodies.length;
    this.numRings = rings.length;
    this.numOrbits = orbits.length;

    const dev = this.device;
    const mk = (rows, floatsPerRow) => {
      const data = new Float32Array(Math.max(rows.length, 1) * floatsPerRow);
      rows.forEach((r, i) => data.set(r, i * floatsPerRow));
      const buf = dev.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      dev.queue.writeBuffer(buf, 0, data);
      return buf;
    };
    this.bodyBuffer = mk(bodies, 12);
    this.ringBuffer = mk(rings, 8);
    this.orbitBuffer = mk(orbits, 4);

    // Belt particles, deterministic from the belt seed
    let beltData = new Float32Array(8);
    if (system.belt) {
      const { a, width, count, seed } = system.belt;
      const rb = mulberry32(seed);
      beltData = new Float32Array(count * 8);
      for (let i = 0; i < count; i++) {
        const pa = a + (rb() * 2 - 1) * width;
        const o = i * 8;
        beltData[o] = pa;
        beltData[o + 1] = rb() * 2 * Math.PI;
        beltData[o + 2] = orbitRate(pa);
        beltData[o + 3] = 0.6 + rb() * 0.4;        // size
        beltData[o + 4] = (rb() * 2 - 1) * 0.04;   // inclination
        beltData[o + 5] = rb() * 2 * Math.PI;      // node
        beltData[o + 6] = 0.4 + rb() * 0.6;        // brightness
      }
      this.numBelt = count;
    }
    this.beltBuffer = dev.createBuffer({
      size: beltData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(this.beltBuffer, 0, beltData);

    this.bindGroup = dev.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bodyBuffer } },
        { binding: 2, resource: { buffer: this.ringBuffer } },
        { binding: 3, resource: { buffer: this.beltBuffer } },
        { binding: 4, resource: { buffer: this.orbitBuffer } },
      ],
    });
  }

  /** Write uniforms: galaxy-camera orientation at system-space radius rSys. */
  updateCamera(camera, rSys, fade, time) {
    const eye = [
      -camera.forward[0] * rSys,
      -camera.forward[1] * rSys,
      -camera.forward[2] * rSys,
    ];
    const vp = new Float32Array(16);
    makeViewProj(vp, eye, camera.right, camera.up, camera.forward,
                 camera.fovY, camera.aspect, 0.005, 3000);
    const u = new Float32Array(32);
    u.set(vp, 0);
    u.set(camera.right, 16); u[19] = time;
    u.set(camera.up, 20);    u[23] = fade;
    u.set(eye, 24);          u[27] = this.system?.star.tempK ?? 5778;
    u[28] = this.width; u[29] = this.height;
    u[30] = Math.tan(camera.fovY / 2);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u);
  }

  /** Encode the system pass into the shared HDR target. */
  encode(encoder, sceneView, width, height) {
    if (!this.bindGroup || this.numBodies === 0) return;
    if (!this.depthTex || this.depthTex.width !== width || this.depthTex.height !== height) {
      this.depthTex?.destroy();
      this.depthTex = this.device.createTexture({
        label: 'sysDepth',
        size: [width, height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.depthView = this.depthTex.createView();
    }
    this.width = width;
    this.height = height;

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: sceneView, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.bodyPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, this.numBodies);
    pass.end();
  }

  destroy() {
    for (const b of ['bodyBuffer', 'ringBuffer', 'beltBuffer', 'orbitBuffer', 'uniformBuffer']) {
      this[b]?.destroy();
    }
    this.depthTex?.destroy();
  }
}
