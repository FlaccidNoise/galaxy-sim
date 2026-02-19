# Galaxy Sim Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a browser-based N-body galaxy simulation with WebGPU compute shaders, Three.js WebGPU rendering, and a lil-gui parameter panel.

**Architecture:** Three.js WebGPU renderer owns the canvas and camera. Physics runs in a raw WGSL compute shader on the same `GPUDevice` as the renderer. Particle data lives in a single GPU-side storage buffer — the compute shader writes to it, the vertex shader reads from it. Zero CPU readback per frame.

**Tech Stack:** Three.js r170 (WebGPU renderer), raw WGSL compute shaders, lil-gui 0.19, ES modules via CDN importmap, `npx serve .`

---

## Before You Start

### Verify WebGPU support
Open Chrome (117+) or Edge. In console: `!!navigator.gpu` → must return `true`. Safari and Firefox do not support WebGPU as of early 2026.

### importmap pattern
All imports use bare specifiers resolved via importmap in `index.html`. No npm install, no bundler. Example:
```json
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
    "lil-gui": "https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm"
  }
}
```

### Three.js WebGPU internals pattern
After `await renderer.init()`, the raw WebGPU device is at:
```javascript
const device = renderer.backend.device; // GPUDevice
```
This is an internal API — works on r163–r170, may change. Verify it logs a `GPUDevice` object after init.

---

## Task 1: Project Scaffold

**Files:**
- Create: `index.html`
- Create: `package.json`

**Step 1: Create `package.json`**

```json
{
  "type": "module",
  "scripts": {
    "start": "npx serve ."
  }
}
```

**Step 2: Create `index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Galaxy Sim</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #000; overflow: hidden; }
    canvas { display: block; }
  </style>
  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
      "lil-gui": "https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm"
    }
  }
  </script>
</head>
<body>
  <script type="module" src="src/main.js"></script>
</body>
</html>
```

**Step 3: Verify scaffold runs**

```bash
cd ~/CodingProjects/galaxy-sim
npx serve .
```

Open `http://localhost:3000` — expect a black page, no console errors.

**Step 4: Commit**

```bash
git add index.html package.json
git commit -m "feat: project scaffold with importmap"
```

---

## Task 2: Initial Conditions Generator

**Files:**
- Create: `src/physics/InitialConditions.js`

This module is pure JS — no WebGPU, no Three.js. It generates a `Float32Array` of particle data in the layout `[x, y, z, mass, vx, vy, vz, pad]` per particle (8 floats × 4 bytes = 32 bytes per particle).

**Step 1: Create `src/physics/InitialConditions.js`**

```javascript
// Particle layout: [x, y, z, mass, vx, vy, vz, pad] per particle (stride = 8 floats)
const STRIDE = 8;

/**
 * Galaxy disk: exponential surface density, circular velocity support,
 * Gaussian scale height, optional central mass.
 */
export function galaxyDisk(params) {
  const {
    N = 10000,
    diskRadius = 50.0,
    diskHeight = 2.0,
    centralMass = 1000.0,
    starMass = 1.0,
    velDispersion = 0.5,
    G = 1.0,
  } = params;

  const data = new Float32Array(N * STRIDE);

  for (let i = 0; i < N; i++) {
    const base = i * STRIDE;

    // Exponential radial distribution via inverse CDF sampling
    // CDF of exp(-r/r0): r = -r0 * ln(1 - u), r0 = diskRadius / 3
    const r0 = diskRadius / 3.0;
    const u = Math.random();
    const r = -r0 * Math.log(1.0 - u * (1.0 - Math.exp(-diskRadius / r0)));
    const theta = Math.random() * 2 * Math.PI;

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);
    // Gaussian scale height
    const z = gaussRandom() * diskHeight;

    // Circular velocity: v_c = sqrt(G * M_enc / r)
    // Simplified enclosed mass: central mass + stars inside r (assume uniform for velocity)
    const massEnc = centralMass + starMass * N * (r / diskRadius);
    const vc = r > 0.01 ? Math.sqrt(G * massEnc / r) : 0;

    // Tangential velocity (perpendicular to radial direction in xy plane)
    const vx = -vc * Math.sin(theta) + gaussRandom() * velDispersion;
    const vy =  vc * Math.cos(theta) + gaussRandom() * velDispersion;
    const vz = gaussRandom() * velDispersion * 0.2; // small vertical motion

    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = starMass;
    data[base + 4] = vx;
    data[base + 5] = vy;
    data[base + 6] = vz;
    data[base + 7] = 0; // pad
  }

  // Place central mass as particle 0
  data[0] = 0; data[1] = 0; data[2] = 0;
  data[3] = centralMass;
  data[4] = 0; data[5] = 0; data[6] = 0;
  data[7] = 0;

  return data;
}

/**
 * Uniform sphere: random positions, zero net velocity.
 */
export function uniformSphere(params) {
  const { N = 5000, radius = 30.0, starMass = 1.0 } = params;
  const data = new Float32Array(N * STRIDE);
  for (let i = 0; i < N; i++) {
    const base = i * STRIDE;
    // Rejection sample inside sphere
    let x, y, z;
    do {
      x = (Math.random() * 2 - 1) * radius;
      y = (Math.random() * 2 - 1) * radius;
      z = (Math.random() * 2 - 1) * radius;
    } while (x*x + y*y + z*z > radius * radius);
    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = starMass;
    // zero velocity
  }
  return data;
}

/**
 * Two galaxy disks on a collision course.
 */
export function twoGalaxyCollision(params) {
  const {
    N = 10000,
    diskRadius = 30.0,
    diskHeight = 1.5,
    centralMass = 800.0,
    starMass = 1.0,
    velDispersion = 0.3,
    G = 1.0,
    separation = 80.0,
    approachVel = 0.8,
  } = params;

  const halfN = Math.floor(N / 2);
  const paramsA = { N: halfN, diskRadius, diskHeight, centralMass, starMass, velDispersion, G };
  const paramsB = { ...paramsA, N: N - halfN };

  const diskA = galaxyDisk(paramsA);
  const diskB = galaxyDisk(paramsB);

  const data = new Float32Array(N * STRIDE);

  // Galaxy A: offset -separation/2 along x, moving +x
  for (let i = 0; i < halfN; i++) {
    const src = i * STRIDE;
    const dst = i * STRIDE;
    data[dst + 0] = diskA[src + 0] - separation / 2;
    data[dst + 1] = diskA[src + 1];
    data[dst + 2] = diskA[src + 2];
    data[dst + 3] = diskA[src + 3];
    data[dst + 4] = diskA[src + 4] + approachVel;
    data[dst + 5] = diskA[src + 5];
    data[dst + 6] = diskA[src + 6];
    data[dst + 7] = 0;
  }

  // Galaxy B: offset +separation/2 along x, tilted 30°, moving -x
  const tilt = Math.PI / 6;
  const cosT = Math.cos(tilt), sinT = Math.sin(tilt);
  for (let i = 0; i < N - halfN; i++) {
    const src = i * STRIDE;
    const dst = (halfN + i) * STRIDE;
    const bx = diskB[src + 0];
    const by = diskB[src + 1];
    const bz = diskB[src + 2];
    data[dst + 0] = bx + separation / 2;
    data[dst + 1] = by * cosT - bz * sinT;
    data[dst + 2] = by * sinT + bz * cosT;
    data[dst + 3] = diskB[src + 3];
    data[dst + 4] = diskB[src + 4] - approachVel;
    data[dst + 5] = diskB[src + 5];
    data[dst + 6] = diskB[src + 6];
    data[dst + 7] = 0;
  }

  return data;
}

// Box-Muller Gaussian random
function gaussRandom() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
```

**Step 2: Smoke-test in browser console**

Add temporarily to `index.html` body script tag:
```html
<script type="module">
import { galaxyDisk } from './src/physics/InitialConditions.js';
const d = galaxyDisk({ N: 100 });
console.assert(d.length === 800, 'wrong size');
console.assert(d[3] === 1000, 'particle 0 should be central mass'); // centralMass default
console.log('InitialConditions OK', d.slice(0, 8));
</script>
```
Open browser, verify no errors, remove test script.

**Step 3: Commit**

```bash
git add src/physics/InitialConditions.js
git commit -m "feat: initial conditions generator (disk, sphere, collision)"
```

---

## Task 3: WGSL Compute Shader

**Files:**
- Create: `src/physics/shaders/nbody.wgsl`

**Step 1: Create `src/physics/shaders/nbody.wgsl`**

```wgsl
// Particle layout: vec4 pos_mass, vec4 vel_pad
// [x, y, z, mass,  vx, vy, vz, pad] per particle

struct Particle {
    pos  : vec3f,
    mass : f32,
    vel  : vec3f,
    pad  : f32,
}

struct Params {
    dt       : f32,
    G        : f32,
    epsilon2 : f32,  // softening^2, prevents singularity
    N        : u32,
}

@group(0) @binding(0) var<storage, read>       particlesIn  : array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut : array<Particle>;
@group(0) @binding(2) var<uniform>             params       : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let pi  = particlesIn[i];
    var acc = vec3f(0.0);

    // Direct summation: O(N²) per frame, but massively parallel on GPU
    for (var j : u32 = 0u; j < params.N; j++) {
        if (j == i) { continue; }
        let pj  = particlesIn[j];
        let r   = pj.pos - pi.pos;
        let d2  = dot(r, r) + params.epsilon2;
        // Force magnitude / distance: G * mj / d^3 (Plummer softening)
        let inv = params.G * pj.mass / (d2 * sqrt(d2));
        acc    += r * inv;
    }

    // Symplectic Euler (velocity Verlet without second force eval)
    // For proper leapfrog, split into two dispatches (kick-drift-kick).
    // Symplectic Euler is energy-conserving enough for visual simulation.
    let newVel = pi.vel + acc * params.dt;
    let newPos = pi.pos + newVel * params.dt;

    particlesOut[i] = Particle(newPos, pi.mass, newVel, 0.0);
}
```

No test for this step — will be validated when the compute pipeline runs.

**Step 2: Commit**

```bash
git add src/physics/shaders/nbody.wgsl
git commit -m "feat: WGSL N-body compute shader (direct summation, symplectic Euler)"
```

---

## Task 4: NBodyGPU — WebGPU Compute Pipeline

**Files:**
- Create: `src/physics/NBodyGPU.js`

**Step 1: Create `src/physics/NBodyGPU.js`**

```javascript
export class NBodyGPU {
  constructor(device, particleData) {
    this.device = device;
    this.N = particleData.length / 8;
    this.currentBuffer = 0; // ping-pong index
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
      size: 16, // 4 floats
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

    // Two bind groups (one per ping-pong direction)
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
    const arr = new Float32Array([dt, G, epsilon * epsilon, 0]);
    const uintArr = new Uint32Array(arr.buffer);
    uintArr[3] = N; // N as u32 in same buffer
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

  /** Buffer containing current (post-step) positions — for rendering */
  get positionBuffer() {
    // After step(), currentBuffer points to the OUTPUT buffer
    return this.buffers[this.currentBuffer];
  }
}
```

**Step 2: Verify NBodyGPU initializes without errors**

Add temporary test to `index.html`:
```html
<script type="module">
import { NBodyGPU } from './src/physics/NBodyGPU.js';
import { galaxyDisk } from './src/physics/InitialConditions.js';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const data = galaxyDisk({ N: 1000 });
const sim = new NBodyGPU(device, data);
await new Promise(r => setTimeout(r, 500)); // let async init complete
console.assert(sim.ready, 'NBodyGPU not ready');
console.assert(sim.buffers[0] instanceof GPUBuffer, 'buffer not created');
console.log('NBodyGPU OK, N =', sim.N);
</script>
```

Expected: `NBodyGPU OK, N = 1000` in console. Remove test after verifying.

**Step 3: Commit**

```bash
git add src/physics/NBodyGPU.js
git commit -m "feat: WebGPU compute pipeline for N-body simulation"
```

---

## Task 5: Three.js Renderer Setup

**Files:**
- Create: `src/render/Renderer.js`

**Step 1: Create `src/render/Renderer.js`**

```javascript
import * as THREE from 'three';
import WebGPURenderer from 'three/addons/renderers/webgpu/WebGPURenderer.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export class Renderer {
  constructor() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 10000);
    this.camera.position.set(0, 60, 120);
    this.camera.lookAt(0, 0, 0);
  }

  async init() {
    this.renderer = new WebGPURenderer({ antialias: false });
    this.renderer.setSize(innerWidth, innerHeight);
    this.renderer.setPixelRatio(devicePixelRatio);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    document.body.appendChild(this.renderer.domElement);

    await this.renderer.init();

    // OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // Bloom postprocessing
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(innerWidth, innerHeight),
      1.5,  // strength
      0.4,  // radius
      0.1   // threshold
    );
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    // Resize handler
    window.addEventListener('resize', () => {
      this.camera.aspect = innerWidth / innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(innerWidth, innerHeight);
      this.composer.setSize(innerWidth, innerHeight);
    });

    // Expose underlying WebGPU device for physics
    this.device = this.renderer.backend.device;
    console.log('WebGPU device:', this.device);

    return this;
  }

  render() {
    this.controls.update();
    this.composer.render();
  }

  setBloom(strength, radius, threshold) {
    this.bloomPass.strength = strength;
    this.bloomPass.radius = radius;
    this.bloomPass.threshold = threshold;
  }
}
```

**Step 2: Smoke-test renderer**

Create a temporary `src/main.js`:
```javascript
import { Renderer } from './render/Renderer.js';

const r = await new Renderer().init();
r.scene.add(new (await import('three')).default.Mesh(
  new (await import('three')).default.SphereGeometry(1),
  new (await import('three')).default.MeshBasicMaterial({ color: 0xffffff })
));

function loop() {
  requestAnimationFrame(loop);
  r.render();
}
loop();
```

Expected: white sphere visible, orbit controls work, no console errors. Clear `main.js` after verifying.

**Step 3: Commit**

```bash
git add src/render/Renderer.js
git commit -m "feat: Three.js WebGPU renderer with bloom and orbit controls"
```

---

## Task 6: Particle System — GPU Buffer to Three.js

**Files:**
- Create: `src/render/ParticleSystem.js`

The challenge: physics buffers are raw `GPUBuffer`s. Three.js needs to read positions from them in a vertex shader. The solution is a custom `THREE.RawShaderMaterial` with a vertex shader that reads from a storage binding — but Three.js WebGPU renderer's shader system uses TSL nodes, not raw GLSL.

**Simpler fallback:** Create a `THREE.Points` geometry with a `THREE.BufferAttribute` backed by a CPU `Float32Array`. Each frame, copy positions from GPU buffer to CPU, then update the geometry. At 50k particles × 12 bytes = 600KB/frame — acceptable.

**Step 1: Create `src/render/ParticleSystem.js`**

```javascript
import * as THREE from 'three';

export class ParticleSystem {
  constructor(scene, N) {
    this.N = N;
    this.positions = new Float32Array(N * 3);
    this.colors = new Float32Array(N * 3);
    this.speeds = new Float32Array(N); // for color mapping

    const geometry = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.positions, 3);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.colAttr = new THREE.BufferAttribute(this.colors, 3);
    this.colAttr.setUsage(THREE.DynamicDrawUsage);

    geometry.setAttribute('position', this.posAttr);
    geometry.setAttribute('color', this.colAttr);

    const material = new THREE.PointsMaterial({
      size: 0.3,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.points = new THREE.Points(geometry, material);
    scene.add(this.points);

    // Staging buffer for GPU readback (created lazily)
    this._stagingBuffer = null;
    this._stagingMapped = false;
  }

  /** Create staging buffer on the device (call once after device is available) */
  initStaging(device, physicsBufferByteSize) {
    this._device = device;
    this._physicsByteSize = physicsBufferByteSize;
    this._stagingBuffer = device.createBuffer({
      size: physicsBufferByteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Readback positions from GPU buffer, update Three.js geometry.
   * Async: queues a copy command, returns a promise that resolves when done.
   */
  async readbackAndUpdate(commandEncoder, gpuBuffer, colorMode, speedScale) {
    // Queue copy: physics buffer → staging buffer
    commandEncoder.copyBufferToBuffer(gpuBuffer, 0, this._stagingBuffer, 0, this._physicsByteSize);
    // Note: commandEncoder must be submitted BEFORE mapAsync
    // Caller submits encoder, then calls updateFromStaging()
  }

  async updateFromStaging(colorMode, speedScale = 50.0) {
    await this._stagingBuffer.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(this._stagingBuffer.getMappedRange());

    const N = this.N;
    for (let i = 0; i < N; i++) {
      const src = i * 8;
      const dst = i * 3;
      this.positions[dst]     = raw[src];
      this.positions[dst + 1] = raw[src + 1];
      this.positions[dst + 2] = raw[src + 2];

      const vx = raw[src + 4], vy = raw[src + 5], vz = raw[src + 6];
      const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
      this.speeds[i] = speed;

      if (colorMode === 'velocity') {
        velocityColor(speed, speedScale, this.colors, dst);
      } else {
        massColor(raw[src + 3], this.colors, dst);
      }
    }

    this._stagingBuffer.unmap();

    this.posAttr.needsUpdate = true;
    this.colAttr.needsUpdate = true;
  }

  setPointSize(size) {
    this.points.material.size = size;
  }
}

// Hot (fast) = blue-white, slow = red-orange
function velocityColor(speed, scale, out, i) {
  const t = Math.min(speed / scale, 1.0);
  // Red → orange → yellow → white → blue
  if (t < 0.33) {
    const s = t / 0.33;
    out[i]     = 1.0;
    out[i + 1] = s * 0.5;
    out[i + 2] = 0.0;
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    out[i]     = 1.0;
    out[i + 1] = 0.5 + s * 0.5;
    out[i + 2] = s * 0.5;
  } else {
    const s = (t - 0.66) / 0.34;
    out[i]     = 1.0 - s * 0.5;
    out[i + 1] = 1.0 - s * 0.5;
    out[i + 2] = 0.5 + s * 0.5;
  }
}

// Massive = blue-white (O/B), light = red (M)
function massColor(mass, out, i) {
  const t = Math.min(Math.log(mass + 1) / 10, 1.0);
  out[i]     = t > 0.5 ? 1.0 : t * 2;
  out[i + 1] = t > 0.5 ? 1.0 : t * 2;
  out[i + 2] = 0.3 + t * 0.7;
}
```

**Step 2: Commit**

```bash
git add src/render/ParticleSystem.js
git commit -m "feat: particle system with GPU readback and velocity/mass coloring"
```

---

## Task 7: Controls Panel

**Files:**
- Create: `src/ui/Controls.js`

**Step 1: Create `src/ui/Controls.js`**

```javascript
import GUI from 'lil-gui';

export class Controls {
  constructor(sim, particles, renderer) {
    this.sim = sim;
    this.particles = particles;
    this.renderer = renderer;
    this._restartPending = false;
    this._onRestart = null;

    this.state = {
      // Simulation
      paused: false,
      speed: 1.0,
      // Initial conditions
      preset: 'Galaxy Disk',
      N: 10000,
      diskRadius: 50.0,
      diskHeight: 2.0,
      centralMass: 1000.0,
      velDispersion: 0.5,
      // Physics
      G: 1.0,
      epsilon: 1.0,
      dt: 0.02,
      // Display
      colorMode: 'Velocity',
      bloomStrength: 1.5,
      pointSize: 0.3,
    };

    this._build();
  }

  _build() {
    const gui = new GUI({ title: 'Galaxy Sim' });

    // Simulation folder
    const simFolder = gui.addFolder('Simulation');
    simFolder.add(this.state, 'paused').name('Pause').onChange(v => this.sim.paused = v);
    simFolder.add(this.state, 'speed', 0.1, 10, 0.1).name('Speed');
    simFolder.add({ reset: () => this._restart() }, 'reset').name('Reset');

    // Initial conditions folder
    const icFolder = gui.addFolder('Initial Conditions');
    icFolder.add(this.state, 'preset', ['Galaxy Disk', 'Uniform Sphere', 'Collision'])
      .name('Preset').onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'N', 1000, 100000, 1000).name('N (stars)')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'diskRadius', 10, 200).name('Disk Radius')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'diskHeight', 0.5, 20).name('Disk Height')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'centralMass', 0, 5000).name('Central Mass')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'velDispersion', 0, 5).name('Vel Dispersion')
      .onChange(() => this._markRestartPending(icFolder));

    // Physics folder
    const physFolder = gui.addFolder('Physics');
    physFolder.add(this.state, 'G', 0.1, 10).name('G');
    physFolder.add(this.state, 'epsilon', 0.1, 10).name('Softening ε');
    physFolder.add(this.state, 'dt', 0.001, 0.1).name('Timestep dt');

    // Display folder
    const dispFolder = gui.addFolder('Display');
    dispFolder.add(this.state, 'colorMode', ['Velocity', 'Mass']).name('Color Mode');
    dispFolder.add(this.state, 'bloomStrength', 0, 3).name('Bloom').onChange(v => {
      this.renderer.setBloom(v, 0.4, 0.1);
    });
    dispFolder.add(this.state, 'pointSize', 0.05, 2).name('Point Size').onChange(v => {
      this.particles.setPointSize(v);
    });
  }

  _markRestartPending(folder) {
    if (!this._restartPending) {
      this._restartPending = true;
      folder.domElement.style.borderLeft = '3px solid orange';
    }
  }

  onRestart(fn) { this._onRestart = fn; }

  _restart() {
    this._restartPending = false;
    document.querySelectorAll('.lil-gui').forEach(el => el.style.borderLeft = '');
    if (this._onRestart) this._onRestart(this.state);
  }
}
```

**Step 2: Commit**

```bash
git add src/ui/Controls.js
git commit -m "feat: lil-gui controls panel with restart-pending indicator"
```

---

## Task 8: Main Wiring — Animation Loop

**Files:**
- Create: `src/main.js`

**Step 1: Create `src/main.js`**

```javascript
import { Renderer } from './render/Renderer.js';
import { ParticleSystem } from './render/ParticleSystem.js';
import { NBodyGPU } from './physics/NBodyGPU.js';
import { galaxyDisk, uniformSphere, twoGalaxyCollision } from './physics/InitialConditions.js';
import { Controls } from './ui/Controls.js';

let renderer, particles, sim, controls;
let paused = false;

async function init(params) {
  // 1. Generate initial conditions
  const presetFns = {
    'Galaxy Disk': galaxyDisk,
    'Uniform Sphere': uniformSphere,
    'Collision': twoGalaxyCollision,
  };
  const icParams = {
    N: params.N,
    diskRadius: params.diskRadius,
    diskHeight: params.diskHeight,
    centralMass: params.centralMass,
    velDispersion: params.velDispersion,
    G: params.G,
  };
  const particleData = presetFns[params.preset](icParams);
  const N = particleData.length / 8;

  // 2. Init renderer (if first time)
  if (!renderer) {
    renderer = await new Renderer().init();
  } else {
    // Clear old particles from scene
    if (particles) renderer.scene.remove(particles.points);
  }

  // 3. Init physics on WebGPU device
  sim = new NBodyGPU(renderer.device, particleData);
  await new Promise(r => setTimeout(r, 100)); // let async GPU init settle

  // 4. Init particle system
  particles = new ParticleSystem(renderer.scene, N);
  particles.initStaging(renderer.device, particleData.byteLength);

  // 5. Init controls (first time only)
  if (!controls) {
    controls = new Controls(sim, particles, renderer);
    controls.onRestart(state => init(state));
  }

  return { sim, particles, controls };
}

async function loop() {
  requestAnimationFrame(loop);

  if (!sim || !sim.ready || controls?.state.paused) {
    if (renderer) renderer.render();
    return;
  }

  const s = controls.state;
  const stepsPerFrame = Math.max(1, Math.round(s.speed));

  for (let i = 0; i < stepsPerFrame; i++) {
    sim.updateParams(s.dt, s.G, s.epsilon, sim.N);

    const enc = renderer.device.createCommandEncoder();
    sim.step(enc);

    // On last step, queue readback
    if (i === stepsPerFrame - 1) {
      await particles.readbackAndUpdate(enc, sim.positionBuffer, s.colorMode, 50.0);
      renderer.device.queue.submit([enc.finish()]);
      await particles.updateFromStaging(s.colorMode, 50.0);
    } else {
      renderer.device.queue.submit([enc.finish()]);
    }
  }

  renderer.render();
}

// Bootstrap
const defaultState = {
  preset: 'Galaxy Disk',
  N: 10000,
  diskRadius: 50,
  diskHeight: 2,
  centralMass: 1000,
  velDispersion: 0.5,
  G: 1.0,
  epsilon: 1.0,
  dt: 0.02,
};

await init(defaultState);
loop();
```

**Step 2: Test full pipeline**

Run `npx serve .`, open browser. Expected:
- Galaxy disk visible — points spread in a disk shape
- Stars colored orange/red/white by velocity
- Bloom glow visible around dense center
- Orbit controls work (drag, zoom, pan)
- GUI panel visible, sliders functional
- No console errors

**Step 3: Tune default parameters if needed**

If the galaxy collapses immediately: increase `epsilon` (softening).
If stars barely move: increase `G` or `dt`.
If too slow at N=10k: reduce N to 5k to verify physics is correct first, then increase.

**Step 4: Commit**

```bash
git add src/main.js
git commit -m "feat: main animation loop — full sim + render pipeline wired"
```

---

## Task 9: Polish Pass

**Goal:** Ensure the default view is visually compelling and controls are intuitive.

**Step 1: Tune defaults to look good**

In `src/main.js` `defaultState`, adjust until the galaxy looks right:
- Galaxy should not immediately collapse — epsilon = 1.0–2.0 usually works
- Disk should be clearly disk-shaped from the default camera angle
- Stars should visibly orbit the core

**Step 2: Add FPS counter**

In `src/main.js`, above `loop()`:
```javascript
let frameCount = 0, lastFPS = performance.now();
const fpsEl = Object.assign(document.createElement('div'), {
  style: 'position:fixed;top:8px;left:8px;color:#fff;font:12px monospace;pointer-events:none',
});
document.body.appendChild(fpsEl);
```

Inside `loop()`, at the top:
```javascript
frameCount++;
const now = performance.now();
if (now - lastFPS > 1000) {
  fpsEl.textContent = `${frameCount} fps | N=${sim?.N ?? 0}`;
  frameCount = 0;
  lastFPS = now;
}
```

**Step 3: Final commit**

```bash
git add src/main.js
git commit -m "polish: default parameter tuning and FPS counter"
```

---

## Known Limitations & Upgrade Notes

**GPU readback bottleneck:** `mapAsync` introduces a GPU stall each frame. At N>50k this may become the limiting factor. Upgrade path: implement a custom WGSL vertex shader that reads positions directly from the storage buffer (no CPU copy needed). This requires using Three.js TSL nodes or switching to a raw WebGPU render pipeline.

**O(N²) scaling:** At N=50k, each frame runs 2.5B multiply-adds. On an M2 Mac GPU this is ~1–5ms/frame. At N=100k it becomes 10B ops — may drop below 60fps. The Phase C upgrade (Particle-Mesh / Barnes-Hut on GPU) addresses this.

**Leapfrog accuracy:** Current integrator is symplectic Euler (one force evaluation per step). For higher accuracy, split `step()` into two dispatches: kick, drift, kick. The `NBodyGPU` interface supports this cleanly.

---

## Running

```bash
cd ~/CodingProjects/galaxy-sim
npx serve .
# Open http://localhost:3000 in Chrome (WebGPU required)
```
