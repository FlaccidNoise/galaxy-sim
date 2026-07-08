# Solar-System Zoom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Click a star in the galaxy sim, zoom past a radius threshold, and crossfade into a deterministic procedurally generated solar system (planets, moons, rings, asteroid belt) rendered in its own AU-scale scene while the live galaxy stays behind as a backdrop.

**Architecture:** Two-scene crossfade (see `docs/plans/2026-07-08-solar-system-zoom-design.md`). CPU click-picking against the tree-worker position snapshot; selection tracked through Morton permutations; pure deterministic generator seeded from the star's stellar record; new impostor-sphere render pass drawing into the existing HDR target before bloom, with its own depth texture.

**Tech Stack:** Raw WebGPU + WGSL, vanilla ES modules (no bundler — served via `npx serve .`), lil-gui via import map, Node built-in test runner for pure modules.

## Global Constraints

- No build step. Static ES modules; bare imports resolve only through the import map in `index.html`. No new runtime dependencies.
- Unit tests use Node's built-in runner: `node --test` (Node ≥ 18). No test framework installs.
- Do NOT modify `src/physics/shaders/nbody_bh.wgsl`, `src/physics/mortonTree.js`, or `src/physics/bhWorker.js`.
- The galaxy scene pass stays depth-free. Only the new system pass owns a depth attachment.
- The galaxy renderer's uniform buffer stays 176 bytes — new fields use the existing `_q0`/`_q1` spare slots (float indices 41/42).
- All new per-frame GPU work is encoded into the existing single command encoder in `main.js`'s `loop()`.
- WGSL modules are fetched at runtime (`new URL(..., import.meta.url)`) and compile-checked via `getCompilationInfo()` — copy the existing pattern (`RendererGPU.js:39-46`).
- The repo has a pre-commit guard: git commit messages must NOT contain `Co-Authored-By` trailers.
- Commit after every task.

**Key verified facts** (do not re-derive):
- Particle layout: 8 floats `[x,y,z,mass,vx,vy,vz,gasFlag]`; position snapshot `posTransfer` is vec4/particle `[x,y,z,mass]` (`NBodyGPU.js:335-350`).
- Stellar record: 4 floats `[stellarMass M☉, birthTime, seed, dustWeight]`; gas has `stellarMass === -1`, central BHs `0` (`InitialConditions.js:1-2,243,266`).
- Permutation is a gather: `particlesOut[i] = particlesIn[perm[i]]` (`nbody_bh.wgsl:325`), so a retained old index remaps to the `new` index where `perm[new] === old`.
- Camera: disk in XY plane, damped orbit, `viewProj` column-major; wheel clamps radius to `[2, 4000]` (`Camera.js:63`).
- Renderer uniform floats: 40 = hiiStrength, 41/42/43 = spare (`render.wgsl:26`, `RendererGPU.js:273`).

---

### Task 1: SystemGen — deterministic system generator

**Files:**
- Create: `src/system/SystemGen.js`
- Test: `test/SystemGen.test.js`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces:
  - `mulberry32(seed: u32) → () => float in [0,1)`
  - `orbitRate(a: number) → number` — angular rate for semi-major axis `a`
  - `spectralClass(tempK: number) → 'O'|'B'|'A'|'F'|'G'|'K'|'M'`
  - `KIND = { SUN:0, SCORCHED:1, VENUSIAN:2, TERRAN:3, ICY:4, GAS:5, ICE_GIANT:6 }`
  - `generateSystem(stellarMass, birthTime, seed) → { star, planets, belt }` where:
    - `star: { mass, lum, tempK, class, radius }`
    - `planets: [{ a, phase, rate, radius, kind, seed, p1, p2, rings: null|{inner, outer, seed}, moons: [{a, phase, rate, radius}] }]`
    - `belt: null | { a, width, count, seed }` (`seed` is a u32 for mulberry32)

- [ ] **Step 1: Write the failing tests**

```js
// test/SystemGen.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, generateSystem, spectralClass, KIND } from '../src/system/SystemGen.js';

test('mulberry32 is deterministic and in [0,1)', () => {
  const a = mulberry32(42), b = mulberry32(42);
  for (let i = 0; i < 1000; i++) {
    const x = a();
    assert.equal(x, b());
    assert.ok(x >= 0 && x < 1);
  }
});

test('generateSystem is deterministic for identical stellar records', () => {
  assert.deepEqual(generateSystem(1.0, -4500, 0.7315), generateSystem(1.0, -4500, 0.7315));
});

test('different seeds give different systems', () => {
  assert.notDeepEqual(generateSystem(1.0, -4500, 0.7315), generateSystem(1.0, -4500, 0.7316));
});

test('planet kinds respect the frost line', () => {
  for (let i = 0; i < 200; i++) {
    const sys = generateSystem(0.5 + i * 0.01, -i * 13, i / 200);
    const frost = 2.7 * Math.sqrt(sys.star.lum);
    for (const p of sys.planets) {
      if (p.kind === KIND.GAS || p.kind === KIND.ICE_GIANT) assert.ok(p.a >= frost);
      else assert.ok(p.a < frost);
    }
  }
});

test('orbits strictly increase; moons stay outside the planet; ≤4 moons', () => {
  for (let i = 0; i < 200; i++) {
    const sys = generateSystem(0.2 + i * 0.05, -i * 7, (i * 37 % 100) / 100);
    for (let j = 1; j < sys.planets.length; j++) {
      assert.ok(sys.planets[j].a > sys.planets[j - 1].a);
    }
    for (const p of sys.planets) {
      assert.ok(p.moons.length <= 4);
      for (const m of p.moons) assert.ok(m.a > p.radius);
    }
  }
});

test('massive stars get at most 3 planets', () => {
  for (let i = 0; i < 100; i++) {
    assert.ok(generateSystem(8 + i * 0.3, -i, i / 100).planets.length <= 3);
  }
});

test('only gas giants get rings', () => {
  for (let i = 0; i < 200; i++) {
    const sys = generateSystem(0.3 + i * 0.02, -i * 3, (i * 7 % 100) / 100);
    for (const p of sys.planets) {
      if (p.rings) assert.equal(p.kind, KIND.GAS);
    }
  }
});

test('spectral classes', () => {
  assert.equal(spectralClass(5778), 'G');
  assert.equal(spectralClass(35000), 'O');
  assert.equal(spectralClass(3000), 'M');
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `node --test`
Expected: FAIL — `Cannot find module .../src/system/SystemGen.js`

- [ ] **Step 3: Implement SystemGen**

```js
// src/system/SystemGen.js
// Deterministic procedural solar-system generation. Pure module — no GPU, no
// DOM. Distances in AU-ish units; body radii are visual (inflated so planets
// read at orbit distances), not to scale.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Appearance palette ids — must match the switch in system.wgsl
export const KIND = { SUN: 0, SCORCHED: 1, VENUSIAN: 2, TERRAN: 3, ICY: 4, GAS: 5, ICE_GIANT: 6 };

// Sim-time units; inner planets complete an orbit in ~10 s of real time
const ORBIT_SPEED = 0.08;

export function orbitRate(a) {
  return 2 * Math.PI * ORBIT_SPEED / Math.pow(a, 1.5);
}

export function spectralClass(tempK) {
  if (tempK >= 30000) return 'O';
  if (tempK >= 10000) return 'B';
  if (tempK >= 7500) return 'A';
  if (tempK >= 6000) return 'F';
  if (tempK >= 5200) return 'G';
  if (tempK >= 3700) return 'K';
  return 'M';
}

export function generateSystem(stellarMass, birthTime, seed) {
  // Seed from the star's intrinsic float bits — stable across tree re-sorts
  const f = new Float32Array([stellarMass, birthTime, seed]);
  const u = new Uint32Array(f.buffer);
  const rand = mulberry32(u[0] ^ Math.imul(u[1], 0x9E3779B9) ^ Math.imul(u[2], 0x85EBCA6B));

  const M = Math.max(stellarMass, 0.08);
  const lum = Math.pow(M, 3.5);
  const tempK = 5778 * Math.pow(M, 0.54);   // matches starState() in render.wgsl
  const sqrtL = Math.sqrt(lum);
  const frostLine = 2.7 * sqrtL;

  const star = {
    mass: M, lum, tempK,
    class: spectralClass(tempK),
    radius: 0.35 * Math.pow(M, 0.8),
  };

  // Hot massive stars: sparse scorched systems. Red dwarfs: tight and small
  // (orbit spacing scales with √L).
  const maxPlanets = M >= 8 ? 3 : 8;
  const count = Math.min(Math.floor(rand() * 9), maxPlanets);

  const planets = [];
  let a = (0.3 + 0.2 * rand()) * sqrtL + star.radius * 3;
  for (let i = 0; i < count; i++) {
    const rocky = a < frostLine;
    let kind, radius;
    if (rocky) {
      radius = 0.02 + 0.05 * rand();
      const hz = a >= 0.95 * sqrtL && a <= 1.4 * sqrtL;
      kind = hz ? KIND.TERRAN
           : a < 0.5 * sqrtL ? KIND.SCORCHED
           : a < 0.95 * sqrtL ? KIND.VENUSIAN
           : KIND.ICY;
    } else {
      kind = rand() < 0.6 ? KIND.GAS : KIND.ICE_GIANT;
      radius = kind === KIND.GAS ? 0.15 + 0.2 * rand() : 0.1 + 0.08 * rand();
    }
    const planet = {
      a, phase: rand() * 2 * Math.PI, rate: orbitRate(a),
      radius, kind, seed: rand(), p1: rand(), p2: rand(),
      rings: null, moons: [],
    };
    if (kind === KIND.GAS && rand() < 0.35) {
      planet.rings = { inner: radius * 1.4, outer: radius * (2.2 + rand()), seed: rand() };
    }
    const giant = kind === KIND.GAS || kind === KIND.ICE_GIANT;
    const nMoons = giant ? 1 + Math.floor(rand() * 4)
                 : (radius > 0.045 && rand() < 0.4 ? 1 : 0);
    let ma = radius * (2.5 + rand());
    for (let m = 0; m < nMoons; m++) {
      planet.moons.push({
        a: ma, phase: rand() * 2 * Math.PI,
        rate: orbitRate(ma) * 8,      // fast — visible motion at moon scale
        radius: radius * (0.12 + 0.15 * rand()),
      });
      ma *= 1.6 + 0.4 * rand();
    }
    planets.push(planet);
    a *= 1.4 + 0.6 * rand();
  }

  // Belt: 70% of systems, in the widest orbit gap (or beyond the planets)
  let belt = null;
  if (rand() < 0.7) {
    let beltA = planets.length ? planets[planets.length - 1].a * 1.5 : 2 * sqrtL;
    let widest = 0;
    for (let i = 0; i + 1 < planets.length; i++) {
      const ratio = planets[i + 1].a / planets[i].a;
      if (ratio > 1.8 && ratio > widest) {
        widest = ratio;
        beltA = Math.sqrt(planets[i].a * planets[i + 1].a);
      }
    }
    belt = { a: beltA, width: beltA * 0.15, count: 3000, seed: Math.floor(rand() * 0xffffffff) };
  }

  return { star, planets, belt };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `node --test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/system/SystemGen.js test/SystemGen.test.js
git commit -m "feat: deterministic procedural solar-system generator"
```

---

### Task 2: Star picking module

**Files:**
- Create: `src/ui/StarSelection.js`
- Modify: `src/physics/NBodyGPU.js:85-88` (stellar buffers gain `COPY_SRC`)
- Test: `test/StarSelection.test.js`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pickCandidates(positions: Float32Array /*vec4 stride*/, N, viewProj: Float32Array16 /*column-major*/, clickX, clickY, width, height, maxPx = 15, k = 4) → [{idx, distPx}]` sorted best-first, empty if no hit. Click/size args are in the same pixel space (device pixels).
  - `async readStellarRecords(device: GPUDevice, stellarBuffer: GPUBuffer, indices: number[]) → Float32Array` — 4 floats per requested index, in request order.

- [ ] **Step 1: Write the failing tests**

```js
// test/StarSelection.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickCandidates } from '../src/ui/StarSelection.js';

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `node --test`
Expected: FAIL — `Cannot find module .../src/ui/StarSelection.js`

- [ ] **Step 3: Implement the module**

```js
// src/ui/StarSelection.js
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
```

- [ ] **Step 4: Add `COPY_SRC` to the stellar buffers**

In `src/physics/NBodyGPU.js` (currently lines 85-88), change:

```js
    this.stellarBuffers = [0, 1].map(() => device.createBuffer({
      size: this.N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }));
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `node --test`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/ui/StarSelection.js test/StarSelection.test.js src/physics/NBodyGPU.js
git commit -m "feat: star click-picking with stellar-record readback"
```

---

### Task 3: Camera — min-radius clamp and reusable view math

**Files:**
- Modify: `src/gpu/Camera.js`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `export function makeViewProj(out: Float32Array16, eye, right, up, forward, fovY, aspect, near, far)` — column-major view-projection.
  - `Camera.minRadius: number` (default 2) — wheel zoom clamp floor.
  - `Camera.setMinRadius(r)` — sets the floor and raises `_goal.radius` if below it.
  - `Camera.viewFrom(radius) → { position, right, up, forward, viewProj, fovY, aspect }` — this camera's POV recomputed at a different radius (same target/orientation). Duck-types as `camera` for `RendererGPU._writeUniforms`.

- [ ] **Step 1: Add `minRadius` and use it in the wheel handler**

In the constructor (after `this.radius = 130;`):

```js
    this.minRadius = 2;        // zoom floor — drops to 0.02 while a star is selected
```

Replace the wheel handler body (`Camera.js:63`):

```js
      this._goal.radius = Math.min(Math.max(this._goal.radius * Math.exp(e.deltaY * 0.001), this.minRadius), 4000);
```

- [ ] **Step 2: Extract the matrix build into an exported helper**

Replace the tail of `update()` (everything from `// View matrix (lookAt)...` through the `mat4mul` call, `Camera.js:90-108`) with:

```js
    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1);
    this.aspect = aspect;
    makeViewProj(this.viewProj, this.position, r, u, f, this.fovY, aspect, this.near, this.far);
```

Add at module level (above the existing helpers):

```js
/** Column-major view-projection from an eye + orthonormal basis. */
export function makeViewProj(out, eye, r, u, f, fovY, aspect, near, far) {
  const view = [
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    -dot3(r, eye), -dot3(u, eye), dot3(f, eye), 1,
  ];
  const fpr = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const proj = [
    fpr / aspect, 0, 0, 0,
    0, fpr, 0, 0,
    0, 0, far * nf, -1,
    0, 0, far * near * nf, 0,
  ];
  mat4mul(out, proj, view);
}
```

- [ ] **Step 3: Add `setMinRadius` and `viewFrom` methods**

```js
  setMinRadius(r) {
    this.minRadius = r;
    if (this._goal.radius < r) this._goal.radius = r;
  }

  /** This camera's POV at a different radius (same target and orientation). */
  viewFrom(radius) {
    const eye = [
      this.target[0] - this.forward[0] * radius,
      this.target[1] - this.forward[1] * radius,
      this.target[2] - this.forward[2] * radius,
    ];
    const viewProj = new Float32Array(16);
    makeViewProj(viewProj, eye, this.right, this.up, this.forward,
                 this.fovY, this.aspect, this.near, this.far);
    return { position: eye, right: this.right, up: this.up, forward: this.forward,
             viewProj, fovY: this.fovY, aspect: this.aspect };
  }
```

- [ ] **Step 4: Verify no regression**

Run: `npm start`, open the served URL in Chrome.
Expected: galaxy renders and orbits exactly as before — rotate, pan, wheel zoom (still clamps at radius 2), presets restart. Also `node --test` still passes.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/Camera.js
git commit -m "refactor: exported view math, min-radius clamp, viewFrom for backdrop pinning"
```

---

### Task 4: Galaxy renderer — selection uniforms + highlight ring

**Files:**
- Modify: `src/render/shaders/render.wgsl`
- Modify: `src/render/RendererGPU.js`
- Modify: `src/main.js` (one debug line)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `RendererGPU.selectedIdx: number` (default `0xffffffff` = none) and `RendererGPU.systemFade: number` (0..1) — set by `main.js`, consumed by shaders.
  - `render.wgsl` uniform fields `systemFade : f32` (float idx 41) and `selectedIdx : u32` (idx 42).
  - Selected star sprite dims by `1 − systemFade`; a pulsing blue ring marks the selection.

- [ ] **Step 1: Update the Uniforms struct in render.wgsl**

Replace line 26 (`hiiStrength   : f32,  _q0 : f32,  _q1 : f32,  _q2 : f32,`):

```wgsl
    hiiStrength   : f32,    systemFade : f32,  selectedIdx : u32,  _q2 : f32,
```

- [ ] **Step 2: Fade the selected star's sprite in vs_particle**

At the end of `vs_particle`, immediately before `return out;` (after the color if/else chain):

```wgsl
    if (ii == u.selectedIdx) {
        out.color *= (1.0 - u.systemFade);   // the system's sun takes over
    }
```

- [ ] **Step 3: Add highlight ring entry points to render.wgsl**

After the H-II section (`fs_hii`):

```wgsl
// ── Selection highlight ring ────────────────────────────────────────────────
@vertex
fn vs_highlight(@builtin(vertex_index) vi : u32) -> VSOut {
    var quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    var out : VSOut;
    if (u.selectedIdx >= arrayLength(&particles)) {
        out.pos = collapse();
        out.quadUV = vec2f(0.0);
        out.color = vec3f(0.0);
        return out;
    }
    let p = particles[u.selectedIdx];
    var clip = u.viewProj * vec4f(p.pos, 1.0);
    let corner = quad[vi];
    let sizePx = 26.0 * u.sizeScale;
    clip = vec4f(clip.xy + corner * (sizePx / u.resolution) * clip.w, clip.zw);
    out.pos = clip;
    out.quadUV = corner;
    out.color = vec3f(0.35, 0.6, 1.0) * (1.0 - u.systemFade);
    return out;
}

@fragment
fn fs_highlight(in : VSOut) -> @location(0) vec4f {
    let d = length(in.quadUV);
    let ring = smoothstep(0.10, 0.02, abs(d - 0.8)) * (0.75 + 0.25 * sin(u.time * 4.0));
    if (d > 1.0 || ring < 0.01) { discard; }
    return vec4f(in.color * ring, 1.0);
}
```

- [ ] **Step 4: Wire the pipeline and uniforms in RendererGPU.js**

Constructor (after `this.hiiStrength = 0.6;`):

```js
    this.selectedIdx = 0xffffffff;   // no selection
    this.systemFade = 0;
```

In `_createPipelines()` (after `hiiPipeline`):

```js
    this.highlightPipeline = device.createRenderPipeline({
      label: 'highlight',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_highlight' },
      fragment: {
        module, entryPoint: 'fs_highlight',
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
```

**Important:** `vs_highlight`/`fs_highlight` reference only bindings 0 and 1, so the `layout: 'auto'` bind-group layout excludes the stellar buffer — the shared `_sceneBindGroup` helper (which binds 0/1/2) will NOT validate. Add a dedicated cached bind group:

```js
  _highlightBindGroup(buffer) {
    if (this._hlBuffer !== buffer) {
      this._hlBG = this.device.createBindGroup({
        layout: this.highlightPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.uniformBuffer } },
        ],
      });
      this._hlBuffer = buffer;
    }
    return this._hlBG;
  }
```

In `clearSceneCache()` add:

```js
    this._hlBuffer = null;
```

In `_writeUniforms`, after the `hiiStrength` line (`u[40] = ...`):

```js
    u[41] = this.systemFade;
    new Uint32Array(u.buffer)[42] = this.selectedIdx;
```

In `encode()`, inside the scene pass after the H-II draw and before `pass.end()`:

```js
      if (this.selectedIdx !== 0xffffffff) {
        pass.setPipeline(this.highlightPipeline);
        pass.setBindGroup(0, this._highlightBindGroup(particleBuffer));
        pass.draw(6, 1);
      }
```

- [ ] **Step 5: Expose the renderer for console verification**

In `src/main.js`, next to the existing debug hooks (`window.__sim = sim;`, main.js:133):

```js
  window.__renderer = renderer;
```

- [ ] **Step 6: Verify in the browser**

Run: `npm start`, open in Chrome, then in DevTools console:

```js
__renderer.selectedIdx = 5000   // → pulsing blue ring appears around a star
__renderer.systemFade = 1       // → ring AND that star's sprite fade out
__renderer.systemFade = 0; __renderer.selectedIdx = 0xffffffff  // → back to normal
```

Expected: exactly those behaviors; no WGSL compile errors in console; fps unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/render/shaders/render.wgsl src/render/RendererGPU.js src/main.js
git commit -m "feat: selection highlight ring + selected-star fade in galaxy renderer"
```

---

### Task 5: Selection wiring in main.js

**Files:**
- Modify: `src/main.js`

**Interfaces:**
- Consumes: `pickCandidates`, `readStellarRecords` (Task 2), `generateSystem` (Task 1), `camera.setMinRadius` (Task 3), `renderer.selectedIdx` (Task 4).
- Produces (module-level in main.js, used by Task 6):
  - `selection: null | { idx, record: Float32Array(4), system }`
  - `lastPositions: Float32Array | null` — retained vec4/particle snapshot
  - `select(idx, record)`, `deselect()`, `updateStarInfo()` (stub for now)
  - `window.__select(idx)` debug hook, `window.__system`

- [ ] **Step 1: Add imports and state**

```js
import { pickCandidates, readStellarRecords } from './ui/StarSelection.js';
import { generateSystem } from './system/SystemGen.js';
```

Module-level state (next to `let centralIdx = 0;`):

```js
let lastPositions = null;   // retained CPU snapshot (vec4/particle) for picking
let selection = null;       // { idx, record: Float32Array(4), system }
```

- [ ] **Step 2: Retain a snapshot copy before handing the buffer to the worker**

In `loop()`, inside `sim.readPositions(posTransfer).then(() => {...})` (main.js:255-259), after the `if (sim !== mySim || !treeWorker)` guard and before `treeWorker.postMessage(...)`:

```js
      if (!lastPositions || lastPositions.length !== posTransfer.length) {
        lastPositions = new Float32Array(posTransfer.length);
      }
      lastPositions.set(posTransfer);
```

- [ ] **Step 3: Add select/deselect/pick functions**

At module level:

```js
function updateStarInfo() {}   // filled in by the UI task

function select(idx, record) {
  selection = {
    idx,
    record: Float32Array.from(record),
    system: generateSystem(record[0], record[1], record[2]),
  };
  renderer.selectedIdx = idx;
  camera.setMinRadius(0.02);
  window.__system = selection.system;
  updateStarInfo();
}

function deselect() {
  if (!selection) return;
  selection = null;
  renderer.selectedIdx = 0xffffffff;
  renderer.systemFade = 0;
  camera.setMinRadius(2);
  if (camera._goal.radius < 2.5) camera._goal.radius = 2.5;
  window.__system = null;
  updateStarInfo();
}

async function handlePick(x, y) {
  if (!sim?.ready || !lastPositions) return;
  const dpr = devicePixelRatio || 1;
  const cands = pickCandidates(lastPositions, sim.N, camera.viewProj,
                               x * dpr, y * dpr, renderer.width, renderer.height);
  if (!cands.length) { deselect(); return; }
  const records = await readStellarRecords(gpuDevice, sim.stellarBuffer, cands.map(c => c.idx));
  for (let i = 0; i < cands.length; i++) {
    if (records[i * 4] <= 0) continue;   // gas (−1) and central BHs (0) unpickable
    select(cands[i].idx, records.subarray(i * 4, i * 4 + 4));
    return;
  }
  deselect();
}
```

- [ ] **Step 4: Bind input events (once, where the canvas is created)**

In `init()`, inside the `if (!renderer)` first-call block, after `camera = new Camera(canvas);`:

```js
    let downX = 0, downY = 0;
    canvas.addEventListener('pointerdown', e => { downX = e.clientX; downY = e.clientY; });
    canvas.addEventListener('pointerup', e => {
      // Click = left button released within 4 px of where it went down
      if (e.button !== 0 || Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return;
      handlePick(e.clientX, e.clientY).catch(console.error);
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') deselect();
    });
```

- [ ] **Step 5: Clear selection on restart**

In `init()`, in the teardown block (next to `if (sim) { ... }`, main.js:87):

```js
  deselect();
```

- [ ] **Step 6: Track the selection through tree permutations**

In `loop()`, inside the `if (pendingTree)` block, right after `renderer.centralIdx = centralIdx;` (main.js:207):

```js
    // Permutation is a gather (out[i] = in[perm[i]]): old index → the new
    // slot that references it.
    if (selection) {
      const perm = pendingTree.perm;
      let ni = -1;
      for (let i = 0; i < perm.length; i++) {
        if (perm[i] === selection.idx) { ni = i; break; }
      }
      if (ni >= 0) { selection.idx = ni; renderer.selectedIdx = ni; }
      else deselect();
    }
```

- [ ] **Step 7: Camera follows the selected star**

In `loop()`, immediately before `camera.update();` (main.js:167):

```js
  if (selection && lastPositions) {
    const i = selection.idx * 4;
    camera._goal.target[0] = lastPositions[i];
    camera._goal.target[1] = lastPositions[i + 1];
    camera._goal.target[2] = lastPositions[i + 2];
  }
```

- [ ] **Step 8: Add the debug hook**

Next to the other `window.__*` hooks in `init()`:

```js
  window.__select = async idx => {
    const rec = await readStellarRecords(gpuDevice, sim.stellarBuffer, [idx]);
    select(idx, rec.subarray(0, 4));
  };
```

- [ ] **Step 9: Verify in the browser**

Run: `npm start`, open in Chrome:
- Click a star → blue pulsing ring appears, camera glides to center it.
- Drag to rotate → no selection change. Click empty sky → deselects.
- Esc → deselects; if zoomed below radius 2, camera eases back out.
- Wheel in with a selection → radius now goes below 2 (sprites get huge — expected until Task 6).
- Collision preset: click each galaxy's bright center → BH never selects.
- Determinism: `await __select(3000); s1 = JSON.stringify(__system); await __select(3000); JSON.stringify(__system) === s1` → `true`.
- Restart (Reset button) → selection cleared.
- `node --test` still passes.

- [ ] **Step 10: Commit**

```bash
git add src/main.js
git commit -m "feat: click star selection with camera tracking and perm remap"
```

---

### Task 6: System scene — sun, planets, moons + transition wiring

**Files:**
- Create: `src/render/shaders/system.wgsl`
- Create: `src/render/SystemRenderer.js`
- Modify: `src/render/RendererGPU.js` (encode signature)
- Modify: `src/main.js` (fade/rSys mapping, backdrop pinning, renderer hookup)

**Interfaces:**
- Consumes: `generateSystem` output shape (Task 1), `makeViewProj` (Task 3), `camera.viewFrom` (Task 3), `renderer.systemFade` (Task 4), `selection` (Task 5), `KIND`, `mulberry32`, `orbitRate` (Task 1).
- Produces:
  - `SystemRenderer` class: `constructor(device)`, `async init() → this`, `setSystem(system|null)`, `updateCamera(camera, rSys, fade, time)`, `encode(encoder, sceneView, width, height)`.
  - `RendererGPU.encode(encoder, camera, particleBuffer, stellarBuffer, particleCount, time, systemScene = null)` — calls `systemScene?.encode(...)` after the galaxy scene pass, before bloom.
  - Constants in main.js: `R_ENTER = 1.5`, `R_FULL = 0.8`.

- [ ] **Step 1: Write system.wgsl (bodies only; ring/belt/orbit bindings declared, entry points come in Task 7)**

```wgsl
// Solar-system scene — impostor spheres in a local AU-scale coordinate space.
// Draws into the galaxy's HDR target (loadOp: load) with its own depth buffer.
// Orbits lie in the XY plane (matches the galaxy disk convention).

struct SysUniforms {
    viewProj   : mat4x4f,
    camRight   : vec3f,  time       : f32,
    camUp      : vec3f,  fade       : f32,
    camPos     : vec3f,  starTemp   : f32,
    resolution : vec2f,  tanHalfFov : f32,  _pad : f32,
}

struct Body {
    orbit  : vec4f,   // a, phase, rate, radius
    parent : vec4f,   // parentA, parentPhase, parentRate, kind (as f32)
    look   : vec4f,   // seed, p1, p2, unused
}

struct Ring {
    orbit : vec4f,    // parentA, parentPhase, parentRate, seed
    size  : vec4f,    // inner, outer, unused, unused
}

@group(0) @binding(0) var<uniform> su : SysUniforms;
@group(0) @binding(1) var<storage, read> bodies : array<Body>;
@group(0) @binding(2) var<storage, read> rings : array<Ring>;
// Belt particle = 2 vec4s: [a, phase, rate, size], [incl, node, bright, _]
@group(0) @binding(3) var<storage, read> belt : array<vec4f>;
@group(0) @binding(4) var<storage, read> orbitLines : array<vec4f>;  // a in .x

const QUAD = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
);

fn orbitPos(a : f32, phase : f32, rate : f32) -> vec3f {
    let ang = phase + rate * su.time;
    return vec3f(cos(ang) * a, sin(ang) * a, 0.0);
}

fn bodyCenter(b : Body) -> vec3f {
    // parent orbit (zeros for planets/sun) + own orbit
    return orbitPos(b.parent.x, b.parent.y, b.parent.z)
         + orbitPos(b.orbit.x, b.orbit.y, b.orbit.z);
}

// Planckian blackbody fit — same as render.wgsl
fn blackbodyRGB(tempK : f32) -> vec3f {
    let t = clamp(tempK, 1000.0, 40000.0) / 100.0;
    var r : f32; var g : f32; var b : f32;
    if (t <= 66.0) {
        r = 1.0;
        g = clamp((99.4708025861 * log(t) - 161.1195681661) / 255.0, 0.0, 1.0);
    } else {
        r = clamp(1.292936186 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
        g = clamp(1.1298908609 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
    }
    if (t >= 66.0) {
        b = 1.0;
    } else if (t <= 19.0) {
        b = 0.0;
    } else {
        b = clamp((138.5177312231 * log(t - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0);
    }
    return vec3f(r, g, b);
}

// ── 3D value noise + fbm (procedural surfaces) ──────────────────────────────
fn hash13(p : vec3f) -> f32 {
    var q = fract(p * 443.8975);
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

fn vnoise(p : vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i),                          hash13(i + vec3f(1.0, 0.0, 0.0)), u.x),
            mix(hash13(i + vec3f(0.0, 1.0, 0.0)),   hash13(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash13(i + vec3f(0.0, 0.0, 1.0)),   hash13(i + vec3f(1.0, 0.0, 1.0)), u.x),
            mix(hash13(i + vec3f(0.0, 1.0, 1.0)),   hash13(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y),
        u.z);
}

fn fbm(p : vec3f) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        v += amp * vnoise(q);
        q = q * 2.03;
        amp *= 0.5;
    }
    return v;
}

// Surface color as a function of the unit sphere-surface point — this is the
// piece that ports unchanged to a terrain-mesh pipeline for surface zoom.
fn surfaceColor(kind : u32, n : vec3f, look : vec3f, t : f32) -> vec3f {
    let sp = n * 3.0 + vec3f(look.x * 91.7);   // seed offsets the noise domain
    let h = fbm(sp);
    let lat = abs(n.z);                        // poles along ±Z
    switch kind {
        case 1u: {   // scorched rock
            return mix(vec3f(0.25, 0.12, 0.08), vec3f(0.55, 0.30, 0.15), h);
        }
        case 2u: {   // venusian: thick slowly-churning haze
            let bands = fbm(sp * 1.5 + vec3f(0.0, 0.0, t * 0.02));
            return mix(vec3f(0.75, 0.60, 0.30), vec3f(0.90, 0.80, 0.55), bands);
        }
        case 3u: {   // terran: oceans, continents, ice caps, drifting clouds
            let water = look.y * 0.35 + 0.35;
            var col : vec3f;
            if (h < water) {
                col = mix(vec3f(0.01, 0.05, 0.18), vec3f(0.02, 0.12, 0.30), h / max(water, 0.01));
            } else {
                col = mix(vec3f(0.10, 0.30, 0.12), vec3f(0.45, 0.38, 0.22), fbm(sp * 2.7));
            }
            col = mix(col, vec3f(0.90, 0.92, 0.95), smoothstep(0.78, 0.92, lat));
            let clouds = smoothstep(0.55, 0.75, fbm(sp * 4.0 + vec3f(t * 0.01, 0.0, 0.0)));
            return mix(col, vec3f(1.0), clouds * 0.7);
        }
        case 4u: {   // icy
            return mix(vec3f(0.55, 0.62, 0.70), vec3f(0.85, 0.90, 0.95), h);
        }
        case 5u: {   // gas giant: latitude bands + storms
            let bands = fbm(vec3f(sp.x * 0.3, sp.y * 0.3, n.z * (6.0 + look.z * 6.0)));
            let col = mix(vec3f(0.45, 0.30, 0.18), vec3f(0.80, 0.65, 0.45), bands);
            let storm = smoothstep(0.72, 0.85, fbm(sp * 2.0 + vec3f(31.7)));
            return mix(col, vec3f(0.85, 0.40, 0.25), storm * 0.6);
        }
        default: {   // 6: ice giant
            let bands = fbm(vec3f(sp.x * 0.4, sp.y * 0.4, n.z * 5.0));
            return mix(vec3f(0.15, 0.30, 0.50), vec3f(0.35, 0.60, 0.80), bands);
        }
    }
}

// ── Impostor sphere bodies ──────────────────────────────────────────────────
struct BodyVSOut {
    @builtin(position) pos : vec4f,
    @location(0) worldPos : vec3f,
    @location(1) @interpolate(flat) center : vec3f,
    @location(2) @interpolate(flat) radius : f32,
    @location(3) @interpolate(flat) kind : u32,
    @location(4) @interpolate(flat) look : vec3f,
}

@vertex
fn vs_body(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> BodyVSOut {
    let b = bodies[ii];
    let corner = QUAD[vi] * 1.05;   // slight inflation: silhouette never clips
    let c = bodyCenter(b);
    let world = c + (su.camRight * corner.x + su.camUp * corner.y) * b.orbit.w;
    var out : BodyVSOut;
    out.pos = su.viewProj * vec4f(world, 1.0);
    out.worldPos = world;
    out.center = c;
    out.radius = b.orbit.w;
    out.kind = u32(b.parent.w);
    out.look = b.look.xyz;
    return out;
}

struct BodyFSOut {
    @location(0) color : vec4f,
    @builtin(frag_depth) depth : f32,
}

@fragment
fn fs_body(in : BodyVSOut) -> BodyFSOut {
    let ro = su.camPos;
    let rd = normalize(in.worldPos - ro);
    let oc = ro - in.center;
    let bq = dot(oc, rd);
    let cq = dot(oc, oc) - in.radius * in.radius;
    let disc = bq * bq - cq;
    if (disc < 0.0) { discard; }
    let thit = -bq - sqrt(disc);
    if (thit < 0.0) { discard; }
    let hit = ro + rd * thit;
    let n = normalize(hit - in.center);

    var out : BodyFSOut;
    let clip = su.viewProj * vec4f(hit, 1.0);
    out.depth = clip.z / clip.w;

    if (in.kind == 0u) {   // the star: emissive, limb-darkened, HDR-hot
        let limb = 0.4 + 0.6 * abs(dot(n, rd));
        let granule = 0.9 + 0.2 * fbm(n * 8.0 + vec3f(su.time * 0.05));
        let col = blackbodyRGB(su.starTemp) * 14.0 * limb * granule;
        out.color = vec4f(col * su.fade, su.fade);
        return out;
    }

    let lightDir = -normalize(hit);   // the sun sits at the origin
    let diff = max(dot(n, lightDir), 0.0);
    var col = surfaceColor(in.kind, n, in.look, su.time) * (diff * 1.2 + 0.03);
    // Atmosphere rim on the lit edge
    let rim = pow(1.0 - abs(dot(n, -rd)), 3.0);
    var atmo = vec3f(0.0);
    if (in.kind == 3u) { atmo = vec3f(0.30, 0.50, 1.00); }
    if (in.kind == 5u) { atmo = vec3f(0.20, 0.16, 0.12); }
    col += atmo * rim * (diff + 0.15);
    out.color = vec4f(col * su.fade, su.fade);   // premultiplied crossfade
    return out;
}
```

- [ ] **Step 2: Write SystemRenderer.js**

```js
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
```

- [ ] **Step 3: Add the systemScene hook to RendererGPU.encode**

Change the signature (`RendererGPU.js:286`):

```js
  encode(encoder, camera, particleBuffer, stellarBuffer, particleCount, time, systemScene = null) {
```

After the scene pass's closing brace (after `pass.end();` / `}` at RendererGPU.js:317, before the bloom chain):

```js
    // 1.5 System scene (selected star's solar system) — into the same HDR
    // target so it shares bloom + AgX
    systemScene?.encode(encoder, this.sceneView, this.width, this.height);
```

- [ ] **Step 4: Wire the transition in main.js**

Import and construct (in `init()`'s first-call block, after `renderer = await ...init();`):

```js
import { SystemRenderer } from './render/SystemRenderer.js';
```

```js
    systemRenderer = await new SystemRenderer(gpuDevice).init();
```

with module-level declarations:

```js
let systemRenderer = null;
const R_ENTER = 1.5;   // galaxy-camera radius where the system starts fading in
const R_FULL = 0.8;    // fully faded in
```

Hook `setSystem` into selection (Task 5's functions):

- In `select()`, after `window.__system = selection.system;`: `systemRenderer.setSystem(selection.system);`
- In `deselect()`, after `renderer.systemFade = 0;`: `systemRenderer?.setSystem(null);`

In `loop()`, right after `camera.update();`, compute the transition (used by BOTH encode calls):

```js
  // Solar-system crossfade: below R_ENTER the system scene fades in and the
  // galaxy backdrop pins at R_ENTER so neighboring sprites don't fly past.
  let systemScene = null;
  let galaxyCam = camera;
  if (selection) {
    const t = Math.min(Math.max((R_ENTER - camera.radius) / (R_ENTER - R_FULL), 0), 1);
    const fade = t * t * (3 - 2 * t);
    renderer.systemFade = fade;
    if (fade > 0) {
      const rSys = 0.5 * Math.pow(camera.radius / 0.02, 1.27);   // 0.02→0.5, 1.5→~120
      systemRenderer.updateCamera(camera, rSys, fade, simTime);
      systemScene = systemRenderer;
      if (camera.radius < R_ENTER) galaxyCam = camera.viewFrom(R_ENTER);
    }
  }
```

Update BOTH render calls to use them — the paused/not-ready branch (main.js:170-174):

```js
      renderer.encode(enc, galaxyCam, sim.positionBuffer, sim.stellarBuffer, sim.N, simTime, systemScene);
```

and the main path (main.js:249):

```js
  renderer.encode(enc, galaxyCam, sim.positionBuffer, sim.stellarBuffer, sim.N, simTime, systemScene);
```

(The transition block must sit above the paused early-return so both paths see it.)

- [ ] **Step 5: Verify in the browser**

Run: `npm start`, open in Chrome:
- Click a bright star, scroll in. Below radius ~1.5 the system fades in: emissive sun (blooming, star-colored), planets as lit spheres with visible surface detail, moons around giants. The galaxy stays behind as a live backdrop that stops zooming.
- The selected star's own sprite is gone at full fade (the sun replaced it).
- Keep scrolling — you can approach the sun to rSys ≈ 0.5. Scroll out past 1.5 → clean return to the galaxy.
- `__select(i)` with various `i` until `__system.planets.length > 0` to check each kind renders (terran blue/white, gas banded).
- Pause → planets freeze. Un-pause → they orbit.
- Esc from deep inside → camera eases out, system unloads.
- Console: no WebGPU validation errors.

- [ ] **Step 6: Commit**

```bash
git add src/render/shaders/system.wgsl src/render/SystemRenderer.js src/render/RendererGPU.js src/main.js
git commit -m "feat: solar-system scene — impostor sun/planets/moons with crossfade zoom"
```

---

### Task 7: Rings, asteroid belt, orbit lines

**Files:**
- Modify: `src/render/shaders/system.wgsl`
- Modify: `src/render/SystemRenderer.js`

**Interfaces:**
- Consumes: buffers/counts already packed by `setSystem` (Task 6: `ringBuffer`/`numRings`, `beltBuffer`/`numBelt`, `orbitBuffer`/`numOrbits`).
- Produces: three more pipelines drawn inside the existing system pass.

- [ ] **Step 1: Add ring/belt/orbit entry points to system.wgsl**

Append:

```wgsl
// ── Planetary rings: flat annulus in the orbital plane ─────────────────────
struct RingVSOut {
    @builtin(position) pos : vec4f,
    @location(0) local : vec2f,
    @location(1) @interpolate(flat) inner : f32,
    @location(2) @interpolate(flat) outer : f32,
    @location(3) @interpolate(flat) seed : f32,
}

@vertex
fn vs_ring(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> RingVSOut {
    let rg = rings[ii];
    let c = orbitPos(rg.orbit.x, rg.orbit.y, rg.orbit.z);
    let local = QUAD[vi] * rg.size.y;
    var out : RingVSOut;
    out.pos = su.viewProj * vec4f(c + vec3f(local, 0.0), 1.0);
    out.local = local;
    out.inner = rg.size.x;
    out.outer = rg.size.y;
    out.seed = rg.orbit.w;
    return out;
}

@fragment
fn fs_ring(in : RingVSOut) -> @location(0) vec4f {
    let rad = length(in.local);
    if (rad < in.inner || rad > in.outer) { discard; }
    let t = (rad - in.inner) / (in.outer - in.inner);
    let bands = 0.35 + 0.65 * vnoise(vec3f(t * 24.0, in.seed * 97.0, 0.0));
    let edge = smoothstep(0.0, 0.06, t) * smoothstep(1.0, 0.94, t);
    let a = bands * edge * 0.55 * su.fade;
    return vec4f(vec3f(0.75, 0.68, 0.58) * a, a);
}

// ── Asteroid belt: instanced Keplerian point sprites ────────────────────────
struct BeltVSOut {
    @builtin(position) pos : vec4f,
    @location(0) quadUV : vec2f,
    @location(1) @interpolate(flat) bright : f32,
}

@vertex
fn vs_belt(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> BeltVSOut {
    let d0 = belt[ii * 2u];        // a, phase, rate, size
    let d1 = belt[ii * 2u + 1u];   // incl, node, bright, _
    let ang = d0.y + d0.z * su.time;
    let p = vec3f(cos(ang) * d0.x, sin(ang) * d0.x, sin(ang + d1.y) * d0.x * d1.x);
    var clip = su.viewProj * vec4f(p, 1.0);
    let corner = QUAD[vi];
    let sizePx = 1.0 + d0.w * 1.5;
    clip = vec4f(clip.xy + corner * (sizePx / su.resolution) * clip.w, clip.zw);
    var out : BeltVSOut;
    out.pos = clip;
    out.quadUV = corner;
    out.bright = d1.z;
    return out;
}

@fragment
fn fs_belt(in : BeltVSOut) -> @location(0) vec4f {
    let d2 = dot(in.quadUV, in.quadUV);
    if (d2 > 1.0) { discard; }
    let a = exp(-d2 * 2.5) * in.bright * 0.5 * su.fade;
    return vec4f(vec3f(0.70, 0.65, 0.55) * a, 1.0);
}

// ── Faint orbit guide lines ─────────────────────────────────────────────────
@vertex
fn vs_orbitline(@builtin(vertex_index) vi : u32,
                @builtin(instance_index) ii : u32) -> @builtin(position) vec4f {
    let a = orbitLines[ii].x;
    let ang = f32(vi) / 128.0 * 6.2831853;
    return su.viewProj * vec4f(cos(ang) * a, sin(ang) * a, 0.0, 1.0);
}

@fragment
fn fs_orbitline() -> @location(0) vec4f {
    return vec4f(vec3f(0.25, 0.40, 0.80) * 0.05 * su.fade, 1.0);
}
```

- [ ] **Step 2: Add the three pipelines in SystemRenderer.init()**

After `this.bodyPipeline = ...`:

```js
    const additive = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    };
    // Depth read-only: occluded by planets, but never self-sorting
    const depthRead = { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' };
    this.ringPipeline = device.createRenderPipeline({
      label: 'sys-rings',
      layout: this.pipelineLayout,
      vertex: { module: this.module, entryPoint: 'vs_ring' },
      fragment: { module: this.module, entryPoint: 'fs_ring', targets: [{ format: HDR, blend: premul }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: depthRead,
    });
    this.beltPipeline = device.createRenderPipeline({
      label: 'sys-belt',
      layout: this.pipelineLayout,
      vertex: { module: this.module, entryPoint: 'vs_belt' },
      fragment: { module: this.module, entryPoint: 'fs_belt', targets: [{ format: HDR, blend: additive }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: depthRead,
    });
    this.orbitPipeline = device.createRenderPipeline({
      label: 'sys-orbits',
      layout: this.pipelineLayout,
      vertex: { module: this.module, entryPoint: 'vs_orbitline' },
      fragment: { module: this.module, entryPoint: 'fs_orbitline', targets: [{ format: HDR, blend: additive }] },
      primitive: { topology: 'line-strip' },
      depthStencil: depthRead,
    });
```

- [ ] **Step 3: Draw them in encode()**

After the bodies draw, before `pass.end()`:

```js
    if (this.numRings > 0) {
      pass.setPipeline(this.ringPipeline);
      pass.draw(6, this.numRings);
    }
    if (this.numBelt > 0) {
      pass.setPipeline(this.beltPipeline);
      pass.draw(6, this.numBelt);
    }
    if (this.numOrbits > 0) {
      pass.setPipeline(this.orbitPipeline);
      pass.draw(129, this.numOrbits);
    }
```

- [ ] **Step 4: Verify in the browser**

Run: `npm start`, then `__select(i)` over a few indices until systems with `__system.belt` and a ringed gas giant (`__system.planets.some(p => p.rings)`) appear. Zoom in:
- Belt: a faint speckled ring of drifting particles at the belt radius, hiding behind planets that cross in front.
- Rings: banded annulus around the giant, occluded by the planet's near side, visible on both flanks.
- Orbit lines: faint blue circles under each planet.
- Note: `su.resolution` is written from the previous frame's size, so belt sprite size lags one frame after a window resize — acceptable.

- [ ] **Step 5: Commit**

```bash
git add src/render/shaders/system.wgsl src/render/SystemRenderer.js
git commit -m "feat: planetary rings, asteroid belt, orbit guide lines"
```

---

### Task 8: Star info chip + README

**Files:**
- Modify: `index.html`
- Modify: `src/main.js` (fill in `updateStarInfo`)
- Modify: `README.md`

**Interfaces:**
- Consumes: `selection` (Task 5) with `system.star.{class, mass, tempK}`, `system.planets.length`, `system.belt`.
- Produces: `#star-info` overlay element.

- [ ] **Step 1: Add the element and styles to index.html**

Element (next to the other HUD elements, e.g. after the `fps` element):

```html
  <div id="star-info" aria-live="polite"></div>
```

CSS (inside the existing `<style>` block, following the design-system variables):

```css
    #star-info {
      position: fixed;
      left: 16px;
      bottom: 16px;
      z-index: 100;
      padding: 10px 14px;
      background: var(--clr-panel);
      border-radius: 8px;
      box-shadow: var(--shadow-panel);
      color: var(--clr-star);
      font-size: 0.72rem;
      line-height: 1.6;
      opacity: 0;
      transition: opacity 0.3s var(--ease);
      pointer-events: none;
    }
    #star-info.visible { opacity: 1; }
    #star-info .cls {
      color: var(--clr-accent);
      font-family: 'Orbitron', sans-serif;
      letter-spacing: 0.08em;
    }
```

- [ ] **Step 2: Fill in updateStarInfo in main.js**

Replace the Task 5 stub:

```js
function updateStarInfo() {
  const el = document.getElementById('star-info');
  if (!el) return;
  if (!selection) { el.classList.remove('visible'); return; }
  const { star, planets, belt } = selection.system;
  const what = planets.length
    ? `${planets.length} planet${planets.length > 1 ? 's' : ''}`
    : 'barren system';
  el.innerHTML =
    `<span class="cls">CLASS ${star.class}</span> · ${star.mass.toFixed(2)} M☉ · ${Math.round(star.tempK).toLocaleString()} K<br>` +
    `${what}${belt ? ' · asteroid belt' : ''} — scroll in to visit · Esc to release`;
  el.classList.add('visible');
}
```

- [ ] **Step 3: Update README**

Add to the features section of `README.md` (match its existing tone/format):

```markdown
- **Solar-system zoom** — click any star to select it, then keep zooming in:
  below a threshold the view crossfades into that star's procedurally
  generated solar system (planets, moons, rings, asteroid belt), deterministic
  per star and rendered as ray-traced impostor spheres in a local AU-scale
  scene while the live galaxy stays behind as backdrop. Esc (or zooming out)
  returns to the galaxy.
```

- [ ] **Step 4: Verify**

Run: `npm start` — select stars: chip appears with class/mass/temp and planet count; "barren system" shows for zero-planet rolls; chip hides on Esc/restart/empty-sky click.

- [ ] **Step 5: Commit**

```bash
git add index.html src/main.js README.md
git commit -m "feat: star info chip; README solar-system zoom section"
```

---

### Task 9: End-to-end verification

**Files:** none (verification only; fix anything found and commit fixes).

- [ ] **Step 1: Unit tests**

Run: `node --test`
Expected: all PASS.

- [ ] **Step 2: Full browser pass**

Run `npm start`, then walk this checklist in Chrome (use Chrome DevTools MCP or manually; take screenshots at each ✓):

1. Galaxy loads at 25k, fps in HUD ≈ previous baseline (~120 on M-series), ΔE HUD unaffected over 60 s with a selection active.
2. Click star → ring + camera glide + info chip. Drag ≠ click. Empty-sky click deselects.
3. Zoom in: smooth crossfade at radius 1.5→0.8; sun blooms; backdrop galaxy stays alive but stops zooming; no star sprites blast through the camera.
4. Inside a system: planets orbit, moons orbit planets, belt drifts, rings occlude correctly against their planet; orbit lines visible but subtle.
5. Zoom out → clean return; ring reappears; Esc → full release.
6. Determinism: `__select(3000)`, `s1 = JSON.stringify(__system)`, `__select(1)`, `__select(3000)`, `JSON.stringify(__system) === s1` → `true`.
7. Collision preset: BHs unpickable; select a star in each galaxy; systems work mid-collision.
8. Pause inside a system → orbits freeze; resume → motion resumes.
9. Restart + N change to 100k → selection cleared, no validation errors, picking still works at 100k.
10. Console clean: no WebGPU validation warnings/errors across all of the above.

- [ ] **Step 3: Commit any fixes, then wrap up**

```bash
git status   # confirm clean
```

Report results (including fps numbers) and hand back for review.
