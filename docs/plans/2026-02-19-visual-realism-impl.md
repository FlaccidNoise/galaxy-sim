# Visual Realism Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the galaxy sim into a cinematic spiral galaxy with spectral star colors, luminosity-scaled glowing sprites, logarithmic spiral arm seeding, and configurable arm count.

**Architecture:** Four independent changes layered together: (1) a new `StarShader.js` with GLSL vertex/fragment shaders replaces `PointsMaterial`; (2) `ParticleSystem.js` gains a `size` vertex attribute and spectral color computation; (3) `InitialConditions.js` gets a new spiral arm seeding algorithm with two stellar populations; (4) `Controls.js` adds an `armCount` slider. No physics changes — purely visual and IC improvements.

**Tech Stack:** Three.js r170 (ShaderMaterial, GLSL), lil-gui 0.19, ES modules, WebGPU compute (unchanged), WebGL rendering.

---

### Task 1: Star Shader (`src/render/StarShader.js`)

**Files:**
- Create: `src/render/StarShader.js`

**Step 1: Create the file with vertex and fragment shader strings**

```js
// src/render/StarShader.js

export const vertexShader = /* glsl */`
  attribute float size;
  varying vec3 vColor;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const fragmentShader = /* glsl */`
  varying vec3 vColor;

  void main() {
    // Distance from center of point sprite, in [0, 1]
    float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
    if (d > 1.0) discard;

    // Gaussian glow falloff
    float intensity = exp(-d * d * 3.0);

    // Hot white core blending to spectral color at edge
    vec3 col = mix(vColor, vec3(1.0), pow(max(0.0, 1.0 - d), 3.0));

    gl_FragColor = vec4(col, intensity * 0.9);
  }
`;
```

**Step 2: Verify file exists**

```bash
cat src/render/StarShader.js
```
Expected: both shader strings printed.

**Step 3: Commit**

```bash
git add src/render/StarShader.js
git commit -m "feat: GLSL star shaders — round gaussian glow sprites"
```

---

### Task 2: ParticleSystem — ShaderMaterial + size attribute + spectral colors

**Files:**
- Modify: `src/render/ParticleSystem.js` (full rewrite of constructor, updateFromStaging, color helpers)

**Context:** The current file uses `THREE.PointsMaterial`. We replace it with `THREE.ShaderMaterial` using the shaders from Task 1. We also add a `size` `Float32Array` vertex attribute and a new `spectralColor()` helper.

**Step 1: Replace the import and add size attribute to constructor**

Replace the top of `ParticleSystem.js` so it imports the shaders and creates the ShaderMaterial:

```js
import * as THREE from 'three';
import { vertexShader, fragmentShader } from './StarShader.js';

export class ParticleSystem {
  constructor(scene, N) {
    this.N = N;
    this.positions = new Float32Array(N * 3);
    this.colors    = new Float32Array(N * 3);
    this.sizes     = new Float32Array(N);

    const geometry = new THREE.BufferGeometry();
    this.posAttr  = new THREE.BufferAttribute(this.positions, 3);
    this.colAttr  = new THREE.BufferAttribute(this.colors, 3);
    this.sizeAttr = new THREE.BufferAttribute(this.sizes, 1);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.colAttr.setUsage(THREE.DynamicDrawUsage);
    this.sizeAttr.setUsage(THREE.DynamicDrawUsage);

    geometry.setAttribute('position', this.posAttr);
    geometry.setAttribute('color',    this.colAttr);
    geometry.setAttribute('size',     this.sizeAttr);

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.points = new THREE.Points(geometry, material);
    scene.add(this.points);

    this._stagingBuffer    = null;
    this._device           = null;
    this._physicsByteSize  = null;
  }
```

**Step 2: Update `updateFromStaging` to compute sizes and spectral colors**

Replace the `updateFromStaging` method:

```js
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

      const mass  = raw[src + 3];
      const vx = raw[src + 4], vy = raw[src + 5], vz = raw[src + 6];
      const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);

      if (colorMode === 'Spectral') {
        spectralColor(mass, this.colors, dst);
      } else if (colorMode === 'Velocity') {
        velocityColor(speed, speedScale, this.colors, dst);
      } else {
        massColor(mass, this.colors, dst);
      }

      // Size from luminosity: L ∝ M^3.5, log-scaled to pixel range
      if (i === 0) {
        // Central black hole / core — fixed large bright size
        this.sizes[i] = 16.0;
      } else {
        const L = Math.pow(Math.max(mass, 0.1), 3.5);
        this.sizes[i] = Math.min(Math.max(Math.log(L + 1.0) * 2.0 + 1.0, 1.0), 12.0);
      }
    }

    this._stagingBuffer.unmap();

    this.posAttr.needsUpdate  = true;
    this.colAttr.needsUpdate  = true;
    this.sizeAttr.needsUpdate = true;
  }
```

**Step 3: Remove `setPointSize` (no longer meaningful with per-particle sizes) and add `spectralColor` helper**

Remove the `setPointSize` method entirely.

Add `spectralColor` function below the class, alongside existing `velocityColor` and `massColor`:

```js
/**
 * Map stellar mass (M☉) to spectral type color.
 * O: >16  blue-white   B: 2-16  blue      A: 1.4-2  white-blue
 * F: 1.04-1.4  white   G: 0.8-1.04  yellow  K: 0.45-0.8  orange  M: <0.45  red-orange
 */
function spectralColor(mass, out, i) {
  if (mass > 16)       { out[i]=0.61; out[i+1]=0.69; out[i+2]=1.00; } // O
  else if (mass > 2)   { out[i]=0.67; out[i+1]=0.75; out[i+2]=1.00; } // B
  else if (mass > 1.4) { out[i]=0.79; out[i+1]=0.85; out[i+2]=1.00; } // A
  else if (mass > 1.04){ out[i]=0.97; out[i+1]=0.97; out[i+2]=1.00; } // F
  else if (mass > 0.8) { out[i]=1.00; out[i+1]=0.96; out[i+2]=0.92; } // G
  else if (mass > 0.45){ out[i]=1.00; out[i+1]=0.82; out[i+2]=0.63; } // K
  else                 { out[i]=1.00; out[i+1]=0.80; out[i+2]=0.44; } // M
}
```

**Step 4: Verify in browser**

Run `npx serve .` and open `http://localhost:<port>`. Stars should render as round glowing sprites, varying in size. Spectral mode should show a mix of colors even before IC changes.

**Step 5: Commit**

```bash
git add src/render/ParticleSystem.js
git commit -m "feat: ShaderMaterial with per-star glow, size-by-luminosity, spectral colors"
```

---

### Task 3: Spiral Arm Initial Conditions

**Files:**
- Modify: `src/physics/InitialConditions.js` — replace `galaxyDisk` body

**Context:** Current `galaxyDisk` seeds stars uniformly in angle. We replace it with logarithmic spiral arm seeding. The existing `gaussRandom()` helper, `uniformSphere`, and `twoGalaxyCollision` are unchanged.

**Step 1: Replace the `galaxyDisk` function body**

The signature stays identical (`export function galaxyDisk(params)`). Replace the entire body:

```js
export function galaxyDisk(params) {
  const {
    N = 10000,
    diskRadius = 50.0,
    diskHeight = 2.0,
    centralMass = 1000.0,
    starMass = 1.0,   // unused directly — mass now sampled per-star
    velDispersion = 0.5,
    G = 1.0,
    armCount = 2,
    armTightness = 0.35,
  } = params;

  const data = new Float32Array(N * STRIDE);

  // 20% of stars form a tight central bulge (old, low-mass, red/yellow)
  const nBulge = Math.floor(N * 0.20);
  const nArm   = N - 1 - nBulge; // remaining go in arms; index 0 reserved for central mass

  for (let i = 1; i < N; i++) {
    const base = i * STRIDE;
    const isBulge = i <= nBulge;

    let r, theta;

    if (isBulge) {
      // Tight Gaussian bulge
      r     = Math.abs(gaussRandom()) * diskRadius * 0.15;
      theta = Math.random() * 2 * Math.PI;
    } else {
      // Exponential radial CDF (same as before)
      const r0 = diskRadius / 3.0;
      const u  = Math.random();
      r = -r0 * Math.log(1.0 - u * (1.0 - Math.exp(-diskRadius / r0)));

      // Assign to an arm
      const armIndex  = (i - nBulge - 1) % armCount;
      const armOffset = armIndex * (2 * Math.PI / armCount);

      // Logarithmic spiral centerline angle at radius r
      const r0Spiral  = diskRadius * 0.1;
      const thetaArm  = Math.log(Math.max(r, r0Spiral) / r0Spiral) / armTightness + armOffset;

      // Scatter around arm — wider at larger radii
      const scatter = 0.15 + (r / diskRadius) * 0.40;
      theta = thetaArm + gaussRandom() * scatter;
    }

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);
    const z = gaussRandom() * diskHeight;

    // Stellar mass: arm stars → upper IMF (hot blue), bulge → lower IMF (cool red)
    // Simple power-law approximation: draw uniform, map to range
    let mass;
    if (isBulge) {
      // Lower IMF: 0.2 – 1.2 M☉, weighted toward low end
      mass = 0.2 + Math.pow(Math.random(), 2.0) * 1.0;
    } else {
      // Check angular distance from arm centerline
      const armIndex  = (i - nBulge - 1) % armCount;
      const armOffset = armIndex * (2 * Math.PI / armCount);
      const r0Spiral  = diskRadius * 0.1;
      const thetaArm  = Math.log(Math.max(r, r0Spiral) / r0Spiral) / armTightness + armOffset;
      const dTheta    = Math.abs(((theta - thetaArm) % (2 * Math.PI) + Math.PI) % (2 * Math.PI) - Math.PI);

      if (dTheta < 0.4) {
        // In arm: upper IMF — young hot stars, 1 – 20 M☉, weighted toward low end
        mass = 1.0 + Math.pow(Math.random(), 2.5) * 19.0;
      } else {
        // Inter-arm disk: mixed, 0.3 – 2.0 M☉
        mass = 0.3 + Math.pow(Math.random(), 1.5) * 1.7;
      }
    }

    // Circular velocity (same physics as before)
    const massEnc = centralMass + mass * N * (r / diskRadius);
    const vc = r > 0.01 ? Math.sqrt(G * massEnc / r) : 0;

    const vx = -vc * Math.sin(theta) + gaussRandom() * velDispersion;
    const vy =  vc * Math.cos(theta) + gaussRandom() * velDispersion;
    const vz =  gaussRandom() * velDispersion * 0.2;

    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = mass;
    data[base + 4] = vx;
    data[base + 5] = vy;
    data[base + 6] = vz;
    data[base + 7] = 0;
  }

  // Particle 0: central mass (black hole / dense core)
  data[0] = 0; data[1] = 0; data[2] = 0;
  data[3] = centralMass;
  data[4] = 0; data[5] = 0; data[6] = 0;
  data[7] = 0;

  return data;
}
```

**Step 2: Verify in browser**

Restart the sim. The galaxy should show distinct blue spiral arms wrapping a warm gold/orange central bulge. Try changing arm count (after Task 4 adds the slider) to confirm parameter threads through.

**Step 3: Commit**

```bash
git add src/physics/InitialConditions.js
git commit -m "feat: logarithmic spiral arm seeding with two stellar populations"
```

---

### Task 4: Arm Count Slider + Wire armCount through

**Files:**
- Modify: `src/ui/Controls.js`
- Modify: `src/main.js`

**Context:** `armCount` needs to be in `controls.state`, shown as a slider (1–6), auto-restart on change, and passed through `icParams` in `main.js`.

**Step 1: Add `armCount` to `controls.state` and add slider in `Controls.js`**

In `this.state`, add after `velDispersion`:
```js
armCount: 2,
```

In `_build()`, after the `velDispersion` slider line, add:
```js
icFolder.add(this.state, 'armCount', 1, 6, 1).name('Arm Count')
  .onChange(() => this._scheduleRestart());
```

**Step 2: Pass `armCount` through `icParams` in `main.js`**

In `init(params)`, find the `icParams` object and add:
```js
armCount: params.armCount,
```

Also add `armCount` to `defaultState` at the bottom of `main.js`:
```js
armCount: 2,
```

**Step 3: Update `colorMode` default in `Controls.js`**

In `this.state`, change:
```js
colorMode: 'Velocity',
```
to:
```js
colorMode: 'Spectral',
```

Also update the dropdown options in `_build()`:
```js
dispFolder.add(this.state, 'colorMode', ['Spectral', 'Velocity', 'Mass']).name('Color Mode');
```

**Step 4: Remove the `pointSize` slider** (no longer meaningful — sizes are per-particle from luminosity)

Remove these lines from `_build()`:
```js
dispFolder.add(this.state, 'pointSize', 0.05, 2).name('Point Size').onChange(v => {
  this.particles.setPointSize(v);
});
```

And remove `pointSize: 0.3` from `this.state`.

**Step 5: Verify in browser**

- Arm Count slider appears, changing it restarts with new arm count
- Color mode defaults to Spectral
- Blue spiral arms visible against gold bulge

**Step 6: Commit**

```bash
git add src/ui/Controls.js src/main.js
git commit -m "feat: arm count slider, spectral default color mode, wire armCount through IC"
```
