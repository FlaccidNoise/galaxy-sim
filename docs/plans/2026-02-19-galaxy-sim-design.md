# Galaxy Sim — Design Document
**Date:** 2026-02-19
**Status:** Approved

## Goal

A browser-based N-body galaxy simulation focused on physics accuracy and interactivity. Primary use: studying galaxy formation and evolution over time. Star count is tunable from ~1k up to ~100k; architecture supports a future upgrade to a Particle-Mesh solver for 500k–1M particles.

---

## Architecture

Particle positions and velocities live entirely on the GPU throughout the simulation loop. The CPU only writes initial conditions and reads/writes UI state.

```
┌─────────────────────────────────────────────┐
│               GPU Memory                    │
│  ┌──────────────────────────────────────┐   │
│  │  StorageBuffer: positions + velocities│   │
│  └─────┬──────────────────────┬─────────┘   │
│        │                      │              │
│  ┌─────▼──────┐        ┌──────▼─────────┐   │
│  │  Compute   │        │  Render Shader │   │
│  │  (WGSL)    │        │  (Three.js)    │   │
│  │  N-body    │        │  Points mesh   │   │
│  └────────────┘        └────────────────┘   │
└─────────────────────────────────────────────┘
         ▲
    CPU controls:
    timestep, pause,
    restart, GUI params
```

**Stack:**
- Three.js r170 (WebGPU renderer) — rendering, camera, postprocessing
- WGSL compute shaders — N-body physics (hand-written)
- lil-gui — parameter panel
- ES modules via CDN importmap — zero build toolchain
- `npx serve .` to run locally

---

## Physics Engine

### Integrator
Leapfrog (Störmer-Verlet) — symplectic, conserves energy over long timescales.

Each step:
1. Half-kick: `v += 0.5 * a * dt`
2. Drift: `x += v * dt`
3. Recompute forces
4. Half-kick: `v += 0.5 * a * dt`

### Force Computation
Direct N-body summation in WGSL. Each GPU thread handles one particle, loops over all N others:

```
F = G * m₁ * m₂ / (r² + ε²)
```

Softening parameter `ε` prevents infinite force at near-zero separation.

### GPU Memory Layout
Two ping-pong buffers (A and B) to avoid read/write races:
```
Buffer: [x, y, z, mass, vx, vy, vz, _pad] × N   (f32 × 8 per particle)
```
Alternates A→B→A→B each step.

### Initial Conditions (CPU-side)

**Galaxy Disk (default):**
- Exponential surface density: `ρ ∝ e^(-r/r₀)`
- Gaussian scale height for 3D structure
- Circular velocities derived from enclosed mass + dark matter halo term
- Small random velocity dispersion on top

**Uniform Sphere:**
- Random positions in a ball, zero net angular momentum

**Two-Galaxy Collision:**
- Two offset disks on intersecting trajectories

### Tunable Parameters

| Parameter | Effect |
|---|---|
| `N` | Star count (restart required) |
| `G` | Gravitational constant |
| `dt` | Timestep (accuracy vs. speed) |
| `ε` | Softening length |
| `diskRadius` | Galaxy disk extent |
| `diskHeight` | Disk scale height |
| `centralMass` | Central black hole mass multiplier |
| `velDispersion` | Random velocity scatter |

### Upgrade Path to Particle-Mesh
`NBodyGPU` exposes a single `step()` method. A future `PMBodyGPU` class with the same interface can replace it — nothing else in the codebase changes.

---

## Renderer

**Particle rendering:** Three.js `Points` with a custom WGSL vertex shader reading directly from the physics StorageBuffer.

**Star coloring (switchable):**
- **Velocity-based (default):** Fast = blue-white, slow = red-orange. Shows dynamics clearly.
- **Mass-based:** Massive = bright blue-white (O/B type), small = dim red (M type).

**Point size:** Scales with mass. Dense core visually stands out.

**Postprocessing:**
- Bloom (UnrealBloomPass) — galaxy glow effect
- ACES filmic tone mapping — prevents overexposure

**Camera:**
- OrbitControls — left-drag rotate, scroll zoom, right-drag pan
- Default: ~15° tilt above disk plane

---

## UI Controls

Single collapsible lil-gui panel:

```
▼ Simulation
    Play / Pause          [button]
    Speed                 [0.1× – 10×]
    Reset                 [button]

▼ Initial Conditions
    Preset          [Galaxy Disk | Sphere | Collision]
    N (stars)       [1k – 100k, log scale]
    Disk Radius     [slider]
    Disk Height     [slider]
    Central Mass    [slider]
    Vel Dispersion  [slider]

▼ Physics
    G               [slider]
    Softening ε     [slider]
    Timestep dt     [slider]

▼ Display
    Color Mode      [Velocity | Mass]
    Bloom Intensity [slider]
    Point Size      [slider]
```

Changing Initial Conditions shows a "Restart required" warning. Reset re-generates and uploads a fresh particle set.

---

## File Structure

```
~/CodingProjects/galaxy-sim/
├── index.html                  # Entry point, importmap, canvas
├── package.json                # type: module, scripts.start = npx serve .
│
├── src/
│   ├── main.js                 # Wiring: init, animation loop
│   │
│   ├── physics/
│   │   ├── NBodyGPU.js         # WebGPU compute pipeline, ping-pong buffers, step()
│   │   ├── InitialConditions.js # Galaxy disk, sphere, collision generators
│   │   └── shaders/
│   │       └── nbody.wgsl      # WGSL compute shader (force + integrate)
│   │
│   ├── render/
│   │   ├── Renderer.js         # Three.js WebGPU renderer, bloom, tone mapping
│   │   └── ParticleSystem.js   # StorageBufferAttribute, Points mesh, color shader
│   │
│   └── ui/
│       └── Controls.js         # lil-gui panel, parameter bindings, restart logic
│
└── docs/
    └── plans/
        └── 2026-02-19-galaxy-sim-design.md
```

---

## Running

```bash
cd ~/CodingProjects/galaxy-sim
npx serve .
# open http://localhost:3000
```

No install, no build step, no config.
