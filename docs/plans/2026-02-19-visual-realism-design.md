# Visual Realism Design

**Goal:** Transform the galaxy sim from colored dots into a cinematic spiral galaxy with spectral star colors, luminosity-scaled glowing sprites, and realistic arm structure.

**Approach:** Option B — custom ShaderMaterial for star rendering, spectral coloring from stellar mass, logarithmic spiral arm seeding with two stellar populations, configurable arm count.

**Files touched:**
- `src/render/StarShader.js` — new GLSL vertex + fragment shaders
- `src/render/ParticleSystem.js` — use ShaderMaterial, add size attribute, spectral color mode
- `src/physics/InitialConditions.js` — spiral arm seeding, two populations, armCount param
- `src/ui/Controls.js` — add armCount slider (1–6), default colorMode to Spectral

---

## Section 1: Custom Star Shader (`src/render/StarShader.js`)

Replace `PointsMaterial` with a `ShaderMaterial`.

**Vertex shader:**
- Receives per-particle `size` attribute (derived from luminosity)
- Sets `gl_PointSize = size * uPixelRatio` so stars scale correctly on retina displays
- Passes `vColor` to fragment shader

**Fragment shader:**
- Uses `gl_PointCoord` (0–1 UV within the point sprite square)
- Computes distance from center: `d = length(gl_PointCoord - 0.5) * 2.0`
- Discards fragments outside unit circle (`d > 1.0`) — round sprite, not square
- Gaussian glow: `intensity = exp(-d * d * 3.0)`
- Core: bright white-hot center blending to spectral color at edge
  - `color = mix(vColor, vec3(1.0), pow(1.0 - d, 3.0))`
- Final alpha: `intensity * opacity` — transparent falloff, works with AdditiveBlending

**Uniforms:** `uPixelRatio` (float)

**Size computation** (CPU side, in `updateFromStaging`):
- `L = pow(mass, 3.5)` (main-sequence mass-luminosity relation)
- `visualSize = clamp(log(L + 1.0) * 2.0 + 1.0, 1.0, 12.0)`
- Central mass (particle 0): fixed size 16.0

---

## Section 2: Spectral Colors

Computed CPU-side in `ParticleSystem.updateFromStaging()` as a new `'Spectral'` color mode (default).

Stellar classification by mass (M☉):

| Type | Mass range | RGB (linear approx) |
|------|-----------|----------------------|
| O    | >16       | `(0.61, 0.69, 1.0)` — blue-white |
| B    | 2–16      | `(0.67, 0.75, 1.0)` — blue |
| A    | 1.4–2     | `(0.79, 0.85, 1.0)` — white-blue |
| F    | 1.04–1.4  | `(0.97, 0.97, 1.0)` — white-yellow |
| G    | 0.8–1.04  | `(1.0, 0.96, 0.92)` — yellow (Sun) |
| K    | 0.45–0.8  | `(1.0, 0.82, 0.63)` — orange |
| M    | <0.45     | `(1.0, 0.80, 0.44)` — red-orange |

Central black hole (particle 0): `(0.9, 0.95, 1.0)` bright white-blue, fixed large size.

GUI color mode dropdown: **Spectral** (new default), Velocity, Mass.

---

## Section 3: Spiral Arm Initial Conditions

Replaces `galaxyDisk` in `InitialConditions.js`.

**Parameters added:** `armCount` (default 2), `armTightness` (default 0.35, not exposed to GUI).

**Algorithm:**
1. Assign each star to an arm: `armIndex = i % armCount`, arm angle offset = `armIndex * (2π / armCount)`
2. Sample radius from existing exponential CDF (`r = -r₀ * ln(1 - u)`)
3. Compute arm centerline angle: `θ_arm = log(r / r₀) / armTightness + armOffset`
4. Scatter star around centerline: `θ = θ_arm + gaussRandom() * scatter(r)` where `scatter(r) = 0.15 + r / diskRadius * 0.4` (arms diffuse outward)
5. Population split by arm vs. bulge:
   - **Arm stars** (scatter within 0.5 rad of arm centerline): mass from upper IMF `U(1.0, 20.0)` power-law → O/B/A → blue-white
   - **Bulge/inter-arm stars**: mass from lower IMF `U(0.2, 1.2)` → G/K/M → yellow/orange/red
6. Central bulge overdensity: 20% of stars placed in tight Gaussian at origin (bulge population), `r ~ |gaussRandom()| * diskRadius * 0.15`

`twoGalaxyCollision` calls `galaxyDisk` internally — collision preset automatically gets spiral arms.

---

## Section 4: Arm Count Control

New slider in Controls.js under Initial Conditions:
- `armCount`: range 1–6, step 1, default 2
- `onChange`: calls `_scheduleRestart()` (debounced auto-restart, already implemented)

Added to `this.state` and passed through `icParams` in `main.js`.

---

## Non-Goals (deferred)

- Dust lane mesh overlay
- Skybox / Milky Way background texture
- Barnes-Hut or Particle-Mesh for N > 50k
- Leapfrog integrator
