# Solar-System Zoom — Design

2026-07-08. Approved approach: **two-scene crossfade** (galaxy scene + separate local-scale system scene).

## Goal

Click a star → it highlights and the camera targets it. Keep scrolling in → below a radius
threshold, a procedurally generated solar system (planets, moons, asteroid belt) crossfades in
around the star, rendered in its own AU-scale coordinate space. Scroll out → back to the galaxy.
No modes; fully reversible. The N-body sim keeps running; the system rides the star's live
position. Deterministic per star: revisiting regenerates the identical system.

## Why two scenes

A real system is ~10⁻⁹ of galaxy scale — no single f32 coordinate space survives that. The system
scene has the star at origin, 1 unit = 1 AU. The galaxy camera's radius maps into system-space
radius by a power law; the galaxy keeps rendering behind as a live parallax backdrop with its
radius pinned at the threshold.

## Components

### 1. Selection — `src/ui/StarSelection.js` (new)

- **Position snapshot**: `main.js` copies `posTransfer` (vec4/particle: x,y,z,mass) into a retained
  `lastPositions` array after each `readPositions()` resolves, before the buffer transfers to the
  tree worker (main.js:255-259). ~400 KB memcpy per rebuild at 25k — negligible; revisit if 500k.
- **Pick**: on click (pointerup with < 4 px movement), project all N snapshot positions through
  `camera.viewProj`, collect the ≤4 best candidates within ~15 px of the click. One staged GPU
  readback of their 16-byte stellar records; reject gas (`stellarMass === -1`) and the central BHs
  (`stellarMass === 0`); remnants are pickable like any star. Best valid candidate wins; none → deselect.
  O(N) on click only.
- **Seed + star props**: the winning record `[stellarMass, birthTime, seed, dustWeight]`
  (InitialConditions.js:2) is kept; `seed`/`stellarMass`/`birthTime` are intrinsic and permute with
  the particle, so the system seed is stable across tree rebuilds.
- **Permutation tracking**: when `pendingTree` is applied (main.js:204), remap
  `selectedIdx` by scanning `perm` for `perm[new] === old` — same pattern as `centralIdx`, one
  O(N) scan per rebuild.
- **Deselect**: Esc, clicking empty space, or restart (`init()` clears selection).

### 2. Camera & transition — `src/gpu/Camera.js`, `main.js`

- Min radius clamp becomes a field: 2 normally, 0.02 while a star is selected.
- While selected, `camera._goal.target` follows the star's latest snapshot position every frame;
  existing damping (Camera.js:69-73) smooths the few-frame staleness.
- `R_enter = 1.5` galaxy units. `systemFade = smoothstep(1.5 → 0.8)` of camera radius.
- System-space camera: same theta/phi/target-relative orientation, radius
  `rSys = 0.5 · (r / 0.02)^1.27` (power law spanning 0.5–120 AU over r ∈ 0.02–1.5).
- Galaxy backdrop: while `r < R_enter`, the galaxy renders with its radius clamped at `R_enter`
  (orientation and target still live) so neighboring sprites don't fly through the camera.

### 3. Generation — `src/system/SystemGen.js` (new, pure)

`generateSystem(stellarMass, birthTime, seed) → system` — mulberry32 PRNG seeded from the three
values' float bits. Plausible-ish rules:

- Star: luminosity `L ≈ M^3.5`, temperature/color from the same mass→blackbody mapping the star
  sprites use; visual radius from mass.
- Frost line `≈ 2.7·√L` AU; habitable zone `≈ [0.95, 1.4]·√L` AU.
- Planet count: 0–8, weighted down for hot massive stars (scorched, sparse), compressed tight
  orbits for red dwarfs.
- Spacing: first orbit `0.3–0.5·√L` AU, each next `×1.4–2.0` (geometric with jitter).
- Types: inside frost line → rocky (scorched / venusian / terran-with-water-and-clouds if in HZ /
  icy beyond); outside → gas giant (banded, 35% rings) or ice giant.
- Moons: giants 1–4, rockies 0–1, orbit radii in planet-radius multiples.
- Belt: 70% of systems, placed in the widest orbit gap (or beyond the rockies), ~3000 particles
  with radial spread ±15%, small inclination scatter.
- Motion: analytic circular orbits, `angle = phase + simTime · 2π/a^1.5 · timeScale`. Zero physics
  cost; pause freezes orbits (consistent with the sim).
- Output: plain object `{star, planets:[{a, phase, radius, type, palette, rings?, moons[]}], belt?}`.

### 4. Rendering — `src/render/SystemRenderer.js` + `src/render/shaders/system.wgsl` (new)

One additional render pass between the galaxy scene pass and bloom, drawing into the same HDR
`sceneView` (`loadOp: 'load'`) with its **own depth texture** (depth24plus, cleared per frame) —
the galaxy pass stays depth-free. Bloom + AgX composite then apply to both scenes for free (the
emissive sun blooms like a star).

- **All bodies are impostor spheres**: billboard quads, analytic ray-sphere in the fragment shader,
  `@builtin(frag_depth)` from the hit point for correct mutual occlusion, alpha-blended over the
  backdrop.
- **Surfaces**: 3D fbm noise evaluated at the sphere-surface point — continents/oceans/ice caps +
  a drifting cloud layer for rockies; latitude bands + storm spots for giants; fresnel rim
  atmosphere tint. **This shading is a function of the surface point, so it ports unchanged to a
  terrain-mesh pipeline for future surface-level zoom.**
- **Sun**: emissive HDR blackbody color, limb darkening, unlit.
- **Rings**: annulus quad in the orbital plane, banded noise, alpha-blended, depth-tested.
- **Belt**: instanced point sprites, Keplerian position computed in the vertex shader from orbital
  elements + time; additive, depth-tested (read-only) so they pass behind planets.
- **Orbit lines**: faint line-strip circles (128 segments), additive, for readability.
- Everything scales with `systemFade` for the crossfade.

### 5. Galaxy renderer changes — `RendererGPU.js`, `render.wgsl`

- Uniform buffer grows (currently exactly full at 176 B, RendererGPU.js:50): add `selectedIdx`
  (u32) and `systemFade` (f32).
- Selected-star highlight: one extra `draw(6, 1)` ring sprite reading `particles[selectedIdx]`.
- The selected star's own sprite fades out with `1 − systemFade` (it's *become* the system's sun).
- `encode()` gains an optional system-scene argument; when present and `systemFade > 0`, the
  system pass is encoded with the system camera.

### 6. UI — `index.html`, `main.js`

- Info chip (HTML overlay): spectral class (OBAFGKM from mass), mass, planet count, "Esc to
  deselect". Appears on selection, updates on entering the system.
- Debug hooks: `window.__system` (current system object), `window.__select(idx)`.

## Edge cases

- **Star recycles (supernova) mid-visit**: system was generated from the record read at selection —
  it persists for the visit. Reselecting afterwards yields a new system (it's a new star). Known
  v1 quirk.
- **Zero planets rolled**: sun + belt (or barren) still renders, info chip says "no planets" — not
  a bug state.
- **Collision preset**: both central BHs unpickable via `stellarMass === 0`.
- **Restart / preset switch**: selection cleared, min-radius clamp restored, system scene dropped.
- **Pause**: orbital motion freezes with `simTime` — consistent, not a bug.

## Verification

- Manual: click stars at 25k/100k, enter/exit systems repeatedly, Esc, restart, Collision preset
  (BH unpickable), pause inside a system, check fps unchanged at 25k and ΔE HUD unaffected.
- Determinism: `__select(i)` twice → identical `__system` JSON.
- Picking correctness: click empty sky → deselect; click near a gas-dust lane → never selects gas.

## Out of scope (roadmap)

- Surface-level planet zoom (terrain mesh LOD) — the impostor shading function is the reusable
  piece.
- Elliptical orbits, moons of moons, comets, binary stars.
- Selecting the central BH (accretion-disk easter egg).
