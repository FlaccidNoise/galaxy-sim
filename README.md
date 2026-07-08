# Galaxy Sim

Real-time N-body galaxy simulation in the browser. Pure WebGPU — physics and
rendering share GPU buffers; particle state never touches the CPU during the
frame loop.

```
npm start        # serves on http://localhost:3000 (or: npx serve .)
```

Requires a WebGPU browser (Chrome 113+, Safari 26+).

## What's simulated

- **Gravity**: Barnes-Hut octree, O(N log N). Morton-sorted particles with
  bucketed leaves; tree *topology* rebuilds asynchronously in a Web Worker
  while node masses re-sum on the GPU every step, so forces always follow the
  live state. Plummer softening, sub-softening monopole acceptance.
- **Dark matter**: analytic Hernquist halo per galaxy (flat rotation curves),
  centred on a smoothed track of each galaxy's central black hole — colliding
  galaxies carry their halos and merge properly.
- **Equilibrium initial conditions**: exponential disk + sech² vertical
  profile + Hernquist bulge; circular velocities from the measured enclosed
  mass; velocity dispersion set by Toomre Q with asymmetric drift.
- **Gas & star formation**: ~30% of the disk is gas that cools (drag on
  non-orbital motion), settles thin, and converts to new stars where dense.
  Massive stars supernova and return their matter to the gas phase.
- **Stellar lifecycle**: Kroupa IMF masses; blackbody (Planckian-locus)
  colors; main sequence → red giant → supernova / white dwarf — derived
  statelessly in the render shader from mass + birth time.
- **Solar-system zoom** — click any star to select it, then keep zooming in:
  below a threshold the view crossfades into that star's procedurally
  generated solar system (planets, moons, rings, asteroid belt), deterministic
  per star and rendered as ray-traced impostor spheres in a local AU-scale
  scene while the live galaxy stays behind as backdrop. Esc (or zooming out)
  returns to the galaxy.

## Rendering

HDR (rgba16float) pipeline: instanced sprites vertex-pulled from the physics
buffer, dust extinction pass (multiplicative, rides the gas), H-II emission
around hot O stars, procedural starfield, mip-chain bloom (Karis-averaged),
AgX tone mapping.

## Controls

- Drag to rotate, right-drag to pan, scroll to zoom, **H** toggles the panel.
- **Initial Conditions**: preset (disk / sphere / collision), N up to 500k,
  Toomre Q, arm count.
- **Physics**: softening, timestep, BH opening angle θ, halo mass/scale, gas
  cooling, star-formation efficiency and threshold.
- **Display**: aging rate (Myr per time unit; 0 freezes stellar evolution),
  exposure, bloom, dust, nebulae.
- HUD (bottom-left) shows fps, N, and total-energy drift. Negative drift is
  gas cooling; slow positive drift at low N is two-body relaxation (each
  particle stands in for ~10⁴ stars) and shrinks as N grows.

## Layout

```
src/main.js                  frame loop, worker orchestration
src/gpu/Camera.js            damped orbit camera
src/physics/NBodyGPU.js      compute pipelines (force, refresh, SF, sort)
src/physics/mortonTree.js    Morton codes, radix sort, BFS bucketed tree
src/physics/bhWorker.js      async topology rebuild worker
src/physics/shaders/         WGSL compute kernels
src/render/RendererGPU.js    HDR render graph
src/render/shaders/          WGSL render kernels
docs/plans/                  historical design documents (pre-rewrite)
```
