# Galaxy Sim

Real-time N-body galaxy simulation in the browser. Pure WebGPU — physics and
rendering share GPU buffers; particle state never touches the CPU during the
frame loop.

```bash
git clone https://github.com/FlaccidNoise/galaxy-sim.git
cd galaxy-sim
npm start        # serves on http://localhost:3000 (or: npx serve .)
```

Requires a WebGPU browser (Chrome 113+, Safari 26+). There are no dependencies
to install — `npm start` just runs `npx serve .`, and any static file server
works equally well.

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
- **Solar-system zoom**: click any star to select it, then keep zooming in:
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

- Drag to rotate, right-drag to pan, scroll to zoom.
- **Controls** (top right) or **H** opens the control sheet. On a wide screen
  the rail on the right lists its sections and About. Click one to open the
  sheet at that section. **Reset** restarts the simulation.
- Click a star to select it and keep zooming to enter its solar system. Inside,
  click a planet or moon to orbit it; the zoom floor keeps the camera outside
  the body. Esc steps back one level at a time: planet focus, then the open
  sheet, then the selected star. Clicking empty sky does the same.
- While you are inside a solar system the galaxy stops, and the planets keep
  their own pace whatever the galaxy's speed or pause.
- Touch: one finger rotates, two fingers pinch to zoom and drag to pan, tap selects a star.
- **Initial Conditions**: preset (disk / sphere / collision), N up to 500k,
  Toomre Q, arm count.
- **Physics**: softening, timestep, BH opening angle θ, halo mass/scale, gas
  cooling, star-formation efficiency and threshold.
- **Display**: aging rate (Myr per time unit; 0 freezes stellar evolution),
  exposure, bloom, dust, nebulae.
- The caption (bottom left) shows the star count and elapsed time in Gyr.
  The readout (bottom right) shows fps, N, and total-energy drift. Negative drift is
  gas cooling; slow positive drift at low N is two-body relaxation (each
  particle stands in for ~10⁴ stars) and shrinks as N grows.

## Layout

```
src/main.js                  frame loop, worker orchestration, HUD
src/ui/Controls.js           control sheet built from a schema of native inputs
src/ui/StarSelection.js      click picking and permutation remap
src/system/SystemGen.js      deterministic solar-system generator
src/gpu/Camera.js            damped orbit camera, pointer and touch gestures
src/physics/NBodyGPU.js      compute pipelines (force, refresh, SF, sort)
src/physics/mortonTree.js    Morton codes, radix sort, BFS bucketed tree
src/physics/bhWorker.js      async topology rebuild worker
src/physics/shaders/         WGSL compute kernels
src/render/RendererGPU.js    HDR render graph
src/render/SystemRenderer.js solar-system scene (impostor spheres, rings, belt)
src/render/shaders/          WGSL render kernels
docs/plans/                  historical design documents (pre-rewrite)
```

## License

MIT — see [LICENSE](LICENSE).
