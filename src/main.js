import { Renderer } from './render/Renderer.js';
import { ParticleSystem } from './render/ParticleSystem.js';
import { NBodyGPU, requestGPUDevice } from './physics/NBodyGPU.js';
import { galaxyDisk, uniformSphere, twoGalaxyCollision } from './physics/InitialConditions.js';
import { Controls } from './ui/Controls.js';
import { BHTree } from './physics/BHTree.js';

let renderer, particles, sim, controls, gpuDevice;
let bhTree = null;
let particleMasses = null;  // extracted from IC; covers all presets including multi-central-mass
let readbackInProgress = false;
let stepAccumulator = 0;

// FPS counter — element is declared in index.html
let frameCount = 0, lastFPSTime = performance.now();
const fpsEl = document.getElementById('fps');

function setLoadingStatus(msg) {
  const el = document.getElementById('loading-status');
  if (el) el.textContent = msg;
}

async function init(params) {
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
    armCount: params.armCount,
    G: params.G,
  };

  const particleData = presetFns[params.preset](icParams);
  const N = particleData.length / 8;

  // Init renderer and GPU device on first call only
  if (!renderer) {
    setLoadingStatus('Requesting GPU device\u2026');
    renderer = await new Renderer().init();
    gpuDevice = await requestGPUDevice();
    console.log('WebGPU device:', gpuDevice);
  } else {
    // Remove old particles from scene
    if (particles) renderer.scene.remove(particles.points);
  }

  // Create new physics simulation
  setLoadingStatus(`Building compute pipeline\u2026`);
  sim = new NBodyGPU(gpuDevice, particleData);

  // Wait for async GPU init to complete
  await new Promise(resolve => {
    const check = () => sim.ready ? resolve() : requestAnimationFrame(check);
    check();
  });

  setLoadingStatus(`Spawning ${N.toLocaleString()} stars\u2026`);
  // Create new particle system
  particles = new ParticleSystem(renderer.scene, N);
  particles.initStaging(gpuDevice, particleData.byteLength);

  // Extract per-particle masses from IC (handles all presets: single galaxy,
  // collision with two central masses, uniform sphere, etc.).
  particleMasses = new Float32Array(N);
  const initPos  = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    initPos[i*3]     = particleData[i*8];
    initPos[i*3 + 1] = particleData[i*8 + 1];
    initPos[i*3 + 2] = particleData[i*8 + 2];
    particleMasses[i] = particleData[i*8 + 3];
  }

  // Build initial BH tree from IC data so frame 0 has valid physics.
  bhTree = new BHTree(N * 2 + 1000);
  bhTree.build(initPos, N, particleMasses);
  sim.uploadBHTree(bhTree);

  // Init controls on first call only
  if (!controls) {
    controls = new Controls(sim, particles, renderer);
    controls.onRestart(state => init(state));
  }
}

async function loop() {
  requestAnimationFrame(loop);

  // FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastFPSTime > 1000) {
    fpsEl.textContent = `${frameCount} fps | N=${sim?.N ?? 0}`;
    frameCount = 0;
    lastFPSTime = now;
  }

  if (!sim || !sim.ready || controls?.state.paused) {
    if (renderer) renderer.render();
    return;
  }

  const s = controls.state;

  // Fractional speed: accumulate sub-step credits so speed < 1 skips frames
  stepAccumulator += s.speed;
  const stepsThisFrame = Math.floor(stepAccumulator);
  stepAccumulator -= stepsThisFrame;

  if (stepsThisFrame === 0) {
    renderer.render();
    return;
  }

  for (let i = 0; i < stepsThisFrame; i++) {
    sim.updateParams(s.dt, s.G, s.epsilon, sim.N, s.theta);

    const enc = gpuDevice.createCommandEncoder();
    sim.step(enc);

    if (i === stepsThisFrame - 1 && !readbackInProgress) {
      // Last step: queue readback in same encoder, submit, then update geometry
      particles.queueReadback(enc, sim.positionBuffer);
      gpuDevice.queue.submit([enc.finish()]);
      readbackInProgress = true;
      particles.updateFromStaging(s.colorMode, 50.0).then(() => {
        readbackInProgress = false;
        // Rebuild BH tree from latest positions (1-frame lag; imperceptible visually).
        if (bhTree) {
          bhTree.build(particles.positions, particles.N, particleMasses);
          sim.uploadBHTree(bhTree);
        }
      });
    } else {
      gpuDevice.queue.submit([enc.finish()]);
    }
  }

  renderer.render();
}

// Bootstrap — N/velDispersion must match Controls.js defaults exactly
const defaultState = {
  preset: 'Galaxy Disk',
  N: 25000,
  diskRadius: 50,
  diskHeight: 2,
  centralMass: 10000,
  velDispersion: 0.2,
  armCount: 2,
  G: 1.0,
  epsilon: 1.5,
  dt: 0.01,
};

await init(defaultState);

// Reveal simulation UI once WebGPU + particles are ready
const loadingEl  = document.getElementById('loading');
if (loadingEl) {
  loadingEl.classList.add('hidden');
  setTimeout(() => loadingEl.remove(), 700);
}
['hud-title', 'fps', 'controls-hint'].forEach(id => {
  document.getElementById(id)?.classList.add('ready');
});
document.querySelectorAll('.lil-gui.root').forEach(el => el.classList.add('ready'));

// H key — toggle controls panel visibility
document.addEventListener('keydown', e => {
  if ((e.key === 'h' || e.key === 'H') && !e.ctrlKey && !e.altKey && !e.metaKey) {
    document.querySelectorAll('.lil-gui.root').forEach(el => {
      el.style.display = el.style.display === 'none' ? '' : 'none';
    });
  }
});

loop();
