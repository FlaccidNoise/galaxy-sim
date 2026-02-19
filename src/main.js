import { Renderer } from './render/Renderer.js';
import { ParticleSystem } from './render/ParticleSystem.js';
import { NBodyGPU, requestGPUDevice } from './physics/NBodyGPU.js';
import { galaxyDisk, uniformSphere, twoGalaxyCollision } from './physics/InitialConditions.js';
import { Controls } from './ui/Controls.js';

let renderer, particles, sim, controls, gpuDevice;
let readbackInProgress = false;
let stepAccumulator = 0;

// FPS counter elements
let frameCount = 0, lastFPSTime = performance.now();
const fpsEl = Object.assign(document.createElement('div'), {
  style: 'position:fixed;top:8px;left:8px;color:#fff;font:12px monospace;pointer-events:none;z-index:1000',
});
document.body.appendChild(fpsEl);

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
    renderer = await new Renderer().init();
    gpuDevice = await requestGPUDevice();
    console.log('WebGPU device:', gpuDevice);
  } else {
    // Remove old particles from scene
    if (particles) renderer.scene.remove(particles.points);
  }

  // Create new physics simulation
  sim = new NBodyGPU(gpuDevice, particleData);

  // Wait for async GPU init to complete
  await new Promise(resolve => {
    const check = () => sim.ready ? resolve() : requestAnimationFrame(check);
    check();
  });

  // Create new particle system
  particles = new ParticleSystem(renderer.scene, N);
  particles.initStaging(gpuDevice, particleData.byteLength);

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
    sim.updateParams(s.dt, s.G, s.epsilon, sim.N);

    const enc = gpuDevice.createCommandEncoder();
    sim.step(enc);

    if (i === stepsThisFrame - 1 && !readbackInProgress) {
      // Last step: queue readback in same encoder, submit, then update geometry
      particles.queueReadback(enc, sim.positionBuffer);
      gpuDevice.queue.submit([enc.finish()]);
      readbackInProgress = true;
      particles.updateFromStaging(s.colorMode, 50.0).then(() => { readbackInProgress = false; });
    } else {
      gpuDevice.queue.submit([enc.finish()]);
    }
  }

  renderer.render();
}

// Bootstrap
const defaultState = {
  preset: 'Galaxy Disk',
  N: 10000,
  diskRadius: 50,
  diskHeight: 3,
  centralMass: 2000,
  velDispersion: 0.3,
  armCount: 2,
  G: 1.0,
  epsilon: 2.0,
  dt: 0.01,
};

await init(defaultState);
loop();
