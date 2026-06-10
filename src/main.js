import { RendererGPU } from './render/RendererGPU.js';
import { Camera } from './gpu/Camera.js';
import { NBodyGPU, requestGPUDevice } from './physics/NBodyGPU.js';
import { galaxyDisk, uniformSphere, twoGalaxyCollision } from './physics/InitialConditions.js';
import { Controls } from './ui/Controls.js';
import { sortAndBuild } from './physics/mortonTree.js';

let renderer, camera, sim, controls, gpuDevice;
let treeWorker = null;
let posTransfer = null;        // Float32Array (vec4/particle) ping-ponged with worker
let readbackInFlight = false;  // GPU staging map pending
let workerBusy = false;        // tree build in progress
let pendingTree = null;        // {tree, numNodes, perm, centralIdx} awaiting GPU application
let centralIdx = 0;            // current index of the central-mass particle
let haloEnabled = false;       // analytic halo active for this preset
let stepAccumulator = 0;
let simTime = 0;

// FPS counter — element is declared in index.html
let frameCount = 0, lastFPSTime = performance.now();
const fpsEl = document.getElementById('fps');

function setLoadingStatus(msg) {
  const el = document.getElementById('loading-status');
  if (el) el.textContent = msg;
}

function fail(err) {
  console.error(err);
  setLoadingStatus(`Error: ${err.message ?? err}`);
  const spinner = document.querySelector('.loading-spinner');
  if (spinner) spinner.style.display = 'none';
}

async function init(params) {
  const presetFns = {
    'Galaxy Disk': galaxyDisk,
    'Uniform Sphere': uniformSphere,
    'Collision': twoGalaxyCollision,
  };

  // Halo applies to the single-galaxy preset only: the analytic potential is
  // pinned at the origin, which is wrong for moving/colliding systems.
  haloEnabled = params.preset === 'Galaxy Disk';
  const icParams = {
    N: params.N,
    diskRadius: params.diskRadius,
    diskHeight: params.diskHeight,
    centralMass: params.centralMass,
    toomreQ: params.toomreQ,
    armCount: params.armCount,
    G: params.G,
    haloMass: haloEnabled ? params.haloFactor * params.N : 0,
    haloA: params.haloScale * params.diskRadius,
  };

  const { data: particleData, stellar: stellarData } = presetFns[params.preset](icParams);
  const N = particleData.length / 8;

  // Init renderer, camera, GPU device on first call only
  if (!renderer) {
    setLoadingStatus('Requesting GPU device…');
    gpuDevice = await requestGPUDevice();
    gpuDevice.lost.then(info => fail(new Error(`GPU device lost: ${info.message}`)));

    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;';
    (document.getElementById('sim') ?? document.body).appendChild(canvas);

    camera = new Camera(canvas);
    setLoadingStatus('Building render pipeline…');
    renderer = await new RendererGPU(gpuDevice, canvas).init();
  }

  // Tear down previous sim + worker state
  if (sim) { const old = sim; sim = null; old.destroy(); }
  if (treeWorker) { treeWorker.terminate(); treeWorker = null; }
  readbackInFlight = false;
  workerBusy = false;
  pendingTree = null;
  simTime = 0;

  setLoadingStatus(`Spawning ${N.toLocaleString()} stars…`);

  // Morton-sort the ICs on the CPU so the GPU starts in Z-curve order, and
  // build the frame-0 tree from the same sorted snapshot.
  const boot = sortAndBuild(particleData, stellarData, N);
  centralIdx = boot.centralIdx;

  setLoadingStatus('Building compute pipeline…');
  const newSim = new NBodyGPU(gpuDevice, boot.sorted, boot.sortedStellar);
  await newSim.initPromise;
  newSim.uploadBHTree(boot.tree._buf, boot.tree.numNodes, boot.tree.levelRanges);

  // Tree worker: owns all subsequent rebuilds, off the main thread
  treeWorker = new Worker(new URL('./physics/bhWorker.js', import.meta.url), { type: 'module' });
  treeWorker.postMessage({ type: 'config', N, masses: boot.masses, centralIdx });
  posTransfer = new Float32Array(N * 4);
  treeWorker.onmessage = (e) => {
    if (e.data.type !== 'tree') return;
    // Tree leaf ranges use the new sorted order — defer upload until the
    // permutation has been applied on the GPU (next frame, same encoder).
    pendingTree = e.data;
    posTransfer = new Float32Array(e.data.positions);  // reclaim transferred buffer
    workerBusy = false;
  };
  treeWorker.onerror = (e) => fail(new Error(`Tree worker: ${e.message}`));

  sim = newSim;
  renderer.centralIdx = centralIdx;
  window.__sim = sim;  // debug/verification hook

  // Init controls on first call only
  if (!controls) {
    controls = new Controls(renderer);
    controls.onRestart(state => init(state).catch(fail));
    controls.onCentralMass(v => {
      if (!sim) return;
      sim.updateCentralMass(v, centralIdx);
      treeWorker.postMessage({ type: 'centralMass', value: v });
    });
  }
}

function loop() {
  requestAnimationFrame(loop);

  // FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastFPSTime > 1000) {
    fpsEl.textContent = `${frameCount} fps | N=${(sim?.N ?? 0).toLocaleString()}`;
    frameCount = 0;
    lastFPSTime = now;
  }

  camera.update();

  if (!sim || !sim.ready || controls?.state.paused) {
    if (renderer && sim?.ready) {
      const enc = gpuDevice.createCommandEncoder();
      renderer.encode(enc, camera, sim.positionBuffer, sim.stellarBuffer, sim.N, simTime);
      gpuDevice.queue.submit([enc.finish()]);
    }
    return;
  }

  const s = controls.state;

  // Fractional speed: accumulate sub-step credits so speed < 1 skips steps
  stepAccumulator += s.speed;
  // Cap steps/frame: physics quality degrades when many steps run against
  // one (increasingly stale) tree snapshot.
  const stepsThisFrame = Math.min(Math.floor(stepAccumulator), 8);
  stepAccumulator -= stepsThisFrame;

  const enc = gpuDevice.createCommandEncoder();

  // Apply any freshly built tree: reorder particles along the Morton curve,
  // then upload the tree (whose leaf indices assume the new order).
  const haloMass = haloEnabled ? s.haloFactor * sim.N : 0;
  const haloA = s.haloScale * s.diskRadius;

  if (pendingTree) {
    sim.updateParams(s.dt, s.G, s.epsilon, s.theta, haloMass, haloA);
    sim.applyPermutation(enc, pendingTree.perm);
    sim.uploadBHTree(pendingTree.tree, pendingTree.numNodes, pendingTree.levelRanges);
    centralIdx = pendingTree.centralIdx;
    renderer.centralIdx = centralIdx;
    pendingTree = null;
  }

  if (stepsThisFrame > 0) {
    sim.updateParams(s.dt, s.G, s.epsilon, s.theta, haloMass, haloA);  // once per frame
    for (let i = 0; i < stepsThisFrame; i++) {
      sim.encodeRefresh(enc);  // re-sum tree COMs from current positions
      sim.step(enc);
      simTime += s.dt;
    }

    // Queue a positions readback for the tree worker when the previous
    // round trip (GPU map + worker build) has fully completed.
    if (!readbackInFlight && !workerBusy) {
      sim.encodePositionReadback(enc);
      readbackInFlight = true;
    }
  }

  renderer.encode(enc, camera, sim.positionBuffer, sim.stellarBuffer, sim.N, simTime);
  gpuDevice.queue.submit([enc.finish()]);

  if (readbackInFlight && !workerBusy && posTransfer) {
    const mySim = sim;
    workerBusy = true;
    sim.readPositions(posTransfer).then(() => {
      readbackInFlight = false;
      if (sim !== mySim || !treeWorker) { workerBusy = false; return; }
      treeWorker.postMessage({ type: 'build', positions: posTransfer }, [posTransfer.buffer]);
      posTransfer = null;  // owned by worker until it returns
    }).catch(() => { readbackInFlight = false; workerBusy = false; });
  }
}

// Bootstrap — must match Controls.js defaults exactly
const defaultState = {
  preset: 'Galaxy Disk',
  N: 25000,
  diskRadius: 50,
  diskHeight: 2,
  centralMass: 500,
  toomreQ: 1.3,
  armCount: 2,
  G: 1.0,
  epsilon: 1.5,
  dt: 0.01,
  haloFactor: 3.0,
  haloScale: 1.2,
};

try {
  await init(defaultState);

  // Reveal simulation UI once WebGPU + particles are ready
  const loadingEl = document.getElementById('loading');
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
} catch (err) {
  fail(err);
}
