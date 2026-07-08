import { RendererGPU } from './render/RendererGPU.js';
import { Camera } from './gpu/Camera.js';
import { NBodyGPU, requestGPUDevice } from './physics/NBodyGPU.js';
import { galaxyDisk, uniformSphere, twoGalaxyCollision } from './physics/InitialConditions.js';
import { Controls } from './ui/Controls.js';
import { sortAndBuild } from './physics/mortonTree.js';
import { pickCandidates, readStellarRecords, remapIndex } from './ui/StarSelection.js';
import { generateSystem } from './system/SystemGen.js';
import { SystemRenderer } from './render/SystemRenderer.js';

let renderer, camera, sim, controls, gpuDevice;
let treeWorker = null;
let posTransfer = null;        // Float32Array (vec4/particle) ping-ponged with worker
let readbackInFlight = false;  // GPU staging map pending
let workerBusy = false;        // tree build in progress
let pendingTree = null;        // {tree, numNodes, perm, centralIdx, …} awaiting GPU application
let centralIdx = 0;            // current index of galaxy A's central particle
let lastPositions = null;   // retained CPU snapshot (vec4/particle) for picking
// Snapshot lags the GPU by up to one rebuild cycle: between a perm application
// and the next readback, indices refer to pre-perm order (picking may hit a
// Morton-neighbor; self-corrects on next snapshot).
let selection = null;       // { idx, system }
let centralIdx2 = -1;          // galaxy B's central (Collision preset), -1 = none
let haloPerGalaxy = 0;         // halo mass per halo for this preset (0 = off)
let haloCenters = [];          // SMOOTHED galaxy centres (EMA of BH positions)
let sfRho0 = 1;                // mean disk density — SF threshold reference
let energyBaseline = null;     // first total-energy reading after restart
let energyDrift = 0;
let stepAccumulator = 0;
let simTime = 0;
let systemRenderer = null;
const R_ENTER = 1.5;   // galaxy-camera radius where the system starts fading in
const R_FULL = 0.8;    // fully faded in

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

function select(idx, record) {
  selection = {
    idx,
    system: generateSystem(record[0], record[1], record[2]),
  };
  renderer.selectedIdx = idx;
  camera.setMinRadius(0.02);
  window.__system = selection.system;
  systemRenderer.setSystem(selection.system);
  updateStarInfo();
}

function deselect() {
  if (!selection) return;
  selection = null;
  renderer.selectedIdx = 0xffffffff;
  renderer.systemFade = 0;
  systemRenderer?.setSystem(null);
  camera.setMinRadius(2);
  camera.radius = Math.max(camera.radius, R_ENTER);
  if (camera._goal.radius < 2.5) camera._goal.radius = 2.5;
  window.__system = null;
  updateStarInfo();
}

async function handlePick(x, y) {
  if (renderer.systemFade > 0) return;   // inside a system: clicks are not picks
  if (!sim?.ready || !lastPositions) return;
  const mySim = sim;
  const dpr = devicePixelRatio || 1;
  const cands = pickCandidates(lastPositions, sim.N, camera.viewProj,
                               x * dpr, y * dpr, renderer.width, renderer.height);
  if (!cands.length) { deselect(); return; }
  const records = await readStellarRecords(gpuDevice, sim.stellarBuffer, cands.map(c => c.idx));
  if (sim !== mySim) return;
  for (let i = 0; i < cands.length; i++) {
    if (records[i * 4] <= 0) continue;   // gas (−1) and central BHs (0) unpickable
    select(cands[i].idx, records.subarray(i * 4, i * 4 + 4));
    return;
  }
  deselect();
}

async function init(params) {
  const presetFns = {
    'Galaxy Disk': galaxyDisk,
    'Uniform Sphere': uniformSphere,
    'Collision': twoGalaxyCollision,
  };

  // Halos are centred on each galaxy's central particle, so they move with
  // their galaxy (essential for the Collision preset). Mass is PER halo.
  haloPerGalaxy = params.preset === 'Galaxy Disk' ? params.haloFactor * params.N
                : params.preset === 'Collision' ? params.haloFactor * Math.floor(params.N / 2)
                : 0;
  const icParams = {
    N: params.N,
    diskRadius: params.diskRadius,
    diskHeight: params.diskHeight,
    centralMass: params.centralMass,
    toomreQ: params.toomreQ,
    armCount: params.armCount,
    G: params.G,
    haloMass: haloPerGalaxy,
    haloA: params.haloScale * params.diskRadius,
  };
  // Mean midplane density heuristic — reference for the SF threshold
  sfRho0 = params.N / (Math.PI * params.diskRadius * params.diskRadius * params.diskHeight);
  energyBaseline = null;
  energyDrift = 0;

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

    setLoadingStatus('Building render pipeline…');
    renderer = await new RendererGPU(gpuDevice, canvas).init();
    systemRenderer = await new SystemRenderer(gpuDevice).init();
  }

  // Tear down previous sim + worker state
  deselect();
  lastPositions = null;
  if (sim) { const old = sim; sim = null; old.destroy(); renderer?.clearSceneCache(); }
  if (treeWorker) { treeWorker.terminate(); treeWorker = null; }
  readbackInFlight = false;
  workerBusy = false;
  pendingTree = null;
  simTime = 0;

  setLoadingStatus(`Spawning ${N.toLocaleString()} stars…`);

  // Morton-sort the ICs on the CPU so the GPU starts in Z-curve order, and
  // build the frame-0 tree from the same sorted snapshot.
  const trackIdx2 = params.preset === 'Collision' ? Math.floor(N / 2) : -1;
  const boot = sortAndBuild(particleData, stellarData, N, trackIdx2);
  centralIdx = boot.centralIdx;
  centralIdx2 = boot.centralIdx2;

  // Halo centres start at the BH initial positions, then track them via EMA
  haloCenters = [];
  if (haloPerGalaxy > 0) {
    haloCenters.push([...boot.sorted.subarray(centralIdx * 8, centralIdx * 8 + 3)]);
    if (centralIdx2 >= 0) {
      haloCenters.push([...boot.sorted.subarray(centralIdx2 * 8, centralIdx2 * 8 + 3)]);
    }
  }

  setLoadingStatus('Building compute pipeline…');
  const newSim = new NBodyGPU(gpuDevice, boot.sorted, boot.sortedStellar);
  await newSim.initPromise;
  newSim.uploadBHTree(boot.tree._buf, boot.tree.numNodes, boot.tree.levelRanges);

  // Tree worker: owns all subsequent rebuilds, off the main thread
  treeWorker = new Worker(new URL('./physics/bhWorker.js', import.meta.url), { type: 'module' });
  treeWorker.postMessage({ type: 'config', N, masses: boot.masses, centralIdx, centralIdx2 });
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
  window.__sim = sim;  // debug/verification hooks
  window.__renderer = renderer;
  window.__dbg = () => ({ centralIdx, centralIdx2, haloPerGalaxy, sfRho0, simTime });
  window.__state = controls?.state;
  window.__restart = (over = {}) => {
    Object.assign(controls.state, over);
    return init(controls.state).catch(fail);
  };
  window.__select = async idx => {
    if (!sim?.ready || idx < 0 || idx >= sim.N) return;
    const mySim = sim;
    const rec = await readStellarRecords(gpuDevice, sim.stellarBuffer, [idx]);
    if (sim !== mySim) return;
    select(idx, rec.subarray(0, 4));
  };

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
  window.__state = controls.state;
}

function loop() {
  requestAnimationFrame(loop);

  // FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastFPSTime > 1000) {
    const drift = energyBaseline !== null ? ` | ΔE ${(energyDrift * 100).toFixed(1)}%` : '';
    fpsEl.textContent = `${frameCount} fps | N=${(sim?.N ?? 0).toLocaleString()}${drift}`;
    frameCount = 0;
    lastFPSTime = now;
  }

  if (selection && lastPositions) {
    const i = selection.idx * 4;
    camera._goal.target[0] = lastPositions[i];
    camera._goal.target[1] = lastPositions[i + 1];
    camera._goal.target[2] = lastPositions[i + 2];
  }
  camera.update();

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

  if (!sim || !sim.ready || controls?.state.paused) {
    if (renderer && sim?.ready) {
      const enc = gpuDevice.createCommandEncoder();
      renderer.encode(enc, galaxyCam, sim.positionBuffer, sim.stellarBuffer, sim.N, simTime, systemScene);
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
  const paramsNow = {
    dt: s.dt, G: s.G, epsilon: s.epsilon, theta: s.theta,
    haloMass: haloPerGalaxy,
    haloA: s.haloScale * s.diskRadius,
    haloCenters,
    gasDrag: s.gasDrag,
    time: simTime,
    agingRate: s.agingRate,
    sfDensity: s.sfThreshold * sfRho0,
    sfEff: s.sfEff,
    sfDt: s.dt * stepsThisFrame,
  };

  if (pendingTree) {
    centralIdx = pendingTree.centralIdx;
    centralIdx2 = pendingTree.centralIdx2 ?? -1;
    renderer.centralIdx = centralIdx;
    // Permutation is a gather (out[i] = in[perm[i]]): old index → the new
    // slot that references it.
    if (selection) {
      const ni = remapIndex(pendingTree.perm, selection.idx);
      if (ni >= 0) { selection.idx = ni; renderer.selectedIdx = ni; }
      else deselect();
    }
    // Drift the halo centres toward the BH positions (EMA: follows the
    // galaxy's orbit, ignores the BH's encounter jitter)
    const k = 0.1;
    const targets = [pendingTree.bh1, pendingTree.bh2].filter(Boolean);
    for (let h = 0; h < haloCenters.length && h < targets.length; h++) {
      for (let c = 0; c < 3; c++) {
        haloCenters[h][c] += (targets[h][c] - haloCenters[h][c]) * k;
      }
    }
    sim.updateParams(paramsNow);
    sim.applyPermutation(enc, pendingTree.perm);
    sim.uploadBHTree(pendingTree.tree, pendingTree.numNodes, pendingTree.levelRanges);
    if (typeof pendingTree.energy === 'number' && Number.isFinite(pendingTree.energy)) {
      if (energyBaseline === null && pendingTree.energy !== 0) {
        energyBaseline = pendingTree.energy;
      } else if (energyBaseline) {
        energyDrift = (pendingTree.energy - energyBaseline) / Math.abs(energyBaseline);
      }
    }
    pendingTree = null;
  }

  if (stepsThisFrame > 0) {
    sim.updateParams(paramsNow);  // once per frame
    for (let i = 0; i < stepsThisFrame; i++) {
      sim.encodeRefresh(enc);  // re-sum tree COMs from current positions
      sim.step(enc);
      simTime += s.dt;
    }

    // Star formation + stellar recycling (uses densities the steps just wrote)
    sim.encodeStarFormation(enc);

    // Queue a positions readback for the tree worker when the previous
    // round trip (GPU map + worker build) has fully completed.
    if (!readbackInFlight && !workerBusy) {
      sim.encodePositionReadback(enc);
      readbackInFlight = true;
    }
  }

  renderer.encode(enc, galaxyCam, sim.positionBuffer, sim.stellarBuffer, sim.N, simTime, systemScene);
  gpuDevice.queue.submit([enc.finish()]);

  if (readbackInFlight && !workerBusy && posTransfer) {
    const mySim = sim;
    workerBusy = true;
    sim.readPositions(posTransfer).then(() => {
      readbackInFlight = false;
      if (sim !== mySim || !treeWorker) { workerBusy = false; return; }
      if (!lastPositions || lastPositions.length !== posTransfer.length) {
        lastPositions = new Float32Array(posTransfer.length);
      }
      lastPositions.set(posTransfer);
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
