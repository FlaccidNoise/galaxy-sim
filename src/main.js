import { RendererGPU } from './render/RendererGPU.js';
import { Camera } from './gpu/Camera.js';
import { NBodyGPU, requestGPUDevice } from './physics/NBodyGPU.js';
import { galaxyDisk, uniformSphere, twoGalaxyCollision } from './physics/InitialConditions.js';
import { Controls } from './ui/Controls.js';
import { sortAndBuild } from './physics/mortonTree.js';
import { pickCandidates, readStellarRecords, remapIndex } from './ui/StarSelection.js';
import { generateSystem, bodyPositions } from './system/SystemGen.js';
import { SystemRenderer } from './render/SystemRenderer.js';

let renderer, camera, sim, controls, gpuDevice;
let treeWorker = null;
let posTransfer = null;        // Float32Array (vec4/particle) ping-ponged with worker
let readbackInFlight = false;  // GPU staging map pending
let workerBusy = false;        // tree build in progress
let pendingTree = null;        // {tree, numNodes, perm, centralIdx, …} awaiting GPU application
let centralIdx = 0;            // current index of galaxy A's central particle
let lastPositions = null;   // retained CPU snapshot (vec4/particle) in GPU order, for picking
let selection = null;       // { idx, system }
let centralIdx2 = -1;          // galaxy B's central (Collision preset), -1 = none
let haloPerGalaxy = 0;         // halo mass per halo for this preset (0 = off)
let haloCenters = [];          // SMOOTHED galaxy centres (EMA of BH positions)
let sfRho0 = 1;                // mean disk density — SF threshold reference
let energyBaseline = null;     // first total-energy reading after restart
let energyDrift = 0;
let stepAccumulator = 0;
let lastFrameTime = performance.now();
let simTime = 0;
let systemTime = 0;   // solar-system clock: runs at a fixed pace, independent of galaxy speed and pause
let systemRenderer = null;
let focusIdx = 0;            // focused system body, bodyPositions order (0 = the star)
let focusFrom = [0, 0, 0];   // focus point when the focus last changed
let focusCur = [0, 0, 0];    // blended point the system camera orbits
let focusT = 1;              // seconds since the focus changed
let sysUiShown = 0;          // in-system UI state last written to the DOM
const R_ENTER = 1.5;   // galaxy-camera radius where the system starts fading in
const R_FULL = 0.8;    // fully faded in
const FOCUS_BLEND = 0.4;   // seconds to move the system camera to a new focus
// System zoom law: galaxy-camera radius → system eye distance (0.02 → 0.5 AU, 1.5 → ~120)
const sysRadius = r => 0.5 * Math.pow(r / 0.02, 1.27);
const camRadiusFor = rSys => 0.02 * Math.pow(rSys / 0.5, 1 / 1.27);
const KIND_NAME = ['Star', 'Scorched world', 'Venusian world', 'Terran world', 'Icy world', 'Gas giant', 'Ice giant', 'Rocky world', 'Sub-Neptune'];
const scHintEls = [...document.querySelectorAll('#star-info .sc-hint span')];   // mouse, touch
const scHintText = scHintEls.map(n => n.textContent);
const hintEls = [...document.querySelectorAll('.hint span')];                   // mouse, touch
const hintText = [hintEls.map(n => n.textContent), ['Click a planet · Scroll · Esc', 'Tap a planet · Pinch · Tap sky']];

// FPS counter — element is declared in index.html
let frameCount = 0, lastFPSTime = performance.now();
const fpsEl = document.getElementById('fps');
const sheetFpsEl = document.getElementById('sheet-fps');
const capNEl = document.getElementById('cap-n');
const capTEl = document.getElementById('cap-t');

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
  const lit = 'OBAFGKM'.indexOf(star.class);
  el.querySelector('.sc-cls').textContent = star.class;
  el.querySelector('.sc-mass').textContent = `${star.mass.toFixed(2)} M☉`;
  el.querySelector('.sc-temp').textContent = star.tempK ? `${Math.round(star.tempK).toLocaleString()} K` : '—';
  el.querySelector('.sc-planets').textContent = `${planets.length}${belt ? ' · belt' : ''}`;
  el.querySelector('.sc-bar').setAttribute('aria-label', `Blackbody colour, class ${star.class}`);
  el.querySelectorAll('.sc-bar i, .sc-scale span').forEach((n, i) => n.classList.toggle('on', i % 7 === lit));
  const b = focusIdx ? bodyPositions(selection.system, 0)[focusIdx] : null;
  const focused = b && `Focused: ${KIND_NAME[b.planet.kind]}${b.moon ? ' moon' : ''}, ${b.planet.au.toFixed(2)} AU, ${Math.round(b.planet.teq)} K.\n`;
  const inSys = renderer.systemFade > 0;
  scHintEls[0].textContent = focused ? focused + 'Esc to release.' : inSys ? 'Esc to leave.' : scHintText[0];
  scHintEls[1].textContent = focused ? focused + 'Tap sky to release.' : inSys ? 'Tap sky to leave.' : scHintText[1];
  el.classList.add('visible');
}

// The galaxy renders in a frame centred on the selected star. When that frame
// changes, shift the camera target by the star's snapshot position so the view
// stays continuous; the damped camera then settles on the new origin.
function shiftCameraTarget(idx, sign) {
  if (!lastPositions) return;
  for (let c = 0; c < 3; c++) {
    const d = sign * lastPositions[idx * 4 + c];
    camera.target[c] += d;
    camera._goal.target[c] += d;
  }
}

// Zoom floor: the eye stays at least 1.6 star radii from the star's centre
// (never closer than the default 0.02), or 2.5 radii from a focused planet or moon.
function focusMinRadius() {
  const sys = selection.system;
  return focusIdx ? camRadiusFor(2.5 * bodyPositions(sys, 0)[focusIdx].radius)
                  : Math.max(0.02, camRadiusFor(1.6 * sys.star.radius));
}

function setFocus(i) {
  if (i === focusIdx) return;
  focusIdx = i;
  focusFrom = [...focusCur];
  focusT = 0;
  camera.setMinRadius(focusMinRadius());
  updateStarInfo();
}

// Nearest body whose projected disc, or an 18 px target, covers the click; -1 on a miss.
function pickBody(x, y) {
  const dpr = devicePixelRatio || 1;
  const px = x * dpr, py = y * dpr, W = renderer.width, H = renderer.height;
  const m = systemRenderer.viewProj;
  const tanHalf = Math.tan(camera.fovY / 2);
  let best = -1, bestD = Infinity;
  bodyPositions(selection.system, systemTime).forEach(({ pos: [bx, by, bz], radius }, i) => {
    const w = m[3] * bx + m[7] * by + m[11] * bz + m[15];
    if (w <= 0) return;
    const sx = ((m[0] * bx + m[4] * by + m[8] * bz + m[12]) / w * 0.5 + 0.5) * W;
    const sy = (-(m[1] * bx + m[5] * by + m[9] * bz + m[13]) / w * 0.5 + 0.5) * H;
    const d = Math.hypot(sx - px, sy - py);
    const rPx = radius / (w * tanHalf) * H / 2;
    if (d <= Math.max(rPx, 18 * dpr) && d < bestD) { best = i; bestD = d; }
  });
  return best;
}

function select(idx, record) {
  if (selection) shiftCameraTarget(selection.idx, 1);
  shiftCameraTarget(idx, -1);
  selection = {
    idx,
    system: generateSystem(record[0], record[1], record[2], simTime * (controls?.state.agingRate ?? 0) - record[1]),
  };
  renderer.selectedIdx = idx;
  focusIdx = 0; focusCur = [0, 0, 0]; focusT = 1;
  camera.setMinRadius(focusMinRadius());
  window.__system = selection.system;
  systemRenderer.setSystem(selection.system);
  updateStarInfo();
}

function deselect() {
  if (!selection) return;
  shiftCameraTarget(selection.idx, 1);
  selection = null;
  focusIdx = 0; focusCur = [0, 0, 0]; focusT = 1;
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
  if (selection && renderer.systemFade > 0) {
    const b = pickBody(x, y);
    if (b >= 0) setFocus(b);
    else if (focusIdx) setFocus(0);
    else deselect();
    return;
  }
  if (!sim?.ready || !lastPositions) return;
  const mySim = sim;
  const dpr = devicePixelRatio || 1;
  const origin = selection ? lastPositions.subarray(selection.idx * 4, selection.idx * 4 + 3) : undefined;
  const cands = pickCandidates(lastPositions, sim.N, camera.viewProj,
                               x * dpr, y * dpr, renderer.width, renderer.height, 15, 4, origin);
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
    canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;touch-action:none;';
    (document.getElementById('sim') ?? document.body).appendChild(canvas);

    camera = new Camera(canvas);

    let downX = 0, downY = 0, pointersDown = 0, multiTouch = false;
    canvas.addEventListener('pointerdown', e => {
      if (pointersDown++ === 0) { downX = e.clientX; downY = e.clientY; multiTouch = false; }
      else multiTouch = true;
    });
    canvas.addEventListener('pointercancel', () => { pointersDown = Math.max(pointersDown - 1, 0); });
    canvas.addEventListener('pointerup', e => {
      pointersDown = Math.max(pointersDown - 1, 0);
      // Click = single left button/finger released within 4 px of where it went down
      if (multiTouch || e.button !== 0 || Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return;
      handlePick(e.clientX, e.clientY).catch(console.error);
    });
    document.addEventListener('keydown', e => {
      if (e.key !== 'Escape') return;
      if (focusIdx) setFocus(0);
      else if (controls?.isOpen) controls.open(false);
      else deselect();
    });

    setLoadingStatus('Building render pipeline…');
    renderer = await new RendererGPU(gpuDevice, canvas).init();
    systemRenderer = await new SystemRenderer(gpuDevice).init();
  }

  // Tear down previous sim + worker state
  deselect();
  lastPositions = null;
  if (sim) { const old = sim; sim = null; old.destroy(); renderer?.clearSceneCache(); camera.home(); }
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
  window.__camera = camera;
  window.__systemRenderer = systemRenderer;
  window.__dbg = () => ({ centralIdx, centralIdx2, haloPerGalaxy, sfRho0, simTime, systemTime, focusIdx, focusCur });
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
    const n = (sim?.N ?? 0).toLocaleString();
    const drift = energyBaseline !== null ? ` · ΔE ${(energyDrift * 100).toFixed(1)}%` : '';
    fpsEl.innerHTML = `<b>${frameCount} FPS</b> <span>· ${n}${drift}</span>`;
    sheetFpsEl.textContent = `${frameCount} FPS · ${n}`;
    capNEl.textContent = n;
    capTEl.textContent = (simTime * (controls?.state.agingRate ?? 0) / 1000).toFixed(1);
    frameCount = 0;
    lastFPSTime = now;
  }
  const dtWall = Math.min((now - lastFrameTime) / 1000, 0.1);
  lastFrameTime = now;

  systemTime += dtWall * 0.6;   // 0.6 = 60 steps/s × dt 0.01, the galaxy's pace at speed 1

  if (selection) camera._goal.target.fill(0);   // the selected star is the origin; no panning off it
  camera.update();

  // Solar-system crossfade: below R_ENTER the system scene fades in and the
  // galaxy backdrop pins at R_ENTER so neighboring sprites don't fly past.
  let systemScene = null;
  let galaxyCam = camera;
  if (selection) {
    const t = Math.min(Math.max((R_ENTER - camera.radius) / (R_ENTER - R_FULL), 0), 1);
    const fade = t * t * (3 - 2 * t);
    renderer.systemFade = fade;
    if (fade === 0) setFocus(0);
    else {
      focusT += dtWall;
      const k = Math.min(focusT / FOCUS_BLEND, 1), blend = k * k * (3 - 2 * k);
      const to = focusIdx ? bodyPositions(selection.system, systemTime)[focusIdx].pos : [0, 0, 0];
      for (let c = 0; c < 3; c++) focusCur[c] = focusFrom[c] + (to[c] - focusFrom[c]) * blend;
      systemRenderer.updateCamera(camera, sysRadius(camera.radius), fade, systemTime, focusCur);
      systemScene = systemRenderer;
      if (camera.radius < R_ENTER) galaxyCam = camera.viewFrom(R_ENTER);
    }
  }

  // In a system: planet-picking hint; past half fade, a solid card over the sun's glow
  const sysUi = (renderer.systemFade > 0) + (renderer.systemFade > 0.5);
  if (sysUi !== sysUiShown) {
    sysUiShown = sysUi;
    document.getElementById('star-info').classList.toggle('in-system', sysUi === 2);
    hintEls.forEach((n, i) => { n.textContent = hintText[sysUi > 0 ? 1 : 0][i]; });
    updateStarInfo();
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

  // Fractional speed: accumulate sub-step credits so speed < 1 skips steps.
  // Wall-clock based: speed 1 = 60 steps/s at any refresh rate at or above
  // 60 Hz. Below that, a slow frame earns at most `speed` steps, so a slow
  // frame cannot demand more steps and slow the next one further.
  // The galaxy slows to a stop as a solar system fades in.
  stepAccumulator += s.speed * 60 * Math.min(dtWall, 1 / 60) * (1 - renderer.systemFade);
  // Cap steps/frame: physics quality degrades when many steps run against
  // one (increasingly stale) tree snapshot. Credit beyond the cap is dropped.
  const stepsThisFrame = Math.min(Math.floor(stepAccumulator), 8);
  stepAccumulator = Math.min(stepAccumulator - stepsThisFrame, 1);

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
    // Keep the CPU snapshot in GPU order so picking and frame shifts read the right star
    if (lastPositions && lastPositions.length === pendingTree.perm.length * 4) {
      const perm = pendingTree.perm, src = lastPositions, dst = new Float32Array(src.length);
      for (let i = 0; i < perm.length; i++) {
        const o = i * 4, q = perm[i] * 4;
        dst[o] = src[q]; dst[o + 1] = src[q + 1]; dst[o + 2] = src[q + 2]; dst[o + 3] = src[q + 3];
      }
      lastPositions = dst;
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
  document.getElementById('bl').classList.add('ready');

  // H key — toggle controls sheet
  document.addEventListener('keydown', e => {
    if (e.target.closest?.('input, select, textarea')) return;
    if ((e.key === 'h' || e.key === 'H') && !e.ctrlKey && !e.altKey && !e.metaKey) controls.toggle();
  });

  frameCount = 0;
  lastFPSTime = lastFrameTime = performance.now();
  loop();
} catch (err) {
  fail(err);
}
