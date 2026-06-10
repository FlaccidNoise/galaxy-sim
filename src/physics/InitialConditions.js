// Particle layout: [x, y, z, mass, vx, vy, vz, pad] per particle (stride = 8 floats)
// Stellar layout:  [stellarMass M☉, birthTime (sim units, ≤0 at t=0), seed, 0]
const STRIDE = 8;

function gaussRandom() {
  // `|| EPSILON` guards Math.random() returning 0 → log(0) = -Infinity → NaN
  const u = Math.random() || Number.EPSILON, v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Sample a stellar mass from the Kroupa (2001) IMF:
 *   dN/dM ∝ M^-1.3 for 0.08–0.5 M☉, ∝ M^-2.3 above 0.5 M☉.
 * ~3/4 of stars come out below 0.5 M☉ — dim red dwarfs — which is what
 * gives the rare luminous giants room to blaze in HDR.
 * @param {number} mMax  upper cutoff (already-dead stars excluded at birth)
 */
export function kroupaIMF(mMax = 100) {
  // Segment number fractions for [0.08, 0.5] vs [0.5, mMax] (α = 1.3 / 2.3)
  // ∫M^-1.3 over [0.08,0.5] = (0.08^-0.3 − 0.5^-0.3)/0.3 ≈ 2.74
  // ∫M^-2.3 · 0.5-continuity over [0.5,mMax] ≈ 0.5^-1.3·(1−(2mMax)^-1.3)/1.3·…
  const lowInt = (Math.pow(0.08, -0.3) - Math.pow(0.5, -0.3)) / 0.3;
  const hiCut = Math.max(Math.min(mMax, 100), 0.5001);
  const k2 = Math.pow(0.5, -1.3 + 2.3); // continuity factor 0.5^1.0
  const hiInt = k2 * (Math.pow(0.5, -1.3) - Math.pow(hiCut, -1.3)) / 1.3;
  const pLow = lowInt / (lowInt + hiInt);
  const u = Math.random();
  if (u < pLow) {
    // inverse CDF on M^-1.3 segment
    const v = Math.random();
    const a = Math.pow(0.08, -0.3), b = Math.pow(0.5, -0.3);
    return Math.pow(a - v * (a - b), -1 / 0.3);
  }
  const v = Math.random();
  const a = Math.pow(0.5, -1.3), b = Math.pow(hiCut, -1.3);
  return Math.pow(a - v * (a - b), -1 / 1.3);
}

/** Main-sequence lifetime in Myr: ~10 Gyr · (M/M☉)^-2.5 */
export function mainSequenceLifetimeMyr(m) {
  return 10000 * Math.pow(m, -2.5);
}

/**
 * Stellar population entry: mass from IMF (capped so stars that would
 * ALREADY have exploded/faded long before `ageMyr` aren't oversampled),
 * birth time in Myr before t=0.
 */
function sampleStar(stellar, i, ageMinMyr, ageMaxMyr, dustWeight = 0) {
  const ageMyr = ageMinMyr + Math.random() * (ageMaxMyr - ageMinMyr);
  stellar[i * 4] = kroupaIMF();
  stellar[i * 4 + 1] = -ageMyr;           // birthTime in Myr (scaled later)
  stellar[i * 4 + 2] = Math.random();     // per-star seed
  stellar[i * 4 + 3] = dustWeight;        // dust-lane carrier weight (0 = none)
}

/**
 * Equilibrium spiral-galaxy disk.
 *
 * Structure: SMBH (particle 0) + Hernquist bulge (20%) + log-spiral arm
 * overdensity (15%) + exponential disk (65%), embedded in an ANALYTIC
 * dark-matter halo (Hernquist profile, applied in the force kernel — pass
 * the same haloMass/haloA here so the rotation curve matches).
 *
 * Kinematics are derived from the actually-sampled mass profile:
 *   1. sample radii for every star
 *   2. measure M_enc(r) and surface density Σ(r) on a radial grid
 *   3. v_c² = G·M_enc/r including halo; κ from the measured curve
 *   4. σ_R set by Toomre Q (σ_R = Q·3.36·G·Σ/κ), σ_φ = σ_R·κ/2Ω,
 *      mean v_φ reduced by asymmetric drift
 *
 * All stellar dynamical masses are 1.0 (well-tested: keeps the velocity
 * grid exact and inter-star forces well-behaved); spectral class lives in
 * colorMass (slot 7).
 */
export function galaxyDisk(params) {
  const {
    N = 10000,
    diskRadius = 50.0,
    diskHeight = 2.0,
    centralMass = 10000.0,
    G = 1.0,
    armCount = 2,
    armTightness = 0.5,
    toomreQ = 1.3,
    haloMass = 0.0,
    haloA = 1.0,
  } = params;

  const R = diskRadius;
  const data = new Float32Array(N * STRIDE);
  const stellar = new Float32Array(N * 4);
  const DYMASS = 1.0;

  const nBulge = Math.floor(N * 0.20);
  const nArm = Math.floor(N * 0.15);
  const aBulge = R * 0.1;        // Hernquist bulge scale
  const hDisk = R * 0.45;        // exponential disk scale length
  const z0 = diskHeight * 0.5;   // sech² scale height
  const r0Spiral = R * 0.08;
  const rArmMin = R * 0.30;

  // ── 1. sample radius + component for every star (index 1..N-1) ───────────
  // comp: 0 = bulge, 1 = arm, 2 = disk
  const radii = new Float32Array(N);
  const comp = new Uint8Array(N);
  for (let i = 1; i < N; i++) {
    if (i <= nBulge) {
      // Hernquist inverse CDF: r = a·s/(1−s), s = √u
      let r;
      do {
        const s = Math.sqrt(Math.random());
        r = aBulge * s / Math.max(1 - s, 1e-6);
      } while (r > R * 0.5);
      radii[i] = r;
      comp[i] = 0;
    } else if (i <= nBulge + nArm) {
      // sqrt-uniform between rArmMin and R (uniform surface density)
      radii[i] = Math.sqrt(rArmMin * rArmMin + Math.random() * (R * R - rArmMin * rArmMin));
      comp[i] = 1;
    } else {
      // exponential disk: p(r) ∝ r·exp(-r/h) = Gamma(2, h)
      let r, guard = 0;
      do {
        r = -hDisk * Math.log((Math.random() || Number.EPSILON) * (Math.random() || Number.EPSILON));
      } while ((r > R || r < R * 0.02) && ++guard < 64);
      radii[i] = Math.min(Math.max(r, R * 0.02), R);
      comp[i] = 2;
    }
  }

  // ── 2. measured mass profile on a radial grid ─────────────────────────────
  const NB = 256;
  const dr = R / NB;
  const counts = new Float64Array(NB);
  for (let i = 1; i < N; i++) {
    const b = Math.min(Math.floor(radii[i] / dr), NB - 1);
    counts[b] += DYMASS;
  }
  // cumulative stellar mass and smoothed surface density
  const mEnc = new Float64Array(NB);
  const sigma = new Float64Array(NB);
  let cum = 0;
  for (let b = 0; b < NB; b++) {
    cum += counts[b];
    mEnc[b] = cum;
    const rMid = (b + 0.5) * dr;
    sigma[b] = counts[b] / (2 * Math.PI * rMid * dr);
  }
  // 5-bin box smooth on Σ to tame sampling noise
  const sigmaS = new Float64Array(NB);
  for (let b = 0; b < NB; b++) {
    let s = 0, n = 0;
    for (let k = Math.max(0, b - 2); k <= Math.min(NB - 1, b + 2); k++) { s += sigma[k]; n++; }
    sigmaS[b] = s / n;
  }

  // ── 3. rotation curve, epicyclic frequency, dispersions on the grid ──────
  const vc = new Float64Array(NB);
  for (let b = 0; b < NB; b++) {
    const r = (b + 0.5) * dr;
    const mHalo = haloMass * r * r / ((r + haloA) * (r + haloA));
    vc[b] = Math.sqrt(G * (centralMass + mEnc[b] + mHalo) / r);
  }
  const sigR = new Float64Array(NB);
  const sigPhi = new Float64Array(NB);
  const vPhiMean = new Float64Array(NB);
  for (let b = 0; b < NB; b++) {
    const r = (b + 0.5) * dr;
    const omega = vc[b] / r;
    // d ln v / d ln r via centered difference
    const b0 = Math.max(0, b - 1), b1 = Math.min(NB - 1, b + 1);
    const dlnv = (Math.log(vc[b1]) - Math.log(vc[b0])) /
                 (Math.log((b1 + 0.5) / Math.max(b0 + 0.5, 0.01)));
    const kappa = Math.sqrt(Math.max(2 * omega * omega * (1 + dlnv), 0.1 * omega * omega));
    // Toomre criterion sets the radial dispersion
    const sR = Math.min(toomreQ * 3.36 * G * sigmaS[b] / kappa, 0.6 * vc[b]);
    sigR[b] = sR;
    sigPhi[b] = sR * kappa / (2 * omega);
    // Asymmetric drift: hotter populations orbit slower than v_c
    const drift2 = sR * sR * (2 * r / hDisk);
    vPhiMean[b] = Math.sqrt(Math.max(vc[b] * vc[b] - drift2, 0.01 * vc[b] * vc[b]));
  }

  const gridAt = (arr, r) => arr[Math.min(Math.floor(r / dr), NB - 1)];

  // ── 4. place stars ────────────────────────────────────────────────────────
  for (let i = 1; i < N; i++) {
    const base = i * STRIDE;
    const r = radii[i];
    const c = comp[i];

    let theta;
    if (c === 1) {
      // log-spiral arm with outward-growing scatter
      const arm = i % armCount;
      const armOffset = arm * (2 * Math.PI / armCount);
      const scatter = 0.04 + 0.09 * (r / R);
      theta = Math.log(r / r0Spiral) / armTightness + armOffset + gaussRandom() * scatter;
    } else {
      theta = Math.random() * 2 * Math.PI;
    }
    const cosT = Math.cos(theta), sinT = Math.sin(theta);

    let x = r * cosT, y = r * sinT, z;
    let vx, vy, vz;

    if (c === 0) {
      // Bulge: quasi-spherical, isotropic dispersion + mild net rotation
      z = gaussRandom() * aBulge * 0.55;
      const sB = Math.sqrt(G * (centralMass + gridAt(mEnc, r) * 0.5) / (3 * (r + aBulge)));
      const vRot = 0.35 * gridAt(vc, r);
      vx = -vRot * sinT + gaussRandom() * sB;
      vy =  vRot * cosT + gaussRandom() * sB;
      vz =  gaussRandom() * sB * 0.8;
    } else {
      // Disk/arm: sech² vertical profile, epicyclic velocity ellipsoid
      const u = Math.min(Math.max(Math.random(), 1e-6), 1 - 1e-6);
      z = z0 * Math.atanh(2 * u - 1) * 0.5;
      const sR = gridAt(sigR, r);
      const vR = gaussRandom() * sR;
      const vPhi = gridAt(vPhiMean, r) + gaussRandom() * gridAt(sigPhi, r);
      const sZ = 0.5 * sR;
      vx = vR * cosT - vPhi * sinT;
      vy = vR * sinT + vPhi * cosT;
      vz = gaussRandom() * sZ;
    }

    // Stellar population: IMF mass + age. Bulge is ancient (8–12 Gyr),
    // disk spans the galaxy's life, arms are newborn (O/B stars alive).
    // Dust rides the cold thin-disk material: strongest along arms,
    // moderate in the disk, none in the bulge.
    const dustW = Math.exp(-Math.abs(z) / Math.max(z0, 0.1));
    if (c === 0)      sampleStar(stellar, i, 8000, 12000, 0);
    else if (c === 1) sampleStar(stellar, i, 0, 50, dustW);
    else              sampleStar(stellar, i, 0, 10000, dustW * 0.55);

    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = DYMASS;
    data[base + 4] = vx;
    data[base + 5] = vy;
    data[base + 6] = vz;
    data[base + 7] = 0;
  }

  // ── particle 0: supermassive central mass (black hole — stellarMass 0) ────
  data[0] = 0; data[1] = 0; data[2] = 0;
  data[3] = centralMass;
  data[4] = 0; data[5] = 0; data[6] = 0;
  data[7] = 0;
  stellar[0] = 0; stellar[1] = 0; stellar[2] = 0; stellar[3] = 0;

  return { data, stellar };
}

export function uniformSphere(params) {
  const { N = 5000, diskRadius = 30.0, starMass = 1.0 } = params;
  const radius = diskRadius;
  const data = new Float32Array(N * STRIDE);
  const stellar = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const base = i * STRIDE;
    let x, y, z;
    do {
      x = (Math.random() * 2 - 1) * radius;
      y = (Math.random() * 2 - 1) * radius;
      z = (Math.random() * 2 - 1) * radius;
    } while (x * x + y * y + z * z > radius * radius);
    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = starMass;
    sampleStar(stellar, i, 0, 5000);
    // zero velocity (indices 4-6 default to 0)
  }
  return { data, stellar };
}

export function twoGalaxyCollision(params) {
  const {
    N = 10000,
    diskRadius = 30.0,
    diskHeight = 1.5,
    centralMass = 800.0,
    toomreQ = 1.3,
    G = 1.0,
    separation = 80.0,
    approachVel = 0.8,
  } = params;

  const halfN = Math.floor(N / 2);
  // No analytic halo in the collision preset (it would be pinned at the
  // origin); each galaxy is self-bound by stars + central mass.
  const paramsA = { N: halfN, diskRadius, diskHeight, centralMass, toomreQ, G, haloMass: 0 };
  const paramsB = { ...paramsA, N: N - halfN };

  const A = galaxyDisk(paramsA);
  const B = galaxyDisk(paramsB);
  const diskA = A.data, diskB = B.data;

  const data = new Float32Array(N * STRIDE);
  const stellar = new Float32Array(N * 4);
  stellar.set(A.stellar.subarray(0, halfN * 4), 0);
  stellar.set(B.stellar.subarray(0, (N - halfN) * 4), halfN * 4);

  // Galaxy A: offset -separation/2 along x, moving +x
  for (let i = 0; i < halfN; i++) {
    const src = i * STRIDE;
    const dst = i * STRIDE;
    data[dst + 0] = diskA[src + 0] - separation / 2;
    data[dst + 1] = diskA[src + 1];
    data[dst + 2] = diskA[src + 2];
    data[dst + 3] = diskA[src + 3];
    data[dst + 4] = diskA[src + 4] + approachVel;
    data[dst + 5] = diskA[src + 5];
    data[dst + 6] = diskA[src + 6];
  }

  // Galaxy B: offset +separation/2 along x, tilted 30°, moving -x
  const tilt = Math.PI / 6;
  const cosT = Math.cos(tilt), sinT = Math.sin(tilt);
  for (let i = 0; i < N - halfN; i++) {
    const src = i * STRIDE;
    const dst = (halfN + i) * STRIDE;
    const bx = diskB[src + 0];
    const by = diskB[src + 1];
    const bz = diskB[src + 2];
    data[dst + 0] = bx + separation / 2;
    data[dst + 1] = by * cosT - bz * sinT;
    data[dst + 2] = by * sinT + bz * cosT;
    data[dst + 3] = diskB[src + 3];
    data[dst + 4] = diskB[src + 4] - approachVel;
    data[dst + 5] = diskB[src + 5];
    data[dst + 6] = diskB[src + 6];
  }

  return { data, stellar };
}
