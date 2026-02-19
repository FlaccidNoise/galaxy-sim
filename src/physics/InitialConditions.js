// Particle layout: [x, y, z, mass, vx, vy, vz, pad] per particle (stride = 8 floats)
const STRIDE = 8;

function gaussRandom() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function galaxyDisk(params) {
  const {
    N = 10000,
    diskRadius = 50.0,
    diskHeight = 2.0,
    centralMass = 10000.0,
    velDispersion = 0.2,
    G = 1.0,
    armCount = 2,
    armTightness = 0.5,
  } = params;

  const data = new Float32Array(N * STRIDE);

  // Three distinct stellar populations, explicitly seeded for visual clarity:
  //   25% bulge  — tight Gaussian, old K/M stars (orange-red)
  //   20% arms   — directly placed on logarithmic spirals, young OB stars (blue-white)
  //   55% disk   — random angle, old G/K stars (yellow-orange background)
  const nBulge = Math.floor(N * 0.20);
  const nArm   = Math.floor(N * 0.15);
  // nDisk fills the rest (particle 0 is the central mass, so N-1 total non-central)

  const r0Spiral = diskRadius * 0.08;  // spiral reference radius
  // Arms start at r=0.30*R — at smaller r, the log spiral wraps nearly 360°, causing
  // arm stars to appear as diffuse "fog" rather than distinct winding features.
  // From r=0.30*R outward, each arm sweeps ~125° total, appearing as clear arcs.
  const rArmMin  = diskRadius * 0.30;
  const rArmMin2 = rArmMin * rArmMin;
  const rArmMax2 = diskRadius * diskRadius;

  let idx = 1;  // idx=0 is the central mass, filled last

  // KEY DESIGN DECISION: all stellar masses are set to 1.0 for dynamics.
  // With arm stars at 5-12 M☉, their combined self-gravity rivals the central mass
  // and causes them to clump into clusters rather than maintaining smooth arms.
  // With mass=1.0 for all stars, inter-stellar forces are <1% of the central force
  // and the velocity formula (massEnc = centralMass + N*r/R) is exactly correct.
  // Spectral color type is stored in the 'pad' slot (index 7) as a separate "color mass".
  const DYMASS = 1.0;  // dynamical mass for all non-central stars

  // ── BULGE: tight Gaussian, orange-red K/M-type stars ─────────────────────────
  for (let i = 0; i < nBulge && idx < N; i++, idx++) {
    const base      = idx * STRIDE;
    const r         = Math.abs(gaussRandom()) * diskRadius * 0.12;
    const theta     = Math.random() * 2 * Math.PI;
    const colorMass = 0.1 + Math.pow(Math.random(), 2.5) * 0.55;  // 0.1–0.65 M☉ K/M

    const massEnc = centralMass + N * (r / diskRadius);
    const vc = r > 0.01 ? Math.sqrt(G * massEnc / r) : 0;

    data[base + 0] = r * Math.cos(theta);
    data[base + 1] = r * Math.sin(theta);
    data[base + 2] = gaussRandom() * diskHeight * 0.4;  // thin bulge
    data[base + 3] = DYMASS;
    data[base + 4] = -vc * Math.sin(theta) + gaussRandom() * velDispersion;
    data[base + 5] =  vc * Math.cos(theta) + gaussRandom() * velDispersion;
    data[base + 6] =  gaussRandom() * velDispersion * 0.1;
    data[base + 7] = colorMass;
  }

  // ── SPIRAL ARMS: directly placed on logarithmic spirals, blue OB stars ────────
  const nPerArm = Math.floor(nArm / armCount);
  for (let arm = 0; arm < armCount; arm++) {
    const armOffset = arm * (2 * Math.PI / armCount);
    const count = (arm < armCount - 1) ? nPerArm : nArm - arm * nPerArm;
    for (let i = 0; i < count && idx < N; i++, idx++) {
      const base      = idx * STRIDE;
      // sqrt-sample for uniform projected density (equal stars per unit disk area)
      const r         = Math.sqrt(rArmMin2 + Math.random() * (rArmMax2 - rArmMin2));
      const thetaArm  = Math.log(r / r0Spiral) / armTightness + armOffset;
      // Scatter grows outward: inner arm tight, outer arm diffuses slightly.
      // At r=0.30*R: σ≈0.06 rad (arm width ~2.7 units), r=R: σ≈0.13 rad (~13 units).
      // Each arm sweeps 138° total; with 3σ padding the arm spans ~180° → clear arcs.
      const scatter   = 0.04 + 0.09 * (r / diskRadius);
      const theta     = thetaArm + gaussRandom() * scatter;
      // All arm stars are O/B type — young, luminous, hot.  Keeping fraction low (15%)
      // means individual blue stars are clearly separated rather than piling into white blobs.
      const colorMass = 2.0 + Math.pow(Math.random(), 1.5) * 8.0;  // 2–10 M☉  B/O (blue)

      const massEnc = centralMass + N * (r / diskRadius);
      const vc = r > 0.01 ? Math.sqrt(G * massEnc / r) : 0;

      data[base + 0] = r * Math.cos(theta);
      data[base + 1] = r * Math.sin(theta);
      data[base + 2] = gaussRandom() * diskHeight;
      data[base + 3] = DYMASS;
      data[base + 4] = -vc * Math.sin(theta) + gaussRandom() * velDispersion;
      data[base + 5] =  vc * Math.cos(theta) + gaussRandom() * velDispersion;
      data[base + 6] =  gaussRandom() * velDispersion * 0.1;
      data[base + 7] = colorMass;
    }
  }

  // ── DISK: exponential surface-density profile, G/K stars ──────────────────────────
  // Real galaxy disks follow Σ(r) ∝ exp(-r/h). Sampling the 2D profile means
  // sampling r from p(r) ∝ r·exp(-r/h), which is Gamma(2,h): r = -h·ln(U₁·U₂).
  // Scale length h = 0.45*R gives a bright center fading to ~10% at the outer edge.
  const h = diskRadius * 0.45;
  while (idx < N) {
    const base = idx * STRIDE;
    let r;
    do {
      r = -h * Math.log(Math.random() * Math.random());
    } while (r > diskRadius || r < diskRadius * 0.02);
    const theta     = Math.random() * 2 * Math.PI;
    const colorMass = 0.35 + Math.pow(Math.random(), 1.5) * 0.65;  // 0.35–1.0 M☉ G/K

    const massEnc = centralMass + N * (r / diskRadius);
    const vc = r > 0.01 ? Math.sqrt(G * massEnc / r) : 0;

    data[base + 0] = r * Math.cos(theta);
    data[base + 1] = r * Math.sin(theta);
    data[base + 2] = gaussRandom() * diskHeight;
    data[base + 3] = DYMASS;
    data[base + 4] = -vc * Math.sin(theta) + gaussRandom() * velDispersion;
    data[base + 5] =  vc * Math.cos(theta) + gaussRandom() * velDispersion;
    data[base + 6] =  gaussRandom() * velDispersion * 0.1;
    data[base + 7] = colorMass;
    idx++;
  }

  // ── PARTICLE 0: supermassive central mass ─────────────────────────────────────
  data[0] = 0; data[1] = 0; data[2] = 0;
  data[3] = centralMass;
  data[4] = 0; data[5] = 0; data[6] = 0;
  data[7] = 0;

  return data;
}

export function uniformSphere(params) {
  const { N = 5000, radius = 30.0, starMass = 1.0 } = params;
  const data = new Float32Array(N * STRIDE);
  for (let i = 0; i < N; i++) {
    const base = i * STRIDE;
    let x, y, z;
    do {
      x = (Math.random() * 2 - 1) * radius;
      y = (Math.random() * 2 - 1) * radius;
      z = (Math.random() * 2 - 1) * radius;
    } while (x*x + y*y + z*z > radius * radius);
    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = starMass;
    // zero velocity (indices 4-6 default to 0)
  }
  return data;
}

export function twoGalaxyCollision(params) {
  const {
    N = 10000,
    diskRadius = 30.0,
    diskHeight = 1.5,
    centralMass = 800.0,
    starMass = 1.0,
    velDispersion = 0.3,
    G = 1.0,
    separation = 80.0,
    approachVel = 0.8,
  } = params;

  const halfN = Math.floor(N / 2);
  const paramsA = { N: halfN, diskRadius, diskHeight, centralMass, starMass, velDispersion, G };
  const paramsB = { ...paramsA, N: N - halfN };

  const diskA = galaxyDisk(paramsA);
  const diskB = galaxyDisk(paramsB);

  const data = new Float32Array(N * STRIDE);

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
    data[dst + 7] = 0;
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
    data[dst + 7] = 0;
  }

  return data;
}
