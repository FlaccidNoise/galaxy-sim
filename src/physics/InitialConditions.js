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
    centralMass = 1000.0,
    starMass = 1.0,   // unused directly — mass now sampled per-star
    velDispersion = 0.5,
    G = 1.0,
    armCount = 2,
    armTightness = 0.35,
  } = params;

  const data = new Float32Array(N * STRIDE);

  // 20% of stars form a tight central bulge (old, low-mass, red/yellow)
  const nBulge = Math.floor(N * 0.20);

  for (let i = 1; i < N; i++) {
    const base = i * STRIDE;
    const isBulge = i <= nBulge;

    let r, theta, armIndex, armOffset, r0Spiral, thetaArm;

    if (isBulge) {
      // Tight Gaussian bulge
      r     = Math.abs(gaussRandom()) * diskRadius * 0.15;
      theta = Math.random() * 2 * Math.PI;
    } else {
      // Exponential radial CDF (same as before)
      const r0 = diskRadius / 3.0;
      const u  = Math.random();
      r = -r0 * Math.log(1.0 - u * (1.0 - Math.exp(-diskRadius / r0)));

      // Assign to an arm
      armIndex  = (i - nBulge - 1) % armCount;
      armOffset = armIndex * (2 * Math.PI / armCount);

      // Logarithmic spiral centerline angle at radius r
      r0Spiral  = diskRadius * 0.1;
      thetaArm  = Math.log(Math.max(r, r0Spiral) / r0Spiral) / armTightness + armOffset;

      // Scatter around arm — wider at larger radii
      const scatter = 0.15 + (r / diskRadius) * 0.40;
      theta = thetaArm + gaussRandom() * scatter;
    }

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);
    const z = gaussRandom() * diskHeight;

    // Stellar mass: arm stars → upper IMF (hot blue), bulge → lower IMF (cool red)
    let mass;
    if (isBulge) {
      // Lower IMF: 0.2 – 1.2 M☉, weighted toward low end
      mass = 0.2 + Math.pow(Math.random(), 2.0) * 1.0;
    } else {
      // Check angular distance from arm centerline
      const diff = Math.abs(theta - thetaArm);
      const dTheta = Math.min(diff, 2 * Math.PI - diff);

      if (dTheta < 0.4) {
        // In arm: upper IMF — young hot stars, 1 – 20 M☉
        mass = 1.0 + Math.pow(Math.random(), 2.5) * 19.0;
      } else {
        // Inter-arm disk: mixed, 0.3 – 2.0 M☉
        mass = 0.3 + Math.pow(Math.random(), 1.5) * 1.7;
      }
    }

    // Circular velocity
    const massEnc = centralMass + mass * N * (r / diskRadius);
    const vc = r > 0.01 ? Math.sqrt(G * massEnc / r) : 0;

    const vx = -vc * Math.sin(theta) + gaussRandom() * velDispersion;
    const vy =  vc * Math.cos(theta) + gaussRandom() * velDispersion;
    const vz =  gaussRandom() * velDispersion * 0.2;

    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = mass;
    data[base + 4] = vx;
    data[base + 5] = vy;
    data[base + 6] = vz;
    data[base + 7] = 0;
  }

  // Particle 0: central mass (black hole / dense core)
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
