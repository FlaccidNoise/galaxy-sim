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
    starMass = 1.0,
    velDispersion = 0.5,
    G = 1.0,
  } = params;

  const data = new Float32Array(N * STRIDE);

  for (let i = 0; i < N; i++) {
    const base = i * STRIDE;

    // Exponential radial distribution via inverse CDF sampling
    // CDF of exp(-r/r0): r = -r0 * ln(1 - u), r0 = diskRadius / 3
    const r0 = diskRadius / 3.0;
    const u = Math.random();
    const r = -r0 * Math.log(1.0 - u * (1.0 - Math.exp(-diskRadius / r0)));
    const theta = Math.random() * 2 * Math.PI;

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);
    // Gaussian scale height
    const z = gaussRandom() * diskHeight;

    // Circular velocity: v_c = sqrt(G * M_enc / r)
    // Simplified enclosed mass: central mass + stars inside r
    const massEnc = centralMass + starMass * N * (r / diskRadius);
    const vc = r > 0.01 ? Math.sqrt(G * massEnc / r) : 0;

    // Tangential velocity (perpendicular to radial direction in xy plane)
    const vx = -vc * Math.sin(theta) + gaussRandom() * velDispersion;
    const vy =  vc * Math.cos(theta) + gaussRandom() * velDispersion;
    const vz = gaussRandom() * velDispersion * 0.2;

    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = starMass;
    data[base + 4] = vx;
    data[base + 5] = vy;
    data[base + 6] = vz;
    data[base + 7] = 0; // pad
  }

  // Place central mass as particle 0
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
