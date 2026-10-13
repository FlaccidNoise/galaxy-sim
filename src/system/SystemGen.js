// Deterministic procedural solar-system generation. Pure module — no GPU, no
// DOM. Distances in AU-ish units; body radii are visual (inflated so planets
// read at orbit distances), not to scale.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Appearance palette ids — must match the switch in system.wgsl
export const KIND = { SUN: 0, SCORCHED: 1, VENUSIAN: 2, TERRAN: 3, ICY: 4, GAS: 5, ICE_GIANT: 6, ROCKY: 7 };

// Sim-time units; inner planets complete an orbit in ~10 s of real time
const ORBIT_SPEED = 0.08;

export function orbitRate(a) {
  return 2 * Math.PI * ORBIT_SPEED / Math.pow(a, 1.5);
}

/**
 * Body centres at system time t, in SystemRenderer's buffer order: the star,
 * then each planet followed by its moons. Mirrors orbitPos/bodyCenter in
 * system.wgsl. planet is null for the star; moons carry their parent planet.
 */
export function bodyPositions(system, t) {
  const orbit = (a, phase, rate) => {
    const ang = phase + rate * t;
    return [Math.cos(ang) * a, Math.sin(ang) * a, 0];
  };
  const out = [{ pos: [0, 0, 0], radius: system.star.radius, planet: null, moon: false }];
  for (const p of system.planets) {
    const c = orbit(p.a, p.phase, p.rate);
    out.push({ pos: c, radius: p.radius, planet: p, moon: false });
    for (const m of p.moons) {
      const o = orbit(m.a, m.phase, m.rate);
      out.push({ pos: [c[0] + o[0], c[1] + o[1], 0], radius: m.radius, planet: p, moon: true });
    }
  }
  return out;
}

export function spectralClass(tempK) {
  if (tempK >= 30000) return 'O';
  if (tempK >= 10000) return 'B';
  if (tempK >= 7500) return 'A';
  if (tempK >= 6000) return 'F';
  if (tempK >= 5200) return 'G';
  if (tempK >= 3700) return 'K';
  return 'M';
}

export function generateSystem(stellarMass, birthTime, seed) {
  // Seed from the star's intrinsic float bits — stable across tree re-sorts
  const f = new Float32Array([stellarMass, birthTime, seed]);
  const u = new Uint32Array(f.buffer);
  const rand = mulberry32(u[0] ^ Math.imul(u[1], 0x9E3779B9) ^ Math.imul(u[2], 0x85EBCA6B));

  const M = Math.max(stellarMass, 0.08);
  const lum = Math.pow(M, 3.5);
  const tempK = 5778 * Math.pow(M, 0.54);   // matches starState() in render.wgsl
  const sqrtL = Math.sqrt(lum);
  const frostLine = 2.7 * sqrtL;

  const star = {
    mass: M, lum, tempK,
    class: spectralClass(tempK),
    radius: 0.35 * Math.pow(M, 0.8),
  };

  // Hot massive stars: sparse scorched systems. Red dwarfs: tight and small
  // (orbit spacing scales with √L).
  const maxPlanets = M >= 8 ? 3 : 8;
  const count = Math.min(Math.floor(rand() * 9), maxPlanets);

  const planets = [];
  let a = (0.3 + 0.2 * rand()) * sqrtL + star.radius * 3;
  for (let i = 0; i < count; i++) {
    const rocky = a < frostLine;
    let kind, radius;
    if (rocky) {
      radius = 0.02 + 0.05 * rand();
      const hz = a >= 0.95 * sqrtL && a <= 1.4 * sqrtL;
      kind = hz ? KIND.TERRAN
           : a < 0.5 * sqrtL ? KIND.SCORCHED
           : a < 0.95 * sqrtL ? KIND.VENUSIAN
           : KIND.ICY;
    } else {
      kind = rand() < 0.6 ? KIND.GAS : KIND.ICE_GIANT;
      radius = kind === KIND.GAS ? 0.15 + 0.2 * rand() : 0.1 + 0.08 * rand();
    }
    radius = Math.min(radius, 0.12 * a);   // tight red-dwarf orbits: no engulfing the star or neighbours
    const planet = {
      a, phase: rand() * 2 * Math.PI, rate: orbitRate(a),
      radius, kind, seed: rand(), p1: rand(), p2: rand(),
      rings: null, moons: [],
    };
    if (kind === KIND.GAS && rand() < 0.35) {
      planet.rings = { inner: radius * 1.4, outer: radius * (2.2 + rand()), seed: rand() };
    }
    const giant = kind === KIND.GAS || kind === KIND.ICE_GIANT;
    const nMoons = giant ? 1 + Math.floor(rand() * 4)
                 : (radius > 0.045 && rand() < 0.4 ? 1 : 0);
    let ma = radius * (2.5 + rand());
    for (let m = 0; m < nMoons; m++) {
      planet.moons.push({
        a: ma, phase: rand() * 2 * Math.PI,
        rate: orbitRate(ma) * 8,      // fast — visible motion at moon scale
        radius: radius * (0.12 + 0.15 * rand()),
      });
      ma *= 1.6 + 0.4 * rand();
    }
    planets.push(planet);
    a *= 1.4 + 0.6 * rand();
  }

  // Belt: 70% of systems, in the widest orbit gap (or beyond the planets)
  let belt = null;
  if (rand() < 0.7) {
    let beltA = planets.length ? planets[planets.length - 1].a * 1.5 : 2 * sqrtL;
    let widest = 0;
    for (let i = 0; i + 1 < planets.length; i++) {
      const ratio = planets[i + 1].a / planets[i].a;
      if (ratio > 1.8 && ratio > widest) {
        widest = ratio;
        beltA = Math.sqrt(planets[i].a * planets[i + 1].a);
      }
    }
    belt = { a: beltA, width: beltA * 0.15, count: 3000, seed: Math.floor(rand() * 0xffffffff) };
  }

  return { star, planets, belt };
}
