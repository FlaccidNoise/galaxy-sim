// Deterministic procedural solar-system generation. Pure module — no GPU, no
// DOM. Orbits are laid out in physical AU (planet.au) and classified there;
// rendered distances (planet.a) are the same ladder scaled up so the innermost
// orbit clears the inflated visual star. Body radii are visual, not to scale.

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
export const KIND = { SUN: 0, SCORCHED: 1, VENUSIAN: 2, TERRAN: 3, ICY: 4, GAS: 5, ICE_GIANT: 6, ROCKY: 7, SUB_NEPTUNE: 8 };

// Sim-time units; inner planets complete an orbit in ~10 s of real time
const ORBIT_SPEED = 0.08;

// Keplerian, but capped: the fastest planet in any system takes at least ~6 s
// of real time per orbit (0.6 sim units/s), so tight red-dwarf orbits stay watchable.
const MAX_RATE = 1.75;

export function orbitRate(a) {
  return Math.min(2 * Math.PI * ORBIT_SPEED / Math.pow(a, 1.5), MAX_RATE);
}

/**
 * Position on a Kepler ellipse (semi-major axis a, eccentricity e, pericentre
 * direction w) at mean anomaly phase + rate·t. Three fixed Newton steps, the
 * same as orbitPos in system.wgsl, so CPU picking lands where the GPU draws.
 */
export function orbitPos(a, phase, rate, e, w, t) {
  const M = phase + rate * t;
  let E = M + e * Math.sin(M);
  for (let i = 0; i < 3; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const x = a * (Math.cos(E) - e), y = a * Math.sqrt(1 - e * e) * Math.sin(E);
  return [x * Math.cos(w) - y * Math.sin(w), x * Math.sin(w) + y * Math.cos(w), 0];
}

/**
 * Body centres at system time t, in SystemRenderer's buffer order: the star,
 * then each planet followed by its moons. Mirrors bodyCenter in system.wgsl.
 * planet is null for the star; moons carry their parent planet.
 */
export function bodyPositions(system, t) {
  const out = [{ pos: [0, 0, 0], radius: system.star.radius, planet: null, moon: false }];
  for (const p of system.planets) {
    const c = orbitPos(p.a, p.phase, p.rate, p.e, p.w, t);
    out.push({ pos: c, radius: p.radius, planet: p, moon: false });
    for (const m of p.moons) {
      const o = orbitPos(m.a, m.phase, m.rate, 0, 0, t);
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

// Main-sequence mass–luminosity and mass–temperature relations, piecewise fits
// to Pecaut & Mamajek's dwarf table. Both are mirrored in render.wgsl.
export function luminosity(M) {
  if (M < 0.43) return 0.23 * Math.pow(M, 2.3);
  if (M < 2) return Math.pow(M, 4);
  return 1.4 * Math.pow(M, 3.5);
}

/**
 * Current state of a star aged ageMyr: mirrors starState() in render.wgsl.
 * Visual radius grows as √(R/R_ms) so a red giant swells past the inner
 * orbits without filling the view. lum feeds planet temperatures. reach is
 * the largest radius the star has had: what it swallowed stays swallowed.
 */
export function temperature(M) {
  return M < 0.57 ? 4100 * Math.pow(M / 0.57, 0.25) : 5778 * Math.pow(M, 0.6);
}

export function evolveStar(M, ageMyr) {
  const tMS = 10000 * Math.pow(M, -2.5);
  const tGiant = tMS * 0.12 + 15;
  const tempMS = temperature(M);
  const lumMS = luminosity(M);
  const rMS = 0.35 * Math.pow(M, 0.8);
  if (ageMyr < tMS) return { phase: 'main', tempK: tempMS, lum: lumMS, radius: rMS, reach: rMS, class: spectralClass(tempMS) };
  const giantRadius = f => {   // L = R²T⁴, with L and T at giant-phase fraction f
    const tempK = tempMS + (3400 - tempMS) * Math.min(f * 2.5, 1);
    return rMS * Math.sqrt(Math.sqrt(1 + 59 * f * f) * Math.pow(tempMS / tempK, 2));
  };
  if (ageMyr < tMS + tGiant) {
    const f = (ageMyr - tMS) / tGiant;
    const tempK = tempMS + (3400 - tempMS) * Math.min(f * 2.5, 1);
    const radius = giantRadius(f);
    return { phase: 'giant', tempK, lum: lumMS * (1 + 59 * f * f), radius, reach: radius, class: spectralClass(tempK) };
  }
  const reach = giantRadius(1);
  if (M >= 8) return { phase: 'neutron star', tempK: 40000, lum: 0.0005, radius: rMS * 0.08, reach, class: 'NS' };
  const tempK = 14000 / (1 + (ageMyr - tMS - tGiant) * 2e-4);
  return { phase: 'white dwarf', tempK, lum: 0.004, radius: rMS * 0.15, reach, class: 'WD' };
}

export function generateSystem(stellarMass, birthTime, seed, ageMyr = -birthTime) {
  // Seed from the star's intrinsic float bits — stable across tree re-sorts.
  // Age must not enter the seed: the same star reselected later keeps its planets.
  const f = new Float32Array([stellarMass, birthTime, seed]);
  const u = new Uint32Array(f.buffer);
  const rand = mulberry32(u[0] ^ Math.imul(u[1], 0x9E3779B9) ^ Math.imul(u[2], 0x85EBCA6B));

  const M = Math.max(stellarMass, 0.08);
  const lum = luminosity(M);   // at formation: sets the disc's frost line and zones
  const sqrtL = Math.sqrt(lum);
  const frostLine = 2.7 * sqrtL;
  const rMS = 0.35 * Math.pow(M, 0.8);

  const star = { mass: M, ...evolveStar(M, ageMyr) };

  // Hot massive stars: sparse scorched systems. Red dwarfs: tight and small
  // (orbit spacing scales with √L).
  const maxPlanets = M >= 8 ? 3 : 8;
  const count = Math.min(Math.floor(rand() * 9), maxPlanets);
  // Giant cores need a massive, metal-rich disc: odds per outer slot scale with
  // stellar mass and halve for the oldest (metal-poor) stars. birthTime is Myr before t=0.
  const ageGyr = Math.min(Math.max(-birthTime / 1000, 0), 12);
  const pGiant = Math.min(1, 0.5 * M) * (1 - ageGyr / 24);
  const warmJupiter = M > 0.5 && rand() < 0.02;   // migrated giant at the innermost slot

  let planets = [];
  let au = (0.1 + 0.3 * rand()) * sqrtL;   // innermost orbit, physical AU
  const scale = Math.max(1, rMS * 3 / au);   // rendered units per AU
  for (let i = 0; i < count; i++) {
    const a = au * scale;
    let kind, radius;
    if (i === 0 && warmJupiter) {
      kind = KIND.GAS;
      radius = 0.15 + 0.2 * rand();
    } else if (au < frostLine && rand() < 0.35) {
      kind = KIND.SUB_NEPTUNE;   // the commonest planet class: a small core under a thick haze
      radius = 0.07 + 0.03 * rand();
    } else if (au < frostLine) {
      radius = 0.02 + 0.05 * rand();
      const hz = au >= 0.95 * sqrtL && au <= 1.4 * sqrtL;
      const small = radius < 0.035;   // Mars-sized: too little gravity to keep an atmosphere
      kind = hz ? (small ? KIND.ROCKY : KIND.TERRAN)
           : au < 0.5 * sqrtL ? KIND.SCORCHED
           : au < 0.95 * sqrtL ? (small ? KIND.SCORCHED : KIND.VENUSIAN)
           : KIND.ICY;
    } else if (rand() < pGiant) {
      // Gas giants form just past the frost line; ice giants dominate further out
      kind = rand() < Math.min(0.9, 0.1 + (au / frostLine - 1) / 4) ? KIND.ICE_GIANT : KIND.GAS;
      radius = kind === KIND.GAS ? 0.15 + 0.2 * rand() : 0.1 + 0.08 * rand();
    } else {
      kind = KIND.ICY;   // failed core: an ice dwarf
      radius = 0.02 + 0.04 * rand();
    }
    radius = Math.min(radius, 0.12 * a);   // tight red-dwarf orbits: no engulfing the star or neighbours
    const planet = {
      a, au, teq: 255 * Math.pow(star.lum, 0.25) / Math.sqrt(au),   // K today, Earth-like albedo
      phase: rand() * 2 * Math.PI, rate: orbitRate(a),
      e: rand(), w: rand() * 2 * Math.PI,   // e is a uniform draw here; mapped once neighbours are known
      radius, kind, seed: rand(), p1: rand(), p2: rand(),
      rings: null, moons: [],
    };
    if (kind === KIND.GAS && !(i === 0 && warmJupiter) && rand() < 0.35) {   // close-in rings don't survive
      planet.rings = { inner: radius * 1.4, outer: radius * (2.2 + rand()), seed: rand() };
    }
    const giant = kind === KIND.GAS || kind === KIND.ICE_GIANT;
    const nMoons = giant ? 1 + Math.floor(rand() * 4)
                 : (radius > 0.045 && rand() < 0.4 ? 1 : 0);
    let ma = radius * (2.5 + rand());
    for (let m = 0; m < nMoons; m++) {
      planet.moons.push({
        a: ma, phase: rand() * 2 * Math.PI,
        rate: Math.min(orbitRate(ma) * 8, 2 * MAX_RATE),   // fast, but at most ~3 s per orbit
        radius: radius * (0.12 + 0.15 * rand()),
      });
      ma *= 1.6 + 0.4 * rand();
    }
    planets.push(planet);
    au *= 1.4 + 0.6 * rand();
  }

  // Eccentricity: Rayleigh, small in packed systems and larger for isolated
  // planets, then shrunk so neighbouring orbits never cross.
  planets.forEach((p, i) => {
    const packed = (i > 0 && p.au < 1.7 * planets[i - 1].au) || (i + 1 < planets.length && planets[i + 1].au < 1.7 * p.au);
    p.e = Math.min((packed ? 0.04 : 0.12) * Math.sqrt(-2 * Math.log(1 - p.e)), 0.4);
  });
  for (let i = 0; i + 1 < planets.length; i++) {
    const p = planets[i], q = planets[i + 1];
    const f = (q.a - p.a - p.radius - q.radius) / (p.a * p.e + q.a * q.e);
    if (f < 1) { p.e *= 0.9 * f; q.e *= 0.9 * f; }
  }
  // Belt: 70% of systems, in the widest orbit gap (or beyond the planets),
  // placed on the formation ladder so it does not jump as the star ages
  const last = planets[planets.length - 1];
  let beltA = last ? last.a * (1 + last.e) * 1.5 : 2 * sqrtL * scale;
  let widest = 0;
  for (let i = 0; i + 1 < planets.length; i++) {
    const apo = planets[i].a * (1 + planets[i].e), peri = planets[i + 1].a * (1 - planets[i + 1].e);
    const ratio = peri / apo;
    if (ratio > 1.8 && ratio > widest) {
      widest = ratio;
      beltA = Math.sqrt(apo * peri);
    }
  }
  const hasBelt = rand() < 0.7, beltSeed = Math.floor(rand() * 0xffffffff);
  const belt = hasBelt && beltA > star.reach * 1.5 ? { a: beltA, width: beltA * 0.15, count: 3000, seed: beltSeed } : null;

  // A swollen giant star engulfs its inner planets, for good
  planets = planets.filter(p => p.a * (1 - p.e) > star.reach * 1.3);
  if (star.phase !== 'main') {   // an evolved star boils or freezes the worlds that had air
    for (const p of planets) {
      if (p.kind !== KIND.TERRAN && p.kind !== KIND.VENUSIAN) continue;
      if (p.teq > 400) p.kind = KIND.SCORCHED;
      else if (p.teq < 180) p.kind = KIND.ICY;
    }
  }

  return { star, planets, belt, frost: frostLine * scale };   // frost: rendered units
}
