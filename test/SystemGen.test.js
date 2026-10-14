import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, generateSystem, evolveStar, luminosity, temperature, spectralClass, KIND, bodyPositions, orbitPos } from '../src/system/SystemGen.js';

test('mulberry32 is deterministic and in [0,1)', () => {
  const a = mulberry32(42), b = mulberry32(42);
  for (let i = 0; i < 1000; i++) {
    const x = a();
    assert.equal(x, b());
    assert.ok(x >= 0 && x < 1);
  }
});

test('generateSystem is deterministic for identical stellar records', () => {
  assert.deepEqual(generateSystem(1.0, -4500, 0.7315), generateSystem(1.0, -4500, 0.7315));
});

test('different seeds give different systems', () => {
  assert.notDeepEqual(generateSystem(1.0, -4500, 0.7315), generateSystem(1.0, -4500, 0.7316));
});

test('giants form past the frost line, except a lone warm Jupiter at the innermost slot', () => {
  let warm = 0;
  for (let i = 0; i < 2000; i++) {
    const sys = generateSystem(0.5 + i * 0.001, -i * 13, (i * 7 % 1000) / 1000, 500);
    const frost = 2.7 * Math.sqrt(luminosity(sys.star.mass));   // the disc's frost line, at formation
    sys.planets.forEach((p, j) => {
      if (p.kind !== KIND.GAS && p.kind !== KIND.ICE_GIANT) return;
      if (p.au < frost) { assert.equal(j, 0); assert.equal(p.kind, KIND.GAS); warm++; }
    });
  }
  assert.ok(warm > 0 && warm < 100, `${warm} warm Jupiters in 2000 systems`);
});

test('giants vanish above 3 M☉ (Reffert 2015) and a few systems carry a hot Jupiter', () => {
  let heavy = 0, hot = 0, hotLonely = true, hotInside = true;
  for (let i = 0; i < 2000; i++) {
    const big = generateSystem(3 + i * 0.002, -i, i / 2000, 10);
    if (big.planets.some(p => p.kind === KIND.GAS || p.kind === KIND.ICE_GIANT)) heavy++;
    const sys = generateSystem(0.6 + i * 0.0005, -i * 3, (i * 7 % 1000) / 1000, 500);
    const sqrtL = Math.sqrt(luminosity(sys.star.mass));
    const hj = sys.planets[0];
    if (hj && hj.kind === KIND.GAS && hj.au < 0.1 * sqrtL) {
      hot++;
      if (sys.planets.length > 3) hotLonely = false;
      if (hj.e > 0.1 || hj.rings) hotInside = false;
      assert.ok(hj.a - hj.radius > sys.star.radius);
    }
  }
  assert.equal(heavy, 0);
  assert.ok(hot >= 5 && hot <= 60, `${hot} hot Jupiters in 2000 systems`);
  assert.ok(hotLonely && hotInside);
});

test('giants are rare around red dwarfs and old stars', () => {
  const giantShare = (M, birth) => {
    let n = 0;
    for (let i = 0; i < 500; i++) {
      if (generateSystem(M, birth - i, i / 500, 500).planets.some(p => p.kind === KIND.GAS || p.kind === KIND.ICE_GIANT)) n++;
    }
    return n / 500;
  };
  assert.ok(giantShare(0.2, -3000) < giantShare(1.0, -3000) / 2);
  assert.ok(giantShare(1.0, -11000) < giantShare(1.0, -1000));
});

test('rendered orbits are the physical ladder under one scale, and red dwarfs get rocky planets', () => {
  let rocky = 0, systems = 0;
  for (let i = 0; i < 300; i++) {
    const sys = generateSystem(0.1 + i * 0.001, -i * 5, (i * 11 % 100) / 100);
    const scale = sys.planets[0]?.a / sys.planets[0]?.au;
    for (const p of sys.planets) assert.ok(Math.abs(p.a / p.au - scale) < 1e-9);
    if (sys.planets.length) {
      systems++;
      if (sys.planets.some(p => p.kind <= KIND.ICY)) rocky++;
    }
  }
  assert.ok(rocky / systems > 0.8, `only ${rocky}/${systems} red-dwarf systems have a rocky planet`);
});

test('orbits strictly increase; moons stay outside the planet; ≤4 moons', () => {
  for (let i = 0; i < 200; i++) {
    const sys = generateSystem(0.2 + i * 0.05, -i * 7, (i * 37 % 100) / 100);
    for (let j = 1; j < sys.planets.length; j++) {
      assert.ok(sys.planets[j].a > sys.planets[j - 1].a);
    }
    for (const p of sys.planets) {
      assert.ok(p.moons.length <= 4);
      for (const m of p.moons) assert.ok(m.a > p.radius);
    }
  }
});

test('planets never engulf the star or cross each other, even at pericentre and apocentre', () => {
  let eccentric = 0;
  for (let i = 0; i < 300; i++) {
    const sys = generateSystem(0.08 + i * 0.03, -i * 11, (i * 13 % 100) / 100);
    sys.planets.forEach((p, j) => {
      assert.ok(p.e >= 0 && p.e <= 0.8);
      if (p.e > 0.1) eccentric++;
      assert.ok(p.a * (1 - p.e) - p.radius > sys.star.radius);
      if (j > 0) {
        const q = sys.planets[j - 1];
        assert.ok(p.a * (1 - p.e) - p.radius > q.a * (1 + q.e) + q.radius);
      }
    });
  }
  assert.ok(eccentric > 20, `only ${eccentric} planets with e > 0.1`);
});

test('orbitPos: circular case, pericentre at the pericentre direction, apocentre opposite', () => {
  assert.deepEqual(orbitPos(2, 0.5, 0.1, 0, 0, 3), [Math.cos(0.8) * 2, Math.sin(0.8) * 2, 0]);
  const peri = orbitPos(2, 0, 1, 0.3, Math.PI / 2, 0);   // mean anomaly 0
  assert.ok(Math.abs(peri[0]) < 1e-12 && Math.abs(peri[1] - 2 * 0.7) < 1e-12);
  const apo = orbitPos(2, Math.PI, 1, 0.3, Math.PI / 2, 0);
  assert.ok(Math.abs(apo[0]) < 1e-12 && Math.abs(apo[1] + 2 * 1.3) < 1e-12);
  // Off the axes the Newton steps do the work: compare with a converged solve
  const M = 1, e = 0.4;
  let E = M; for (let i = 0; i < 50; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const [x, y] = orbitPos(2, M, 0, e, 0, 0);
  assert.ok(Math.abs(x - 2 * (Math.cos(E) - e)) < 1e-9 && Math.abs(y - 2 * Math.sqrt(1 - e * e) * Math.sin(E)) < 1e-9);
});

test('massive stars get at most 3 planets', () => {
  for (let i = 0; i < 100; i++) {
    assert.ok(generateSystem(8 + i * 0.3, -i, i / 100).planets.length <= 3);
  }
});

test('only gas giants past the frost line get rings; warm Jupiters never do', () => {
  let warm = 0;
  for (let i = 0; i < 3000; i++) {
    const sys = generateSystem(0.6 + i * 0.0005, -i * 3, (i * 7 % 1000) / 1000, 500);
    const frost = 2.7 * Math.sqrt(luminosity(sys.star.mass));
    for (const p of sys.planets) {
      if (p.kind === KIND.GAS && p.au < frost) { warm++; assert.equal(p.rings, null); }
      if (p.rings) assert.ok(p.kind === KIND.GAS && p.au >= frost);
    }
  }
  assert.ok(warm > 10, `only ${warm} warm Jupiters sampled`);
});

test('main-sequence laws: Sun-normalised, continuous, M dwarfs near 3000 K', () => {
  assert.equal(temperature(1), 5778);
  assert.ok(Math.abs(luminosity(1) - 1) < 0.05);
  assert.ok(Math.abs(temperature(0.57) / temperature(0.5699) - 1) < 0.02);
  for (const m of [0.45, 0.72, 1.05, 2.4, 7, 31]) assert.ok(Math.abs(luminosity(m) / luminosity(m * 0.9999) - 1) < 0.05, `luminosity join at ${m}`);
  assert.ok(Math.abs(Math.log10(luminosity(59)) - 5.82) < 0.15);   // O3V
  assert.ok(Math.abs(temperature(0.37) / temperature(0.3699) - 1) < 0.02);
  assert.ok(Math.abs(temperature(1) / temperature(0.9999) - 1) < 0.02);
  assert.ok(temperature(0.15) > 2800 && temperature(0.15) < 3200);
  assert.equal(spectralClass(temperature(0.15)), 'M');
  assert.equal(spectralClass(temperature(0.5)), 'M');   // M1V, not K
  assert.equal(spectralClass(temperature(0.7)), 'K');
  assert.equal(spectralClass(temperature(3)), 'B');
  assert.ok(temperature(60) < 46000);   // O3V tops out near 45,000 K
});

test('spectral classes', () => {
  assert.equal(spectralClass(5778), 'G');
  assert.equal(spectralClass(35000), 'O');
  assert.equal(spectralClass(3000), 'M');
});

test('bodyPositions follows the shader orbit formula and buffer order', () => {
  const sys = generateSystem(1.2, 0.3, 0.77);
  const giant = sys.planets.find(p => p.moons.length);
  assert.ok(giant, 'fixture needs a planet with moons');
  const t = 3.7;
  const bodies = bodyPositions(sys, t);
  assert.equal(bodies.length, 1 + sys.planets.reduce((n, p) => n + 1 + p.moons.length, 0));
  assert.deepEqual(bodies[0].pos, [0, 0, 0]);
  assert.equal(bodies[0].radius, sys.star.radius);

  let i = 1;
  for (const p of sys.planets) {
    const c = orbitPos(p.a, p.phase, p.rate, p.e, p.w, t);
    assert.deepEqual(bodies[i].pos, c);
    assert.equal(bodies[i].planet, p);
    assert.equal(bodies[i++].moon, false);
    for (const m of p.moons) {
      const o = orbitPos(m.a, m.phase, m.rate, 0, 0, t);
      assert.deepEqual(bodies[i].pos, [c[0] + o[0], c[1] + o[1], 0]);
      assert.equal(bodies[i].radius, m.radius);
      assert.equal(bodies[i++].moon, true);
    }
  }
});

test('stellar evolution: a red giant swallows inner planets, remnants keep the rest', () => {
  const young = generateSystem(1.0, -1000, 0, 1000);
  const giant = generateSystem(1.0, -1000, 0, 8540 + 520);   // t_MS = 8.54 Gyr, giant phase 1.04 Gyr
  const dwarf = generateSystem(1.0, -1000, 0, 20000);
  assert.ok(young.planets.length >= 3, 'fixture needs planets');
  assert.equal(giant.star.phase, 'giant');
  assert.ok(giant.star.radius > young.star.radius * 2);
  assert.ok(giant.planets.length < young.planets.length);
  assert.equal(dwarf.star.phase, 'white dwarf');
  assert.equal(dwarf.star.class, 'WD');
  assert.ok(dwarf.star.tempK > 3800 && dwarf.star.tempK < 5000);   // ~10.4 Gyr of cooling: Montreal track gives ~4,000 K
  // Engulfment is physical: everything with pericentre inside ~1.15 AU is gone by the giant's end
  assert.ok(young.planets.some(p => p.au * (1 - p.e) < 1.15), 'fixture needs an inner planet');
  assert.ok(generateSystem(1.0, -1000, 0, 9579).planets.every(p => p.au * (1 - p.e) > 1.1));
  assert.equal(evolveStar(30, 1e6).phase, 'black hole');
  assert.equal(evolveStar(30, 1e6).lum, 0);
  // Same ladder regardless of age: survivors sit on the same orbits; only worlds with air change kind
  for (const p of giant.planets) {
    const twin = young.planets.find(q => q.au === p.au);
    assert.ok(twin && (twin.kind === p.kind || twin.kind === KIND.TERRAN || twin.kind === KIND.VENUSIAN));
  }
  // What the giant swallowed stays swallowed: the remnant keeps exactly the late giant's survivors
  const lateGiant = generateSystem(1.0, -1000, 0, 9579);
  assert.ok(lateGiant.planets.length <= giant.planets.length);
  // ...and their orbits widen by M / M_wd as the envelope goes
  const grow = 1 / 0.55;
  assert.deepEqual(dwarf.planets.map(p => +(p.au / grow).toFixed(9)), lateGiant.planets.map(p => +p.au.toFixed(9)));
  assert.ok(dwarf.star.reachAU > 1);
  assert.equal(dwarf.belt?.a, young.belt?.a);
  assert.ok(dwarf.planets[0].teq < young.planets[0].teq / 3);
  // Worlds with air freeze around a white dwarf and boil under a red giant
  assert.ok(young.planets.some(p => p.kind === KIND.TERRAN || p.kind === KIND.VENUSIAN), 'fixture needs a world with air');
  assert.ok(!dwarf.planets.some(p => p.kind === KIND.TERRAN || p.kind === KIND.VENUSIAN));
  assert.ok(!giant.planets.some(p => (p.kind === KIND.TERRAN || p.kind === KIND.VENUSIAN) && p.teq > 400));
  assert.equal(evolveStar(10, 1e6).phase, 'neutron star');
});
