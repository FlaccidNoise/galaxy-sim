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
      assert.ok(p.e >= 0 && p.e <= 0.4);
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
});

test('massive stars get at most 3 planets', () => {
  for (let i = 0; i < 100; i++) {
    assert.ok(generateSystem(8 + i * 0.3, -i, i / 100).planets.length <= 3);
  }
});

test('only gas giants get rings', () => {
  for (let i = 0; i < 200; i++) {
    const sys = generateSystem(0.3 + i * 0.02, -i * 3, (i * 7 % 100) / 100);
    for (const p of sys.planets) {
      if (p.rings) assert.equal(p.kind, KIND.GAS);
    }
  }
});

test('main-sequence laws: Sun-normalised, continuous, M dwarfs near 3000 K', () => {
  assert.equal(temperature(1), 5778);
  assert.equal(luminosity(1), 1);
  assert.ok(Math.abs(temperature(0.57) / temperature(0.5699) - 1) < 0.02);
  assert.ok(Math.abs(luminosity(0.43) / luminosity(0.4299) - 1) < 0.05);
  assert.ok(Math.abs(luminosity(2) / luminosity(1.999) - 1) < 0.05);
  assert.ok(temperature(0.15) > 2800 && temperature(0.15) < 3200);
  assert.equal(spectralClass(temperature(0.15)), 'M');
  assert.equal(spectralClass(temperature(3)), 'B');
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
  const giant = generateSystem(1.0, -1000, 0, 10000 + 600);   // t_MS = 10 Gyr, giant phase 1.2 Gyr
  const dwarf = generateSystem(1.0, -1000, 0, 20000);
  assert.ok(young.planets.length >= 3, 'fixture needs planets');
  assert.equal(giant.star.phase, 'giant');
  assert.ok(giant.star.radius > young.star.radius * 2);
  assert.ok(giant.planets.length < young.planets.length);
  assert.equal(dwarf.star.phase, 'white dwarf');
  assert.equal(dwarf.star.class, 'WD');
  // Same ladder regardless of age: survivors sit on the same orbits with the same kinds
  for (const p of giant.planets) {
    const twin = young.planets.find(q => q.au === p.au);
    assert.ok(twin && twin.kind === p.kind);
  }
  assert.deepEqual(dwarf.planets.map(p => [p.au, p.kind]), young.planets.map(p => [p.au, p.kind]));
  assert.ok(dwarf.planets[0].teq < young.planets[0].teq / 3);
  assert.equal(evolveStar(10, 1e6).phase, 'neutron star');
});
