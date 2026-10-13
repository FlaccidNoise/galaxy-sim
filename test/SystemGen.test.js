import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, generateSystem, spectralClass, KIND } from '../src/system/SystemGen.js';

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

test('planet kinds respect the frost line', () => {
  for (let i = 0; i < 200; i++) {
    const sys = generateSystem(0.5 + i * 0.01, -i * 13, i / 200);
    const frost = 2.7 * Math.sqrt(sys.star.lum);
    for (const p of sys.planets) {
      if (p.kind === KIND.GAS || p.kind === KIND.ICE_GIANT) assert.ok(p.a >= frost);
      else assert.ok(p.a < frost);
    }
  }
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

test('planets never engulf the star or each other', () => {
  for (let i = 0; i < 300; i++) {
    const sys = generateSystem(0.08 + i * 0.03, -i * 11, (i * 13 % 100) / 100);
    sys.planets.forEach((p, j) => {
      assert.ok(p.a - p.radius > sys.star.radius);
      if (j > 0) assert.ok(p.a - p.radius > sys.planets[j - 1].a + sys.planets[j - 1].radius);
    });
  }
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

test('spectral classes', () => {
  assert.equal(spectralClass(5778), 'G');
  assert.equal(spectralClass(35000), 'O');
  assert.equal(spectralClass(3000), 'M');
});
