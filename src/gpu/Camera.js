/**
 * Orbit camera with damping — replaces THREE.OrbitControls.
 * Produces a column-major viewProj matrix plus basis vectors for shaders.
 *
 * Left-drag / one finger: rotate · Right-drag / two-finger drag: pan
 * Wheel / pinch: zoom
 */
export class Camera {
  constructor(canvas) {
    this.canvas = canvas;
    this.fovY = 60 * Math.PI / 180;
    this.near = 0.1;
    this.far = 10000;

    // Spherical state around target (current vs. damped goal)
    this.target = [0, 0, 0];
    this.theta = 0;            // azimuth
    // Disk lies in the XY plane (normal = +Z); phi=0 is face-on, π/2 edge-on.
    this.phi = 0.85;           // ~49° inclination: classic galaxy-photo angle
    // Home distance: the disc fills a landscape view; a portrait phone pulls back so it still fits
    this.homeRadius = 130 / Math.min(canvas.clientWidth / Math.max(canvas.clientHeight, 1), 1);
    this.radius = this.homeRadius;
    this.minRadius = 2;        // zoom floor — drops to 0.02 while a star is selected
    this._goal = { theta: this.theta, phi: this.phi, radius: this.radius, target: [...this.target] };

    // Outputs (recomputed in update())
    this.position = [0, 0, 0];
    this.right = [1, 0, 0];
    this.up = [0, 1, 0];
    this.forward = [0, 0, -1];
    this.viewProj = new Float32Array(16);

    this._bind();
    this.update();
  }

  _bind() {
    const el = this.canvas;
    const pointers = new Map();   // pointerId → last {x, y}
    let mode = 1;                 // single pointer: 1 = rotate, 2 = pan

    const pan = (dx, dy) => {
      // Pan in screen plane, scaled by distance
      const s = this._goal.radius * 0.0012;
      const t = this._goal.target;
      t[0] -= (dx * this.right[0] - dy * this.up[0]) * s;
      t[1] -= (dx * this.right[1] - dy * this.up[1]) * s;
      t[2] -= (dx * this.right[2] - dy * this.up[2]) * s;
    };
    const zoom = f => {
      this._goal.radius = Math.min(Math.max(this._goal.radius * f, this.minRadius), 4000);
    };
    const pair = () => {
      const [a, b] = pointers.values();
      return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) };
    };
    const release = e => pointers.delete(e.pointerId);

    el.addEventListener('contextmenu', e => e.preventDefault());
    el.addEventListener('pointerdown', e => {
      if (pointers.size === 0) mode = e.button === 2 ? 2 : 1;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);
    el.addEventListener('pointermove', e => {
      const p = pointers.get(e.pointerId);
      if (!p) return;
      if (pointers.size === 1) {
        const dx = e.clientX - p.x, dy = e.clientY - p.y;
        p.x = e.clientX; p.y = e.clientY;
        if (mode === 1) {
          this._goal.theta -= dx * 0.005;
          this._goal.phi = Math.min(Math.max(this._goal.phi + dy * 0.005, -1.55), 1.55);
        } else {
          pan(dx, dy);
        }
      } else if (pointers.size === 2) {
        const before = pair();
        p.x = e.clientX; p.y = e.clientY;
        const after = pair();
        if (before.d > 0 && after.d > 0) zoom(before.d / after.d);
        pan(after.x - before.x, after.y - before.y);
      } else {
        p.x = e.clientX; p.y = e.clientY;
      }
    });
    el.addEventListener('wheel', e => {
      e.preventDefault();
      zoom(Math.exp(e.deltaY * 0.001));
    }, { passive: false });
  }

  update() {
    // Damped approach to goal state
    const k = 0.12;
    this.theta += (this._goal.theta - this.theta) * k;
    this.phi += (this._goal.phi - this.phi) * k;
    this.radius += (this._goal.radius - this.radius) * k;
    for (let i = 0; i < 3; i++) this.target[i] += (this._goal.target[i] - this.target[i]) * k;

    const ct = this.target;
    const cp = Math.cos(this.phi), sp = Math.sin(this.phi);
    const st = Math.sin(this.theta), c2 = Math.cos(this.theta);
    this.position = [
      ct[0] + this.radius * cp * st,
      ct[1] + this.radius * sp,
      ct[2] + this.radius * cp * c2,
    ];

    // Basis
    const f = norm3(sub3(ct, this.position));         // forward
    const r = norm3(cross3(f, [0, 1, 0]));            // right
    const u = cross3(r, f);                            // up
    this.forward = f; this.right = r; this.up = u;

    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1);
    this.aspect = aspect;
    makeViewProj(this.viewProj, this.position, r, u, f, this.fovY, aspect, this.near, this.far);
  }

  /** Back to the home view (restart): origin target, home distance, same orientation. */
  home() {
    this._goal.radius = this.homeRadius;
    this._goal.target = [0, 0, 0];
  }

  setMinRadius(r) {
    this.minRadius = r;
    if (this._goal.radius < r) this._goal.radius = r;
  }

  /** This camera's POV at a different radius (same target and orientation). */
  viewFrom(radius) {
    const eye = [
      this.target[0] - this.forward[0] * radius,
      this.target[1] - this.forward[1] * radius,
      this.target[2] - this.forward[2] * radius,
    ];
    const viewProj = new Float32Array(16);
    makeViewProj(viewProj, eye, this.right, this.up, this.forward,
                 this.fovY, this.aspect, this.near, this.far);
    return { position: eye, right: this.right, up: this.up, forward: this.forward,
             viewProj, fovY: this.fovY, aspect: this.aspect };
  }
}

/** Column-major view-projection from an eye + orthonormal basis. */
export function makeViewProj(out, eye, r, u, f, fovY, aspect, near, far) {
  const view = [
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    -dot3(r, eye), -dot3(u, eye), dot3(f, eye), 1,
  ];
  const fpr = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const proj = [
    fpr / aspect, 0, 0, 0,
    0, fpr, 0, 0,
    0, 0, far * nf, -1,
    0, 0, far * near * nf, 0,
  ];
  mat4mul(out, proj, view);
}

function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm3(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
/** out = a * b, all column-major 4x4 */
function mat4mul(out, a, b) {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
}
