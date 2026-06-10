/**
 * Orbit camera with damping — replaces THREE.OrbitControls.
 * Produces a column-major viewProj matrix plus basis vectors for shaders.
 *
 * Left-drag: rotate · Right-drag / two-finger drag: pan · Wheel: zoom
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
    this.radius = 130;
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
    let dragging = 0; // 1 = rotate, 2 = pan
    let lastX = 0, lastY = 0;

    el.addEventListener('contextmenu', e => e.preventDefault());
    el.addEventListener('pointerdown', e => {
      dragging = e.button === 2 ? 2 : 1;
      lastX = e.clientX; lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointerup', () => { dragging = 0; });
    el.addEventListener('pointermove', e => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (dragging === 1) {
        this._goal.theta -= dx * 0.005;
        this._goal.phi = Math.min(Math.max(this._goal.phi + dy * 0.005, -1.55), 1.55);
      } else {
        // Pan in screen plane, scaled by distance
        const s = this._goal.radius * 0.0012;
        const t = this._goal.target;
        t[0] -= (dx * this.right[0] - dy * this.up[0]) * s;
        t[1] -= (dx * this.right[1] - dy * this.up[1]) * s;
        t[2] -= (dx * this.right[2] - dy * this.up[2]) * s;
      }
    });
    el.addEventListener('wheel', e => {
      e.preventDefault();
      this._goal.radius = Math.min(Math.max(this._goal.radius * Math.exp(e.deltaY * 0.001), 2), 4000);
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

    // View matrix (lookAt) then proj, multiplied into viewProj (column-major)
    const e = this.position;
    const view = [
      r[0], u[0], -f[0], 0,
      r[1], u[1], -f[1], 0,
      r[2], u[2], -f[2], 0,
      -dot3(r, e), -dot3(u, e), dot3(f, e), 1,
    ];
    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1);
    this.aspect = aspect;
    const fpr = 1 / Math.tan(this.fovY / 2);
    const nf = 1 / (this.near - this.far);
    const proj = [
      fpr / aspect, 0, 0, 0,
      0, fpr, 0, 0,
      0, 0, this.far * nf, -1,
      0, 0, this.far * this.near * nf, 0,
    ];
    mat4mul(this.viewProj, proj, view);
  }
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
