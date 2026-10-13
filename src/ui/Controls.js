const COLOR_MODES = { 'Spectral': 0, 'Velocity': 1, 'Mass': 2 };

function fmt(c, v) {
  const d = c.step < 1 ? Math.max(2, String(c.step).split('.')[1].length) : 0;
  return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}

export class Controls {
  constructor(renderer) {
    this.renderer = renderer;
    this._onRestart = null;
    this._onCentralMass = null;
    this._restartTimer = null;

    this.state = {
      // Simulation
      paused: false,
      speed: 1.0,
      // Initial conditions
      preset: 'Galaxy Disk',
      N: 25000,
      diskRadius: 50.0,
      diskHeight: 2.0,
      // ~2% of stellar mass — with the halo carrying the rotation curve, a
      // dominant point mass only creates under-resolved tight orbits.
      centralMass: 500.0,
      toomreQ: 1.3,
      armCount: 2,
      // Physics
      G: 1.0,
      epsilon: 1.5,
      dt: 0.01,
      theta: 0.5,
      haloFactor: 3.0,   // halo mass as multiple of stellar mass (0 = no halo)
      haloScale: 1.2,    // halo scale radius as multiple of disk radius
      gasDrag: 0.25,     // gas cooling rate (damps non-orbital motion)
      sfEff: 0.012,      // star-formation efficiency (per time unit at threshold)
      sfThreshold: 2.5,  // SF density threshold, × mean disk density
      // Display
      colorMode: 'Spectral',
      agingRate: 25.0,   // Myr of stellar aging per sim time unit (0 = frozen)
      exposureEV: 0.0,
      bloomStrength: 0.35,
      starIntensity: 1.0,
      sizeScale: 1.0,
      dustStrength: 0.45,
      hiiStrength: 0.6,
    };

    this.isOpen = false;
    this.panel = document.getElementById('panel');
    this._link = document.getElementById('nav-controls');
    this._about = document.getElementById('about');
    this._build();
  }

  _build() {
    const r = this.renderer;
    const schema = [
      { section: 'Simulation', key: 'paused', label: 'Pause' },
      { key: 'speed', label: 'Speed', min: 0.01, max: 10, step: 0.01 },
      { label: 'Reset' },
      // Initial conditions — changes auto-restart after 600ms debounce
      { section: 'Initial conditions', key: 'preset', label: 'Preset',
        options: [['Galaxy Disk', 'Galaxy disc'], ['Uniform Sphere', 'Uniform sphere'], ['Collision', 'Collision']], restart: true },
      { key: 'N', label: 'N (stars)', min: 1000, max: 500000, step: 1000, restart: true },
      { key: 'diskRadius', label: 'Disc radius', min: 10, max: 200, step: 1, restart: true },
      { key: 'diskHeight', label: 'Disc height', min: 0.5, max: 20, step: 0.1, restart: true },
      { key: 'centralMass', label: 'Central mass', min: 0, max: 50000, step: 100,
        onChange: v => this._onCentralMass?.(v) },
      { key: 'toomreQ', label: 'Toomre Q', min: 0.8, max: 3, step: 0.05, restart: true },
      { key: 'armCount', label: 'Arm count', min: 1, max: 6, step: 1, restart: true },
      // Physics — live (read from state each frame)
      { section: 'Physics', key: 'G', label: 'G', min: 0.1, max: 10, step: 0.01 },
      { key: 'epsilon', label: 'Softening <i>ε</i>', min: 0.1, max: 10, step: 0.01 },
      { key: 'dt', label: 'Timestep <i>dt</i>', min: 0.001, max: 0.1, step: 0.001 },
      { key: 'theta', label: 'BH theta <i>θ</i>', min: 0.1, max: 1, step: 0.05 },
      { key: 'haloFactor', label: 'Halo mass ×M⋆', min: 0, max: 10, step: 0.1, restart: true },
      { key: 'haloScale', label: 'Halo scale ×R', min: 0.3, max: 4, step: 0.1, restart: true },
      { key: 'gasDrag', label: 'Gas cooling', min: 0, max: 1, step: 0.01 },
      { key: 'sfEff', label: 'SF efficiency', min: 0, max: 1, step: 0.001 },
      { key: 'sfThreshold', label: 'SF threshold ×<i>ρ̄</i>', min: 0.5, max: 10, step: 0.1 },
      // Display — drives renderer directly
      { section: 'Display', key: 'colorMode', label: 'Colour mode', options: Object.keys(COLOR_MODES).map(k => [k, k]),
        onChange: v => { r.colorMode = COLOR_MODES[v]; } },
      { key: 'agingRate', label: 'Aging <i>(Myr/t)</i>', min: 0, max: 200, step: 1,
        onChange: v => { r.agingRate = v; } },
      { key: 'exposureEV', label: 'Exposure (EV)', min: -4, max: 4, step: 0.1,
        onChange: v => { r.exposure = Math.pow(2, v); } },
      { key: 'bloomStrength', label: 'Bloom', min: 0, max: 1.5, step: 0.01,
        onChange: v => { r.bloomStrength = v; } },
      { key: 'starIntensity', label: 'Star intensity', min: 0.05, max: 4, step: 0.05,
        onChange: v => { r.starIntensity = v; } },
      { key: 'sizeScale', label: 'Star size', min: 0.3, max: 3, step: 0.05,
        onChange: v => { r.sizeScale = v; } },
      { key: 'dustStrength', label: 'Dust lanes', min: 0, max: 1.5, step: 0.05,
        onChange: v => { r.dustStrength = v; } },
      { key: 'hiiStrength', label: 'H-II nebulae', min: 0, max: 2, step: 0.05,
        onChange: v => { r.hiiStrength = v; } },
    ];

    const s = this.state;
    const byKey = {};
    let html = '', rail = '', n = 0;
    for (const c of schema) {
      if (c.section) {
        n++;
        html += `${html ? '</details>' : ''}<details open id="sec-${n}"><summary><b>0${n}</b>${c.section}</summary>`;
        rail += `<li><button type="button" data-to="sec-${n}" aria-controls="panel">${c.section}</button></li>`;
      }
      if (!c.key) {
        html += `<div class="row act"><button type="button" class="btn">${c.label}</button></div>`;
        continue;
      }
      byKey[c.key] = c;
      const id = `c-${c.key}`;
      const v = s[c.key];
      if (typeof v === 'boolean') {
        html += `<label class="row tog"><span class="lbl">${c.label}</span><input type="checkbox" class="sr" name="${c.key}"${v ? ' checked' : ''}><span class="state" aria-hidden="true"></span></label>`;
      } else if (c.options) {
        const opts = c.options.map(([o, name]) => `<option value="${o}"${o === v ? ' selected' : ''}>${name}</option>`).join('');
        html += `<div class="row"><label for="${id}">${c.label}</label><select id="${id}" name="${c.key}">${opts}</select></div>`;
      } else {
        html += `<div class="row"><label for="${id}">${c.label}</label><output for="${id}">${fmt(c, v)}</output><input id="${id}" name="${c.key}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" value="${v}"></div>`;
      }
    }
    this._about.insertAdjacentHTML('beforebegin', html + '</details>');
    const railList = document.querySelector('#rail ol');
    railList.innerHTML = rail + '<li><button type="button" data-to="about" aria-controls="panel">About</button></li>';
    this._rail = [...railList.querySelectorAll('button')];
    this._sections = this._rail.map(b => document.getElementById(b.dataset.to));
    this._top = this.panel.querySelector('.p-top');

    this.panel.addEventListener('input', e => {
      const c = byKey[e.target.name];
      if (!c) return;
      const t = e.target;
      const v = t.type === 'checkbox' ? t.checked : c.options ? t.value : +t.value;
      s[c.key] = v;
      if (t.type === 'range') t.previousElementSibling.value = fmt(c, v);
      c.onChange?.(v);
      if (c.restart) this._scheduleRestart();
    });
    this.panel.querySelector('.btn').addEventListener('click', () => this._restart());

    this._link.addEventListener('click', () => this.toggle());
    document.getElementById('sheet-close').addEventListener('click', () => this.open(false));
    document.getElementById('nav-reset').addEventListener('click', () => this._restart());
    railList.addEventListener('click', e => {
      const to = e.target.closest('button')?.dataset.to;
      if (!to) return;
      this.open(true);
      this.panel.scrollTo({ top: document.getElementById(to).offsetTop - this._top.offsetHeight });
    });
    this.panel.addEventListener('scroll', () => this._spy());
  }

  open(on) {
    this.isOpen = on;
    this.panel.classList.toggle('open', on);
    this._link.setAttribute('aria-expanded', on);
    this._spy();
  }

  toggle() { this.open(!this.isOpen); }

  // Marks the rail entry of the section at the top of the sheet
  _spy() {
    const y = this.panel.scrollTop + this._top.offsetHeight + 4;
    let at = 0;
    this._sections.forEach((d, i) => { if (d.offsetTop <= y) at = i; });
    this._rail.forEach((b, i) => b.classList.toggle('on', this.isOpen && i === at));
  }

  _scheduleRestart() {
    clearTimeout(this._restartTimer);
    this._restartTimer = setTimeout(() => this._restart(), 600);
  }

  onRestart(fn) { this._onRestart = fn; }
  onCentralMass(fn) { this._onCentralMass = fn; }

  _restart() {
    clearTimeout(this._restartTimer);
    this._restartTimer = null;
    if (this._onRestart) this._onRestart(this.state);
  }
}
