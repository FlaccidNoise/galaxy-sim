import GUI from 'lil-gui';

const COLOR_MODES = { 'Spectral': 0, 'Velocity': 1, 'Mass': 2 };

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
      // Display
      colorMode: 'Spectral',
      agingRate: 25.0,   // Myr of stellar aging per sim time unit (0 = frozen)
      exposureEV: 0.0,
      bloomStrength: 0.35,
      starIntensity: 1.0,
      sizeScale: 1.0,
    };

    this._build();
  }

  _build() {
    const gui = new GUI({ title: 'Galaxy Sim' });

    // Simulation folder
    const simFolder = gui.addFolder('Simulation');
    simFolder.add(this.state, 'paused').name('Pause');
    simFolder.add(this.state, 'speed', 0.01, 10, 0.01).name('Speed');
    simFolder.add({ reset: () => this._restart() }, 'reset').name('Reset');

    // Initial conditions folder — changes auto-restart after 600ms debounce
    const icFolder = gui.addFolder('Initial Conditions');
    icFolder.add(this.state, 'preset', ['Galaxy Disk', 'Uniform Sphere', 'Collision'])
      .name('Preset').onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'N', 1000, 500000, 1000).name('N (stars)')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'diskRadius', 10, 200).name('Disk Radius')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'diskHeight', 0.5, 20).name('Disk Height')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'centralMass', 0, 50000).name('Central Mass')
      .onChange(v => this._onCentralMass?.(v));
    icFolder.add(this.state, 'toomreQ', 0.8, 3, 0.05).name('Toomre Q')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'armCount', 1, 6, 1).name('Arm Count')
      .onChange(() => this._scheduleRestart());

    // Physics folder — live (read from state each frame)
    const physFolder = gui.addFolder('Physics');
    physFolder.add(this.state, 'G', 0.1, 10).name('G');
    physFolder.add(this.state, 'epsilon', 0.1, 10).name('Softening ε');
    physFolder.add(this.state, 'dt', 0.001, 0.1).name('Timestep dt');
    physFolder.add(this.state, 'theta', 0.1, 1.0, 0.05).name('BH Theta θ');
    physFolder.add(this.state, 'haloFactor', 0, 10, 0.1).name('Halo Mass ×M⋆')
      .onChange(() => this._scheduleRestart());
    physFolder.add(this.state, 'haloScale', 0.3, 4, 0.1).name('Halo Scale ×R')
      .onChange(() => this._scheduleRestart());

    // Display folder — drives renderer directly
    const r = this.renderer;
    const dispFolder = gui.addFolder('Display');
    dispFolder.add(this.state, 'colorMode', Object.keys(COLOR_MODES)).name('Color Mode')
      .onChange(v => { r.colorMode = COLOR_MODES[v]; });
    dispFolder.add(this.state, 'agingRate', 0, 200, 1).name('Aging (Myr/t)')
      .onChange(v => { r.agingRate = v; });
    dispFolder.add(this.state, 'exposureEV', -4, 4, 0.1).name('Exposure (EV)')
      .onChange(v => { r.exposure = Math.pow(2, v); });
    dispFolder.add(this.state, 'bloomStrength', 0, 1.5, 0.01).name('Bloom')
      .onChange(v => { r.bloomStrength = v; });
    dispFolder.add(this.state, 'starIntensity', 0.05, 4, 0.05).name('Star Intensity')
      .onChange(v => { r.starIntensity = v; });
    dispFolder.add(this.state, 'sizeScale', 0.3, 3, 0.05).name('Star Size')
      .onChange(v => { r.sizeScale = v; });
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
