import GUI from 'lil-gui';

export class Controls {
  constructor(sim, particles, renderer) {
    this.sim = sim;
    this.particles = particles;
    this.renderer = renderer;
    this._onRestart = null;
    this._restartTimer = null;

    this.state = {
      // Simulation
      paused: false,
      speed: 0.03,
      // Initial conditions
      preset: 'Galaxy Disk',
      N: 25000,
      diskRadius: 50.0,
      diskHeight: 2.0,
      centralMass: 10000.0,
      velDispersion: 0.2,
      armCount: 2,
      // Physics
      G: 1.0,
      epsilon: 1.5,
      dt: 0.01,
      // Display
      colorMode: 'Spectral',
      bloomStrength: 0.55,
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
    icFolder.add(this.state, 'N', 1000, 100000, 1000).name('N (stars)')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'diskRadius', 10, 200).name('Disk Radius')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'diskHeight', 0.5, 20).name('Disk Height')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'centralMass', 0, 50000).name('Central Mass')
      .onChange(v => this.sim.updateCentralMass(v));
    icFolder.add(this.state, 'velDispersion', 0, 5).name('Vel Dispersion')
      .onChange(() => this._scheduleRestart());
    icFolder.add(this.state, 'armCount', 1, 6, 1).name('Arm Count')
      .onChange(() => this._scheduleRestart());

    // Physics folder — already live (read from state each frame)
    const physFolder = gui.addFolder('Physics');
    physFolder.add(this.state, 'G', 0.1, 10).name('G');
    physFolder.add(this.state, 'epsilon', 0.1, 10).name('Softening ε');
    physFolder.add(this.state, 'dt', 0.001, 0.1).name('Timestep dt');

    // Display folder
    const dispFolder = gui.addFolder('Display');
    dispFolder.add(this.state, 'colorMode', ['Spectral', 'Velocity', 'Mass']).name('Color Mode');
    dispFolder.add(this.state, 'bloomStrength', 0, 3).name('Bloom').onChange(v => {
      this.renderer.setBloom(v, 0.4, 0.6);
    });
  }

  _scheduleRestart() {
    clearTimeout(this._restartTimer);
    this._restartTimer = setTimeout(() => this._restart(), 600);
  }

  onRestart(fn) {
    this._onRestart = fn;
  }

  _restart() {
    clearTimeout(this._restartTimer);
    this._restartTimer = null;
    if (this._onRestart) this._onRestart(this.state);
  }
}
