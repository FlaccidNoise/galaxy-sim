import GUI from 'lil-gui';

export class Controls {
  constructor(sim, particles, renderer) {
    this.sim = sim;
    this.particles = particles;
    this.renderer = renderer;
    this._restartPending = false;
    this._onRestart = null;

    this.state = {
      // Simulation
      paused: false,
      speed: 1.0,
      // Initial conditions
      preset: 'Galaxy Disk',
      N: 10000,
      diskRadius: 50.0,
      diskHeight: 2.0,
      centralMass: 1000.0,
      velDispersion: 0.5,
      // Physics
      G: 1.0,
      epsilon: 1.0,
      dt: 0.02,
      // Display
      colorMode: 'Velocity',
      bloomStrength: 1.5,
      pointSize: 0.3,
    };

    this._build();
  }

  _build() {
    const gui = new GUI({ title: 'Galaxy Sim' });

    // Simulation folder
    const simFolder = gui.addFolder('Simulation');
    simFolder.add(this.state, 'paused').name('Pause');
    simFolder.add(this.state, 'speed', 0.1, 10, 0.1).name('Speed');
    simFolder.add({ reset: () => this._restart() }, 'reset').name('Reset');

    // Initial conditions folder
    const icFolder = gui.addFolder('Initial Conditions');
    icFolder.add(this.state, 'preset', ['Galaxy Disk', 'Uniform Sphere', 'Collision'])
      .name('Preset').onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'N', 1000, 100000, 1000).name('N (stars)')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'diskRadius', 10, 200).name('Disk Radius')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'diskHeight', 0.5, 20).name('Disk Height')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'centralMass', 0, 5000).name('Central Mass')
      .onChange(() => this._markRestartPending(icFolder));
    icFolder.add(this.state, 'velDispersion', 0, 5).name('Vel Dispersion')
      .onChange(() => this._markRestartPending(icFolder));

    // Physics folder
    const physFolder = gui.addFolder('Physics');
    physFolder.add(this.state, 'G', 0.1, 10).name('G');
    physFolder.add(this.state, 'epsilon', 0.1, 10).name('Softening ε');
    physFolder.add(this.state, 'dt', 0.001, 0.1).name('Timestep dt');

    // Display folder
    const dispFolder = gui.addFolder('Display');
    dispFolder.add(this.state, 'colorMode', ['Velocity', 'Mass']).name('Color Mode');
    dispFolder.add(this.state, 'bloomStrength', 0, 3).name('Bloom').onChange(v => {
      this.renderer.setBloom(v, 0.4, 0.1);
    });
    dispFolder.add(this.state, 'pointSize', 0.05, 2).name('Point Size').onChange(v => {
      this.particles.setPointSize(v);
    });
  }

  _markRestartPending(folder) {
    if (!this._restartPending) {
      this._restartPending = true;
      folder.domElement.style.borderLeft = '3px solid orange';
    }
  }

  onRestart(fn) {
    this._onRestart = fn;
  }

  _restart() {
    this._restartPending = false;
    // Clear orange border from all folders
    document.querySelectorAll('.lil-gui .children').forEach(el => {
      el.style.borderLeft = '';
    });
    if (this._onRestart) this._onRestart(this.state);
  }
}
