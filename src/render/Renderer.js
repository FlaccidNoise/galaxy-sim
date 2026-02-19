import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export class Renderer {
  constructor() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 10000);
    this.camera.position.set(0, 15, 115);
    this.camera.lookAt(0, 0, 0);
  }

  async init() {
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setSize(innerWidth, innerHeight);
    this.renderer.setPixelRatio(devicePixelRatio);
    // ReinhardToneMapping (x/(1+x) per-channel) provides the highlight compression we need:
    // - Dense arm clusters (3+ B stars): pre-Reinhard (1.26,1.58,2.10) → (0.56,0.61,0.68) = blue ✓
    // - Isolated K star: pre-Reinhard (0.42,0.27,0.12) → (0.30,0.21,0.11) = orange ✓
    // - Even 10-star clusters stay distinctly blue rather than white-saturating.
    // Linear tone mapping couldn't compress highlights; exposure boost made clusters white.
    this.renderer.toneMapping = THREE.ReinhardToneMapping;
    this.renderer.toneMappingExposure = 2.3;
    (document.getElementById('sim') ?? document.body).appendChild(this.renderer.domElement);

    // OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // Bloom postprocessing
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(innerWidth, innerHeight),
      0.55, // strength — enough to make arms visibly luminous
      0.3,  // radius — tight halos so individual stars bloom separately
      0.40  // threshold — lower so individual bright OB stars bloom
    );
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    // Resize handler
    window.addEventListener('resize', () => {
      this.camera.aspect = innerWidth / innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(innerWidth, innerHeight);
      this.composer.setSize(innerWidth, innerHeight);
    });

    // Background starfield for depth
    const bgGeo = new THREE.BufferGeometry();
    const bgPositions = new Float32Array(2000 * 3);
    for (let i = 0; i < 2000; i++) {
      bgPositions[i * 3]     = (Math.random() - 0.5) * 2000;
      bgPositions[i * 3 + 1] = (Math.random() - 0.5) * 2000;
      bgPositions[i * 3 + 2] = (Math.random() - 0.5) * 2000;
    }
    bgGeo.setAttribute('position', new THREE.BufferAttribute(bgPositions, 3));
    const bgMat = new THREE.PointsMaterial({ size: 0.5, color: 0x334455, sizeAttenuation: true });
    this.scene.add(new THREE.Points(bgGeo, bgMat));

    return this;
  }

  render() {
    this.controls.update();
    this.composer.render();
  }

  setBloom(strength, radius, threshold) {
    this.bloomPass.strength = strength;
    this.bloomPass.radius = radius;
    this.bloomPass.threshold = threshold;
  }
}
