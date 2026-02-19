import * as THREE from 'three';
import { vertexShader, fragmentShader } from './StarShader.js';

export class ParticleSystem {
  constructor(scene, N) {
    this.N = N;
    this.positions = new Float32Array(N * 3);
    this.colors    = new Float32Array(N * 3);
    this.sizes     = new Float32Array(N);

    const geometry = new THREE.BufferGeometry();
    this.posAttr  = new THREE.BufferAttribute(this.positions, 3);
    this.colAttr  = new THREE.BufferAttribute(this.colors, 3);
    this.sizeAttr = new THREE.BufferAttribute(this.sizes, 1);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.colAttr.setUsage(THREE.DynamicDrawUsage);
    this.sizeAttr.setUsage(THREE.DynamicDrawUsage);

    geometry.setAttribute('position', this.posAttr);
    geometry.setAttribute('color',    this.colAttr);
    geometry.setAttribute('size',     this.sizeAttr);

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.points = new THREE.Points(geometry, material);
    scene.add(this.points);

    this._stagingBuffer    = null;
    this._device           = null;
    this._physicsByteSize  = null;
  }

  initStaging(device, physicsBufferByteSize) {
    this._device = device;
    this._physicsByteSize = physicsBufferByteSize;
    this._stagingBuffer = device.createBuffer({
      size: physicsBufferByteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  queueReadback(commandEncoder, gpuBuffer) {
    commandEncoder.copyBufferToBuffer(gpuBuffer, 0, this._stagingBuffer, 0, this._physicsByteSize);
  }

  async updateFromStaging(colorMode, speedScale = 50.0) {
    await this._stagingBuffer.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(this._stagingBuffer.getMappedRange());

    const N = this.N;
    for (let i = 0; i < N; i++) {
      const src = i * 8;
      const dst = i * 3;
      this.positions[dst]     = raw[src];
      this.positions[dst + 1] = raw[src + 1];
      this.positions[dst + 2] = raw[src + 2];

      // mass (slot 3) is the dynamical mass (1.0 for all stars).
      // colorMass (slot 7, the 'pad' field) is the spectral classification mass.
      const mass      = raw[src + 3];
      const colorMass = raw[src + 7] > 0 ? raw[src + 7] : mass;
      const vx = raw[src + 4], vy = raw[src + 5], vz = raw[src + 6];
      const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);

      if (colorMode === 'Spectral') {
        spectralColor(colorMass, this.colors, dst);
      } else if (colorMode === 'Velocity') {
        velocityColor(speed, speedScale, this.colors, dst);
      } else {
        massColor(colorMass, this.colors, dst);
      }

      // Size from luminosity: L ∝ M^3.5 using colorMass for visual classification.
      // Min 2.5 ensures K/M stars render as ~2px glows so their orange color is visible.
      // At camera depth ~200, gl_PointSize = size * 150/200. So size=2.5 → ~1.9px glow.
      if (i === 0) {
        this.sizes[i] = 8.0;
      } else {
        // Wider size spread so OB stars pop as individual blue jewels against the warm disk.
        // OB (>2 M☉) → 4.5–4.8px, A/F → 2.7–3.3px, G/K → 1.6–2.0px, M → 1.5px.
        // At camera depth ~140, gl_PointSize ≈ size * 1.07, so OB ≈ 5px screen.
        const L = Math.pow(Math.max(colorMass, 0.1), 3.5);
        this.sizes[i] = Math.min(Math.max(Math.log(L + 1.0) * 1.0 + 1.5, 2.0), 4.8);
      }
    }

    this._stagingBuffer.unmap();

    this.posAttr.needsUpdate  = true;
    this.colAttr.needsUpdate  = true;
    this.sizeAttr.needsUpdate = true;
  }
}

/**
 * Map stellar mass (M☉) to spectral type color.
 * Colors are intentionally saturated for visibility at small angular sizes —
 * a real star's spectral color isn't perfectly pure, but for a sim you need
 * the contrast to read clearly at 2-5 pixel scale.
 *
 * O: >16  vivid blue    B: 2-16  blue         A: 1.4-2  blue-white
 * F: 1.04-1.4  white    G: 0.8-1.04  yellow-white
 * K: 0.45-0.8  orange   M: <0.45  deep orange-red
 */
function spectralColor(mass, out, i) {
  if (mass > 16)        { out[i]=0.35; out[i+1]=0.55; out[i+2]=1.00; } // O  deep blue
  else if (mass > 2)    { out[i]=0.48; out[i+1]=0.65; out[i+2]=1.00; } // B  vivid blue
  else if (mass > 1.4)  { out[i]=0.78; out[i+1]=0.87; out[i+2]=1.00; } // A  blue-white
  else if (mass > 1.04) { out[i]=1.00; out[i+1]=1.00; out[i+2]=0.95; } // F  near-white
  else if (mass > 0.8)  { out[i]=1.00; out[i+1]=0.95; out[i+2]=0.75; } // G  yellow-white
  else if (mass > 0.45) { out[i]=1.00; out[i+1]=0.65; out[i+2]=0.30; } // K  orange
  else                  { out[i]=1.00; out[i+1]=0.45; out[i+2]=0.15; } // M  deep orange-red
}

function velocityColor(speed, scale, out, i) {
  const t = Math.min(speed / scale, 1.0);
  if (t < 0.33) {
    const s = t / 0.33;
    out[i]     = 1.0;
    out[i + 1] = s * 0.5;
    out[i + 2] = 0.0;
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    out[i]     = 1.0;
    out[i + 1] = 0.5 + s * 0.5;
    out[i + 2] = s * 0.5;
  } else {
    const s = (t - 0.66) / 0.34;
    out[i]     = 1.0 - s * 0.5;
    out[i + 1] = 1.0 - s * 0.5;
    out[i + 2] = 0.5 + s * 0.5;
  }
}

function massColor(mass, out, i) {
  const t = Math.min(Math.log(mass + 1) / 10, 1.0);
  out[i]     = t > 0.5 ? 1.0 : t * 2;
  out[i + 1] = t > 0.5 ? 1.0 : t * 2;
  out[i + 2] = 0.3 + t * 0.7;
}
