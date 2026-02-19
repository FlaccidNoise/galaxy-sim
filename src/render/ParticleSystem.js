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

      const mass  = raw[src + 3];
      const vx = raw[src + 4], vy = raw[src + 5], vz = raw[src + 6];
      const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);

      if (colorMode === 'Spectral') {
        spectralColor(mass, this.colors, dst);
      } else if (colorMode === 'Velocity') {
        velocityColor(speed, speedScale, this.colors, dst);
      } else {
        massColor(mass, this.colors, dst);
      }

      // Size from luminosity: L ∝ M^3.5, log-scaled to pixel range
      if (i === 0) {
        // Central black hole / core — fixed large bright size
        this.sizes[i] = 16.0;
      } else {
        const L = Math.pow(Math.max(mass, 0.1), 3.5);
        this.sizes[i] = Math.min(Math.max(Math.log(L + 1.0) * 2.0 + 1.0, 1.0), 12.0);
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
 * O: >16  blue-white   B: 2-16  blue      A: 1.4-2  white-blue
 * F: 1.04-1.4  white   G: 0.8-1.04  yellow  K: 0.45-0.8  orange  M: <0.45  red-orange
 */
function spectralColor(mass, out, i) {
  if (mass > 16)        { out[i]=0.61; out[i+1]=0.69; out[i+2]=1.00; } // O
  else if (mass > 2)    { out[i]=0.67; out[i+1]=0.75; out[i+2]=1.00; } // B
  else if (mass > 1.4)  { out[i]=0.79; out[i+1]=0.85; out[i+2]=1.00; } // A
  else if (mass > 1.04) { out[i]=0.97; out[i+1]=0.97; out[i+2]=1.00; } // F
  else if (mass > 0.8)  { out[i]=1.00; out[i+1]=0.96; out[i+2]=0.92; } // G
  else if (mass > 0.45) { out[i]=1.00; out[i+1]=0.82; out[i+2]=0.63; } // K
  else                  { out[i]=1.00; out[i+1]=0.80; out[i+2]=0.44; } // M
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
