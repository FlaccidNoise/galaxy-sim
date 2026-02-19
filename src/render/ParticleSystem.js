import * as THREE from 'three';

export class ParticleSystem {
  constructor(scene, N) {
    this.N = N;
    this.positions = new Float32Array(N * 3);
    this.colors = new Float32Array(N * 3);

    const geometry = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.positions, 3);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.colAttr = new THREE.BufferAttribute(this.colors, 3);
    this.colAttr.setUsage(THREE.DynamicDrawUsage);

    geometry.setAttribute('position', this.posAttr);
    geometry.setAttribute('color', this.colAttr);

    const material = new THREE.PointsMaterial({
      size: 0.3,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.points = new THREE.Points(geometry, material);
    scene.add(this.points);

    this._stagingBuffer = null;
    this._device = null;
    this._physicsByteSize = null;
  }

  initStaging(device, physicsBufferByteSize) {
    this._device = device;
    this._physicsByteSize = physicsBufferByteSize;
    this._stagingBuffer = device.createBuffer({
      size: physicsBufferByteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Queue a GPU buffer copy from physics buffer to staging buffer.
   * Must be called before commandEncoder.finish() — adds a copy command to the encoder.
   * After submitting the encoder, call updateFromStaging() to read back and update geometry.
   */
  queueReadback(commandEncoder, gpuBuffer) {
    commandEncoder.copyBufferToBuffer(gpuBuffer, 0, this._stagingBuffer, 0, this._physicsByteSize);
  }

  /**
   * Map staging buffer, read positions and velocities, update Three.js geometry.
   * Must be called after the command encoder containing queueReadback() has been submitted.
   */
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

      const vx = raw[src + 4], vy = raw[src + 5], vz = raw[src + 6];
      const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);

      if (colorMode === 'Velocity') {
        velocityColor(speed, speedScale, this.colors, dst);
      } else {
        massColor(raw[src + 3], this.colors, dst);
      }
    }

    this._stagingBuffer.unmap();

    this.posAttr.needsUpdate = true;
    this.colAttr.needsUpdate = true;
  }

  setPointSize(size) {
    this.points.material.size = size;
  }
}

/**
 * Map speed to color: slow = red-orange, fast = blue-white.
 * t = clamp(speed / scale, 0, 1)
 * t < 0.33: red → orange
 * t < 0.66: orange → yellow-white
 * t >= 0.66: yellow-white → blue-white
 */
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

/**
 * Map mass to color: low mass = dim red (M-type), high mass = bright blue-white (O/B-type).
 */
function massColor(mass, out, i) {
  const t = Math.min(Math.log(mass + 1) / 10, 1.0);
  out[i]     = t > 0.5 ? 1.0 : t * 2;
  out[i + 1] = t > 0.5 ? 1.0 : t * 2;
  out[i + 2] = 0.3 + t * 0.7;
}
