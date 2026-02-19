export const vertexShader = /* glsl */`
  attribute float size;
  varying vec3 vColor;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const fragmentShader = /* glsl */`
  varying vec3 vColor;

  void main() {
    // Distance from center of point sprite, in [0, 1]
    float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
    if (d > 1.0) discard;

    // Gaussian glow falloff
    float intensity = exp(-d * d * 3.0);

    // Hot white core blending to spectral color at edge
    vec3 col = mix(vColor, vec3(1.0), pow(max(0.0, 1.0 - d), 3.0));

    gl_FragColor = vec4(col, intensity * 0.9);
  }
`;
