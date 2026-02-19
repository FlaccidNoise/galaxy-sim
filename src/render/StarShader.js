export const vertexShader = /* glsl */`
  attribute float size;
  varying vec3 vColor;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (150.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const fragmentShader = /* glsl */`
  varying vec3 vColor;

  void main() {
    // Distance from center of point sprite, in [0, 1]
    float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
    if (d > 1.0) discard;

    // Gaussian glow — tighter gaussian keeps stars crisp
    float intensity = exp(-d * d * 3.5);

    // Alpha 0.35 prevents additive-blending white-out in dense arm clusters:
    // 3× B-type overlap: 3*(0.60,0.75,1.0)*0.35 = (0.63,0.79,1.05) → clamped blue, not white.
    // Individual isolated stars remain clearly visible at this alpha with exposure boost.
    gl_FragColor = vec4(vColor, intensity * 0.35);
  }
`;
