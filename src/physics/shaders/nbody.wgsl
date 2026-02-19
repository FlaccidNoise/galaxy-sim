// Particle layout: vec4 pos_mass, vec4 vel_pad
// [x, y, z, mass,  vx, vy, vz, pad] per particle

struct Particle {
    pos  : vec3f,
    mass : f32,
    vel  : vec3f,
    pad  : f32,
}

struct Params {
    dt       : f32,
    G        : f32,
    epsilon2 : f32,  // softening^2, prevents singularity
    N        : u32,
}

@group(0) @binding(0) var<storage, read>       particlesIn  : array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut : array<Particle>;
@group(0) @binding(2) var<uniform>             params       : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let pi  = particlesIn[i];
    var acc = vec3f(0.0);

    // Direct summation: O(N²) per frame, but massively parallel on GPU
    for (var j : u32 = 0u; j < params.N; j++) {
        if (j == i) { continue; }
        let pj  = particlesIn[j];
        let r   = pj.pos - pi.pos;
        let d2  = dot(r, r) + params.epsilon2;
        // Force magnitude / distance: G * mj / d^3 (Plummer softening)
        let inv = params.G * pj.mass / (d2 * sqrt(d2));
        acc    += r * inv;
    }

    // Symplectic Euler (velocity Verlet without second force eval)
    // For proper leapfrog, split into two dispatches (kick-drift-kick).
    // Symplectic Euler is energy-conserving enough for visual simulation.
    let newVel = pi.vel + acc * params.dt;
    let newPos = pi.pos + newVel * params.dt;

    particlesOut[i] = Particle(newPos, pi.mass, newVel, 0.0);
}
