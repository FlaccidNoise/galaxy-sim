// Barnes-Hut N-body compute kernel — O(N log N) per step.
//
// BHNode layout (64 bytes = 16 × f32/i32):
//   comMass  : vec4f  — xyz = centre of mass, w = total mass
//   sizeLeaf : vec4f  — x = node width, y = isLeaf (1.0 or 0.0), zw = padding
//   children : array<i32, 8>  — child indices, −1 = empty

struct Particle {
    pos  : vec3f,
    mass : f32,
    vel  : vec3f,
    pad  : f32,   // spectral colorMass — must be preserved across steps
}

struct Params {
    dt       : f32,
    G        : f32,
    epsilon2 : f32,  // softening², prevents singularity
    N        : u32,
    theta    : f32,  // Barnes-Hut opening angle (smaller = more accurate)
    _pad0    : f32,
    _pad1    : f32,
    _pad2    : f32,
}

struct BHNode {
    comMass  : vec4f,          // xyz = COM, w = total mass
    sizeLeaf : vec4f,          // x = node size, y = isLeaf flag
    children : array<i32, 8>,
}

@group(0) @binding(0) var<storage, read>       particlesIn  : array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut : array<Particle>;
@group(0) @binding(2) var<uniform>             params       : Params;
@group(0) @binding(3) var<storage, read>       bhNodes      : array<BHNode>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let pi  = particlesIn[i];
    var acc = vec3f(0.0);

    // Iterative BH traversal with an explicit per-thread stack.
    // Stack depth 64 is ample: worst-case is 8 × tree_depth ≈ 8 × log₈(N).
    var stack : array<i32, 64>;
    var sp    : i32 = 0;
    stack[0] = 0;   // start at root
    sp = 1;

    while (sp > 0) {
        sp -= 1;
        let nodeIdx  = stack[sp];
        let node     = bhNodes[nodeIdx];
        let nodeMass = node.comMass.w;

        if (nodeMass == 0.0) { continue; }  // empty node (shouldn't occur but safe)

        let com      = node.comMass.xyz;
        let nodeSize = node.sizeLeaf.x;
        let isLeaf   = node.sizeLeaf.y > 0.5;

        let r     = com - pi.pos;
        let dist2 = dot(r, r) + params.epsilon2;
        let dist  = sqrt(dist2);

        if (isLeaf || nodeSize / dist < params.theta) {
            // Approximate this subtree (or direct interaction for a leaf).
            // Self-interaction: r ≈ 0 → contribution ≈ 0. Plummer softening
            // makes it exactly G·m·ε/(ε²)^(3/2), but r=0 means acc += 0. Safe.
            acc += r * (params.G * nodeMass / (dist2 * dist));
        } else {
            // Open node: push non-empty children.
            for (var c : i32 = 0; c < 8; c++) {
                let ch = node.children[c];
                if (ch >= 0 && sp < 63) {
                    stack[sp] = ch;
                    sp += 1;
                }
            }
        }
    }

    // Symplectic Euler integration (same as original kernel).
    let newVel = pi.vel + acc * params.dt;
    let newPos = pi.pos + newVel * params.dt;

    // Preserve pad — it stores per-star spectral colorMass for rendering.
    particlesOut[i] = Particle(newPos, pi.mass, newVel, pi.pad);
}
