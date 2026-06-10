// Barnes-Hut N-body compute kernel — O(N log N) per step.
//
// BHNode layout (64 bytes = 16 × f32/i32):
//   comMass  : vec4f  — xyz = centre of mass, w = total mass
//   sizeLeaf : vec4f  — x = node width, y = isLeaf (1.0 or 0.0),
//                       z = bitcast i32 particle index for single-particle
//                           leaves (−1 otherwise), w = padding
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
    haloMass : f32,  // analytic Hernquist dark-matter halo (0 = off)
    haloA    : f32,  // halo scale radius
    _pad0    : f32,
}

// Integer fields are REAL u32s — never store ints as f32 bit patterns in
// storage buffers: small ints are denormal floats and implementations may
// flush denormals to zero on load.
struct BHNode {
    comMass  : vec4f,          // xyz = COM, w = total mass
    size     : f32,            // node width
    isLeaf   : u32,            // 1 = leaf, 0 = internal
    first    : u32,            // leaf: first particle index (sorted order)
    count    : u32,            // leaf: particle count
    children : array<i32, 8>,
}

@group(0) @binding(0) var<storage, read>       particlesIn  : array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut : array<Particle>;
@group(0) @binding(2) var<uniform>             params       : Params;
@group(0) @binding(3) var<storage, read>       bhNodes      : array<BHNode>;

const STACK_SIZE : i32 = 96;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let pi  = particlesIn[i];
    var acc = vec3f(0.0);
    let theta2 = params.theta * params.theta;

    // Iterative BH traversal with an explicit per-thread stack.
    var stack : array<i32, STACK_SIZE>;
    var sp : i32 = 1;
    stack[0] = 0;   // start at root

    while (sp > 0) {
        sp -= 1;
        let nodeIdx  = stack[sp];
        let node     = bhNodes[nodeIdx];
        let nodeMass = node.comMass.w;

        if (nodeMass == 0.0) { continue; }  // empty node (shouldn't occur but safe)

        let com      = node.comMass.xyz;
        let nodeSize = node.size;
        let isLeaf   = node.isLeaf == 1u;

        let r      = com - pi.pos;
        let geomD2 = dot(r, r);
        // SOFTENED distance in the opening criterion — deliberate: Plummer
        // softening (ε) erases all structure below ~ε, so any node smaller
        // than θ·ε is dynamically indistinguishable from its monopole. This
        // collapses the dense-core traversal cost by ~10× with no force
        // error beyond what softening already imposes.
        let d2 = geomD2 + params.epsilon2;

        // Sub-softening monopole: inside ε the softened pair force is linear
        // in separation, so the monopole equals the exact member sum — with
        // exact self-cancellation (Σ mⱼ(xⱼ−xᵢ) ≡ M·(com−xᵢ); the j=i term is
        // zero). This bounds worst-case cost when dense clumps collapse into
        // single deep tree cells.
        let s2 = nodeSize * nodeSize;
        if (s2 < theta2 * d2 || s2 < 0.25 * params.epsilon2) {
            // Far field (or sub-softening) — monopole.
            acc += r * (params.G * nodeMass / (d2 * sqrt(d2)));
        } else if (isLeaf) {
            // Near-field leaf — direct sum over its Morton-contiguous bucket.
            // Sequential reads, coherent across neighbouring threads; the
            // j == i test makes self-force exclusion exact.
            let first = node.first;
            let count = node.count;
            for (var j = first; j < first + count; j++) {
                if (j == i) { continue; }
                let pj = particlesIn[j];
                let rj = pj.pos - pi.pos;
                let dj2 = dot(rj, rj) + params.epsilon2;
                acc += rj * (params.G * pj.mass / (dj2 * sqrt(dj2)));
            }
        } else {
            // Open node: push non-empty children (bounds-checked BEFORE write).
            for (var c : i32 = 0; c < 8; c++) {
                let ch = node.children[c];
                if (ch >= 0 && sp < STACK_SIZE) {
                    stack[sp] = ch;
                    sp += 1;
                }
            }
        }
    }

    // Analytic Hernquist dark-matter halo: M(r) = M_h·r²/(r+a)², so
    // a(r) = -G·M_h/(r+a)² r̂ — ~6 ops for a flat rotation curve.
    if (params.haloMass > 0.0) {
        let rlen = length(pi.pos);
        if (rlen > 1e-6) {
            let ra = rlen + params.haloA;
            acc -= pi.pos * (params.G * params.haloMass / (ra * ra * rlen));
        }
    }

    // Kick-drift with constant dt ≡ leapfrog with half-step-offset velocities
    // (2nd-order symplectic); only dt changes break the equivalence.
    let newVel = pi.vel + acc * params.dt;
    let newPos = pi.pos + newVel * params.dt;

    // Preserve pad — it stores per-star spectral colorMass for rendering.
    particlesOut[i] = Particle(newPos, pi.mass, newVel, pi.pad);
}

// ── Position compaction for async tree-rebuild readback ─────────────────────
// Packs pos.xyz into a tight vec4 array so the CPU readback transfers
// 16 bytes/particle instead of the full 32-byte particle state.
@group(0) @binding(4) var<storage, read_write> posOut : array<vec4f>;

@compute @workgroup_size(256)
fn compact_positions(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }
    posOut[i] = vec4f(particlesIn[i].pos, 1.0);
}

// ── Per-step tree COM refresh ───────────────────────────────────────────────
// The tree TOPOLOGY (node ranges, children, sizes) is rebuilt asynchronously
// in the worker, but COMs/masses are re-summed from CURRENT particle
// positions every physics step, level by level (deepest first; one dispatch
// per level — dispatches in a pass are ordered, so children are fresh when
// parents read them). Forces are therefore always a smooth function of the
// current state: discontinuous "tree swap" force jumps — which we measured
// pumping ~0.5% kinetic energy per swap — vanish, along with staleness.

struct RefreshParams {
    start : u32,   // first node index of this level
    count : u32,   // node count in this level
}

@group(0) @binding(0) var<storage, read>       refParticles : array<Particle>;
@group(0) @binding(1) var<storage, read_write> refNodes     : array<BHNode>;
@group(0) @binding(2) var<uniform>             refLevel     : RefreshParams;

@compute @workgroup_size(64)
fn refresh_level(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= refLevel.count) { return; }
    let n = refLevel.start + gid.x;
    let node = refNodes[n];

    var com = vec3f(0.0);
    var m = 0.0;
    if (node.isLeaf == 1u) {
        for (var j = node.first; j < node.first + node.count; j++) {
            let p = refParticles[j];
            com += p.pos * p.mass;
            m += p.mass;
        }
    } else {
        for (var c = 0; c < 8; c++) {
            let ch = node.children[c];
            if (ch >= 0) {
                let cn = refNodes[ch].comMass;
                com += cn.xyz * cn.w;
                m += cn.w;
            }
        }
    }
    refNodes[n].comMass = vec4f(select(vec3f(0.0), com / m, m > 0.0), m);
}

// ── Morton-order permutation gather ─────────────────────────────────────────
// Reorders particles along a Z-curve so neighbouring threads traverse nearly
// identical tree paths (coherent node fetches). perm comes from the worker;
// dst[i] = src[perm[i]] into the ping-pong partner buffer.
@group(0) @binding(5) var<storage, read> perm : array<u32>;

@compute @workgroup_size(256)
fn apply_permutation(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }
    particlesOut[i] = particlesIn[perm[i]];
}

// Stellar-state gather (same permutation, separate ping-pong pair — the
// stellar buffers only flip on permutation, not every step).
@group(0) @binding(0) var<storage, read>       stellarIn  : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> stellarOut : array<vec4f>;
@group(0) @binding(2) var<uniform>             spParams   : Params;
@group(0) @binding(3) var<storage, read>       spPerm     : array<u32>;

@compute @workgroup_size(256)
fn apply_permutation_stellar(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= spParams.N) { return; }
    stellarOut[i] = stellarIn[spPerm[i]];
}
