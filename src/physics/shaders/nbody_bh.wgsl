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
    pad  : f32,   // 0 = star; < -0.5 = GAS, encoded as -(1 + localDensity).
                  // The force kernel rewrites the density every step from the
                  // particle's own tree leaf; the star-formation pass reads it.
}

struct Params {
    dt        : f32,
    G         : f32,
    epsilon2  : f32,  // softening², prevents singularity
    N         : u32,
    theta     : f32,  // Barnes-Hut opening angle (smaller = more accurate)
    haloMass  : f32,  // Hernquist halo mass PER HALO (0 = off)
    haloA     : f32,  // halo scale radius
    gasDrag   : f32,  // gas cooling rate (damps non-azimuthal motion)
    time      : f32,  // simulation time (units)
    agingRate : f32,  // Myr of stellar aging per time unit
    sfDensity : f32,  // star-formation density threshold (particles/unit³)
    sfEff     : f32,  // star-formation efficiency
    sfDt      : f32,  // dt · stepsThisFrame (for per-frame SF probability)
    haloCount : u32,  // 0, 1, or 2 active halos
    _pad0     : f32,
    _pad1     : f32,
    // Halo centres are SMOOTHED galaxy positions (EMA of each central BH),
    // updated on the CPU. Slaving the halo to the live BH particle is
    // violently unstable: the massless potential teleports with BH jitter
    // and pumps energy into the disk.
    haloC1    : vec4f,
    haloC2    : vec4f,
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
@group(0) @binding(6) var<storage, read_write> energyOut    : array<f32>;

const STACK_SIZE : i32 = 96;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let pi  = particlesIn[i];
    var acc = vec3f(0.0);
    var pot = 0.0;          // star-star potential (for the energy diagnostic)
    var density = 0.0;      // local density from this particle's own leaf
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

        // Local density: the leaf whose bucket contains this particle.
        if (isLeaf && i >= node.first && i < node.first + node.count) {
            density = f32(node.count) / (nodeSize * nodeSize * nodeSize);
        }

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
            let f = params.G * nodeMass / (d2 * sqrt(d2));
            acc += r * f;
            pot -= f * d2;      // = G·M/√d2
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
                let f = params.G * pj.mass / (dj2 * sqrt(dj2));
                acc += rj * f;
                pot -= f * dj2;
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

    // Hernquist dark-matter halo(s) at smoothed galaxy centres, so colliding
    // galaxies carry their dark matter with them.
    // M(r) = M_h·r²/(r+a)² → a(r) = -G·M_h/(r+a)² r̂; φ = -G·M_h/(r+a).
    var potHalo = 0.0;
    if (params.haloMass > 0.0) {
        for (var h = 0u; h < params.haloCount; h++) {
            let centre = select(params.haloC2.xyz, params.haloC1.xyz, h == 0u);
            let d = pi.pos - centre;
            let rlen = max(length(d), 1e-6);
            let ra = rlen + params.haloA;
            acc -= d * (params.G * params.haloMass / (ra * ra * rlen));
            potHalo -= params.G * params.haloMass / ra;
        }
    }

    // Kick-drift with constant dt ≡ leapfrog with half-step-offset velocities
    // (2nd-order symplectic); only dt changes break the equivalence.
    var newVel = pi.vel + acc * params.dt;

    // Gas cooling: radiative dissipation proxy — exponentially damp the
    // non-azimuthal velocity components, so gas settles into a cold, thin,
    // rotating disk (where star formation then happens).
    let isGas = pi.pad < -0.5;
    if (isGas && params.gasDrag > 0.0) {
        let phiHat = normalize(vec3f(-pi.pos.y, pi.pos.x, 1e-6));
        let vPhi = dot(newVel, phiHat) * phiHat;
        newVel = vPhi + (newVel - vPhi) * exp(-params.gasDrag * params.dt);
    }

    let newPos = pi.pos + newVel * params.dt;

    // pad: stars stay 0; gas carries -(1 + localDensity) for the SF pass.
    let newPad = select(0.0, -(1.0 + density), isGas);
    particlesOut[i] = Particle(newPos, pi.mass, newVel, newPad);

    // Specific energy: ½v² + ½·φ_pairs (half: pairwise double-count) + φ_halo
    energyOut[i] = 0.5 * dot(newVel, newVel) + 0.5 * pot + potHalo;
}

// ── Position compaction for async tree-rebuild readback ─────────────────────
// Packs pos.xyz into a tight vec4 array so the CPU readback transfers
// 16 bytes/particle instead of the full 32-byte particle state.
@group(0) @binding(4) var<storage, read_write> posOut : array<vec4f>;

@compute @workgroup_size(256)
fn compact_positions(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }
    // w carries the per-particle specific energy: the worker sums it for the
    // energy-drift diagnostic, piggybacking on the tree-rebuild readback.
    posOut[i] = vec4f(particlesIn[i].pos, energyOut[i]);
}

// ── Star formation & stellar recycling ──────────────────────────────────────
// Runs once per frame, in place (1:1 index mapping, no ping-pong needed):
//   gas + dense  → new star: Kroupa IMF mass, birthTime = now
//   massive star past its supernova → returns to gas with an ejecta kick
// This closes the baryon cycle: arms stay blue because they FORM stars.
@group(0) @binding(0) var<storage, read_write> sfParticles : array<Particle>;
@group(0) @binding(1) var<storage, read_write> sfStellar   : array<vec4f>;
@group(0) @binding(2) var<uniform>             sfParams    : Params;

fn pcg(v0 : u32) -> u32 {
    var v = v0 * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}
fn rand01(i : u32, salt : u32) -> f32 {
    return f32(pcg(i ^ (salt * 0x9E3779B9u))) / 4294967295.0;
}

// Kroupa (2001) IMF inverse-CDF: α=1.3 on [0.08,0.5], α=2.3 on [0.5,100]
fn imfSample(u1 : f32, u2 : f32) -> f32 {
    let a0 = pow(0.08, -0.3);
    let b0 = pow(0.5, -0.3);
    let lowInt = (a0 - b0) / 0.3;
    let a1 = pow(0.5, -1.3);
    let b1 = pow(100.0, -1.3);
    let hiInt = 0.5 * (a1 - b1) / 1.3;
    if (u1 < lowInt / (lowInt + hiInt)) {
        return pow(a0 - u2 * (a0 - b0), -1.0 / 0.3);
    }
    return pow(a1 - u2 * (a1 - b1), -1.0 / 1.3);
}

@compute @workgroup_size(256)
fn star_formation(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= sfParams.N) { return; }
    let salt = u32(sfParams.time * 977.0) + 1u;
    let p = sfParticles[i];
    let st = sfStellar[i];

    if (p.pad < -0.5) {
        // GAS: collapse probability rises with density above the threshold
        // (∝ √ρ mimics the shorter free-fall time of denser clouds)
        let density = -p.pad - 1.0;
        if (density > sfParams.sfDensity && sfParams.sfEff > 0.0) {
            let prob = sfParams.sfEff * sfParams.sfDt
                     * min(sqrt(density / max(sfParams.sfDensity, 1e-6)), 3.0);
            if (rand01(i, salt) < prob) {
                let m = imfSample(rand01(i, salt + 7u), rand01(i, salt + 13u));
                let birthMyr = sfParams.time * sfParams.agingRate;
                sfStellar[i] = vec4f(m, birthMyr, rand01(i, salt + 29u), 0.0);
                sfParticles[i].pad = 0.0;   // now a star
            }
        }
    } else if (st.x >= 8.0) {
        // Massive STAR: after the supernova has faded (+30 Myr), the remnant
        // and ejecta rejoin the gas phase, with a velocity kick.
        let ageMyr = sfParams.time * sfParams.agingRate - st.y;
        let tMS = 10000.0 * pow(st.x, -2.5);
        let tGiant = tMS * 0.12 + 15.0;
        if (ageMyr > tMS + tGiant + 15.0 + 30.0) {
            sfStellar[i] = vec4f(-1.0, 0.0, st.z, 0.7);   // gas again, dusty
            sfParticles[i].pad = -1.0;
            let kick = vec3f(
                rand01(i, salt + 41u) - 0.5,
                rand01(i, salt + 43u) - 0.5,
                rand01(i, salt + 47u) - 0.5,
            ) * 2.0;
            sfParticles[i].vel += kick;
        }
    }
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
