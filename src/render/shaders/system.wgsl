// Solar-system scene — impostor spheres in a local AU-scale coordinate space.
// Draws into the galaxy's HDR target (loadOp: load) with its own depth buffer.
// Orbits lie in the XY plane (matches the galaxy disk convention).

struct SysUniforms {
    viewProj   : mat4x4f,
    camRight   : vec3f,  time       : f32,
    camUp      : vec3f,  fade       : f32,
    camPos     : vec3f,  starTemp   : f32,
    resolution : vec2f,  tanHalfFov : f32,  _pad : f32,
}

struct Body {
    orbit  : vec4f,   // a, phase, rate, radius
    parent : vec4f,   // parentA, parentPhase, parentRate, kind (as f32)
    look   : vec4f,   // seed, p1, p2, unused
}

struct Ring {
    orbit : vec4f,    // parentA, parentPhase, parentRate, seed
    size  : vec4f,    // inner, outer, unused, unused
}

@group(0) @binding(0) var<uniform> su : SysUniforms;
@group(0) @binding(1) var<storage, read> bodies : array<Body>;
@group(0) @binding(2) var<storage, read> rings : array<Ring>;
// Belt particle = 2 vec4s: [a, phase, rate, size], [incl, node, bright, _]
@group(0) @binding(3) var<storage, read> belt : array<vec4f>;
@group(0) @binding(4) var<storage, read> orbitLines : array<vec4f>;  // a in .x

const QUAD = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
);

fn orbitPos(a : f32, phase : f32, rate : f32) -> vec3f {
    let ang = phase + rate * su.time;
    return vec3f(cos(ang) * a, sin(ang) * a, 0.0);
}

fn bodyCenter(b : Body) -> vec3f {
    // parent orbit (zeros for planets/sun) + own orbit
    return orbitPos(b.parent.x, b.parent.y, b.parent.z)
         + orbitPos(b.orbit.x, b.orbit.y, b.orbit.z);
}

// Planckian blackbody fit — same as render.wgsl
fn blackbodyRGB(tempK : f32) -> vec3f {
    let t = clamp(tempK, 1000.0, 40000.0) / 100.0;
    var r : f32; var g : f32; var b : f32;
    if (t <= 66.0) {
        r = 1.0;
        g = clamp((99.4708025861 * log(t) - 161.1195681661) / 255.0, 0.0, 1.0);
    } else {
        r = clamp(1.292936186 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
        g = clamp(1.1298908609 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
    }
    if (t >= 66.0) {
        b = 1.0;
    } else if (t <= 19.0) {
        b = 0.0;
    } else {
        b = clamp((138.5177312231 * log(t - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0);
    }
    return vec3f(r, g, b);
}

// ── 3D value noise + fbm (procedural surfaces) ──────────────────────────────
fn hash13(p : vec3f) -> f32 {
    var q = fract(p * 443.8975);
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

fn vnoise(p : vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i),                          hash13(i + vec3f(1.0, 0.0, 0.0)), u.x),
            mix(hash13(i + vec3f(0.0, 1.0, 0.0)),   hash13(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash13(i + vec3f(0.0, 0.0, 1.0)),   hash13(i + vec3f(1.0, 0.0, 1.0)), u.x),
            mix(hash13(i + vec3f(0.0, 1.0, 1.0)),   hash13(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y),
        u.z);
}

fn fbm(p : vec3f) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        v += amp * vnoise(q);
        q = q * 2.03;
        amp *= 0.5;
    }
    return v;
}

// Surface color as a function of the unit sphere-surface point — this is the
// piece that ports unchanged to a terrain-mesh pipeline for surface zoom.
fn surfaceColor(kind : u32, n : vec3f, look : vec3f, t : f32) -> vec3f {
    let sp = n * 3.0 + vec3f(look.x * 91.7);   // seed offsets the noise domain
    let h = fbm(sp);
    let lat = abs(n.z);                        // poles along ±Z
    switch kind {
        case 1u: {   // scorched rock
            return mix(vec3f(0.25, 0.12, 0.08), vec3f(0.55, 0.30, 0.15), h);
        }
        case 2u: {   // venusian: thick slowly-churning haze
            let bands = fbm(sp * 1.5 + vec3f(0.0, 0.0, t * 0.02));
            return mix(vec3f(0.75, 0.60, 0.30), vec3f(0.90, 0.80, 0.55), bands);
        }
        case 3u: {   // terran: oceans, continents, ice caps, drifting clouds
            let water = look.y * 0.35 + 0.35;
            var col : vec3f;
            if (h < water) {
                col = mix(vec3f(0.01, 0.05, 0.18), vec3f(0.02, 0.12, 0.30), h / max(water, 0.01));
            } else {
                col = mix(vec3f(0.10, 0.30, 0.12), vec3f(0.45, 0.38, 0.22), fbm(sp * 2.7));
            }
            col = mix(col, vec3f(0.90, 0.92, 0.95), smoothstep(0.78, 0.92, lat));
            let clouds = smoothstep(0.55, 0.75, fbm(sp * 4.0 + vec3f(t * 0.01, 0.0, 0.0)));
            return mix(col, vec3f(1.0), clouds * 0.7);
        }
        case 4u: {   // icy
            return mix(vec3f(0.55, 0.62, 0.70), vec3f(0.85, 0.90, 0.95), h);
        }
        case 5u: {   // gas giant: latitude bands + storms
            let bands = fbm(vec3f(sp.x * 0.3, sp.y * 0.3, n.z * (6.0 + look.z * 6.0)));
            let col = mix(vec3f(0.45, 0.30, 0.18), vec3f(0.80, 0.65, 0.45), bands);
            let storm = smoothstep(0.72, 0.85, fbm(sp * 2.0 + vec3f(31.7)));
            return mix(col, vec3f(0.85, 0.40, 0.25), storm * 0.6);
        }
        default: {   // 6: ice giant
            let bands = fbm(vec3f(sp.x * 0.4, sp.y * 0.4, n.z * 5.0));
            return mix(vec3f(0.15, 0.30, 0.50), vec3f(0.35, 0.60, 0.80), bands);
        }
    }
}

// ── Impostor sphere bodies ──────────────────────────────────────────────────
struct BodyVSOut {
    @builtin(position) pos : vec4f,
    @location(0) worldPos : vec3f,
    @location(1) @interpolate(flat) center : vec3f,
    @location(2) @interpolate(flat) radius : f32,
    @location(3) @interpolate(flat) kind : u32,
    @location(4) @interpolate(flat) look : vec3f,
}

@vertex
fn vs_body(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> BodyVSOut {
    let b = bodies[ii];
    let corner = QUAD[vi] * 1.05;   // slight inflation: silhouette never clips
    let c = bodyCenter(b);
    let world = c + (su.camRight * corner.x + su.camUp * corner.y) * b.orbit.w;
    var out : BodyVSOut;
    out.pos = su.viewProj * vec4f(world, 1.0);
    out.worldPos = world;
    out.center = c;
    out.radius = b.orbit.w;
    out.kind = u32(b.parent.w);
    out.look = b.look.xyz;
    return out;
}

struct BodyFSOut {
    @location(0) color : vec4f,
    @builtin(frag_depth) depth : f32,
}

@fragment
fn fs_body(in : BodyVSOut) -> BodyFSOut {
    let ro = su.camPos;
    let rd = normalize(in.worldPos - ro);
    let oc = ro - in.center;
    let bq = dot(oc, rd);
    let cq = dot(oc, oc) - in.radius * in.radius;
    let disc = bq * bq - cq;
    if (disc < 0.0) { discard; }
    let thit = -bq - sqrt(disc);
    if (thit < 0.0) { discard; }
    let hit = ro + rd * thit;
    let n = normalize(hit - in.center);

    var out : BodyFSOut;
    let clip = su.viewProj * vec4f(hit, 1.0);
    out.depth = clip.z / clip.w;

    if (in.kind == 0u) {   // the star: emissive, limb-darkened, HDR-hot
        let limb = 0.4 + 0.6 * abs(dot(n, rd));
        let granule = 0.9 + 0.2 * fbm(n * 8.0 + vec3f(su.time * 0.05));
        let col = blackbodyRGB(su.starTemp) * 14.0 * limb * granule;
        out.color = vec4f(col * su.fade, su.fade);
        return out;
    }

    let lightDir = -normalize(hit);   // the sun sits at the origin
    let diff = max(dot(n, lightDir), 0.0);
    var col = surfaceColor(in.kind, n, in.look, su.time) * (diff * 1.2 + 0.03);
    // Atmosphere rim on the lit edge
    let rim = pow(1.0 - abs(dot(n, -rd)), 3.0);
    var atmo = vec3f(0.0);
    if (in.kind == 3u) { atmo = vec3f(0.30, 0.50, 1.00); }
    if (in.kind == 5u) { atmo = vec3f(0.20, 0.16, 0.12); }
    col += atmo * rim * (diff + 0.15);
    out.color = vec4f(col * su.fade, su.fade);   // premultiplied crossfade
    return out;
}
