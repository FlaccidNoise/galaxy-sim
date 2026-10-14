// Solar-system scene — impostor spheres in a local AU-scale coordinate space.
// Draws into the galaxy's HDR target (loadOp: load) with its own depth buffer.
// Orbits lie in the XY plane (matches the galaxy disk convention).

struct SysUniforms {
    viewProj   : mat4x4f,
    camRight   : vec3f,  time       : f32,
    camUp      : vec3f,  fade       : f32,
    camPos     : vec3f,  starTemp   : f32,
    resolution : vec2f,  tanHalfFov : f32,  frost : f32,   // frost line, rendered units
}

struct Body {
    orbit  : vec4f,   // a, phase, rate, radius
    parent : vec4f,   // parentA, parentPhase, parentRate, kind (as f32)
    look   : vec4f,   // seed, p1, p2, ring index + 1 (0 = none)
}

struct Ring {
    orbit : vec4f,    // parentA, parentPhase, parentRate, seed
    size  : vec4f,    // inner, outer, planet radius, unused
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

// Voronoi craters: x/y/z = height gradient, w = height (bowl < 0, rim > 0)
fn craters(p : vec3f) -> vec4f {
    let i = floor(p);
    var acc = vec4f(0.0);
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let c = i + vec3f(f32(x), f32(y), f32(z));
                let d = p - c - vec3f(hash13(c), hash13(c + 3.3), hash13(c + 5.7));
                let rad = 0.2 + 0.4 * hash13(c + 7.1);
                let len = length(d);
                let u = len / rad;
                if (u < 1.6) {
                    let rim = 0.35 * exp(-(u - 1.0) * (u - 1.0) * 25.0);
                    let inside = select(0.0, 1.0, u < 1.0);
                    let h = inside * 0.6 * (u * u - 1.0) + rim;
                    let dh = inside * 1.2 * u - rim * 50.0 * (u - 1.0);
                    acc += vec4f(d / max(len, 1e-4) * dh / rad, h);
                }
            }
        }
    }
    return acc;
}

// Storm oval centred at (z0, lon0): x = interior, y = dark collar
fn oval(n : vec3f, z0 : f32, lon0 : f32, size : f32) -> vec2f {
    var dl = atan2(n.y, n.x) - lon0;
    dl -= 6.2831853 * round(dl / 6.2831853);
    let q = length(vec2f(dl * sqrt(max(1.0 - n.z * n.z, 0.0)) / (size * 2.2), (n.z - z0) / size));
    return vec2f(1.0 - smoothstep(0.6, 1.0, q), smoothstep(0.7, 0.95, q) * (1.0 - smoothstep(0.95, 1.2, q)));
}

// Gas-giant band palettes (zone, belt): Jupiter ochre, Saturn cream, salmon,
// slate, and the cold teal that outer giants fade toward
const GIANT_ZONE = array<vec3f, 5>(
    vec3f(0.52, 0.39, 0.22), vec3f(0.54, 0.45, 0.27), vec3f(0.54, 0.30, 0.21),
    vec3f(0.34, 0.35, 0.38), vec3f(0.17, 0.35, 0.42),
);
const GIANT_BELT = array<vec3f, 5>(
    vec3f(0.29, 0.13, 0.05), vec3f(0.39, 0.29, 0.14), vec3f(0.34, 0.12, 0.08),
    vec3f(0.17, 0.18, 0.21), vec3f(0.04, 0.13, 0.26),
);

// Surface albedo (rgb) and ocean mask (a) as a function of the unit
// sphere-surface point in the body frame (pole = +Z) — this is the piece that
// ports unchanged to a terrain-mesh pipeline for surface zoom.
fn surfaceColor(kind : u32, n : vec3f, look : vec4f, t : f32, cold : f32) -> vec4f {
    let sp = n * 3.0 + vec3f(look.x * 91.7);   // seed offsets the noise domain
    switch kind {
        case 1u: {   // scorched rock
            return vec4f(mix(vec3f(0.25, 0.12, 0.08), vec3f(0.55, 0.30, 0.15), fbm(sp)), 0.0);
        }
        case 2u: {   // venusian: thick slowly-churning haze
            let warp = fbm(sp * 1.2 + vec3f(0.0, 0.0, t * 0.02));
            let haze = fbm(vec3f(sp.xy * 0.8, n.z * 3.0) + warp * 1.5);
            return vec4f(mix(vec3f(0.78, 0.66, 0.40), vec3f(0.95, 0.88, 0.68), haze), 0.0);
        }
        case 3u: {   // terran: altitude-shaded continents, shelf seas, snow line, polar caps
            let h = fbm(sp);
            let water = 0.47 + 0.1 * look.y;   // ~50–80 % ocean
            let e = clamp((h - water) / 0.2, 0.0, 1.0);
            let lat = abs(n.z);
            var col = mix(vec3f(0.14, 0.19, 0.08), vec3f(0.46, 0.39, 0.26), smoothstep(0.4, 0.6, fbm(sp * 2.7)));
            col = mix(col, vec3f(0.33, 0.26, 0.17), smoothstep(0.2, 0.5, e));
            col = mix(col, vec3f(0.42, 0.40, 0.37), smoothstep(0.5, 0.7, e));
            col = mix(col, vec3f(0.90, 0.92, 0.95), smoothstep(0.7, 0.8, e + lat * 0.6));   // snow line drops poleward
            let sea = 1.0 - smoothstep(water - 0.005, water + 0.005, h);
            col = mix(col, mix(vec3f(0.04, 0.20, 0.28), vec3f(0.01, 0.04, 0.13), smoothstep(0.0, 0.04, water - h)), sea);
            let cap = smoothstep(0.8, 0.92, lat);
            col = mix(col, vec3f(0.90, 0.92, 0.95), cap);
            return vec4f(col, sea * (1.0 - cap));
        }
        case 4u: {   // icy
            return vec4f(mix(vec3f(0.55, 0.62, 0.70), vec3f(0.85, 0.90, 0.95), fbm(sp)), 0.0);
        }
        case 5u: {   // gas giant: domain-warped latitude bands + storm ovals
            let warp = fbm(sp * 1.3 + vec3f(t * 0.003, 0.0, 0.0));
            let z = n.z + (warp - 0.5) * 0.15;
            let freq = 5.0 + look.z * 6.0;
            let belts = 0.5 + 0.5 * sin(z * freq * 3.14159 + look.x * 6.2831853);
            let fine = fbm(vec3f(sp.xy * 0.35, z * freq * 3.0) + warp);
            let f = look.y * 3.999;
            let i = u32(f);
            let j = min(i + 1u, 3u);
            let b = smoothstep(0.7, 1.0, fract(f));
            let zone = mix(mix(GIANT_ZONE[i], GIANT_ZONE[j], b), GIANT_ZONE[4], cold);
            let belt = mix(mix(GIANT_BELT[i], GIANT_BELT[j], b), GIANT_BELT[4], cold);
            let contrast = 0.1 + 0.4 * fract(look.x * 13.7);
            var col = mix(zone, belt, (1.0 - clamp(belts * 0.7 + fine * 0.5 - 0.1, 0.0, 1.0)) * min(contrast * 2.5, 1.0));
            let s1 = oval(n, -0.35 + 0.3 * look.y, look.z * 6.2831853, 0.07 + 0.05 * look.y);
            col = mix(col, select(zone * 1.08, vec3f(0.72, 0.30, 0.18), fract(look.x * 7.3) < 0.45), s1.x);
            col *= 1.0 - 0.3 * s1.y;
            if (look.y > 0.4) {
                let s2 = oval(n, 0.4 - 0.2 * look.z, look.x * 6.2831853 + 2.0, 0.04);
                col = mix(col, vec3f(0.92, 0.90, 0.85), s2.x);
            }
            return vec4f(col, 0.0);
        }
        case 6u, 8u: {   // ice giant and sub-Neptune: faint bands, polar haze, dark spot
            let warp = fbm(sp + vec3f(t * 0.003, 0.0, 0.0));
            let z = n.z + (warp - 0.5) * 0.1;
            let bands = fbm(vec3f(sp.xy * 0.3, z * 8.0));
            var col = mix(mix(vec3f(0.36, 0.64, 0.70), vec3f(0.42, 0.72, 0.78), bands),
                          mix(vec3f(0.03, 0.09, 0.38), vec3f(0.05, 0.14, 0.48), bands), look.y);   // pale cyan → deep azure
            col = mix(col, vec3f(0.75, 0.85, 0.90), smoothstep(0.75, 0.95, abs(z)) * 0.5);
            col = mix(col, col * 0.45, oval(n, -0.3, look.z * 6.2831853, 0.06).x * step(look.y, 0.5));
            return vec4f(col, 0.0);
        }
        default: {   // 7: rocky moon — grey regolith with darker maria
            let maria = smoothstep(0.45, 0.6, fbm(sp * 0.7));
            return vec4f(mix(vec3f(0.42, 0.40, 0.38), vec3f(0.22, 0.21, 0.20), maria)
                         * (0.85 + 0.3 * fbm(sp * 2.0)), 0.0);
        }
    }
}

// Cloud cover drifting around the pole; the surface beneath stays fixed
fn clouds(n : vec3f, seed : f32, t : f32) -> f32 {
    let a = t * 0.012;
    let p = vec3f(n.x * cos(a) - n.y * sin(a), n.x * sin(a) + n.y * cos(a), n.z) * 6.0 + vec3f(seed * 53.1);
    let mid = 0.501 - 0.058 * fract(seed * 5.3);   // 40–60 % cover
    return smoothstep(mid - 0.08, mid + 0.08, fbm(p + fbm(p * 0.5) * 1.8));
}

// Atmosphere tint (rgb) and strength (a); a = 0 means airless
fn atmoTint(kind : u32) -> vec4f {
    switch kind {
        case 2u: { return vec4f(1.00, 0.90, 0.65, 0.9); }   // venusian: yellow-white
        case 3u: { return vec4f(0.35, 0.60, 1.00, 1.0); }   // terran: Rayleigh blue
        case 5u: { return vec4f(0.80, 0.82, 0.85, 0.5); }   // gas giant: pale, neutral across palettes
        case 6u: { return vec4f(0.40, 0.85, 1.00, 0.8); }   // ice giant: cyan
        case 8u: { return vec4f(0.70, 0.80, 0.90, 1.0); }   // sub-Neptune: thick grey-blue haze
        default: { return vec4f(0.0); }
    }
}

// Limb scattering at grazing view: brightest on the lit limb, with a thin
// forward-scatter glow just past the terminator when looking sunward.
fn limbGlow(tint : vec4f, ndl : f32, rd : vec3f, l : vec3f) -> vec3f {
    let lit = smoothstep(-0.3, 0.5, ndl);
    let dusk = smoothstep(-0.35, 0.0, ndl) * (1.0 - smoothstep(0.0, 0.2, ndl));
    let fwd = pow(max(dot(rd, l), 0.0), 6.0);
    return tint.rgb * tint.w * (lit + dusk * (0.25 + 2.0 * fwd));
}

// Rings are equatorial; the tilt is seeded so the ring/planet shadows show
fn ringNormal(seed : f32) -> vec3f {
    let tilt = 0.25 + 0.3 * fract(seed * 7.31);
    let az = seed * 6.2831853;
    return vec3f(sin(tilt) * cos(az), sin(tilt) * sin(az), cos(tilt));
}

fn ringBasis(nrm : vec3f) -> mat3x3f {
    let e1 = normalize(cross(vec3f(0.0, 1.0, 0.0), nrm));
    return mat3x3f(e1, cross(nrm, e1), nrm);
}

fn ringDensity(rad : f32, inner : f32, outer : f32, seed : f32) -> f32 {
    let t = (rad - inner) / (outer - inner);
    if (t < 0.0 || t > 1.0) { return 0.0; }
    let bands = 0.3 + 0.45 * vnoise(vec3f(t * 24.0, seed * 97.0, 0.0)) + 0.25 * vnoise(vec3f(t * 90.0, seed * 31.0, 0.0));
    let gap = smoothstep(0.0, 0.02, abs(t - 0.55 - 0.2 * fract(seed * 3.7)));
    return bands * gap * smoothstep(0.0, 0.06, t) * smoothstep(1.0, 0.94, t);
}

// ── Impostor sphere bodies ──────────────────────────────────────────────────
struct BodyVSOut {
    @builtin(position) pos : vec4f,
    @location(0) worldPos : vec3f,
    @location(1) @interpolate(flat) center : vec3f,
    @location(2) @interpolate(flat) radius : f32,
    @location(3) @interpolate(flat) kind : u32,
    @location(4) @interpolate(flat) look : vec4f,
}

@vertex
fn vs_body(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> BodyVSOut {
    let b = bodies[ii];
    let c = bodyCenter(b);
    // Cover the perspective silhouette plus the atmosphere halo, even up close
    let k = b.orbit.w / max(length(c - su.camPos), 1e-6);
    let corner = QUAD[vi] * 1.15 / sqrt(max(1.0 - k * k, 0.01));
    let world = c + (su.camRight * corner.x + su.camUp * corner.y) * b.orbit.w;
    var out : BodyVSOut;
    out.pos = su.viewProj * vec4f(world, 1.0);
    out.worldPos = world;
    out.center = c;
    out.radius = b.orbit.w;
    out.kind = u32(b.parent.w);
    out.look = b.look;
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
    let sunCol = blackbodyRGB(su.starTemp);
    let tint = atmoTint(in.kind);
    var out : BodyFSOut;

    if (disc < 0.0) {   // ray misses: thin atmosphere halo just outside the limb
        if (tint.w == 0.0 || bq > 0.0) { discard; }
        let p = ro - rd * bq;
        let alt = (length(p - in.center) / in.radius - 1.0) / 0.012;
        let glow = sunCol * limbGlow(tint, dot(normalize(p - in.center), -normalize(p)), rd, -normalize(p)) * exp(-alt);
        out.color = vec4f(glow * su.fade, 0.0);   // additive under premultiplied blend
        out.depth = 0.99999;                      // never occludes what is drawn later
        return out;
    }
    var thit = -bq - sqrt(disc);
    if (thit < 0.0) { thit = -bq + sqrt(disc); }   // camera inside the sphere: far root
    if (thit < 0.0) { discard; }
    let hit = ro + rd * thit;
    let n = normalize(hit - in.center);

    let clip = su.viewProj * vec4f(hit, 1.0);
    out.depth = clip.z / clip.w;

    if (in.kind == 0u) {   // the star: emissive, limb-darkened, granulated, HDR-hot
        let mu = abs(dot(n, rd));
        let limb = 1.0 - 0.7 * (1.0 - mu);
        let gran = 0.5 + 0.8 * smoothstep(0.3, 0.7, fbm(n * 40.0 + vec3f(0.0, 0.0, su.time * 0.03)));
        let spots = 1.0 - 0.85 * smoothstep(0.66, 0.76, fbm(n * 2.5 + vec3f(17.3)));
        let bb = blackbodyRGB(su.starTemp * (0.7 + 0.3 * mu));
        let col = bb * bb * 3.0 * limb * gran * spots;   // squared: keeps the tint through AgX's path to white
        out.color = vec4f(col * su.fade, su.fade);
        return out;
    }

    let l = -normalize(hit);   // the sun sits at the origin
    var frame = ringBasis(vec3f(0.0, 0.0, 1.0));
    var shade = 1.0;
    let ri = u32(in.look.w);
    if (ri > 0u) {   // ringed: pole follows the ring plane, which shadows the planet
        let rg = rings[ri - 1u];
        frame = ringBasis(ringNormal(rg.orbit.w));
        let s = dot(in.center - hit, frame[2]) / dot(l, frame[2]);
        if (s > 0.0) {
            shade = 1.0 - 0.85 * ringDensity(length(hit + l * s - in.center), rg.size.x, rg.size.y, rg.orbit.w);
        }
    }
    let nb = n * frame;   // world → body frame
    // Distance past the frost line: colder giants turn bluer
    let cold = smoothstep(2.5, 7.0, length(in.center) / su.frost);
    let surf = surfaceColor(in.kind, nb, in.look, su.time, cold);
    var albedo = surf.rgb;
    var ns = n;
    if (in.kind == 1u || in.kind == 7u) {   // airless: bump-shaded craters
        let cr = craters(nb * 5.0 + vec3f(in.look.x * 37.0));
        let g = frame * cr.xyz;
        ns = normalize(n - 0.08 * (g - dot(g, n) * n));
        albedo *= 1.0 + 0.25 * cr.w;
    }
    let ndl = dot(n, l);
    let light = sunCol * max((dot(ns, l) + 0.04) / 1.04, 0.0) * 1.3 * shade;   // soft terminator: the sun is a disc
    var col = albedo * (light + 0.015);
    let mu = max(dot(n, -rd), 0.0);

    if (in.kind == 3u) {
        let lb = l * frame;
        let cloud = clouds(nb, in.look.x, su.time);
        let cshadow = 1.0 - 0.6 * clouds(normalize(nb + lb * 0.03), in.look.x, su.time);
        let h = normalize(l - rd);
        let spec = pow(max(dot(n, h), 0.0), 500.0) * surf.a * step(0.0, ndl) * (1.0 - cloud);
        col = albedo * (light * cshadow + 0.015) + sunCol * spec * 3.0 * shade * cshadow;
        col = mix(col, vec3f(0.95) * (light + 0.015), cloud * 0.9);
    }
    if (in.kind == 5u || in.kind == 6u || in.kind == 8u) {
        col *= 0.55 + 0.45 * sqrt(mu);   // limb darkening
    }
    if (tint.w > 0.0) {
        col += sunCol * shade * limbGlow(tint, ndl, rd, l) * pow(1.0 - mu, 4.0);
    }
    out.color = vec4f(col * su.fade, su.fade);   // premultiplied crossfade
    return out;
}

// ── Corona: additive halo around the star, ~1/r² out to 4 radii ────────────
struct CoronaVSOut {
    @builtin(position) pos : vec4f,
    @location(0) worldPos : vec3f,
    @location(1) @interpolate(flat) radius : f32,
}

@vertex
fn vs_corona(@builtin(vertex_index) vi : u32) -> CoronaVSOut {
    let r = bodies[0].orbit.w;
    let world = (su.camRight * QUAD[vi].x + su.camUp * QUAD[vi].y) * r * 4.0;
    var out : CoronaVSOut;
    out.pos = su.viewProj * vec4f(world, 1.0);
    out.worldPos = world;
    out.radius = r;
    return out;
}

@fragment
fn fs_corona(in : CoronaVSOut) -> @location(0) vec4f {
    let rd = normalize(in.worldPos - su.camPos);
    let x = max(length(su.camPos + rd * max(-dot(su.camPos, rd), 0.0)) / in.radius, 1.0);
    let w = 1.0 - min(x / 4.0, 1.0);
    return vec4f(blackbodyRGB(su.starTemp) * 2.5 * w * w * w / (x * x) * su.fade, 0.0);
}

// ── Planetary rings: flat annulus in the tilted equatorial plane ───────────
struct RingVSOut {
    @builtin(position) pos : vec4f,
    @location(0) local : vec2f,
    @location(1) worldPos : vec3f,
    @location(2) @interpolate(flat) center : vec3f,
    @location(3) @interpolate(flat) size : vec4f,
    @location(4) @interpolate(flat) seed : f32,
}

@vertex
fn vs_ring(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> RingVSOut {
    let rg = rings[ii];
    let c = orbitPos(rg.orbit.x, rg.orbit.y, rg.orbit.z);
    let local = QUAD[vi] * rg.size.y;
    let world = c + ringBasis(ringNormal(rg.orbit.w)) * vec3f(local, 0.0);
    var out : RingVSOut;
    out.pos = su.viewProj * vec4f(world, 1.0);
    out.local = local;
    out.worldPos = world;
    out.center = c;
    out.size = rg.size;
    out.seed = rg.orbit.w;
    return out;
}

@fragment
fn fs_ring(in : RingVSOut) -> @location(0) vec4f {
    let rad = length(in.local);
    if (rad < in.size.x || rad > in.size.y) { discard; }
    let d = ringDensity(rad, in.size.x, in.size.y, in.seed);
    // Planet shadow: does the ray toward the sun hit the planet?
    let l = -normalize(in.worldPos);
    let oc = in.worldPos - in.center;
    let b = dot(oc, l);
    var lit = 1.0;
    if (b < 0.0) { lit = smoothstep(0.97, 1.03, sqrt(max(dot(oc, oc) - b * b, 0.0)) / in.size.z); }
    // Seen from the unlit face, only light diffusing through thin regions shows
    let nrm = ringNormal(in.seed);
    if (dot(su.camPos - in.worldPos, nrm) * dot(l, nrm) < 0.0) { lit *= 0.35 * (1.0 - 0.6 * d); }
    let t = (rad - in.size.x) / (in.size.y - in.size.x);
    let col = mix(vec3f(0.75, 0.68, 0.58), vec3f(0.55, 0.50, 0.45), vnoise(vec3f(t * 7.0, in.seed * 13.0, 0.0)));
    let a = d * 0.8 * su.fade;
    return vec4f(col * blackbodyRGB(su.starTemp) * (lit * 1.2 + 0.015) * a, a);
}

// ── Asteroid belt: instanced Keplerian point sprites ────────────────────────
struct BeltVSOut {
    @builtin(position) pos : vec4f,
    @location(0) quadUV : vec2f,
    @location(1) @interpolate(flat) color : vec3f,
}

@vertex
fn vs_belt(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> BeltVSOut {
    let d0 = belt[ii * 2u];        // a, phase, rate, size
    let d1 = belt[ii * 2u + 1u];   // incl, node, bright, tint
    let ang = d0.y + d0.z * su.time;
    let p = vec3f(cos(ang) * d0.x, sin(ang) * d0.x, sin(ang + d1.y) * d0.x * d1.x);
    var clip = su.viewProj * vec4f(p, 1.0);
    let corner = QUAD[vi];
    // Projected diameter of a ~0.002 AU rock: sub-pixel dust from afar, specks up close
    let px = 0.002 * d0.w * su.resolution.y / (max(clip.w, 1e-4) * su.tanHalfFov);
    let sizePx = clamp(px, 1.5, 8.0);   // smaller quads would miss pixel centres
    clip = vec4f(clip.xy + corner * (sizePx / su.resolution) * clip.w, clip.zw);
    var out : BeltVSOut;
    out.pos = clip;
    out.quadUV = corner;
    out.color = mix(vec3f(0.60, 0.60, 0.58), vec3f(0.62, 0.47, 0.33), d1.w)
              * blackbodyRGB(su.starTemp) * d1.z * clamp(px / 1.5, 0.4, 1.0);
    return out;
}

@fragment
fn fs_belt(in : BeltVSOut) -> @location(0) vec4f {
    let d2 = dot(in.quadUV, in.quadUV);
    if (d2 > 1.0) { discard; }
    return vec4f(in.color * exp(-d2 * 3.0) * 0.6 * su.fade, 1.0);
}

// ── Faint orbit guide lines ─────────────────────────────────────────────────
@vertex
fn vs_orbitline(@builtin(vertex_index) vi : u32,
                @builtin(instance_index) ii : u32) -> @builtin(position) vec4f {
    let a = orbitLines[ii].x;
    let ang = f32(vi) / 128.0 * 6.2831853;
    return su.viewProj * vec4f(cos(ang) * a, sin(ang) * a, 0.0, 1.0);
}

@fragment
fn fs_orbitline() -> @location(0) vec4f {
    return vec4f(vec3f(0.25, 0.40, 0.80) * 0.05 * su.fade, 1.0);
}
