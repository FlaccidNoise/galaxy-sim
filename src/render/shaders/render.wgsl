// Galaxy render pipeline — raw WebGPU.
// Entry points:
//   vs_particle / fs_particle    — instanced billboard sprites, vertex-pulled
//                                  straight from the physics storage buffer
//   vs_fullscreen / fs_background — procedural deep-sky starfield
//   vs_fullscreen / fs_downsample_first / fs_downsample — bloom mip chain down
//   vs_fullscreen / fs_upsample  — bloom tent-filter up (additive)
//   vs_fullscreen / fs_composite — scene + bloom, exposure, AgX, sRGB out

// ── Shared particle layout (must match nbody_bh.wgsl) ──────────────────────
struct Particle {
    pos       : vec3f,
    mass      : f32,
    vel       : vec3f,
    colorMass : f32,   // spectral classification mass (0 ⇒ fall back to mass)
}

struct Uniforms {
    viewProj      : mat4x4f,
    camRight      : vec3f,  tanHalfFov : f32,
    camUp         : vec3f,  aspect     : f32,
    camForward    : vec3f,  time       : f32,
    camPos        : vec3f,  starIntensity : f32,
    resolution    : vec2f,  sizeScale  : f32,  colorMode : u32,
    speedScale    : f32,    centralIdx : u32,  agingRate : f32,  dustStrength : f32,
    hiiStrength   : f32,    systemFade : f32,  selectedIdx : u32,  _q2 : f32,
}

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform> u : Uniforms;
// Stellar state: [mass M☉, birthTime Myr (≤0 at t=0), seed, reserved]
@group(0) @binding(2) var<storage, read> stellar : array<vec4f>;

// ── Particle pass ───────────────────────────────────────────────────────────
struct VSOut {
    @builtin(position) pos : vec4f,
    @location(0) quadUV : vec2f,   // [-1, 1] across the sprite
    @location(1) color  : vec3f,
    @location(2) spike  : f32,     // quad half-size in core radii (1 = no spikes)
    @location(3) flux   : f32,     // near-field brightness gain for the core
}

// Planckian-locus blackbody color (Tanner Helland's fit to CIE data),
// normalized to [0,1] linear-ish sRGB. Valid ~1000–40000 K.
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

// ── Stellar lifecycle, derived statelessly from (mass, age) ────────────────
// MS lifetime t_MS ≈ 10 Gyr · M^-2.5. Phases: main sequence → red giant
// (cool, ×60 brighten) → for ≥8 M☉ a supernova flash then a near-invisible
// remnant; below 8 M☉ a hot dim white dwarf that cools.
struct StarState {
    temp : f32,
    lum  : f32,
}

fn starState(M : f32, ageIn : f32, seed : f32) -> StarState {
    let tMS = 10000.0 * pow(M, -2.5);
    let tGiant = tMS * 0.12 + 15.0;
    let tempMS = 5778.0 * pow(M, 0.54);
    let snDur = 15.0;   // Myr — stretched so a flash lasts ~1 s of real time

    // Lifecycle is linear: the star-formation compute pass returns massive
    // remnants to the gas phase, and dense gas births new stars — so the
    // supernova rate is sustained by actual star formation.
    let ageMyr = ageIn;

    var s : StarState;
    if (ageMyr < tMS) {
        s.temp = tempMS;
        s.lum = pow(M, 3.5);
    } else if (ageMyr < tMS + tGiant) {
        let f = (ageMyr - tMS) / tGiant;
        s.temp = mix(tempMS, 3400.0, clamp(f * 2.5, 0.0, 1.0));
        s.lum = pow(M, 3.5) * mix(1.0, 60.0, f * f);
    } else if (M >= 8.0) {
        let sn = ageMyr - tMS - tGiant;
        if (sn < snDur) {
            let f = sn / snDur;
            s.temp = 9000.0;
            s.lum = 5.0e4 * exp(-5.0 * f);
        } else {
            s.temp = 40000.0;  // neutron star: hot, vanishingly dim
            s.lum = 0.0005;
        }
    } else {
        let cool = (ageMyr - tMS - tGiant) * 2.0e-4;
        s.temp = 14000.0 / (1.0 + cool);
        s.lum = 0.004;
    }
    return s;
}

fn velocityColor(speed : f32) -> vec3f {
    let t = min(speed / max(u.speedScale, 0.001), 1.0);
    if (t < 0.33) {
        let s = t / 0.33;
        return vec3f(1.0, s * 0.5, 0.0);
    } else if (t < 0.66) {
        let s = (t - 0.33) / 0.33;
        return vec3f(1.0, 0.5 + s * 0.5, s * 0.5);
    }
    let s = (t - 0.66) / 0.34;
    return vec3f(1.0 - s * 0.5, 1.0 - s * 0.5, 0.5 + s * 0.5);
}

fn massColor(mass : f32) -> vec3f {
    let t = min(log(mass + 1.0) / 10.0, 1.0);
    let g = select(t * 2.0, 1.0, t > 0.5);
    return vec3f(g, g, 0.3 + t * 0.7);
}

@vertex
fn vs_particle(@builtin(vertex_index) vi : u32,
               @builtin(instance_index) ii : u32) -> VSOut {
    var quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    let corner = quad[vi];
    let p = particles[ii];
    let st = stellar[ii];

    let origin = select(vec3f(0.0), particles[u.selectedIdx].pos, u.selectedIdx != 0xffffffffu);
    var clip = u.viewProj * vec4f(p.pos - origin, 1.0);

    // Photometry from the stellar lifecycle (or BH / gas special cases)
    var temp : f32;
    var lum : f32;
    if (st.x > -0.5 && st.x <= 0.0) {
        temp = 9000.0;          // central black hole(s): compact hot glow
        lum = 60.0;
    } else if (st.x < -0.5) {
        temp = 10000.0;         // cold gas: barely-there blue shimmer —
        lum = 0.0006;           // its real visual role is the dust it carries
    } else {
        let ageMyr = max(u.time * u.agingRate - st.y, 0.0);
        let s = starState(st.x, ageMyr, st.z);
        temp = s.temp;
        lum = s.lum;
    }

    // Perceptual luminosity compression: full L spans ~1e-4..1e4+, which
    // additive blending can't take raw. L^0.55 keeps giants dominant
    // without nuking the frame; AgX rolls off the rest.
    let intensity = pow(max(lum, 1e-5), 0.55) * u.starIntensity;

    // Sprite size grows slowly with brightness; supernovae get big.
    // Beyond d0 the sprite shrinks as 1/d (far-field look). Inside d0 a star
    // stays a point: the sprite grows only as 1/√d up to ~7 px, and the area
    // it no longer covers goes into brightness (total flux stays 1/d²),
    // capped so AgX and bloom carry the rest.
    let d0 = 60.0;
    let size = clamp(1.1 + 1.05 * log(1.0 + intensity), 1.4, 10.0);
    let near = max(d0 / max(clip.w, 0.1), 1.0);
    let base = size * 150.0 / max(clip.w, d0);
    let sizePx = max(min(base * sqrt(near), max(base, 7.0)), 1.25) * u.sizeScale;
    let areaRatio = base * near * u.sizeScale / sizePx;
    let flux = clamp(areaRatio * areaRatio, 1.0, 12.0);

    // Giants, supernovae and O stars get diffraction spikes on a larger quad.
    let spike = select(1.0, clamp(96.0 * u.sizeScale / sizePx, 1.0, 4.0), lum > 2000.0);

    // Screen-aligned billboard: offset corners in NDC, scaled back by w.
    clip = vec4f(clip.xy + corner * spike * (sizePx / u.resolution) * clip.w, clip.zw);

    var out : VSOut;
    out.pos = clip;
    out.quadUV = corner;
    out.spike = spike;
    out.flux = flux;

    if (u.colorMode == 1u) {
        out.color = velocityColor(length(p.vel)) * max(intensity, 0.4);
    } else if (u.colorMode == 2u) {
        out.color = massColor(max(st.x, 0.05)) * max(intensity, 0.4);
    } else {
        out.color = blackbodyRGB(temp) * intensity;
    }

    if (ii == u.selectedIdx) {
        out.color *= (1.0 - u.systemFade);   // the system's sun takes over
    }
    return out;
}

@fragment
fn fs_particle(in : VSOut) -> @location(0) vec4f {
    // Point-spread: Gaussian core + wide dim halo (~9 % of the energy), same
    // total energy as the old single Gaussian. Luminosity is in the color.
    // Near-field flux boosts the core; the halo gets only √flux so close
    // neighbours don't wash the frame.
    let p = in.quadUV * in.spike;
    let px = fwidth(p.x);
    let d2 = dot(p, p);
    if (in.spike == 1.0 && d2 > 1.0) { discard; }
    var f = in.flux * 1.15 * exp(-d2 * 4.5) + sqrt(in.flux) * 0.043 * exp(-d2 * 1.2);
    if (in.spike > 1.0) {
        // 4 spikes: narrow Gaussian across, exponential falloff along
        let a = abs(p) / in.spike;
        let along = (1.0 - a) * exp(-3.0 * a);
        let w = max(0.08, 0.6 * px);
        f += in.flux * 0.15 * (along.x * exp(-p.y * p.y / (w * w)) + along.y * exp(-p.x * p.x / (w * w)));
    }
    return vec4f(in.color * f, 1.0);
}

// ── Dust extinction sprites ─────────────────────────────────────────────────
// Disk/arm particles near the midplane carry dust (stellar.w). Rendered as
// large WORLD-SPACE soft sprites with multiplicative blending
// (dst *= 1 − srcRGB): wavelength-dependent extinction — blue absorbed more
// than red — gives the brown lanes real galaxies have.
struct DustOut {
    @builtin(position) pos : vec4f,
    @location(0) quadUV : vec2f,
    @location(1) strength : f32,
}

fn collapse() -> vec4f { return vec4f(0.0, 0.0, 2.0, 1.0); }  // clipped away

// A world-space sprite that would cover over ~half the screen height
// dissolves instead of blanking the view.
fn nearFade(sizePx : f32) -> f32 {
    return 1.0 - smoothstep(0.25, 0.6, sizePx / u.resolution.y);
}

@vertex
fn vs_dust(@builtin(vertex_index) vi : u32,
           @builtin(instance_index) ii : u32) -> DustOut {
    var quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    var out : DustOut;
    let st = stellar[ii];
    // Dust lives in the GAS (stellar.x = -1); stars carry none.
    if (st.x > -0.5 || st.w < 0.1 || u.dustStrength <= 0.0) {
        out.pos = collapse();
        out.quadUV = vec2f(0.0);
        out.strength = 0.0;
        return out;
    }
    let p = particles[ii];
    let origin = select(vec3f(0.0), particles[u.selectedIdx].pos, u.selectedIdx != 0xffffffffu);
    var clip = u.viewProj * vec4f(p.pos - origin, 1.0);
    // world-space size: fat fuzzy blobs that overlap into lanes
    let worldSize = 2.2 + st.z * 6.0;
    let projScale = u.resolution.y / (2.0 * u.tanHalfFov);
    let sizePx = worldSize * projScale / max(clip.w, 0.1);
    let fade = nearFade(sizePx);
    let corner = quad[vi];
    clip = vec4f(clip.xy + corner * (sizePx / u.resolution) * clip.w, clip.zw);
    out.pos = select(clip, collapse(), fade <= 0.0);
    out.quadUV = corner;
    out.strength = st.w * u.dustStrength * fade;
    return out;
}

@fragment
fn fs_dust(in : DustOut) -> @location(0) vec4f {
    let d2 = dot(in.quadUV, in.quadUV);
    if (d2 > 1.0) { discard; }
    let a = exp(-d2 * 3.0) * in.strength * 0.16;
    // extinction per channel: blue suffers most → lanes go warm brown
    return vec4f(a * 0.55, a * 0.8, a * 1.0, 1.0);
}

// ── H-II emission nebulae ───────────────────────────────────────────────────
// Hot young O stars ionize their surroundings: a soft pink Hα halo,
// rendered additively at world scale.
@vertex
fn vs_hii(@builtin(vertex_index) vi : u32,
          @builtin(instance_index) ii : u32) -> DustOut {
    var quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    var out : DustOut;
    let st = stellar[ii];
    var ok = false;
    if (st.x >= 12.0 && u.hiiStrength > 0.0 && ii != u.centralIdx) {
        // only while the star is actually on the (possibly recycled) MS
        let ageMyr = max(u.time * u.agingRate - st.y, 0.0);
        let s = starState(st.x, ageMyr, st.z);
        ok = s.lum > 100.0 && s.temp > 15000.0;
    }
    if (!ok) {
        out.pos = collapse();
        out.quadUV = vec2f(0.0);
        out.strength = 0.0;
        return out;
    }
    let p = particles[ii];
    let origin = select(vec3f(0.0), particles[u.selectedIdx].pos, u.selectedIdx != 0xffffffffu);
    var clip = u.viewProj * vec4f(p.pos - origin, 1.0);
    let worldSize = 3.0 + st.z * 4.0;
    let projScale = u.resolution.y / (2.0 * u.tanHalfFov);
    let sizePx = worldSize * projScale / max(clip.w, 0.1);
    let fade = nearFade(sizePx);
    let corner = quad[vi];
    clip = vec4f(clip.xy + corner * (sizePx / u.resolution) * clip.w, clip.zw);
    out.pos = select(clip, collapse(), fade <= 0.0);
    out.quadUV = corner;
    out.strength = u.hiiStrength * fade;
    return out;
}

@fragment
fn fs_hii(in : DustOut) -> @location(0) vec4f {
    let d2 = dot(in.quadUV, in.quadUV);
    if (d2 > 1.0) { discard; }
    let a = exp(-d2 * 4.0) * in.strength * 0.10;
    // Hα-dominated emission: saturated pink-red
    return vec4f(vec3f(1.0, 0.35, 0.48) * a, 1.0);
}

// ── Selection highlight ring ────────────────────────────────────────────────
@vertex
fn vs_highlight(@builtin(vertex_index) vi : u32) -> VSOut {
    var quad = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    );
    var out : VSOut;
    if (u.selectedIdx >= arrayLength(&particles)) {
        out.pos = collapse();
        out.quadUV = vec2f(0.0);
        out.color = vec3f(0.0);
        out.spike = 1.0;
        return out;
    }
    var clip = u.viewProj * vec4f(0.0, 0.0, 0.0, 1.0);
    let corner = quad[vi];
    let sizePx = 26.0 * u.sizeScale;
    clip = vec4f(clip.xy + corner * (sizePx / u.resolution) * clip.w, clip.zw);
    out.pos = clip;
    out.quadUV = corner;
    out.color = vec3f(0.6) * (1.0 - u.systemFade);
    out.spike = 1.0;
    return out;
}

@fragment
fn fs_highlight(in : VSOut) -> @location(0) vec4f {
    // 1.5 px hairline: box-filtered coverage from the pixel distance to r = 0.8
    let d = length(in.quadUV);
    let px = length(vec2f(dpdx(d), dpdy(d)));
    let ring = clamp(1.25 - abs(d - 0.8) / px, 0.0, 1.0);
    if (d > 1.0 || ring < 0.01) { discard; }
    return vec4f(in.color * ring, 1.0);
}

// ── Fullscreen triangle ─────────────────────────────────────────────────────
struct FSOut {
    @builtin(position) pos : vec4f,
    @location(0) uv : vec2f,   // [0,1], y-down
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi : u32) -> FSOut {
    var out : FSOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.pos = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f(x, -y) * 0.5 + 0.5;
    return out;
}

// ── Background starfield (procedural, view-direction hashed) ────────────────
fn hash31(p : vec3f) -> f32 {
    var q = fract(p * 443.8975);
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}
fn hash33(p : vec3f) -> vec3f {
    var q = fract(p * vec3f(443.897, 441.423, 437.195));
    q += dot(q, q.yxz + 19.19);
    return fract((q.xxy + q.yxx) * q.zyx);
}

fn starLayer(dir : vec3f, cells : f32, density : f32, brightness : f32, sharp : f32) -> vec3f {
    let g = dir * cells;
    let cell = floor(g);
    let h = hash33(cell);
    if (h.z > density) { return vec3f(0.0); }
    let starPos = cell + vec3f(0.15) + h * 0.7;
    let d = length(g - starPos);
    // Steep magnitude distribution: most stars dim, a rare few bright
    let mag = pow(hash31(cell + 7.77), 18.0) * brightness + 0.003;
    let core = exp(-d * d * sharp) * mag;
    // Subtle temperature variation: blue-white to warm
    let t = hash31(cell + 3.33);
    let col = mix(vec3f(0.65, 0.75, 1.0), vec3f(1.0, 0.85, 0.65), t);
    return col * core;
}

@fragment
fn fs_background(in : FSOut) -> @location(0) vec4f {
    let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let dir = normalize(
        u.camForward
        + u.camRight * ndc.x * u.tanHalfFov * u.aspect
        + u.camUp    * ndc.y * u.tanHalfFov
    );
    var col = starLayer(dir, 220.0, 0.10, 0.35, 380.0);
    col += starLayer(dir, 90.0, 0.05, 0.8, 900.0);
    return vec4f(col, 1.0);
}

// ── Bloom: 13-tap downsample (CoD: AW) ──────────────────────────────────────
@group(0) @binding(0) var srcTex  : texture_2d<f32>;
@group(0) @binding(1) var srcSamp : sampler;

fn luma(c : vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

fn downsample13(uv : vec2f, texel : vec2f, karis : bool) -> vec3f {
    let a = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f(-2.0,  2.0), 0.0).rgb;
    let b = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f( 0.0,  2.0), 0.0).rgb;
    let c = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f( 2.0,  2.0), 0.0).rgb;
    let d = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f(-2.0,  0.0), 0.0).rgb;
    let e = textureSampleLevel(srcTex, srcSamp, uv,                              0.0).rgb;
    let f = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f( 2.0,  0.0), 0.0).rgb;
    let g = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f(-2.0, -2.0), 0.0).rgb;
    let h = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f( 0.0, -2.0), 0.0).rgb;
    let i = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f( 2.0, -2.0), 0.0).rgb;
    let j = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f(-1.0,  1.0), 0.0).rgb;
    let k = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f( 1.0,  1.0), 0.0).rgb;
    let l = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f(-1.0, -1.0), 0.0).rgb;
    let m = textureSampleLevel(srcTex, srcSamp, uv + texel * vec2f( 1.0, -1.0), 0.0).rgb;

    if (karis) {
        // Firefly suppression: luminance-weighted groups (first downsample only)
        var g0 = (a + b + d + e) * 0.25;
        var g1 = (b + c + e + f) * 0.25;
        var g2 = (d + e + g + h) * 0.25;
        var g3 = (e + f + h + i) * 0.25;
        var g4 = (j + k + l + m) * 0.25;
        let w0 = 0.125 / (1.0 + luma(g0));
        let w1 = 0.125 / (1.0 + luma(g1));
        let w2 = 0.125 / (1.0 + luma(g2));
        let w3 = 0.125 / (1.0 + luma(g3));
        let w4 = 0.5   / (1.0 + luma(g4));
        return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4)
             / (w0 + w1 + w2 + w3 + w4);
    }
    return e * 0.125
         + (a + c + g + i) * 0.03125
         + (b + d + f + h) * 0.0625
         + (j + k + l + m) * 0.125;
}

@fragment
fn fs_downsample_first(in : FSOut) -> @location(0) vec4f {
    let texel = 1.0 / vec2f(textureDimensions(srcTex, 0));
    var c = downsample13(in.uv, texel, true);
    c = min(c, vec3f(64.0));   // clamp extreme HDR to keep bloom stable
    return vec4f(c, 1.0);
}

@fragment
fn fs_downsample(in : FSOut) -> @location(0) vec4f {
    let texel = 1.0 / vec2f(textureDimensions(srcTex, 0));
    return vec4f(downsample13(in.uv, texel, false), 1.0);
}

// ── Bloom: 3x3 tent upsample (rendered additively via blend state) ─────────
@fragment
fn fs_upsample(in : FSOut) -> @location(0) vec4f {
    let texel = 1.0 / vec2f(textureDimensions(srcTex, 0));
    let r = 1.0;
    var s = textureSampleLevel(srcTex, srcSamp, in.uv, 0.0).rgb * 4.0;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f(-r,  0.0), 0.0).rgb * 2.0;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f( r,  0.0), 0.0).rgb * 2.0;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f( 0.0, -r), 0.0).rgb * 2.0;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f( 0.0,  r), 0.0).rgb * 2.0;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f(-r,  -r), 0.0).rgb;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f( r,  -r), 0.0).rgb;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f(-r,   r), 0.0).rgb;
    s += textureSampleLevel(srcTex, srcSamp, in.uv + texel * vec2f( r,   r), 0.0).rgb;
    return vec4f(s / 16.0, 1.0);
}

// ── Composite: scene + bloom → exposure → AgX → sRGB ────────────────────────
struct CompositeParams {
    exposure      : f32,   // linear multiplier (2^EV)
    bloomStrength : f32,
    _c0 : f32,
    _c1 : f32,
}

@group(0) @binding(0) var sceneTex  : texture_2d<f32>;
@group(0) @binding(1) var bloomTex  : texture_2d<f32>;
@group(0) @binding(2) var compSamp  : sampler;
@group(0) @binding(3) var<uniform> comp : CompositeParams;

// AgX tone mapping — ported from three.js (Filament-derived matrices,
// Blender-matched contrast curve).
fn agxContrast(x : vec3f) -> vec3f {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    return -17.86 * x6 * x + 78.01 * x6 - 126.7 * x4 * x + 92.06 * x4
           - 28.72 * x2 * x + 4.361 * x2 - 0.1718 * x + vec3f(0.002857);
}

fn agx(colorIn : vec3f) -> vec3f {
    let AgXInset = mat3x3f(
        vec3f(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3f(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3f(0.0482516061458583, 0.101439036467562, 0.811302368396859),
    );
    let AgXOutset = mat3x3f(
        vec3f(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3f(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        vec3f(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
    );
    let srgbToRec2020 = mat3x3f(
        vec3f(0.6274, 0.0691, 0.0164),
        vec3f(0.3293, 0.9195, 0.0880),
        vec3f(0.0433, 0.0114, 0.8956),
    );
    let rec2020ToSrgb = mat3x3f(
        vec3f(1.6605, -0.1246, -0.0182),
        vec3f(-0.5876, 1.1329, -0.1006),
        vec3f(-0.0728, -0.0083, 1.1187),
    );
    const AgxMinEv = -12.47393;
    const AgxMaxEv = 4.026069;

    var color = srgbToRec2020 * colorIn;
    color = AgXInset * color;
    color = max(color, vec3f(1e-10));
    color = log2(color);
    color = (color - vec3f(AgxMinEv)) / (AgxMaxEv - AgxMinEv);
    color = clamp(color, vec3f(0.0), vec3f(1.0));
    color = agxContrast(color);
    color = AgXOutset * color;
    color = pow(max(color, vec3f(0.0)), vec3f(2.2));  // back to linear
    color = rec2020ToSrgb * color;
    return clamp(color, vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c : vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs_composite(in : FSOut) -> @location(0) vec4f {
    let scene = textureSampleLevel(sceneTex, compSamp, in.uv, 0.0).rgb;
    let bloom = textureSampleLevel(bloomTex, compSamp, in.uv, 0.0).rgb;
    var hdr = (scene + bloom * comp.bloomStrength) * comp.exposure;
    // ±1/255 triangular-PDF dither breaks 8-bit banding in dark bloom gradients:
    // interleaved-gradient noise remapped from uniform to triangular.
    let n = fract(52.9829189 * fract(dot(in.pos.xy, vec2f(0.06711056, 0.00583715)))) * 2.0 - 1.0;
    let tri = sign(n) * (1.0 - sqrt(1.0 - abs(n)));
    return vec4f(linearToSrgb(agx(hdr)) + tri / 255.0, 1.0);
}
