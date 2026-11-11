// --- the roof surface ---------------------------------------------------------
// What a flat roof is covered in, building by building: black membrane
// (bitumen, EPDM), grey gravel ballast, or a white coating (the "cool
// roof"), the share of each about what an aerial of a city shows. Membrane
// goes down in sheets, a tone apart, with a darker lapped seam between: the
// pattern's mean is exactly 1, so past a few pixels the roof is its own
// albedo and nothing else. A white roof is what makes a roof legible from
// the air at night: it takes the city's glow.
const CITY_ROOF_BLACK: f32 = 0.070;
const CITY_ROOF_GRAVEL: f32 = 0.150;
const CITY_ROOF_WHITE: f32 = 0.420;
const CITY_ROOF_SEAM_W: f32 = 0.06;
const CITY_ROOF_SEAM_K: f32 = 0.55;       // a seam's albedo against its sheet
fn city_roof_albedo(cc: CityCell, h: CityHit, fp: f32) -> f32 {
    if (cc.land_use != CITY_USE_URBAN) {
        return CITY_ROOF_ALBEDO;
    }
    let r = city_rand4(cc.seed ^ vec2<u32>(0x4c957f2du, 0x2c1b3c6du));
    var base = CITY_ROOF_BLACK;
    if (r.x > 0.75) {
        base = CITY_ROOF_WHITE;
    } else if (r.x > 0.45) {
        return CITY_ROOF_GRAVEL;   // ballast: no sheets to see
    }
    // Sheets 1-3 m wide along the roof's long axis, each its own tone in
    // [0.88, 1.12], seams darker; normalised so the mean over a sheet's
    // pitch is 1.
    let w = 1.0 + 2.0 * r.y;
    let along_x = (cc.b1max.x - cc.b1min.x) > (cc.b1max.y - cc.b1min.y);
    let across = select(h.pos.x, h.pos.y, along_x) / w;
    let i = floor(across);
    let tone = 0.88 + 0.24 * city_u01(pcg2d(vec2<u32>(
        cc.seed.x ^ (bitcast<u32>(i32(i)) * 0x9e3779b9u), cc.seed.y)).x);
    let fs = CITY_ROOF_SEAM_W / w;
    let seam = select(1.0, CITY_ROOF_SEAM_K, fract(across) < fs);
    let norm = 1.0 / (1.0 - fs * (1.0 - CITY_ROOF_SEAM_K));
    let pattern = tone * seam * norm;
    // The tones average to 1 over sheets and the seam over a pitch, so the
    // dissolve is to 1 itself.
    return base * mix(pattern, 1.0, smoothstep(0.08, 0.5 * w, fp));
}

// Surface radiance at a city hit, before fog. fp is the pixel footprint at
// the hit (m/px): the analytic LOD that dissolves windows and lamp pools
// into their means once they are sub-pixel, so the far city is smooth
// glow rather than shimmer.
// fp is the footprint the city's LOD runs on; fp_filter the true pixel's
// (or a reflection lobe's), which a surface's texture is filtered over;
// direct false for a surface seen in a reflection.
fn city_shade(h: CityHit, dir: vec3<f32>, fp: f32, fp_filter: f32,
              direct: bool) -> vec3<f32> {
    let moon = u.sun_dir;
    let cc = city_cell(h.cell);
    let local_glow = city_glow_sample(h.pos.xy, 3.0);
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * local_glow);

    if (h.kind >= 100) {   // a component's hit; its shader owns the look
        return cc_component_shade(h, cc, dir, fp);
    }
    if (h.kind == 4) {   // beacon
        return CITY_BEACON_COLOR;
    }
    if (h.kind == 3) {   // mast steel
        return 0.05 * fill
               + 0.04 * CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0);
    }
    if (h.kind == 2) {   // roof
        var alb = city_roof_albedo(cc, h, fp_filter);
        // Parapet ring: the roof edge catches the street glow from below.
        // The ring belongs to whichever tier this roof is — pick the box
        // whose top the hit sits on.
        var rmin = cc.b1min.xy;
        var rmax = cc.b1max.xy;
        if (cc.tiers >= 2 && h.pos.z > cc.b2min.z + 0.5) {
            rmin = cc.b2min.xy;
            rmax = cc.b2max.xy;
        }
        if (cc.tiers >= 3 && h.pos.z > cc.b3min.z + 0.5) {
            rmin = cc.b3min.xy;
            rmax = cc.b3max.xy;
        }
        // A roof above the summit is a crown's: no ring on it (the draw
        // that would say where its edge is costs more than the ring).
        if (h.pos.z > cc.height + 0.5) {
            rmin = vec2<f32>(-1e30);
            rmax = vec2<f32>(1e30);
        }
        let din = min(
            min(h.pos.x - rmin.x, rmax.x - h.pos.x),
            min(h.pos.y - rmin.y, rmax.y - h.pos.y)
        );
        if (din < 1.2) {
            alb = alb * 2.2;
        }
        let base = alb * (fill + CITY_MOONLIGHT * max(moon.z, 0.0));
        return base + cc_roof_detail(cc, h, fp, fp_filter, dir, direct, base);
    }
    if (h.kind == 1) {   // facade, flat: past the shell's reach
        let f = city_face_at_mass(cc, h.pos, h.normal);
        return city_facade_flat(cc, f, h, dir, fp, fp_filter, direct, fill);
    }

    // Ground: the street surface (see THE STREET SURFACE above).
    return city_ground_radiance(h.pos, h.cell, cc, dir, fp);
}

// Night ground-fog: extinction toward the hit plus in-scattered city glow.
// The glow source is the mip under the midpoint of the path, coarsened with
// distance — the fog over downtown is orange, the fog over the outskirts is
// barely there.
// The fog's transmittance alone: what a reflection seen THROUGH the
// primary path's fog is multiplied by (city_fog is affine in radiance).
fn city_fog_trans(o: vec3<f32>, dir: vec3<f32>, t_hit: f32,
                  hit_pos: vec3<f32>) -> f32 {
    let z0 = max(o.z, 0.0);
    let z1 = max(hit_pos.z, 0.0);
    let mu = dir.z;
    var tau: f32;
    if (abs(mu) > 1e-6) {
        tau = CITY_FOG_BETA * (CITY_FOG_H / mu)
              * (exp(-z0 / CITY_FOG_H) - exp(-z1 / CITY_FOG_H));
    } else {
        tau = CITY_FOG_BETA * t_hit * exp(-z0 / CITY_FOG_H);
    }
    return exp(-max(tau, 0.0));
}

fn city_fog(radiance: vec3<f32>, o: vec3<f32>, dir: vec3<f32>,
            t_hit: f32, hit_pos: vec3<f32>) -> vec3<f32> {
    let ab = city_fog_ab(o, dir, t_hit, hit_pos);
    return radiance * ab.x + ab.yzw;
}

// The fog as the affine map it is on the surface's radiance: fogged =
// radiance * x + yzw (transmittance, in-scatter). The flat facade needs it
// to average what the eye is shown rather than what the wall emits.
fn city_fog_ab(o: vec3<f32>, dir: vec3<f32>, t_hit: f32,
               hit_pos: vec3<f32>) -> vec4<f32> {
    let ftrans = city_fog_trans(o, dir, t_hit, hit_pos);
    let mid = mix(o.xy, hit_pos.xy, 0.6);
    // @domain: t_hit >= 0
    let lod = 5.0 + log2(1.0 + t_hit / 2000.0);
    // Clamped: over the megatower district the raw mip runs to ~3x the
    // p99.5 level, and unclamped it bleaches the whole frame sepia.
    let glow = min(city_glow_sample(mid, lod), 1.2);
    let in_scatter = (1.0 - ftrans)
        * (CITY_FOG_GLOW * (CITY_FOG_GLOW_AMP * glow) + CITY_FOG_BASE);
    return vec4<f32>(ftrans, in_scatter);
}

// Downward twin of sky_probe_tau: the optical depth between a
// sample and the city below it. The caller applies its own decay — NOT the
// two-stream diffuse transmittance, whose heavy tail (T(20) ~ 0.3) let the
// city glow soak whole cloud masses ochre; the amber belongs to the base
// skin, so the uplight uses an exponential in this tau instead.
fn city_uplight_probe_tau(p: vec3<f32>, jit: f32) -> f32 {
    let span = min(CITY_UP_PROBE_SPAN, max(p.z - u.bmin.z, 0.0));
    if (span <= 0.0) {
        return 0.0;
    }
    let dz = span / f32(CITY_UP_PROBE_STEPS);
    var tau = 0.0;
    for (var k: i32 = 0; k < CITY_UP_PROBE_STEPS; k = k + 1) {
        let z_off = (f32(k) + 0.5 + jit) * dz;
        let q = vec3<f32>(p.x, p.y, p.z - z_off);
        tau = tau + sample_sigma(q) * dz;
    }
    return tau;
}

// The night sky: gradient, stars, the city's own glow dome at the skyline,
// and a crescent moon in the day scene's sun direction (u.sun_dir).
fn night_sky_radiance(dir: vec3<f32>, moon: vec3<f32>) -> vec3<f32> {
    let zc = clamp(dir.z, 0.0, 1.0);
    var col = mix(CITY_NIGHT_HORIZON, CITY_NIGHT_ZENITH, pow(zc, 0.55));

    // Skyline glow dome: sample the city ahead of the sightline.
    // Into the city frame, through the one entry the trace itself uses.
    let probe_xy = city_camera_origin().xy + dir.xy * CITY_SKYGLOW_DOME_DIST;
    let dome = city_glow_sample(probe_xy, 7.0);
    col = col + CITY_SKYGLOW_DOME * (CITY_SKYGLOW_DOME_AMP * dome)
                * exp(-max(dir.z, 0.0) / CITY_SKYGLOW_DOME_SCALE);

    // Stars: hashed cells on the dominant-axis cube face, faded into the
    // horizon haze and small enough that subpixel jitter antialiases them.
    let ad = abs(dir);
    var uv: vec2<f32>;
    var face: i32;
    if (ad.z >= ad.x && ad.z >= ad.y) {
        uv = dir.xy / ad.z;
        face = 0;
    } else if (ad.x >= ad.y) {
        uv = dir.yz / ad.x;
        face = 1;
    } else {
        uv = dir.xz / ad.y;
        face = 2;
    }
    let sc = uv * CITY_STAR_GRID;
    let cid = vec2<i32>(floor(sc));
    let sh = city_rand4(pcg2d(vec2<u32>(
        bitcast<u32>(cid.x * 3 + face),
        bitcast<u32>(cid.y)
    )));
    if (sh.x < CITY_STAR_FRAC) {
        let spos = vec2<f32>(cid) + vec2<f32>(0.15) + 0.70 * sh.yz;
        let d = length(sc - spos);
        if (d < CITY_STAR_RADIUS) {
            // @domain: a hash draw, in [0, 1)
            let b = pow(sh.w, 4.0) * (1.0 - d / CITY_STAR_RADIUS);
            col = col + vec3<f32>(0.9, 0.95, 1.0)
                * (CITY_STAR_AMP * b * smoothstep(0.05, 0.25, dir.z));
        }
    }

    // Moon: halo, tight bloom, then the crescent disc itself.
    let cm = dot(dir, moon);
    let g_h = CITY_MOON_HALO_G;
    let dh = 1.0 + g_h * g_h - 2.0 * g_h * cm;
    let one_minus_gh = 1.0 - g_h;
    let halo = (one_minus_gh * one_minus_gh * one_minus_gh)
        // @domain: d = 1 + g^2 - 2 g cos >= (1 - |g|)^2 >= 0 for |cos| <= 1 (Henyey-Greenstein)
        / max(pow(dh, 1.5), 1e-6);
    col = col + CITY_MOON_HALO_AMP * halo * vec3<f32>(0.75, 0.80, 0.95);
    if (cm > 0.0) {
        let a = CITY_MOON_BLOOM_W / ((1.0 - cm) + CITY_MOON_BLOOM_W);
        col = col + a * u.sky_bloom;
    }
    let off = dir - moon * cm;
    let sin_r = length(off);
    if (cm > 0.0 && sin_r < CITY_MOON_SIN_R) {
        let r = sin_r / CITY_MOON_SIN_R;
        // The frame's helper axis swaps for x near the zenith, where
        // z x moon would vanish (as the sun's frame does).
        let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0),
                        abs(moon.z) > 0.9);
        // @domain: up is swapped for x when |moon.z| > 0.9: never parallel to moon
        let t1 = normalize(cross(up, moon));
        let t2 = cross(moon, t1);
        let a1 = dot(off, t1) / CITY_MOON_SIN_R;
        let a2 = dot(off, t2) / CITY_MOON_SIN_R;
        let c = sqrt(max(1.0 - r * r, 0.0));
        let n = a1 * t1 + a2 * t2 - c * moon;
        // @domain: a combination of an orthonormal frame with a nonzero coefficient (t1, t2, moon)
        let l = normalize(0.82 * t1 - 0.20 * t2 + 0.55 * moon);
        let lit = smoothstep(
            0.0, CITY_MOON_TERMINATOR_SOFT, dot(n, l));
        let edge = smoothstep(1.0, 0.97, r);
        col = col + CITY_MOON_DISC * edge
              * (lit + CITY_MOON_EARTHSHINE);
    }
    return col;
}
