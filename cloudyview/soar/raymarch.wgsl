// CloudyView interactive volume raymarcher (WGSL).
//
// Stage 4 scope (2026-07): ray-box entry, hardware-trilinear sampling of a
// resident r32float 3D extinction texture, witness procedural sky, per-pixel
// jittered ray starts, and the single-domain witness cloud-scattering model
// (adaptive view/light marches, dt-invariant powder, MS octaves, ambient ramp,
// surface bounce, ghost-zero boundary taper, and the witness FIF ocean). Nested levels are still staged follow-ups,
// ported function by function against the numba golden reference
// (docs/architecture.md).
//
// Conventions shared with the CPU renderer (cloudyview/witness.py):
// - World space is absolute meters, +x east, +y north, +z up.
// - The volume AABB is cell-edge aligned (half-cell padding around centers).
// - Vertical FOV; pixel row 0 is the top of the image.
// - Tone map: Reinhard + gamma 1.4 (witness.tone_map), applied in-shader.
//
// Texture layout note: the density texture is uploaded with
// width=nz, height=ny, depth=nx so the (nx,ny,nz) C-order numpy array is
// uploaded with zero host-side reshuffling. Normalized sample coords are
// therefore (gz, gy, gx) — see sample_sigma().

struct Uniforms {
    // xyz = camera origin (m), w = tan(fov_vertical / 2)
    cam_origin: vec4<f32>,
    // xyz = forward, w = aspect (width / height)
    cam_forward: vec4<f32>,
    // xyz = right, w = exposure (Reinhard pre-scale)
    cam_right: vec4<f32>,
    // xyz = up, w = jitter enable (0.0 or 1.0)
    cam_up: vec4<f32>,
    // xyz = unit direction toward the sun, w = frame index (jitter decorrelation)
    sun_dir: vec4<f32>,
    // xyz = volume AABB min (m), w = view-ray step dt (m)
    bmin: vec4<f32>,
    // xyz = volume AABB max (m), w = light-march step dt (m)
    bmax: vec4<f32>,
    // x = image width (px), y = image height (px), z = HG asymmetry g, w = ambient strength
    params: vec4<f32>,
    // x = ocean z (m), yzw = ocean reflectance (witness.py:104-106)
    ocean: vec4<f32>,
    // x = FIF dx (m), y = FIF tile extent (m), z = ocean enabled, w = max normal LOD
    ocean_params: vec4<f32>,
    // x = subpixel camera-ray jitter enable (0.0 or 1.0),
    // y = jitter amplitude scale, zw = unused
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;
@group(0) @binding(3) var ocean_normals: texture_2d<f32>;
@group(0) @binding(4) var ocean_samp: sampler;

// ---------------------------------------------------------------------------
// Constants (witness.py values where the concept carries over)
// ---------------------------------------------------------------------------

const SUN_COLOR: vec3<f32> = vec3<f32>(22.0, 21.0, 17.0); // witness.py:66
const POWDER_COEFF: f32 = 1.5;       // witness.py:63
const AMBIENT_TINT: vec3<f32> = vec3<f32>(0.22, 0.23, 0.28); // witness.py:82-84
const AMBIENT_HEIGHT_FLOOR: f32 = 0.3; // witness.py:85
const BOUNCE_STRENGTH: f32 = 0.05;   // witness.py:95
const BOUNCE_TINT: vec3<f32> = vec3<f32>(1.0, 0.97, 0.92); // witness.py:96-98
const MS_OCTAVES: i32 = 6;           // witness.py:76
const MS_ATTEN: f32 = 0.4;           // witness.py:77
const MS_BLEND_RATE: f32 = 0.35;     // witness.py:78
const MAX_VIEW_STEPS: i32 = 2048;   // witness.py:102
const MAX_LIGHT_STEPS: i32 = 512;   // witness.py:72
const TRANSMITTANCE_CUTOFF: f32 = 0.002; // witness early-exit
const LIGHT_TAU_CUTOFF: f32 = 80.0; // witness.py:363
const EMPTY_DTAU_CUTOFF: f32 = 1e-5; // witness.py:701
const DENSE_SIGMA_CUTOFF: f32 = 0.01; // witness.py:688
const TAU_STEP_MAX: f32 = 0.5;       // witness.py:689
const GAMMA: f32 = 1.4;             // witness tone_map gamma
const ISO_PHASE: f32 = 0.07957747154594767; // 1 / (4*pi)
const OCEAN_SHADOW_FLOOR: f32 = 0.35; // witness.py:539

// TODO(occupancy-grid): empty-space skipping. Cloud fields are sparse; a
// coarse (e.g. 16^3-voxel-block) occupancy grid bound after the ocean slots would
// let both the view march and the light march leap over empty bricks. This
// is the known next lever for the full 1024x1024x255 domain
// (docs/architecture.md "Interactive techniques").

// TODO(fp16): store density as a filterable 16-bit format to halve resident
// memory on large domains. r32float + float32-filterable works everywhere we
// care about today, so fp16 is an optimization, not a requirement.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Slab-method ray/AABB intersection. Returns (t_near, t_far); miss when
// t_near > t_far.
fn ray_box(origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    let t0 = (u.bmin.xyz - origin) * inv_dir;
    let t1 = (u.bmax.xyz - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

fn sigma_data_dims_xyz() -> vec3<f32> {
    let tex_dims = vec3<f32>(textureDimensions(vol, 0));
    return vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x) - vec3<f32>(2.0);
}

// Extinction (m^-1) at world point p via hardware trilinear filtering.
// Coordinate swizzle: texture is (w=nz+2, h=ny+2, d=nx+2), see file header.
// The host uploads original data into padded texels [1..N]. Witness uses
// gx=(p-bmin)/dx with data value i at gx=i and ghost zeros at -1 and N.
// Therefore padded_texel = gx+1 and normalized coord = (gx+1.5)/(N+2),
// preserving every pre-padding world/data sample while hardware filtering
// supplies the linear taper against the zero border. The march interval stays
// on the original outer AABB, matching witness._active_level's containment
// test before it calls _sample_sigma_level.
fn sample_sigma(p: vec3<f32>) -> f32 {
    let tex_dims = vec3<f32>(textureDimensions(vol, 0));
    let data_g = ((p - u.bmin.xyz) / (u.bmax.xyz - u.bmin.xyz))
                 * sigma_data_dims_xyz();
    let tex_coord = vec3<f32>(
        data_g.z + 1.5,
        data_g.y + 1.5,
        data_g.x + 1.5
    ) / tex_dims;
    return textureSampleLevel(vol, vol_samp, tex_coord, 0.0).r;
}

// Henyey-Greenstein phase function.
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (12.566370614 * pow(max(denom, 1e-6), 1.5));
}

// Per-pixel hash for jittered ray starts (Dave Hoskins style hash12).
// Blue-noise would be better; a white-noise hash is enough to break the
// coherent step-shell banding this exists to kill.
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        hash12(p + vec2<f32>(17.17, 41.93)),
        hash12(p + vec2<f32>(71.31, 11.57))
    );
}

fn step_dt_for_sigma(sigma: f32, dt_max: f32) -> f32 {
    if (sigma > DENSE_SIGMA_CUTOFF) {
        return min(dt_max, TAU_STEP_MAX / sigma);
    }
    return dt_max;
}

// Sun optical depth from p toward the sun: witness adaptive-step shadow march
// to the box exit with tau saturation (witness.py:299-367).
fn light_march_tau(p: vec3<f32>, sun: vec3<f32>) -> f32 {
    let inv_dir = 1.0 / sun;
    let hit = ray_box(p, inv_dir);
    if (hit.x > hit.y || hit.y <= 0.0) {
        return 0.0;
    }
    let t_exit = hit.y;
    let dt_max = u.bmax.w;
    var tau = 0.0;
    var t = max(hit.x, 0.0);
    for (var i: i32 = 0; i < MAX_LIGHT_STEPS; i = i + 1) {
        if (t >= t_exit) {
            break;
        }
        let sigma = sample_sigma(p + t * sun);
        var dt = dt_max;
        if (t + dt > t_exit) {
            dt = t_exit - t;
        }
        tau = tau + sigma * dt;
        if (tau > LIGHT_TAU_CUTOFF) {
            break;
        }
        t = t + dt;
    }
    return tau;
}

// Procedural sky ported from witness._sky_radiance (witness.py lines 371-413).
fn sky_color(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    var t = max(0.0, dir.z);
    let one_minus = 1.0 - t;
    t = 1.0 - one_minus * one_minus * one_minus;

    let zenith = vec3<f32>(0.0044, 0.035, 0.1156);
    let horizon = vec3<f32>(0.10, 0.18, 0.38);
    var col = horizon + (zenith - horizon) * t;

    let cos_sun = dot(dir, sun);
    if (cos_sun > 0.0) {
        let sun_half_width = 0.002;
        let a = sun_half_width / ((1.0 - cos_sun) + sun_half_width);
        col = col + a * vec3<f32>(0.8, 0.6, 0.3);
    }
    if (cos_sun > 0.9998) {
        col = col + vec3<f32>(50.0, 45.0, 35.0);
    }
    return col;
}

// Ocean reflection sky ported from witness._reflection_sky
// (witness.py lines 445-470).
fn ocean_reflection_sky(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    var t = max(0.0, dir.z);
    let one_minus = 1.0 - t;
    t = 1.0 - one_minus * one_minus * one_minus;

    let zenith = vec3<f32>(0.0044, 0.035, 0.1156);
    let horizon = vec3<f32>(0.10, 0.18, 0.38);
    var col = horizon + (zenith - horizon) * t;

    let cos_rs = dot(dir, sun);
    if (cos_rs > 0.0) {
        let glint_w = 0.02;
        let a = glint_w / ((1.0 - cos_rs) + glint_w);
        col = col + a * vec3<f32>(1.2, 1.0, 0.6);
    }
    return col;
}

// FIF normal sampling follows witness._ocean_wave_normal_fif
// (witness.py lines 417-442): periodic wrap, bilinear interpolation, then
// renormalization. The host uploads the generated nx/ny/nz arrays as RGB.
fn ocean_normal_lod(t_hit: f32, dir: vec3<f32>) -> f32 {
    let pixel_span = 2.0 * max(t_hit, 0.0) * u.cam_origin.w / u.params.y;
    let ocean_span = pixel_span / max(abs(dir.z), 0.03);
    let texel_span = max(ocean_span / u.ocean_params.x, 1.0);
    return clamp(log2(texel_span), 0.0, u.ocean_params.w);
}

fn ocean_wave_normal(world_xy: vec2<f32>, lod: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(ocean_normals, 0));
    let coord = world_xy / u.ocean_params.y + 0.5 / dims;
    let n = textureSampleLevel(ocean_normals, ocean_samp, coord, lod).rgb;
    return normalize(n);
}

// Ocean shade ported from witness._ocean_shade (witness.py lines 473-541).
fn ocean_shade(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>, t_hit: f32) -> vec3<f32> {
    var normal_lod = 0.0;
    if (u.cam_up.w > 0.5) {
        normal_lod = ocean_normal_lod(t_hit, dir);
    }
    let n = ocean_wave_normal(hit.xy, normal_lod);

    let vdotn = dot(dir, n);
    var refl = dir - 2.0 * vdotn * n;
    if (refl.z < 0.0) {
        refl.z = -refl.z;
    }
    let sky = ocean_reflection_sky(refl, sun);

    let cos_i = clamp(-vdotn, 0.0, 1.0);
    let one_minus = 1.0 - cos_i;
    let om2 = one_minus * one_minus;
    let fresnel = 0.02 + 0.98 * om2 * om2 * one_minus;

    let tau_ocean = light_march_tau(hit, sun);
    let t_sun_ocean = exp(-tau_ocean);
    let cos_sun_n = max(0.0, dot(sun, n));
    let diff_irr = t_sun_ocean * cos_sun_n * 0.3183098861837907;
    let diffuse = diff_irr * SUN_COLOR * u.ocean.yzw;

    let lit = fresnel * sky + (1.0 - fresnel) * diffuse;
    let t_eff = OCEAN_SHADOW_FLOOR + (1.0 - OCEAN_SHADOW_FLOOR) * t_sun_ocean;
    return lit * t_eff;
}

// Reinhard + gamma, matching radiative_transfer.tone_map (lines 675-680)
// with witness's default exposure=4.0 (witness.py lines 955-959, 1072-1073).
fn tone_map(hdr: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = hdr * exposure;
    let mapped = exposed / (1.0 + exposed);
    return pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / GAMMA));
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle.
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let img_w = u.params.x;
    let img_h = u.params.y;
    let g_hg = u.params.z;
    let ambient_strength = u.params.w;
    let tan_half_fov = u.cam_origin.w;
    let aspect = u.cam_forward.w;
    let exposure = u.cam_right.w;
    let jitter_on = u.cam_up.w;
    let subpixel_on = u.flags.x;
    let jitter_scale = clamp(u.flags.y, 0.0, 1.0);
    let sun = u.sun_dir.xyz;

    // Pixel -> camera ray. Framebuffer y=0 is the image top, matching the
    // witness convention ndc_y = 1 - 2*(py+0.5)/h.
    var sample_pos = frag_pos.xy;
    if (subpixel_on > 0.5) {
        let subpixel_seed = frag_pos.xy + vec2<f32>(
            u.sun_dir.w * 61.803,
            u.sun_dir.w * 17.271
        );
        sample_pos = sample_pos
                     + (hash22(subpixel_seed) - vec2<f32>(0.5)) * jitter_scale;
    }
    let ndc_x = (2.0 * sample_pos.x / img_w - 1.0) * aspect * tan_half_fov;
    let ndc_y = (1.0 - 2.0 * sample_pos.y / img_h) * tan_half_fov;
    let dir = normalize(u.cam_forward.xyz
                        + ndc_x * u.cam_right.xyz
                        + ndc_y * u.cam_up.xyz);

    let inv_dir = 1.0 / dir;
    let hit = ray_box(u.cam_origin.xyz, inv_dir);
    var t_near = max(hit.x, 0.0);
    let t_far = hit.y;

    let ocean_on = u.ocean_params.z > 0.5;
    var t_ocean = 1e30;
    if (ocean_on && dir.z < -1e-8) {
        let t_ocean_candidate = (u.ocean.x - u.cam_origin.z) / dir.z;
        if (t_ocean_candidate > 0.0) {
            t_ocean = t_ocean_candidate;
        }
    }

    let phase = hg_phase(dot(dir, sun), g_hg);

    // Jittered first step: decorrelates the sampling shells between
    // neighboring pixels, killing the coherent ring/banding artifact.
    // frame index is folded in so temporal accumulation stays unbiased.
    let jitter = hash12(frag_pos.xy + vec2<f32>(u.sun_dir.w * 61.803, 0.0));
    let dt_max = u.bmin.w;
    var t = t_near + jitter_on * jitter * jitter_scale * dt_max;

    var transmittance = 1.0;
    var col = vec3<f32>(0.0);
    var tau_depth = 0.0;

    if (t_near >= 0.0 && t_near < t_far) {
        for (var i: i32 = 0; i < MAX_VIEW_STEPS; i = i + 1) {
            // witness.py:621-646 tests ocean before the t_far break so an
            // ocean plane coincident with the box floor is still shaded.
            if (ocean_on && t >= t_ocean) {
                let ocean_hit = u.cam_origin.xyz + t_ocean * dir;
                col = col + transmittance * ocean_shade(ocean_hit, dir, sun, t_ocean);
                transmittance = 0.0;
                break;
            }

            if (t >= t_far || transmittance < TRANSMITTANCE_CUTOFF) {
                break;
            }
            let p = u.cam_origin.xyz + t * dir;
            let sigma = sample_sigma(p);

            var dt = step_dt_for_sigma(sigma, dt_max);
            if (t + dt > t_far) {
                dt = t_far - t;
            }
            if (ocean_on && t + dt > t_ocean) {
                dt = max(0.0001, t_ocean - t);
            }

            let d_tau = sigma * dt;
            if (d_tau < EMPTY_DTAU_CUTOFF) {
                tau_depth = 0.0;
                t = t + dt;
                continue;
            }

            tau_depth = tau_depth + d_tau;

            let tau_sun = light_march_tau(p, sun);

            var ms = vec3<f32>(0.0);
            var ms_atten = 1.0;
            for (var octave: i32 = 0; octave < MS_OCTAVES; octave = octave + 1) {
                let t_sun_ms = exp(-tau_sun * ms_atten);
                let blend = min(1.0, f32(octave) * MS_BLEND_RATE);
                let oct_phase = phase * (1.0 - blend) + ISO_PHASE * blend;
                let contrib = ms_atten * t_sun_ms * oct_phase;
                ms = ms + contrib * SUN_COLOR;
                ms_atten = ms_atten * MS_ATTEN;
            }

            // Powder is a function of cumulative optical depth since the current
            // cloud entry, not the current step size (witness.py:729-732).
            let powder = 1.0 - exp(-POWDER_COEFF * tau_depth);
            let scatter_weight = d_tau * powder * transmittance;
            col = col + scatter_weight * ms;

            // Ambient: height-based on the outer box (witness.py:738-747).
            let h = clamp((p.z - u.bmin.z) / (u.bmax.z - u.bmin.z), 0.0, 1.0);
            let amb = ambient_strength * (AMBIENT_HEIGHT_FLOOR
                                          + (1.0 - AMBIENT_HEIGHT_FLOOR) * h);
            col = col + transmittance * d_tau * amb * AMBIENT_TINT;

            // Surface bounce is anchored at physical z=0, not the AABB floor
            // (witness.py:749-760).
            if (BOUNCE_STRENGTH > 0.0) {
                let bounce_frac = clamp(1.0 - p.z / u.bmax.z, 0.0, 1.0);
                col = col + transmittance * d_tau * BOUNCE_STRENGTH
                            * bounce_frac * BOUNCE_TINT;
            }

            transmittance = transmittance * exp(-d_tau);
            t = t + dt;
        }
    }

    // Ocean for rays that exit/miss the outer box without becoming opaque;
    // witness.py:766-790 also limits far-open-water hits to 50 outer widths.
    if (ocean_on
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_ocean < 1e29
        && t_ocean > t_far) {
        let ocean_hit = u.cam_origin.xyz + t_ocean * dir;
        let outer_size = u.bmax.xyz - u.bmin.xyz;
        let center = 0.5 * (u.bmin.xy + u.bmax.xy);
        if (abs(ocean_hit.x - center.x) < outer_size.x * 50.0
            && abs(ocean_hit.y - center.y) < outer_size.y * 50.0) {
            col = col + transmittance * ocean_shade(ocean_hit, dir, sun, t_ocean);
            transmittance = 0.0;
        }
    }

    if (transmittance > TRANSMITTANCE_CUTOFF) {
        let sky = sky_color(dir, sun);
        col = col + transmittance * sky;
    }
    return vec4<f32>(tone_map(col, exposure), 1.0);
}
