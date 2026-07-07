// CloudyView interactive volume raymarcher (WGSL).
//
// Spike scope (2026-07): ray-box entry, hardware-trilinear sampling of a
// resident r32float 3D extinction texture, Beer-Lambert absorption with a
// single Henyey-Greenstein sun light-march, gradient sky, per-pixel jittered
// ray starts. This is NOT the witness look — the full port (FIF ocean,
// multi-scatter octaves, powder, nested levels) happens later, function by
// function against the numba golden reference (docs/architecture.md).
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
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;

// ---------------------------------------------------------------------------
// Constants (witness.py values where the concept carries over)
// ---------------------------------------------------------------------------

const SUN_COLOR: vec3<f32> = vec3<f32>(22.0, 21.0, 17.0); // HDR, slightly warm
const AMBIENT_TINT: vec3<f32> = vec3<f32>(0.75, 0.85, 1.05); // cool skylight
const MAX_VIEW_STEPS: i32 = 2048;   // witness MAX_STEPS
const MAX_LIGHT_STEPS: i32 = 512;   // witness n_light_steps
const TRANSMITTANCE_CUTOFF: f32 = 0.002; // witness early-exit
const LIGHT_TAU_CUTOFF: f32 = 7.0;  // exp(-7) ~ 1e-3: shadow fully dark
const SIGMA_SKIP: f32 = 1e-7;       // below this, skip the light march
const GAMMA: f32 = 1.4;             // witness tone_map gamma

// TODO(occupancy-grid): empty-space skipping. Cloud fields are sparse; a
// coarse (e.g. 16^3-voxel-block) occupancy grid bound at @binding(3) would
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

// Extinction (m^-1) at world point p via hardware trilinear filtering.
// Coordinate swizzle: texture is (w=nz, h=ny, d=nx), see file header.
// Clamp-to-edge sampling extends boundary values across the outer half
// voxel; the CPU reference instead tapers to zero through a ghost layer
// (witness._sample_sigma_level). TODO(ghost-zero): match the taper, either
// with a 1-voxel zero border baked at upload or explicit edge handling here.
fn sample_sigma(p: vec3<f32>) -> f32 {
    let g = (p - u.bmin.xyz) / (u.bmax.xyz - u.bmin.xyz);
    return textureSampleLevel(vol, vol_samp, vec3<f32>(g.z, g.y, g.x), 0.0).r;
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

// Sun transmittance from p toward the sun: short Beer-Lambert march to the
// box exit with early-out once the shadow saturates.
fn light_march(p: vec3<f32>, sun: vec3<f32>) -> f32 {
    let inv_dir = 1.0 / sun;
    let hit = ray_box(p, inv_dir);
    let t_exit = hit.y; // p is inside the box, so t_near < 0 < t_far
    let dt = u.bmax.w;
    var tau = 0.0;
    var t = 0.5 * dt;
    for (var i: i32 = 0; i < MAX_LIGHT_STEPS; i = i + 1) {
        if (t >= t_exit || tau >= LIGHT_TAU_CUTOFF) {
            break;
        }
        tau = tau + sample_sigma(p + t * sun) * dt;
        t = t + dt;
    }
    return exp(-tau);
}

// Simple gradient sky + sun disc. Placeholder for the witness sky port.
fn sky_color(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.z, -1.0, 1.0);
    var col: vec3<f32>;
    if (up >= 0.0) {
        let t = pow(1.0 - up, 4.0);
        col = mix(vec3<f32>(0.10, 0.28, 0.65), vec3<f32>(0.55, 0.68, 0.88), t);
    } else {
        // Below the horizon: dark sea-haze gradient (no ocean in the spike).
        let t = clamp(-up * 6.0, 0.0, 1.0);
        col = mix(vec3<f32>(0.55, 0.68, 0.88), vec3<f32>(0.020, 0.042, 0.075), t);
    }
    let cos_sun = clamp(dot(dir, sun), -1.0, 1.0);
    col = col + vec3<f32>(1.0, 0.95, 0.85) * pow(max(cos_sun, 0.0), 1200.0) * 30.0;
    col = col + vec3<f32>(0.28, 0.24, 0.17) * pow(max(cos_sun, 0.0), 8.0);
    return col;
}

// Reinhard + gamma, matching witness.tone_map(image, exposure, gamma=1.4).
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
    let sun = u.sun_dir.xyz;

    // Pixel -> camera ray. Framebuffer y=0 is the image top, matching the
    // witness convention ndc_y = 1 - 2*(py+0.5)/h.
    let ndc_x = (2.0 * frag_pos.x / img_w - 1.0) * aspect * tan_half_fov;
    let ndc_y = (1.0 - 2.0 * frag_pos.y / img_h) * tan_half_fov;
    let dir = normalize(u.cam_forward.xyz
                        + ndc_x * u.cam_right.xyz
                        + ndc_y * u.cam_up.xyz);

    let sky = sky_color(dir, sun);

    let inv_dir = 1.0 / dir;
    let hit = ray_box(u.cam_origin.xyz, inv_dir);
    var t_near = max(hit.x, 0.0);
    let t_far = hit.y;

    if (t_near >= t_far || t_far <= 0.0) {
        return vec4<f32>(tone_map(sky, exposure), 1.0);
    }

    let dt = u.bmin.w;
    let phase = hg_phase(dot(dir, sun), g_hg);

    // Jittered first step: decorrelates the sampling shells between
    // neighboring pixels, killing the coherent ring/banding artifact.
    // frame index is folded in so temporal accumulation stays unbiased.
    let jitter = hash12(frag_pos.xy + vec2<f32>(u.sun_dir.w * 61.803, 0.0));
    var t = t_near + jitter_on * jitter * dt;

    var transmittance = 1.0;
    var col = vec3<f32>(0.0);

    for (var i: i32 = 0; i < MAX_VIEW_STEPS; i = i + 1) {
        if (t >= t_far || transmittance < TRANSMITTANCE_CUTOFF) {
            break;
        }
        let p = u.cam_origin.xyz + t * dir;
        let sigma = sample_sigma(p);

        if (sigma > SIGMA_SKIP) {
            let t_sun = light_march(p, sun);

            // Height-modulated ambient: crude stand-in for the witness sky
            // integral; brighter near cloud top, floor of 0.3 (witness
            // AMBIENT_HEIGHT_FLOOR) near the base.
            let h = clamp((p.z - u.bmin.z) / (u.bmax.z - u.bmin.z), 0.0, 1.0);
            let ambient = ambient_strength * (0.3 + 0.7 * h) * AMBIENT_TINT;

            let src = SUN_COLOR * (t_sun * phase) + ambient;
            let alpha = 1.0 - exp(-sigma * dt);
            col = col + transmittance * alpha * src;
            transmittance = transmittance * (1.0 - alpha);
        }
        t = t + dt;
    }

    col = col + transmittance * sky;
    return vec4<f32>(tone_map(col, exposure), 1.0);
}
