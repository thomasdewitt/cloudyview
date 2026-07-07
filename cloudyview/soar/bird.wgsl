// CloudyView soar: the bird (WGSL).
//
// A tiny raster pass drawn after the volume raymarch: a stylized swift that
// leads the flight a few metres ahead of the camera. Entirely additive
// garnish — no volume-render code or look lives here.
//
// - Wing flap happens in the vertex shader: each vertex carries a signed
//   span fraction (0 on the body, ±1 at the wingtips) and is rotated about
//   the body's forward (y) axis by flap_angle * fraction, which curls the
//   wing along its span like a real wingbeat. The same rotation is applied
//   to the normal.
// - Occlusion / atmosphere: the fragment stage marches a few samples of the
//   resident sigma texture along the camera→fragment segment and outputs
//   alpha = exp(-tau), so the bird fades naturally into cloud. The volume
//   pass has already painted that cloud behind it; ordinary alpha blending
//   composites the two correctly to first order.
// - Lighting is minimal and scene-consistent: sun-dot-normal with the same
//   HDR sun color as the volume shader, a sky-tinted ambient, then the same
//   Reinhard + gamma 1.4 tone map so the bird sits in the image rather than
//   on it.

struct BirdUniforms {
    // World -> clip (camera view-projection, WebGPU depth 0..1).
    vp: mat4x4<f32>,
    // Bird local -> world (rotation * translation; mesh is pre-scaled).
    model: mat4x4<f32>,
    // Rotation-only copy of `model` for normals.
    nrot: mat4x4<f32>,
    // xyz = camera origin (m), w = exposure (Reinhard pre-scale).
    cam_origin: vec4<f32>,
    // xyz = unit direction toward the sun, w = ambient strength.
    sun_dir: vec4<f32>,
    // xyz = volume AABB min (m), w = current flap angle (rad).
    bmin: vec4<f32>,
    // xyz = volume AABB max (m), w = unused.
    bmax: vec4<f32>,
};

@group(0) @binding(0) var<uniform> bu: BirdUniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;

// Sooty swift plumage: dark, so the silhouette reads against bright cloud
// and blue sky alike, with just enough sun response to model the form.
const ALBEDO: vec3<f32> = vec3<f32>(0.035, 0.031, 0.028);
const SUN_COLOR: vec3<f32> = vec3<f32>(22.0, 21.0, 17.0);   // raymarch.wgsl
const AMBIENT_TINT: vec3<f32> = vec3<f32>(0.75, 0.85, 1.05); // raymarch.wgsl
const GAMMA: f32 = 1.4;                                      // raymarch.wgsl
const INV_PI: f32 = 0.3183098862;
const OCCLUSION_STEPS: i32 = 16;

// ---------------------------------------------------------------------------
// Shared-convention helpers (duplicated from raymarch.wgsl; keep in sync)
// ---------------------------------------------------------------------------

fn ray_box(origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    let t0 = (bu.bmin.xyz - origin) * inv_dir;
    let t1 = (bu.bmax.xyz - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

fn sample_sigma(p: vec3<f32>) -> f32 {
    let g = (p - bu.bmin.xyz) / (bu.bmax.xyz - bu.bmin.xyz);
    return textureSampleLevel(vol, vol_samp, vec3<f32>(g.z, g.y, g.x), 0.0).r;
}

fn tone_map(hdr: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = hdr * exposure;
    let mapped = exposed / (1.0 + exposed);
    return pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)),
               vec3<f32>(1.0 / GAMMA));
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) span_frac: f32,
) -> VSOut {
    // Wing curl: rotate about the local forward (y) axis by an angle
    // proportional to the signed span fraction. The mesh is built with the
    // shoulder line at z = 0, so the rotation needs no pivot offset.
    // Sign: flap_angle > 0 raises both wingtips.
    let a = -bu.bmin.w * span_frac;
    let ca = cos(a);
    let sa = sin(a);
    let p = vec3<f32>(pos.x * ca + pos.z * sa, pos.y,
                      -pos.x * sa + pos.z * ca);
    let n = vec3<f32>(normal.x * ca + normal.z * sa, normal.y,
                      -normal.x * sa + normal.z * ca);

    var out: VSOut;
    let wp = (bu.model * vec4<f32>(p, 1.0)).xyz;
    out.world_pos = wp;
    out.normal = (bu.nrot * vec4<f32>(n, 0.0)).xyz;
    out.clip = bu.vp * vec4<f32>(wp, 1.0);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let exposure = bu.cam_origin.w;
    let ambient_strength = bu.sun_dir.w;
    let cam = bu.cam_origin.xyz;

    // Two-sided flat-ish shading: the mesh is a thin shell, so orient the
    // (face) normal toward the viewer.
    var n = normalize(in.normal);
    let to_cam = cam - in.world_pos;
    if (dot(n, to_cam) < 0.0) {
        n = -n;
    }

    // Minimal scene-consistent light, tuned toward silhouette: a *damped*
    // Lambert sun (a full-strength sun flips thin wing surfaces to pale
    // beige against blue sky and the silhouette inverts), sky-tinted
    // ambient from above, and a soft warm bounce from the bright
    // cloud/ocean deck below. The bird stays clearly darker than sunlit
    // cloud and blue sky from every angle, but is modelled, not a cutout.
    let ndotl = max(dot(n, bu.sun_dir.xyz), 0.0);
    let sun = SUN_COLOR * (INV_PI * 0.3 * ndotl);
    let sky_amb = AMBIENT_TINT * (0.55 + 0.45 * n.z)
                  * (ambient_strength * 3.0);
    let bounce = vec3<f32>(1.0, 0.97, 0.90) * (0.5 - 0.5 * n.z)
                 * (ambient_strength * 2.0);
    let hdr = ALBEDO * (sun + sky_amb + bounce);

    // Occlusion by cloud between the camera and the bird: short Beer-Lambert
    // march of the resident sigma texture over the segment clipped to the
    // volume AABB. alpha = transmittance; the volume pass already painted
    // the occluding cloud, so src-alpha blending composites correctly.
    let dist = length(to_cam);
    let dir = -to_cam / dist;             // camera -> fragment
    let hit = ray_box(cam, 1.0 / dir);
    let t0 = max(hit.x, 0.0);
    let t1 = min(hit.y, dist);
    var tau = 0.0;
    if (t1 > t0) {
        let dt = (t1 - t0) / f32(OCCLUSION_STEPS);
        for (var i: i32 = 0; i < OCCLUSION_STEPS; i = i + 1) {
            let t = t0 + (f32(i) + 0.5) * dt;
            tau = tau + sample_sigma(cam + t * dir) * dt;
        }
    }
    let alpha = exp(-tau);

    return vec4<f32>(tone_map(hdr, exposure), alpha);
}
