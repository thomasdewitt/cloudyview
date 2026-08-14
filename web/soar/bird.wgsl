// The bird (WGSL): a common swift, shaded to sit inside the cloud render.
//
// Drawn as a small raster pass after the volume march. The geometry is in
// birdmesh.js — real anatomy at real size, with the hand's ten primaries as
// separate feathers — and this shader's job is to make that read as an animal
// rather than as a model of one. Three things do most of that work:
//
//   - Backlit transmission. A feather vane is one layer of keratin. With the
//     sun behind the bird the outer primaries light up amber and the shafts
//     stay dark, and that single effect is most of what makes a photograph of
//     a bird against the sky look like a bird against the sky.
//   - The scene's own light. The sun colour and the sky fill come in from the
//     host, spectrally shifted by the same code that lights the clouds, so at
//     a low sun the bird warms with everything else instead of staying the
//     grey it was at noon.
//   - Soft margins. Real feather edges are barbs, not an edge. At this size
//     that is under a pixel, but letting the outer few percent of each vane
//     fade is the difference between a feather and a polygon.
//
// Wingbeat articulation happens in the vertex stage, per structure: the arm
// barely moves, the hand carries the amplitude, and the hand supinates on the
// way up. That is a swift's actual wingbeat, and it is why the silhouette
// changes shape through the stroke instead of just tilting.

// EVERYTHING POSITIONAL HERE IS RELATIVE TO THE CAMERA, and it has to stay
// that way. A swift is 0.40 m across with 2-8 mm of structure in its feathers,
// which is finer than a float32 ulp once world coordinates pass a couple of
// hundred kilometres — and soar's camera is unbounded in altitude. Feeding
// absolute world positions through `model` and `vp`, as this once did, put a
// cancellation of two ~1e6 quantities between the mesh and the screen and the
// bird rasterized as three or four stray slivers. The host differences the
// camera out in doubles (see bird.js writeUniforms); nothing below is ever
// bigger than the handful of metres the bird is away.
struct BirdUniforms {
    // Camera-relative -> clip. Camera basis only: NO translation.
    vp: mat4x4<f32>,
    // Bird local -> camera-relative (rotation, then the offset from the
    // camera to the bird; mesh is pre-scaled).
    model: mat4x4<f32>,
    // Rotation-only copy of `model`, for normals.
    nrot: mat4x4<f32>,
    // xyz = camera origin (m), w = exposure (Reinhard pre-scale).
    cam_origin: vec4<f32>,
    // xyz = unit direction toward the sun, w = ambient strength.
    sun_dir: vec4<f32>,
    // xyz = volume AABB min (m), w = shoulder flap angle (rad).
    bmin: vec4<f32>,
    // xyz = volume AABB max (m), w = tone map gamma.
    bmax: vec4<f32>,
    // xyz = spectral sun radiance (what the clouds see), w = transmission gain.
    sun_rgb: vec4<f32>,
    // xyz = sky/ambient radiance, w = sheen gain.
    sky_rgb: vec4<f32>,
    // x = wrist flex (rad), y = hand twist (rad), z = tail spread, w = unused.
    anim: vec4<f32>,
    // x = tone-map white point, y = display contrast; z/w unused. The
    // scene's tone map is per-frame state now, so the bird's copy takes the
    // same values or it reads as a sticker with its own camera.
    display: vec4<f32>,
};

@group(0) @binding(0) var<uniform> bu: BirdUniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;

const PART_ARM: f32 = 1.0;
const PART_PRIMARY: f32 = 2.0;
const PART_TAIL: f32 = 3.0;

// Sooty brown-black: a swift is not grey, it is a very dark warm brown, and
// the difference shows the moment the sun is low.
const PLUMAGE: vec3<f32> = vec3<f32>(0.055, 0.047, 0.039);
// Flight feathers are glossier and a shade cooler than the body coverts.
const FLIGHT_FEATHER: vec3<f32> = vec3<f32>(0.048, 0.044, 0.041);
// The pale chin — the only light marking a common swift has, and the thing
// that stops the head from reading as a featureless point.
const THROAT: vec3<f32> = vec3<f32>(0.34, 0.31, 0.27);
// Light coming back up off the cloud deck and the sea.
const GROUND_BOUNCE: vec3<f32> = vec3<f32>(1.00, 0.97, 0.90);
// What comes through one layer of keratin: warm, because the blue is
// scattered out of it long before the red is.
const TRANSMISSION_TINT: vec3<f32> = vec3<f32>(1.00, 0.68, 0.44);

const INV_PI: f32 = 0.3183098862;
const OCCLUSION_STEPS: i32 = 16;

// The rachis sits at this fraction across the vane — see birdmesh.js, where
// the feather surface is offset by (v - 0.38). The outer web is the narrow
// one, which is what makes a flight feather a flight feather.
const RACHIS_V: f32 = 0.38;

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

// The volume carries a one-texel ghost ring, so data voxel i lives at texel
// i + 1 and the domain maps to (g * N + 1.5) / (N + 2), not onto [0, 1]. The
// desktop shader mapped it onto [0, 1] and sampled about a texel and a half
// off; there is no reason to inherit that.
fn sample_sigma(p: vec3<f32>) -> f32 {
    let dims = vec3<f32>(textureDimensions(vol, 0));
    let n = max(dims - vec3<f32>(2.0), vec3<f32>(1.0));
    let g = (p - bu.bmin.xyz) / (bu.bmax.xyz - bu.bmin.xyz);
    let t = (clamp(g, vec3<f32>(0.0), vec3<f32>(1.0)) * n + 1.5) / dims;
    return textureSampleLevel(vol, vol_samp, vec3<f32>(t.z, t.y, t.x), 0.0).r;
}

const TONE_MAP_SHOULDER: f32 = 0.35;

fn tone_map(hdr: vec3<f32>, exposure: f32, gamma: f32) -> vec3<f32> {
    let exposed = hdr * exposure;
    let wp = bu.display.x;
    let w2 = wp * wp;
    let per_channel = exposed * (1.0 + exposed / w2) / (1.0 + exposed);
    let y = dot(exposed, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma_preserving = exposed * (1.0 + y / w2) / (1.0 + y);
    let k = TONE_MAP_SHOULDER * smoothstep(1.0, 3.0, y);
    let mapped = mix(per_channel, chroma_preserving, k);
    let encoded = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)),
                      vec3<f32>(1.0 / max(gamma, 0.01)));
    let c = bu.display.y;
    return clamp(vec3<f32>(0.5) + (encoded - vec3<f32>(0.5)) * c,
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

fn rot_y(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

fn rot_x(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

/** Cheap deterministic per-feather jitter, so no two are identical. */
fn hash11(x: f32) -> f32 {
    return fract(sin(x * 78.233) * 43758.5453);
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    // Camera to this vertex, in world axes, metres. Not a world position.
    @location(0) rel_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) local_pos: vec3<f32>,
    // x = signed span, y = chord, z = part, w = feather id.
    @location(3) attrs: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) span: f32,
    @location(3) chord: f32,
    @location(4) part: f32,
    @location(5) feather: f32,
) -> VSOut {
    var p = pos;
    var n = normal;

    // The tail fans open in a turn and closes in a glide.
    if (part > PART_TAIL - 0.5) {
        let spread = 1.0 + bu.anim.z;
        p = vec3<f32>(p.x * spread, p.y, p.z);
        n = normalize(vec3<f32>(n.x / spread, n.y, n.z));
    }

    let mag = abs(span);
    if (mag > 1e-5) {
        let side = sign(span);
        // Supination first, about the wing's own long axis: on the upstroke a
        // swift twists the hand nose-down so it slices rather than pushes.
        // Only the hand does this, and progressively.
        let hand = smoothstep(0.40, 1.0, mag);
        let twist = bu.anim.y * side * hand;
        p = rot_x(p, twist);
        n = rot_x(n, twist);

        // Then the bend. The rotation each point receives grows with its own
        // distance out the span, so the wing curls along its length like a
        // real one instead of hinging like a board. The exponent is what
        // keeps the arm nearly still while the hand carries the stroke.
        let bend = -(bu.bmin.w * side * pow(mag, 1.35)
                     + bu.anim.x * side * hand);
        p = rot_y(p, bend);
        n = rot_y(n, bend);
    }

    var out: VSOut;
    let rel = (bu.model * vec4<f32>(p, 1.0)).xyz;
    out.rel_pos = rel;
    out.local_pos = p;
    out.normal = (bu.nrot * vec4<f32>(n, 0.0)).xyz;
    out.attrs = vec4<f32>(span, chord, part, feather);
    out.clip = bu.vp * vec4<f32>(rel, 1.0);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let exposure = bu.cam_origin.w;
    let gamma = bu.bmax.w;
    let ambient_strength = bu.sun_dir.w;
    let cam = bu.cam_origin.xyz;

    let span = in.attrs.x;
    let chord = in.attrs.y;
    let part = in.attrs.z;
    let feather = in.attrs.w;
    let is_feather = part > PART_ARM - 0.5;
    let is_primary = abs(part - PART_PRIMARY) < 0.5;
    let is_body = part < PART_ARM - 0.5;

    // Thin shells: orient the normal toward the viewer so both faces shade.
    var n = normalize(in.normal);
    // The camera sits at the origin of this frame, so the vector to it is
    // just the negated interpolant — no differencing of world positions, and
    // so no cancellation however far out the camera has flown.
    let to_cam = -in.rel_pos;
    let dist = length(to_cam);
    let view = to_cam / dist;
    if (dot(n, view) < 0.0) { n = -n; }

    let sun = bu.sun_dir.xyz;
    let sun_rgb = bu.sun_rgb.xyz;
    let sky_rgb = bu.sky_rgb.xyz;

    // --- plumage ----------------------------------------------------------

    var albedo = select(PLUMAGE, FLIGHT_FEATHER, is_feather);

    // The rachis: a hard dark shaft down each flight feather, and the reason
    // a backlit wing shows stripes rather than a uniform glow.
    let from_rachis = abs(chord - RACHIS_V);
    var shaft = 0.0;
    if (is_feather) {
        shaft = exp(-pow(from_rachis / 0.075, 2.0));
        albedo = albedo * (1.0 - 0.45 * shaft);
        // No two feathers are quite alike, and wear pales the outer primaries.
        let jitter = 0.88 + 0.24 * hash11(feather * 37.0 + part * 5.0);
        let wear = select(0.0, 0.20 * smoothstep(0.55, 1.0, feather), is_primary);
        albedo = albedo * jitter * (1.0 + wear);
    }

    // The pale throat, on the underside of the head only. `feather` runs nose
    // to tail along the body and `chord` runs belly to back.
    if (is_body) {
        let along = smoothstep(0.10, 0.19, feather)
                  * smoothstep(0.42, 0.30, feather);
        let under = smoothstep(0.46, 0.05, chord);
        albedo = mix(albedo, THROAT, along * under * 0.85);

        // The eye. One dark pixel with a wet catch-light is worth more to a
        // face at this size than any amount of geometry.
        let eye = vec3<f32>(0.0105, 0.0455, 0.0080);
        let d = length(vec3<f32>(abs(in.local_pos.x), in.local_pos.y,
                                 in.local_pos.z) - eye);
        albedo = mix(albedo, vec3<f32>(0.012, 0.010, 0.010),
                     smoothstep(0.0042, 0.0018, d));
    }

    // --- light ------------------------------------------------------------

    let ndotl = max(dot(n, sun), 0.0);
    // Damped Lambert: at full strength a thin sunlit wing goes pale against
    // blue sky and the silhouette inverts, which no photograph of a bird
    // ever does.
    let direct = sun_rgb * (INV_PI * 0.15 * ndotl);

    // Hemispheric fill: sky from above, cloud and sea from below. The deck
    // under a bird flying over cloud is genuinely bright, and leaving it out
    // is why CG birds so often look pasted on.
    let up = 0.5 + 0.5 * n.z;
    let fill = mix(GROUND_BOUNCE * (ambient_strength * 1.0),
                   sky_rgb * (ambient_strength * 1.7), up);

    // Transmission. Needs three things at once: the sun behind the surface,
    // the sun roughly behind the bird from where we stand, and a thin part of
    // a feather to come through. The shaft blocks it, the vane passes it, and
    // the long outer primaries pass the most.
    var glow = vec3<f32>(0.0);
    if (is_feather) {
        let back = max(-dot(n, sun), 0.0);
        let toward = pow(max(dot(-view, sun), 0.0), 9.0);
        let vane = (1.0 - shaft) * smoothstep(0.02, 0.16, from_rachis);
        let thinness = select(0.55, 0.55 + 0.45 * abs(span), is_primary);
        glow = sun_rgb * TRANSMISSION_TINT
             * (back * toward * vane * thinness * bu.sun_rgb.w);
    }

    // A grazing sheen of sky off oiled feathers, and — when the sun is
    // behind — the rim of light that runs round the edge of anything with a
    // surface. The rim is what stops a backlit bird from being a hole in the
    // sky: it is the single cue that says "solid body", and leaving it out is
    // why the first version of this read as a silhouette sticker.
    let fresnel = pow(1.0 - max(dot(n, view), 0.0), 5.0);
    let sheen = sky_rgb * (fresnel * bu.sky_rgb.w * ambient_strength);
    let behind = pow(max(dot(-view, sun), 0.0), 6.0);
    let rim = sun_rgb * (fresnel * behind * 0.030);

    let hdr = albedo * (direct + fill) + glow + sheen + rim;

    // --- occlusion and edges ----------------------------------------------

    // Cloud between camera and bird: a short Beer-Lambert march of the
    // resident sigma texture. The volume pass already painted that cloud, so
    // ordinary alpha blending composites the two correctly to first order.
    //
    // This is the one place absolute world coordinates survive, and they have
    // to: the volume is anchored to the world, not to the camera. It is also
    // the one place they are harmless — this asks which voxels a 5 m segment
    // crosses, and a voxel is metres wide.
    let dir = -view;
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

    // Feather margins are barbs, not an edge. Fading the outer few percent of
    // each vane costs nothing and removes the polygon.
    var edge = 1.0;
    if (is_feather) {
        edge = smoothstep(0.0, 0.09, chord) * smoothstep(1.0, 0.91, chord);
        edge = 0.35 + 0.65 * edge;
    }

    return vec4<f32>(tone_map(hdr, exposure, gamma), exp(-tau) * edge);
}
