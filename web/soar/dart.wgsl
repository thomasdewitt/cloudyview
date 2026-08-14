// The paper dart (WGSL): a thrown sheet of A4, shaded to sit in the cloud
// render.
//
// Written against the bird's uniform block byte for byte — they share a base
// class and a render pass, and only the meaning of two slots differs (see
// BirdUniforms' anim/bmin.w; here they are wing flex and flutter). Everything
// about the LOOK is the bird's inverse, on purpose:
//
//   - Hard panels, not smooth sheets. The swift's whole argument was that
//     light must sweep across a wing rather than step from facet to facet.
//     Paper is the opposite: it is flat between creases and it breaks sharply
//     at them, and as the dart banks its panels light up one at a time. The
//     mesh gives that by not sharing vertices across a fold; this file must
//     not smooth it back out.
//   - Cut edges, not barbs. The bird fades the outer few percent of every
//     vane because a feather ends in barbs. A guillotined sheet does not, and
//     softening it here would cost the one crisp silhouette in the scene.
//   - Transmission by LAYER COUNT. A feather vane is one thickness of keratin
//     and always the same one. Paper comes in a countable number of sheets —
//     dartmesh.js folds the thing for real and counts them — so the nose at
//     eight layers is dead opaque while the wing tip at one lights up like a
//     lampshade. That single effect is most of what says "paper" at ninety
//     pixels, and it is only available because the mesh was folded rather
//     than modelled.
//   - A broad transmission lobe. Feather vane is thin and forward-scattering,
//     so the bird's glow is tight (a ninth power). Paper is a volume
//     diffuser: light that gets through has forgotten where it came from. The
//     lobe here is nearly flat, and getting that wrong makes paper look like
//     plastic film.
//
// This dart has been thrown before. The creases are furred, the left tip is
// bent, one trailing corner has curled, and there are fingerprints on the
// nose. Most of that is geometry; the grime and the crease shadow are here.

struct DartUniforms {
    // Camera-relative -> clip. Camera basis only: NO translation. See the
    // long note in bird.wgsl — the flyers are drawn camera-relative because
    // soar's world coordinates are unbounded and float32 is not.
    vp: mat4x4<f32>,
    // Dart local -> camera-relative (rotation, then the offset from the
    // camera; mesh is pre-scaled).
    model: mat4x4<f32>,
    // Rotation-only copy of `model`, for normals.
    nrot: mat4x4<f32>,
    // xyz = camera origin (m, absolute — the volume is world-anchored),
    // w = exposure.
    cam_origin: vec4<f32>,
    // xyz = unit direction toward the sun, w = ambient strength.
    sun_dir: vec4<f32>,
    // xyz = volume AABB min (m), w = WING FLEX (rad, tip up).
    bmin: vec4<f32>,
    // xyz = volume AABB max (m), w = tone map gamma.
    bmax: vec4<f32>,
    // xyz = spectral sun radiance, w = transmission gain.
    sun_rgb: vec4<f32>,
    // xyz = sky/ambient radiance, w = sheen gain.
    sky_rgb: vec4<f32>,
    // x = flutter phase (rad), y = flutter amplitude (0..1),
    // z = keel deflection (rad), w = unused.
    anim: vec4<f32>,
    // x = tone-map white point, y = display contrast; z/w unused.
    display: vec4<f32>,
};

@group(0) @binding(0) var<uniform> du: DartUniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;

const PART_WING: f32 = 1.0;

// Ordinary office paper: bright, and very slightly warm. Not white — nothing
// in a sky render should be 1.0, and paper is about 0.75 at best.
const PAPER: vec3<f32> = vec3<f32>(0.745, 0.735, 0.700);
// What handling leaves: darker, and yellower than the paper under it.
const GRIME: vec3<f32> = vec3<f32>(0.400, 0.370, 0.310);
// What comes through one sheet. Warm, because paper's own brighteners and
// the fibre both take more from the blue end.
const TRANSMISSION_TINT: vec3<f32> = vec3<f32>(1.00, 0.93, 0.80);
// Fraction of light that survives ONE thickness of 80 gsm. Eight layers of
// this is 1e-3, which is the nose being opaque.
const SHEET_TRANSMIT: f32 = 0.42;
// Light coming back up off the cloud deck and the sea.
const GROUND_BOUNCE: vec3<f32> = vec3<f32>(1.00, 0.97, 0.90);

const INV_PI: f32 = 0.3183098862;
const OCCLUSION_STEPS: i32 = 16;
// Matches CREASE_CAP_M in dartmesh.js: beyond this a point is not near a fold.
const CREASE_CAP: f32 = 0.0035;
// Matches SEMI_SPAN in dartmesh.js: (210/2 - 18) mm of wing outboard of the
// keel fold. Only the flutter needs it, to turn a spanwise rate into a slope.
const SEMI_SPAN: f32 = 0.087;
// Radians of travelling wave across one semi-span, and its peak deflection.
// Six millimetres of flutter on an 87 mm wing is about what a thrown dart
// does; it is small, and it is the only thing on the aircraft that moves fast.
const FLUTTER_WAVES: f32 = 7.5;
const FLUTTER_AMPLITUDE_M: f32 = 0.006;

// ---------------------------------------------------------------------------
// Shared-convention helpers (duplicated from raymarch.wgsl; keep in sync)
// ---------------------------------------------------------------------------

fn ray_box(origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    let t0 = (du.bmin.xyz - origin) * inv_dir;
    let t1 = (du.bmax.xyz - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2<f32>(max(max(tmin.x, tmin.y), tmin.z),
                     min(min(tmax.x, tmax.y), tmax.z));
}

// The volume carries a one-texel ghost ring, so data voxel i lives at texel
// i + 1 and the domain maps to (g * N + 1.5) / (N + 2), not onto [0, 1].
fn sample_sigma(p: vec3<f32>) -> f32 {
    let dims = vec3<f32>(textureDimensions(vol, 0));
    let n = max(dims - vec3<f32>(2.0), vec3<f32>(1.0));
    let g = (p - du.bmin.xyz) / (du.bmax.xyz - du.bmin.xyz);
    let t = (clamp(g, vec3<f32>(0.0), vec3<f32>(1.0)) * n + 1.5) / dims;
    return textureSampleLevel(vol, vol_samp, vec3<f32>(t.z, t.y, t.x), 0.0).r;
}

const TONE_MAP_SHOULDER: f32 = 0.35;

fn tone_map(hdr: vec3<f32>, exposure: f32, gamma: f32) -> vec3<f32> {
    let exposed = hdr * exposure;
    let wp = du.display.x;
    let w2 = wp * wp;
    let per_channel = exposed * (1.0 + exposed / w2) / (1.0 + exposed);
    let y = dot(exposed, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma_preserving = exposed * (1.0 + y / w2) / (1.0 + y);
    let k = TONE_MAP_SHOULDER * smoothstep(1.0, 3.0, y);
    let mapped = mix(per_channel, chroma_preserving, k);
    let encoded = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)),
                      vec3<f32>(1.0 / max(gamma, 0.01)));
    let c = du.display.y;
    return clamp(vec3<f32>(0.5) + (encoded - vec3<f32>(0.5)) * c,
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

fn rot_y(p: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

/** Smooth value noise, for paper formation. */
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
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
    // x = signed span, y = chord, z = part, w = layers.
    @location(3) attrs: vec4<f32>,
    // x = metres to the nearest crease, y = grime.
    @location(4) surf: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) span: f32,
    @location(3) chord: f32,
    @location(4) part: f32,
    @location(5) layers: f32,
    @location(6) crease: f32,
    @location(7) grime: f32,
) -> VSOut {
    var p = pos;
    var n = normal;

    let mag = abs(span);
    if (mag > 1e-5) {
        let side = sign(span);

        // Wing flex. A dart's wing is a single unstiffened sheet, so it bends
        // under load along its whole length rather than hinging at a root —
        // same shape of law as the bird's bend, but a much higher exponent,
        // because paper is stiff until it is not and then gives all at once
        // near the tip.
        let bend = du.bmin.w * side * pow(mag, 1.9);
        p = rot_y(p, bend);
        n = rot_y(n, bend);

        // Trailing-edge flutter: a travelling wave running outboard along the
        // back of the wing. This is what a paper plane does at speed and it
        // is entirely absent from a bird, which is half the reason the two
        // read as different things in motion.
        let aft = smoothstep(0.52, 1.0, chord);
        let phase = du.anim.x - FLUTTER_WAVES * mag;
        let amp = du.anim.y * FLUTTER_AMPLITUDE_M * aft * pow(mag, 2.0);
        p.z = p.z + amp * sin(phase);
        // Tilt the normal by the wave's own slope, or it shades as a flat
        // sheet that happens to be in the wrong place. dz/d(span) of the line
        // above, turned into a slope across the wing by the semi-span.
        let slope = -FLUTTER_WAVES * amp * cos(phase) / SEMI_SPAN;
        n = normalize(n - vec3<f32>(side * slope, 0.0, 0.0));
    } else {
        // The keel swings a little as a rudder. It is hinged along its TOP,
        // where it meets the wing fold at z = 0, and free along its bottom
        // edge at z = -KEEL — so the deflection grows with -z, not with
        // distance from the bottom. Getting that backwards pivots the keel
        // about its free edge and swings the part that is attached to the
        // aeroplane.
        let sweep = du.anim.z * smoothstep(0.35, 1.0, chord);
        p.x = p.x + sweep * (-p.z);
    }

    var out: VSOut;
    let rel = (du.model * vec4<f32>(p, 1.0)).xyz;
    out.rel_pos = rel;
    out.local_pos = p;
    out.normal = (du.nrot * vec4<f32>(n, 0.0)).xyz;
    out.attrs = vec4<f32>(span, chord, part, layers);
    out.surf = vec2<f32>(crease, grime);
    out.clip = du.vp * vec4<f32>(rel, 1.0);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let exposure = du.cam_origin.w;
    let gamma = du.bmax.w;
    let ambient_strength = du.sun_dir.w;
    let cam = du.cam_origin.xyz;

    let span = in.attrs.x;
    let chord = in.attrs.y;
    let layers = max(in.attrs.w, 1.0);
    let is_wing = in.attrs.z > PART_WING - 0.5;
    let crease = in.surf.x;
    let grime = in.surf.y;

    // Thin shell: orient the normal toward the viewer so both faces shade.
    // A sheet of paper genuinely has two sides and both are paper.
    var n = normalize(in.normal);

    // Folded-over paper does not lie flat. Each layer region — the two big
    // corner flaps and what shows of the sheet beneath them — sits at its own
    // slightly different angle, by a degree or so, and that is why a dart
    // seen from above is a mosaic of panels rather than one white shape.
    //
    // The mesh cannot give this: those flaps are not separate surfaces in the
    // final object, they are the same sheet at different thicknesses. But the
    // layer count already names the regions, so tilting the normal by a fixed
    // amount per count separates them exactly where the folds do. Without it
    // the aeroplane renders as a single blown-out triangle, which was the
    // first thing wrong with it.
    let region = hash21(vec2<f32>(layers, 3.7));
    let tilt = (region - 0.5) * 0.055;
    n = normalize(n + vec3<f32>(tilt, 0.0, tilt * 0.6));

    let to_cam = -in.rel_pos;
    let dist = length(to_cam);
    let view = to_cam / dist;
    if (dot(n, view) < 0.0) { n = -n; }

    let sun = du.sun_dir.xyz;
    let sun_rgb = du.sun_rgb.xyz;
    let sky_rgb = du.sky_rgb.xyz;

    // --- the sheet --------------------------------------------------------

    // Formation: paper is a mat of fibres and it is cloudy, not uniform. The
    // grain runs along the sheet's long axis, so the noise is stretched that
    // way — an A4 sheet folded across its grain is a different object from
    // one folded along it, and this is the only place that shows.
    let sheet_uv = vec2<f32>(in.local_pos.x * 620.0, in.local_pos.y * 155.0);
    let formation = vnoise(sheet_uv) - 0.5;

    var albedo = PAPER * (1.0 + 0.035 * formation);
    albedo = mix(albedo, GRIME, 0.55 * grime);

    // The crease. Every fold on used paper is a line of shadow with a slightly
    // pale, slightly furred shoulder either side — the fibres have been broken
    // and stand up. One hairline does more for "this is folded" than the fold
    // geometry does, because at this size the geometry is a pixel wide.
    let near = 1.0 - smoothstep(0.0, CREASE_CAP, crease);
    let core = pow(near, 3.0);
    let fur = near * (1.0 - core);
    albedo = albedo * (1.0 - 0.42 * core) * (1.0 + 0.09 * fur);

    // --- light ------------------------------------------------------------

    // Paper is close to Lambertian and it is bright — but only up to a point.
    // The bird damps its Lambert to 0.15 because a thin sunlit wing that goes
    // pale against blue sky inverts the silhouette. Paper genuinely is the
    // brightest thing in the frame and should read that way, but taken to
    // 0.85 it clipped to flat white in full sun and lost every panel, crease
    // and fold with it. Bright enough to be paper, dark enough to have form.
    let ndotl = max(dot(n, sun), 0.0);
    let direct = sun_rgb * (INV_PI * 0.42 * ndotl);

    // Hemispheric fill: sky above, cloud and sea below.
    let up = 0.5 + 0.5 * n.z;
    let fill = mix(GROUND_BOUNCE * (ambient_strength * 1.0),
                   sky_rgb * (ambient_strength * 1.7), up);

    // Transmission through N thicknesses. The lobe is deliberately broad —
    // pow(.., 1.5) rather than the bird's ninth power — because paper
    // scatters light so many times on the way through that what comes out has
    // lost the sun's direction almost entirely. The fibre clouding shows far
    // more strongly here than in reflection, which is exactly what happens
    // when you hold a sheet up to a window.
    let back = max(-dot(n, sun), 0.0);
    let toward = pow(max(dot(-view, sun), 0.0), 1.5);
    let through = pow(SHEET_TRANSMIT, layers);
    let cloudiness = 1.0 + 0.42 * formation;
    let glow = sun_rgb * TRANSMISSION_TINT
             * (back * toward * through * cloudiness * du.sun_rgb.w);

    // Barely any specular: office paper is matte, and the only thing it does
    // at grazing incidence is pick up a little sky. Overdo this and the dart
    // turns into laminated plastic.
    let fresnel = pow(1.0 - max(dot(n, view), 0.0), 5.0);
    let sheen = sky_rgb * (fresnel * du.sky_rgb.w * ambient_strength * 0.35);

    let hdr = albedo * (direct + fill) + glow + sheen;

    // --- occlusion --------------------------------------------------------

    // Cloud between camera and dart: a short Beer-Lambert march of the
    // resident sigma texture, exactly as the bird does it.
    //
    // This is the one place absolute world coordinates survive, and they have
    // to: the volume is anchored to the world, not to the camera. It is also
    // the one place they are harmless — this asks which voxels a few metres
    // of segment cross, and a voxel is metres wide.
    let dir = -view;
    let hit = ray_box(cam, 1.0 / dir);
    let t0 = max(hit.x, 0.0);
    let t1 = min(hit.y, dist);
    var tau = 0.0;
    if (t1 > t0) {
        let dt = (t1 - t0) / f32(OCCLUSION_STEPS);
        for (var i: i32 = 0; i < OCCLUSION_STEPS; i = i + 1) {
            tau = tau + sample_sigma(cam + (t0 + (f32(i) + 0.5) * dt) * dir) * dt;
        }
    }

    // No edge softening, and that is the decision, not an omission. The bird
    // fades its vane margins because a feather ends in barbs half a pixel
    // wide. A sheet of A4 ends where the guillotine left it.
    return vec4<f32>(tone_map(hdr, exposure, gamma), exp(-tau));
}
