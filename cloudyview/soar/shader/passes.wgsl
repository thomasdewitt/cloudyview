// --- the veil ------------------------------------------------------------------

// What every surface stage writes to the veil target (@location(1)), and
// what the march reads back at group 1 binding 2: premultiplied radiance
// and opacity (half floats), and the composite depth (a full f32 — tens of
// kilometres). Packed into a uint target so no driver's float
// path ever touches the bits. Opacity, 1 - transmittance, rather than
// transmittance: a half float rounds any transmittance above ~0.99976 to
// exactly 1, which would drop a faint plume's fringe (and its radiance with
// it) at a hard edge; 1 - T keeps its full relative precision down there.
struct SurfaceOut {
    @location(0) surf: vec4<f32>,
    @location(1) veil: vec4<u32>,
}

fn veil_pack(rgb: vec3<f32>, trans: f32, t: f32) -> vec4<u32> {
    return vec4<u32>(pack2x16float(rgb.xy),
                     pack2x16float(vec2<f32>(rgb.z, 1.0 - trans)),
                     bitcast<u32>(t), 0u);
}

// Whether the city's steam is the live view's (row 8.w, which the city's row
// 8 has spare): drawn after the accumulation by fs_steam_live at the live
// clock, so it keeps moving over a parked, converging picture. Off, it is
// marched into the veil as it always was — every still, video and replay,
// and the meter's of a capture.
fn city_steam_live() -> bool {
    return CITY && u.ocean.w > 0.5;
}

// No veil: nothing scattered in, everything let through.
fn veil_none() -> vec4<u32> {
    return veil_pack(vec3<f32>(0.0), 1.0, 1e30);
}

struct Veil {
    rgb: vec3<f32>,
    trans: f32,
    t: f32,
}

fn veil_unpack(v: vec4<u32>) -> Veil {
    let rg = unpack2x16float(v.x);
    let bt = unpack2x16float(v.y);
    return Veil(vec3<f32>(rg, bt.x), 1.0 - bt.y, bitcast<f32>(v.z));
}

// The pass itself. Only the CITY specialization is ever run through it; any
// other scene returns a miss and the host never draws it.
@fragment
fn fs_surface(@builtin(position) frag_pos: vec4<f32>) -> SurfaceOut {
    var out: SurfaceOut;
    out.surf = surface_first_stage(frag_pos);
    out.veil = veil_none();
    return out;
}

// The first stage's surface at a pixel (its centre, frag_pos.xy).
fn surface_first_stage(frag_pos: vec4<f32>) -> vec4<f32> {
    let cr = camera_ray(frag_pos);
    var surf = vec4<f32>(0.0, 0.0, 0.0, 1e30);
    if (CITY) {
        surf = surface_pass_city(cr.dir, cr.pixel_angle);
    }
    return surf;
}

// The second stage, for a surface that has one. Most pixels leave at once;
// see each surface's refine function for when.
@fragment
fn fs_surface_refine(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    let plain = textureLoad(surface_plain, vec2<i32>(frag_pos.xy), 0);
    let cr = camera_ray(frag_pos);
    if (CITY) {
        return surface_refine_city(cr, plain);
    }
    return plain;
}

// The third stage, for a surface that has one: the city's street
// (surface_street_city). It reads the second stage's surface at group 1
// binding 1, as the second reads the first's, and writes the final surface
// and the veil.
@fragment
fn fs_surface_street(@builtin(position) frag_pos: vec4<f32>) -> SurfaceOut {
    let prev = textureLoad(surface_plain, vec2<i32>(frag_pos.xy), 0);
    let cr = camera_ray(frag_pos);
    if (CITY) {
        return surface_street_city(cr, prev);
    }
    var out: SurfaceOut;
    out.surf = prev;
    out.veil = veil_none();
    return out;
}

// --- lens water: the state ---------------------------------------------------
//
// Water on the camera's lens after it flies through cloud. The look is
// lens.wgsl's, drawn by each host's present (browser) or post pass (Python)
// AFTER accumulation, never into the accumulated image. This is its state,
// one texel, integrated once per presented frame: x = wetness in [0, 1],
// y = w*, the level it is heading for at this extinction (the browser's
// read-back settles the loop on |w - w*|, whatever the frame rate), z, w 0. The host puts the step's dt in row 24.w: the live view's
// wall-clock frame time, parked or not (the lens keeps wetting in cloud and
// drying out of it while you hover; being drawn after accumulation, it does
// not disturb a parked picture's convergence), a video's resampled frame
// interval. Stills pass no state at all and are dry in both hosts.
//
// The wetting law, exact in dt at constant extinction sigma at the camera:
//   dw/dt = K_WET f (1 - w) - k_dry(f) (1 - f) w,  f = 1 - exp(-sigma L_WET),
//   k_dry(f) = K_DRY in cloud, K_DRY_CLEAR in clear air,
// is linear in w at constant f, whatever k_dry(f) is, so one step of any
// length is
//   w(t + dt) = w* + (w - w*) exp(-a dt),
//   a = K_WET f + k_dry(f) (1 - f),  w* = K_WET f / a,
// and one step of 2 dt equals two of dt at constant f: a replay at another
// frame rate sees the same wetness wherever the extinction at the camera
// changes slowly against the frame interval (f is taken at the step's end).
// f is the cloud's alone (the field's extinction; the haze is not in it).
// Clear air is where the first law's level, K_WET f / (K_WET f + K_DRY
// (1 - f)), is under LENS_DRY_FLOOR, the wetness that counts as dry anyway:
// f below ~6.7e-4, sigma below ~2.2e-5 /m. Not where sigma is exactly 0: a
// field's clear air can carry a faint positive tail (the RCE demo's, median
// f ~3e-5, over a third of its volume), and the camera samples sigma
// trilinearly. Dense cloud (f ~ 1) wets the lens in a few seconds, thin
// cloud (f small) only to a low level, slowly; in cloud this is the first
// law, its w* and its rate unchanged. In clear air the lens drains at
// K_DRY_CLEAR, 3x K_DRY (Thomas, 2026-11-08: outside of cloud the drops
// "should disappear 3x faster"), to 5 % in 5 s. The same law in numpy is
// tests/test_lens_water.py's reference.
const LENS_L_WET: f32 = 30.0;       // m of cloud path that fogs the lens
const LENS_K_WET: f32 = 0.6;        // 1/s
const LENS_K_DRY: f32 = 0.2;        // 1/s, in cloud
const LENS_K_DRY_CLEAR: f32 = 0.6;  // 1/s, in clear air
const LENS_DRY_FLOOR: f32 = 0.002;  // below this, and drying: exactly dry

@fragment
fn fs_lens_state(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    // The previous state, bound where the refine stages bind surface_plain.
    let prev = textureLoad(surface_plain, vec2<i32>(0, 0), 0);
    let dt = max(u.surface_offset.w, 0.0);
    // A NaN extinction reads as none (a comparison with NaN is false, so
    // this is exact where max(NaN, 0) is left to the implementation).
    let s_raw = sample_sigma(u.cam_origin.xyz);
    let sigma = select(0.0, s_raw, s_raw > 0.0);
    let f = 1.0 - exp(-sigma * LENS_L_WET);
    // Clear air: the first law's level would count as dry. Its denominator
    // is >= min(K_WET, K_DRY) > 0.
    let w_first = LENS_K_WET * f / (LENS_K_WET * f + LENS_K_DRY * (1.0 - f));
    let k_dry = select(LENS_K_DRY_CLEAR, LENS_K_DRY, w_first >= LENS_DRY_FLOOR);
    let a = LENS_K_WET * f + k_dry * (1.0 - f);   // >= min(K_WET, K_DRY) > 0
    let w_star = LENS_K_WET * f / a;
    var w = w_star + (prev.x - w_star) * exp(-a * dt);
    // Dry is exactly 0, so a dry lens draws nothing at all (lens.wgsl); a
    // state that is not a number starts again dry rather than stay NaN.
    w = select(w, 0.0, w < LENS_DRY_FLOOR && w_star < LENS_DRY_FLOOR);
    w = select(0.0, w, w >= 0.0);
    return vec4<f32>(w, w_star, 0.0, 0.0);
}

// --- the live view's steam -------------------------------------------------
//
// Thomas (2026-11-08): the steam "should keep going when the flight is
// paused". A parked view converges by averaging frames of one frozen scene,
// so steam marched into those frames would either freeze with them or, with
// the clock running, smear the average. The live view therefore marches
// the city with the steam left out (city_steam_live: the veil is only its
// depth) and draws it here, once per presented frame, at the live clock
// (row 24.z), onto the accumulated image — the lens water's arrangement.
// Group 1: binding 0 the final surface (t in w), binding 1 the accumulated
// image. The composite and what it gets wrong: steam_live_city (a stub that
// returns the image as it came in a module without the city). The image is
// in display values, as the browser accumulates it: the one host that draws
// this pass always tone-maps (TONE_MAP false is the Python host's linear
// output, render_nested, which has no live view).
@fragment
fn fs_steam_live(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    let xy = vec2<i32>(frag_pos.xy);
    let acc = textureLoad(surface_plain, xy, 0);
    return steam_live_city(camera_ray(frag_pos), acc,
                           textureLoad(surface_buf, xy, 0).w);
}

@fragment
fn fs_bake_light(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    let tex_dims = vec3<f32>(textureDimensions(light_tau, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    // frag_pos.xy = (iz + 0.5, iy + 0.5) over the (nz, ny) target; the slice
    // index is the field-x plane.
    let data_g = vec3<f32>(
        u.light_cache.y,
        frag_pos.y - 0.5,
        frag_pos.x - 0.5
    );
    // Texel i sits at box fraction (i+0.5)/N — the same half-cell-padded
    // convention sample_level and sample_light_tau map with, so the read-back
    // at texel centers returns exactly what was baked.
    let p = u.bmin.xyz
        + ((data_g + 0.5) / dims) * (u.bmax.xyz - u.bmin.xyz);
    let tau = light_march_tau(p, u.sun_dir.xyz, 0.0, 0.0);
    return vec4<f32>(tau, 0.0, 0.0, 1.0);
}
