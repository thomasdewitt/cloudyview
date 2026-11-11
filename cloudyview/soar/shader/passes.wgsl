// The fragment passes other than the cloud march (fs_main, march.wgsl): the
// surface passes, which trace the opaque surface under the clouds and its
// veil; the lens-water state; the live view's steam; and the bake of the sun
// optical depth cache.

// --- the surface passes ---------------------------------------------------
//
// The city is an opaque surface traced in closed form (a DDA over its block
// grid), and tracing it takes a large fragment program. Compiled into
// fs_main it would slow every march pixel, whether or not the pixel's ray
// reaches the city (see volume.wgsl). So the surface is traced in passes of
// its own, one fragment per render pixel, before the fs_main pass of a
// frame, into the target fs_main reads at group 1. The march keeps only the
// composite: the t >= t_surface test, the dt clamp and the fallback after
// its loop.
//
// A surface too large for one pass is traced in stages. fs_surface draws
// what every pixel needs. fs_surface_refine and fs_surface_street each read
// the stage before at group 1 binding 1 and re-trace only the pixels that
// need more, so most pixels leave a later stage at once. Each surface
// decides which pixels those are: the city marks the pixels its facade
// shell can touch in the sign bit of red (CITY_SHELL_MARK). Which surfaces
// have the later stages is the hosts' list.
//
// What belongs to the surface and lies in front of it as part of its look
// (the city's fog) is folded into its rgb. A semi-transparent layer that
// must also show against the sky, or have cloud in front of or behind it
// (the city's steam), goes to the veil, which fs_main composites at the
// veil's own depth.

// --- the veil -------------------------------------------------------------

// What a surface stage writes to the veil target (@location(1)) and what the
// march reads back at group 1 binding 2: premultiplied radiance and opacity
// as half floats, and the composite depth as a full f32 (it runs to tens of
// kilometres). The target is a uint format, so the bits are stored as
// packed. Opacity (1 - transmittance) is stored and not transmittance: a
// half float rounds any transmittance above ~0.99976 to exactly 1, which
// would cut a faint plume's fringe off at a hard edge, while 1 - T keeps
// its full relative precision near zero.
struct SurfaceOut {
    @location(0) surf: vec4<f32>,
    @location(1) veil: vec4<u32>,
}

fn veil_pack(rgb: vec3<f32>, trans: f32, t: f32) -> vec4<u32> {
    return vec4<u32>(pack2x16float(rgb.xy),
                     pack2x16float(vec2<f32>(rgb.z, 1.0 - trans)),
                     bitcast<u32>(t), 0u);
}

// Whether the city's steam is left out of the march and drawn after the
// accumulation by fs_steam_live (u.steam_live), where it keeps moving over a
// parked, converging picture. Otherwise it is marched into the veil.
fn city_steam_live() -> bool {
    return CITY && u.steam_live > 0.5;
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

// The first surface stage. Only a CITY module is run through it. Any other
// returns a miss, and the host does not draw the pass.
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

// The second stage, for a surface that has one. See each surface's refine
// function for which pixels it re-traces.
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
// binding 1 and writes the final surface and the veil.
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
// lens.wgsl's, drawn by each host after the accumulation and never into the
// accumulated image. This pass integrates its state, one texel, once per
// presented frame: x = wetness w in [0, 1], y = w*, the level it is heading
// for at the present extinction, z and w = 0. The host puts the step's dt in
// u.lens_dt: the live view's wall-clock frame time (parked or not: the lens
// keeps wetting in cloud and drying out of it while the camera hovers), or a
// video's frame interval. Stills pass no state and are dry.
//
// The wetting law, with sigma the cloud's extinction at the camera (the haze
// is not in it):
//
//   dw/dt = K_WET f (1 - w) - k_dry (1 - f) w,  f = 1 - exp(-sigma L_WET),
//   k_dry = K_DRY in cloud, K_DRY_CLEAR in clear air.
//
// It is linear in w at constant f, so one step of any length is exact:
//
//   w(t + dt) = w* + (w - w*) exp(-a dt),
//   a = K_WET f + k_dry (1 - f),  w* = K_WET f / a.
//
// One step of 2 dt equals two of dt at constant f, so a replay at another
// frame rate sees the same wetness wherever the extinction at the camera
// changes slowly against the frame interval (f is taken at the step's end).
// Dense cloud (f ~ 1) wets the lens in a few seconds; thin cloud only to a
// low level, slowly.
//
// "Clear air" is where the in-cloud law's level, K_WET f / (K_WET f + K_DRY
// (1 - f)), is under LENS_DRY_FLOOR, the wetness that counts as dry anyway:
// f below ~6.7e-4, sigma below ~2.2e-5 m^-1. The test is not sigma == 0
// because a field's clear air can carry a faint positive tail, and the
// camera samples sigma trilinearly. There the lens drains at K_DRY_CLEAR,
// reaching 5% in 5 s. tests/test_lens_water.py holds the same law in numpy.
const LENS_L_WET: f32 = 30.0;       // m of cloud path that fogs the lens
const LENS_K_WET: f32 = 0.6;        // 1/s
const LENS_K_DRY: f32 = 0.2;        // 1/s, in cloud
const LENS_K_DRY_CLEAR: f32 = 0.6;  // 1/s, in clear air
const LENS_DRY_FLOOR: f32 = 0.002;  // below this, and drying: exactly dry

@fragment
fn fs_lens_state(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    // The previous state, at the binding the later surface stages read
    // their input from.
    let prev = textureLoad(surface_plain, vec2<i32>(0, 0), 0);
    let dt = max(u.lens_dt, 0.0);
    // A NaN extinction reads as none. A comparison with NaN is false, so the
    // select is defined where max(NaN, 0) is left to the implementation.
    let s_raw = sample_sigma(u.cam_origin);
    let sigma = select(0.0, s_raw, s_raw > 0.0);
    let f = 1.0 - exp(-sigma * LENS_L_WET);
    // Clear air: the in-cloud law's level would count as dry. Its
    // denominator is >= min(K_WET, K_DRY) > 0.
    let w_first = LENS_K_WET * f / (LENS_K_WET * f + LENS_K_DRY * (1.0 - f));
    let k_dry = select(LENS_K_DRY_CLEAR, LENS_K_DRY, w_first >= LENS_DRY_FLOOR);
    let a = LENS_K_WET * f + k_dry * (1.0 - f);   // >= min(K_WET, K_DRY) > 0
    let w_star = LENS_K_WET * f / a;
    var w = w_star + (prev.x - w_star) * exp(-a * dt);
    // Dry is exactly 0, so that a dry lens draws nothing at all (lens.wgsl).
    // A state that is not a number starts again dry.
    w = select(w, 0.0, w < LENS_DRY_FLOOR && w_star < LENS_DRY_FLOOR);
    w = select(0.0, w, w >= 0.0);
    return vec4<f32>(w, w_star, 0.0, 0.0);
}

// --- the live view's steam -------------------------------------------------
//
// A parked view converges by averaging frames of one frozen scene. Steam
// marched into those frames would either freeze with them or, with the clock
// running, smear the average. The live view therefore marches the city with
// the steam left out (city_steam_live: the veil then holds only its depth)
// and draws it here, once per presented frame, at the scene clock (row
// 24.z), onto the accumulated image, as the lens water is.
//
// Group 1: binding 0 is the final surface (t in w), binding 1 the
// accumulated image. The composite itself is steam_live_city (in a module
// without the city, a stub that returns the image as it came). The image is
// in display values, as the browser accumulates it: the one host that draws
// this pass always tone-maps.
@fragment
fn fs_steam_live(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    let xy = vec2<i32>(frag_pos.xy);
    let acc = textureLoad(surface_plain, xy, 0);
    return steam_live_city(camera_ray(frag_pos), acc,
                           textureLoad(surface_buf, xy, 0).w);
}

// One field-x plane of the sun optical depth cache. The host renders this
// into a 2D r16float target of size (cache_nz, cache_ny) for the slice
// u.bake_slice, then copies the target into that depth slice of the
// light_tau texture. A 3D texture cannot be a render target and r16float is
// not a storage format, but r16float can be rendered to, so the bake is a
// render pass per slice and not a compute pass. Slice by slice also lets
// the browser spread a bake over frames.
//
// Each texel is light_march_tau at the texel's own world position, nest
// included, with dt_floor = 0 (the bake is paid once, so it takes the fine
// step everywhere) and no jitter (a static cache cannot average a random
// phase away).
@fragment
fn fs_bake_light(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    let tex_dims = vec3<f32>(textureDimensions(light_tau, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    // frag_pos.xy = (iz + 0.5, iy + 0.5) over the (nz, ny) target. The slice
    // index is the field-x plane.
    let data_g = vec3<f32>(
        u.bake_slice,
        frag_pos.y - 0.5,
        frag_pos.x - 0.5
    );
    // Texel i sits at box fraction (i + 0.5) / N, the convention
    // sample_level and sample_light_tau map with, so a read at a texel
    // centre returns exactly what was baked there.
    let p = u.bmin
        + ((data_g + 0.5) / dims) * (u.bmax - u.bmin);
    let tau = light_march_tau(p, u.sun_dir, 0.0, 0.0);
    return vec4<f32>(tau, 0.0, 0.0, 1.0);
}
