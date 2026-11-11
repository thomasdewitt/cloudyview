

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

// One field-x plane of the sun-tau cache. The host renders this into a 2D
// r16float target of size (cache_nz, cache_ny) for slice u.light_cache.y,
// then copies the target into that depth slice of the light_tau texture —
// 3D textures are not renderable, and r16float is not a storage format, but
// it IS renderable, so the bake is a render pass rather than a compute pass.
// Rendering slice by slice is also what lets the browser spread a bake over
// frames instead of stalling one.
//
// Each texel evaluates the exact live march — light_march_tau through
// sample_level_at, nest included — at the cache texel's own world position,
// with dt_floor = 0 (full quadrature: the bake is paid once, so it takes the
// fine step everywhere) and zero jitter (a static cache cannot average a
// randomized phase away, so it takes the unbiased-enough left-endpoint rule
// the near field always used). Texel i sits AT data coordinate i, matching
// sample_level's convention, so sample_light_tau reads back exactly what was
// baked at texel centers.
// ---------------------------------------------------------------------------
// The surface pre-pass
// ---------------------------------------------------------------------------
//
// The city is an opaque surface traced in closed form — a DDA over the block
// grid — and its tracing is register-hungry. In fs_main it shared a kernel
// with the cloud march, so its registers were the march's too: every surface feature compiled in cost
// every march pixel occupancy, whether its rays ever reached that feature or
// not. So the surface is traced in a pass of its own, fs_surface, one
// fragment per render pixel, before every fs_main pass of a frame, into the
// target fs_main reads at group 1. The march keeps only the composite (the t >= t_surface test,
// the dt clamp and the post-loop fallback), with the same radiance at the
// same t as when it traced the surface itself.
//
// Both entries take their ray from camera_ray, so the pass traces exactly
// the ray the march composites: the same pixel, subpixel stream and frame
// index, which is what keeps sub-pixel surface detail accumulating.
//
// A surface whose full detail is too register-hungry even for its own pass
// splits in two: fs_surface draws what every pixel needs cheaply, and
// fs_surface_refine re-traces only the pixels that need more, reading the
// first stage's result at group 1 binding 1 and writing the final one. The
// second stage's registers are paid only by its own kernel, which most
// pixels leave at once. Each surface decides its own early-out. The city
// marks the pixels its facade shell can touch in the sign bit of red
// (CITY_SHELL_MARK: set by surface_pass_city for a ray that entered a
// building's relief envelope within the shell's reach),
// returns every other one untouched, and clears the mark on the rest; t
// leaves it as t came, sign included, unless the shell replaced the hit.
// Which surfaces have the second stage is the hosts' list
// (soar_host.SURFACE_REFINE_SURFACES, renderer.js _hasRefineStage).
//
// One function per surface, chosen by the scene's specialization. What
// belongs to the surface and sits in front of it along the ray as part of
// its look (city fog) is folded into its rgb here. A semi-transparent layer
// that must also show against the sky, or with a cloud in front of or
// behind it (the city's steam), goes to the VEIL instead (@location(1),
// veil_pack), which fs_main composites at the veil's own depth.

struct CameraRay {
    dir: vec3<f32>,
    frame_index: u32,       // this frame's index in the accumulation
    strat_pixel: vec2<u32>, // the pixel that seeds the stratified scrambles
    pixel_angle: f32,       // the city's resolving angle (rad/px)
    true_pixel_angle: f32,  // one pixel's angle, without the LOD slider
    jitter_amp: f32,        // jitter_on * jitter_scale
}

// One pixel's angular size: the frame's own, before any LOD floor. What a
// surface filters its texture over (a wider footprint only blurs it; the
// view-step LOD angle decides how much work is done, not how sharp).
fn true_pixel_angle() -> f32 {
    return 2.0 * u.cam_origin.w / u.params.x;
}

// The angle a surface's level of detail resolves at: the true pixel's, or
// the view-step LOD angle (u.periodic.z, the app's one LOD slider) when that
// is coarser. See camera_ray.
fn lod_pixel_angle() -> f32 {
    return max(true_pixel_angle(), u.periodic.z);
}

fn camera_ray(frag_pos: vec4<f32>) -> CameraRay {
    var cr: CameraRay;
    let img_w = u.params.x;
    let img_h = u.params.y;
    let tan_half_fov = u.cam_origin.w;
    let aspect = u.cam_forward.w;
    let subpixel_on = u.flags.x;
    let jitter_scale = clamp(u.flags.y, 0.0, 1.0);
    // This frame's index in the accumulation, the sequence coordinate every
    // stratified stream is indexed by, and this pixel, which seeds their
    // scrambles (see the strat1/strat2 block).
    cr.frame_index = u32(max(u.sun_dir.w, 0.0));
    cr.strat_pixel = vec2<u32>(frag_pos.xy);

    // Pixel -> camera ray. Framebuffer y=0 is the image top, matching the
    // witness convention ndc_y = 1 - 2*(py+0.5)/h.
    var sample_pos = frag_pos.xy;
    if (subpixel_on > 0.5) {
        let subpixel_offset = strat2(cr.frame_index, cr.strat_pixel,
                                     STRAT_STREAM_SUBPIXEL,
                                     vec2<f32>(STRAT_ALPHA_1, STRAT_ALPHA_2));
        sample_pos = sample_pos
                     + (subpixel_offset - vec2<f32>(0.5)) * jitter_scale;
    }
    let ndc_x = (2.0 * sample_pos.x / img_w - 1.0) * tan_half_fov;
    let ndc_y = (1.0 - 2.0 * sample_pos.y / img_h) * tan_half_fov / aspect;
    // @domain: forward plus in-plane offsets of the camera's orthonormal basis
    cr.dir = normalize(u.cam_forward.xyz
                       + ndc_x * u.cam_right.xyz
                       + ndc_y * u.cam_up.xyz);
    // The surfaces' LOD angle is the same knob as the cloud march's: the
    // larger of the true pixel angle and the view-step LOD angle
    // (u.periodic.z, the app's one LOD slider). Coarsening the slider cuts
    // the surfaces' work (how far the city's facade shell reaches) exactly
    // as it coarsens the march — one
    // lever, one degrees-not-meters law for the whole scene. What a surface
    // FILTERS its texture over stays the true pixel's footprint: a wider
    // one would only blur it.
    cr.true_pixel_angle = true_pixel_angle();
    cr.pixel_angle = lod_pixel_angle();
    cr.jitter_amp = u.cam_up.w * jitter_scale;
    return cr;
}
