// The vertex stage and the camera ray that every fragment pass shares.

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // One triangle that covers the screen: (-1, -1), (-1, 3), (3, -1).
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// The ray through a fragment, with what its stratified draws are indexed by.
struct CameraRay {
    dir: vec3<f32>,
    frame_index: u32,       // this frame's index in the accumulation
    strat_pixel: vec2<u32>, // the pixel that seeds the stratified rotations
    pixel_angle: f32,       // the angle the surfaces resolve detail at (rad)
    true_pixel_angle: f32,  // one pixel's angle, without the LOD floor (rad)
    jitter_amp: f32,        // march jitter enable * jitter amplitude
}

// One pixel's angular size (rad, small-angle): 2 tan(fov / 2) / width. A
// surface filters its texture over this footprint whatever the LOD angle is:
// a wider one would only blur it.
fn true_pixel_angle() -> f32 {
    return 2.0 * u.tan_half_fov / u.image_width;
}

// The angle a surface's level of detail resolves at: the true pixel's, or
// the view-step LOD angle (u.view_lod_tan) where that is coarser. It is the
// same angle that sets the cloud march's step, so one setting coarsens the
// march and cuts the surfaces' work together.
fn lod_pixel_angle() -> f32 {
    return max(true_pixel_angle(), u.view_lod_tan);
}

// fs_main and the surface passes all take their ray from here, so a surface
// pass traces exactly the ray the march composites: the same pixel, the same
// sub-pixel draw, the same frame index. That is what lets sub-pixel surface
// detail accumulate.
fn camera_ray(frag_pos: vec4<f32>) -> CameraRay {
    var cr: CameraRay;
    let img_w = u.image_width;
    let img_h = u.image_height;
    let tan_half_fov = u.tan_half_fov;
    let aspect = u.aspect;
    let subpixel_on = u.subpixel_jitter;
    let jitter_scale = clamp(u.jitter_scale, 0.0, 1.0);
    // The frame's index in the accumulation is the sequence coordinate of
    // every stratified stream, and the pixel seeds their rotations (see
    // math.wgsl).
    cr.frame_index = u32(max(u.frame_index, 0.0));
    cr.strat_pixel = vec2<u32>(frag_pos.xy);

    // Pixel -> camera ray. frag_pos.xy is the pixel centre, (px + 0.5,
    // py + 0.5), and framebuffer y = 0 is the top of the image.
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
    cr.dir = normalize(u.cam_forward
                       + ndc_x * u.cam_right
                       + ndc_y * u.cam_up);
    // The two angles: see true_pixel_angle and lod_pixel_angle.
    cr.true_pixel_angle = true_pixel_angle();
    cr.pixel_angle = lod_pixel_angle();
    cr.jitter_amp = u.march_jitter * jitter_scale;
    return cr;
}
