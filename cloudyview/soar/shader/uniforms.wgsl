// soar's shader: a volume ray marcher for cloud fields.
//
// The module a host compiles is these files joined in manifest.json's order
// (the shared files, the files of the surface the world stands on, and the
// `absent` stand-ins of every other surface), behind a header of compile-time
// switches the host writes:
//
//     const PERIODIC_DOMAIN: bool   the field tiles in x and y
//     const NESTED: bool            a second, finer field sits inside the first
//     const TONE_MAP: bool          false returns linear radiance, not an image
//     const LIGHT_CACHE: bool       cloud samples read the baked sun optical
//                                   depth instead of marching to the sun
//
// A fifth switch, CITY (the surface under the clouds is the night city, not
// the sea), comes with the surface: city/layout.wgsl declares it true and
// the stand-in city/absent.wgsl declares it false.
//
// They are constants rather than uniforms so that the branch not taken costs
// nothing: every `if` on one folds away when the module is compiled.
//
// Conventions:
// - World space is absolute metres, +x east, +y north, +z up.
// - The volume's box is cell-edge aligned (half a cell beyond the outermost
//   cell centres).
// - The field is extinction sigma (m^-1); tau is optical depth; radiance is
//   linear RGB until tone_map.
// - The field of view is horizontal; pixel row 0 is the top of the image.
// - The density texture is uploaded with width = nz, height = ny, depth = nx,
//   so an (nx, ny, nz) C-order array uploads as it lies in memory. Normalized
//   sample coordinates are therefore (z, y, x); see sample_level.
//
// This file: what the host hands the shader. One uniform block and the
// textures of group 0, shared by every pass; group 1 holds the per-frame
// targets of the surface passes, which are sized to the frame.
//
// The hosts fill the block as rows of four floats, numbered from 0, and the
// JavaScript and Python hosts must fill it identically. Each row is declared here as the fields it holds: a vec3 and
// a scalar, four scalars, or one vec4. The whole block is also the hosts'
// scene key: a change in any row restarts the accumulation, except the
// sampling state (frame_index, subpixel_jitter, jitter_scale). That is why a
// few slots the shader never reads are packed.

struct Uniforms {
    // Row 0. The camera's position (m) and tan(horizontal field of view / 2).
    cam_origin: vec3<f32>,
    tan_half_fov: f32,
    // Row 1. Unit forward, and the image's aspect (width / height).
    cam_forward: vec3<f32>,
    aspect: f32,
    // Row 2. Unit right, and the exposure: the factor on radiance ahead of
    // the tone curve.
    cam_right: vec3<f32>,
    exposure: f32,
    // Row 3. Unit up, and the march jitter enable (0.0 or 1.0): the entry
    // phase of the view march, the quadrature phases of the sun march and
    // the forward pre-march, and the draw on the solar disc.
    cam_up: vec3<f32>,
    march_jitter: f32,
    // Row 4. Unit direction toward the sun (the moon under CITY), and this
    // frame's index in the accumulation, which every stratified stream is
    // indexed by (math.wgsl).
    sun_dir: vec3<f32>,
    frame_index: f32,
    // Rows 5-6. The volume's box (m), and the step of the view march and of
    // the light march in it (m).
    bmin: vec3<f32>,
    dt_view: f32,
    bmax: vec3<f32>,
    dt_light: f32,
    // Rows 7-8. The nest (NESTED), as rows 5-6 are for the outer level: its
    // box (m) and its own view-march and light-march steps (m). Zero when
    // NESTED is false.
    nest_bmin: vec3<f32>,
    nest_dt_view: f32,
    nest_bmax: vec3<f32>,
    nest_dt_light: f32,
    // Rows 9-13: the lighting spectra, computed by the host each frame from
    // the sun's elevation (fixed night values under CITY), each with one
    // scalar of the atmosphere beside it.
    // Row 9. The colour of the direct beam on cloud, and the light-transfer
    // split strength, which the host fades out as the sun rises (always 0
    // under CITY).
    cloud_sun: vec3<f32>,
    light_split: f32,
    // Row 10. The colour of the ambient (skylight) fill, and the haze, the
    // one aerosol amount, from which the host derives haze_beta0 and
    // ocean_haze_beta. It is negative at the clear end of its range, and the
    // sky terms clamp it at 0.
    ambient_tint: vec3<f32>,
    haze: f32,
    // Row 11. The sky colour at the horizon (the night haze colour under
    // CITY), and the haze extinction at sea level, beta0 (m^-1).
    sky_horizon: vec3<f32>,
    haze_beta0: f32,
    // Row 12. The circumsolar bloom colour, and the haze scale height H (m):
    // extinction is beta0 exp(-z / H). H <= 0 selects a uniform haze, beta0
    // at every altitude. That is unphysical but bounds the periodic march:
    // in an exponential atmosphere an upward ray can leave the haze without
    // reaching the cutoff optical depth and so runs to the range ceiling,
    // while in a uniform one every ray ends at the same distance
    // (periodic_march_cap).
    sky_bloom: vec3<f32>,
    haze_scale_height: f32,
    // Row 13. The solar disc colour, and the haze extinction along the
    // sightline to the water (m^-1).
    sky_disc: vec3<f32>,
    ocean_haze_beta: f32,
    // Row 14. The RGB reflectance of the water body (zero under CITY), and
    // the surface's z (m): sea level, or the city's ground.
    water_reflectance: vec3<f32>,
    surface_z: f32,
    // Row 15. The surface tile (the sea's wave normals, or the city's tile):
    // its cell size (m), its extent (m), whether the surface is drawn (0.0
    // or 1.0), and its highest mip level.
    tile_cell: f32,
    tile_extent: f32,
    surface_enabled: f32,
    tile_max_lod: f32,
    // Row 16. surface_offset (m) is where the surface tile's origin sits in
    // the world in this frame, folded into [0, tile extent). The
    // water-normal lookups read the tile at (world - offset), and the city
    // builds its whole frame from it (city_camera_origin). The hosts write
    // the camera's own phase within the tile, cam.xy - surface position, so
    // that folding the camera back by a cloud period moves the offset with
    // it and the surface does not jump. Only the offset modulo the tile can
    // matter, because everything drawn on the surface is periodic in the
    // tile. A scene whose surface is fixed in the world writes its static
    // offset, which for the sea is exactly 0.0. scene_clock is the scene's
    // time (s; scene_time). lens_dt is the time step of the lens-water
    // state (s), read by fs_lens_state only and 0 in every other pass.
    surface_offset: vec2<f32>,
    scene_clock: f32,
    lens_dt: f32,
    // Row 17. The image size (px); the sub-pixel camera-ray jitter enable
    // (0.0 or 1.0); and the jitter amplitude in [0, 1], a scale on every
    // jittered quantity.
    image_width: f32,
    image_height: f32,
    subpixel_jitter: f32,
    jitter_scale: f32,
    // Row 18. The tone map: the white point (the exposed radiance that
    // displays as 1.0), the gamma (the display value is the tone curve
    // raised to 1 / gamma), and the contrast about mid-grey (1.0 =
    // identity). Then ice_mode (0.0 or 1.0), the ice-detection view: cloud
    // source terms are recoloured by the per-voxel ice fraction (binding 8)
    // and the sky is drawn green (ice.wgsl).
    white_point: f32,
    gamma: f32,
    contrast: f32,
    ice_mode: f32,
    // Row 19. Distance level of detail, as tan(theta). The sun march's step
    // is at least view distance * light_lod_tan, so distant cloud gets a
    // coarser (never a truncated) tau quadrature. The view march's step is
    // at least t * view_lod_tan, a step of fixed angle rather than fixed
    // length; it is also the angle the surfaces resolve detail at
    // (lod_pixel_angle). 0 in either = fixed dt. Then the city's two:
    // roof_density, a factor on the rooftop clutter probabilities (1 = as
    // tuned), and steam_live, 1.0 when the steam is drawn after the
    // accumulation rather than marched into it (city_steam_live). Both are
    // zero on a day scene.
    light_lod_tan: f32,
    view_lod_tan: f32,
    roof_density: f32,
    steam_live: f32,
    // Row 20. Two slots the shader does not read: 1.0 when the domain is
    // periodic, and 1.0 when cloud samples read tau_sun from the light_tau
    // texture. They are the host's copies of the PERIODIC_DOMAIN and
    // LIGHT_CACHE switches, packed so that changing one changes the scene
    // key. Then a slot nothing uses, and bake_slice, the field-x plane of
    // the cache being rendered, read by fs_bake_light only.
    periodic_flag: f32,
    light_cache_flag: f32,
    unused_20_2: f32,
    bake_slice: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// The outer level's extinction field (m^-1). r32float or filterable r16float;
// either reads as texture_3d<f32>, and both levels share one format.
@group(0) @binding(1) var vol: texture_3d<f32>;
// Trilinear, clamp-to-edge on all three axes.
@group(0) @binding(2) var vol_samp: sampler;
// The surface tile: the sea's wave normals under the ocean, the city's tile
// under CITY.
@group(0) @binding(3) var ocean_normals: texture_2d<f32>;
// Repeat-wrapping trilinear sampler for binding 3.
@group(0) @binding(4) var ocean_samp: sampler;
// The finer nested level. Always bound (a 1x1x1 zero texture when there is
// no nest) so one bind-group layout serves both cases.
@group(0) @binding(5) var nest_vol: texture_3d<f32>;
// The periodic way to read a volume texture: repeat on the two lateral
// texture axes (depth indexes field x, height indexes field y), clamp on the
// vertical (width). See sample_level.
@group(0) @binding(6) var vol_wrap_samp: sampler;
// The light cache over the outer box (sample_light_cache, fs_bake_light):
// red is the optical depth to the sun, green the optical depth of the cloud
// overhead. Always bound: a 1x1x1 zero texture while there is no finished
// bake.
@group(0) @binding(7) var light_tau: texture_3d<f32>;
// The ice fraction of extinction, sigma_ice / (sigma_liq + sigma_ice), per
// voxel of the outer level, in the texel layout of `vol`. Always bound: a
// 1x1x1 zero texture when the field has no ice variable. Read only in
// ice-detection mode (u.ice_mode).
@group(0) @binding(8) var ice_frac: texture_3d<f32>;

// Group 1 holds what is sized to the frame, apart from the scene resources
// of group 0. Its three bindings mean different things to different passes;
// a pass that does not use one is given a 1x1 stand-in.
//
// Binding 0, for fs_main: the opaque surface under the clouds (the traced
// city) as the surface passes left it for this frame, (radiance, t) per
// render pixel, with t = 1e30 on a miss. Read only under CITY.
@group(1) @binding(0) var surface_buf: texture_2d<f32>;
// Binding 1: the previous stage's output, for a pass that continues one:
// the surface so far (fs_surface_refine, fs_surface_street), the previous
// lens state (fs_lens_state), the accumulated image (fs_steam_live).
@group(1) @binding(1) var surface_plain: texture_2d<f32>;
// Binding 2, the veil: one semi-transparent layer per pixel in front of the
// surface (the city's steam), as the last surface stage left it:
// premultiplied in-scattered radiance, opacity, and the depth the march
// composites it at (veil_pack / veil_unpack). Read by fs_main only under
// CITY.
@group(1) @binding(2) var veil_buf: texture_2d<u32>;

// The scene clock (s, u.scene_clock). Anything in the scene that moves reads the
// time here (the city's steam). The hosts pack 0 for a surface where nothing
// moves (the sea).
fn scene_time() -> f32 {
    return u.scene_clock;
}
