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
//     const SKY_PROBE: bool         run the vertical sky-visibility march
//
// They are constants rather than uniforms so that the branch not taken costs
// nothing: every `if` on one folds away when the module is compiled.
//
// Conventions:
// - World space is absolute metres, +x east, +y north, +z up.
// - The volume's box is cell-edge aligned (half a cell beyond the outermost
//   cell centres).
// - The field of view is horizontal; pixel row 0 is the top of the image.
// - The density texture is uploaded with width = nz, height = ny, depth = nx,
//   so an (nx, ny, nz) C-order array uploads as it lies in memory. Normalized
//   sample coordinates are therefore (z, y, x); see sample_level.
//
// This file: what the host hands the shader. One uniform block and the
// textures of group 0, shared by every pass; group 1 holds the surface
// pre-pass's targets, which are sized to the frame.

struct Uniforms {
    // xyz = camera origin (m), w = tan(fov_horizontal / 2)
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
    // x = ocean z (m), yzw = ocean reflectance (witness.py:104-106).
    // Under CITY: x = ground z, yz = the tile's STATIC world offset, w =
    // the live view's steam (1.0: drawn after the accumulation at the live
    // clock, city_steam_live). yz is informational since row 24 — the
    // shader builds the city frame from the per-frame offset there
    // (city_camera_origin) and no longer reads this pair; hosts keep
    // writing it as the scene's own statement of where the tile was pinned.
    ocean: vec4<f32>,
    // x = FIF dx (m), y = FIF tile extent (m), z = ocean enabled, w = max normal LOD
    ocean_params: vec4<f32>,
    // x = subpixel camera-ray jitter enable (0.0 or 1.0),
    // y = jitter amplitude scale (x and y are sampling flags, excluded from
    //     the host's scene-identity key),
    // z = haze in [0, HAZE_MAX] — the user's one aerosol knob, which also
    //     sets row 16.w and row 19.y on the host side;
    // w = tone-map white point (exposed radiance that displays as 1.0)
    flags: vec4<f32>,
    // x = gradient shading, y = deep-shadow MS suppression,
    // z = directional ambient occlusion, w = bounce depth attenuation
    cb_realism: vec4<f32>,
    // x = gradient coarse weight, y = gradient coarse radius (m),
    // z = directional ambient occlusion floor,
    // w = cone stencil tan(theta) (0 = legacy fixed coarse radius)
    cb_params: vec4<f32>,
    // Rows 13-17: witness realism package (per-frame CPU spectral precompute,
    // see witness._spectral_lighting_colors / engine.write_uniforms).
    // xyz = spectral direct-beam color for cloud MS octaves,
    // w = low-sun sky field strength (0 = iter_006 azimuth-only field)
    cloud_sun: vec4<f32>,
    // xyz = spectral ambient tint, w = effective light-transfer split
    // strength (already elevation-faded on the CPU; 0 = legacy shader)
    ambient_tint: vec4<f32>,
    // xyz = spectral sky horizon color, w = aerial perspective strength
    // (0 = exact legacy: no haze extinction, no in-scatter)
    sky_horizon: vec4<f32>,
    // xyz = circumsolar bloom color, w = aerial beta0 (m^-1, sea level)
    sky_bloom: vec4<f32>,
    // xyz = solar disc color, w = aerial haze scale height (m), where
    //     w <= 0 means the haze does NOT thin with height: the sea-level
    //     extinction applies at every altitude. That is unphysical and it is
    //     the point — an exponential atmosphere lets an upward ray leave the
    //     haze without ever reaching the cutoff optical depth, so it marches
    //     to the range ceiling, while a uniform one caps every ray at the
    //     same distance. It is the cheapest range lever there is.
    sky_disc: vec4<f32>,
    // Rows 18-19: ocean realism (witness iter_004/009/011).
    // x = master gate (0 = exact legacy ocean shader), y = mip bias,
    // z = GGX glint strength, w = GGX base roughness
    ocean_realism_a: vec4<f32>,
    // x = ocean slope resolved fraction, y = ocean haze extinction
    // (m^-1), z = sky-reflection cloud-shadow floor,
    // w = display contrast about mid-grey (1.0 = identity)
    ocean_realism_b: vec4<f32>,
    // Row 20: periodic domain + distance LOD (2026-07-17 perf pass).
    // x = periodic enable in host scene identity (0.0 or 1.0); the shader
    //     itself branches on the compile-time PERIODIC_DOMAIN const.
    // y = light-march LOD tan(theta): sun-march dt floor grows as
    //     view_distance * y so distant cloud copies get coarser (never
    //     truncated) tau quadrature. 0 = fixed dt.
    // z = view-step LOD tan(theta): view-march dt floor grows as t * z —
    //     the degrees-not-meters step. 0 = fixed dt.
    periodic: vec4<f32>,
    // Rows 21-22: the optional nest (NESTED). Same packing as rows 5-6 but
    // for the finer level: xyz = nest AABB min/max (m), w = the nest's own
    // view-ray / light-march step dt (m). Zero-filled when NESTED is false.
    nest_bmin: vec4<f32>,
    nest_bmax: vec4<f32>,
    // Row 23: the lighting method, and the bake's slice index.
    //
    // x and z are NOT read by fs_main any more (2026-09-23): each is a
    // compile-time specialization (LIGHT_CACHE, SKY_PROBE — see their
    // declarations), because a run-time branch here paid for its untaken
    // side on every sample. The hosts still pack both, from the same flags
    // that pick the specialization, because the row is part of the
    // accumulation's scene key: a bake completing or a tier switching
    // methods is a scene change (and the browser restarts a parked view's
    // average on it outright — renderer.js drawFrame).
    //
    // x = read tau_sun for CLOUD samples from the light_tau texture instead
    //     of marching to the sun per sample (0.0 or 1.0). The ocean's two
    //     shadow marches and the sky probe stay live either way. The cached
    //     value is the central sun direction at zero quadrature jitter, so
    //     the solar-disc penumbra sampling (iter_007) does not apply to it —
    //     trilinear filtering of the cache is the only softening left, which
    //     is exactly what the A/B is judging.
    // y = bake slice index (the fs_bake_light pass only; fs_main never
    //     reads it): the field-x plane of the cache being rendered.
    // z = skip the vertical sky probe (0.0 or 1.0): every consumer of
    //     t_sky then sees a fully open sky. A cost/look toggle while the
    //     per-tier fate of these marches is decided.
    // w = ice-detection mode (0.0 or 1.0), a false-color view:
    //     cloud source terms are tinted by the per-voxel ice fraction
    //     (binding 8) and the sky is remapped to an alien green. At 0.0
    //     every multiply below is by exactly 1.0 and the frame is
    //     bit-identical to a build without the mode.
    light_cache: vec4<f32>,
    // Row 24: the surface offset (2026-09-01). xy = where the surface tile
    // sits in the world THIS FRAME renders, folded into [0, tile extent).
    // The water-normal reads sample at (world - offset), and the city
    // system derives its whole frame from it (see city_camera_origin). The
    // hosts write it per frame as cam.xy - surfacePosition — the camera's
    // own tile phase — so a cloud-period camera fold moves the offset with
    // the camera and the surface never jumps; a static render writes the
    // scene's static offset, folded. The fold is harmless BECAUSE the city
    // is a pure function of the tile frame: every cell index folds at the
    // tile (city_tile_cell) before it seeds anything, so only the offset
    // mod the tile can matter. A day scene with a parked camera writes
    // exactly 0.0, making the water subtraction bit-neutral — which is
    // what keeps the frozen judge-view goldens byte-identical.
    // z = the scene clock (scene_time, per surface: the sea and the city
    // run it). w = lens water's step dt for fs_lens_state (s; 0 in every
    // other pass, which never reads it).
    surface_offset: vec4<f32>,
    // Row 25: the city's look knobs, zero on a day scene, which reads none.
    // x = roof clutter density (the hosts' CITY_ROOF_DENSITY_BY_TIER at the
    //     tier: a factor on rooftopworks' clutter slot probabilities, 1 as
    //     the roofs were tuned); yzw spare.
    city_look: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;
// The surface tile: the sea's normals under the ocean, the city's tile under
// CITY.
@group(0) @binding(3) var ocean_normals: texture_2d<f32>;
// Repeat-wrapping trilinear sampler for binding 3.
@group(0) @binding(4) var ocean_samp: sampler;
// The finer nested level. Always bound (a 1x1x1 zero texture when there is
// no nest) so one bind-group layout serves both specializations.
@group(0) @binding(5) var nest_vol: texture_3d<f32>;
// The same volume texture read the periodic way: repeat on the two lateral
// texture axes (w indexes field x, v indexes field y), clamp on the vertical.
// A doubly periodic domain wraps in hardware rather than through an uploaded
// ghost ring — see sample_level.
@group(0) @binding(6) var vol_wrap_samp: sampler;
// The baked sun-tau cache over the outer AABB — see Uniforms.light_cache and
// sample_light_tau. Always bound (a 1x1x1 zero texture while the cache is
// off or stale) so one bind-group layout serves both paths.
@group(0) @binding(7) var light_tau: texture_3d<f32>;
// Ice-detection mode: per-voxel ice extinction fraction
// sigma_ice / (sigma_liq + sigma_ice) over the OUTER level's grid, same
// texel layout as `vol`. Always bound — a 1x1x1 zero stand-in when the field
// carries no ice variable — so one bind-group layout serves both. Read only
// when u.light_cache.w is set.
@group(0) @binding(8) var ice_frac: texture_3d<f32>;

// Group 1: the surface pre-pass's target, size-specific, so it lives apart
// from the scene resources in group 0 (which every pass shares unchanged):
// the opaque surface under the clouds — the traced city — as the pre-pass
// left it for this frame, (radiance, t) per render pixel, t = 1e30 on a
// miss. The march reads it, and only under CITY; other scenes bind a 1x1
// stand-in and never run the pass. See the surface pre-pass section.
@group(1) @binding(0) var surface_buf: texture_2d<f32>;
// The previous stage's surface, for a surface traced in stages
// (fs_surface_refine, fs_surface_street) — the city's plain boxes, then its
// facade shell — and a 1x1 stand-in for every other pass.
@group(1) @binding(1) var surface_plain: texture_2d<f32>;
// The VEIL: one semi-transparent layer per pixel in front of the surface —
// the city's steam — as the last surface stage left it: premultiplied
// in-scattered radiance, opacity, and the depth the march composites it at
// (veil_pack / veil_unpack). Read by the march only under CITY.
@group(1) @binding(2) var veil_buf: texture_2d<u32>;

// The scene clock (s, row 24.z). The live view advances it every frame,
// parked and paused included, and keeps it out of its accumulation: the
// march leaves the steam to fs_steam_live, drawn at this clock after the
// accumulation, so a held view converges under steam that moves. Every
// still holds it fixed; a recorded track carries it per sample (track.js),
// and `witness --time` sets it. Anything in the scene that moves reads it here (the
// city's steam); a surface where nothing moves (the sea) has no clock, and
// the hosts pack 0 for it.
fn scene_time() -> f32 {
    return u.surface_offset.z;
}
