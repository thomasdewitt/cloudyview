// Ice-detection mode (false-color phase) ------------------------------------
//
// Ice extinction fraction at a world point, from the outer level's grid —
// the outer AABB covers any nest, so one texture serves both levels. Same
// coordinate convention as sample_light_tau: no edge taper (a fraction is
// not extinction; ramping it to zero at the boundary would paint the domain
// edge liquid-red), lateral wrap in a periodic domain, clamp vertically.
fn ice_fraction_at(p: vec3<f32>) -> f32 {
    let q = wrap_to_domain(p);
    let tex_dims = vec3<f32>(textureDimensions(ice_frac, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    let data_g = ((q - u.bmin.xyz) / (u.bmax.xyz - u.bmin.xyz)) * dims - 0.5;
    let tex_coord = vec3<f32>(
        data_g.z + 0.5,
        data_g.y + 0.5,
        data_g.x + 0.5
    ) / tex_dims;
    if (PERIODIC_DOMAIN) {
        return textureSampleLevel(ice_frac, vol_wrap_samp, tex_coord, 0.0).r;
    }
    return textureSampleLevel(ice_frac, vol_samp, tex_coord, 0.0).r;
}

// The false-color ramp: liquid burns crimson, ice glows cyan, and the mixed
// phase between them crosses through electric violet. NOT luminance-
// normalized: dividing by the hue's luminance blows the red channel far past
// the tone map's shoulder and deep red comes out pink. Max-component ~1
// instead, so red reads darker than cyan — deep red to vibrant cyan, which
// is the intended reading.
fn ice_tint(f_in: f32) -> vec3<f32> {
    let f = clamp(f_in, 0.0, 1.0);
    // Authored for the display: the tone map's gamma lifts weak channels,
    // so the linear palette is deeper than the hue it lands on.
    let c_liquid = vec3<f32>(1.00, 0.010, 0.030);
    let c_mixed  = vec3<f32>(0.60, 0.015, 1.00);
    let c_ice    = vec3<f32>(0.010, 0.85, 1.00);
    return mix(mix(c_liquid, c_mixed, smoothstep(0.0, 0.5, f)),
               c_ice, smoothstep(0.5, 1.0, f));
}

const ICE_LUM: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// The sea under the alien sky: its shaded luminance on a dark teal, so the
// glint and wave structure survive the recolor.
const ICE_OCEAN_TINT: vec3<f32> = vec3<f32>(0.04, 0.30, 0.24);

// A cloud source term recolored for the false-color view: its luminance —
// which carries all the shading, shadowing and silhouette structure — on the
// ramp's hue. With the mode off the term passes through untouched, so every
// call site is bit-identical to the pre-mode shader.
fn ice_recolor(x: vec3<f32>, tint: vec3<f32>, on: bool) -> vec3<f32> {
    if (on) {
        return dot(x, ICE_LUM) * tint;
    }
    return x;
}

// The alien sky: the physical sky's own luminance — gradients, circumsolar
// bloom, disc, haze behavior all intact — recolored onto a green palette,
// acid chartreuse at the horizon falling to deep emerald-teal overhead.
// Luminance-preserving, so exposure, tone map and aerial haze all keep
// working; `disc` is passed through so the aerial-perspective caller can
// exclude the solar disc exactly as it does from the physical sky.
fn alien_sky(dir: vec3<f32>, sun: vec3<f32>, disc: vec3<f32>) -> vec3<f32> {
    let phys = sky_radiance(dir, sun, u.sky_horizon.xyz, u.sky_bloom.xyz,
                            disc, u.cloud_sun.w);
    let lum = dot(phys, ICE_LUM);
    let up = clamp(dir.z, 0.0, 1.0);
    // Dimmer than the physical sky it replaces (the ramp luminances are
    // ~0.7 and ~0.3, not 1), so the red/cyan clouds carry the frame.
    let horizon = vec3<f32>(0.35, 0.90, 0.10);
    let zenith  = vec3<f32>(0.01, 0.38, 0.26);
    return lum * mix(horizon, zenith, pow(up, 0.45));
}
