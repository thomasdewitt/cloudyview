// Ice-detection mode: a false-colour view of cloud phase. Cloud source terms
// keep their luminance and take a hue from the local ice fraction, the sky
// is redrawn green and the sea dark teal, so the red-to-cyan clouds carry
// the frame. Everything here runs only when u.light_cache.w is set.

// Ice fraction of extinction at a world point, from the outer level's grid.
// The outer box covers any nest, so one texture serves both levels.
// Coordinates as in sample_light_tau: lateral wrap in a periodic domain,
// clamp on the vertical, and no edge taper (a fraction is not extinction,
// and ramping it to zero at the boundary would colour the domain's edge as
// liquid).
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

// The false-colour ramp: liquid crimson, ice cyan, mixed phase through
// violet. It is normalized to a maximum component of ~1 and not to unit
// luminance: dividing by a deep red's luminance drives the red channel far
// past the tone curve's shoulder, and it displays as pink. Red therefore
// reads darker than cyan.
fn ice_tint(f_in: f32) -> vec3<f32> {
    let f = clamp(f_in, 0.0, 1.0);
    // Chosen for the displayed colour: the tone map's gamma lifts weak
    // channels, so the linear palette is deeper than the hue it lands on.
    let c_liquid = vec3<f32>(1.00, 0.010, 0.030);
    let c_mixed  = vec3<f32>(0.60, 0.015, 1.00);
    let c_ice    = vec3<f32>(0.010, 0.85, 1.00);
    return mix(mix(c_liquid, c_mixed, smoothstep(0.0, 0.5, f)),
               c_ice, smoothstep(0.5, 1.0, f));
}

const ICE_LUM: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722); // Rec. 709

// The sea in this view: its shaded luminance on a dark teal, which keeps
// the glint and the wave structure.
const ICE_OCEAN_TINT: vec3<f32> = vec3<f32>(0.04, 0.30, 0.24);

// A cloud source term recoloured: its luminance, which carries the shading,
// the shadows and the silhouette, on the hue `tint`. With the mode off the
// term is returned as it came.
fn ice_recolor(x: vec3<f32>, tint: vec3<f32>, on: bool) -> vec3<f32> {
    if (on) {
        return dot(x, ICE_LUM) * tint;
    }
    return x;
}

// The sky in this view: the physical sky's luminance (gradient, circumsolar
// lobe, bloom, disc and haze behaviour) on a green palette, chartreuse at
// the horizon falling to a deep teal overhead. `disc` is passed through so
// that a caller can leave the solar disc out, as with sky_radiance.
fn alien_sky(dir: vec3<f32>, sun: vec3<f32>, disc: vec3<f32>) -> vec3<f32> {
    let phys = sky_radiance(dir, sun, u.sky_horizon.xyz, u.sky_bloom.xyz,
                            disc, u.cloud_sun.w);
    let lum = dot(phys, ICE_LUM);
    let up = clamp(dir.z, 0.0, 1.0);
    // Dimmer than the physical sky: the palette's luminances are ~0.7 and
    // ~0.3, not 1.
    let horizon = vec3<f32>(0.35, 0.90, 0.10);
    let zenith  = vec3<f32>(0.01, 0.38, 0.26);
    return lum * mix(horizon, zenith, pow(up, 0.45));
}
