// The tone map, linear radiance to display values, and its inverse.
//
// With e = exposure * radiance, w the white point (u.flags.w) and
// y = luminance(e), the curve is extended Reinhard,
//
//     per channel:        m = e (1 + e / w^2) / (1 + e)
//     chroma preserving:  m = e (1 + y / w^2) / (1 + y)
//
// mixed by k = TONE_MAP_SHOULDER * smoothstep(1, 3, y), then
//
//     display = 0.5 + c (clamp(m, 0, 1)^(1 / gamma) - 0.5),
//
// c the contrast (u.ocean_realism_b.w; 1.0 is the identity). The white point
// is the exposed radiance that displays as 1.0: plain Reinhard e / (1 + e)
// only reaches 1 at infinity, which holds sunlit cloud faces under a grey
// ceiling. For e << w^2 the curve is plain Reinhard.
//
// The gamma (u.periodic.w) is the only display encoding there is. The
// output target must be a plain unorm format, never an sRGB one, or the
// encoding would be applied twice.

// How far bright regions (y > 1) move from the per-channel curve toward the
// chroma-preserving one. Per-channel compression desaturates highlights
// toward grey; the other keeps the beam's colour in them. Sky and shadow
// (y < 1) take the per-channel curve alone.
const TONE_MAP_SHOULDER: f32 = 0.35;

fn tone_map(hdr: vec3<f32>, exposure: f32, gamma: f32) -> vec3<f32> {
    let exposed = hdr * exposure;
    let wp = u.flags.w;
    let w2 = wp * wp;
    let per_channel = exposed * (1.0 + exposed / w2) / (1.0 + exposed);
    let y = dot(exposed, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma_preserving = exposed * (1.0 + y / w2) / (1.0 + y);
    let k = TONE_MAP_SHOULDER * smoothstep(1.0, 3.0, y);
    let mapped = mix(per_channel, chroma_preserving, k);
    let encoded = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)),
                      vec3<f32>(1.0 / gamma));
    let c = u.ocean_realism_b.w;
    return clamp(vec3<f32>(0.5) + (encoded - vec3<f32>(0.5)) * c,
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// tone_map inverted: the linear radiance a display value came from. The live
// steam (fs_steam_live) needs it to composite in linear light onto an image
// accumulated in display values.
//
// It is exact wherever tone_map is invertible. What tone_map's clamps
// discarded does not come back: a channel whose curve reached 1 (at the
// white point, or below it for a saturated colour in the shoulder, where the
// chroma-preserving curve overshoots) returns the radiance at which it
// reached 1, and under contrast > 1 a value clipped to 0 or 1 returns the
// edge of the clip.
//
// Below the shoulder (y < 1) the curve is per channel and inverts in closed
// form. In the shoulder, y fixes the mix k and the chroma gain
// g = (1 + y / w^2) / (1 + y), and with them every channel, so the unknown
// is the one y that reproduces itself. It is found by bisection, because a
// fixed-point iteration on y can oscillate there.
fn tone_unmap(display: vec3<f32>, exposure: f32, gamma: f32) -> vec3<f32> {
    let c = u.ocean_realism_b.w;
    let encoded = clamp((display - vec3<f32>(0.5)) / c + vec3<f32>(0.5),
                        vec3<f32>(0.0), vec3<f32>(1.0));
    let m = pow(encoded, vec3<f32>(gamma));
    let w2 = u.flags.w * u.flags.w;
    let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
    let e0 = tone_unmap_channels(m, 0.0, 1.0, w2);
    if (dot(e0, lum) < 1.0) {
        return e0 / exposure;
    }
    // y lies in [1, y_hi]. Every channel has m <= 1, and with k at most
    // TONE_MAP_SHOULDER the per-channel part alone bounds e by the value at
    // which the plain curve reaches 1 / (1 - TONE_MAP_SHOULDER).
    var lo = 1.0;
    var hi = tone_unmap_channels(vec3<f32>(1.0 / (1.0 - TONE_MAP_SHOULDER)),
                                 0.0, 1.0, w2).x;
    for (var i: i32 = 0; i < 24; i = i + 1) {
        let y = 0.5 * (lo + hi);
        let k = TONE_MAP_SHOULDER * smoothstep(1.0, 3.0, y);
        let e = tone_unmap_channels(m, k, (1.0 + y / w2) / (1.0 + y), w2);
        if (dot(e, lum) > y) {
            lo = y;
        } else {
            hi = y;
        }
    }
    let y = 0.5 * (lo + hi);
    let k = TONE_MAP_SHOULDER * smoothstep(1.0, 3.0, y);
    return tone_unmap_channels(m, k, (1.0 + y / w2) / (1.0 + y), w2)
        / exposure;
}

// Each channel's exposed e from (1 - k) e (1 + e/w2)/(1 + e) + k g e = m,
// with the shoulder's mix k and chroma gain g held fixed. Multiplying through
// by (1 + e) gives a e^2 + b e - m = 0 with a = (1 - k)/w2 + k g > 0 and
// b = (1 - k) + k g - m. The root e >= 0 is taken in the form that does not
// cancel for either sign of b; the textbook form loses a dark value to 0 in
// float32.
fn tone_unmap_channels(m: vec3<f32>, k: f32, g: f32, w2: f32) -> vec3<f32> {
    let a = (1.0 - k) / w2 + k * g;
    let b = vec3<f32>((1.0 - k) + k * g) - m;
    // @domain: a square plus 4 a m, with a > 0 and m >= 0
    let q = sqrt(b * b + 4.0 * a * m);
    return select((q - b) / (2.0 * a), 2.0 * m / (q + b), b > vec3<f32>(0.0));
}
