// Highlight shoulder: how far bright regions blend from per-channel
// Reinhard toward a luminance-normalized curve. Per-channel compression
// desaturates highlights toward grey; the luminance-preserving branch
// keeps the beam's chroma, so lit faces read vibrant rather than chalky.
// Gated on luminance so sky and shadow keep the per-channel curve exactly.
const TONE_MAP_SHOULDER: f32 = 0.35;

// Reinhard + gamma, matching radiative_transfer.tone_map (lines 675-680)
// with witness's default exposure=4.0 (witness.py lines 955-959, 1072-1073).
// Gamma arrives per-frame (u.periodic.w) rather than as the GAMMA const:
// it is the one knob that decides how much the far field lifts, and it is
// the only place the encode may happen — the swapchain must be a plain
// unorm format, never *-srgb, or this runs a second time.
// Extended Reinhard: plain e/(1+e) crushes lit cloud faces into a grey
// ceiling (a face at exposed radiance ~4.7 displays at 227/255 and cannot
// move). The white point (u.flags.w, user slider) is the exposed radiance
// that reaches 1.0, so sunlit faces read vibrant white; below ~10% of it
// the curve is within 0.3% of plain Reinhard, so sky and shadow are
// untouched. Contrast (u.ocean_realism_b.w) pivots the encoded value on
// mid-grey; at 1.0 the multiply-add is the exact identity.

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

// tone_map run backwards: the linear radiance a displayed value came from,
// for the live steam (fs_steam_live), which composites in linear light onto
// an image already accumulated in display values. Exact wherever tone_map
// is invertible (tests/test_tone_unmap.py holds the round trip on the GPU);
// what its clamps threw away does not come back: a channel whose curve
// reached 1 (at the white point, or below it for a saturated colour in the
// shoulder, whose chroma-preserving branch overshoots) returns where it
// reached it, and under contrast > 1 a value clipped to 0 or 1 returns the
// clip's edge. Below the shoulder (exposed luminance
// y < 1) the curve is per channel and inverts in closed form. In it, y fixes
// the shoulder's mix k and chroma gain g and with them every channel, so
// the solve is for the one y that reproduces itself, by bisection: a fixed
// point iterated on y can oscillate there (codex, 2026-11-08).
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
    // y lies in [1, y_hi]: every channel's m <= 1, and with k at most
    // TONE_MAP_SHOULDER the per-channel part alone caps e where the plain
    // curve reaches 1 / (1 - TONE_MAP_SHOULDER).
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

// Each channel's exposed e with (1 - k) e (1 + e/w2)/(1 + e) + k g e = m,
// the shoulder's mix k and chroma gain g held fixed: a e^2 + b e - m = 0,
// a = (1 - k)/w2 + k g > 0, b = (1 - k) + k g - m. Its root e >= 0 in the
// form that does not cancel for either sign of b (the plain one lost a dark
// value to 0 in float32, codex 2026-11-08).
fn tone_unmap_channels(m: vec3<f32>, k: f32, g: f32, w2: f32) -> vec3<f32> {
    let a = (1.0 - k) / w2 + k * g;
    let b = vec3<f32>((1.0 - k) + k * g) - m;
    // @domain: a square plus 4 a m, with a > 0 and m >= 0
    let q = sqrt(b * b + 4.0 * a * m);
    return select((q - b) / (2.0 * a), 2.0 * m / (q + b), b > vec3<f32>(0.0));
}
