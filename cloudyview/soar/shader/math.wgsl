// Shared math: the phase function, the hashes, the stratified sample streams
// and the GGX masking term.

// Henyey-Greenstein phase function (sr^-1; integrates to 1 over the sphere):
//   p(mu) = (1 - g^2) / (4 pi (1 + g^2 - 2 g mu)^(3/2)),
// mu the cosine of the scattering angle, g the asymmetry parameter.
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (12.566370614 * pow(max(denom, 1e-6), 1.5));
}

// White-noise hash of a float pair to [0, 1) (Hoskins' hash12). The cloud
// renderer's own draws come from strat1 / strat2 below; this one serves the
// city.
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Two independent hash12 values.
fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        hash12(p + vec2<f32>(17.17, 41.93)),
        hash12(p + vec2<f32>(71.31, 11.57))
    );
}

// ---------------------------------------------------------------------------
// Stratified draws over the accumulation.
//
// Several quantities are estimated by Monte Carlo with one draw per pixel per
// frame: the sub-pixel camera offset, the view march's entry phase, the sun
// march's quadrature phase, the sky probe's and the forward pre-march's
// quadrature phases, the direction within the solar disc, and the phase of
// the view march where it leaves a nest. A pixel's N accumulated frames are N
// samples of one fixed integral, so the draws form a low-discrepancy point
// set rather than white noise: a rank-1 lattice, the frame index times a
// generating vector, wrapped to [0, 1) and shifted by a random offset per
// pixel (a Cranley-Patterson rotation):
//
//     x_j(frame) = fract(frame * alpha_j + rotation(pixel, stream))
//
// The generating vector is Roberts' generalised golden ratio in ten
// dimensions, alpha_j = phi_10^-j with phi_10 the real root of x^11 = x + 1.
// Each stream takes a component of it.
//
// All streams share the one lattice, indexed by the one frame counter. A
// pixel's radiance depends on the entry phase, the sun-march phase and the
// sub-pixel offset jointly, so it is the joint space that has to be covered
// evenly. An independent low-discrepancy sequence per stream would stratify
// each margin and leave the joint distribution random (Latin-hypercube
// sampling), which removes only the variance that is additive across
// streams, and little of this renderer's is.
//
// What the construction keeps:
//   * It is unbiased. The rotation is uniform, so every single frame is a
//     uniform draw and the converged image is the one white noise would
//     converge to.
//   * Within one frame, neighbouring pixels are decorrelated: the rotation
//     is per pixel and the lattice term is common to the frame. At frame 0
//     the lattice term is zero and each draw is its rotation alone.
//   * It is progressive. Every prefix of an additive recurrence is well
//     distributed, so N need not be known in advance, and the live view
//     accumulates for as long as the camera is parked.
//
// The rotations come from an integer hash of the pixel and the stream
// (pcg2d). A float hash such as hash12 does not produce enough distinct
// values to give every pixel of a frame its own rotation, and two pixels
// that share a rotation draw the identical sequence at every frame.
fn pcg2d(v_in: vec2<u32>) -> vec2<u32> {
    var v = v_in * 1664525u + vec2<u32>(1013904223u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return v;
}

// The per-pixel, per-stream seed of the rotation.
fn strat_seed(pixel: vec2<u32>, stream: u32) -> vec2<u32> {
    return pcg2d(pixel + vec2<u32>(stream * 0x9e3779b9u,
                                   stream * 0x85ebca6bu));
}

// Stream indices. They need only be distinct: each selects its own rotation,
// so two streams that share a lattice component (the city probe and the nest
// seam both take STRAT_ALPHA_10) still land on unrelated offsets.
const STRAT_STREAM_SUBPIXEL: u32 = 1u;
const STRAT_STREAM_ENTRY: u32 = 2u;
const STRAT_STREAM_CITY_PROBE: u32 = 3u;
const STRAT_STREAM_SUN_MARCH: u32 = 4u;
const STRAT_STREAM_SUN_CONE: u32 = 5u;
const STRAT_STREAM_PREMARCH: u32 = 7u;
const STRAT_STREAM_NEST_SEAM: u32 = 8u;

const STRAT_INV_2P32: f32 = 2.3283064365386963e-10; // 2^-32

// The generating vector: alpha_j = phi_10^-j, phi_10 the real root of
// x^11 = x + 1 (Roberts 2018). Which component a stream takes is arbitrary:
// every component is an equally good 1D generator, and the joint quality is
// a property of the vector as a whole.
const STRAT_ALPHA_1: f32 = 0.9360691110777584;
const STRAT_ALPHA_2: f32 = 0.8762253807139048;
const STRAT_ALPHA_3: f32 = 0.8202075132286352;
const STRAT_ALPHA_4: f32 = 0.7677709178072273;
const STRAT_ALPHA_5: f32 = 0.7186866405431660;
const STRAT_ALPHA_6: f32 = 0.6727403647567018;
const STRAT_ALPHA_9: f32 = 0.5517867016256194;
const STRAT_ALPHA_10: f32 = 0.5165104872952219;

// One draw in [0, 1) for `stream`, at accumulation frame `index`.
fn strat1(index: u32, pixel: vec2<u32>, stream: u32, alpha: f32) -> f32 {
    let seed = strat_seed(pixel, stream);
    return fract(f32(index) * alpha + f32(seed.x) * STRAT_INV_2P32);
}

// Two draws at once: `alpha` holds two components of the generating vector.
fn strat2(index: u32, pixel: vec2<u32>, stream: u32,
          alpha: vec2<f32>) -> vec2<f32> {
    let seed = strat_seed(pixel, stream);
    return fract(f32(index) * alpha
                 + vec2<f32>(seed) * STRAT_INV_2P32);
}

// Smith masking term for a GGX microfacet distribution:
//   G1(x) = 2 (n.x) / ((n.x) + sqrt(alpha^2 + (1 - alpha^2) (n.x)^2)),
// alpha the GGX roughness and x the view or the light direction.
fn ggx_smith_g1(n_dot_x: f32, alpha_squared: f32) -> f32 {
    // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
    let root = sqrt(alpha_squared + (1.0 - alpha_squared) * n_dot_x * n_dot_x);
    return (2.0 * n_dot_x) / (n_dot_x + root);
}

// A u32 as a float in [0, 1]: x / 2^32. Used by the city.
fn city_u01(x: u32) -> f32 {
    return f32(x) * 2.3283064365386963e-10; // 2^-32
}
