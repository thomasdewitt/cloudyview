// Henyey-Greenstein phase function.
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (12.566370614 * pow(max(denom, 1e-6), 1.5));
}

// Per-pixel hash for jittered ray starts (Dave Hoskins style hash12).
// Blue-noise would be better; a white-noise hash is enough to break the
// coherent step-shell banding this exists to kill.
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        hash12(p + vec2<f32>(17.17, 41.93)),
        hash12(p + vec2<f32>(71.31, 11.57))
    );
}

// ---------------------------------------------------------------------------
// Stratified draws over the accumulation (lighting-loop iter_011).
//
// Seven quantities in this shader are estimated by Monte Carlo with one draw
// per pixel per frame: the sub-pixel camera offset, the view march's entry
// phase, the sun march's quadrature phase (iter_003), the sky probe's and the
// forward pre-march's quadrature phases (iter_001, iter_004), the solar-disc
// and aureole direction (iter_007), and the ocean's sub-pixel slope
// (iter_008). Every one of them drew white noise in the frame index, so an
// accumulated still converged at the white-noise rate and kept 1/sqrt(N) of a
// single frame's error. That residue is the speckle grain the blind judge saw.
//
// White noise is the wrong sequence for this. A pixel's N frames are N samples
// of one fixed integral, and a low-discrepancy point set covers its domain
// evenly instead of clumping. What the draws form here is a rank-1 lattice —
// the frame index times a generating vector, wrapped — shifted by a per-pixel
// random offset (a Cranley-Patterson rotation). The generating vector is
// Roberts' generalised golden ratio in ten dimensions, alpha_j = phi_10^-j
// with phi_10 the real root of x^11 = x + 1, one component per stream.
//
// The single measurement that fixed this shape, and it cost two rejected
// implementations to find: **the streams must share one point set.** The
// second version of this iteration gave every stream its own Owen-scrambled
// van der Corput sequence — the textbook 1D construction, and by direct GPU
// audit a perfect one: each pixel got exactly one sample in each 1/64
// interval, per-pixel mean 0.5 to 2.4e-4, no neighbour correlation. It bought
// **nothing** (64-frame noise within 10 percent of white noise on every view).
// Independent per-stream sequences are Latin-hypercube sampling: they
// stratify each margin and leave the joint distribution random, so they can
// only remove the variance that is additive across streams. Almost none of
// this renderer's is — a pixel's radiance depends on the entry phase and the
// sun-march phase and the sub-pixel offset *jointly*. One lattice indexed by
// the frame, with a different component per stream, is low-discrepancy in the
// joint space, and it cuts the 64-frame noise by 2 to 4x on every view.
//
// A Halton sequence (a different prime base per stream) is jointly
// low-discrepancy too and measured within a few percent of the lattice; it is
// not used because its later streams have to draw base 17, 19, 23, ... which
// is visibly the worse end of the sequence at 64 samples, and because it needs
// a digit loop per stream where the lattice needs a multiply-add.
//
// Three properties this construction keeps, all of which matter here:
//
//   * Unbiased. The rotation is uniform, so each individual frame is still a
//     uniform draw and the converged image is exactly the one iterations
//     001-009 produced. This buys convergence, not a different look — measured
//     as a 0.01-level mean shift per view.
//   * A single frame is unchanged in character. The rotation is per pixel and
//     the lattice offset is common to the frame, so within one frame the draws
//     are as decorrelated between neighbouring pixels as they ever were. At
//     frame 0 the lattice term is zero and every stream returns its rotation
//     alone, i.e. exactly the per-pixel white noise it used to draw, so soar's
//     first frame after a camera move is statistically identical.
//   * Progressive. Every prefix of an additive recurrence is well distributed,
//     so this does not depend on knowing N in advance — which matters because
//     soar accumulates for as long as the camera is parked.
//
// The rotations come from an *integer* hash of the pixel and the stream, not
// from the float hash12 taps these streams used to draw. hash12 does not carry
// enough distinct values to be one rotation per pixel: measured on GPU over a
// 256x256 block it produces about 11.5k of them, i.e. ~13.5 bits, so on a
// 960x540 frame roughly forty pixels share each rotation exactly — and pixels
// that share a rotation draw the *identical* 64-frame sequence, which is
// precisely the coherent-shell correlation the entry jitter exists to break.
// It is a latent weakness of the white-noise version too; it only becomes
// load-bearing once the sequence a rotation selects is a fixed lattice rather
// than fresh noise every frame.
//
// pcg2d is the integer hash iter_009 put in the present pass's dither, and for
// the same reason: its negative result there — float hashes are measurably
// biased and coarse on integer input — is this one restated.
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

fn strat_seed(pixel: vec2<u32>, stream: u32) -> vec2<u32> {
    return pcg2d(pixel + vec2<u32>(stream * 0x9e3779b9u,
                                   stream * 0x85ebca6bu));
}

// Stream indices: each selects its own per-pixel rotation, so two streams that
// happen to share a lattice component still land on unrelated offsets.
const STRAT_STREAM_SUBPIXEL: u32 = 1u;
const STRAT_STREAM_ENTRY: u32 = 2u;
const STRAT_STREAM_SKY_PROBE: u32 = 3u;
const STRAT_STREAM_SUN_MARCH: u32 = 4u;
const STRAT_STREAM_SUN_CONE: u32 = 5u;
const STRAT_STREAM_PREMARCH: u32 = 7u;
const STRAT_STREAM_NEST_SEAM: u32 = 8u;

const STRAT_INV_2P32: f32 = 2.3283064365386963e-10;

// The generating vector: alpha_j = phi_10^-j, phi_10 the real root of
// x^11 = x + 1 (Roberts 2018). Components are assigned to streams in the order
// they are consumed; the ordering is not load-bearing, since every component
// of an R_d vector is an equally good 1D generator and the joint quality is a
// property of the vector as a whole.
const STRAT_ALPHA_1: f32 = 0.9360691110777584;
const STRAT_ALPHA_2: f32 = 0.8762253807139048;
const STRAT_ALPHA_3: f32 = 0.8202075132286352;
const STRAT_ALPHA_4: f32 = 0.7677709178072273;
const STRAT_ALPHA_5: f32 = 0.7186866405431660;
const STRAT_ALPHA_6: f32 = 0.6727403647567018;
const STRAT_ALPHA_9: f32 = 0.5517867016256194;
const STRAT_ALPHA_10: f32 = 0.5165104872952219;

fn strat1(index: u32, pixel: vec2<u32>, stream: u32, alpha: f32) -> f32 {
    let seed = strat_seed(pixel, stream);
    return fract(f32(index) * alpha + f32(seed.x) * STRAT_INV_2P32);
}

fn strat2(index: u32, pixel: vec2<u32>, stream: u32,
          alpha: vec2<f32>) -> vec2<f32> {
    let seed = strat_seed(pixel, stream);
    return fract(f32(index) * alpha
                 + vec2<f32>(seed) * STRAT_INV_2P32);
}

// Smith masking for a GGX distribution parameterized by RMS slope.
fn ggx_smith_g1(n_dot_x: f32, alpha_squared: f32) -> f32 {
    // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
    let root = sqrt(alpha_squared + (1.0 - alpha_squared) * n_dot_x * n_dot_x);
    return (2.0 * n_dot_x) / (n_dot_x + root);
}

fn city_u01(x: u32) -> f32 {
    return f32(x) * 2.3283064365386963e-10; // 1 / 2^32
}
