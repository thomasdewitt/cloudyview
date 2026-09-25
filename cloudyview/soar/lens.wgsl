// Lens water: drops on the camera's lens after it flies through cloud.
//
// One look, drawn by both hosts AFTER accumulation — the browser's present
// pass (web/soar/renderer.js presentShader) and the Python host's post pass
// (cloudyview/soar_host.py _LENS_POST_SHADER) include this file — so the
// accumulated image never holds a drop, a parked view converges under a
// still lens, and `soar TRACK.json` draws what the browser drew.
//
// The includer provides:
//   fn lens_src(uv: vec2<f32>) -> vec3<f32>   the image at uv: both hosts
//       define it as lens_bilinear(<their source texture>, uv)
// and passes the lens state texel raymarch.wgsl's fs_lens_state wrote:
//   x wetness [0, 1], y the lens clock (s, folded at LENS_PERIOD_S),
//   zw "down" on the screen (unit, x right, y down).
//
// Wetness 0 returns the pixel untouched, bit for bit: a dry lens (every
// still, every capture, the toggle off) costs one compare.
//
// Everything is procedural and measured in screen heights, so the drops keep
// their size at any resolution. The beads sit on the lens (screen space);
// the runners run with gravity on the image plane ("down", st.zw) — today
// always straight down the screen, since soar's camera does not roll, and
// down whatever roll a camera might have.
// A drop is a small lens: it shows the scene behind it inverted and
// magnified, darkened and blurred at its rim where the water curves away.
// Each pixel looks only at its own cell of each layer (a drop never leaves
// its cell), so the cost is fixed and small.

const LENS_PERIOD_S_L: f32 = 600.0;    // raymarch.wgsl LENS_PERIOD_S
const LENS_BEAD_CELL: f32 = 0.085;     // screen heights: the large beads
const LENS_FINE_CELL: f32 = 0.038;     // the fine mist of small ones
const LENS_RUN_PITCH: f32 = 0.075;     // screen heights between drip columns
const LENS_RUN_R: f32 = 0.024;         // a running head's half-width
const LENS_RUN_WANDER: f32 = 0.004;    // how far a head wanders side to side
const LENS_TRAIL: f32 = 0.45;          // screen heights of trail behind it
const LENS_TRAIL_W: f32 = 0.006;       // the trail's half-width
const LENS_REFRACT: f32 = 1.6;         // how far a drop throws the view
const LENS_RIM_DARK: f32 = 0.45;       // rim darkening at a drop's edge
const LENS_GLINT: f32 = 0.35;          // the highlight on a drop's upper side

// Clamp-to-edge bilinear from an unfilterable float texture, by hand (core
// WebGPU cannot filter rgba32float): what each includer's lens_src reads
// the image through, so the two hosts throw the view identically.
fn lens_bilinear(src: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(src, 0));
    let p = uv * size - vec2<f32>(0.5);
    let base = floor(p);
    let f = p - base;
    let hi = vec2<i32>(size) - vec2<i32>(1);
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), hi);
    let i1 = clamp(vec2<i32>(base) + vec2<i32>(1), vec2<i32>(0), hi);
    let c00 = textureLoad(src, vec2<i32>(i0.x, i0.y), 0).rgb;
    let c10 = textureLoad(src, vec2<i32>(i1.x, i0.y), 0).rgb;
    let c01 = textureLoad(src, vec2<i32>(i0.x, i1.y), 0).rgb;
    let c11 = textureLoad(src, vec2<i32>(i1.x, i1.y), 0).rgb;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn lens_hash4(c: vec2<i32>, salt: u32) -> vec4<f32> {
    var v = vec4<u32>(bitcast<u32>(c.x), bitcast<u32>(c.y), salt,
                      bitcast<u32>(c.x) ^ bitcast<u32>(c.y));
    v = (v * 1664525u) + vec4<u32>(1013904223u);
    v.x = v.x + (v.y * v.w);
    v.y = v.y + (v.z * v.x);
    v.z = v.z + (v.x * v.y);
    v.w = v.w + (v.y * v.z);
    v = v ^ (v >> vec4<u32>(16u));
    v.x = v.x + (v.y * v.w);
    v.y = v.y + (v.z * v.x);
    v.z = v.z + (v.x * v.y);
    v.w = v.w + (v.y * v.z);
    return vec4<f32>(v) * (1.0 / 4294967296.0);
}

// One drop in the cell containing g (cell units): its refraction offset
// (cell units, in the turned frame) and how much of this pixel it covers.
struct LensDrop {
    off: vec2<f32>,
    cover: f32,
    rim: f32,
    glint: f32,
}

fn lens_bead(g: vec2<f32>, salt: u32, share: f32, grow: f32) -> LensDrop {
    var d: LensDrop;
    d.off = vec2<f32>(0.0);
    d.cover = 0.0;
    d.rim = 0.0;
    d.glint = 0.0;
    let ci = floor(g);
    let h = lens_hash4(vec2<i32>(ci), salt);
    if (h.x >= share) {
        return d;
    }
    // Inside the cell, so no neighbour ever needs looking at: centre in
    // [0.25, 0.75], semi-axes <= 0.21 across and 0.2415 down. Beads sag a
    // little: taller than wide.
    let c = vec2<f32>(0.25, 0.25) + 0.5 * h.yz;
    let r = (0.10 + 0.11 * h.w) * grow;
    let q = (g - ci - c) / vec2<f32>(r, 1.15 * r);
    let dd = dot(q, q);
    if (dd >= 1.0) {
        return d;
    }
    // A spherical cap: the view through it is inverted and thrown by the
    // slope, and the rim, where the surface turns away, goes dark.
    d.off = -q * (LENS_REFRACT * r);
    d.cover = 1.0 - smoothstep(0.85, 1.0, dd);
    d.rim = smoothstep(0.30, 1.0, dd);
    // The sky's reflection on the cap's upper side: a small soft spot.
    let gq = q - vec2<f32>(-0.25, -0.4);
    d.glint = 1.0 - smoothstep(0.0, 0.09, dot(gq, gq));
    return d;
}

// `uv` is the pixel's place on the image ([0, 1], y down), `aspect` the
// image's width / height and `texel` one source texel in uv: nothing here
// depends on the output's pixel count, so the browser's upscaling present
// and the Python post pass at the render size draw the same lens.
fn lens_apply(rgb: vec3<f32>, uv: vec2<f32>, aspect: f32, texel: vec2<f32>,
              st: vec4<f32>) -> vec3<f32> {
    let w = st.x;
    if (w <= 0.0) {
        return rgb;
    }
    // Screen heights about the image centre (x right, y down).
    let p = (uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);

    // The beads sit on the lens, so they live on the screen: they appear as
    // the lens wets, first the fine mist, then the large ones, and grow
    // toward their full size. Offsets here are screen heights.
    let grow = 0.55 + 0.45 * smoothstep(0.2, 0.9, w);
    let fine = lens_bead(p / LENS_FINE_CELL, 11u,
                         0.45 * smoothstep(0.02, 0.35, w), grow);
    let big = lens_bead(p / LENS_BEAD_CELL, 23u,
                        0.30 * smoothstep(0.15, 0.7, w), grow);
    var off = fine.off * LENS_FINE_CELL;
    var cover = fine.cover;
    var rim = fine.rim;
    var glint = fine.glint;
    if (big.cover > 0.0) {
        off = big.off * LENS_BEAD_CELL;
        cover = big.cover;
        rim = big.rim;
        glint = big.glint;
    }

    // The runners run with gravity, so they live in the frame turned to
    // "down" (st.zw): once the lens is properly wet, a few columns carry a
    // drop that runs down it, leaving a thin wet trail. A column's speed is
    // a whole number of passes per lens-clock period, so the pattern is
    // exactly periodic with the clock (it folds without a jump). A head
    // starts above the screen's top and wraps only once its trail has left
    // the bottom, over the screen's diagonal (whatever "down" is), so no
    // trail is on screen when its drop wraps.
    let down = st.zw;
    let side = vec2<f32>(-down.y, down.x);
    let t = vec2<f32>(dot(p, side), dot(p, down));
    let run_share = 0.35 * smoothstep(0.45, 0.95, w);
    let col = floor(t.x / LENS_RUN_PITCH);
    let hc = lens_hash4(vec2<i32>(i32(col), 7), 37u);
    if (hc.x < run_share) {
        let r = LENS_RUN_R;
        let half_diag = 0.5 * length(vec2<f32>(aspect, 1.0));
        let top = -half_diag - r / 0.65;
        let span = 2.0 * half_diag + LENS_TRAIL + 2.0 * r / 0.65;
        let laps = 35.0 + floor(25.0 * hc.y);         // passes per period
        let head = top + span * fract(hc.z + laps * st.y / LENS_PERIOD_S_L);
        // Inside its column with the wander and a margin to spare: a head
        // never reaches its column's edge, where the next column's pixels
        // would not see it.
        let x_lo = r + LENS_RUN_WANDER + 0.002;
        let x0 = (col * LENS_RUN_PITCH) + x_lo
            + (LENS_RUN_PITCH - 2.0 * x_lo) * hc.w;
        let xw = x0 + LENS_RUN_WANDER * sin(31.0 * t.y + 6.3 * hc.y);
        let q = vec2<f32>(t.x - xw, (t.y - head) * 0.65) / r;
        let dd = dot(q, q);
        var off_t = vec2<f32>(0.0);
        var hit = false;
        if (dd < 1.0) {
            off_t = -q * (LENS_REFRACT * r);
            cover = 1.0 - smoothstep(0.85, 1.0, dd);
            rim = smoothstep(0.30, 1.0, dd);
            let gq = q - vec2<f32>(-0.25, -0.4);
            glint = 1.0 - smoothstep(0.0, 0.09, dot(gq, gq));
            hit = true;
        } else {
            // The trail: behind the head, thin, fading along its length. A
            // wet streak is a thin cylinder lens: it throws the view across
            // itself, and it is darkest at its two edges.
            let behind = head - t.y;
            let v = (t.x - xw) / LENS_TRAIL_W;                  // -1 .. 1
            if (behind > 0.0 && behind < LENS_TRAIL && abs(v) < 1.0) {
                let along = 1.0 - behind / LENS_TRAIL;
                off_t = vec2<f32>(-v * 0.012, 0.0) * along;
                cover = max(cover,
                            0.9 * along * (1.0 - smoothstep(0.7, 1.0, abs(v))));
                rim = max(rim, (0.3 + 0.5 * v * v) * along);
                glint = 0.0;
                hit = true;
            }
        }
        if (hit) {
            off = off_t.x * side + off_t.y * down;        // back to the screen
        }
    }
    if (cover <= 0.0) {
        return rgb;
    }
    // The view through the water: the thrown sample and a small cross of
    // taps around it for the water's softness.
    let c = uv + off * vec2<f32>(1.0 / aspect, 1.0);
    let b = 1.2 * texel;
    let seen = 0.4 * lens_src(c)
        + 0.15 * (lens_src(c + vec2<f32>(b.x, 0.0))
                  + lens_src(c - vec2<f32>(b.x, 0.0))
                  + lens_src(c + vec2<f32>(0.0, b.y))
                  + lens_src(c - vec2<f32>(0.0, b.y)));
    let wet = seen * (1.0 - LENS_RIM_DARK * rim) + vec3<f32>(LENS_GLINT * glint);
    return mix(rgb, wet, cover);
}
