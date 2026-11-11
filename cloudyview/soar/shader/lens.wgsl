// Lens water: small drops on the camera's lens after it flies through cloud.
//
// This file is not part of the march's module. Each host includes it in the
// pass that draws the finished image (the browser's present pass, the Python
// host's post pass), after the accumulation. The accumulated image therefore
// never holds a drop, a parked view converges under the lens whatever the
// lens does, and the two hosts draw the same lens.
//
// The includer provides
//   fn lens_src(uv: vec2<f32>) -> vec3<f32>   the image at uv, defined in
//       both hosts as lens_bilinear(<their source texture>, uv)
// and passes the state texel that fs_lens_state (passes.wgsl) wrote, of
// which only x, the wetness in [0, 1], is read here.
//
// Wetness 0 returns the pixel untouched, so a dry lens costs one compare.
//
// The drops are small beads that neither grow nor run. A wetter lens carries
// more of them: two interleaved layers fill in as the wetness rises.
// Everything is procedural and measured in screen heights, so a bead keeps
// its size at any resolution. A bead is a small lens: it shows the scene
// behind it inverted, darkened at its rim where the water curves away, with
// a glint on its upper side. Each pixel looks only at its own cell of each
// layer (a bead never leaves its cell), so the cost is fixed.

const LENS_CELL_A: f32 = 0.038;        // screen heights: the first layer
const LENS_CELL_B: f32 = 0.029;        // the second, filling in when wetter
const LENS_BEAD_R: f32 = 0.07;         // a bead's radius, in cells (+ up to
const LENS_BEAD_R_VAR: f32 = 0.08;     // this much more)
const LENS_REFRACT: f32 = 1.6;         // how far a bead throws the view
const LENS_RIM_DARK: f32 = 0.45;       // rim darkening at a bead's edge
const LENS_GLINT: f32 = 0.35;          // the highlight on a bead's upper side

// Clamp-to-edge bilinear read of an unfilterable float texture, done by hand
// (core WebGPU cannot filter rgba32float). Each includer's lens_src reads
// the image through it, so the two hosts refract the view identically.
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

// Four uniforms in [0, 1] from a cell index and a salt (an integer hash).
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

// One bead in the cell containing g (cell units): its refraction offset
// (cell units), how much of this pixel it covers, its rim and its glint.
struct LensDrop {
    off: vec2<f32>,
    cover: f32,
    rim: f32,
    glint: f32,
}

// The bead of layer `salt` in g's cell, if the cell has one: a fraction
// `share` of the cells do.
fn lens_bead(g: vec2<f32>, salt: u32, share: f32) -> LensDrop {
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
    // The bead stays inside its cell, so no neighbouring cell ever needs
    // looking at: centre in [0.25, 0.75], semi-axes at most 0.15 across and
    // 0.1725 down. A bead sags a little, so it is taller than wide.
    let c = vec2<f32>(0.25, 0.25) + 0.5 * h.yz;
    let r = LENS_BEAD_R + LENS_BEAD_R_VAR * h.w;
    let q = (g - ci - c) / vec2<f32>(r, 1.15 * r);
    let dd = dot(q, q);
    if (dd >= 1.0) {
        return d;
    }
    // A spherical cap: the view through it is inverted and displaced in
    // proportion to the slope, and the rim, where the surface turns away,
    // goes dark.
    d.off = -q * (LENS_REFRACT * r);
    d.cover = 1.0 - smoothstep(0.85, 1.0, dd);
    d.rim = smoothstep(0.30, 1.0, dd);
    // The sky's reflection on the cap's upper side: a small soft spot.
    let gq = q - vec2<f32>(-0.25, -0.4);
    d.glint = 1.0 - smoothstep(0.0, 0.09, dot(gq, gq));
    return d;
}

// `uv` is the pixel's place on the image ([0, 1], y down), `aspect` the
// image's width / height and `texel` one source texel in uv. Nothing here
// depends on the output's pixel count, so a present pass that upscales and a
// post pass at the render size draw the same lens.
fn lens_apply(rgb: vec3<f32>, uv: vec2<f32>, aspect: f32, texel: vec2<f32>,
              st: vec4<f32>) -> vec3<f32> {
    let w = st.x;
    if (w <= 0.0) {
        return rgb;
    }
    // Screen heights about the image centre (x right, y down).
    let p = (uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
    // The first layer appears as the lens wets; the second fills in between
    // as it gets wetter. The beads' size never changes.
    let a = lens_bead(p / LENS_CELL_A, 11u, 0.45 * smoothstep(0.02, 0.45, w));
    let b = lens_bead(p / LENS_CELL_B, 29u, 0.40 * smoothstep(0.35, 0.95, w));
    var d = a;
    var cell = LENS_CELL_A;
    if (b.cover > a.cover) {
        d = b;
        cell = LENS_CELL_B;
    }
    if (d.cover <= 0.0) {
        return rgb;
    }
    // The view through the water: the displaced sample, and a small cross
    // of taps around it for the water's softness.
    let c = uv + (d.off * cell) * vec2<f32>(1.0 / aspect, 1.0);
    let t = 1.2 * texel;
    let seen = 0.4 * lens_src(c)
        + 0.15 * (lens_src(c + vec2<f32>(t.x, 0.0))
                  + lens_src(c - vec2<f32>(t.x, 0.0))
                  + lens_src(c + vec2<f32>(0.0, t.y))
                  + lens_src(c - vec2<f32>(0.0, t.y)));
    let wet = seen * (1.0 - LENS_RIM_DARK * d.rim)
        + vec3<f32>(LENS_GLINT * d.glint);
    return mix(rgb, wet, d.cover);
}
