// Reprojected history: what the picture is made of while the camera moves.
//
// A march is one noisy sample of the image, at the tier's render size. A
// still camera averages them in place (renderer.js ACCUM_SHADER). A moving
// one cannot: the pixel a ray lands on changes from frame to frame. This
// pass keeps a history image at its own, finer size and, each frame,
//
//   1. finds where each history pixel's content was in the previous frame,
//      from the two cameras and the march's motion guide (march.wgsl
//      MarchOut.guide: the mean inverse distance of what the pixel shows),
//   2. reads the history there,
//   3. blends this frame's samples into it, each at the place its jittered
//      ray actually went, so that samples from many frames fill a grid finer
//      than any one of them.
//
// The geometry of a frame is still the march's: every sample is a ray through
// the field from this frame's camera. What is approximate is the history,
// and how long it is believed is decided per pixel (see `life` in the
// shader):
//   - A rotation moves every ray's whole picture together, exactly.
//   - A translation moves each contribution along a ray by its own inverse
//     distance. One motion per pixel fits them only as well as their inverse
//     distances agree, and the guide's variance says how well that is. The
//     history is kept for as long as the misfit stays under
//     REPROJECT_DRIFT_BUDGET_PX, and never longer than REPROJECT_LIFETIME_S.
//   - History that disagrees with what this frame's samples around it show
//     is pulled into their range before it is used.
// Where nothing can be kept, the pixel is this frame's samples alone, which
// is what a pixel of the unreprojected renderer is.
//
// A parked view goes through the same pass (Renderer._reprojectionPlan): the
// two cameras are then one, every history pixel is exactly where it was, and
// the pass is a plain running average on the history's grid, begun from
// whatever the flight left there.

"use strict";

import * as K from "./constants.js";

/** The march's guide target, and the history's second attachment
 *  (inverse distance, weight). */
export const GUIDE_FORMAT = "rgba16float";
export const HISTORY_AUX_FORMAT = "rg16float";

/** The unit of distance (m) the march's guide is written in: march.wgsl's
 *  GUIDE_UNIT_M, which a test holds this to. */
export const GUIDE_UNIT_M = 100.0;

/** Floats in the resolve pass's uniform block (ResolveUniforms). */
export const RESOLVE_UNIFORM_FLOATS = 44;

/**
 * The resolve shader. `mathSource` is shader/math.wgsl, whose strat2 gives
 * each march pixel's sub-pixel offset: the resolve recomputes the offset the
 * march used (camera.wgsl camera_ray) to know where each sample is.
 */
export const resolveShader = (mathSource) => `
${mathSource}

struct ResolveUniforms {
    // This frame's camera, as rows 0-3 of the march's block hold it.
    cam_origin: vec3<f32>,
    tan_half_fov: f32,
    cam_forward: vec3<f32>,
    aspect: f32,
    cam_right: vec3<f32>,
    frame_index: f32,
    cam_up: vec3<f32>,
    jitter_scale: f32,
    // This camera's position minus the previous frame's, in the guide's
    // unit of distance (GUIDE_UNIT_M), and the previous camera.
    delta_g: vec3<f32>,
    prev_tan_half_fov: f32,
    prev_forward: vec3<f32>,
    history_valid: f32,
    prev_right: vec3<f32>,
    life_frames: f32,
    prev_up: vec3<f32>,
    splat_sigma: f32,
    render_size: vec2<f32>,
    history_size: vec2<f32>,
    clamp_gamma: f32,
    drift_budget: f32,
    fill_weight: f32,
    subpixel: f32,
    // History pixels per radian at the image centre.
    px_per_rad: f32,
    // The most weight (samples) the history is carried in at.
    count_cap: f32,
    // 1.0 reads the history's colour through the Catmull-Rom kernel, 0.0
    // bilinearly.
    sharp_history: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> ru: ResolveUniforms;
@group(0) @binding(1) var sample_tex: texture_2d<f32>;
@group(0) @binding(2) var guide_tex: texture_2d<f32>;
@group(0) @binding(3) var hist_color: texture_2d<f32>;
@group(0) @binding(4) var hist_aux: texture_2d<f32>;

struct ResolveOut {
    @location(0) color: vec4<f32>,
    // x: the mean inverse distance used (guide units). y: the history's
    // weight, in samples.
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// The Catmull-Rom weights of the four texels around a point a fraction f
// past the second of them.
fn catmull_rom(f: f32) -> vec4<f32> {
    let f2 = f * f;
    let f3 = f2 * f;
    return vec4<f32>(
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2);
}

// The fill's extra weight (samples) at a history that lives one frame; it
// falls linearly to nothing at one that lives for ever.
const FILL_SHORT_LIFE: f32 = 0.3;

// The clamp's width. It is clamp_gamma standard deviations where the
// history's place is uncertain by CLAMP_UNSURE_PX history pixels a frame or
// more, and CLAMP_EXACT_EXTRA wider where it is exact. The uncertainty
// counts CLAMP_PARALLAX_SHARE of the parallax the guide's distance gave.
const CLAMP_EXACT_EXTRA: f32 = 6.0;
const CLAMP_UNSURE_PX: f32 = 0.05;
const CLAMP_PARALLAX_SHARE: f32 = 0.25;

// Below this opacity a pixel is mostly sky, which is at infinity, and its
// inverse distance fades to the sky's zero with what little it holds.
const GUIDE_MIN_OPACITY: f32 = 0.02;

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> ResolveOut {
    // This history pixel's centre, in the image (uv) and in march pixels.
    let uv = frag_pos.xy / ru.history_size;
    let rp = uv * ru.render_size;
    let to_hist = ru.history_size / ru.render_size;
    let centre = vec2<i32>(floor(rp));
    let hi = vec2<i32>(ru.render_size) - vec2<i32>(1);
    let frame = u32(max(ru.frame_index, 0.0));
    let inv_2s2 = 0.5 / (ru.splat_sigma * ru.splat_sigma);

    // This frame's samples around the pixel. Two estimates come of them:
    //   splat  the samples whose rays landed on this history pixel, weighted
    //          by how near (a Gaussian, splat_sigma history pixels wide).
    //          Sharp, and on most frames at a coarse march there are none.
    //   fill   the same samples over a kernel one march pixel wide: always
    //          there, and as soft as the march is coarse.
    var w_splat = 0.0;
    var c_splat = vec4<f32>(0.0);
    var w_fill = 0.0;
    var c_fill = vec4<f32>(0.0);
    var g_fill = vec3<f32>(0.0);
    var s1 = vec3<f32>(0.0);
    var s2 = vec3<f32>(0.0);
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let q = clamp(centre + vec2<i32>(i, j), vec2<i32>(0), hi);
            let c = textureLoad(sample_tex, q, 0);
            let g = textureLoad(guide_tex, q, 0).xyz;
            // Where this sample's ray went: camera_ray's own draw.
            var sp = vec2<f32>(q) + vec2<f32>(0.5);
            if (ru.subpixel > 0.5) {
                sp = sp + (strat2(frame, vec2<u32>(q), STRAT_STREAM_SUBPIXEL,
                                  vec2<f32>(STRAT_ALPHA_1, STRAT_ALPHA_2))
                           - vec2<f32>(0.5)) * ru.jitter_scale;
            }
            let dr = sp - rp;
            let dh = dr * to_hist;
            let ws = exp(-dot(dh, dh) * inv_2s2);
            let wf = exp(-dot(dr, dr) * 2.0);
            w_splat = w_splat + ws;
            c_splat = c_splat + ws * c;
            w_fill = w_fill + wf;
            c_fill = c_fill + wf * c;
            g_fill = g_fill + wf * g;
            s1 = s1 + c.rgb;
            s2 = s2 + c.rgb * c.rgb;
        }
    }
    c_fill = c_fill / w_fill;
    g_fill = g_fill / w_fill;
    if (w_splat > 1e-6) {
        c_splat = c_splat / w_splat;
    } else {
        c_splat = c_fill;
    }
    let w_new = min(w_splat, 1.0);

    // What the pixel shows, as the guide summarizes it: the mean inverse
    // distance (in the guide's unit) and its variance over the contributions.
    let opacity = max(g_fill.z, GUIDE_MIN_OPACITY);
    let rho = g_fill.x / opacity;
    let rho_var = max(g_fill.y / opacity - rho * rho, 0.0);

    // Where that was in the previous frame. A point at distance 1 / rho along
    // this pixel's ray is, seen from the previous camera, along
    // dir + rho * delta: the ray's own direction when rho is 0 (the sky, or a
    // camera that only turned).
    let ndc = vec2<f32>(
        (2.0 * uv.x - 1.0) * ru.tan_half_fov,
        (1.0 - 2.0 * uv.y) * ru.tan_half_fov / ru.aspect);
    let dir = normalize(ru.cam_forward + ndc.x * ru.cam_right
                        + ndc.y * ru.cam_up);
    let v = dir + rho * ru.delta_g;
    let zf = dot(v, ru.prev_forward);
    let pn = vec2<f32>(dot(v, ru.prev_right), dot(v, ru.prev_up))
             / max(zf, 1e-6);
    let uv_prev = vec2<f32>(
        0.5 * (pn.x / ru.prev_tan_half_fov + 1.0),
        0.5 * (1.0 - pn.y * ru.aspect / ru.prev_tan_half_fov));
    let valid = ru.history_valid > 0.5 && zf > 1e-6
        && all(uv_prev >= vec2<f32>(0.0)) && all(uv_prev <= vec2<f32>(1.0));

    // The history there. The colour is read through a Catmull-Rom kernel
    // (sixteen loads; the history is rgba32float, which core WebGPU cannot
    // filter): a history that moves a fraction of a pixel every frame is
    // resampled every frame, and a bilinear read would low-pass it each
    // time until nothing finer than a few pixels was left. The kernel's
    // overshoot is taken off by the clamp below. The weight is read
    // bilinearly; it has nothing fine to keep. (sharp_history off reads the
    // colour bilinearly too: four loads for sixteen, and a softer picture.)
    var hist = vec4<f32>(0.0);
    var count = 0.0;
    if (valid) {
        let hmax = vec2<i32>(ru.history_size) - vec2<i32>(1);
        let p = uv_prev * ru.history_size - vec2<f32>(0.5);
        let base = floor(p);
        let f = p - base;
        let i0 = clamp(vec2<i32>(base), vec2<i32>(0), hmax);
        let i1 = clamp(vec2<i32>(base) + vec2<i32>(1), vec2<i32>(0), hmax);
        let w00 = (1.0 - f.x) * (1.0 - f.y);
        let w10 = f.x * (1.0 - f.y);
        let w01 = (1.0 - f.x) * f.y;
        let w11 = f.x * f.y;
        if (ru.sharp_history > 0.5) {
            let wx = catmull_rom(f.x);
            let wy = catmull_rom(f.y);
            for (var j: i32 = 0; j < 4; j = j + 1) {
                var row = vec4<f32>(0.0);
                let y = clamp(i32(base.y) + j - 1, 0, hmax.y);
                for (var i: i32 = 0; i < 4; i = i + 1) {
                    let x = clamp(i32(base.x) + i - 1, 0, hmax.x);
                    row = row + wx[i]
                        * textureLoad(hist_color, vec2<i32>(x, y), 0);
                }
                hist = hist + wy[j] * row;
            }
        } else {
            hist = w00 * textureLoad(hist_color, vec2<i32>(i0.x, i0.y), 0)
                 + w10 * textureLoad(hist_color, vec2<i32>(i1.x, i0.y), 0)
                 + w01 * textureLoad(hist_color, vec2<i32>(i0.x, i1.y), 0)
                 + w11 * textureLoad(hist_color, vec2<i32>(i1.x, i1.y), 0);
        }
        count = w00 * textureLoad(hist_aux, vec2<i32>(i0.x, i0.y), 0).y
              + w10 * textureLoad(hist_aux, vec2<i32>(i1.x, i0.y), 0).y
              + w01 * textureLoad(hist_aux, vec2<i32>(i0.x, i1.y), 0).y
              + w11 * textureLoad(hist_aux, vec2<i32>(i1.x, i1.y), 0).y;
    }

    count = min(count, ru.count_cap);

    // How long the history may live, in frames. One motion was used for
    // contributions whose inverse distances spread by sqrt(rho_var); each
    // frame of sideways travel therefore misplaces them by that spread times
    // the travel, and the errors add up over the history's life.
    let side = ru.delta_g - dot(ru.delta_g, dir) * dir;
    let drift_px = length(side) * sqrt(rho_var) * ru.px_per_rad;
    let life = min(ru.life_frames, ru.drift_budget / max(drift_px, 1e-6));
    let decay = clamp(1.0 - 1.0 / max(life, 1.0), 0.0, 1.0);

    // History is pulled into the range of this frame's samples around the
    // pixel: what it holds that none of them shows is something that is no
    // longer there. The range is as tight as the history's place is
    // uncertain. Under a turn, or for what is far away, the place is exact
    // and the range is wide, because the history then holds detail finer
    // than this frame's few samples can vouch for. What the translation
    // moved is placed only as well as its inverse distance is known: by the
    // misfit above, and by a share of the parallax itself.
    let parallax_px = length(side) * rho * ru.px_per_rad;
    let unsure_px = drift_px + CLAMP_PARALLAX_SHARE * parallax_px;
    let gamma = ru.clamp_gamma
        + CLAMP_EXACT_EXTRA * exp(-unsure_px / CLAMP_UNSURE_PX);
    let mean = s1 / 9.0;
    let sd = sqrt(max(s2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let lo = min(mean - gamma * sd, c_fill.rgb);
    let hi_c = max(mean + gamma * sd, c_fill.rgb);
    hist = vec4<f32>(clamp(hist.rgb, lo, hi_c), hist.a);

    // The blend: the history at its decayed weight, this frame's splat at
    // the weight of the samples that reached the pixel, and the fill. The
    // fill is what keeps a pixel no sample reached following the present,
    // and it is soft, so it weighs little where the history is trusted and
    // more the faster the history is being let go.
    var out: ResolveOut;
    let carried = count * decay;
    let w_soft = ru.fill_weight + FILL_SHORT_LIFE * (1.0 - decay);
    if (carried < 1e-3) {
        out.color = c_fill;
        out.aux = vec4<f32>(rho, w_soft + w_new, 0.0, 0.0);
        return out;
    }
    let total = carried + w_new + w_soft;
    out.color = (hist * carried + c_splat * w_new + c_fill * w_soft)
                / total;
    out.aux = vec4<f32>(rho, total, 0.0, 0.0);
    return out;
}
`;

/**
 * The camera's displacement since the previous frame (m), with a fold at the
 * domain's period taken out: a periodic camera that crosses a lateral face
 * reappears at the opposite one, and the picture does not move.
 */
export function cameraDelta(origin, prevOrigin, periodic, bmin, bmax) {
  const d = [origin[0] - prevOrigin[0], origin[1] - prevOrigin[1],
             origin[2] - prevOrigin[2]];
  if (periodic) {
    for (let a = 0; a < 2; a++) {
      const L = bmax[a] - bmin[a];
      d[a] -= L * Math.round(d[a] / L);
    }
  }
  return d;
}

/** The settings a Renderer starts with (its \`reproject\` record): whether
 *  the pass is used at all, and the constants.js values it runs on. */
export function defaultReprojection() {
  return {
    on: K.REPROJECT_DEFAULT,
    lifetimeS: K.REPROJECT_LIFETIME_S,
    driftBudgetPx: K.REPROJECT_DRIFT_BUDGET_PX,
    splatSigma: K.REPROJECT_SPLAT_SIGMA,
    clampGamma: K.REPROJECT_CLAMP_GAMMA,
    fillWeight: K.REPROJECT_FILL_WEIGHT,
    jitterScale: K.REPROJECT_JITTER_SCALE,
    sharpHistory: K.REPROJECT_SHARP_HISTORY,
  };
}

/**
 * Fill `out` (a Float32Array of RESOLVE_UNIFORM_FLOATS) for one frame.
 * `u` is this frame's march block and `prev` the previous frame's first 16
 * floats (its camera rows); `deltaM` is cameraDelta's answer; `params` is
 * the renderer's reprojection record, as this frame uses it.
 */
export function packResolveUniforms(out, u, prev, deltaM, {
  frameIndex, jitterScale, subpixel, historyValid, lifeFrames,
  renderSize, historySize, params, countCap,
}) {
  out.set(u.subarray(0, 16), 0);
  // Rows 2.w and 3.w hold exposure and the march-jitter switch in the march
  // block; here they are the frame index and the jitter scale.
  out[11] = frameIndex;
  out[15] = jitterScale;
  out[16] = deltaM[0] / GUIDE_UNIT_M;
  out[17] = deltaM[1] / GUIDE_UNIT_M;
  out[18] = deltaM[2] / GUIDE_UNIT_M;
  out[19] = prev[3];                      // previous tan(fov / 2)
  out.set(prev.subarray(4, 7), 20);       // previous forward
  out[23] = historyValid ? 1.0 : 0.0;
  out.set(prev.subarray(8, 11), 24);      // previous right
  out[27] = lifeFrames;
  out.set(prev.subarray(12, 15), 28);     // previous up
  out[31] = params.splatSigma;
  out[32] = renderSize[0];
  out[33] = renderSize[1];
  out[34] = historySize[0];
  out[35] = historySize[1];
  out[36] = params.clampGamma;
  out[37] = params.driftBudgetPx;
  out[38] = params.fillWeight;
  out[39] = subpixel ? 1.0 : 0.0;
  out[40] = historySize[0] / (2.0 * u[3]);
  out[41] = countCap;
  out[42] = params.sharpHistory ? 1.0 : 0.0;
  out[43] = 0.0;
  return out;
}
