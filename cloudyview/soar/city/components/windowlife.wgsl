// windowlife — what stands between the light and the glass.
//
// A lit pane is 2.45 m wide by 2.23 m tall (CITY_WIN_PITCH_U and
// CITY_FLOOR_H times the pane fractions). Most rooms show nothing but their
// light. A quarter have something drawn across them. One in seven holds a
// person, dark against the room behind them — the thing that makes a wall
// read as inhabited rather than as a lamp array. One in a hundred holds
// something whose eyes are lit from inside.
//
// Everything here is a transmission MULTIPLIER on the pane's own emission,
// so it can only take light away — except the android's eyes, which return
// a multiplier above 1 and so read as emission through the same interface.
//
// LOD is the whole discipline here. These glyphs are sub-window detail:
// they must be gone before the core's window->block dissolve even starts
// (CITY_WIN_LOD_START, fp_eff 1.6 m/px). Each case blends to its OWN mean
// transmission over a footprint window sized to that case's FINEST feature,
// not to the pane — a blind's 0.4 m stripe and an android's 5 cm eye alias
// long before a curtain's edge does, so they die sooner. Past its window a
// case returns a constant, which is what lets the core's own dissolve stay
// smooth. Nothing here ever just vanishes.
//
// The means are measured, not guessed: 4e6 Monte Carlo draws of (window
// hash, pane_uv) against a numpy port of this file, so the hand-off from
// pattern to mean is invisible and the far-field facade statistics are
// exactly what they were before this component existed.

const cc_windowlife_PANE_W: f32 = 2.448;   // CITY_WIN_PITCH_U * (U_HI - U_LO)
const cc_windowlife_PANE_H: f32 = 2.232;   // CITY_FLOOR_H * (V_HI - V_LO)
const cc_windowlife_ASPECT: f32 = 1.0968;  // W / H; makes circles round

// Case shares. Nothing takes the rest (60%).
const cc_windowlife_T_DRAPE: f32 = 0.60;    // .60-.85  drapes    25%
const cc_windowlife_T_FIGURE: f32 = 0.85;   // .85-.99  figures   14%
const cc_windowlife_T_ANDROID: f32 = 0.99;  // .99-1.0  androids   1%

// Transmissions through each thing, and the ensemble mean each dissolves
// into. Curtain cloth is warm (it is lit from behind by a warm room and
// eats blue); a body is near-neutral and much darker; aluminium blinds are
// neutral. See the header on how the means were obtained.
const cc_windowlife_CURTAIN_T: vec3<f32> = vec3<f32>(0.32, 0.26, 0.19);
const cc_windowlife_CURTAIN_MEAN: vec3<f32> = vec3<f32>(0.686, 0.660, 0.631);
const cc_windowlife_BLIND_T: f32 = 0.42;
const cc_windowlife_BLIND_MEAN: vec3<f32> = vec3<f32>(0.733);
const cc_windowlife_BODY_T: vec3<f32> = vec3<f32>(0.10, 0.09, 0.09);
const cc_windowlife_FIGURE_MEAN: vec3<f32> = vec3<f32>(0.822, 0.820, 0.820);
// The eyes: a multiplier well above 1 in G and B and near zero in R, so the
// dot lands cyan whatever colour the room behind it is.
const cc_windowlife_EYE_GAIN: vec3<f32> = vec3<f32>(0.05, 3.0, 2.8);
const cc_windowlife_EYE_SEP: f32 = 0.028;   // half-separation, pane widths
const cc_windowlife_EYE_SIGMA: f32 = 0.0060; // ~0.02 pane: a point, not a lamp

// Footprint windows (m/px) over which each case blends into its mean. Set
// by the case's finest feature, all of them closed before fp_eff 1.6.
const cc_windowlife_LOD_CURTAIN: vec2<f32> = vec2<f32>(0.40, 1.50);
const cc_windowlife_LOD_BLIND: vec2<f32> = vec2<f32>(0.18, 0.60);
const cc_windowlife_LOD_FIGURE: vec2<f32> = vec2<f32>(0.30, 1.20);
const cc_windowlife_LOD_EYE: vec2<f32> = vec2<f32>(0.05, 0.18);

struct cc_windowlife_Pose {
    head: vec2<f32>,   // head centre in pane_uv
    head_r: f32,       // head radius in u units (u is the round one)
    torso_x: f32,      // torso centre in u
    torso_w: f32,      // torso half-width in u
    shoulder: f32,     // v of the shoulder line
    tilt: f32,         // shoulder line slope; one side rides higher
}

// wh's components already carry the core's meanings — x decides lit, y the
// palette, z the brightness — so a draw taken straight off them would be
// conditioned on the window being lit and correlated with its colour. Mix
// all four back through the city's own hash instead.
fn cc_windowlife_hash(wh: vec4<f32>) -> vec4<f32> {
    let a = u32(wh.x * 16777216.0) * 0x9e3779b9u
          ^ u32(wh.w * 16777216.0) * 0x85ebca6bu;
    let b = u32(wh.y * 16777216.0) * 0xc2b2ae35u
          ^ u32(wh.z * 16777216.0) * 0x27d4eb2fu;
    return city_rand4(vec2<u32>(a ^ 0x7f4a7c15u, b ^ 0x1b56c4e9u));
}

// Half a pixel in pane-width units: every soft edge below opens by this, so
// an edge stays about a pixel wide instead of crawling as the camera moves.
fn cc_windowlife_aa_u(fp: f32) -> f32 {
    return min(0.6 * fp / cc_windowlife_PANE_W, 0.30);
}

fn cc_windowlife_aa_v(fp: f32) -> f32 {
    return min(0.6 * fp / cc_windowlife_PANE_H, 0.30);
}

// Curtains drawn part-way, or blinds. Both are whole-pane furniture, so
// both hold to a coarser footprint than a figure does — but the blind's
// stripe is finer than the curtain's edge, and it dies first.
fn cc_windowlife_drape(h: vec4<f32>, uv: vec2<f32>, fp: f32) -> vec3<f32> {
    if (h.y < 0.55) {
        // Two masses closing in from the sides, 20-70% of the pane between
        // them, split unevenly — one side is nearly always drawn further.
        let lod = smoothstep(cc_windowlife_LOD_CURTAIN.x,
                             cc_windowlife_LOD_CURTAIN.y, fp);
        if (lod >= 1.0) {
            return cc_windowlife_CURTAIN_MEAN;
        }
        let cov = 0.20 + 0.50 * h.z;
        let split = 0.15 + 0.70 * h.w;
        let soft = 0.022 + cc_windowlife_aa_u(fp);
        // Cloth hangs: the drawn edge leans a few centimetres off vertical,
        // and that lean is most of what separates a curtain from a gradient.
        let lean = (0.055 * (h.z - 0.5)) * (uv.y - 0.5);
        let l = cov * split + lean;
        let r = 1.0 - cov * (1.0 - split) + lean;
        let m = clamp(max(1.0 - smoothstep(l - soft, l + soft, uv.x),
                          smoothstep(r - soft, r + soft, uv.x)),
                      0.0, 1.0);
        // The hem leaks a little more light than the rod end. Two touches
        // of cloth, both monotone in position and so free of any repeat
        // frequency to alias: the vertical lean of the light, and a denser
        // gather in the first few centimetres behind the drawn edge, which
        // is what a curtain has and a gradient does not.
        let din = max(max(l - uv.x, uv.x - r), 0.0);
        let bunch = 1.0 - 0.25 * exp(-din * 18.0);
        let cloth = cc_windowlife_CURTAIN_T
                  * ((0.89 + 0.22 * (1.0 - uv.y)) * bunch);
        return mix(mix(vec3<f32>(1.0), cloth, m),
                   cc_windowlife_CURTAIN_MEAN, lod);
    }
    // Blinds: 4-6 slats. The duty cycle is symmetric about its edges so
    // widening them for the footprint blurs without darkening.
    let lod = smoothstep(cc_windowlife_LOD_BLIND.x,
                         cc_windowlife_LOD_BLIND.y, fp);
    if (lod >= 1.0) {
        return cc_windowlife_BLIND_MEAN;
    }
    let nb = floor(4.0 + 2.99 * h.z);
    let s = fract(uv.y * nb + h.w);
    let e = 0.07 + cc_windowlife_aa_v(fp) * nb;
    let band = smoothstep(0.20 - e, 0.20 + e, s)
             * (1.0 - smoothstep(0.66 - e, 0.66 + e, s));
    return mix(mix(vec3<f32>(1.0), vec3<f32>(cc_windowlife_BLIND_T), band),
               cc_windowlife_BLIND_MEAN, lod);
}

// Three poses. Not three people — three ways a body stands against a lit
// room, which is all that survives 30 m of night air anyway. The shoulder
// line is always tilted and the head always carried a little off the torso
// centre: a square-on symmetric figure reads as a pictogram, not a person,
// and that is the failure mode this case has to avoid.
fn cc_windowlife_pose(h: vec4<f32>) -> cc_windowlife_Pose {
    let g = city_rand4(vec2<u32>(u32(h.y * 16777216.0) ^ 0x68e31da4u,
                                 u32(h.z * 16777216.0) ^ 0xb5297a4du));
    let side = select(-1.0, 1.0, g.x < 0.5);
    let scale = 0.86 + 0.30 * g.z;
    let tilt = side * (0.05 + 0.13 * g.w);
    if (h.w < 0.45) {
        // Standing at one edge of the pane, looking out; often half out of
        // frame, which is how you actually catch someone at a window.
        let hx = 0.5 + side * (0.22 + 0.17 * g.y);
        let hy = 0.66 + 0.10 * g.w;
        return cc_windowlife_Pose(vec2<f32>(hx, hy), 0.085 * scale,
                                  hx - side * (0.02 + 0.03 * g.y),
                                  0.190 * scale, hy - 0.080 * scale, tilt);
    }
    if (h.w < 0.80) {
        // Sitting: low in the frame and wide, the shoulders nearer the sill
        // than the head is to the lintel.
        let hx = 0.5 + (g.y - 0.5) * 0.58;
        let hy = 0.40 + 0.12 * g.w;
        return cc_windowlife_Pose(vec2<f32>(hx, hy), 0.082 * scale,
                                  hx + side * (0.02 + 0.04 * g.y),
                                  0.215 * scale, hy - 0.076 * scale, tilt * 0.7);
    }
    // Leaning: the head carried well off the line of the torso.
    let hx = 0.5 + (g.y - 0.5) * 0.55;
    let hy = 0.62 + 0.09 * g.w;
    return cc_windowlife_Pose(vec2<f32>(hx, hy), 0.084 * scale,
                              hx - side * (0.06 + 0.04 * g.w),
                              0.170 * scale, hy - 0.078 * scale, tilt * 1.3);
}

// Head plus shoulders-and-torso, both soft-edged. A hard-edged version of
// this reads as a sticker; the soft edge is what makes it a person seen
// through glass.
fn cc_windowlife_figure_mask(p: cc_windowlife_Pose, uv: vec2<f32>,
                             fp: f32) -> f32 {
    let au = cc_windowlife_aa_u(fp);
    // Heads are taller than they are wide.
    let d = vec2<f32>(uv.x - p.head.x,
                      (uv.y - p.head.y) / (cc_windowlife_ASPECT * 1.12));
    let head = 1.0 - smoothstep(p.head_r * 0.76,
                                p.head_r * 1.14 + au, length(d));
    // The shoulder line is tilted and starts INSIDE the head, so the body
    // grows out from behind it — a shoulder line below the jaw leaves a gap
    // of light at the neck and turns the whole thing into a lollipop. Full
    // width about 17 cm down, narrowing again toward the waist.
    let dx = uv.x - p.torso_x;
    let below = p.shoulder - uv.y + p.tilt * dx;
    let w = p.torso_w * smoothstep(-0.015, 0.105, below)
                      * (1.0 - 0.16 * smoothstep(0.12, 0.45, below));
    let soft = 0.028 + au;
    let torso = (1.0 - smoothstep(w - soft, w + soft, abs(dx)))
              * smoothstep(-0.02 - au, 0.03 + au, below);
    return clamp(max(head, torso), 0.0, 1.0);
}

// Two points of light where the eyes are. The gaussian is widened by the
// footprint with its integral held fixed, so the dots dim honestly as they
// go sub-pixel instead of flickering, and are gone by 0.18 m/px — this is a
// thing you find by flying close, not a feature of the skyline.
fn cc_windowlife_eyes(p: cc_windowlife_Pose, uv: vec2<f32>,
                      fp: f32) -> vec3<f32> {
    let fade = 1.0 - smoothstep(cc_windowlife_LOD_EYE.x,
                                cc_windowlife_LOD_EYE.y, fp);
    if (fade <= 0.0) {
        return vec3<f32>(0.0);
    }
    let s0 = cc_windowlife_EYE_SIGMA;
    let px = 0.5 * fp / cc_windowlife_PANE_W;
    let s = sqrt(s0 * s0 + px * px);
    let amp = (s0 * s0) / (s * s);      // conserve the integral
    let cy = p.head.y + 0.014 * (p.head_r / 0.085);
    let d1 = vec2<f32>(uv.x - p.head.x + cc_windowlife_EYE_SEP,
                       (uv.y - cy) / cc_windowlife_ASPECT);
    let d2 = vec2<f32>(uv.x - p.head.x - cc_windowlife_EYE_SEP,
                       (uv.y - cy) / cc_windowlife_ASPECT);
    let k = -0.5 / (s * s);
    let g = exp(k * dot(d1, d1)) + exp(k * dot(d2, d2));
    return cc_windowlife_EYE_GAIN * (g * amp * fade);
}

fn cc_windowlife_figure(h: vec4<f32>, uv: vec2<f32>, fp: f32,
                        android: bool) -> vec3<f32> {
    let lod = smoothstep(cc_windowlife_LOD_FIGURE.x,
                         cc_windowlife_LOD_FIGURE.y, fp);
    if (lod >= 1.0) {
        return cc_windowlife_FIGURE_MEAN;   // the eyes average into nothing
    }
    let p = cc_windowlife_pose(h);
    let m = cc_windowlife_figure_mask(p, uv, fp);
    let body = mix(vec3<f32>(1.0), cc_windowlife_BODY_T, m);
    if (!android) {
        return mix(body, cc_windowlife_FIGURE_MEAN, lod);
    }
    // The eyes ride on top of the silhouette and above 1: they are the one
    // thing in this file that adds light rather than taking it.
    let lit = body + cc_windowlife_eyes(p, uv, fp) * m;
    return mix(lit, cc_windowlife_FIGURE_MEAN, lod);
}

fn cc_windowlife_glyph(cc: CityCell, wh: vec4<f32>, pane_uv: vec2<f32>,
                       fp: f32) -> vec3<f32> {
    let h = cc_windowlife_hash(wh);
    if (h.x < cc_windowlife_T_DRAPE) {
        return vec3<f32>(1.0);      // most windows are just light
    }
    if (h.x < cc_windowlife_T_FIGURE) {
        return cc_windowlife_drape(h, pane_uv, fp);
    }
    return cc_windowlife_figure(h, pane_uv, fp,
                                h.x >= cc_windowlife_T_ANDROID);
}
