// adscreens — the big screens, and the weird things they are selling.
//
// About one building in twelve over 50 m carries screens — but weighted by
// the cascade, so downtown runs nearer one in five and the outskirts one in
// forty, because advertising clusters where the crowds are and a uniform
// sprinkle left the megatower district (which is the shot) with almost none.
// A chosen building offers each of its vertical box facades to a second draw,
// and the ones that take it get ONE screen — 30-70% of the facade wide (up to
// 86% on a merged superblock, which is where the giants live), aspect
// anywhere from 4:3 to 1:2.5. That is a 10-50 m rectangle on an ordinary
// tower and a 140 m one on a superblock: the loudest thing on the wall.
//
// Content features are sized in METRES, not in fractions of the panel —
// character cells 3-6 m, dead-channel pixel pitch 0.7-1.6 m, test-card bars
// 1.6-4 m. Sizing them as panel fractions instead (which is what this file
// did first) gives a small screen 10 cm noise that is sub-pixel from across
// the street and a superblock 30 m characters, so the same archetype read as
// flat grey on one wall and as three enormous blobs on another.
//
// Six content archetypes, all pure math in screen-local uv, all static:
//   FACE     a giant face suggestion — two eye blobs and a mouth bar over a
//            skin vignette, with a neon pinprick in each eye. Abstracted
//            enough that it reads as a face without ever being one.
//   PRODUCT  a capsule silhouette, luminous, rimmed, on a saturated gradient
//            with a halo bleeding off it. The object is never named.
//   GLYPH    rows of pseudo-ideograms: three axis-aligned strokes per cell,
//            endpoints snapped to a 3x3 lattice out of one pcg2d draw, so the
//            characters have the regularity of writing without being any
//            writing. No real text — the calibration forbids it and it would
//            be worse if it were legible.
//   STRIPES  a test card. Bar count a multiple of three, colours cycling
//            through the screen's three-neon scheme, dark foot band.
//   STATIC   a dead channel: hash noise, a few rows shear-glitched, a few
//            more torn bright. Frozen — this is a broken screen, not a
//            playing one.
//   EYE      iris rings and radial spokes around a black pupil, one glint.
//
// Each screen draws a 3-colour scheme from an 8-hue neon family, and every
// archetype is written as three scalar FIELDS against those three colours:
// content = qa*f.x + qb*f.y + qc*f.z. That factoring is what makes the LOD
// honest — the screen's mean is qa*m.x + qb*m.y + qc*m.z with m the fields'
// own means, so the far-field colour is computed, not guessed.
//
// THE SCREEN IS OPAQUE. The facade hook is additive, so a bright emitter
// dropped on the wall leaves the wall's own lit windows punching through it —
// which is exactly what the first cut of this file did, and it read as a
// projection rather than a panel. So the panel first SUBTRACTS the core's
// window ladder over its whole outer rectangle (cc_adscreens_wall below
// recomputes it, at the core's own fp_eff, which we can form because
// u.cam_origin is in the uniform block) and then adds its own emission. The
// bezel subtracts and adds nothing, so it is a genuinely dark frame rather
// than a strip of bare wall, and the screen sits ON the building.
//
// BRIGHTNESS, and why it is not the brief's 2.5. At exposure 6 against a
// white point of 15, radiance 2.5 lands at display 1.0 EXACTLY: a screen
// normalised to a mean radiance of 2.5 is a wall-sized white rectangle with
// no hue and no content, which is what it rendered as. The tone map is what
// sets this scale, so the normalisation targets the mean colour's LARGEST
// channel (the one that clips first) rather than its luminance, and puts it
// near 0.6-1.4. The content's own contrast then carries highlights — glints,
// hot cores, torn static rows — up past 4, where they clip to white on
// purpose, the way a real LED wall photographs: a white-hot core in a field
// that still has colour.
//
// LOD. Content dissolves into the screen's mean over fp in [0.055, 0.33] x
// the screen's short side: full detail while the short side is wider than
// ~18 px, gone by ~3. The screen itself NEVER goes — past the dissolve it is
// a flat rectangle of exactly the colour its content averaged to, which is
// what a city mega-screen is at two kilometres. Inside that window the four
// periodic generators (glyph strokes, test-card bars, noise cells, iris
// rings) each blend to their OWN mean over their OWN feature size, which is
// far finer than the screen's, and the point highlights are gaussians
// widened by the pixel with their integral held. Nothing aliases on the way
// down and nothing vanishes at the bottom.
//
// THE MEANS ARE PER-SCREEN, NOT ENSEMBLE. A single-feature archetype's mean
// depends strongly on its own hash draw — a fat capsule covers three times
// the panel a thin one does — so an ensemble constant is wrong by 15-25% for
// any particular screen, and that error IS the brightness step between the
// resolved screen and the flat one. Every compact feature here therefore
// carries a closed form in its own draw: capsule and annulus areas, Steiner
// offsets for the rim and halo bands, exact gaussian integrals for the
// glints. The two panel-filling gradients (FACE's vignette, EYE's radial)
// are cubics in the aspect fitted to their erf integrals (max error 0.02%).
// The many-cell archetypes (GLYPH, STRIPES, STATIC) need no per-screen form:
// a screen holds dozens of cells, so the ensemble mean IS the screen's mean —
// except STRIPES' bar brightness, which is drawn per colour index rather than
// per bar so that its per-screen mean stays exact with only 2-5 bars a colour.
//
// Validation, twice over. On the CPU: a numpy port of these functions,
// integrated on a 192^2 quadrature grid per screen over 120 screens x 6
// aspects, against the closed forms. Per-screen error is under 1% on every
// dominant field for aspects 0.55 and wider, rising to ~7% of the panel mean
// at the narrowest 1:2.5 banner, where compact features clip against the
// panel edge and the unclipped areas overstate them. (For contrast, the
// ensemble constants this file started with were off by 2.8x on GLYPH and 3x
// on PRODUCT's halo, and gave FACE's glint a NEGATIVE mean.)
//
// And on the GPU, which is the claim that matters: fp is inversely
// proportional to render width at fixed fov, so rendering one parked camera
// at widths from 1920 down to 120 walks every screen in the frame through
// the entire LOD ladder with the geometry, the occlusion and the hash draws
// all held fixed. Over that 16x sweep the panels' mean displayed colour
// moves by at most 1.3% between adjacent steps and 2.6% end to end, with no
// step at either hand-off. That is the no-pops-on-a-dolly claim, measured
// rather than asserted.

const cc_adscreens_MIN_H: f32 = 50.0;      // shorter buildings get no screen
const cc_adscreens_BLDG_FRAC: f32 = 0.085; // of tall buildings
const cc_adscreens_FACE_FRAC: f32 = 0.55;  // of a chosen building's facades
const cc_adscreens_MIN_FACADE_W: f32 = 14.0;
const cc_adscreens_MIN_SCREEN: f32 = 6.0;
const cc_adscreens_BASE_LO: f32 = 8.0;     // lower edge above the tier base
const cc_adscreens_BASE_HI: f32 = 25.0;
const cc_adscreens_BEZEL_FRAC: f32 = 0.028;
const cc_adscreens_BEZEL_MIN: f32 = 0.40;
// Content -> mean, as a fraction of the screen's short side (m/px).
const cc_adscreens_LOD_LO: f32 = 0.055;
const cc_adscreens_LOD_HI: f32 = 0.33;
// Peak-channel radiance of the screen's MEAN colour (see the header): the
// band a screen's overall level is drawn from.
const cc_adscreens_WANT_LO: f32 = 0.60;
const cc_adscreens_WANT_HI: f32 = 1.40;
// Light the screen throws on the wall around it. A screen this bright with a
// perfectly sharp edge reads as a decal; a little spill is what puts it in
// front of the masonry.
const cc_adscreens_SPILL: f32 = 0.055;
const cc_adscreens_SPILL_SIG: f32 = 1.6;   // metres, scaled by screen size

// --- field means -----------------------------------------------------------
// The panel-filling gradients, as cubics in the aspect ratio: the mean of
// exp(-(a x^2 + b y^2)) over the panel, which is an erf product WGSL has no
// erf for. Fitted over ar in [0.40, 1.34]; max relative error 0.02%.
// FACE's vignette, exp(-3 x^2 - 4.05 y^2):
const cc_adscreens_VIGN_C: vec4<f32> =
    vec4<f32>(0.743370, 0.012539, -0.231555, 0.069387);
// EYE's background, exp(-2.2 (x^2 + y^2)):
const cc_adscreens_EYEBG_C: vec4<f32> =
    vec4<f32>(0.840833, 0.016997, -0.198110, 0.051518);
// GLYPH: mean ink of the 3-stroke alphabet, blanks (1 cell in 8) included.
// Measured over 3000 alphabet draws on a 96^2 grid: 0.1388.
const cc_adscreens_M_GLYPH: f32 = 0.1388;
// STRIPES: the bottom 18% of the panel dimmed to 0.15 -> 0.82 + 0.18*0.15.
const cc_adscreens_STRIPE_FOOT: f32 = 0.847;
// STATIC: a uniform hash averages to a half; 6% of rows are torn bright.
const cc_adscreens_M_STATIC: vec2<f32> = vec2<f32>(0.5, 0.06);
// EYE: E[(0.45 + 0.75*rings)(0.70 + 0.55*spokes)] at rings = spokes = 1/2.
const cc_adscreens_IRIS_MEAN: f32 = 0.80438;
const cc_adscreens_TAU: f32 = 6.2831853;
const cc_adscreens_SQRT_HALF_PI: f32 = 1.2533141;

fn cc_adscreens_cubic(c: vec4<f32>, x: f32) -> f32 {
    return c.x + x * (c.y + x * (c.z + x * c.w));
}

// The advertising palette: saturated where the windows are not. Deliberately
// louder than city_window_color — a screen that shares the facade's spectrum
// stops being a screen.
fn cc_adscreens_neon(k: f32) -> vec3<f32> {
    let i = i32(clamp(k, 0.0, 0.9999) * 8.0);
    if (i == 0) { return vec3<f32>(1.00, 0.13, 0.42); }  // rose
    if (i == 1) { return vec3<f32>(0.10, 0.90, 1.00); }  // cyan
    if (i == 2) { return vec3<f32>(1.00, 0.60, 0.10); }  // amber
    if (i == 3) { return vec3<f32>(0.52, 1.00, 0.20); }  // acid green
    if (i == 4) { return vec3<f32>(0.60, 0.30, 1.00); }  // violet
    if (i == 5) { return vec3<f32>(1.00, 0.22, 0.10); }  // hot red
    if (i == 6) { return vec3<f32>(0.82, 0.90, 1.00); }  // cold white
    return vec3<f32>(1.00, 0.84, 0.32);                  // warm yellow
}

fn cc_adscreens_sd_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * t);
}

// Area of a disc of radius r centred in a strip of half-width w: the iris of
// a narrow vertical banner runs off both sides of it, and an unclipped pi r^2
// would overstate its mean by half.
fn cc_adscreens_disc_clip(r: f32, w: f32) -> f32 {
    let full = 3.14159265 * r * r;
    if (r <= w) {
        return full;
    }
    let seg = r * r * acos(clamp(w / r, -1.0, 1.0))
            - w * sqrt(max(r * r - w * w, 0.0));
    return full - 2.0 * seg;
}

// Does this hit sit on the +nh face of that box? The facade hook gets a
// position and a normal but not which of the building's boxes it belongs to,
// and the screen has to be laid out on THAT box's extent.
fn cc_adscreens_on_face(p: vec3<f32>, nh: vec2<f32>,
                        bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    if (p.z < bmin.z - 0.05 || p.z > bmax.z + 0.05) {
        return false;
    }
    let far_c = select(bmin.xy, bmax.xy, nh > vec2<f32>(0.0));
    if (abs(dot(p.xy, nh) - dot(far_c, nh)) > 0.06) {
        return false;
    }
    let tg = vec2<f32>(-nh.y, nh.x);
    let ua = dot(bmin.xy, tg);
    let ub = dot(bmax.xy, tg);
    let u = dot(p.xy, tg);
    return u > min(ua, ub) - 0.06 && u < max(ua, ub) + 0.06;
}

// --- what the panel covers -------------------------------------------------
//
// The core's window ladder at this point on the wall, recomputed so the panel
// can subtract it. This mirrors the facade branch of city_shade: the same
// lattice, the same style switch, the same two far-field octaves and the same
// fp_eff hand-offs. It is the one place this component reaches into core
// behaviour rather than calling it, and it is duplication only because the
// core's facade emission is inlined in city_shade rather than factored into a
// function a component could call. If it is ever factored out, this goes.
//
// It includes the cc_window_glyph term, because it has to: the core
// multiplies a lit pane by whatever the glyph components put behind the
// glass, and leaving it out left windowlife's occupants standing on the
// panel as bright silhouettes — visible, and the one defect an
// occlusion-only test render still showed. That dispatcher is a composer
// guarantee rather than another component's symbol (it exists, returning
// 1, whatever is registered), and WGSL resolves module-scope functions out
// of order, so calling it forward is safe. With it in, the wall under the
// panel goes exactly black.
fn cc_adscreens_wall(cc: CityCell, uc: f32, vc: f32, fp: f32,
                     fp_eff: f32) -> vec3<f32> {
    let fp_glyph = fp;
    let iu = i32(floor(uc / cc.win_pitch));
    let iv = i32(floor(vc / CITY_FLOOR_H));
    let wh = city_rand4(vec2<u32>(
        cc.seed.x ^ bitcast<u32>(iu),
        cc.seed.y ^ bitcast<u32>(iv)
    ));
    let fh = city_rand4(vec2<u32>(
        cc.seed.x ^ 0x2545f491u,
        bitcast<u32>(iv) * 0x9e3779b9u
    ));
    let floor_dark = fh.x < CITY_DARK_FLOOR_FRAC;
    let fu = fract(uc / cc.win_pitch);
    let fv = fract(vc / CITY_FLOOR_H);
    let pane = fu > cc.pane_lo.x && fu < cc.pane_hi.x
            && fv > cc.pane_lo.y && fv < cc.pane_hi.y;
    var is_on = wh.x < cc.lit_frac;
    if (cc.win_style == 1) {
        let sh = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu >> 1) * 0x9e3779b9u),
            cc.seed.y ^ (bitcast<u32>(iv) * 0x51ed270bu)));
        is_on = sh.x < cc.lit_frac * 1.15;
    } else if (cc.win_style == 2) {
        let sh = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu) * 0x9e3779b9u),
            cc.seed.y ^ (bitcast<u32>(iv >> 2) * 0x51ed270bu)));
        is_on = sh.x < cc.lit_frac * 1.10;
    } else if (cc.win_style == 3) {
        let fl = city_rand4(vec2<u32>(
            cc.seed.x ^ 0x7feb352du,
            bitcast<u32>(iv) * 0x846ca68bu));
        is_on = fl.x < cc.lit_frac * 1.5 && wh.x < 0.88;
    }
    let lit = pane && !floor_dark && is_on;
    var e_win = vec3<f32>(0.0);
    if (lit) {
        let style_gain = select(
            select(1.0, 0.80, cc.win_style == 1),
            0.55, cc.win_style == 3);
        let bright = (0.25 + 5.0 * pow(wh.z, 7.0)) * style_gain;
        var cdraw = wh.y;
        if (cc.win_mono >= 0.0) {
            cdraw = clamp(cc.win_mono + (wh.y - 0.5) * 0.08, 0.0, 1.0);
        }
        e_win = city_window_color(cdraw, cc.palette_bias)
                * (CITY_WIN_RADIANCE * bright);
        let pane_uv = (vec2<f32>(fu, fv) - cc.pane_lo)
                      / max(cc.pane_hi - cc.pane_lo, vec2<f32>(1e-4));
        e_win = e_win * cc_window_glyph(cc, wh, pane_uv, fp_glyph);
    }
    let e_mean_base = CITY_PALETTE_MEAN
        * (cc.lit_frac * cc.pane_frac * (1.0 - CITY_DARK_FLOOR_FRAC)
           * CITY_WIN_RADIANCE * CITY_WIN_BRIGHT_MEAN);
    let ibu = iu >> 2;
    let ibv = iv >> 2;
    let bh1 = city_rand4(vec2<u32>(
        cc.seed.x ^ (bitcast<u32>(ibu) * 0x85ebca6bu),
        cc.seed.y ^ (bitcast<u32>(ibv) * 0xc2b2ae35u)
    ));
    var block_color = CITY_PALETTE_MEAN;
    if (bh1.y < 0.03) {
        block_color = city_window_color(0.90 + 0.09 * bh1.z, cc.palette_bias);
    }
    let block_var = 0.10 + 1.8 * pow(bh1.x, 5.0);
    let e_block = block_color
        * (cc.lit_frac * cc.pane_frac * (1.0 - CITY_DARK_FLOOR_FRAC)
           * CITY_WIN_RADIANCE * CITY_WIN_BRIGHT_MEAN
           * CITY_MEAN_COMP_BLOCK * block_var);
    let bh2 = city_rand4(vec2<u32>(
        cc.seed.x ^ 0x51ed270bu,
        bitcast<u32>(iv >> 4) * 0x9e3779b9u
    ));
    let band = 0.70 + 0.60 * bh2.x;
    let e_flat = e_mean_base * (CITY_MEAN_COMP_FLAT * band);
    let b1 = smoothstep(CITY_WIN_LOD_START, CITY_WIN_LOD_FULL, fp_eff);
    let b2 = smoothstep(CITY_BLOCK_LOD_START, CITY_BLOCK_LOD_FULL, fp_eff);
    return mix(e_win, mix(e_block, e_flat, b2), b1);
}

// --- the archetypes --------------------------------------------------------
//
// All of them take screen-local coordinates and return the three scalar
// fields that multiply the screen's three colours, plus — where the mean
// depends on the screen's own draw — the closed form of those fields' means.
// px and py are in units of the screen HEIGHT (px = (su - 0.5) * ar), so a
// circle drawn in them is round on the wall whatever the panel's shape. aa is
// half a pixel in those units; ps is the pixel's own half-width, for the
// gaussians that have to conserve their integral rather than shrink out of
// existence.

// A giant face. The eyes are blobs and the mouth is a bar, because a face
// assembled from parts you can name is a cartoon; what makes this one land is
// that the parts are just dark masses in the right places, over a vignette
// that could as easily be a light. The neon pinpricks in the eyes are the
// wrongness: nobody's eyes do that.
fn cc_adscreens_face(ch: vec4<f32>, px: f32, py: f32, aa: f32,
                     ps: f32) -> vec3<f32> {
    let sep = 0.13 + 0.07 * ch.x;
    let ey = 0.10 + 0.06 * ch.y;
    let er = 0.055 + 0.030 * ch.z;
    let my = -0.14 - 0.09 * ch.w;
    let mhw = 0.13 + 0.08 * ch.x;
    let mhh = 0.016 + 0.022 * ch.y;
    // Eyes are a little taller than wide, and the mouth is a slab.
    let d1 = length(vec2<f32>(px + sep, (py - ey) * 0.82));
    let d2 = length(vec2<f32>(px - sep, (py - ey) * 0.82));
    let e1 = 1.0 - smoothstep(er * 0.70, er * 1.15 + aa, d1);
    let e2 = 1.0 - smoothstep(er * 0.70, er * 1.15 + aa, d2);
    let mo = (1.0 - smoothstep(mhw - 0.02 - aa, mhw + 0.02 + aa, abs(px)))
           * (1.0 - smoothstep(mhh - aa, mhh + 0.012 + aa, abs(py - my)));
    let mask = clamp(max(max(e1, e2), mo), 0.0, 1.0);
    let vign = 0.42 + 0.58 * exp(-3.0 * (px * px + 1.35 * py * py));
    let skin = vign * (1.0 - 0.90 * mask);
    // Two point highlights, set off-centre in the eyes so the gaze is not
    // straight out. Widened by the pixel with the integral held, so they dim
    // honestly instead of flickering.
    let s0 = er * 0.26;
    let s = sqrt(s0 * s0 + ps * ps);
    let amp = 3.2 * (s0 * s0) / (s * s);
    let k = -0.5 / (s * s);
    let o = er * 0.28;
    let g1 = vec2<f32>(px + sep - o, py - ey - er * 0.18);
    let g2 = vec2<f32>(px - sep - o, py - ey - er * 0.18);
    let glint = amp * (exp(k * dot(g1, g1)) + exp(k * dot(g2, g2)));
    return vec3<f32>(skin, glint, 0.0);
}

fn cc_adscreens_face_mean(ch: vec4<f32>, ar: f32) -> vec3<f32> {
    let sep = 0.13 + 0.07 * ch.x;
    let ey = 0.10 + 0.06 * ch.y;
    let er = 0.055 + 0.030 * ch.z;
    let my = -0.14 - 0.09 * ch.w;
    let mhw = 0.13 + 0.08 * ch.x;
    let mhh = 0.016 + 0.022 * ch.y;
    let vign = 0.42 + 0.58 * cc_adscreens_cubic(cc_adscreens_VIGN_C, ar);
    // The blobs cut the vignette where they sit, so each area is weighted by
    // the vignette at its own centre, not by the panel average.
    let re = 0.925 * er;
    let a_eye = 3.14159265 * re * re / 0.82;
    let a_mouth = 4.0 * mhw * (mhh + 0.006);
    let ve = 0.42 + 0.58 * exp(-3.0 * (sep * sep + 1.35 * ey * ey));
    let vm = 0.42 + 0.58 * exp(-3.0 * (1.35 * my * my));
    let skin = vign - 0.90 * (2.0 * a_eye * ve + a_mouth * vm) / ar;
    let s0 = er * 0.26;
    let glint = 3.2 * cc_adscreens_TAU * s0 * s0 * 2.0 / ar;
    return vec3<f32>(skin, glint, 0.0);
}

// A product: one luminous capsule, a rim where the silhouette turns away,
// and a halo bleeding into a saturated gradient. Bottle, ampoule, pill,
// canister — the point is that it is never quite any of them.
fn cc_adscreens_product(ch: vec4<f32>, sv: f32, px: f32, py: f32,
                        aa: f32) -> vec3<f32> {
    let th = (ch.x - 0.5) * 0.9;
    let ln = 0.10 + 0.11 * ch.y;
    let r = 0.055 + 0.050 * ch.z;
    let ax = vec2<f32>(sin(th), cos(th));
    let p = vec2<f32>(px, py + 0.02);
    let d = cc_adscreens_sd_seg(p, -ax * ln, ax * ln) - r;
    let body = 1.0 - smoothstep(-aa - 0.004, aa + 0.004, d);
    let rim = exp(-abs(d) * 22.0);
    let bg = (0.18 + 0.82 * pow(1.0 - sv, 1.3)) * 0.55;
    let dh = max(d, 0.0);
    let halo = 1.1 * exp(-dh * dh / (2.0 * 0.075 * 0.075));
    return vec3<f32>(bg, body * 2.4 + rim * 1.1, halo);
}

// Areas by Steiner: a convex body's offset region at distance s has area
// (perimeter + 2 pi s) ds, which integrates the rim's exponential and the
// halo's gaussian in closed form. The inward rim saturates at the inradius.
fn cc_adscreens_product_mean(ch: vec4<f32>, ar: f32) -> vec3<f32> {
    let ln = 0.10 + 0.11 * ch.y;
    let r = 0.055 + 0.050 * ch.z;
    let area = 4.0 * ln * r + 3.14159265 * r * r;
    let peri = 4.0 * ln + 2.0 * 3.14159265 * r;
    let k = 22.0;
    let ekr = exp(-k * r);
    let rim_out = peri / k + 2.0 * 3.14159265 / (k * k);
    let rim_in = peri * (1.0 - ekr) / k
               - 2.0 * 3.14159265 * (1.0 - ekr * (1.0 + k * r)) / (k * k);
    let sig = 0.075;
    let halo = 1.1 * (area + peri * sig * cc_adscreens_SQRT_HALF_PI
                      + 2.0 * 3.14159265 * sig * sig);
    // The background gradient is exact: mean of (0.18 + 0.82 (1-sv)^1.3)*0.55.
    return vec3<f32>(0.29509,
                     (2.4 * area + 1.1 * (rim_out + rim_in)) / ar,
                     halo / ar);
}

// A wall of writing that is not writing. Three axis-aligned strokes per cell
// with endpoints snapped to a 3x3 lattice: that quantization is the whole
// trick — free-floating segments read as scribble, snapped ones read as
// characters, and a character set you cannot read is more unsettling than
// one you can. One cell in eight is blank, which is what gives the rows
// their rhythm.
fn cc_adscreens_glyph_cell(bits: vec2<u32>, cu: f32, cv: f32,
                           aa: f32) -> f32 {
    if ((bits.y & 7u) == 0u) {
        return 0.0;
    }
    let p = vec2<f32>(cu, cv);
    let t = 0.075;
    var m = 0.0;
    for (var k: u32 = 0u; k < 3u; k = k + 1u) {
        let sft = k * 8u;
        let horiz = ((bits.x >> sft) & 1u) == 0u;
        let c = 0.14 + 0.72 * f32((bits.x >> (sft + 1u)) & 3u) / 3.0;
        let s = 0.14 + 0.72 * f32((bits.x >> (sft + 3u)) & 3u) / 3.0;
        let e = min(0.86, s + 0.72 * (0.30 + 0.60
                    * f32((bits.x >> (sft + 5u)) & 3u) / 3.0));
        var a: vec2<f32>;
        var b: vec2<f32>;
        if (horiz) {
            a = vec2<f32>(s, c);
            b = vec2<f32>(e, c);
        } else {
            a = vec2<f32>(c, s);
            b = vec2<f32>(c, e);
        }
        let d = cc_adscreens_sd_seg(p, a, b);
        m = max(m, 1.0 - smoothstep(t - aa, t + aa, d));
    }
    return m;
}

// Iris rings around a black pupil. The staple, and it earns its place: an
// eye at forty metres across is the one image on this wall that looks back.
fn cc_adscreens_eye(ch: vec4<f32>, px: f32, py: f32, aa: f32, ps: f32,
                    damp: f32) -> vec3<f32> {
    let rr = 0.28 + 0.12 * ch.x;
    let pu = rr * (0.24 + 0.14 * ch.y);
    let cx = (ch.z - 0.5) * 0.10;
    let cy = (ch.w - 0.5) * 0.08;
    let q = vec2<f32>(px - cx, py - cy);
    let d = length(q);
    let bg = 0.12 + 0.55 * exp(-2.2 * d * d);
    let nr = 14.0 + 10.0 * ch.y;
    let ns = 14.0 + floor(ch.x * 14.0);
    // Rings and spokes are the finest periodic structure on this screen and
    // blend to their own means (1/2 and 1/2) well before the panel does.
    let rings = mix(0.5 + 0.5 * sin(nr * cc_adscreens_TAU * (d / rr)
                                    + ch.z * cc_adscreens_TAU), 0.5, damp);
    let spokes = mix(0.5 + 0.5 * sin(atan2(q.y, q.x) * ns), 0.5, damp);
    let shape = (1.0 - smoothstep(rr * 0.90, rr + aa, d))
              * smoothstep(pu * 0.85, pu * 1.15 + aa, d);
    let iris = shape * (0.45 + 0.75 * rings) * (0.70 + 0.55 * spokes);
    let s0 = rr * 0.10;
    let s = sqrt(s0 * s0 + ps * ps);
    let amp = 5.0 * (s0 * s0) / (s * s);
    let g = vec2<f32>(q.x + rr * 0.30, q.y - rr * 0.34);
    let glint = amp * exp(-0.5 * dot(g, g) / (s * s));
    return vec3<f32>(bg, iris, glint);
}

fn cc_adscreens_eye_mean(ch: vec4<f32>, ar: f32) -> vec3<f32> {
    let rr = 0.28 + 0.12 * ch.x;
    let pu = rr * (0.24 + 0.14 * ch.y);
    let bg = 0.12 + 0.55 * cc_adscreens_cubic(cc_adscreens_EYEBG_C, ar);
    // The annulus between the two smoothstep midpoints, clipped by the panel
    // sides — a tall banner cuts the iris off well inside its radius.
    let a_ann = cc_adscreens_disc_clip(0.95 * rr, 0.5 * ar)
              - 3.14159265 * pu * pu;
    let s0 = rr * 0.10;
    return vec3<f32>(bg,
                     cc_adscreens_IRIS_MEAN * a_ann / ar,
                     5.0 * cc_adscreens_TAU * s0 * s0 / ar);
}

// --- the hook --------------------------------------------------------------

fn cc_adscreens_facade(cc: CityCell, h: CityHit, uc: f32, vc: f32,
                       fp: f32) -> vec3<f32> {
    if (!cc.built || cc.height < cc_adscreens_MIN_H) {
        return vec3<f32>(0.0);
    }
    // Screens hang on flat vertical wall. A tapered shaft's skin is neither,
    // and the same rule that keeps antennas off a spire keeps screens off it.
    if (abs(h.normal.z) > 0.02) {
        return vec3<f32>(0.0);
    }
    // Which buildings. The base rate is one tall building in twelve, but not
    // spread evenly: screens CLUSTER. A uniform sprinkle put a handful on the
    // whole skyline and none at all in either canyon the harness frames, and
    // that is also not how a city works — the walls that carry advertising
    // are the ones a crowd walks past. So the draw is weighted by the
    // cascade percentile that already sets height and occupancy: downtown
    // runs about 2.5x the base rate, the outskirts a quarter of it, and the
    // megatower district (which is the shot) reliably carries screens.
    let dens_w = 0.30 + 2.20 * smoothstep(0.50, 0.95, cc.rank);
    let bd = city_rand4(cc.seed ^ vec2<u32>(0x1f83d9abu, 0x5be0cd19u));
    if (bd.x >= cc_adscreens_BLDG_FRAC * dens_w) {
        return vec3<f32>(0.0);
    }

    let nh = normalize(h.normal.xy + vec2<f32>(1e-9, 0.0));
    let tg = vec2<f32>(-nh.y, nh.x);
    var bmin = cc.b1min;
    var bmax = cc.b1max;
    var tier: u32 = 0u;
    var found = cc_adscreens_on_face(h.pos, nh, cc.b1min, cc.b1max);
    if (cc.tiers >= 2 && cc_adscreens_on_face(h.pos, nh, cc.b2min, cc.b2max)) {
        bmin = cc.b2min; bmax = cc.b2max; tier = 1u; found = true;
    }
    if (cc.tiers >= 3 && cc_adscreens_on_face(h.pos, nh, cc.b3min, cc.b3max)) {
        bmin = cc.b3min; bmax = cc.b3max; tier = 2u; found = true;
    }
    if (!found) {
        return vec3<f32>(0.0);
    }
    let ua = dot(bmin.xy, tg);
    let ub = dot(bmax.xy, tg);
    let u0 = min(ua, ub);
    let fw = abs(ub - ua);
    if (fw < cc_adscreens_MIN_FACADE_W) {
        return vec3<f32>(0.0);
    }

    var fid: u32 = 0u;
    if (nh.x < -0.5) { fid = 1u; }
    else if (nh.y > 0.5) { fid = 2u; }
    else if (nh.y < -0.5) { fid = 3u; }
    let key = vec2<u32>(cc.seed.x ^ (fid * 0x9e3779b9u) ^ 0x243f6a88u,
                        cc.seed.y ^ (tier * 0x85ebca6bu) ^ 0x13198a2eu);
    let ph = city_rand4(key);
    if (ph.x >= cc_adscreens_FACE_FRAC) {
        return vec3<f32>(0.0);
    }
    let ph2 = city_rand4(vec2<u32>(key.x ^ 0xa4093822u, key.y ^ 0x299f31d0u));

    // Size. A merged superblock's facade is already twice as wide, and it is
    // allowed a bigger fraction of it on top: those are the giants.
    let wlo = select(0.30, 0.35, cc.merged);
    let whi = select(0.70, 0.86, cc.merged);
    var sw = fw * mix(wlo, whi, ph.y);
    let ar = 0.40 + 0.93 * ph.z;          // width : height, 1:2.5 .. 4:3
    var sz = sw / ar;
    // Where the screen's lower edge sits. A mid-rise takes the plain law —
    // 8 to 25 m above the tier it stands on, which is where a screen goes on
    // a building you walk past. A megatower cannot: its tier-1 wall is
    // hundreds of metres of facade, soar's camera flies well above the
    // street, and a 25-m-high mark on a 400 m shaft is a thing no flight
    // ever sees. So a tall wall places its screen proportionally instead,
    // anywhere in the lower two thirds of itself — the same argument
    // facadeworks makes for signage, and the same crossover.
    let wall_h = bmax.z - bmin.z;
    let z_low = cc_adscreens_BASE_LO
              + (cc_adscreens_BASE_HI - cc_adscreens_BASE_LO) * ph.w;
    let z_tall = wall_h * (0.06 + 0.60 * ph.w);
    let z0 = bmin.z + mix(z_low, max(z_tall, z_low),
                          smoothstep(90.0, 320.0, wall_h));
    let avail = bmax.z - 3.0 - z0;
    if (avail < cc_adscreens_MIN_SCREEN) {
        return vec3<f32>(0.0);
    }
    if (sz > avail) {                      // keep the aspect, lose the size
        sw = sw * (avail / sz);
        sz = avail;
    }
    if (sw < cc_adscreens_MIN_SCREEN) {
        return vec3<f32>(0.0);
    }
    let slack = max(fw - 2.0 - sw, 0.0);
    let su = (uc - (u0 + 1.0 + slack * ph2.x)) / sw;
    let sv = (vc - z0) / sz;
    // The spill reaches beyond the panel, so the reject has to as well.
    let spill_sig = clamp(cc_adscreens_SPILL_SIG * (1.0 + 0.02 * min(sw, sz)),
                          0.8, 5.0);
    let mu = 2.6 * spill_sig / sw;
    let mv = 2.6 * spill_sig / sz;
    if (su < -mu || su > 1.0 + mu || sv < -mv || sv > 1.0 + mv) {
        return vec3<f32>(0.0);
    }

    // The view direction, and with it the core's own foreshortened footprint:
    // the panel has to subtract the wall at exactly the LOD the core drew it.
    // h.pos is in the CITY FRAME, so the camera comes from the frame's one
    // entry (city_camera_origin) rather than from u.cam_origin raw.
    let dir = normalize(h.pos - city_camera_origin());
    let fp_eff = fp / clamp(abs(dot(dir, h.normal)), 0.20, 1.0);

    // The outer rectangle: everything inside it is panel or bezel, and the
    // wall behind all of it is covered.
    let au = clamp(0.6 * fp / sw, 0.0015, 0.25);
    let av = clamp(0.6 * fp / sz, 0.0015, 0.25);
    let outer =
        smoothstep(-au, au, su) * (1.0 - smoothstep(1.0 - au, 1.0 + au, su))
        * smoothstep(-av, av, sv) * (1.0 - smoothstep(1.0 - av, 1.0 + av, sv));
    // The bezel: a thin dark frame the screen sits inside, so it reads as
    // mounted on the wall rather than painted into it.
    let bez = max(cc_adscreens_BEZEL_MIN,
                  cc_adscreens_BEZEL_FRAC * min(sw, sz));
    let bu = bez / sw;
    let bv = bez / sz;
    let panel =
        smoothstep(bu - au, bu + au, su)
        * (1.0 - smoothstep(1.0 - bu - au, 1.0 - bu + au, su))
        * smoothstep(bv - av, bv + av, sv)
        * (1.0 - smoothstep(1.0 - bv - av, 1.0 - bv + av, sv));

    // Three colours out of the neon family, never the same one twice.
    let qa0 = cc_adscreens_neon(ph2.z);
    let qb0 = cc_adscreens_neon(fract(ph2.z + 0.19 + 0.60 * ph2.w));
    let qc0 = cc_adscreens_neon(fract(ph2.z + 0.44 + 0.30 * ph.y));
    let ch = city_rand4(vec2<u32>(key.x ^ 0x082efa98u, key.y ^ 0xec4e6c89u));

    let px = (su - 0.5) * ar;
    let py = sv - 0.5;
    let aa = clamp(0.6 * fp / sz, 0.0008, 0.20);
    let ps = 0.5 * fp / sz;

    var f: vec3<f32>;
    var m: vec3<f32>;
    var qa: vec3<f32>;
    var qb: vec3<f32>;
    var qc: vec3<f32>;
    // Not every screen runs at the same level. A dead channel is the dimmest
    // thing on the street, not the brightest; a face lit to full white loses
    // the skin it is supposed to have.
    var lvl = 1.0;
    let ad = ph2.y;
    if (ad < 0.17) {                                   // FACE
        f = cc_adscreens_face(ch, px, py, aa, ps);
        m = cc_adscreens_face_mean(ch, ar);
        qa = mix(vec3<f32>(1.00, 0.68, 0.52), qa0, 0.35);
        qb = qb0;
        qc = vec3<f32>(0.0);
        lvl = 0.72;
    } else if (ad < 0.38) {                            // PRODUCT
        f = cc_adscreens_product(ch, sv, px, py, aa);
        m = cc_adscreens_product_mean(ch, ar);
        qa = qa0 * 0.60;
        qb = mix(qb0, vec3<f32>(1.0), 0.55);
        qc = qc0;
    } else if (ad < 0.59) {                            // GLYPH
        // Characters are sized in METRES, not in panel fractions: a shop
        // screen carries three of them and a superblock carries thirty,
        // which is the difference between a sign and a wall of writing.
        // Counting rows as a fixed fraction of the panel instead made every
        // screen three characters tall, so the small ones read as scribble
        // and the giants as three enormous blobs.
        let cell_m = 3.2 + 3.0 * ch.x;
        let nr = clamp(round(sz / cell_m), 2.0, 40.0);
        let nc = clamp(round(sw / (cell_m * (0.85 + 0.35 * ch.y))),
                       1.0, 60.0);
        let gx = su * nc;
        let gy = sv * nr;
        let ix = i32(floor(gx));
        let iy = i32(floor(gy));
        let bits = pcg2d(vec2<u32>(
            key.x ^ (bitcast<u32>(ix) * 0x9e3779b9u),
            key.y ^ (bitcast<u32>(iy) * 0x85ebca6bu)));
        // Cell size on the wall sets when the strokes stop resolving.
        let cs = min(sw / nc, sz / nr);
        let dmp = smoothstep(0.30 * cs, 1.1 * cs, fp);
        let gm = mix(cc_adscreens_glyph_cell(
                         bits, fract(gx), fract(gy),
                         clamp(0.6 * fp / cs, 0.002, 0.25)),
                     cc_adscreens_M_GLYPH, dmp);
        f = vec3<f32>(0.40 + 0.30 * (1.0 - sv), gm * 2.6, 0.0);
        m = vec3<f32>(0.55, cc_adscreens_M_GLYPH * 2.6, 0.0);
        qa = select(vec3<f32>(0.42, 0.03, 0.06), vec3<f32>(0.04, 0.07, 0.42),
                    ch.w > 0.5);
        qb = qa0;
        qc = vec3<f32>(0.0);
    } else if (ad < 0.70) {                            // STRIPES
        // Bars are metres wide too, and their count stays a multiple of
        // three so each colour gets exactly a third of the panel.
        let bar_m = 1.6 + 2.4 * ch.y;
        let nb = 3.0 * clamp(round(sw / (3.0 * bar_m)), 2.0, 12.0);
        let j = floor(clamp(su, 0.0, 0.99999) * nb);
        let idx = i32(j) - 3 * (i32(j) / 3);
        // One brightness per COLOUR, not per bar: with only 2-5 bars to a
        // colour, a per-bar draw would leave the screen's true mean 15% away
        // from any constant we could write down, and that gap is the LOD
        // step. Per colour it is exact.
        let bh = city_rand4(vec2<u32>(key.x ^ 0x27d4eb2fu, key.y ^ 0x165667b1u));
        let brs = vec3<f32>(0.55 + 0.90 * bh.x, 0.55 + 0.90 * bh.y,
                            0.55 + 0.90 * bh.z);
        let br = select(select(brs.z, brs.y, idx == 1), brs.x, idx == 0);
        let foot = mix(0.15, 1.0, smoothstep(0.18 - av, 0.18 + av, sv));
        let v = br * foot;
        m = brs * (cc_adscreens_STRIPE_FOOT / 3.0);
        let dmp = smoothstep(0.45 * sw / nb, 1.4 * sw / nb, fp);
        f = mix(vec3<f32>(select(0.0, v, idx == 0), select(0.0, v, idx == 1),
                          select(0.0, v, idx == 2)), m, dmp);
        qa = qa0;
        qb = qb0;
        qc = qc0;
    } else if (ad < 0.81) {                            // STATIC
        // The dead channel's pixel pitch is a physical size — a coarse LED
        // module, 0.7-1.6 m — so the grain reads at the same distance on
        // every screen instead of vanishing on the small ones.
        let pitch_m = 0.7 + 0.9 * ch.x;
        let ncx = clamp(round(sw / pitch_m), 6.0, 220.0);
        let ncy = max(1.0, round(sz / pitch_m));
        let row = floor(sv * ncy);
        let rh = city_rand4(vec2<u32>(
            key.x ^ (bitcast<u32>(i32(row)) * 0x9e3779b9u),
            key.y ^ 0x51ed270bu));
        // A tenth of the rows are sheared: a frozen glitch, not a moving one.
        let uu = fract(su + select(0.0, (rh.y - 0.5) * 0.35, rh.x < 0.10));
        let nz = city_rand4(vec2<u32>(
            key.x ^ (bitcast<u32>(i32(floor(uu * ncx))) * 0x85ebca6bu),
            key.y ^ (bitcast<u32>(i32(row)) * 0xc2b2ae35u)));
        m = vec3<f32>(cc_adscreens_M_STATIC, 0.0);
        let cs = min(sw / ncx, sz / ncy);
        let dmp = smoothstep(0.35 * cs, 1.3 * cs, fp);
        f = vec3<f32>(mix(nz.x, m.x, dmp),
                      mix(select(0.0, 1.0, rh.z < 0.06), m.y, dmp), 0.0);
        qa = vec3<f32>(0.78, 0.82, 0.92);
        qb = qa0;
        qc = vec3<f32>(0.0);
        lvl = 0.55;
    } else {                                           // EYE
        let rr = 0.28 + 0.12 * ch.x;
        let ring_p = rr * sz / (14.0 + 10.0 * ch.y);
        let dmp = smoothstep(0.40 * ring_p, 1.3 * ring_p, fp);
        f = cc_adscreens_eye(ch, px, py, aa, ps, dmp);
        m = cc_adscreens_eye_mean(ch, ar);
        qa = qa0 * 0.55;
        qb = qb0;
        qc = vec3<f32>(1.0, 0.98, 0.95);
    }

    // The screen's own mean colour: the three colours against the three field
    // means. It is both the far-field asymptote and the yardstick the content
    // is normalized against, so the level a screen was designed to have
    // survives whatever its palette draw turned out to be. Normalising on the
    // LARGEST channel rather than on luminance is what keeps the hue: the
    // tone map clips channels one at a time, and a screen normalised by
    // luminance drives its dominant channel far past the white point and
    // comes back white.
    let mean_c = max(qa * m.x + qb * m.y + qc * m.z, vec3<f32>(0.0));
    let peak = max(max(mean_c.r, mean_c.g), mean_c.b);
    let want = mix(cc_adscreens_WANT_LO, cc_adscreens_WANT_HI, ph2.x);
    let gain = want * lvl / max(peak, 1e-3);

    let lod = smoothstep(cc_adscreens_LOD_LO * min(sw, sz),
                         cc_adscreens_LOD_HI * min(sw, sz), fp);
    let content = mix(qa * f.x + qb * f.y + qc * f.z, mean_c, lod);

    // Spill: the wall around the panel catches its light. Cheap, and it is
    // what stops the screen reading as a decal printed on the masonry.
    let dxu = max(max(-su, su - 1.0), 0.0) * sw;
    let dyv = max(max(-sv, sv - 1.0), 0.0) * sz;
    let dout = length(vec2<f32>(dxu, dyv));
    let spill = cc_adscreens_SPILL
              * exp(-0.5 * dout * dout / (spill_sig * spill_sig));

    // Opaque: take the wall out over the whole outer rectangle, then put the
    // screen back inside the bezel.
    return content * (gain * panel)
         + mean_c * (gain * spill * (1.0 - outer))
         - cc_adscreens_wall(cc, uc, vc, fp, fp_eff) * outer;
}
