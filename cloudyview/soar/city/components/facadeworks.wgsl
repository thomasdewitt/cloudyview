// facadeworks — additive facade detail on top of the core window ladder.
//
// Three layers, in the order they matter to the shot:
//
//   1. VERTICAL NEON SIGNAGE. The cyberpunk cue. A hashed minority of
//      buildings (~12%) carries one narrow vertical sign box mounted on a
//      corner of the main mass, so both walls meeting there show it:
//      saturated amber / cyan / magenta at radiance 5-8, broken into
//      glyph-like blocks that suggest characters without being any script.
//      A mid-rise gets a 12-40 m strip starting 6 m up; a tower gets a
//      ribbon proportional to its own height. This is the layer the whole
//      component exists for — at street range it should read as typography,
//      and from kilometres out it should still be a colored vertical line,
//      which is what makes a skyline read as a CITY and not a lit grid.
//   2. BALCONY BANDS. ~25% of buildings get balcony ledges every 2nd or 3rd
//      storey. We are ADDITIVE, so the ledge itself cannot be drawn (it is a
//      dark band, and there is no multiply here); what we CAN draw is the
//      light that spills past a balcony door, so the ledge reads by its warm
//      underglow line rather than by its shadow. Deliberately faint:
//      balconies are rhythm, not decoration.
//   3. FIRE-ESCAPE ZIGZAGS. Filigree: a stair lattice rim-lit by street
//      light on one wall of ~15% of mid-rises. Sub-pixel almost everywhere,
//      so it is the layer with the most aggressive LOD.
//
// LOD discipline (SPEC rule 3). Every layer states its own mean and blends
// to it as the pixel footprint grows; nothing switches off. Two subtleties
// this file cares about:
//
//   * The sign strip is 1.2 m wide, so past ~2.5 m/px it is thinner than a
//     pixel and a naive gate would make it strobe. Instead the cross-section
//     WIDENS to the pixel while its amplitude drops by the same factor, so
//     the integrated energy across the strip is invariant. That is why a
//     sign is still a clean colored line at 3 km instead of a sparkle.
//   * The core hands us `fp`, not its own `fp_eff` (which divides by the
//     view/normal cosine); we have no `dir` here. So our thresholds are on
//     the un-foreshortened footprint and are set a little conservatively.
//
// The composer's namespace rule covers function-local `var` too, so this
// file is written without mutable locals — which is no loss: every layer is
// a mask times a color.

// --- Vertical neon signage -------------------------------------------------
const cc_facadeworks_SIGN_FRAC: u32 = 31u;      // of 256 buildings (~12%)
const cc_facadeworks_SIGN_HW: f32 = 0.60;      // half-width of the strip (m)
const cc_facadeworks_SIGN_V0: f32 = 6.0;       // strip starts this high (m)
const cc_facadeworks_SEG_PITCH: f32 = 1.15;    // glyph block pitch (m)
const cc_facadeworks_SEG_H: f32 = 0.90;        // lit height within the pitch
const cc_facadeworks_SEG_DUTY: f32 = 0.7826;   // SEG_H / SEG_PITCH
// Mean ink coverage of the 5-stroke glyph alphabet below, each stroke drawn
// with probability 1/2. Integrated over the alphabet rather than guessed:
// this is the value a block collapses to once its strokes go sub-pixel.
const cc_facadeworks_GLYPH_MEAN: f32 = 0.312;
// Both of these are calibrated against exposure 6, where anything over
// ~0.05 radiance is already a clearly visible grey: the first version of
// this halo ran at 0.33 and rendered the sign as a solid tan panel.
const cc_facadeworks_SIGN_HALO_AMP: f32 = 0.014; // wall spill, x the strip mean
const cc_facadeworks_SIGN_HALO_SIG: f32 = 0.85;  // spill sigma (m)
// A dim backing panel behind the glyphs, so a sign block reads as an
// illuminated box with writing on it rather than as free-floating strokes.
const cc_facadeworks_SIGN_PANEL: f32 = 0.005;

// --- Balcony bands ---------------------------------------------------------
const cc_facadeworks_BAL_FRAC: u32 = 64u;      // of 256 buildings (25%)
const cc_facadeworks_BAL_GLOW_V0: f32 = 0.55;  // door spill starts above the
const cc_facadeworks_BAL_GLOW_H: f32 = 0.45;   // 0.5 m ledge, and is this tall
// The brief's 0.6 rendered as a bright hairline rule ruled clean across a
// whole wall — the single most prominent thing on the facade, and the exact
// "stripe" the brief warns against. At exposure 6 a spill that reads as
// spill, not as a drawn line, wants roughly a tenth of that.
const cc_facadeworks_BAL_RADIANCE: f32 = 0.09;
const cc_facadeworks_BAL_COLOR: vec3<f32> = vec3<f32>(1.0, 0.52, 0.22);
const cc_facadeworks_BAL_FLOOR_P: f32 = 0.60;  // storeys whose doors are open
const cc_facadeworks_BAL_UNIT_P: f32 = 0.45;   // doors within such a storey
// One door per window pitch, lighting the middle two thirds of it: what
// breaks the line into a dashed rhythm instead of a rule.
const cc_facadeworks_BAL_DOOR_LO: f32 = 0.18;
const cc_facadeworks_BAL_DOOR_HI: f32 = 0.82;
const cc_facadeworks_BAL_DOOR_FRAC: f32 = 0.64;  // HI - LO
// Mean of the soft vertical profile below over the GLOW_H window.
const cc_facadeworks_BAL_PROFILE_MEAN: f32 = 0.625;
const cc_facadeworks_BAL_MAX_H: f32 = 260.0;   // towers have sealed walls

// --- Fire escapes ----------------------------------------------------------
const cc_facadeworks_FE_FRAC: u32 = 38u;        // of 256 buildings (~15%)
const cc_facadeworks_FE_HW: f32 = 1.50;        // half-width of the column (m)
// 0.30 was the brief's number and it renders as a solid tan scaffold at
// exposure 6 — brighter than most of the windows behind it. This is the
// rim a fire escape catches off a street lamp, so it is set where it reads
// as structure on black and nothing more, and it falls off with height
// because the lamp that lights it is on the ground.
const cc_facadeworks_FE_RADIANCE: f32 = 0.055;
const cc_facadeworks_FE_FALL_H: f32 = 55.0;    // e-fold of the street lamp
const cc_facadeworks_FE_COLOR: vec3<f32> = vec3<f32>(1.0, 0.55, 0.28);
const cc_facadeworks_FE_MEAN: f32 = 0.240;     // ink coverage of the lattice

// Which tier of the building this facade pixel belongs to, and how far the
// wall runs in facade-u. The u-extent is what lets signage and fire escapes
// sit a fixed distance from a CORNER rather than at a hashed offset that
// would sometimes land mid-wall and sometimes off the end of it.
struct cc_facadeworks_Face {
    u0: f32,
    u1: f32,
    z1: f32,      // top of this tier
    base: bool,   // this is tier 1, the main mass
    side: i32,    // 0 = +x wall, 1 = -x, 2 = +y, 3 = -y
}

fn cc_facadeworks_face(cc: CityCell, h: CityHit) -> cc_facadeworks_Face {
    let t2 = cc.tiers >= 2 && h.pos.z > cc.b2min.z;
    let t3 = cc.tiers >= 3 && h.pos.z > cc.b3min.z;
    let bmn = select(select(cc.b1min, cc.b2min, t2), cc.b3min, t3);
    let bmx = select(select(cc.b1max, cc.b2max, t2), cc.b3max, t3);
    // The core's facade tangent, so our uc is exactly the core's uc.
    let tangent = vec2<f32>(-h.normal.y, h.normal.x);
    let a = dot(bmn.xy, tangent);
    let b = dot(bmx.xy, tangent);
    let side = select(
        select(select(3, 2, h.normal.y > 0.5), 1, h.normal.x < -0.5),
        0, h.normal.x > 0.5);
    return cc_facadeworks_Face(min(a, b), max(a, b), bmx.z, !(t2 || t3), side);
}

// One glyph block: up to five strokes from an alphabet of three horizontals
// and two verticals, each present with probability 1/2. Not a script — the
// point is that a stack of them reads as writing at a glance and as nothing
// in particular on inspection, which is what neon signage in a language you
// do not speak looks like.
fn cc_facadeworks_glyph(bits: u32, gu: f32, gv: f32) -> f32 {
    let inl = gu > 0.12 && gu < 0.88;   // horizontal strokes stop short
    let inb = gv > 0.10 && gv < 0.90;   // vertical strokes stop short
    let on = (inl && (bits & 1u) != 0u && abs(gv - 0.86) < 0.10)
          || (inl && (bits & 2u) != 0u && abs(gv - 0.50) < 0.10)
          || (inl && (bits & 4u) != 0u && abs(gv - 0.14) < 0.10)
          || (inb && (bits & 8u) != 0u && abs(gu - 0.20) < 0.075)
          || (inb && (bits & 16u) != 0u && abs(gu - 0.80) < 0.075);
    return select(0.0, 1.0, on);
}

// Layer 1. `s` is the building's structural draw (see cc_facadeworks_detail).
fn cc_facadeworks_sign(cc: CityCell, f: cc_facadeworks_Face, s: vec2<u32>,
                       uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    // Cheap rejects first: one bit-slice of the building draw, then whether
    // this wall carries the sign, then whether this mass can carry one at
    // all. Signs live on tier 1 only — one that climbed onto a setback would
    // hang off the inset wall's corner instead of the street's.
    //
    // The draw picks a CORNER of the plan, not a wall, and the sign box is
    // mounted on it, so both walls meeting there carry the strip, on
    // whichever of their two u-ends that corner is. One sign, two faces —
    // which is how corner signage actually works, and it is also what makes
    // the layer land at street level: gating on a single wall put a sign in
    // front of the camera roughly never, since 12% of buildings times one
    // wall in four is 3% of the walls in a frame and a street view holds
    // about a dozen.
    //
    // The wall->corner table: wall s has corner (0x0231 >> 4s) & 3 at its u0
    // end and that corner plus one at its u1 end, walking the plan corners
    // (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax) counterclockwise.
    let c0 = i32((0x0231u >> (4u * u32(f.side))) & 3u);
    let corner = i32((s.x >> 8u) & 3u);
    let right_edge = corner == ((c0 + 1) & 3);
    if ((s.x & 255u) >= cc_facadeworks_SIGN_FRAC
        || !(right_edge || corner == c0)
        || !f.base || f.z1 <= 17.0 || (f.u1 - f.u0) <= 6.5) {
        return vec3<f32>(0.0);
    }
    // The sign is sized to the mass it hangs on. A fixed 12-40 m strip
    // starting at 6 m is right for a mid-rise and invisible on everything
    // else: soar is a fly-through whose camera lives well above 46 m, and
    // this city's megatower tier-1 walls are a KILOMETRE of facade that
    // would carry a 40 m mark at the very bottom and nothing above it. So a
    // tall wall takes a length proportional to itself instead — a ribbon
    // running most of the tower, which is the Cloudpunk cue — while a
    // mid-rise keeps the fixed law exactly. The cross-section and the glyph
    // blocks scale as the square root of the length ratio, which holds the
    // character aspect fixed while a tower's sign reads at a tower's size.
    let r_len = city_u01(s.y);
    let tall = mix(0.0, 0.85, smoothstep(60.0, 600.0, f.z1))
               * (0.5 + 0.5 * r_len);
    let len = max(12.0 + 28.0 * r_len, tall * f.z1);
    let gscale = sqrt(clamp(len / 26.0, 1.0, 12.0));
    let hw0 = cc_facadeworks_SIGN_HW * gscale;
    let seg_pitch = cc_facadeworks_SEG_PITCH * gscale;
    let seg_h = cc_facadeworks_SEG_H * gscale;

    let v0 = cc_facadeworks_SIGN_V0;
    let v1 = min(v0 + len, f.z1 - 2.0);
    if (v1 - v0 < 6.0 || vc < v0 - 1.0 || vc > v1 + 1.0) {
        return vec3<f32>(0.0);
    }
    let uanch = select(f.u0 + 1.9 * gscale, f.u1 - 1.9 * gscale, right_edge);
    let du = abs(uc - uanch);
    // The strip's cross-section, widened to the pixel once it is thinner
    // than one and dimmed by the same factor: the integral across u is
    // invariant, so a sign neither strobes nor washes out with range.
    let hw = max(hw0, 0.50 * fp);
    let soft = 0.30 * hw;
    let sig = max(cc_facadeworks_SIGN_HALO_SIG * gscale, 0.65 * fp);
    if (du > max(hw, 2.6 * sig)) {
        return vec3<f32>(0.0);
    }

    // The sign's own draw: color, brightness, and whether a third of its
    // tubes are out. No time in this shader, so a broken sign is permanently
    // broken rather than flickering — the dead blocks are the tell either way.
    let sh = city_rand4(cc.seed ^ vec2<u32>(0x1d8e4a37u, 0x6c9f2b05u));
    // Pushed draws: three explicit slots in city_window_color's palette at a
    // fixed bias, so signage lands on saturated amber / cyan / magenta and
    // never on the fluorescent white that would read as an office window.
    let sdraw = select(select(0.20, 0.89, sh.x > 0.34), 0.97, sh.x > 0.67);
    let scol = city_window_color(sdraw, 0.5);
    let rad = 5.0 + 3.0 * sh.y;
    let dead_frac = select(0.04, 0.35, sh.z < 0.15);

    // Glyph blocks up the strip.
    let sv = vc - v0;
    let iseg = i32(floor(sv / seg_pitch));
    let fsv = sv - f32(iseg) * seg_pitch;
    let sb = pcg2d(vec2<u32>(cc.seed.x ^ 0x51a3f7ddu,
                             bitcast<u32>(iseg) * 0x9e3779b9u));
    let alive = select(0.0, 1.0, city_u01(sb.y) >= dead_frac);
    let gu = (uc - (uanch - hw0)) / (2.0 * hw0);
    let ink = cc_facadeworks_glyph(sb.x, gu, fsv / seg_h);

    // The ladder. Strokes are ~0.2 m, so they go sub-pixel well before the
    // 1.15 m block pitch does: glyph interiors dissolve into the block mean
    // first, then blocks and their gaps dissolve into a continuous line
    // carrying the same mean energy. Both thresholds ride gscale, because a
    // tower's sign is physically bigger and stays resolved further out.
    let l1 = smoothstep(0.30 * gscale, 1.5 * gscale, fp);
    let l2 = smoothstep(1.2 * gscale, 6.0 * gscale, fp);
    let in_seg = select(0.0, 1.0, fsv < seg_h);
    let pan = cc_facadeworks_SIGN_PANEL;
    let lum = mix(ink, cc_facadeworks_GLYPH_MEAN, l1) * (1.0 - pan) + pan;
    let near = lum * in_seg * alive;
    let far = (cc_facadeworks_GLYPH_MEAN * (1.0 - pan) + pan)
              * cc_facadeworks_SEG_DUTY * (1.0 - dead_frac);
    let val = mix(near, far, l2);

    // Profile across the strip (integral 2*hw - soft, hence the amplitude)
    // and the wall spill either side of it, which is what tells the eye the
    // sign is a tube in front of a wall and not a painted rectangle.
    let prof = 1.0 - smoothstep(hw - soft, hw, du);
    let amp = hw0 / max(hw - 0.5 * soft, 1e-3);
    let halo = cc_facadeworks_SIGN_HALO_AMP
               * (cc_facadeworks_SIGN_HALO_SIG * gscale / sig) * far
               * exp(-0.5 * du * du / (sig * sig));
    // Both ends written ascending: smoothstep with edge0 > edge1 is undefined
    // in WGSL, however reasonably a given backend happens to evaluate it.
    let vgate = smoothstep(v0 - 0.3, v0 + 0.3, vc)
                * (1.0 - smoothstep(v1 - 0.3, v1 + 0.3, vc));
    return scol * (rad * vgate * (prof * amp * val + halo));
}

// Two gates: whether this storey's doors are open at all, and which of them
// along it spill. The second is what keeps the line dashed — a continuous
// band ruled across a whole wall reads as a stripe, and balconies are
// supposed to read as rhythm. One door per window pitch, and the spill
// fades top and bottom rather than ending on an edge, because it is light
// falling on a ledge and not a painted line.
fn cc_facadeworks_bal_lit(cc: CityCell, per: i32, uc: f32, vc: f32) -> f32 {
    let ivf = floor(vc / CITY_FLOOR_H);
    let iv = i32(ivf);
    let vloc = vc - ivf * CITY_FLOOR_H;
    if (((iv % per) + per) % per != 0
        || vloc <= cc_facadeworks_BAL_GLOW_V0
        || vloc >= cc_facadeworks_BAL_GLOW_V0 + cc_facadeworks_BAL_GLOW_H) {
        return 0.0;
    }
    let fu = fract(uc / CITY_WIN_PITCH_U);
    if (fu <= cc_facadeworks_BAL_DOOR_LO || fu >= cc_facadeworks_BAL_DOOR_HI) {
        return 0.0;
    }
    let fb = pcg2d(vec2<u32>(cc.seed.x ^ 0x3c6ef372u,
                             bitcast<u32>(iv) * 0x85ebca6bu));
    if (city_u01(fb.x) >= cc_facadeworks_BAL_FLOOR_P) {
        return 0.0;
    }
    let iu = i32(floor(uc / CITY_WIN_PITCH_U));
    let ub = pcg2d(vec2<u32>(fb.y, bitcast<u32>(iu) * 0xc2b2ae35u));
    if (city_u01(ub.x) >= cc_facadeworks_BAL_UNIT_P) {
        return 0.0;
    }
    let q = abs(vloc - (cc_facadeworks_BAL_GLOW_V0
                        + 0.5 * cc_facadeworks_BAL_GLOW_H))
            / (0.5 * cc_facadeworks_BAL_GLOW_H);
    return 1.0 - smoothstep(0.25, 1.0, q);
}

// Layer 2.
fn cc_facadeworks_balcony(cc: CityCell, f: cc_facadeworks_Face, s: vec2<u32>,
                          uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    if (((s.x >> 11u) & 255u) >= cc_facadeworks_BAL_FRAC
        || cc.height >= cc_facadeworks_BAL_MAX_H) {
        return vec3<f32>(0.0);
    }
    let per = select(2, 3, ((s.x >> 19u) & 1u) != 0u);
    let duty = cc_facadeworks_BAL_GLOW_H * cc_facadeworks_BAL_PROFILE_MEAN
               / (f32(per) * CITY_FLOOR_H);
    let far = duty * cc_facadeworks_BAL_FLOOR_P * cc_facadeworks_BAL_UNIT_P
              * cc_facadeworks_BAL_DOOR_FRAC;
    // A 0.45 m spill at a 7.2-10.8 m pitch is sub-pixel past ~1 m/px, so it
    // dissolves early. Its mean is ~0.0005 radiance — three orders under a
    // lit facade — which is why it can dissolve without leaving a mark.
    let l = smoothstep(0.40, 1.5, fp);
    if (l >= 0.999) {
        return cc_facadeworks_BAL_COLOR * (cc_facadeworks_BAL_RADIANCE * far);
    }
    let near = cc_facadeworks_bal_lit(cc, per, uc, vc);
    return cc_facadeworks_BAL_COLOR
           * (cc_facadeworks_BAL_RADIANCE * mix(near, far, l));
}

// Layer 3.
fn cc_facadeworks_escape(cc: CityCell, f: cc_facadeworks_Face, s: vec2<u32>,
                         uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    // Filigree: fade the pattern into its mean first, then let that mean —
    // 0.013 radiance, a two-hundredth of a lit window — go out over a wide
    // footprint band. Fading a pedestal that small across two octaves of
    // range is not a pop; fading the PATTERN would have been.
    let l2 = smoothstep(2.0, 6.0, fp);
    if (((s.x >> 20u) & 255u) >= cc_facadeworks_FE_FRAC
        || cc.height <= 30.0 || cc.height >= 120.0
        || i32((s.x >> 28u) & 3u) != f.side
        || !f.base || (f.u1 - f.u0) <= 13.0 || l2 >= 0.999) {
        return vec3<f32>(0.0);
    }
    // Set in from the corner far enough to clear a sign strip on the same
    // wall (the two land together on well under 1% of buildings, but where
    // they do they should be neighbours, not the same column).
    let ufe = select(f.u0 + 4.5, f.u1 - 4.5, ((s.x >> 30u) & 1u) != 0u);
    let du = uc - ufe;
    if (abs(du) > cc_facadeworks_FE_HW) {
        return vec3<f32>(0.0);
    }
    let x = (du + cc_facadeworks_FE_HW) / (2.0 * cc_facadeworks_FE_HW);
    let ivf = floor(vc / CITY_FLOOR_H);
    let y = vc / CITY_FLOOR_H - ivf;
    // Stringers alternate direction storey to storey: the switchback is the
    // whole silhouette of a fire escape.
    let xx = select(x, 1.0 - x, ((i32(ivf) % 2) + 2) % 2 == 1);
    let on = abs(y - xx) < 0.055        // stair stringer
          || y < 0.055                  // landing at the floor line
          || abs(x - 0.055) < 0.024     // rails
          || abs(x - 0.945) < 0.024;
    let l1 = smoothstep(0.35, 1.6, fp);
    let val = mix(select(0.0, 1.0, on), cc_facadeworks_FE_MEAN, l1)
              * (1.0 - l2);
    let reach = 0.25 + 0.75 * exp(-max(vc - 6.0, 0.0)
                                  / cc_facadeworks_FE_FALL_H);
    return cc_facadeworks_FE_COLOR
           * (cc_facadeworks_FE_RADIANCE * reach * val);
}

fn cc_facadeworks_detail(cc: CityCell, h: CityHit, uc: f32, vc: f32,
                         fp: f32) -> vec3<f32> {
    // One draw per building for everything structural. pcg2d gives 64 bits;
    // slicing them beats three city_rand4 calls on a hook that runs at every
    // facade pixel, and the continuous draws each layer needs are taken
    // later, behind that layer's own reject.
    //
    //   bits  0-7   sign presence      8-9   sign wall      10  sign edge
    //        11-18  balcony presence    19   balcony pitch
    //        20-27  escape presence   28-29  escape wall     30  escape edge
    // (bit 10 is spare: signage picks a corner, so it needs no edge bit.)
    //   s.y         sign length
    let s = pcg2d(cc.seed ^ vec2<u32>(0x7f4a7c15u, 0x2b3d8e11u));
    let f = cc_facadeworks_face(cc, h);
    return cc_facadeworks_sign(cc, f, s, uc, vc, fp)
         + cc_facadeworks_balcony(cc, f, s, uc, vc, fp)
         + cc_facadeworks_escape(cc, f, s, uc, vc, fp);
}
