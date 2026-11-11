// facadeworks — additive facade detail on top of the core's windows.
//
// VERTICAL NEON SIGNAGE. The cyberpunk cue. A hashed minority of buildings
// (~12%) carries one narrow vertical sign box mounted on a corner of the
// main mass, so both walls meeting there show it: saturated amber / cyan /
// magenta at radiance 5-8, broken into glyph-like blocks that suggest
// characters without being any script. A mid-rise gets a 12-40 m strip
// starting 6 m up; a tower gets a ribbon proportional to its own height. At
// street range it should read as typography, and from kilometres out it
// should still be a colored vertical line, which is what makes a skyline
// read as a CITY and not a lit grid.
//
// (Balconies and fire escapes used to be drawn here as light on a flat wall;
// they are the core facade shell's now, in 3D, with their flat twins in
// city_facade_flat.)
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

// Which tier of the building this facade pixel belongs to, and how far the
// wall runs in facade-u. The u-extent is what lets signage sit a fixed
// distance from a CORNER rather than at a hashed offset that
// would sometimes land mid-wall and sometimes off the end of it.
struct cc_facadeworks_Face {
    u0: f32,
    u1: f32,
    z1: f32,      // top of this tier
    base: bool,   // this is tier 1, the main mass
    side: i32,    // 0 = +x wall, 1 = -x, 2 = +y, 3 = -y
}

// The core's own face lookup, so our uc (metres from the face's start
// corner) is exactly the core's. A frustum skin is no one's base wall.
fn cc_facadeworks_face(cc: CityCell, h: CityHit) -> cc_facadeworks_Face {
    let f = city_face_at(cc, h.pos, h.normal);
    return cc_facadeworks_Face(0.0, f.len, f.z1, f.aligned && f.tier == 0,
                               f.side);
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

// Where this wall's sign strip sits, if it has one: (u of the strip's
// centre, half-width, v0, v1), half-width 0 for none. Shared by the sign
// and by the cover it reports, so the wall it blanks is the wall it lights.
fn cc_facadeworks_sign_box(cc: CityCell, f: cc_facadeworks_Face,
                           s: vec2<u32>) -> vec4<f32> {
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
    // No signs on a warehouse: the industry component owns its walls.
    if ((s.x & 255u) >= cc_facadeworks_SIGN_FRAC
        || cc.land_use == CITY_USE_INDUSTRIAL
        || !(right_edge || corner == c0)
        || !f.base || f.z1 <= 17.0 || (f.u1 - f.u0) <= 6.5) {
        return vec4<f32>(0.0);
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

    let v0 = cc_facadeworks_SIGN_V0;
    let v1 = min(v0 + len, f.z1 - 2.0);
    if (v1 - v0 < 6.0) {
        return vec4<f32>(0.0);
    }
    let uanch = select(f.u0 + 1.9 * gscale, f.u1 - 1.9 * gscale, right_edge);
    return vec4<f32>(uanch, hw0, v0, v1);
}

// Layer 1. `s` is the building's structural draw (see cc_facadeworks_detail).
fn cc_facadeworks_sign(cc: CityCell, f: cc_facadeworks_Face, s: vec2<u32>,
                       uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    let sbox = cc_facadeworks_sign_box(cc, f, s);
    let hw0 = sbox.y;
    let v0 = sbox.z;
    let v1 = sbox.w;
    if (hw0 <= 0.0 || vc < v0 - 1.0 || vc > v1 + 1.0) {
        return vec3<f32>(0.0);
    }
    let gscale = hw0 / cc_facadeworks_SIGN_HW;
    let seg_pitch = cc_facadeworks_SEG_PITCH * gscale;
    let seg_h = cc_facadeworks_SEG_H * gscale;
    let uanch = sbox.x;
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

fn cc_facadeworks_draw(cc: CityCell) -> vec2<u32> {
    return pcg2d(cc.seed ^ vec2<u32>(0x7f4a7c15u, 0x2b3d8e11u));
}

fn cc_facadeworks_detail(cc: CityCell, h: CityHit, dir: vec3<f32>,
                         uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    // One draw per building for the sign's structure. pcg2d gives 64 bits;
    // the continuous draws are taken later, behind the sign's own reject.
    //
    //   bits  0-7   sign presence      8-9   sign corner
    //   s.y         sign length
    let s = cc_facadeworks_draw(cc);
    let f = cc_facadeworks_face(cc, h);
    return cc_facadeworks_sign(cc, f, s, uc, vc, fp);
}

// The sign box is a box on the wall: where it hangs there are no windows
// and no relief behind it, only the sign.
fn cc_facadeworks_cover(cc: CityCell, h: CityHit, uc: f32, vc: f32,
                        fp: f32) -> f32 {
    let s = cc_facadeworks_draw(cc);
    if ((s.x & 255u) >= cc_facadeworks_SIGN_FRAC) {
        return 0.0;
    }
    let sb = cc_facadeworks_sign_box(cc, cc_facadeworks_face(cc, h), s);
    let inside = sb.y > 0.0 && abs(uc - sb.x) < sb.y
              && vc > sb.z && vc < sb.w;
    return select(0.0, 1.0, inside);
}
