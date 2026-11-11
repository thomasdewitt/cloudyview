// The city under this ray: fogged radiance and distance, t = 1e30 on a miss.
fn surface_pass_city(dir: vec3<f32>, pixel_angle: f32) -> vec4<f32> {
    // THE CITY FRAME: world xy minus the row-24 surface offset, folded into
    // the primary tile (city_camera_origin — the one entry). The whole city
    // system — the DDA, every cell index and pcg2d seed, plot slabs, street
    // lines, component routes, hit positions, the fog's glow probe — runs in
    // this frame, and because the city is a pure function of it, the
    // frame's whole-tile ambiguity is invisible. t values are
    // translation-invariant, so depth compositing against the cloud march
    // needs no conversion; the one city input that must STAY world is the
    // uplight probe's cloud sampling (see the city_glow_sample call in the
    // march loop).
    //
    // This stage traces and shades only. What stands on the street and what
    // the street reflects are the street stage's (surface_street_city): kept
    // out of here, this stage stays in master's register tier (128 on the
    // 5080; the wet reflection, the steam and the street furniture compiled
    // in took it to 168, and every city pixel paid for that occupancy). A
    // ground hit is FLAGGED for that stage by a negative t — city-facades'
    // refinement passes it through, sign and all — and the street stage
    // writes it back positive.
    let city_o = city_camera_origin();
    let chit = city_trace(city_o, dir, 0.0, CITY_TRACE_RANGE, CITY_TRACE_ALL);
    // The refinement stage's flag, CITY_SHELL_MARK: the sign bit of red
    // (radiance is never negative) marks a pixel whose ray entered a
    // building's relief envelope within the shell's reach before its plain
    // hit, the only pixels the facade shell can change. surface_refine_city
    // returns every other pixel untouched and clears the bit on every marked
    // one, so no later reader sees it. A bit rather than a negated value: a
    // miss has red 0, and -0.0 only survives as a bit. (t's sign is the
    // street stage's ground flag, below.)
    let t_lim = min(select(1e30, chit.t, chit.hit),
                    city_shell_reach(city_o, dir, pixel_angle));
    let near = city_shell_on_wall(chit, t_lim)
               || city_shell_envelope(city_o, dir, t_lim);
    let mark = select(0u, CITY_SHELL_MARK, near);
    if (!chit.hit) {
        return vec4<f32>(bitcast<f32>(mark), 0.0, 0.0, 1e30);
    }
    let rgb = city_fog(city_shade(chit, dir, pixel_angle * chit.t,
                                  true_pixel_angle() * chit.t, true),
                       city_o, dir, chit.t, chit.pos);
    // Flagged only where a reflection can be traced: past CITY_REFL_LOD the
    // wet street's reflection is its environment mean, measured at under
    // 1e-4 of display brightness on the wide views (base 8e-5, horizon
    // 3e-5), while evaluating the ground material again for every distant
    // ground pixel just to weight it cost the street stage ~12% on the
    // horizon view. So it is left out, and the stand-in only fades the
    // traced reflection out across the hand-off band.
    let flag = chit.kind == 0 && pixel_angle * chit.t < CITY_REFL_LOD.y;
    return vec4<f32>(bitcast<f32>(bitcast<u32>(rgb.x) | mark), rgb.yz,
                     select(chit.t, -chit.t, flag));
}

// The refinement stage: the facade shell, traced where the plain surface
// pass left the camera's view within its reach. The plain pass (city_trace)
// found t_prev, the nearest opaque thing along the ray with every building a
// plain box; this walks the same cells again, but only as far as
// min(t_prev, the shell's reach), testing each building's box inflated by
// its relief. A building that yields a nearer hit — relief standing proud
// of a wall the ray would have missed or met later — replaces t_prev; so
// does the building that OWNED t_prev (its own wall or roof), whose shell
// may put the hit deeper (the glass behind a reveal) or keep it where it was.
// Everything else the plain pass found stands. Returns the hit that
// replaces t_prev, or no hit when t_prev stands.
const CITY_SHELL_REFINE_MAX_CELLS: i32 = 64;
// How far along this ray the block walk reaches before its cell cap: the
// shell's reach ends there too. Only a narrow view at a high resolution
// with the view-step LOD at 0 ever sees it (4K at a 12 degree fov: 90 m
// cells, about 4.8 km); per ray, continuous in direction, since each step
// covers cell / (|dx| + |dy|).
fn city_shell_cap(o: vec3<f32>, dir: vec3<f32>) -> f32 {
    var t0 = 0.0;
    if (o.z > CITY_SLAB_TOP && dir.z < -1e-7) {
        t0 = (CITY_SLAB_TOP - o.z) / dir.z;
    }
    return t0 + f32(CITY_SHELL_REFINE_MAX_CELLS - 2) * u.ocean_params.x
                / max(abs(dir.x) + abs(dir.y), 1e-6);
}

// The relief's fade to flat before the cap: 0 until the last quarter, 1 at it.
fn city_shell_cap_fade(t: f32, t_cap: f32) -> f32 {
    return smoothstep(0.75 * t_cap, t_cap, t);
}

// The shell's reach along a ray: its footprint limit, or the walk's cap.
fn city_shell_reach(o: vec3<f32>, dir: vec3<f32>, pixel_angle: f32) -> f32 {
    return min(CITY_SHELL_FP_FLAT / max(pixel_angle, 1e-9),
               city_shell_cap(o, dir));
}

// surface_pass_city's mark for the refinement: the sign bit of red.
const CITY_SHELL_MARK: u32 = 0x80000000u;

// The block-grid walk the shell's two passes share: the envelope probe and
// the refinement. The same DDA as city_trace, from the slab top.
struct CityWalk {
    ci: vec2<i32>,
    stp: vec2<i32>,
    inv3: vec3<f32>,
    tmax: vec2<f32>,
    tdelta: vec2<f32>,
    t_cur: f32,
}

// `live` false: the ray never enters the slab.
fn city_walk_start(o: vec3<f32>, dir: vec3<f32>, w: ptr<function, CityWalk>)
        -> bool {
    let cell = u.ocean_params.x;
    var t0 = 0.0;
    if (o.z > CITY_SLAB_TOP) {
        if (dir.z >= -1e-7) {
            return false;
        }
        t0 = (CITY_SLAB_TOP - o.z) / dir.z;
    }
    let p0 = o + t0 * dir;
    let ci = vec2<i32>(floor(p0.xy / cell));
    let inv3 = vec3<f32>(
        select(1e30, 1.0 / dir.x, abs(dir.x) > 1e-9),
        select(1e30, 1.0 / dir.y, abs(dir.y) > 1e-9),
        select(1e30, 1.0 / dir.z, abs(dir.z) > 1e-9));
    let cmin0 = vec2<f32>(ci) * cell;
    (*w).ci = ci;
    (*w).stp = vec2<i32>(select(-1, 1, dir.x > 0.0),
                         select(-1, 1, dir.y > 0.0));
    (*w).inv3 = inv3;
    (*w).tmax = vec2<f32>(
        select(1e30, t0 + (select(cmin0.x, cmin0.x + cell, dir.x > 0.0)
                           - p0.x) * inv3.x, abs(dir.x) > 1e-9),
        select(1e30, t0 + (select(cmin0.y, cmin0.y + cell, dir.y > 0.0)
                           - p0.y) * inv3.y, abs(dir.y) > 1e-9));
    (*w).tdelta = vec2<f32>(cell * abs(inv3.x), cell * abs(inv3.y));
    (*w).t_cur = t0;
    return true;
}

fn city_walk_step(w: ptr<function, CityWalk>) {
    if ((*w).tmax.x < (*w).tmax.y) {
        (*w).ci.x = (*w).ci.x + (*w).stp.x;
        (*w).t_cur = (*w).tmax.x;
        (*w).tmax.x = (*w).tmax.x + (*w).tdelta.x;
    } else {
        (*w).ci.y = (*w).ci.y + (*w).stp.y;
        (*w).t_cur = (*w).tmax.y;
        (*w).tmax.y = (*w).tmax.y + (*w).tdelta.y;
    }
}

// Does the ray enter this building's relief envelope (every tier's box grown
// by CITY_SHELL_OUT_MAX) by t_lim? Everything the shell can return for it —
// relief, glass, its plain slab — lies inside.
fn city_shell_enters(o: vec3<f32>, inv3: vec3<f32>, cc: CityCell,
                     t_lim: f32) -> bool {
    let grow = vec3<f32>(CITY_SHELL_OUT_MAX, CITY_SHELL_OUT_MAX, 0.0);
    for (var k: i32 = 0; k < cc.tiers; k = k + 1) {
        let bx = city_tier_box(cc, k);
        let hg = city_box_hit(o, inv3, bx.bmin - grow, bx.bmax + grow);
        if (hg.x <= hg.y && hg.y > 0.0 && max(hg.x, 0.0) <= t_lim) {
            return true;
        }
    }
    // The crown's envelope: first where a crown could stand, without its
    // draw (the summit deck under the crown's headroom), and only a ray
    // that enters that pays for the draw. The bound alone marked every
    // ray over a tall tower's roof, four in five of them crownless.
    if (city_crown_possible(cc)) {
        let bd = city_crown_bound(cc);
        let hb = city_box_hit(o, inv3, bd.bmin - grow, bd.bmax + grow);
        if (hb.x <= hb.y && hb.y > 0.0 && max(hb.x, 0.0) <= t_lim) {
            let cb = city_crown_box(cc);
            let hc = city_box_hit(o, inv3, cb.bmin - grow, cb.bmax + grow);
            return cb.bmin.x <= cb.bmax.x && hc.x <= hc.y && hc.y > 0.0
                   && max(hc.x, 0.0) <= t_lim;
        }
    }
    return false;
}

// A plain hit on a building's wall, inside the shell's reach, is inside that
// building's envelope, which the ray entered strictly before it: the 1.5 m
// the envelope stands out from the wall. Most marked pixels are these, and
// they need no walk. (The z bounds are open: a hit on a box edge is left to
// the walk.)
fn city_shell_on_wall(h: CityHit, t_lim: f32) -> bool {
    if (!h.hit || h.kind != 1 || h.t >= t_lim) {
        return false;
    }
    let cc = city_cell(h.cell);
    if (!cc.built || cc.land_use == CITY_USE_INDUSTRIAL) {
        return false;
    }
    let grow = CITY_SHELL_OUT_MAX - 0.01;
    // A wall hit above the roof is a crown's (the mast is kind 3): inside
    // the crown's envelope, as a tier's wall is inside its own.
    if (h.pos.z > cc.height + 0.01 && city_crown_possible(cc)) {
        return true;
    }
    for (var k: i32 = 0; k < cc.tiers; k = k + 1) {
        let bx = city_tier_box(cc, k);
        if (all(h.pos.xy > bx.bmin.xy - grow)
            && all(h.pos.xy < bx.bmax.xy + grow)
            && h.pos.z > bx.bmin.z && h.pos.z < bx.bmax.z) {
            return true;
        }
    }
    return false;
}

// Does the ray enter any building's relief envelope (its boxes grown by the
// deepest relief, CITY_SHELL_OUT_MAX) before t_lim? Only such a pixel can
// hold a shell hit. The plain pass asks after its own trace, with a walk of
// its own: carried through city_trace's loop, even one running minimum cost
// that kernel 32 registers.
fn city_shell_envelope(o: vec3<f32>, dir: vec3<f32>, t_lim: f32) -> bool {
    var w: CityWalk;
    if (!city_walk_start(o, dir, &w)) {
        return false;
    }
    for (var i: i32 = 0; i < CITY_SHELL_REFINE_MAX_CELLS; i = i + 1) {
        if (w.t_cur >= t_lim) {
            return false;
        }
        let t_exit = min(w.tmax.x, w.tmax.y);
        let z_lo = min(o.z + w.t_cur * dir.z, o.z + t_exit * dir.z);
        let cc = city_cell(w.ci);
        if (cc.built && cc.land_use != CITY_USE_INDUSTRIAL
            && z_lo < cc.top_z + 1.0
            && city_shell_enters(o, w.inv3, cc, t_lim)) {
            return true;
        }
        city_walk_step(&w);
    }
    return false;
}

fn city_shell_refine(o: vec3<f32>, dir: vec3<f32>, pixel_angle: f32,
                     t_prev: f32) -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    // Past this every box is its plain slab, which the plain pass has done.
    let t_cap = city_shell_cap(o, dir);
    let t_reach = min(t_prev, city_shell_reach(o, dir, pixel_angle));
    var w: CityWalk;
    if (!city_walk_start(o, dir, &w)) {
        return res;
    }
    let inv3 = w.inv3;
    var best = CityBoxHit(1e30, vec3<f32>(0.0), -1);
    var best_cell = w.ci;
    var owns_prev = false;
    for (var i: i32 = 0; i < CITY_SHELL_REFINE_MAX_CELLS; i = i + 1) {
        if (w.t_cur >= t_reach || w.t_cur >= best.t) {
            break;
        }
        let ci = w.ci;
        let t_exit = min(w.tmax.x, w.tmax.y);
        let z_a = o.z + w.t_cur * dir.z;
        let z_b = o.z + t_exit * dir.z;
        let cc = city_cell(ci);
        // Industrial walls are the industry component's (city-layout's):
        // cladding, dock doors and their own relief. The shell leaves them.
        if (cc.built && cc.land_use != CITY_USE_INDUSTRIAL
            && min(z_a, z_b) < cc.top_z + 1.0
            && city_shell_enters(o, inv3, cc, min(best.t, t_reach))) {
            let st = city_shell_style(cc);
            // The crown's draw, behind this cell's ray segment reaching the
            // crown's band (city_shell_enters may have drawn it too).
            var cb = CityBox(vec3<f32>(1e30), vec3<f32>(-1e30));
            if (city_crown_possible(cc) && max(z_a, z_b) >= cc.height
                && min(z_a, z_b) <= cc.height + CITY_CROWN_H_HI) {
                cb = city_crown_box(cc);
            }
            for (var k: i32 = 0; k < CITY_CROWN_TIER + 1; k = k + 1) {
                if (!city_mass_has(cc, cb, k)) {
                    continue;
                }
                let bx = city_mass_box(cc, cb, k);
                let bh = city_building_box(o, dir, inv3, t_cap, bx.bmin,
                                           bx.bmax, cc, k, st, pixel_angle);
                if (bh.kind >= 0 && bh.t < best.t) {
                    best = bh;
                    best_cell = ci;
                    // Did this building own the plain pass's hit? Its core
                    // slab entry is computed exactly as the plain pass did.
                    let hc = city_box_hit(o, inv3, bx.bmin, bx.bmax);
                    owns_prev = hc.x <= hc.y && hc.x > 0.0
                                && abs(hc.x - t_prev) <= 1e-5 * t_prev;
                }
            }
        }
        city_walk_step(&w);
    }
    // Only a shell hit changes anything: a plain one is what t_prev already
    // is. A shell hit stands if it is nearer, or if it belongs to the
    // building whose wall t_prev was.
    if (best.kind >= CITY_KIND_SHELL
        && (best.t < t_prev || owns_prev)) {
        res.hit = true;
        res.t = best.t;
        res.pos = o + best.t * dir;
        res.normal = best.normal;
        res.kind = best.kind;
        res.cell = best_cell;
    }
    return res;
}

// The city's second stage: the facade shell over the plain surface. Its
// own kernel, because the shell's registers would otherwise be paid by every
// step of the plain pass's DDA — and this one only walks the few cells
// within the shell's reach.
fn surface_refine_city(cr: CameraRay, plain: vec4<f32>) -> vec4<f32> {
    let red = bitcast<u32>(plain.x);
    if ((red & CITY_SHELL_MARK) == 0u) {
        return plain;   // no relief envelope on this ray (surface_pass_city)
    }
    // t is read by magnitude, and a pixel the shell leaves keeps its sign
    // bit for bit: its sign belongs to whatever stage follows.
    let t_prev = abs(plain.w);
    let city_o = city_camera_origin();
    let h = city_shell_refine(city_o, cr.dir, cr.pixel_angle, t_prev);
    if (!h.hit) {
        return vec4<f32>(bitcast<f32>(red & ~CITY_SHELL_MARK), plain.yzw);
    }
    let cc = city_cell(h.cell);
    let glow = city_glow_sample(h.pos.xy, 3.0);
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * glow);
    return vec4<f32>(city_fog(city_shell_shade(h, cc, cr.dir,
                                               cr.pixel_angle * h.t, fill,
                                               glow),
                              city_o, cr.dir, h.t, h.pos), h.t);
}

// --- the street stage -----------------------------------------------------------
//
// The third city stage (fs_surface_street), after the facade refinement: it
// reads the surface as the first two left it and adds what belongs to the
// street — the furniture standing in front of it, the wet reflection off
// it, and the steam over it, which goes to the VEIL so it shows against the
// sky and in front of any wall. A pixel whose ray never comes down within
// CITY_NEAR_TOP of the street, within its reach (city_near_reach), and did
// not land on the ground, leaves at once with the surface unchanged.
const CITY_NEAR_TOP: f32 = 15.0;     // the tallest near thing: a plume
const CITY_NEAR_RANGE: f32 = 400.0;  // past the steam's cutoff
const CITY_FURNITURE_TOP: f32 = 3.0; // an open dumpster lid on the kerb
// The near_props contract (SPEC.md): no hook draws anything where the props'
// footprint, max(0.35 pixel_angle, true_pixel_angle) * t, exceeds this. The
// wall lane's fade ends there (streetfurniture WALL_FAR), and the parked
// cars the street stage owns (pixel_angle * t < 0.5) end well short of it.
const CITY_NEAR_FP: f32 = 0.75;

// How far along the ray the street stage looks: the steam's fixed range, or
// the props' footprint-bound one, whichever is longer. At high resolution the
// props reach well past CITY_NEAR_RANGE (at 1440p, fov 70, the cars' switch
// to boxes sits ~900 m out and the reach ~1370 m).
fn city_near_reach(cr: CameraRay) -> f32 {
    return max(CITY_NEAR_RANGE,
               CITY_NEAR_FP / max(0.35 * cr.pixel_angle,
                                  cr.true_pixel_angle));
}

fn surface_street_city(cr: CameraRay, prev: vec4<f32>) -> SurfaceOut {
    let dir = cr.dir;
    let pixel_angle = cr.pixel_angle;
    let ground_hit = prev.w < 0.0;
    let t_prev = abs(prev.w);
    var out: SurfaceOut;
    out.surf = vec4<f32>(prev.xyz, t_prev);
    out.veil = veil_none();
    let city_o = city_camera_origin();
    let t_low = city_t_low(city_o, dir);
    let t_near = min(t_prev, city_near_reach(cr));
    let near = t_low < t_near;
    if (!near && !ground_hit) {
        return out;
    }
    var rgb = prev.xyz;
    var t = t_prev;
    var ground = ground_hit;
    // The street furniture (the near_props hooks) stands below
    // CITY_FURNITURE_TOP, so only the stretch of the ray under it is walked:
    // at eye level a ray climbing into the sky leaves it within metres, and
    // one from above enters it just short of the street.
    var t_f0 = 0.0;
    if (city_o.z > CITY_FURNITURE_TOP) {
        t_f0 = select(1e30, (city_o.z - CITY_FURNITURE_TOP) / -dir.z,
                      dir.z < 0.0);
    }
    var t_f1 = t_near;
    if (city_o.z < CITY_FURNITURE_TOP && dir.z > 0.0) {
        t_f1 = min(t_f1, (CITY_FURNITURE_TOP - city_o.z) / dir.z);
    }
    if (t_f0 < t_f1) {
        // From the camera's own origin, so t stays the distance from the eye
        // that the props take their footprint from.
        let hp = city_trace(city_o, dir, t_f0, t_f1, CITY_TRACE_NEAR);
        if (hp.hit) {
            t = hp.t;
            // Always a component's hit (kind >= 100: no near_props hook
            // returns core kinds), so its shader directly — a second whole
            // city_shade beside the reflection's cost registers here.
            rgb = city_fog(cc_component_shade(hp, city_cell(hp.cell), dir,
                                              pixel_angle * t),
                           city_o, dir, t, hp.pos);
            ground = false;
        }
    }
    if (ground) {
        rgb = rgb + city_wet_reflection(city_o, dir, t, pixel_angle,
                                        cr.frame_index, cr.strat_pixel);
    }
    if (near && city_steam_live()) {
        // The live view draws its steam after the accumulation
        // (fs_steam_live); the veil only marks the depth it goes in at, so
        // the march notes the cloud's transmittance there.
        out.veil = veil_pack(vec3<f32>(0.0), 1.0, t_low);
    } else if (near) {
        // The steam in front of whatever is there — or of nothing: over the
        // sky the veil is all that shows of it.
        let med = cc_media(city_o, dir, city_inv_dir(dir), t, pixel_angle,
                           city_frame_rand(cr.strat_pixel, cr.frame_index,
                                           CITY_RAND_MEDIA).x);
        out.veil = veil_pack(med.rgb, med.a, t_low);
    }
    out.surf = vec4<f32>(rgb, t);
    return out;
}

// Where a ray from the camera first comes down to CITY_NEAR_TOP (0 from
// below it, 1e30 for a ray that never does): the depth the veil goes in at.
fn city_t_low(city_o: vec3<f32>, dir: vec3<f32>) -> f32 {
    var t_low = 0.0;
    if (city_o.z > CITY_NEAR_TOP) {
        t_low = select(1e30, (city_o.z - CITY_NEAR_TOP) / -dir.z, dir.z < 0.0);
    }
    return t_low;
}

// The steam's slab tests take 1/dir with an axis-parallel component at 1e30.
fn city_inv_dir(dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(select(1e30, 1.0 / dir.x, abs(dir.x) > 1e-9),
                     select(1e30, 1.0 / dir.y, abs(dir.y) > 1e-9),
                     select(1e30, 1.0 / dir.z, abs(dir.z) > 1e-9));
}

// The live view's steam (fs_steam_live), over `acc`, the accumulated image
// (display values; alpha the cloud's transmittance T_f from the camera to
// the veil's depth, as fs_main noted it), for the surface at t_surf. The
// steam is marched at the live clock exactly as the street stage marches it
// (cc_media, to the same surface, from the same depth), but once per
// presented frame, so instead of the accumulation's many offsets of its
// march it averages STEAM_LIVE_SAMPLES, stratified over the offset's range
// from a start fixed per pixel: a parked view shows the steam moving through
// a still, fine dither rather than a fresh grain every frame (one offset
// alone was a coarse grain, 2026-11-08). In linear light, with B the
// accumulated image's radiance and
// S, T_s the steam's in-scattered radiance and transmittance,
//     L = B (1 - T_f (1 - T_s)) + T_f S.
// The exact composite is C_f + T_f S + T_s (B - C_f), C_f the cloud
// radiance in front of the steam; with no cloud in front (T_f = 1, C_f = 0)
// and behind a cloud that hides it (T_f = 0) the two agree. In between this
// takes the share of B arriving from behind the veil's depth to be T_f, as
// though the cloud in front glowed like the scene behind it. What else is
// inexact here, and only here (stills and videos march the steam):
//   - B is tone_unmap of an AVERAGE of tone-mapped samples, not the average
//     radiance: a pixel whose samples differ (a window edge behind the
//     steam) is dimmed through the curve at its mean display value;
//   - T_f is averaged apart from the radiance it belongs to;
//   - a background past the curve's 1 (a lamp head, or a saturated colour
//     in the shoulder) reads as that clip, so steam in front of it dims it
//     more than it should;
//   - under contrast > 1 a background the contrast clipped to black has
//     lost its radiance: a channel displayed at exactly 0 is taken as black
//     (B = 0, the clip's low end, where thin steam then reads as itself)
//     rather than as the clip's edge, which made a plume over the night sky
//     larger and greyer than the still's (loose-ends' review, 2026-11-08);
//     a dark background just above the clip keeps its inverse;
//   - the steam takes the LAST sample's ray and surface depth, so its edge
//     along a silhouette is one sample's, not the average's.
// A pixel with no steam returns acc untouched, bit for bit. That is against
// the live accumulation, not the marched one: the live march adds a step at
// the veil's depth for every near pixel (to note T_f there), so its cloud
// samples there are not the marched view's.
const STEAM_LIVE_SAMPLES: u32 = 4u;
fn steam_live_city(cr: CameraRay, acc: vec4<f32>, t_surf: f32) -> vec4<f32> {
    let dir = cr.dir;
    let city_o = city_camera_origin();
    let t_low = city_t_low(city_o, dir);
    if (!(t_low < min(t_surf, city_near_reach(cr)))) {
        return acc;
    }
    let inv = city_inv_dir(dir);
    let start = city_frame_rand(cr.strat_pixel, 0u, CITY_RAND_MEDIA).x;
    var med = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < STEAM_LIVE_SAMPLES; k = k + 1u) {
        med = med + cc_media(city_o, dir, inv, t_surf, cr.pixel_angle,
                             (f32(k) + start) / f32(STEAM_LIVE_SAMPLES));
    }
    med = med / f32(STEAM_LIVE_SAMPLES);
    if (med.a >= 1.0 && all(med.rgb == vec3<f32>(0.0))) {
        return acc;
    }
    // fs_main's exposure and gamma, read as it reads them (the exposure
    // slider and auto-exposure keep it >= 0.5).
    let exposure = u.cam_right.w;
    let gamma = u.periodic.w;
    let t_front = acc.a;
    let b = select(tone_unmap(acc.rgb, exposure, gamma), vec3<f32>(0.0),
                   acc.rgb <= vec3<f32>(0.0));
    let l = b * (1.0 - t_front * (1.0 - med.a)) + t_front * med.rgb;
    return vec4<f32>(tone_map(l, exposure, gamma), acc.a);
}

// What the wet street at distance t along the view ray reflects, through the
// view path's fog (city_fog is affine, so it enters times that path's
// transmittance). One microfacet sample per frame (city_wet_lobe); the
// reflected ray is traced PLAIN — buildings and ground, no component hooks —
// and a lobe carrying under CITY_REFL_RR is traced only by Russian
// roulette. What the puddles lose is the parked cars, the poles, the skyway
// and the furniture; the lamps they would have shown are city_lamp_glint's,
// analytically, in the ground's own shading.
// The reflection's four draws per pixel and frame — the GGX normal (x, y),
// the lobe (z) and the roulette (w) — as one 4-D R-sequence over the frames
// (Roberts 2018: alpha_k = phi_4^-k, phi_4 the real root of x^5 = x + 1),
// rotated per pixel by an integer hash (strat_seed). A parked view's frames
// then stratify all four together, and the lobe and the roulette stay
// independent of each other, which the estimator's 1/p weights need (two
// draws on one 1-D lattice would not be). White noise (city_frame_rand)
// drew them before; the sky probe's jitter still does.
// Each dimension has a near neighbour among the strat streams the city
// runs, and two Weyl sequences whose steps differ by d move nearly in step
// for ~1/(2d) frames (loose-ends' review): x by the subpixel offset's y
// (STRAT_ALPHA_2, d 0.020, ~25 frames), y by the moon cone's u (ALPHA_5,
// 0.015, ~32), z by its v (ALPHA_6, 0.044, ~11), w by the pre-march jitter
// (ALPHA_9, 0.013, ~37). With eight 1-D streams already on the circle no
// four steps sit much farther (a search of the golden families and of
// integer strides of this lattice found 0.02 at best), and none of these
// pairs shares an integrand (a microfacet normal against a subpixel
// position, the roulette against the cloud pre-march), so early in a
// parked accumulation a pair samples near a diagonal of its square instead
// of all of it and converges regardless — the first 8 frames' error is 8 %
// lower than dev's white draws all the same.
const CITY_REFL_R4: vec4<f32> = vec4<f32>(0.8566748838545029, 0.7338918566271260,
                                          0.6287067210378086, 0.5385972572236101);
const STRAT_STREAM_CITY_REFLECT: u32 = 9u;
fn city_refl_rand(pixel: vec2<u32>, frame: u32) -> vec4<f32> {
    let a = strat_seed(pixel, STRAT_STREAM_CITY_REFLECT);
    let b = pcg2d(a);
    return fract(f32(frame) * CITY_REFL_R4
                 + vec4<f32>(vec2<f32>(a), vec2<f32>(b)) * STRAT_INV_2P32);
}

fn city_wet_reflection(city_o: vec3<f32>, dir: vec3<f32>, t: f32,
                       pixel_angle: f32, frame_index: u32,
                       strat_pixel: vec2<u32>) -> vec3<f32> {
    let pos = city_o + t * dir;
    let cell = vec2<i32>(floor(pos.xy / u.ocean_params.x));
    let cc = city_cell(cell);
    let paved = city_ground_paved(cc, cell, pos.xy);
    if (city_ground_is_lot(paved, cc)) {
        return vec3<f32>(0.0);   // yards and scrub are dry dirt
    }
    let h = CityHit(true, t, pos, vec3<f32>(0.0, 0.0, 1.0), 0, cell);
    let fp = pixel_angle * t;
    let g = city_ground(pos.xy, cell, cc, fp);
    let rnd = city_refl_rand(strat_pixel, frame_index);
    let lobe = city_wet_lobe(h, g, dir, pixel_angle, rnd.xyz);
    if (!lobe.ok) {
        return vec3<f32>(0.0);
    }
    let w = lobe.w * city_fog_trans(city_o, dir, t, pos);
    var add = (w * lobe.blend) * lobe.env;
    // Russian roulette on what is left to trace: a lobe whose weight is
    // under CITY_REFL_RR (a steep view, a damp film, a far pixel) is traced
    // with probability w / RR and counted RR / w times over when it is.
    let wt = w * (1.0 - lobe.blend);
    let p_tr = min(wt / CITY_REFL_RR, 1.0);
    if (wt > 0.0 && rnd.w < p_tr) {
        let ro = vec3<f32>(pos.xy, pos.z + 0.02);
        let hr = city_trace(ro, lobe.r, 0.0, CITY_REFL_RANGE,
                            CITY_TRACE_PLAIN);
        var l = vec3<f32>(0.0);
        if (!hr.hit) {
            // Fogged over the range it searched, as a hit at the range
            // would have been.
            let far = ro + CITY_REFL_RANGE * lobe.r;
            l = city_fog(city_reflect_sky(u.cam_origin.xyz + t * dir, lobe.r,
                                          city_frame_rand(strat_pixel,
                                                          frame_index,
                                                          CITY_RAND_REFLECT).w),
                         ro, lobe.r, CITY_REFL_RANGE, far);
        } else {
            let fp_r = pixel_angle * (t + hr.t) + 0.5 * lobe.alpha * hr.t;
            // The texture under the lobe, filtered over its footprint on the
            // true pixel's terms.
            let fp_f = true_pixel_angle() * (t + hr.t)
                       + 0.5 * lobe.alpha * hr.t;
            l = city_fog(city_shade(hr, lobe.r, fp_r, fp_f, false), ro,
                         lobe.r, hr.t, hr.pos);
        }
        add = add + (wt / p_tr) * l;
    }
    return add;
}
