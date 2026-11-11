// --- the flat facade's windows, as the eye is shown them --------------------
//
// A lit window is ten to a hundred times the radiance of the wall around
// it, and the display transform (tone_map) is steeply compressive. When a
// pixel holds several windows, what the renderer shows at rest is the
// average of what each sample SHOWS: the accumulator averages tone-mapped
// samples. A far facade's plain mean radiance tone-maps to a cream wash far
// brighter than the resolved facade it stands for. So the pixel's footprint
// on the wall is averaged as it would be shown: exactly, window by window,
// while it holds few of them (city_windows_exact), and in expectation
// storey by storey past that (city_windows_rows), blending the two across
// the last cell of the exact sum's room. The storeys are the building's own
// (city_storey_rate), so a far tower reads in fine bands of floors, never
// in squares.
//
// What comes back is the LUMINANCE the eye is shown at rest, in the colour
// the light actually has (city_windows_resolve): the plain area mean's
// chromaticity, scaled until it shows that luminance. Bright panes clip
// toward white on the display, so a mean taken of displayed colours greys
// every window population out, while the light itself, which is what a
// camera's optics and sensor average, keeps its warmth; the city's rooms
// are warm (city_room_color). A footprint holding one element comes back
// as that element, exactly. The result is handed back as a radiance, so
// fog, clouds and the tone map downstream treat it like any other.
//
// Linear consumers get the plain area mean, which is what they average:
// TONE_MAP false (witness's linear return, the auto-exposure meter, any
// HDR output), where it does not depend on the exposure, so the meter
// never chases its own setting through the far city; and a facade seen in
// a reflection, whose radiance is weighted into the ground's before
// anything is shown.

// The window cells the exact sum walks at most, across and up. A box
// spans at most floor(2 h / pitch) + 2 cells on an axis, so the exact sum
// is whole while 2 h / pitch < MAX - 1; it hands over to the storey-by-
// storey expectation while the span is between MAX - 3 and MAX - 2, well
// inside that.
const CITY_WIN_EXACT_MAX_U: i32 = 5;
const CITY_WIN_EXACT_MAX_V: i32 = 4;
// Storeys the expectation walks one by one; past this it blends, over the
// last CITY_WIN_ROWS_BLEND of them, into one building mean.
const CITY_WIN_ROWS_MAX: i32 = 16;
const CITY_WIN_ROWS_BLEND: f32 = 4.0;
// The reveal, sill and frame inside an opening, off the glass: the shell
// shades them in the recess's shadow (ao 0.5-0.85); their mean.
const CITY_OPENING_AO: f32 = 0.6;
// The per-bay covers' fade (m of foreshortened footprint).
const CITY_COVER_LOD: vec2<f32> = vec2<f32>(1.6, 6.5);

// The display curve on exposed radiance: tone_map without its contrast
// pivot (an affine map, which commutes with an average), its highlight
// shoulder included.
fn city_display_curve(e: vec3<f32>) -> vec3<f32> {
    let w2 = u.flags.w * u.flags.w;
    let per_channel = e * (1.0 + e / w2) / (1.0 + e);
    let y = dot(e, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma_preserving = e * (1.0 + y / w2) / (1.0 + y);
    let k = TONE_MAP_SHOULDER * smoothstep(1.0, 3.0, y);
    let mapped = mix(per_channel, chroma_preserving, k);
    return pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)),
               vec3<f32>(1.0 / u.periodic.w));
}

// One area element, weighted `a`, of radiance l seen through the fog: its
// displayed value and its light, summed side by side. `shown` is kept only
// when the pixel will be shown directly (see above).
struct CityWinSum {
    shown: vec3<f32>,
    light: vec3<f32>,
}

fn city_win_zero() -> CityWinSum {
    return CityWinSum(vec3<f32>(0.0), vec3<f32>(0.0));
}

fn city_win_add(acc: CityWinSum, l: vec3<f32>, a: f32, fog: vec4<f32>,
                shown: bool) -> CityWinSum {
    if (a <= 0.0) {
        return acc;
    }
    let x = max(l * fog.x + fog.yzw, vec3<f32>(0.0));
    var d = vec3<f32>(0.0);
    if (shown) {
        d = city_display_curve(x * u.cam_right.w);
    }
    return CityWinSum(acc.shown + d * a, acc.light + x * a);
}

// The radiance the averaged footprint hands back. `mean` is the per-area
// sum. Shown directly, it is the light's own colour (its fogless mean)
// scaled by s until it shows the averaged luminance: the least s that
// does (to within CITY_RESOLVE_EPS). s is bracketed about
// a first guess (the per-channel curve inverted at that luminance, held
// to [1e-6, 1e6]), the bracket widened by fours until it holds, and
// bisected in log steps, two more for each widening. One element, alone
// in the footprint, comes back as itself.
//
// The rule needs the displayed luminance to rise with s up to the goal.
// It does for a light with every channel lit, as the room palette's
// lights are, in any fog, and for any light in clear air, where it at most
// plateaus once every lit channel is clipped (hence the least s, which
// keeps a rounding residue in an empty channel from being scaled up past
// the plateau). A near-primary in fog is the exception: the shoulder's
// chroma-preserving blend dims the fog's own channels as the primary
// brightens, so its luminance can peak and fall. The widening stops at
// the peak, stepping hi back to the brightest point it found, and s stays
// near it.
const CITY_RESOLVE_EPS: f32 = 1e-5;
fn city_windows_resolve(mean: CityWinSum, fog: vec4<f32>,
                        shown: bool) -> vec3<f32> {
    let light = max(mean.light - fog.yzw, vec3<f32>(0.0)) / max(fog.x, 1e-6);
    if (!shown) {
        return light;
    }
    let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
    let target_y = dot(mean.shown, lum);
    let goal = target_y - CITY_RESOLVE_EPS;
    let e = u.cam_right.w;
    let x0 = light * fog.x;
    let y_light = dot(x0 * e, lum);
    if (y_light <= 0.0) {
        return light;
    }
    // First guess: the exposed luminance the per-channel curve shows as
    // target_y, over the light's own.
    let m = pow(clamp(target_y, 0.0, 0.999999), u.periodic.w);
    let w2 = u.flags.w * u.flags.w;
    // @domain: (1 - m)^2 >= 0, m >= 0, and w2 >= 16 (both hosts hold the white point to [4, 40])
    let e_y = 2.0 * m / ((1.0 - m) + sqrt((1.0 - m) * (1.0 - m) + 4.0 * m / w2));
    let guess = clamp((e_y - dot(fog.yzw * e, lum)) / y_light, 1e-6, 1e6);
    // lo shows short of the goal, hi reaches it. The goal is always
    // reachable: every channel an averaged element lit, the light has.
    // Twelve widenings reach 4^12 either way, past anything the palette
    // and the hosts' gamma and white-point limits produce.
    var lo = 0.5 * guess;
    var hi = 2.0 * guess;
    var steps = 10;
    var y_hi_last = -1.0;
    for (var k: i32 = 0; k < 12; k = k + 1) {
        let lo_over = dot(city_display_curve(e * (lo * x0 + fog.yzw)), lum)
                      >= goal;
        let y_hi = dot(city_display_curve(e * (hi * x0 + fog.yzw)), lum);
        let hi_under = y_hi < goal;
        if (!lo_over && !hi_under) {
            break;
        }
        // Past a peak short of the goal (see above): back to the last hi,
        // the brighter.
        if (hi_under && y_hi <= y_hi_last) {
            hi = 0.25 * hi;
            steps = steps - 2;
            break;
        }
        y_hi_last = y_hi;
        if (lo_over) {
            lo = 0.25 * lo;
        }
        if (hi_under) {
            hi = 4.0 * hi;
        }
        steps = steps + 2;
    }
    for (var k: i32 = 0; k < steps; k = k + 1) {
        // @domain: lo and hi stay positive (guess >= 1e-6, quartered at most 12 times after halving)
        let mid = sqrt(lo * hi);
        if (dot(city_display_curve(e * (mid * x0 + fog.yzw)), lum) < goal) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // @domain: as above
    return light * sqrt(lo * hi);
}

// Half extents (u, v) of the box that holds one pixel's footprint on the
// wall: the pixel's fp x fp cross-section carried along the ray onto the
// wall plane. dl is the ray in face-local terms (z out of the wall); the
// stretch is held to the core's grazing limit (cosine 0.2).
fn city_wall_footprint(dl: vec3<f32>, fp: f32) -> vec2<f32> {
    // @domain: zero only at dl.x = 0 and dl.z = -1e-9 exactly (a unit dl into the wall)
    let e1 = normalize(vec3<f32>(dl.z, 0.0, -dl.x) + vec3<f32>(1e-9, 0.0, 0.0));
    let e2 = cross(dl, e1);
    // An offset e across the ray lands on the wall at e - dl (e.z / dl.z).
    let k = 1.0 / max(-dl.z, 0.20);
    let w1 = e1 + dl * (e1.z * k);
    let w2 = e2 + dl * (e2.z * k);
    return 0.5 * fp * vec2<f32>(abs(w1.x) + abs(w2.x), abs(w1.y) + abs(w2.y));
}

// The face's bays along u: the lattice between the corner strips, or on a
// frustum's skin everything.
fn city_bay_strip(f: CityFace) -> vec2<f32> {
    if (!f.aligned) {
        return vec2<f32>(-1e30, 1e30);
    }
    return vec2<f32>(f.margin, f.margin + f32(f.nbays) * f.pitch);
}

fn city_overlap(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    return max(min(a1, b1) - max(a0, b0), 0.0);
}

// Length of [u0, u1] covered by the interval [r0, r1] repeated every
// `pitch` from `start` over `n` periods (r0 <= r1 within one period).
fn city_periodic_cover(u0: f32, u1: f32, start: f32, pitch: f32, n: f32,
                       r0: f32, r1: f32) -> f32 {
    let lo = clamp((u0 - start) / pitch, 0.0, n);
    let hi = clamp((u1 - start) / pitch, 0.0, n);
    let a = r0 / pitch;
    let w = max(r1 - r0, 0.0) / pitch;
    let c_hi = floor(hi) * w + clamp(fract(hi) - a, 0.0, w);
    let c_lo = floor(lo) * w + clamp(fract(lo) - a, 0.0, w);
    return max(c_hi - c_lo, 0.0) * pitch;
}

// The probability a window on storey iv is lit, from the draws city_window
// makes (its unit's draw is uniform, so this is exact).
fn city_storey_lit(cc: CityCell, iv: i32) -> f32 {
    return min(city_storey_rate(cc, iv), 1.0);
}

// The share of the building's lit windows drawing room colour band k
// (city_room_color): city_window_rated's colour draw is uniform on [lo, hi] and
// then clamped to [0, 1] (a whole [0, 1] for a mixed house, 0.08 wide about
// its one draw for a monochrome one), and band k takes the clamped draws in
// [its lower edge, its upper edge). So the share is the difference of the
// clamped draw's distribution at the two edges, and the six shares sum to
// 1 wherever the edges fall, 1 and past it included.
fn city_room_band_share(k: i32, lo: f32, hi: f32, bias: f32) -> f32 {
    let a = select(city_room_band_edge(k - 1, bias), -1e9, k == 0);
    let b = select(city_room_band_edge(k, bias), 1e9, k == 5);
    return city_room_draw_below(b, lo, hi) - city_room_draw_below(a, lo, hi);
}

// P(clamp(draw, 0, 1) < x) for the draw uniform on [lo, hi].
fn city_room_draw_below(x: f32, lo: f32, hi: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
}

// A lit pane of this building, over the population of its lit windows:
// what it shows (the brightness draw, 0.25 + 5 w^7, in four strata by
// their exact means, times the palette's bands by their exact shares) and
// its light (exact: the mean brightness times the mean colour).
// `cyan` moves that share of the population to the cyan band (the shops'
// cyan interiors among all the lit glass the caller averages).
fn city_lit_pane(cc: CityCell, refl: vec3<f32>, through: f32,
                 fog: vec4<f32>, shown: bool, cyan: f32) -> CityWinSum {
    let style_gain = select(select(1.0, 0.80, cc.win_style == 1),
                            0.55, cc.win_style == 3);
    let g = cc_window_glyph_mean() * (through * CITY_ROOM_RADIANCE
                                      * style_gain);
    var lo = 0.0;
    var hi = 1.0;
    if (cc.win_mono >= 0.0) {
        lo = cc.win_mono - 0.04;
        hi = cc.win_mono + 0.04;
    }
    var colour = vec3<f32>(0.0);
    var acc = city_win_zero();
    for (var k: i32 = 0; k < 6; k = k + 1) {
        let share = city_room_band_share(k, lo, hi, cc.palette_bias)
                    * (1.0 - cyan) + select(0.0, cyan, k == 4);
        if (share <= 0.0) {
            continue;
        }
        let c = city_room_band_colour(k);
        colour = colour + c * share;
        if (shown) {
            // The brightness strata: w in [0, .7), [.7, .85), [.85, .95),
            // [.95, 1), each at its exact mean 0.25 + 5 (b^8 - a^8) /
            // (8 (b - a)).
            for (var s: i32 = 0; s < 4; s = s + 1) {
                let wa = select(select(select(0.0, 0.70, s == 1), 0.85,
                                       s == 2), 0.95, s == 3);
                let wb = select(select(select(0.70, 0.85, s == 1), 0.95,
                                       s == 2), 1.0, s == 3);
                // @domain: wa and wb lie in [0, 1]
                let b_mean = 0.25 + 5.0 * (pow(wb, 8.0) - pow(wa, 8.0))
                                        / (8.0 * (wb - wa));
                let x = max((refl + c * (g * b_mean)) * fog.x + fog.yzw,
                            vec3<f32>(0.0));
                acc.shown = acc.shown
                    + city_display_curve(x * u.cam_right.w)
                      * (share * (wb - wa));
            }
        }
    }
    acc.light = max((refl + colour * (g * CITY_WIN_BRIGHT_MEAN)) * fog.x
                    + fog.yzw, vec3<f32>(0.0));
    return acc;
}

struct CityWinGeom {
    plo: vec2<f32>,    // pane, bay-local
    phi: vec2<f32>,
    olo: vec2<f32>,    // opening, bay-local
    ohi: vec2<f32>,
    off: vec2<f32>,    // where the glass seen at a wall point lies, relative
    i_shift: i32,      // the lattice cell c's u is counted from (skins)
}

// Window by window: every window cell the footprint box touches. Returns
// the sums over the box (not yet divided by its area).
// The storeys at a warehouse's foot whose panes would stand in its door
// band: none there, in either path.
fn city_door_storeys(g: CityWinGeom) -> i32 {
    return i32(ceil((CITY_INDUSTRY_DOOR_BAND - g.phi.y) / CITY_FLOOR_H));
}

fn city_windows_exact(cc: CityCell, f: CityFace, g: CityWinGeom,
                      zp: vec3<u32>, c: vec2<f32>, hb: vec2<f32>,
                      l_wall: vec3<f32>, l_plant: vec3<f32>,
                      refl: vec3<f32>, through: f32, fog: vec4<f32>,
                      skin: bool, shown: bool) -> CityWinSum {
    let warehouse = cc.land_use == CITY_USE_INDUSTRIAL;
    // A frustum's skin has an unbounded, world-anchored lattice: every
    // cell is a bay (city_bay).
    var i0 = i32(floor((c.x - hb.x - f.margin) / f.pitch));
    var i1 = i32(floor((c.x + hb.x - f.margin) / f.pitch));
    if (f.aligned) {
        i0 = max(i0, 0);
        i1 = min(i1, f.nbays - 1);
    }
    let j0 = i32(floor((c.y - hb.y) / CITY_FLOOR_H));
    let j1 = i32(floor((c.y + hb.y) / CITY_FLOOR_H));
    let loft_lf = city_loft_lit_frac(cc);
    let zones = city_zones_unpack(zp);
    var acc = city_win_zero();
    var a_open = 0.0;
    var a_ring = 0.0;
    var a_plant = 0.0;
    for (var jj: i32 = 0; jj < CITY_WIN_EXACT_MAX_V; jj = jj + 1) {
        let j = j0 + jj;
        if (j > j1) {
            break;
        }
        // A warehouse's ground band is its doors, which the industry
        // component draws: no windows, no glass.
        if (warehouse && j < city_door_storeys(g)) {
            continue;
        }
        let vfl = f32(j) * CITY_FLOOR_H;
        let rate = city_storey_rate(cc, j);
        // The storey's zoning, once (city_zone_at, by storey and then by
        // bay, in its order).
        let row_shop = zones.store && j == 0;
        let row_plant = city_storey_plant(zones, j);
        let row_loft = j >= zones.loft.x && j < zones.loft.y;
        for (var ii: i32 = 0; ii < CITY_WIN_EXACT_MAX_U; ii = ii + 1) {
            let i = i0 + ii;
            if (i > i1) {
                break;
            }
            let base = vec2<f32>(f.margin + f32(i) * f.pitch, vfl);
            // The bay's zone, as the shell draws it: no opening on a solid
            // bay (wall) or a plant one (louvres), a loft's or a shop's
            // own glass and its own rate.
            var zone = CITY_ZONE_WIN;
            if (zones.pent) {
                zone = CITY_ZONE_PENT;
            } else if (row_shop) {
                zone = CITY_ZONE_STORE;
            } else if (zones.blank) {
                zone = CITY_ZONE_BLANK;
            } else if (i >= zones.core.x && i < zones.core.y && j >= 1) {
                zone = CITY_ZONE_CORE;
            } else if (row_plant) {
                zone = CITY_ZONE_PLANT;
            } else if (row_loft && (i < zones.loft.z
                                    || i >= f.nbays - zones.loft.z)) {
                zone = CITY_ZONE_LOFT;
            }
            if (!city_zone_open(zone)) {
                if (zone == CITY_ZONE_PLANT) {
                    a_plant = a_plant
                        + city_overlap(c.x - hb.x, c.x + hb.x, base.x,
                                       base.x + f.pitch)
                        * city_overlap(c.y - hb.y, c.y + hb.y,
                                       max(vfl, f.z0),
                                       min(vfl + CITY_FLOOR_H, f.z1));
                }
                continue;
            }
            let gz = city_zone_geom(g, f, zone);
            var rate_w = rate;
            if (zone == CITY_ZONE_LOFT) {
                rate_w = city_storey_rate_at(cc, j, loft_lf);
            } else if (zone == CITY_ZONE_STORE) {
                rate_w = city_storey_rate_at(cc, j, CITY_STORE_LIT_FRAC);
            }
            // The opening, on this face, and the glass seen through it.
            let olo = max(base + gz.olo, vec2<f32>(-1e9, f.z0));
            let ohi = min(base + gz.ohi, vec2<f32>(1e9, f.z1));
            let rlo = max(base + gz.plo - g.off, olo);
            let rhi = min(base + gz.phi - g.off, ohi);
            let ao = city_overlap(c.x - hb.x, c.x + hb.x, olo.x, ohi.x)
                     * city_overlap(c.y - hb.y, c.y + hb.y, olo.y, ohi.y);
            if (ao <= 0.0) {
                continue;
            }
            let ag = city_overlap(c.x - hb.x, c.x + hb.x, rlo.x, rhi.x)
                     * city_overlap(c.y - hb.y, c.y + hb.y, rlo.y, rhi.y);
            var win = city_window_rated(cc, f, i + g.i_shift, j, rate_w);
            if (zone == CITY_ZONE_STORE) {
                win = city_shop_window(cc, win);
            }
            if (zone == CITY_ZONE_PENT) {
                win = city_pent_window(cc, f, i + g.i_shift, j);
            }
            var glass = refl;
            if (win.lit) {
                // Where the pixel's centre sees this pane, for a glyph a
                // pixel can still resolve; far larger footprints get the
                // glyph's own mean from it.
                let uv = clamp((c - base + g.off - gz.plo)
                               / max(gz.phi - gz.plo, vec2<f32>(1e-4)),
                               vec2<f32>(0.0), vec2<f32>(1.0));
                glass = refl + win.e * through
                        * cc_window_glyph(cc, win.wh, uv, uv,
                                          2.0 * max(hb.x, hb.y));
            }
            acc = city_win_add(acc, glass, ag, fog, shown);
            a_ring = a_ring + (ao - ag);
            a_open = a_open + ao;
        }
    }
    acc = city_win_add(acc, CITY_OPENING_AO * l_wall, a_ring, fog, shown);
    acc = city_win_add(acc, l_plant + select(vec3<f32>(0.0), refl, skin),
                       a_plant, fog, shown);
    // The rest is wall: skin (a curtain wall's glass spandrels reflect) on
    // the bays, plain wall on the corner strips and off the face. Openings
    // and plant lie inside the bays.
    let strip = city_bay_strip(f);
    let a_bays = city_overlap(c.x - hb.x, c.x + hb.x, strip.x, strip.y)
                 * city_overlap(c.y - hb.y, c.y + hb.y, f.z0, f.z1);
    acc = city_win_add(acc, l_wall + select(vec3<f32>(0.0), refl, skin),
                       max(a_bays - a_open - a_plant, 0.0), fog, shown);
    return city_win_add(acc, l_wall,
                        max(4.0 * hb.x * hb.y - a_bays, 0.0), fog, shown);
}

// A zone's window geometry: a loft's or a shop's own panes in place of the
// building's (g for every other zone). The recess offset is the wall's.
fn city_zone_geom(g: CityWinGeom, f: CityFace, zone: i32) -> CityWinGeom {
    if (zone != CITY_ZONE_LOFT && zone != CITY_ZONE_STORE
        && zone != CITY_ZONE_PENT) {
        return g;
    }
    var z = g;
    let shop = zone == CITY_ZONE_STORE;
    let pent = zone == CITY_ZONE_PENT;
    let cell = vec2<f32>(f.pitch, CITY_FLOOR_H);
    z.plo = select(select(CITY_LOFT_PANE_LO, CITY_STORE_PANE_LO, shop),
                   CITY_PENT_PANE_LO, pent) * cell;
    z.phi = select(select(CITY_LOFT_PANE_HI, CITY_STORE_PANE_HI, shop),
                   CITY_PENT_PANE_HI, pent) * cell;
    z.olo = z.plo - vec2<f32>(CITY_SHELL_FRAME);
    z.ohi = z.phi + vec2<f32>(CITY_SHELL_FRAME);
    return z;
}

// Storey by storey, in expectation: each storey's windows lit at the rate
// its own draw gives, their panes averaged over the draws. Past
// CITY_WIN_ROWS_MAX storeys, the building's mean rate instead, blended in
// over the last CITY_WIN_ROWS_BLEND. Returns the sums over the box.
// The glass room in expectation over a box [c - hb, c + hb] (face-local u
// and world z), for a footprint too wide for the exact sum: every bay's
// opening, its pane inside (behind the recess g.off) and the frame between,
// each by exact periodic cover on both axes; the panes lit at the room's
// rate, the level uniform on [0.45, 1.55] by three-point Gauss-Legendre;
// the rest wall. Returns the sums over the box, as city_windows_rows does.
fn city_pent_rows(cc: CityCell, f: CityFace, g: CityWinGeom, c: vec2<f32>,
                  hb: vec2<f32>, l_wall: vec3<f32>, refl: vec3<f32>,
                  through: f32, fog: vec4<f32>, skin: bool,
                  shown: bool) -> CityWinSum {
    let u0 = c.x - hb.x;
    let u1 = c.x + hb.x;
    let n = f32(max(f.nbays, 0));
    let rlo = max(g.plo - g.off, g.olo);
    let rhi = min(g.phi - g.off, g.ohi);
    let len_g = city_periodic_cover(u0, u1, f.margin, f.pitch, n, rlo.x,
                                    rhi.x);
    let len_o = city_periodic_cover(u0, u1, f.margin, f.pitch, n, g.olo.x,
                                    g.ohi.x);
    let ns = (f.z1 - f.z0) / CITY_FLOOR_H;
    let v0 = c.y - hb.y;
    let v1 = c.y + hb.y;
    let ht_g = city_periodic_cover(v0, v1, f.z0, CITY_FLOOR_H, ns, rlo.y,
                                   rhi.y);
    let ht_o = city_periodic_cover(v0, v1, f.z0, CITY_FLOOR_H, ns, g.olo.y,
                                   g.ohi.y);
    let a_glass = len_g * ht_g;
    let a_open = len_o * ht_o;
    let e = city_pent_colour(cc) * (CITY_PENT_RAD * through)
            * cc_window_glyph_mean();
    let gl = 0.55 * 0.7745966692;
    let lit = a_glass * CITY_PENT_LIT;
    var acc = city_win_add(city_win_zero(), refl + e, lit * (4.0 / 9.0), fog,
                           shown);
    acc = city_win_add(acc, refl + e * (1.0 - gl), lit * (5.0 / 18.0), fog,
                       shown);
    acc = city_win_add(acc, refl + e * (1.0 + gl), lit * (5.0 / 18.0), fog,
                       shown);
    acc = city_win_add(acc, refl, a_glass - lit, fog, shown);
    acc = city_win_add(acc, CITY_OPENING_AO * l_wall,
                       max(a_open - a_glass, 0.0), fog, shown);
    let strip = city_bay_strip(f);
    let a_bays = city_overlap(u0, u1, strip.x, strip.y)
                 * city_overlap(v0, v1, f.z0, f.z1);
    acc = city_win_add(acc, l_wall + select(vec3<f32>(0.0), refl, skin),
                       max(a_bays - a_open, 0.0), fog, shown);
    return city_win_add(acc, l_wall, max(4.0 * hb.x * hb.y - a_bays, 0.0),
                        fog, shown);
}

fn city_windows_rows(cc: CityCell, f: CityFace, g: CityWinGeom,
                     zp: vec3<u32>, c: vec2<f32>, hb: vec2<f32>,
                     l_wall: vec3<f32>, l_plant: vec3<f32>,
                     refl: vec3<f32>, through: f32, fog: vec4<f32>,
                     skin: bool, shown: bool) -> CityWinSum {
    let warehouse = cc.land_use == CITY_USE_INDUSTRIAL;
    let u0 = c.x - hb.x;
    let u1 = c.x + hb.x;
    // The lattice's first bay and its count; a skin's runs past the box.
    var start = f.margin;
    var n = f32(max(f.nbays, 0));
    if (!f.aligned) {
        start = f.margin + (floor((u0 - f.margin) / f.pitch) - 1.0) * f.pitch;
        n = (u1 - start) / f.pitch + 2.0;
    }
    let rlo = max(g.plo - g.off, g.olo);
    let rhi = min(g.phi - g.off, g.ohi);
    let len_g = city_periodic_cover(u0, u1, start, f.pitch, n, rlo.x, rhi.x);
    let len_o = city_periodic_cover(u0, u1, start, f.pitch, n,
                                    g.olo.x, g.ohi.x);
    // A warehouse's door band holds no windows (the storeys the exact sum
    // skips).
    var v0 = max(c.y - hb.y, f.z0);
    let v1 = min(c.y + hb.y, f.z1);
    if (warehouse) {
        v0 = max(v0, f32(city_door_storeys(g)) * CITY_FLOOR_H);
    }
    let j0 = i32(floor(v0 / CITY_FLOOR_H));
    let j1 = i32(floor(v1 / CITY_FLOOR_H));
    // The storeys the box touches are at most (v1 - v0) / FLOOR_H + 2, so
    // the walk is whole wherever the blend leaves it any weight, and the
    // blend moves continuously with the box.
    let w_mean = smoothstep(f32(CITY_WIN_ROWS_MAX) - CITY_WIN_ROWS_BLEND,
                            f32(CITY_WIN_ROWS_MAX),
                            (v1 - v0) / CITY_FLOOR_H + 2.0);
    let zones = city_zones_unpack(zp);
    // The zones along the storey (city_zones): which bays a core takes and
    // the corner bays the lofts take, as lengths of [u0, u1] like len_g and
    // len_o. Only these ride the walk; the shop row (storey 0 alone) and
    // the lofts' own glass are summed apart, after it, so the loop carries
    // five values more than the lattice's, not twenty (every one is
    // registers for the whole surface pass).
    let c0 = f.margin + f32(zones.core.x) * f.pitch;
    let nc = f32(zones.core.y - zones.core.x);
    let core = vec3<f32>(
        city_periodic_cover(u0, u1, c0, f.pitch, nc, rlo.x, rhi.x),
        city_periodic_cover(u0, u1, c0, f.pitch, nc, g.olo.x, g.ohi.x),
        city_overlap(u0, u1, c0, c0 + nc * f.pitch));
    let nl = f32(zones.loft.z);
    let l1 = f.margin + (f32(f.nbays) - nl) * f.pitch;
    let loft_lat = vec2<f32>(
        city_periodic_cover(u0, u1, f.margin, f.pitch, nl, rlo.x, rhi.x)
        + city_periodic_cover(u0, u1, l1, f.pitch, nl, rlo.x, rhi.x),
        city_periodic_cover(u0, u1, f.margin, f.pitch, nl, g.olo.x, g.ohi.x)
        + city_periodic_cover(u0, u1, l1, f.pitch, nl, g.olo.x, g.ohi.x));
    let shop_row = zones.store && j0 == 0;
    // The lit glass, the glass, the openings and the plant, walked storey
    // by storey and as the building's mean, blended.
    var walk = vec4<f32>(0.0);
    var walk_shop = 0.0;   // the shop row's lit glass (city_shop_window)
    if (w_mean < 1.0) {
        for (var jj: i32 = 0; jj < CITY_WIN_ROWS_MAX; jj = jj + 1) {
            let j = j0 + jj;
            if (j > j1) {
                break;
            }
            if (zones.blank || (shop_row && j == 0)) {
                continue;
            }
            let vfl = f32(j) * CITY_FLOOR_H;
            // A core's bays are solid from storey 1 up.
            let cz = select(vec3<f32>(0.0), core, j >= 1);
            if (city_storey_plant(zones, j)) {
                walk.w = walk.w + (city_overlap(u0, u1, f.margin, f.margin
                                                + f32(f.nbays) * f.pitch)
                                   - cz.z)
                         * city_overlap(v0, v1, vfl, vfl + CITY_FLOOR_H);
                continue;
            }
            let lz = select(vec2<f32>(0.0), loft_lat,
                            j >= zones.loft.x && j < zones.loft.y);
            let ag = (len_g - cz.x - lz.x)
                     * city_overlap(v0, v1, vfl + rlo.y, vfl + rhi.y);
            walk = walk + vec4<f32>(city_storey_lit(cc, j) * ag, ag,
                (len_o - cz.y - lz.y)
                * city_overlap(v0, v1, vfl + g.olo.y, vfl + g.ohi.y), 0.0);
        }
        // The shop row: every bay of storey 0, shop glass, every pane lit.
        if (shop_row) {
            let gs = city_zone_geom(g, f, CITY_ZONE_STORE);
            let rs = vec4<f32>(max(gs.plo - g.off, gs.olo),
                               min(gs.phi - g.off, gs.ohi));
            let nb = f32(f.nbays);
            let ag = city_periodic_cover(u0, u1, f.margin, f.pitch, nb,
                                         rs.x, rs.z)
                     * city_overlap(v0, v1, rs.y, rs.w);
            walk_shop = ag;
            walk = walk + vec4<f32>(ag, ag,
                city_periodic_cover(u0, u1, f.margin, f.pitch, nb,
                                    gs.olo.x, gs.ohi.x)
                * city_overlap(v0, v1, gs.olo.y, gs.ohi.y), 0.0);
        }
        // The lofts' own glass, over the loft storeys the box holds.
        if (zones.loft.z > 0 && !zones.blank) {
            let gl = city_zone_geom(g, f, CITY_ZONE_LOFT);
            let rl = vec4<f32>(max(gl.plo - g.off, gl.olo),
                               min(gl.phi - g.off, gl.ohi));
            let lg = city_periodic_cover(u0, u1, f.margin, f.pitch, nl,
                                         rl.x, rl.z)
                     + city_periodic_cover(u0, u1, l1, f.pitch, nl, rl.x,
                                           rl.z);
            let lo = city_periodic_cover(u0, u1, f.margin, f.pitch, nl,
                                         gl.olo.x, gl.ohi.x)
                     + city_periodic_cover(u0, u1, l1, f.pitch, nl,
                                           gl.olo.x, gl.ohi.x);
            let loft_lf = city_loft_lit_frac(cc);
            for (var j: i32 = max(zones.loft.x, j0);
                 j < min(zones.loft.y, j1 + 1); j = j + 1) {
                let vfl = f32(j) * CITY_FLOOR_H;
                let al = lg * city_overlap(v0, v1, vfl + rl.y, vfl + rl.w);
                walk = walk + vec4<f32>(
                    min(city_storey_rate_at(cc, j, loft_lf), 1.0) * al, al,
                    lo * city_overlap(v0, v1, vfl + gl.olo.y,
                                      vfl + gl.ohi.y), 0.0);
            }
        }
    }
    var mean = vec4<f32>(0.0);
    var mean_shop = 0.0;
    if (w_mean > 0.0) {
        let nv = floor(f.z1 / CITY_FLOOR_H) + 1.0;
        let ag = len_g * city_periodic_cover(v0, v1, 0.0, CITY_FLOOR_H, nv,
                                             rlo.y, rhi.y);
        let ao = len_o * city_periodic_cover(v0, v1, 0.0, CITY_FLOOR_H, nv,
                                             g.olo.y, g.ohi.y);
        // Past the walk, the zones in expectation: their shares of the
        // footprint (city_zone_cover), each in place of the lattice there,
        // the lofts and the shops at their own glass's share of a bay cell.
        let zc = city_zone_cover(zones, f, c, hb);
        let area = 4.0 * hb.x * hb.y;
        let zoned = zc.x + zc.y + zc.z + zc.w;
        var keep = 1.0;
        if (zoned > 0.0) {
            keep = max(1.0 - zoned / city_zone_bays(f, c, hb), 0.0);
        }
        let cell = f.pitch * CITY_FLOOR_H;
        let gl = city_zone_geom(g, f, CITY_ZONE_LOFT);
        let gs = city_zone_geom(g, f, CITY_ZONE_STORE);
        let rlo_l = max(gl.plo - g.off, gl.olo);
        let rhi_l = min(gl.phi - g.off, gl.ohi);
        let rlo_s = max(gs.plo - g.off, gs.olo);
        let rhi_s = min(gs.phi - g.off, gs.ohi);
        let fl = vec2<f32>(max(rhi_l.x - rlo_l.x, 0.0)
                           * max(rhi_l.y - rlo_l.y, 0.0),
                           (gl.ohi.x - gl.olo.x) * (gl.ohi.y - gl.olo.y))
                 * (zc.z * area / cell);
        let fs = vec2<f32>(max(rhi_s.x - rlo_s.x, 0.0)
                           * max(rhi_s.y - rlo_s.y, 0.0),
                           (gs.ohi.x - gs.olo.x) * (gs.ohi.y - gs.olo.y))
                 * (zc.w * area / cell);
        let loft_lit = min(city_loft_lit_frac(cc) * city_style_rate(cc)
                           * CITY_LIT_SHARE, 1.0);
        mean = vec4<f32>(city_storey_lit_mean(cc) * ag * keep
                         + loft_lit * fl.x + fs.x,
                         ag * keep + fl.x + fs.x,
                         ao * keep + fl.y + fs.y,
                         zc.y * area);
        mean_shop = fs.x;
    }
    let a = mix(walk, mean, w_mean);   // (lit glass, glass, openings, plant)
    let cyan = CITY_SHOP_CYAN_FRAC * mix(walk_shop, mean_shop, w_mean)
               / max(a.x, 1e-6);
    let lit = city_lit_pane(cc, refl, through, fog, shown, cyan);
    var acc = city_win_add(city_win_zero(), refl, a.y - a.x, fog, shown);
    acc = CityWinSum(acc.shown + lit.shown * a.x,
                     acc.light + lit.light * a.x);
    acc = city_win_add(acc, CITY_OPENING_AO * l_wall, max(a.z - a.y, 0.0),
                       fog, shown);
    acc = city_win_add(acc, l_plant + select(vec3<f32>(0.0), refl, skin),
                       a.w, fog, shown);
    let strip = city_bay_strip(f);
    let a_bays = city_overlap(u0, u1, strip.x, strip.y)
                 * city_overlap(c.y - hb.y, c.y + hb.y, f.z0, f.z1);
    acc = city_win_add(acc, l_wall + select(vec3<f32>(0.0), refl, skin),
                       max(a_bays - a.z - a.w, 0.0), fog, shown);
    return city_win_add(acc, l_wall,
                        max(4.0 * hb.x * hb.y - a_bays, 0.0), fog, shown);
}

// The windows and wall under one pixel's footprint `fp_filter`, as the
// radiance that shows as their mean at rest, in the colour of their light
// (see above); `shown` false for a facade seen in a reflection, which gets
// the plain area mean. `through` is what of a lit pane's light passes the
// glass, (1 - F). Under a component's panel, a fire escape's lattice or a
// balcony (`cover`, 0..1) there are no windows to see: the covered share
// is plain wall, as the shell draws it.
fn city_windows_seen(cc: CityCell, f: CityFace, st: CityShellStyle,
                     zones: CityZones, c_in: vec2<f32>, dl: vec3<f32>,
                     fp_filter: f32, l_wall: vec3<f32>, l_plant: vec3<f32>,
                     refl: vec3<f32>, through: f32, cover: f32,
                     fog: vec4<f32>, direct: bool) -> vec3<f32> {
    let shown = direct && TONE_MAP;
    // The glass is recessed behind its opening, as the shell builds it: the
    // ray that crosses the opening at a wall point meets the glass `off`
    // further along, and sees it only if that point is still inside the
    // opening (else the reveal or the sill) and inside the pane (else the
    // frame). A frustum's skin has no shell and no recess, and neither has
    // a warehouse, whose walls are the industry component's.
    var off = vec2<f32>(0.0);
    if (f.aligned && cc.land_use == CITY_USE_URBAN) {
        off = city_recess_off(st, dl);
    }
    // A skin's u is world-anchored, tens of kilometres at float precision:
    // count it from the lattice cell under it, so a small footprint keeps
    // its width.
    var c = c_in;
    var i_shift = 0;
    if (!f.aligned) {
        let cell = floor((c.x - f.margin) / f.pitch);
        i_shift = i32(cell);
        c.x = c.x - cell * f.pitch;
    }
    let skin = cc.win_style == 3;
    let hb = max(city_wall_footprint(dl, fp_filter), vec2<f32>(1e-3));
    // The facade's zones are the windows' own business: the exact sum asks
    // each bay its zone, the storey walk each storey, and past the walk
    // the zones are taken in expectation (city_windows_rows), so a loft's
    // or a shop's glass, a core, a solid side and a plant floor all come
    // out of the one mean.
    var mean = city_windows_mean(cc, f, city_zones_pack(zones), off, i_shift,
                                 c, hb, l_wall,
                                 l_plant, refl, through, fog, skin, shown);
    if (cover > 0.0) {
        let wall = city_win_add(city_win_zero(),
                                l_wall + select(vec3<f32>(0.0), refl, skin),
                                1.0, fog, shown);
        mean = CityWinSum(mix(mean.shown, wall.shown, cover),
                          mix(mean.light, wall.light, cover));
    }
    return city_windows_resolve(mean, fog, shown);
}

// The lattice's windows and wall under the footprint [c - hb, c + hb], as
// the mean of what each shows (city_windows_seen).
fn city_windows_mean(cc: CityCell, f: CityFace, zp: vec3<u32>,
                     off: vec2<f32>, i_shift: i32, c: vec2<f32>,
                     hb: vec2<f32>, l_wall: vec3<f32>, l_plant: vec3<f32>,
                     refl: vec3<f32>, through: f32, fog: vec4<f32>,
                     skin: bool, shown: bool) -> CityWinSum {
    var g: CityWinGeom;
    g.plo = city_pane_lo(cc, f);
    g.phi = city_pane_hi(cc, f);
    g.olo = g.plo - vec2<f32>(CITY_SHELL_FRAME);
    g.ohi = g.phi + vec2<f32>(CITY_SHELL_FRAME);
    g.off = off;
    g.i_shift = i_shift;
    // Cells the footprint spans, continuously, against the room the exact
    // sum has: it hands over while it is still whole.
    let span = vec2<f32>(2.0 * hb.x / f.pitch, 2.0 * hb.y / CITY_FLOOR_H);
    let room = vec2<f32>(f32(CITY_WIN_EXACT_MAX_U - 2),
                         f32(CITY_WIN_EXACT_MAX_V - 2));
    let w = max(smoothstep(room.x - 1.0, room.x - 0.05, span.x),
                smoothstep(room.y - 1.0, room.y - 0.05, span.y));
    var sum = city_win_zero();
    if (w < 1.0) {
        sum = city_windows_exact(cc, f, g, zp, c, hb, l_wall, l_plant,
                                 refl, through, fog, skin, shown);
    }
    if (w > 0.0) {
        var rows: CityWinSum;
        if (city_zones_unpack(zp).pent) {
            rows = city_pent_rows(cc, f, city_zone_geom(g, f, CITY_ZONE_PENT),
                                  c, hb, l_wall, refl, through, fog, skin,
                                  shown);
        } else {
            rows = city_windows_rows(cc, f, g, zp, c, hb, l_wall, l_plant,
                                     refl, through, fog, skin, shown);
        }
        sum = CityWinSum(mix(sum.shown, rows.shown, w),
                         mix(sum.light, rows.light, w));
    }
    let area = 4.0 * hb.x * hb.y;
    return CityWinSum(sum.shown / area, sum.light / area);
}

// The flat facade: what a wall is past the shell's reach, and what the
// shell blends into at the hand-off.
fn city_facade_flat(cc: CityCell, f: CityFace, h: CityHit, dir: vec3<f32>,
                    fp: f32, fp_filter: f32, direct: bool,
                    fill: vec3<f32>) -> vec3<f32> {
    let moon = u.sun_dir.xyz;
    let l = city_face_local(f, h.pos);
    let uc = l.x;
    let vc = h.pos.z;
    let b_lat = city_bay(f, uc, vc);
    let st = city_shell_style(cc);
    let dl = city_face_dir(f, dir);
    let cosv = abs(dot(dir, h.normal));
    let fp_eff = fp / clamp(cosv, 0.20, 1.0);
    let warehouse = cc.land_use == CITY_USE_INDUSTRIAL;
    let zones = city_zones(cc, f);
    let b = city_bay_zone(zones, f, b_lat);

    // Under a component's panel there are no windows to see; behind a fire
    // escape, only what its lattice leaves. These stay point-sampled at the
    // pixel, each with its own footprint fade.
    let esc_cover = city_escape_cover(city_escape(cc, f, st), uc, vc, fp_eff);
    let bal_cover = city_balcony_cover(cc, f, st, b, fp_eff);
    // A unit set in the window covers its lower sash, dissolving into its
    // share of the pane.
    var ac_cover = 0.0;
    let ac = city_shell_ac(cc, f, st, b);
    if (ac.on && ac.in_window) {
        let inside = b.bu > ac.lo.x && b.bu < ac.hi.x
                  && b.bv > ac.lo.y && b.bv < ac.hi.y;
        let span = city_pane_hi(cc, f) - city_pane_lo(cc, f);
        let share = (ac.hi.x - ac.lo.x) * (ac.hi.y - ac.lo.y)
                    / max(span.x * span.y, 1e-3);
        ac_cover = mix(select(0.0, 1.0, inside), share,
                       smoothstep(0.25, 0.9, fp_eff));
    }
    // The per-bay covers (balconies, in-window units) are a bay pattern;
    // once bays are a few pixels they fade out rather than beat.
    let bay_fade = 1.0 - smoothstep(CITY_COVER_LOD.x, CITY_COVER_LOD.y,
                                    fp_eff);
    let bal_c = bal_cover * bay_fade;
    let ac_c = ac_cover * bay_fade;
    let cover = 1.0 - (1.0 - cc_facade_cover(cc, h, uc, vc, fp))
                      * (1.0 - esc_cover) * (1.0 - bal_c) * (1.0 - ac_c);

    // The wall itself, lit where the pixel looks. A warehouse's walls are
    // the industry component's whole look, reflectance included.
    var l_wall = vec3<f32>(0.0);
    var l_plant = vec3<f32>(0.0);
    if (!warehouse) {
        let alb = mix(mix(mix(st.albedo, CITY_ESCAPE_ALBEDO, esc_cover),
                          st.trim, bal_c), vec3<f32>(0.34, 0.33, 0.30), ac_c);
        let light = CITY_SHELL_FILL_K
                    * (fill + CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0))
                    + city_shell_light(h.pos, h.normal);
        l_wall = alb * light;
        l_plant = CITY_LOUVRE_ALBEDO * light;
    }
    // Glass reflects the night; the room's light leaves through it at
    // (1 - F), as the shell's glass does.
    let fres = city_glass_fresnel(cosv);
    let refl = fres * city_glass_env(reflect(dir, h.normal), h.pos,
                                     city_glow_sample(h.pos.xy, 3.0));

    // The windows, averaged over the pixel's footprint on the wall
    // (city_windows_seen): what the eye is shown, not what the wall emits.
    let fog = city_fog_ab(h.pos - h.t * dir, dir, h.t, h.pos);
    // Filtered over fp_filter: `fp` carries the view-step LOD floor, which
    // decides the work, not the sharpness.
    var e = city_windows_seen(cc, f, st, zones, vec2<f32>(uc, vc), dl,
                              fp_filter, l_wall, l_plant, refl, 1.0 - fres,
                              cover, fog, direct);

    // Whatever the facade-layer components add: signage, screens, a
    // warehouse's cladding and doors.
    return e + cc_facade_detail(cc, h, dir, uc, vc, fp);
}
