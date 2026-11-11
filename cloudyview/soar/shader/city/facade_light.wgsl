// --- shell light ---------------------------------------------------------------

// The shell's lamp light: the street's sodium lattice (city_street_light,
// radiance per unit albedo, district-scaled like the pools), plus the lit
// street's own bounce, which is what reaches a sill's underside three floors
// above the last lamp — the heads light only the faces that look toward
// them.
fn city_shell_light(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let bounce = 0.030 * max(0.5 - 0.5 * n.z, 0.0)
                 / (1.0 + (p.z / 18.0) * (p.z / 18.0));
    return city_street_light(p, n)
           + CITY_LAMP_COLOR
             * ((city_lamp_scale(p.xy) / CITY_LAMP_RADIANCE) * bounce);
}

// Irradiance from a lit rectangle (corner c, edges e1 and e2) at p with
// normal n, as a fraction of pi times its radiance: Lambert's polygon
// formula, closed form. Zero for a rectangle behind the surface.
fn city_rect_edge(n: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let cr = cross(a, b);
    let lc = length(cr);
    return select(0.0, acos(clamp(dot(a, b), -1.0, 1.0)) * dot(n, cr) / lc,
                  lc > 1e-6);
}

fn city_rect_ff(p: vec3<f32>, n: vec3<f32>, c: vec3<f32>, e1: vec3<f32>,
                e2: vec3<f32>) -> f32 {
    // Four corners as scalars, not a function-local array: an array in
    // city-only code has moved day-scene pixels before (the driver compiles
    // the kernel differently around it).
    // Guarded: a point ON a corner of the rectangle has no direction to it.
    let v0 = (c - p) / max(length(c - p), 1e-5);
    let v1 = (c + e1 - p) / max(length(c + e1 - p), 1e-5);
    let v2 = (c + e1 + e2 - p) / max(length(c + e1 + e2 - p), 1e-5);
    let v3 = (c + e2 - p) / max(length(c + e2 - p), 1e-5);
    let sum = city_rect_edge(n, v0, v1) + city_rect_edge(n, v1, v2)
            + city_rect_edge(n, v2, v3) + city_rect_edge(n, v3, v0);
    return max(abs(sum) * 0.15915494 * select(0.0, 1.0,
               dot(n, c + 0.5 * (e1 + e2) - p) > 0.0), 0.0);
}

// Glass reflects the night: the smooth sky and the glow dome above, the lit
// street below. Stars and the moon disc are left out — a sharp source in
// every pane is shimmer, not a reflection.
fn city_glass_env(r: vec3<f32>, p: vec3<f32>, glow: f32) -> vec3<f32> {
    if (r.z >= 0.0) {
        var col = mix(CITY_NIGHT_HORIZON, CITY_NIGHT_ZENITH,
                      pow(clamp(r.z, 0.0, 1.0), 0.55));
        let dome = city_glow_sample(p.xy + r.xy * CITY_SKYGLOW_DOME_DIST, 7.0);
        col = col + CITY_SKYGLOW_DOME * (CITY_SKYGLOW_DOME_AMP * dome)
                    * exp(-r.z / CITY_SKYGLOW_DOME_SCALE);
        return col;
    }
    return CITY_LAMP_COLOR * (0.05 * glow) + CITY_FOG_BASE;
}

fn city_glass_fresnel(cosv: f32) -> f32 {
    let m = 1.0 - clamp(cosv, 0.0, 1.0);
    let m2 = m * m;
    return 0.04 + 0.96 * m2 * m2 * m;
}

// --- the room behind the glass -------------------------------------------------
//
// Interior mapping: the bay cell's own storey, CITY_SHELL_ROOM_D deep, is a
// room, and the ray that passed the glass finds its back wall, side wall,
// floor or ceiling in closed form. What sells it is the parallax as the
// camera moves — the back wall slides against the frame, a side wall opens
// up — so the room is plain: light-coloured walls, a ceiling fixture, a
// darker floor, one piece of furniture, and windowlife's occupant standing
// on a plane between the glass and the back wall. Everything is scaled by
// the pane's own radiance and normalized so the room, averaged over its pane
// and over the views a flight takes of it, is exactly what the flat facade
// paints there: 1 / <shade>, measured per window style by a numpy port of
// city_room (4e5 draws each — pane points uniform, cameras 5-80 m out and
// up to 30 m aside, furniture included; the means run 0.199-0.234).
fn city_room_norm(style: i32) -> f32 {
    if (style == 1) { return 4.415; }   // ribbon
    if (style == 2) { return 4.695; }   // strips
    if (style == 3) { return 5.025; }   // curtain wall
    if (style == 4) { return 4.277; }   // punched
    return 4.613;                       // grid
}

fn city_room(cc: CityCell, f: CityFace, b: CityBay, win: CityWin,
             g: vec3<f32>, dl: vec3<f32>, fp_eff: f32) -> vec3<f32> {
    let plo = city_bay_pane_lo(cc, f, b);
    let phi = city_bay_pane_hi(cc, f, b);
    let vfl = f32(b.iv) * CITY_FLOOR_H;
    let pane_uv = (vec2<f32>(g.x - b.ca, g.y - vfl) - plo)
                  / max(phi - plo, vec2<f32>(1e-4));
    let rh = city_rand4(vec2<u32>(u32(win.wh.w * 16777216.0),
                                  u32(win.wh.y * 16777216.0))
                        ^ vec2<u32>(0x243f6a88u, 0x85a308d3u));
    // The room box, face-local.
    let rmin = vec3<f32>(b.ca, vfl + 0.10, g.z - CITY_SHELL_ROOM_D);
    // The ceiling hangs 0.30 m under the slab, but never below the pane:
    // a curtain wall's glass runs higher than a punched window's.
    let rmax = vec3<f32>(b.ca + f.pitch,
                         vfl + max(CITY_FLOOR_H - 0.30, phi.y + 0.02), g.z);
    let il = vec3<f32>(
        select(1e30, 1.0 / dl.x, abs(dl.x) > 1e-9),
        select(1e30, 1.0 / dl.y, abs(dl.y) > 1e-9),
        select(1e30, 1.0 / dl.z, abs(dl.z) > 1e-9));
    let tx = city_exit_t(g, dl, rmin, rmax);
    var tr = min(min(tx.x, tx.y), tx.z);
    var surf = 0;   // 0 back wall, 1 side wall, 2 floor, 3 ceiling, 4 furniture
    if (tx.x <= tx.y && tx.x <= tx.z) {
        surf = 1;
    } else if (tx.y <= tx.z) {
        surf = select(3, 2, dl.y < 0.0);
    }
    // One piece of furniture against the back wall: a desk, a sofa or a
    // tall cabinet, by the room's draw.
    let fw = 0.9 + 0.8 * rh.x;
    let fu0 = b.ca + 0.2 + (f.pitch - fw - 0.4) * rh.y;
    let fh = select(select(0.75, 0.85, rh.z < 0.66), 1.9, rh.z < 0.25);
    let fd = select(0.75, 0.45, rh.z < 0.25);
    let fmin = vec3<f32>(fu0, rmin.y, rmin.z);
    let fmax = vec3<f32>(fu0 + fw, rmin.y + fh, rmin.z + fd);
    let hf = city_box_hit(g, il, fmin, fmax);
    var fn_l = vec3<f32>(0.0);
    if (hf.x <= hf.y && hf.x > 0.0 && hf.x < tr) {
        tr = hf.x;
        surf = 4;
        fn_l = city_box_normal(g + tr * dl, fmin, fmax);
    }
    let q = g + tr * dl;
    // The fixture: a panel on the ceiling over the middle of the room.
    let lamp = vec3<f32>(b.ca + f.pitch * (0.35 + 0.30 * rh.w),
                         rmax.y - 0.05, g.z - CITY_SHELL_ROOM_D * 0.45);
    let dlmp = lamp - q;
    let r2 = dot(dlmp, dlmp);
    var alb = 0.72;
    var nq = vec3<f32>(0.0, 0.0, 1.0);
    if (surf == 1) { alb = 0.62; nq = vec3<f32>(-sign(dl.x), 0.0, 0.0); }
    if (surf == 2) { alb = 0.34; nq = vec3<f32>(0.0, 1.0, 0.0); }
    if (surf == 3) { alb = 0.55; nq = vec3<f32>(0.0, -1.0, 0.0); }
    if (surf == 4) { alb = 0.22; nq = fn_l; }
    let lit = max(dot(nq, dlmp * inverseSqrt(max(r2, 1e-4))), 0.0);
    var shade = alb * (0.08 + 1.5 * lit / (1.0 + r2 / 1.6));
    if (surf == 3) {
        let pd = abs(q.xz - lamp.xz);
        let panel = (1.0 - smoothstep(0.26, 0.30, pd.x))
                  * (1.0 - smoothstep(0.55, 0.60, pd.y));
        shade = shade + 5.0 * panel;
    }
    // The room's own colour: paint on the walls, wood or carpet underfoot.
    // Each tint has unit mean over its channels, so it moves hue, not the
    // normalization.
    var tint = vec3<f32>(1.0);
    if (surf <= 1) {
        tint = select(select(vec3<f32>(1.12, 1.0, 0.88),
                             vec3<f32>(0.92, 1.06, 1.02), rh.w < 0.5),
                      vec3<f32>(0.96, 0.98, 1.06), rh.x < 0.3);
    } else if (surf == 2) {
        tint = select(vec3<f32>(1.25, 0.98, 0.77),
                      vec3<f32>(0.95, 1.0, 1.05), rh.y < 0.4);
    }
    // The occupant stands on a plane between the glass and the back wall;
    // where the ray crosses it is where windowlife's figure is drawn, so the
    // person slides against the room as the camera moves.
    let fz = g.z - (0.7 + 1.4 * rh.z);
    let tf = (fz - g.z) * il.z;
    var fig_uv = vec2<f32>(-9.0);
    if (tf > 0.0 && tf < tr) {
        let pf = g + tf * dl;
        fig_uv = (vec2<f32>(pf.x - b.ca, pf.y - vfl) - plo)
                 / max(phi - plo, vec2<f32>(1e-4));
    }
    let glyph = cc_window_glyph(cc, win.wh, pane_uv, fig_uv, fp_eff);
    let room = win.e * tint * (city_room_norm(cc.win_style) * shade) * glyph;
    // The room is sub-window detail: it dissolves into the flat pane (the
    // pane's radiance through the same glyph at the glass) before the shell
    // blend begins, so the blend only ever mixes two flat panes.
    let flat = win.e * cc_window_glyph(cc, win.wh, pane_uv, pane_uv, fp_eff);
    return mix(room, flat, smoothstep(0.30, CITY_SHELL_FP_BLEND, fp_eff));
}

// --- shell shading ---------------------------------------------------------------

fn city_shell_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32,
                    fill: vec3<f32>, glow: f32) -> vec3<f32> {
    let code = h.kind - CITY_KIND_SHELL;
    let part = code & 15;
    let side = (code >> 4) & 3;
    let tier = code >> 6;
    var bx = city_tier_box(cc, tier);
    if (tier == CITY_CROWN_TIER) {
        bx = city_crown_box(cc);
    }
    let f = city_face_make(cc, bx.bmin, bx.bmax, side, tier);
    let st = city_shell_style(cc);
    let l = city_face_local(f, h.pos);
    let dl = city_face_dir(f, dir);
    // The numbers the trace used, recovered, so shading and geometry agree
    // on s: the wall-plane crossing for a ray heading into the wall, the hit
    // itself for one that never reaches it (the trace used the layer entry,
    // which a ray running along the wall enters close to where it hits).
    let pa = fp / max(h.t, 1e-3);
    var t_ref = h.t;
    if (dl.z < -1e-3) {
        t_ref = h.t - l.z / dl.z;
    }
    let t_cap = city_shell_cap(h.pos - h.t * dir, dir);
    let s = city_shell_scale(pa, t_ref, dl, st.depth, f.pitch, t_cap);
    // The in-plane pattern dissolves by the core's foreshortened footprint.
    // The shell blends into the flat facade as its relief flattens, on the
    // footprint s runs on, so the blend never completes with the relief
    // still up; and by the walk's cap.
    let fp_eff = pa * t_ref / clamp(abs(dl.z), 0.20, 1.0);
    let blend = max(smoothstep(CITY_SHELL_FP_BLEND, CITY_SHELL_FP_FLAT,
                               pa * max(t_ref, 0.0)),
                    city_shell_cap_fade(t_ref, t_cap));

    // The flat twin is read where the ray crosses the wall plane, which is
    // where the plain pass meets this wall: the flat facade finds the glass
    // behind that point itself (city_recess_off), exactly where the shell's
    // recess put it. A ray that never reaches the wall keeps the hit's foot.
    var hw = h;
    var t_wall = h.t;
    if (dl.z < -1e-3) {
        hw.pos = h.pos + (t_ref - h.t) * dir;
        t_wall = t_ref;
    } else {
        hw.pos = h.pos - l.z * vec3<f32>(f.n, 0.0);
    }
    hw.t = t_wall;
    hw.normal = vec3<f32>(f.n, 0.0);
    hw.kind = 1;
    var flat = vec3<f32>(0.0);
    if (blend > 0.0) {
        flat = city_facade_flat(cc, f, hw, dir, pa * t_wall,
                                true_pixel_angle() * t_wall, true, fill);
        if (blend >= 1.0) {
            return flat;
        }
    }

    let b = city_bay_zoned(city_zones(cc, f), f, l.x, l.y);
    var win = city_window_rated(cc, f, b.iu, b.iv,
                                city_storey_rate_at(cc, b.iv,
                                                    city_zone_lit_frac(cc, b.zone)));
    if (b.zone == CITY_ZONE_STORE) {
        win = city_shop_window(cc, win);
    }
    if (b.zone == CITY_ZONE_PENT) {
        win = city_pent_window(cc, f, b.iu, b.iv);
    }
    let n = h.normal;
    let moon = u.sun_dir.xyz;
    var ao = 1.0;
    var alb = st.albedo;
    var e = vec3<f32>(0.0);
    let plo = city_bay_pane_lo(cc, f, b);
    let phi = city_bay_pane_hi(cc, f, b);
    let vfl = f32(b.iv) * CITY_FLOOR_H;
    if (part == CITY_PART_GLASS) {
        let cosv = abs(dot(dir, n));
        e = city_glass_fresnel(cosv)
            * city_glass_env(reflect(dir, n), h.pos, glow);
        if (win.lit) {
            e = e + city_room(cc, f, b, win, l, dl, fp_eff)
                  * (1.0 - city_glass_fresnel(cosv));
        }
        alb = vec3<f32>(0.0);
    } else if (part == CITY_PART_REVEAL) {
        ao = 0.6;
    } else if (part == CITY_PART_SILL) {
        alb = st.trim;
        ao = 0.85;
    } else if (part == CITY_PART_FRAME) {
        alb = st.frame;
        ao = 0.5;
    } else if (part == CITY_PART_TRIM) {
        alb = st.trim;
    } else if (part == CITY_PART_AC) {
        alb = vec3<f32>(0.34, 0.33, 0.30);
        // The grille: horizontal louvres across the front face, dissolving
        // into their mean before they alias.
        if (n.z > 0.5 || dot(n, vec3<f32>(f.n, 0.0)) > 0.5) {
            let slat = 0.5 + 0.5 * cos(6.2831853 * l.y / 0.045);
            let g = mix(0.45 + 0.55 * slat, 0.72,
                        smoothstep(0.02, 0.06, fp_eff));
            alb = alb * g;
        }
    } else if (part == CITY_PART_PIPE) {
        let ph = pcg2d(vec2<u32>(cc.seed.x ^ f.salt, 0x1b873593u));
        alb = select(vec3<f32>(0.07, 0.07, 0.075),
                     vec3<f32>(0.32, 0.31, 0.29), city_u01(ph.x) < 0.4);
    } else if (part == CITY_PART_STEEL) {
        alb = CITY_ESCAPE_ALBEDO;
    } else if (part == CITY_PART_SLAB) {
        alb = st.trim;
    } else if (part == CITY_PART_CANOPY) {
        // Corrugated sheet: ribs along the wall's normal, every 76 mm.
        let rib = 0.5 + 0.5 * cos(6.2831853 * l.x / 0.076);
        alb = vec3<f32>(0.20, 0.21, 0.20)
              * mix(0.6 + 0.4 * rib, 0.8, smoothstep(0.02, 0.07, fp_eff));
    } else if (part == CITY_PART_PANEL) {
        // Frosted glass: dark, with the night's reflection on it.
        alb = vec3<f32>(0.05, 0.055, 0.06);
        e = city_glass_fresnel(abs(dot(dir, n)))
            * city_glass_env(reflect(dir, n), h.pos, glow);
    } else {
        // The wall. Brick is coursed in running bond — 65 mm courses and
        // 215 mm bricks in 10 mm of lighter mortar, each brick its own shade
        // — and the multiplier averages to exactly 1 (mortar share 0.172 at
        // 1.5, bricks at 0.896), so the courses dissolve into the very
        // albedo the flat facade uses once they are sub-pixel.
        if (b.zone == CITY_ZONE_PLANT) {
            // Louvre blades: a lit top face and a shadowed gap, averaging
            // to exactly the louvre albedo (0.6 x 1.3 + 0.4 x 0.55).
            let lv = fract(l.y / CITY_LOUVRE_PITCH) < 0.6;
            alb = CITY_LOUVRE_ALBEDO
                  * mix(select(0.55, 1.3, lv), 1.0,
                        smoothstep(0.03, 0.10, fp_eff));
        } else if (st.brick) {
            let cy = l.y / 0.075;
            let row = floor(cy);
            let cx = l.x / 0.225 + 0.5 * (row - 2.0 * floor(0.5 * row));
            let bi = vec2<i32>(i32(floor(cx)), i32(row));
            let jb = city_u01(pcg2d(vec2<u32>(bitcast<u32>(bi.x) ^ cc.seed.x,
                                              bitcast<u32>(bi.y))).x);
            let mortar = fract(cy) > 0.8667 || fract(cx) > 0.9556;
            let m = select(0.896 * (0.8 + 0.4 * jb), 1.5, mortar);
            alb = alb * mix(m, 1.0, smoothstep(0.012, 0.05, fp_eff));
        }
        // Below a unit the condensate has left its stain: a
        // darker streak the width of the unit, fading down the wall.
        let ac = city_shell_ac(cc, f, st, b);
        if (ac.on) {
            let bu = l.x - b.ca;
            let bv = l.y - vfl;
            let under = smoothstep(ac.lo.x + 0.1, ac.lo.x + 0.25, bu)
                      * (1.0 - smoothstep(ac.hi.x - 0.25, ac.hi.x - 0.1, bu))
                      * select(0.0, 1.0, bv < ac.lo.y)
                      * exp(-(ac.lo.y - bv) / 1.4);
            alb = alb * (1.0 - 0.55 * under);
        }
    }
    // Light from the pane of this bay, on everything in and around its
    // reveal that can see it.
    var spill = 0.0;
    if (win.lit && part != CITY_PART_GLASS && part != CITY_PART_WALL
        && b.in_bay) {
        let d = st.reveal;
        let c = city_face_world(f, vec3<f32>(b.ca + plo.x, vfl + plo.y, -d))
                + vec3<f32>(f.t * f.ua + f.n * f.wp, 0.0);
        let e1 = city_face_world(f, vec3<f32>(phi.x - plo.x, 0.0, 0.0));
        let e2 = vec3<f32>(0.0, 0.0, phi.y - plo.y);
        spill = city_rect_ff(h.pos, n, c, e1, e2);
    }
    let light = CITY_SHELL_FILL_K * ao
                * (fill + CITY_MOONLIGHT * max(dot(n, moon), 0.0))
              + city_shell_light(h.pos, n)
              + win.e * (0.8 * spill);
    e = e + alb * light;
    // Everything the facade components paint on the wall.
    if (part == CITY_PART_WALL || part == CITY_PART_CORE) {
        e = e + cc_facade_detail(cc, hw, dir, l.x, l.y, fp);
    }
    // A curtain wall's spandrels are glass too, and so is the cladding of
    // its cores and plant floors: the same reflection the flat facade gives
    // the whole skin.
    if (cc.win_style == 3
        && (part == CITY_PART_WALL || part == CITY_PART_CORE)) {
        e = e + city_glass_fresnel(abs(dot(dir, n)))
                * city_glass_env(reflect(dir, n), h.pos, glow);
    }
    return mix(e, flat, blend);
}

// The shop row's radiance as the street sees it, the strip of light the
// sidewalk's spill, the steam and the street furniture read: what its
// windows (the STORE zone) emit on average over the strip, the glass's
// share of it by a lit room's mean radiance through the glass at normal
// incidence, in the house's colour (its one draw for a monochrome house,
// the palette's mean for a mixed one). The frames' and the recess's shade
// on the glass are left out: a street-scale light, not the wall's mean.
// Zero where the ground floor is not commercial.
fn city_storefront_glow(cc: CityCell) -> vec3<f32> {
    if (!cc.built || cc.store_draw >= 0.65
        || cc.land_use != CITY_USE_URBAN) {
        return vec3<f32>(0.0);
    }
    let style_gain = select(select(1.0, 0.80, cc.win_style == 1),
                            0.55, cc.win_style == 3);
    // The glass's share of the strip: its width in a bay, the bays' share
    // of the ground box's perimeter (each face keeps a corner strip of wall
    // at both ends, city_face_make), its height in the strip.
    let w = cc.b1max.xy - cc.b1min.xy;
    let corner = min(vec2<f32>(CITY_SHELL_CORNER), 0.25 * w);
    let bays = max(w.x + w.y - 2.0 * (corner.x + corner.y), 0.0)
               / max(w.x + w.y, 1e-3);
    let glass = (CITY_STORE_PANE_HI.x - CITY_STORE_PANE_LO.x) * bays
                * (CITY_STORE_PANE_HI.y - CITY_STORE_PANE_LO.y)
                * CITY_FLOOR_H / CITY_STORE_H;
    // The shops' mean colour. A mixed house's draws are uniform on [0, 1],
    // so its mean is the bands' colours by their widths, which the
    // building's bias moves (city_room_band_edge, clamped at 1). A
    // monochrome house is taken at its draw's centre: exact unless its
    // +-0.04 spread straddles a band's edge. (The street reads this several
    // times a pixel; the share-weighted mean over the spread cost the street
    // view 9%.)
    var colour = vec3<f32>(0.0);
    if (cc.win_mono >= 0.0) {
        colour = city_room_color(cc.win_mono, cc.palette_bias);
    } else {
        let e = min(vec4<f32>(city_room_band_edge(1, cc.palette_bias),
                              city_room_band_edge(2, cc.palette_bias),
                              city_room_band_edge(3, cc.palette_bias),
                              city_room_band_edge(4, cc.palette_bias)),
                    vec4<f32>(1.0));
        let e0 = min(city_room_band_edge(0, cc.palette_bias), 1.0);
        colour = city_room_band_colour(0) * e0
                 + city_room_band_colour(1) * (e.x - e0)
                 + city_room_band_colour(2) * (e.y - e.x)
                 + city_room_band_colour(3) * (e.z - e.y)
                 + city_room_band_colour(4) * (e.w - e.z)
                 + city_room_band_colour(5) * (1.0 - e.w);
    }
    colour = mix(colour, city_room_band_colour(4), CITY_SHOP_CYAN_FRAC);
    return colour * cc_window_glyph_mean()
           * (CITY_ROOM_RADIANCE * CITY_WIN_BRIGHT_MEAN * style_gain * glass
              * (1.0 - city_glass_fresnel(1.0)));
}

// The recess, seen flat. A pane sits st.reveal behind the wall plane, so a
// ray (face-local direction dl, into the wall) that crosses the opening at
// bay-local p meets the glass at p + city_recess_off.
fn city_recess_off(st: CityShellStyle, dl: vec3<f32>) -> vec2<f32> {
    return st.reveal * dl.xy / max(-dl.z, 1e-4);
}
