fn cc_extra_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    let h_skyway = cc_skyway_trace(o, dir, inv_dir);
    if (h_skyway.hit && h_skyway.t < res.t) { res = h_skyway; }
    return res;
}

fn cc_cell_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                       t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    let h_rooftopworks = cc_rooftopworks_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_rooftopworks.hit && h_rooftopworks.t < res.t) { res = h_rooftopworks; }
    let h_skybridges = cc_skybridges_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_skybridges.hit && h_skybridges.t < res.t) { res = h_skybridges; }
    let h_streetlife = cc_streetlife_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_streetlife.hit && h_streetlife.t < res.t) { res = h_streetlife; }
    let h_manholesteam = cc_manholesteam_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_manholesteam.hit && h_manholesteam.t < res.t) { res = h_manholesteam; }
    let h_industry = cc_industry_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_industry.hit && h_industry.t < res.t) { res = h_industry; }
    return res;
}

fn cc_component_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    if (h.kind >= 200 && h.kind < 300) {
        return cc_skyway_shade(h, cc, dir, fp);
    }
    if (h.kind >= 600 && h.kind < 700) {
        return cc_rooftopworks_shade(h, cc, dir, fp);
    }
    if (h.kind >= 500 && h.kind < 600) {
        return cc_skybridges_shade(h, cc, dir, fp);
    }
    if (h.kind >= 100 && h.kind < 200) {
        return cc_streetlife_shade(h, cc, dir, fp);
    }
    if (h.kind >= 900 && h.kind < 1000) {
        return cc_manholesteam_shade(h, cc, dir, fp);
    }
    if (h.kind >= 700 && h.kind < 800) {
        return cc_industry_shade(h, cc, dir, fp);
    }
    if (h.kind >= 800 && h.kind < 900) {
        return cc_streetfurniture_shade(h, cc, dir, fp);
    }
    // An unclaimed component kind is a bug, and this is its color.
    return vec3<f32>(1.0, 0.0, 1.0);
}

fn cc_facade_detail(cc: CityCell, h: CityHit, dir: vec3<f32>,
                    uc: f32, vc: f32, fp: f32)
        -> vec3<f32> {
    var e = vec3<f32>(0.0);
    e = e + cc_facadeworks_detail(cc, h, dir, uc, vc, fp);
    e = e + cc_adscreens_facade(cc, h, dir, uc, vc, fp);
    e = e + cc_industry_facade(cc, h, dir, uc, vc, fp);
    return e;
}

fn cc_facade_cover(cc: CityCell, h: CityHit, uc: f32, vc: f32, fp: f32)
        -> f32 {
    var m = 0.0;
    m = max(m, cc_facadeworks_cover(cc, h, uc, vc, fp));
    m = max(m, cc_adscreens_cover(cc, h, uc, vc, fp));
    return m;
}

fn cc_window_glyph(cc: CityCell, wh: vec4<f32>, pane_uv: vec2<f32>,
                   deep_uv: vec2<f32>, fp: f32)
        -> vec3<f32> {
    var t = vec3<f32>(1.0);
    t = t * cc_windowlife_glyph(cc, wh, pane_uv, deep_uv, fp);
    return t;
}

fn cc_window_glyph_mean()
        -> vec3<f32> {
    var t = vec3<f32>(1.0);
    t = t * cc_windowlife_glyph_mean();
    return t;
}

fn cc_media(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, t_max: f32,
            pixel_angle: f32, rnd: f32) -> vec4<f32> {
    var acc = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let m_manholesteam = cc_manholesteam_media(o, dir, inv_dir, t_max, pixel_angle, rnd);
    acc = vec4<f32>(acc.rgb + acc.a * m_manholesteam.rgb, acc.a * m_manholesteam.a);
    return acc;
}

fn cc_near_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                       t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    let h_streetlife = cc_streetlife_near_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_streetlife.hit && h_streetlife.t < res.t) { res = h_streetlife; }
    let h_streetfurniture = cc_streetfurniture_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_streetfurniture.hit && h_streetfurniture.t < res.t) { res = h_streetfurniture; }
    return res;
}

fn cc_roof_detail(cc: CityCell, h: CityHit, fp: f32, fp_filter: f32,
                  dir: vec3<f32>, direct: bool, base: vec3<f32>)
        -> vec3<f32> {
    var e = vec3<f32>(0.0);
    e = e + cc_rooftopworks_roof(cc, h, fp, fp_filter, dir, direct, base);
    e = e + cc_industry_roof(cc, h, fp, fp_filter, dir, direct, base);
    return e;
}
