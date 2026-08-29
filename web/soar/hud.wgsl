// CloudyView soar: HUD minimap overlay.
//
// A tiny post-volume pass. The albedo map texture is static; per-frame
// uniforms carry the screen-space rectangle and camera/FOV overlay geometry.

struct HudUniforms {
    // x/y = framebuffer size, z = map opacity, w = marker radius (px).
    frame: vec4<f32>,
    // x/y = top-left of minimap, z/w = width/height (px).
    rect: vec4<f32>,
    // x/y = camera map UV (east, north), z = mode (0 wedge, 1 circle),
    // w = circle radius in screen px.
    camera: vec4<f32>,
    // left endpoint UV.xy, right endpoint UV.zw (wedge mode only).
    rays: vec4<f32>,
    // x = line half-width px, y = border width px, z = halo width px,
    // w = nest rectangle enable (0 or 1).
    style: vec4<f32>,
    // Nested field footprint in map UV: xy = min corner, zw = max corner.
    nest: vec4<f32>,
    // x/y = haze e-folding distance as a radius in screen px per axis (an
    // ellipse when the map is not square in metres); z/w unused.
    haze: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: HudUniforms;
@group(0) @binding(1) var map_tex: texture_2d<f32>;
@group(0) @binding(2) var map_samp: sampler;

// The overlay colour: the app's warm accent, matching soar's --hot, the
// landing page's --amber and basic_render's ACCENT, so the camera marker
// looks the same in a glimpse PNG and in the corner of the flight view.
// It replaces pure red, which read as an error state and fought the cloud
// ramp at both ends.
//
// Duplicated from MAP_ACCENT in constants.js because a shader constant
// cannot import; tests/test_map_ramp_parity.py fails if the two drift.
const ACCENT: vec3<f32> = vec3<f32>(232.0 / 255.0, 131.0 / 255.0, 74.0 / 255.0);
const WHITE: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const INK: vec3<f32> = vec3<f32>(0.02, 0.025, 0.035);

struct VSOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    var out: VSOut;
    out.pos = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

fn over(dst: vec4<f32>, src: vec4<f32>) -> vec4<f32> {
    let src_a = clamp(src.a, 0.0, 1.0);
    let dst_a = clamp(dst.a, 0.0, 1.0);
    let out_a = src_a + dst_a * (1.0 - src_a);
    if (out_a <= 1e-5) {
        return vec4<f32>(0.0);
    }
    let out_rgb = (
        src.rgb * src_a + dst.rgb * dst_a * (1.0 - src_a)
    ) / out_a;
    return vec4<f32>(out_rgb, out_a);
}

fn uv_to_screen(map_uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        u.rect.x + map_uv.x * u.rect.z,
        u.rect.y + (1.0 - map_uv.y) * u.rect.w
    );
}

fn dist_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ba = b - a;
    let denom = max(dot(ba, ba), 1e-5);
    let h = clamp(dot(p - a, ba) / denom, 0.0, 1.0);
    return length(p - (a + ba * h));
}

fn edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    return ab.x * ap.y - ab.y * ap.x;
}

// Inside the ellipse of the haze e-folding distance around the camera —
// the region the view actually reaches. Per-axis radii because a metric
// distance is an ellipse on a map that is not square in metres.
fn within_haze(p: vec2<f32>, cam: vec2<f32>) -> bool {
    let q = (p - cam) / max(u.haze.xy, vec2<f32>(1e-4));
    return dot(q, q) <= 1.0;
}

fn stroke_coverage(distance_px: f32, half_width_px: f32) -> f32 {
    return 1.0 - smoothstep(half_width_px, half_width_px + 1.0, distance_px);
}

fn disk_coverage(distance_px: f32, radius_px: f32) -> f32 {
    return 1.0 - smoothstep(radius_px, radius_px + 1.0, distance_px);
}

// Unsigned distance to a rectangle's border (standard box SDF, absolute
// value): zero on the outline, growing both inward and outward.
fn dist_to_rect_border(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    let d = max(lo - p, p - hi);
    let outside = length(max(d, vec2<f32>(0.0)));
    let inside = min(max(d.x, d.y), 0.0);
    return abs(outside + inside);
}

fn inside_rect(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> bool {
    return p.x >= lo.x && p.y >= lo.y && p.x <= hi.x && p.y <= hi.y;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;
    let r0 = u.rect.xy;
    let r1 = u.rect.xy + u.rect.zw;

    if (p.x < r0.x || p.y < r0.y || p.x > r1.x || p.y > r1.y) {
        return vec4<f32>(0.0);
    }

    let local = (p - r0) / u.rect.zw;
    let tex_uv = vec2<f32>(local.x, 1.0 - local.y);
    let map_rgb = textureSampleLevel(map_tex, map_samp, tex_uv, 0.0).rgb;

    var out = vec4<f32>(map_rgb, u.frame.z);
    let cam = uv_to_screen(u.camera.xy);
    let line_hw = u.style.x;
    let halo_hw = u.style.z;

    // Nested-field footprint, drawn UNDER the camera overlay: it is context,
    // not the thing you steer by. White line over an ink halo so it reads on
    // both the bright cloud and dark ocean ends of the albedo map, and
    // thinner than the camera's rays so the two never compete.
    if (u.style.w > 0.5) {
        // uv_to_screen flips y, so the UV min/max corners come back as
        // opposite screen corners; re-sort them.
        let c0 = uv_to_screen(u.nest.xy);
        let c1 = uv_to_screen(u.nest.zw);
        let lo = min(c0, c1);
        let hi = max(c0, c1);
        if (inside_rect(p, lo, hi)) {
            out = over(out, vec4<f32>(WHITE, 0.10));
        }
        let nest_hw = max(line_hw * 0.7, 1.0);
        let nest_d = dist_to_rect_border(p, lo, hi);
        let halo = stroke_coverage(nest_d, nest_hw + halo_hw);
        if (halo > 0.0) {
            out = over(out, vec4<f32>(INK, 0.30 * halo));
        }
        let line = stroke_coverage(nest_d, nest_hw);
        if (line > 0.0) {
            out = over(out, vec4<f32>(WHITE, 0.80 * line));
        }
    }

    if (u.camera.z < 0.5) {
        // The endpoints sit ON the haze ellipse (minimap.js puts them there),
        // so the rays end where the view does, and the shaded region is the
        // pie slice between them: inside the angular wedge AND closer than
        // the haze distance. The wedge test is two half-planes rather than
        // the endpoint triangle, whose straight far edge would cut the pie's
        // arc off.
        let left = uv_to_screen(u.rays.xy);
        let right = uv_to_screen(u.rays.zw);

        let s = edge(cam, left, right);
        let in_wedge = edge(cam, left, p) * s >= 0.0
            && edge(cam, right, p) * s <= 0.0;
        if (in_wedge && within_haze(p, cam)) {
            out = over(out, vec4<f32>(ACCENT, 0.05));
        }

        let ray_d = min(
            dist_to_segment(p, cam, left),
            dist_to_segment(p, cam, right)
        );
        let halo = stroke_coverage(ray_d, line_hw + halo_hw);
        if (halo > 0.0) {
            out = over(out, vec4<f32>(INK, 0.22 * halo));
        }
        let ray = stroke_coverage(ray_d, line_hw);
        if (ray > 0.0) {
            // Opaque. The coverage factor is antialiasing, not translucency:
            // it is 1 inside the stroke and falls off over the last pixel, so
            // multiplying by it keeps the edge smooth while the body of the
            // line sits at full strength over the map.
            out = over(out, vec4<f32>(ACCENT, ray));
        }
    } else {
        // Straight up or down: every bearing is in frame, so the whole
        // closer-than-haze ellipse is the visible region.
        if (within_haze(p, cam)) {
            out = over(out, vec4<f32>(ACCENT, 0.05));
        }
        let ring_d = abs(length(p - cam) - u.camera.w);
        let halo = stroke_coverage(ring_d, line_hw + halo_hw);
        if (halo > 0.0) {
            out = over(out, vec4<f32>(INK, 0.22 * halo));
        }
        let ring = stroke_coverage(ring_d, line_hw);
        if (ring > 0.0) {
            out = over(out, vec4<f32>(ACCENT, ring));
        }
    }

    // The dot, with a rim and a halo that keep it findable at both ends of the
    // albedo ramp: white for the dark-ocean end, ink for the bright-cloud end.
    //
    // Both are sized FROM the dot rather than at fixed pixel offsets. They
    // used to be +1.2 and +2.0 px, which was fine at the old radius and wrong
    // the moment it halved — at a 2.5 px dot a fixed +1.2 rim is half again as
    // wide as the mark it is meant to trim, so the marker would have read as a
    // white blob with an orange centre.
    let marker_d = length(p - cam);
    let rim_w = max(0.7, u.frame.w * 0.45);
    let outer = disk_coverage(marker_d, u.frame.w + rim_w * 2.0);
    if (outer > 0.0) {
        out = over(out, vec4<f32>(INK, 0.22 * outer));
    }
    let rim = disk_coverage(marker_d, u.frame.w + rim_w);
    if (rim > 0.0) {
        out = over(out, vec4<f32>(WHITE, 0.55 * rim));
    }
    let dot = disk_coverage(marker_d, u.frame.w);
    if (dot > 0.0) {
        out = over(out, vec4<f32>(ACCENT, dot));
    }

    let edge_d = min(
        min(p.x - r0.x, r1.x - p.x),
        min(p.y - r0.y, r1.y - p.y)
    );
    let border = 1.0 - smoothstep(u.style.y, u.style.y + 1.0, edge_d);
    if (border > 0.0) {
        out = over(out, vec4<f32>(INK, 0.75 * border));
    }

    return out;
}
