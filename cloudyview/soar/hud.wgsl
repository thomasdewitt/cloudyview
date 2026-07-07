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
    // x = line half-width px, y = border width px, z = halo width px.
    style: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: HudUniforms;
@group(0) @binding(1) var map_tex: texture_2d<f32>;
@group(0) @binding(2) var map_samp: sampler;

const RED: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);
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

fn inside_triangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>,
                   c: vec2<f32>) -> bool {
    let area = edge(a, b, c);
    let e0 = edge(a, b, p);
    let e1 = edge(b, c, p);
    let e2 = edge(c, a, p);
    if (area >= 0.0) {
        return e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0;
    }
    return e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0;
}

fn stroke_coverage(distance_px: f32, half_width_px: f32) -> f32 {
    return 1.0 - smoothstep(half_width_px, half_width_px + 1.0, distance_px);
}

fn disk_coverage(distance_px: f32, radius_px: f32) -> f32 {
    return 1.0 - smoothstep(radius_px, radius_px + 1.0, distance_px);
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

    if (u.camera.z < 0.5) {
        let left = uv_to_screen(u.rays.xy);
        let right = uv_to_screen(u.rays.zw);

        if (inside_triangle(p, cam, left, right)) {
            out = over(out, vec4<f32>(RED, 0.13));
        }

        let ray_d = min(
            dist_to_segment(p, cam, left),
            dist_to_segment(p, cam, right)
        );
        let halo = stroke_coverage(ray_d, line_hw + halo_hw);
        if (halo > 0.0) {
            out = over(out, vec4<f32>(INK, 0.38 * halo));
        }
        let ray = stroke_coverage(ray_d, line_hw);
        if (ray > 0.0) {
            out = over(out, vec4<f32>(RED, 0.96 * ray));
        }
    } else {
        let ring_d = abs(length(p - cam) - u.camera.w);
        let halo = stroke_coverage(ring_d, line_hw + halo_hw);
        if (halo > 0.0) {
            out = over(out, vec4<f32>(INK, 0.38 * halo));
        }
        let ring = stroke_coverage(ring_d, line_hw);
        if (ring > 0.0) {
            out = over(out, vec4<f32>(RED, 0.96 * ring));
        }
    }

    let marker_d = length(p - cam);
    let outer = disk_coverage(marker_d, u.frame.w + 2.0);
    if (outer > 0.0) {
        out = over(out, vec4<f32>(INK, 0.30 * outer));
    }
    let rim = disk_coverage(marker_d, u.frame.w + 1.2);
    if (rim > 0.0) {
        out = over(out, vec4<f32>(WHITE, 0.90 * rim));
    }
    let dot = disk_coverage(marker_d, u.frame.w);
    if (dot > 0.0) {
        out = over(out, vec4<f32>(RED, 0.98 * dot));
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
