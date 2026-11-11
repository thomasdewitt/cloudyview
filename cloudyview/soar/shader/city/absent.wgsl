// The night city is not in this module: what shared code calls of it, with
// neutral bodies. Each is reached only under `if (CITY)`.
const CITY: bool = false;

fn surface_street_city(cr: CameraRay, prev: vec4<f32>) -> SurfaceOut {
    var o: SurfaceOut;
    o.surf = prev;
    o.veil = veil_none();
    return o;
}
fn steam_live_city(cr: CameraRay, acc: vec4<f32>, t_surf: f32) -> vec4<f32> {
    return acc;
}
fn surface_pass_city(dir: vec3<f32>, pixel_angle: f32) -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1e30);
}
fn surface_refine_city(cr: CameraRay, plain: vec4<f32>) -> vec4<f32> {
    return plain;
}
fn city_glow_sample(xy: vec2<f32>, lod: f32) -> f32 {
    return 0.0;
}
fn city_fold_xy(xy: vec2<f32>) -> vec2<f32> {
    return xy;
}
fn city_uplight_probe_tau(p: vec3<f32>, jit: f32) -> f32 {
    return 0.0;
}
fn night_sky_radiance(dir: vec3<f32>, moon: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}
