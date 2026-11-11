// The sea is not in this module: what shared code calls of it, with a
// neutral body. Reached only under ocean_on, which is false there.
fn ocean_shade_dispatch(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>,
                        sun_shadow: vec3<f32>, t_hit: f32,
                        jit: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
