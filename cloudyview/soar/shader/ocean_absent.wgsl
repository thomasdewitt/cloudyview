// The sea's stand-in, for a module compiled for another surface: the one
// function shared code calls of it, with a neutral body. It is never
// reached, because ocean_on is false in such a module.
fn ocean_shade(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>,
               sun_shadow: vec3<f32>, t_hit: f32,
               jit: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
