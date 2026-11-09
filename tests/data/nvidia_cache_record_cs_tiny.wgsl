@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(32, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    var a = f32(g.x) * 0.001 + f32(g.y);
    for (var i = 0; i < 6; i = i + 1) {
        a = sin(a) * 1.3 + cos(a * 0.7);
    }
    buf[g.x + g.y * 1024u] = a;
}
