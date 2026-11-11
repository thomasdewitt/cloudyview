// EMPTY-SPACE SKIPPING: measured three times, lost three times. Do not read
// the sparseness of cloud fields as an unclaimed opportunity — it is a claim
// that has been tested to destruction, and this note replaces the TODO that
// used to invite it.
//
//   2026-07-17  view-march occupancy skip. 1.2-2.5x SLOWER. Fine-dt empty
//               samples are cheap; the skip machinery is not.
//   2026-08-11  8^3 majorant/occupancy grid on the light march. Proven
//               max|diff| == 0.0 and measured 0.5-0.9x. Machinery cost 47 ms;
//               skipping 36% of bricks recovered 0.8 ms.
//   2026-08-13  full sparse bricks: page table, atlas with a 1-voxel apron,
//               brick-DDA traversal that consults only the page table in
//               vacuum, and a skip that lands on the march's own sample
//               lattice so the picture is unchanged. Proven correct against a
//               dense render (the difference is a Monte Carlo realization and
//               falls as 1/sqrt(frames)). Measured 3-6x SLOWER on an RTX 5080
//               across both a 33%-occupied field and an 8%-occupied one, and
//               slower again on Apple silicon — which was the one hypothesis
//               left standing after the first two, since both of those were
//               5080 texture-latency measurements wearing the costume of a
//               measurement about an algorithm.
//
//   2026-09-23  no skipping at all, only the READ: an 8^3 max-sigma grid
//               (one compute pass, dilated for the trilinear footprint),
//               cached per brick crossing, so a step the grid proves empty
//               takes sigma = 0 in place of the fetch and runs the same
//               arithmetic at the same t. Exact (A/B max|diff| == 0) and it
//               removed the fetch from 60-77% of all view steps on TWP-ICE —
//               and was 16-43% SLOWER on the 5080, 18-49% at the clock lock
//               that stands in for an M1; only DYCOMS's view from inside
//               the deck gained (4-8%). The control that settles it: the
//               same code compiled in behind a guard that never fires cost
//               14-20% by itself. The fetches it removed were nearly free;
//               fs_main's register budget is not.
//
// The mechanism, consistent across all four: this march is texture-LATENCY
// bound, and every scheme that knows where the empty space is must ask
// something to find out. In occupied space that ask lands on the critical
// path of a loop that tolerates no added control flow (a bit-identical
// micro-opt — reusing the view sample as the light march's first tap — also
// regressed). The skip cannot outrun the question.
//
// Two things worth knowing before a fifth attempt. The per-crossing fetch
// that the third attempt left untried has now been tried (the fourth) and it
// is not the fetch that costs: any per-step machinery in fs_main pays for
// its registers whether it fires or not. And bricking has a memory
// break-even that is lower than it looks: an 8^3 brick with its apron stores
// 1000 texels to hold 512,
// so it only shrinks a field below 51% brick occupancy (16^3: 70%, 32^3:
// 83%). The whole attempt is in git — see the commit that reverted it for
// the entry point, and benchmarking/soar_frame_results.md for the numbers.
//
// What DID pay, and shipped, is cropping the empty sky off a field at load:
// same insight about sparseness, applied to storage rather than traversal,
// where nothing has to be asked per sample. See web/soar/zcrop.js.

// FS_MAIN IS ONE COMPILE, and the driver treats it as one (2026-09-23). Five
// measurements from one day, all the same fact:
//   - the empty-space guard above cost 14-20% while never firing;
//   - compiling out the lighting method a tier does not use (LIGHT_CACHE,
//     SKY_PROBE) bought 3-14%, where the branch it replaced was never taken;
//   - the city's 3D facade shell cost 35-50% on pixels that never run it,
//     by pushing the whole kernel over a register threshold (which is why
//     the opaque surfaces are moving to their own pass);
//   - a function-local `var c: array<f32, 4>` in CITY-only code moved DAY
//     frames by 2^-11 — a specialization that was compiled out still
//     changed how the rest of the kernel was compiled. Scalars fixed it.
//   - running the forward pre-march lazily, at a ray's first deep-shadow
//     sample instead of before the loop, was exact and saves the march for
//     every ray that never gets there — and cost 28-213% on the views where
//     rays do get there, because a call inside the loop is paid in
//     registers, and lanes reaching it at different steps serialize.
// So a change's cost is not where its code runs, and a bit-identity claim
// has to be proven on scenes the change does not touch. Measure with
// tools/benchmarking/soar_pass_timer.py and prove with
// tools/soar_shader_ab.py --scene day|city|nested, both on the
// specialization the app actually compiles.
//
// And look, rather than guess: tools/soar_regcount.py prints the register
// count, spills and size of the binary wgpu actually built for an entry
// point (read back from the NVIDIA driver's shader cache; a pure compile,
// no GPU work). A count is a diagnosis, not a score — fs_main at Low was
// 96 registers one day against 80 the morning before, and faster — so
// time what it suggests. Near 128 a count is not stable either: the driver
// picks each stage's register target by its own spill-cost estimate, and
// a stage whose demand sits near it flips between 128 with spills (16
// warps an SM) and ~160 without (12 warps) on changes that do no work —
// the city's first stage, 10-14% apart (2026-11-09). Counts quoted before
// that date came from a compile of our own (naga 26), not the binary the
// app runs, and can be wrong in either direction.

// The host may bind either r32float (reference/default) or filterable r16float
// density. Both expose texture_3d<f32> here; fp16 only changes storage/bandwidth.
// Both levels always share one format.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Slab-method ray/AABB intersection. Returns (t_near, t_far); miss when
// t_near > t_far.
fn ray_box(origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    let t0 = (u.bmin.xyz - origin) * inv_dir;
    let t1 = (u.bmax.xyz - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

// ray_box's inv_dir for a direction that may lie along an axis. 1 / 0 is an
// infinity, and in WGSL an operation that produces one is indeterminate: the
// driver may assume it never happens, so what the slab test then returns is
// whatever the codegen of the day makes of it. It did differ — the desert's
// straight-up march, (0, 0, 1) in a bounded domain, measured the whole cloud
// column above the ground in one build and no cloud at all in the next
// (2026-09-24). A component of 1e30 bounds the ray only where the origin is
// outside that slab, which is what the infinity was meant to do.
fn safe_inv_dir(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(select(1e30, 1.0 / d.x, abs(d.x) > 1e-9),
                     select(1e30, 1.0 / d.y, abs(d.y) > 1e-9),
                     select(1e30, 1.0 / d.z, abs(d.z) > 1e-9));
}

// One refinement level's sample, plus the march step sizes that level asks
// for. witness carries the same two things out of _sample_sigma_nested (the
// value and the level index that sets dt_max).
struct LevelSample {
    sigma: f32,
    dt_view: f32,
    dt_light: f32,
    in_nest: bool,
}

// Data dims in FIELD axis order (nx, ny, nz). The texture is stored
// transposed — (w=nz, h=ny, d=nx) — and holds nothing but the field, so this
// is a swizzle and no longer a swizzle-minus-a-border.
fn level_data_dims(t: texture_3d<f32>) -> vec3<f32> {
    let tex_dims = vec3<f32>(textureDimensions(t, 0));
    return vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
}

// Voxel size (m) of whichever level is active at the sample point.
fn level_voxel_size(in_nest: bool) -> vec3<f32> {
    if (NESTED && in_nest) {
        return (u.nest_bmax.xyz - u.nest_bmin.xyz) / level_data_dims(nest_vol);
    }
    return (u.bmax.xyz - u.bmin.xyz) / level_data_dims(vol);
}

// Fold a world point into the outer tile. The scene — nest included — is one
// periodic tile, so this runs before any level test: every domain copy then
// carries a copy of the nest at the same place within the tile.
fn wrap_to_domain(p: vec3<f32>) -> vec3<f32> {
    if (!PERIODIC_DOMAIN) {
        return p;
    }
    let extent = u.bmax.xy - u.bmin.xy;
    let g = fract((p.xy - u.bmin.xy) / extent);
    return vec3<f32>(u.bmin.xy + g * extent, p.z);
}

// Containment test for the already-wrapped point (witness._active_level).
fn in_nest_box(q: vec3<f32>) -> bool {
    if (!NESTED) {
        return false;
    }
    return all(q >= u.nest_bmin.xyz) && all(q <= u.nest_bmax.xyz);
}

// The linear taper a zero ghost texel used to supply, computed instead.
//
// Sampling a zero-bordered volume at data coordinate g gives, along one axis,
// v_0*(g+1) for g in [-1, 0] and v_{N-1}*(N-g) for g in [N-1, N] — a ramp from
// the outermost data value down to zero across the outer half-cell either
// side. Clamp-to-edge on an UNPADDED texture returns exactly those same
// outermost values over that range and holds them beyond it, so multiplying by
// this window reproduces the padded result. Trilinear filtering is separable
// and the border was zero on every axis, so the three windows multiply, which
// makes the identity hold at edges, at corners, and everywhere between.
//
// That is what buys the two texels back: a 2048-cell axis no longer asks a
// browser clamped to the WebGPU spec floor for 2050.
fn edge_taper(g: f32, n: f32) -> f32 {
    return clamp(min(g + 1.0, n - g), 0.0, 1.0);
}

// Extinction (m^-1) from one level via hardware trilinear filtering.
// Coordinate swizzle: texture is (w=nz, h=ny, d=nx), see file header. The
// AABB is half-cell padded, so cell centre i sits at fraction (i+0.5)/N of
// the box: data_g = fraction*N - 0.5 puts data value i at data_g=i, texel i
// IS data value i, and the normalized coord is (data_g+0.5)/N — which is the
// box fraction itself, with no border to skip past.
//
// `periodic_level` picks how the domain edge behaves. A doubly periodic outer
// level wraps: the repeat sampler fetches the far face as the trilinear
// partner across the seam, which is what the uploaded ghost ring used to do,
// exactly and for free, corners included. Everything else — every nest, and a
// non-periodic outer level — tapers into zero at its own boundary, which is
// how a nest blends out into the coarse field around it. Either way the
// vertical is never periodic and always tapers.
fn sample_level(t: texture_3d<f32>, q: vec3<f32>,
                bmin: vec3<f32>, bmax: vec3<f32>, periodic_level: bool) -> f32 {
    let tex_dims = vec3<f32>(textureDimensions(t, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    let data_g = ((q - bmin) / (bmax - bmin)) * dims - 0.5;
    let tex_coord = vec3<f32>(
        data_g.z + 0.5,
        data_g.y + 0.5,
        data_g.x + 0.5
    ) / tex_dims;
    let taper_z = edge_taper(data_g.z, dims.z);
    if (periodic_level) {
        return textureSampleLevel(t, vol_wrap_samp, tex_coord, 0.0).r * taper_z;
    }
    let taper = edge_taper(data_g.x, dims.x) * edge_taper(data_g.y, dims.y)
              * taper_z;
    return textureSampleLevel(t, vol_samp, tex_coord, 0.0).r * taper;
}

// Sun optical depth from the baked cache — the trilinear read that replaces
// light_march_tau for cloud samples under LIGHT_CACHE.
//
// Same coordinate convention as sample_level over the same outer AABB (the
// cache's texel i sits AT data coordinate i, see fs_bake_light), same
// (w=nz, h=ny, d=nx) transposition, same lateral wrap in a periodic domain —
// but NO edge taper: tau is not extinction, and ramping it to zero at the
// domain edge would punch sunlight into the boundary column. Clamp-to-edge
// on the vertical is right at both ends: above the top slice tau is ~0 and
// below the bottom the nearest baked value is the honest answer.
fn sample_light_tau(p: vec3<f32>) -> f32 {
    let q = wrap_to_domain(p);
    let tex_dims = vec3<f32>(textureDimensions(light_tau, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    let data_g = ((q - u.bmin.xyz) / (u.bmax.xyz - u.bmin.xyz)) * dims - 0.5;
    let tex_coord = vec3<f32>(
        data_g.z + 0.5,
        data_g.y + 0.5,
        data_g.x + 0.5
    ) / tex_dims;
    if (PERIODIC_DOMAIN) {
        return textureSampleLevel(light_tau, vol_wrap_samp, tex_coord, 0.0).r;
    }
    return textureSampleLevel(light_tau, vol_samp, tex_coord, 0.0).r;
}
// ---------------------------------------------------------------------------

// Sample a *chosen* level at an already-wrapped point.
fn sample_sigma_pinned(q: vec3<f32>, in_nest: bool) -> f32 {
    if (NESTED && in_nest) {
        return sample_level(nest_vol, q, u.nest_bmin.xyz, u.nest_bmax.xyz,
                            false);
    }
    return sample_level(vol, q, u.bmin.xyz, u.bmax.xyz, PERIODIC_DOMAIN);
}

// Finest level covering p wins (witness._sample_sigma_nested).
fn sample_sigma(p: vec3<f32>) -> f32 {
    let q = wrap_to_domain(p);
    return sample_sigma_pinned(q, in_nest_box(q));
}

// Same dispatch, carrying the active level's step sizes back to the caller.
fn sample_level_at(p: vec3<f32>) -> LevelSample {
    let q = wrap_to_domain(p);
    let nested_here = in_nest_box(q);
    var s: LevelSample;
    s.in_nest = nested_here;
    s.sigma = sample_sigma_pinned(q, nested_here);
    if (NESTED && nested_here) {
        s.dt_view = u.nest_bmin.w;
        s.dt_light = u.nest_bmax.w;
    } else {
        s.dt_view = u.bmin.w;
        s.dt_light = u.bmax.w;
    }
    return s;
}

// One gradient tap. `pin` keeps the tap on the caller's level instead of
// re-dispatching — used for the one-voxel fine stencil, where crossing into
// the coarse field mid-stencil would measure the resolution change rather
// than the cloud. The coarse stencil deliberately does NOT pin: its radius
// routinely leaves the nest, and reading the parent field there is both
// cheaper and more honest than reading the nest's ghost-zero taper (witness
// pins at every radius, which puts a spurious edge on the nest boundary).
fn sigma_tap(p: vec3<f32>, pin: bool, pin_nest: bool) -> f32 {
    if (NESTED && pin) {
        return sample_sigma_pinned(wrap_to_domain(p), pin_nest);
    }
    return sample_sigma(p);
}

fn sigma_gradient_at_radius(p: vec3<f32>, h: vec3<f32>,
                            pin: bool, pin_nest: bool) -> vec3<f32> {
    let sxp = sigma_tap(p + vec3<f32>(h.x, 0.0, 0.0), pin, pin_nest);
    let sxm = sigma_tap(p - vec3<f32>(h.x, 0.0, 0.0), pin, pin_nest);
    let syp = sigma_tap(p + vec3<f32>(0.0, h.y, 0.0), pin, pin_nest);
    let sym = sigma_tap(p - vec3<f32>(0.0, h.y, 0.0), pin, pin_nest);
    let szp = sigma_tap(p + vec3<f32>(0.0, 0.0, h.z), pin, pin_nest);
    let szm = sigma_tap(p - vec3<f32>(0.0, 0.0, h.z), pin, pin_nest);
    return vec3<f32>(
        (sxp - sxm) / (2.0 * h.x),
        (syp - sym) / (2.0 * h.y),
        (szp - szm) / (2.0 * h.z)
    );
}

// One component of the coarse gradient, taken from the fine gradient where the
// two stencils' radii coincide on that axis. The cone stencil's radius is
// floored at a voxel, so for every sample nearer than a voxel / tan(theta) —
// ~29 voxels at the shipped 2 degrees — coarse_h equals fine_h on that axis,
// and the coarse pair of taps IS the fine pair: the same two points, read from
// the same level, differenced over the same radius. The fine component is
// therefore not an approximation of this one but the same number, and taking
// it saves two of the twelve texture reads per cloud sample (on a flat grid
// like DYCOMS, where dz is half of dx, that is the x and y pairs for nearly
// every visible sample).
//
// "The same level" is the one thing to check. The fine taps are pinned to the
// sample's level and the coarse ones are not, so under NESTED the fine
// component only stands in if neither tap lands across the nest boundary.
fn coarse_gradient_component(p: vec3<f32>, offset: vec3<f32>, h: f32,
                             same_radius: bool, fine_component: f32,
                             in_nest: bool) -> f32 {
    var borrow = same_radius;
    if (NESTED && borrow) {
        borrow = in_nest_box(wrap_to_domain(p + offset)) == in_nest
              && in_nest_box(wrap_to_domain(p - offset)) == in_nest;
    }
    if (borrow) {
        return fine_component;
    }
    let sp = sigma_tap(p + offset, false, in_nest);
    let sm = sigma_tap(p - offset, false, in_nest);
    return (sp - sm) / (2.0 * h);
}

fn sigma_gradient(p: vec3<f32>, sigma: f32,
                  coarse_weight_in: f32,
                  coarse_radius_m: f32,
                  sample_distance_m: f32,
                  cone_stencil_tan_theta: f32,
                  in_nest: bool) -> vec4<f32> {
    // One voxel-size lookup for both stencils: the coarse branch below asked
    // level_voxel_size(in_nest) for the same answer a second time.
    let voxel = level_voxel_size(in_nest);
    let fine_h = voxel * GRADIENT_SHADING_RADIUS_VOXELS;
    let fine_grad = sigma_gradient_at_radius(p, fine_h, true, in_nest);
    let fine_len = length(fine_grad);
    let fine_conf = (
        fine_len * min(min(fine_h.x, fine_h.y), fine_h.z)
    ) / (sigma + 1e-4);

    let coarse_weight = clamp(coarse_weight_in, 0.0, 1.0);
    if (coarse_weight <= 0.0) {
        return vec4<f32>(fine_grad, fine_conf);
    }

    let extent = u.bmax.xyz - u.bmin.xyz;

    // Cone stencil (witness iter_001): a fixed angular radius follows
    // apparent cloud scale — distant samples use a broader world-space
    // normal while nearby samples converge on the fine stencil. An exact
    // zero explicitly selects the legacy fixed-radius path, which keeps
    // its larger minimum-voxel floor and the domain-fraction ceiling.
    var cone_radius_m: f32;
    var coarse_min_voxels: f32;
    if (cone_stencil_tan_theta > 0.0) {
        cone_radius_m = sample_distance_m * cone_stencil_tan_theta;
        coarse_min_voxels = GRADIENT_SHADING_RADIUS_VOXELS;
    } else {
        cone_radius_m = coarse_radius_m;
        coarse_min_voxels = GRADIENT_SHADING_COARSE_MIN_VOXELS;
    }
    var coarse_h = max(vec3<f32>(cone_radius_m), voxel * coarse_min_voxels);
    if (cone_stencil_tan_theta == 0.0) {
        coarse_h = min(
            coarse_h,
            extent * GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION
        );
    }
    coarse_h = max(coarse_h, fine_h);
    var coarse_grad: vec3<f32>;
    if (any(coarse_h == fine_h)) {
        coarse_grad = vec3<f32>(
            coarse_gradient_component(p, vec3<f32>(coarse_h.x, 0.0, 0.0),
                                      coarse_h.x, coarse_h.x == fine_h.x,
                                      fine_grad.x, in_nest),
            coarse_gradient_component(p, vec3<f32>(0.0, coarse_h.y, 0.0),
                                      coarse_h.y, coarse_h.y == fine_h.y,
                                      fine_grad.y, in_nest),
            coarse_gradient_component(p, vec3<f32>(0.0, 0.0, coarse_h.z),
                                      coarse_h.z, coarse_h.z == fine_h.z,
                                      fine_grad.z, in_nest)
        );
    } else {
        coarse_grad = sigma_gradient_at_radius(p, coarse_h, false, in_nest);
    }
    let coarse_len = length(coarse_grad);
    let coarse_conf = (
        coarse_len * min(min(coarse_h.x, coarse_h.y), coarse_h.z)
    ) / (sigma + 1e-4);

    if (coarse_weight >= 1.0) {
        return vec4<f32>(coarse_grad, coarse_conf);
    }

    let fine_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, fine_conf
    );
    let coarse_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, coarse_conf
    );
    let fine_w = (1.0 - coarse_weight) * fine_gate;
    let coarse_w = coarse_weight * coarse_gate;

    var blended = vec3<f32>(0.0);
    if (fine_len > 1e-12) {
        blended = blended + fine_w * fine_grad / fine_len;
    }
    if (coarse_len > 1e-12) {
        blended = blended + coarse_w * coarse_grad / coarse_len;
    }
    return vec4<f32>(blended, max(fine_conf, coarse_conf));
}
