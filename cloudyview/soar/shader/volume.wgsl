// The volume: the ray/box intersection, reading extinction from the field's
// one or two levels by hardware trilinear filtering, reading the baked sun
// optical depth, and the density gradient that the cloud shading uses as a
// surface normal.
//
// The density texture is r32float or filterable r16float. Either reads as
// texture_3d<f32>; half floats change only storage and bandwidth.
//
// What sets the march's cost, for anyone changing it:
// - It is bound by texture latency and by the size of the compiled fragment
//   program, not by the number of fetches. fs_main compiles as one kernel,
//   so code added to it costs every pixel whether or not its branch runs,
//   and a change has to be timed on scenes it does not touch as well.
// - Cloud fields are mostly empty, but skipping the empty samples does not
//   pay: the per-step bookkeeping that finds out a sample is empty costs
//   more than the fetch it saves.
// - The sparseness is used at load instead, where the empty sky is cropped
//   off the field (web/soar/zcrop.js) and nothing is asked per sample.

// Slab-method intersection of a ray with the outer box. Returns
// (t_near, t_far); the ray misses when t_near > t_far.
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
// infinity, and WGSL leaves an expression that produces one indeterminate:
// an implementation may assume it never happens, so the slab test's result
// would depend on the compiler. A component of 1e30 does what the infinity
// was meant to: an origin inside that axis's slab is not constrained by it,
// and an origin outside it misses.
fn safe_inv_dir(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(select(1e30, 1.0 / d.x, abs(d.x) > 1e-9),
                     select(1e30, 1.0 / d.y, abs(d.y) > 1e-9),
                     select(1e30, 1.0 / d.z, abs(d.z) > 1e-9));
}

// One level's sample, with the march step sizes that level asks for.
struct LevelSample {
    sigma: f32,
    dt_view: f32,
    dt_light: f32,
    in_nest: bool,
}

// A level's grid size in field axis order (nx, ny, nz). The texture is
// stored transposed, (width = nz, height = ny, depth = nx), and holds the
// field alone, with no border texels.
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

// Fold a world point into the outer tile. The whole scene, nest included, is
// one periodic tile, so this runs before any level test: every copy of the
// domain then carries a copy of the nest at the same place within it.
fn wrap_to_domain(p: vec3<f32>) -> vec3<f32> {
    if (!PERIODIC_DOMAIN) {
        return p;
    }
    let extent = u.bmax.xy - u.bmin.xy;
    let g = fract((p.xy - u.bmin.xy) / extent);
    return vec3<f32>(u.bmin.xy + g * extent, p.z);
}

// Whether an already-wrapped point lies in the nest's box.
fn in_nest_box(q: vec3<f32>) -> bool {
    if (!NESTED) {
        return false;
    }
    return all(q >= u.nest_bmin.xyz) && all(q <= u.nest_bmax.xyz);
}

// The window that fades a level to zero across the half cell outside its
// outermost cell centres.
//
// g is the data coordinate along one axis of N cells (cell centre i at
// g = i). Clamp-to-edge sampling returns the outermost value, v_0 for g < 0
// and v_{N-1} for g > N - 1. Times this window that is v_0 (g + 1) on
// [-1, 0] and v_{N-1} (N - g) on [N - 1, N]: a linear ramp from the
// outermost value down to zero, which is what trilinear filtering against a
// ring of zero texels would return. Trilinear filtering is separable, so the
// windows of the three axes multiply and the identity holds at edges and
// corners too.
//
// Computing the ramp rather than storing the ring keeps the texture at the
// field's own size. A ring would add two texels per axis, and a 2048-cell
// axis would then exceed the 3D texture size WebGPU guarantees (2048).
fn edge_taper(g: f32, n: f32) -> f32 {
    return clamp(min(g + 1.0, n - g), 0.0, 1.0);
}

// Extinction (m^-1) from one level, by hardware trilinear filtering.
//
// The box extends half a cell beyond the outermost cell centres, so cell
// centre i sits at box fraction (i + 0.5) / N. data_g = fraction * N - 0.5
// is the data coordinate (cell i at data_g = i), and the normalized texture
// coordinate (data_g + 0.5) / N is the box fraction itself. The swizzle is
// the texture's (width = nz, height = ny, depth = nx) transposition.
//
// `periodic_level` picks how the lateral edges behave. A doubly periodic
// outer level wraps: the repeat sampler takes the far face as the trilinear
// partner across the seam, corners included. Every nest, and a non-periodic
// outer level, tapers to zero at its own boundary (edge_taper), which is
// also how a nest blends into the coarse field around it. The vertical is
// never periodic and always tapers.
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

// Sun optical depth from the baked cache: the read that stands in for
// light_march_tau on cloud samples under LIGHT_CACHE. The cached value is
// for the sun's central direction with no quadrature jitter, so it carries
// no solar-disc penumbra; trilinear filtering is its only softening.
//
// Coordinates are sample_level's over the same outer box (texel i at data
// coordinate i; see fs_bake_light), with the same lateral wrap in a periodic
// domain, but there is no edge taper: tau is not extinction, and ramping it
// to zero at the domain edge would let sunlight into the boundary column.
// Clamp-to-edge is right on the vertical: above the top slice tau is ~0,
// and below the bottom slice the nearest baked value is the best available.
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

// Extinction from a chosen level at an already-wrapped point.
fn sample_sigma_pinned(q: vec3<f32>, in_nest: bool) -> f32 {
    if (NESTED && in_nest) {
        return sample_level(nest_vol, q, u.nest_bmin.xyz, u.nest_bmax.xyz,
                            false);
    }
    return sample_level(vol, q, u.bmin.xyz, u.bmax.xyz, PERIODIC_DOMAIN);
}

// Extinction at a world point: the finest level that covers it.
fn sample_sigma(p: vec3<f32>) -> f32 {
    let q = wrap_to_domain(p);
    return sample_sigma_pinned(q, in_nest_box(q));
}

// The same, with the active level's step sizes.
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

// One tap of a gradient stencil. `pin` keeps the tap on the level `pin_nest`
// names instead of choosing the level at the tap. The one-voxel fine stencil
// pins: crossing into the coarse field mid-stencil would difference the
// change of resolution rather than the cloud. The coarse stencil does not:
// its radius routinely leaves the nest, and there the parent field is a
// better reading than the nest's taper to zero.
fn sigma_tap(p: vec3<f32>, pin: bool, pin_nest: bool) -> f32 {
    if (NESTED && pin) {
        return sample_sigma_pinned(wrap_to_domain(p), pin_nest);
    }
    return sample_sigma(p);
}

// Central-difference gradient of sigma (m^-2), half-width h (m) per axis.
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

// One component of the coarse gradient, borrowed from the fine gradient
// where the two stencils have the same radius on that axis. The coarse
// radius is floored at the fine one, so for a near sample (under the cone
// stencil, one closer than a voxel / tan(theta)) the coarse pair of taps is
// the fine pair: the same two points, differenced over the same radius.
// Borrowing saves two texture reads per axis.
//
// The fine taps are pinned to the sample's level and the coarse ones are
// not, so under NESTED the fine component stands in only if neither tap
// lands across the nest boundary.
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

// The density gradient at a cloud sample, which the shading uses as the
// local surface normal (outward normal = -grad / |grad|). Returns
// xyz = the gradient and w = a confidence, |grad| h / sigma: the relative
// change of sigma across the stencil radius h, near 0 in a uniform interior
// and of order 1 at an edge.
//
// Two stencils are blended by coarse_weight_in. The fine one, of radius
// GRADIENT_SHADING_RADIUS_VOXELS, resolves the voxel-scale skin. The coarse
// one gives the shape of the billow. For a weight strictly between 0 and 1
// the two unit normals are summed with weights (1 - w) and w, each times a
// smoothstep of its own confidence, so xyz is then a direction and not a
// gradient in m^-2. The callers use only its direction.
fn sigma_gradient(p: vec3<f32>, sigma: f32,
                  coarse_weight_in: f32,
                  coarse_radius_m: f32,
                  sample_distance_m: f32,
                  cone_stencil_tan_theta: f32,
                  in_nest: bool) -> vec4<f32> {
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

    // The coarse radius. Under the cone stencil it is
    // sample_distance_m * tan(theta): a fixed angle, so the normal is taken
    // at the scale the pixel sees, and near samples converge on the fine
    // stencil. A tangent of exactly 0 selects the fixed radius
    // coarse_radius_m, floored at GRADIENT_SHADING_COARSE_MIN_VOXELS voxels
    // and capped at GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION of the box.
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
