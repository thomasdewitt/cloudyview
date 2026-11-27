"""Cheaper density-gradient stencils for soar's march, as soar_menu variants.

`sigma_gradient` (volume.wgsl) gives the shading its surface normal from two
stencils: a fine one of one voxel, which resolves the skin, and a coarse one
of radius distance x tan 2 deg (floored at the fine radius), which gives the
billow its shape. Each is six taps of central differences; where the two
radii agree on an axis the coarse component borrows the fine one. A far
sample therefore reads 12 taps. The blend, the confidences and the return
value are left exactly as stock in every variant here; only how each
stencil's gradient is estimated changes.

The confidence needs no taps of its own: it is |gradient| x (smallest
radius) / (sigma + 1e-4), from the stencil's gradient and the sample's own
sigma, which the march passes in. A variant's confidence is therefore that
same expression over its own gradient.

Variants (taps per far sample in brackets):

  tetr_both     both scales by a tetrahedral stencil of the stencil's
                radius (8): four taps at p + (h / sqrt 3) * k,
                k = (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1), each h from the
                centre. With a = h / sqrt 3, g_b = sum_i k_ib s_i / (4 a_b).
                Exact for a linear field; its error is first order in h,
                from the mixed second derivatives (g_x picks up a H_yz / 2),
                where central differences are second order.
  tet_both      the same with a = h, so each axis's difference spans 2h as
                the central stencil's does (the taps sqrt 3 h out). Defined,
                not in the default set (see GRADIENT_CANDIDATES).
  fwd_both      both scales by forward differences from the sample the march
                already holds (6): g_a = (s(p + h_a e_a) - sigma) / h_a. The
                same first-order error, from the diagonal second derivatives
                (g_a picks up h_a H_aa / 2): the gradient is in effect taken
                half a radius forward along each axis.
  tetr_fine     tetrahedral fine, central coarse (10)
  tetr_coarse   central fine, tetrahedral coarse (10)
  fwd_coarse    central fine, forward coarse (9)
  tetr_alt      tetr_both with the tetrahedron inverted (k -> -k) on
                alternate pixels in a checkerboard that swaps every frame.
                The two tetrahedra together are the cube, whose stencil is
                second order, so the first-order error changes sign from
                one frame to the next and from one pixel to its neighbour,
                and the accumulation averages it out (8)
  fwd_alt       fwd_both with forward and backward differences alternating
                the same way: together they are central differences (6)
  reuse50       stock's fine stencil at every sample; the coarse gradient
                kept from an earlier sample of the same ray while the march
                has moved less than half the coarse radius since it was
                taken (6 + 6 per refresh). The coarse radius is 2 deg of
                the distance and the view step 0.3-1.8 deg of it, so the
                refresh comes every ~3 steps at High and every step at Low.
  reuse100      the same within one whole coarse radius.

The alternating variants' per-frame picture carries the whole first-order
error; only the accumulation removes it. soar_menu's single-frame column
measures that frame.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soar_variants as sv                                    # noqa: E402

# The set soar_menu --set gradient runs. tet_both, tet_fine, tet_coarse and
# tet_alt (the tetrahedron at the per-axis radius) are defined too: a first
# look at one view of rce put them further from stock than the tetrahedron
# at the stencil's radius, at the same cost, so they are left out of it.
GRADIENT_CANDIDATES = ("tetr_both", "tetr_fine", "tetr_coarse", "tetr_alt",
                       "fwd_both", "fwd_coarse", "fwd_alt", "reuse50",
                       "reuse100")

HEAD = ("fn sigma_gradient(p: vec3<f32>, sigma: f32, sample_distance_m: f32,\n"
        "                  in_nest: bool) -> vec4<f32> {\n")
TAIL = "    return vec4<f32>(blended, max(fine_conf, coarse_conf));\n}\n"

HELPERS = """\
// stencil variant: the sign that flips the alternating stencils, set once
// per fragment (a checkerboard that swaps every frame).
var<private> grad_flip: f32 = 1.0;
const INV_SQRT3: f32 = 0.5773502691896258;

// stencil variant: a tetrahedral gradient. Taps at p + flip * h * k for
// k = (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1); g_b = sum k_b s / (4 h_b).
fn sigma_gradient_tetra(p: vec3<f32>, h: vec3<f32>, pin: bool,
                        pin_nest: bool, flip: f32) -> vec3<f32> {
    let o = flip * h;
    let a = sigma_tap(p + o * vec3<f32>(1.0, 1.0, 1.0), pin, pin_nest);
    let b = sigma_tap(p + o * vec3<f32>(1.0, -1.0, -1.0), pin, pin_nest);
    let c = sigma_tap(p + o * vec3<f32>(-1.0, 1.0, -1.0), pin, pin_nest);
    let d = sigma_tap(p + o * vec3<f32>(-1.0, -1.0, 1.0), pin, pin_nest);
    return vec3<f32>(a + b - c - d, a - b + c - d, a - b - c + d) / (4.0 * o);
}

// stencil variant: forward (flip = 1) or backward (flip = -1) differences
// from the sample's own sigma.
fn sigma_gradient_forward(p: vec3<f32>, sigma: f32, h: vec3<f32>, pin: bool,
                          pin_nest: bool, flip: f32) -> vec3<f32> {
    let o = flip * h;
    let sx = sigma_tap(p + vec3<f32>(o.x, 0.0, 0.0), pin, pin_nest);
    let sy = sigma_tap(p + vec3<f32>(0.0, o.y, 0.0), pin, pin_nest);
    let sz = sigma_tap(p + vec3<f32>(0.0, 0.0, o.z), pin, pin_nest);
    return (vec3<f32>(sx, sy, sz) - sigma) / o;
}

// stencil variant: one forward-difference component of the coarse gradient,
// borrowed from the fine one where the radii agree (as
// coarse_gradient_component does for central differences).
fn coarse_forward_component(p: vec3<f32>, sigma: f32, offset: vec3<f32>,
                            h: f32, same_radius: bool, fine_component: f32,
                            in_nest: bool, flip: f32) -> f32 {
    var borrow = same_radius;
    if (NESTED && borrow) {
        borrow = in_nest_box(wrap_to_domain(p + flip * offset)) == in_nest;
    }
    if (borrow) {
        return fine_component;
    }
    return (sigma_tap(p + flip * offset, false, in_nest) - sigma) / (flip * h);
}

"""

PREAMBLE = """\
    let voxel = level_voxel_size(in_nest);
    let fine_h = voxel * GRADIENT_SHADING_RADIUS_VOXELS;
    let flip = FLIP;
    let fine_grad = FINE;
    let fine_len = length(fine_grad);
    let fine_conf = (
        fine_len * min(min(fine_h.x, fine_h.y), fine_h.z)
    ) / (sigma + 1e-4);
    var coarse_h = max(vec3<f32>(sample_distance_m * CONE_STENCIL_TAN),
                       voxel * GRADIENT_SHADING_RADIUS_VOXELS);
    coarse_h = max(coarse_h, fine_h);
    var coarse_grad: vec3<f32>;
"""

FINE = {
    "central": "sigma_gradient_at_radius(p, fine_h, true, in_nest)",
    "tetra": "sigma_gradient_tetra(p, fine_h, true, in_nest, flip)",
    "tetra_r": "sigma_gradient_tetra(p, fine_h * INV_SQRT3, true, in_nest, flip)",
    "forward": "sigma_gradient_forward(p, sigma, fine_h, true, in_nest, flip)",
}

COARSE = {
    "central": """\
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
""",
    "tetra": """\
    if (all(coarse_h == fine_h)) {
        coarse_grad = fine_grad;
    } else {
        coarse_grad = sigma_gradient_tetra(p, coarse_h, false, in_nest, flip);
    }
""",
    "tetra_r": """\
    if (all(coarse_h == fine_h)) {
        coarse_grad = fine_grad;
    } else {
        coarse_grad = sigma_gradient_tetra(p, coarse_h * INV_SQRT3, false,
                                           in_nest, flip);
    }
""",
    "forward": """\
    coarse_grad = vec3<f32>(
        coarse_forward_component(p, sigma, vec3<f32>(coarse_h.x, 0.0, 0.0),
                                 coarse_h.x, coarse_h.x == fine_h.x,
                                 fine_grad.x, in_nest, flip),
        coarse_forward_component(p, sigma, vec3<f32>(0.0, coarse_h.y, 0.0),
                                 coarse_h.y, coarse_h.y == fine_h.y,
                                 fine_grad.y, in_nest, flip),
        coarse_forward_component(p, sigma, vec3<f32>(0.0, 0.0, coarse_h.z),
                                 coarse_h.z, coarse_h.z == fine_h.z,
                                 fine_grad.z, in_nest, flip)
    );
""",
}

# The coarse stencil kept along the ray: `cache` holds the coarse gradient,
# `cache_at` the point it was taken at (w = 1 once filled).
REUSE_COARSE = """\
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
        (*cache_at).w = 0.0;
    } else if ((*cache_at).w > 0.5
               && distance(p, (*cache_at).xyz)
                  <= REUSE * min(min(coarse_h.x, coarse_h.y), coarse_h.z)) {
        coarse_grad = (*cache).xyz;
    } else {
        coarse_grad = sigma_gradient_at_radius(p, coarse_h, false, in_nest);
        *cache = vec4<f32>(coarse_grad, 0.0);
        *cache_at = vec4<f32>(p, 1.0);
    }
"""

# The rest of sigma_gradient, as stock has it.
BLEND = """\
    let coarse_len = length(coarse_grad);
    let coarse_conf = (
        coarse_len * min(min(coarse_h.x, coarse_h.y), coarse_h.z)
    ) / (sigma + 1e-4);

    let fine_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, fine_conf
    );
    let coarse_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, coarse_conf
    );
    let fine_w = (1.0 - GRADIENT_SHADING_COARSE_WEIGHT) * fine_gate;
    let coarse_w = GRADIENT_SHADING_COARSE_WEIGHT * coarse_gate;

    var blended = vec3<f32>(0.0);
    if (fine_len > 1e-12) {
        blended = blended + fine_w * fine_grad / fine_len;
    }
    if (coarse_len > 1e-12) {
        blended = blended + coarse_w * coarse_grad / coarse_len;
    }
    return vec4<f32>(blended, max(fine_conf, coarse_conf));
}
"""

SPECS = {           # name -> (fine, coarse, alternating)
    "tet_both": ("tetra", "tetra", False),
    "fwd_both": ("forward", "forward", False),
    "tet_fine": ("tetra", "central", False),
    "tet_coarse": ("central", "tetra", False),
    "fwd_coarse": ("central", "forward", False),
    "tet_alt": ("tetra", "tetra", True),
    "tetr_both": ("tetra_r", "tetra_r", False),
    "tetr_fine": ("tetra_r", "central", False),
    "tetr_coarse": ("central", "tetra_r", False),
    "tetr_alt": ("tetra_r", "tetra_r", True),
    "fwd_alt": ("forward", "forward", True),
}
REUSE = {"reuse50": 0.5, "reuse100": 1.0}
ALL_GRADIENT_VARIANTS = tuple(SPECS) + tuple(REUSE)

FLIP_SET = """\
    // stencil variant: the alternating stencils' sign for this fragment.
    grad_flip = select(1.0, -1.0, ((u32(frag_pos.x) + u32(frag_pos.y)
                                    + u32(max(u.frame_index, 0.0))) & 1u) == 1u);
"""


def _replace_function(src: str, new_head_and_body: str) -> str:
    if src.count(HEAD) != 1:
        raise RuntimeError("sigma_gradient's signature has moved; update HEAD")
    i = src.index(HEAD)
    j = src.find(TAIL, i)
    if j < 0 or src.count(TAIL) != 1:
        raise RuntimeError("sigma_gradient's return has moved; update TAIL")
    return src[:i] + new_head_and_body + src[j + len(TAIL):]


def gradient_candidate(src: str, name: str) -> str:
    """The stock module `src` with sigma_gradient replaced by variant `name`."""
    if name in SPECS:
        fine, coarse, alt = SPECS[name]
        body = (PREAMBLE.replace("FLIP", "grad_flip" if alt else "1.0")
                .replace("FINE", FINE[fine]) + COARSE[coarse] + BLEND)
        src = _replace_function(src, HELPERS + HEAD + body)
        if alt:
            src = sv.patch(src, "fn fs_main(@builtin(position) frag_pos: vec4<f32>) "
                                "-> MarchOut {\n",
                           "fn fs_main(@builtin(position) frag_pos: vec4<f32>) "
                           "-> MarchOut {\n" + FLIP_SET, what=name + " flip")
        return src
    if name in REUSE:
        head = HEAD.replace(
            "in_nest: bool) -> vec4<f32> {\n",
            "in_nest: bool, cache: ptr<function, vec4<f32>>,\n"
            "                  cache_at: ptr<function, vec4<f32>>) -> vec4<f32> {\n")
        body = (PREAMBLE.replace("FLIP", "1.0").replace("FINE", FINE["central"])
                + REUSE_COARSE.replace("REUSE", f"{REUSE[name]:.2f}") + BLEND)
        src = _replace_function(src, HELPERS + head + body)
        src = sv.patch(src, "    var was_in_nest = false;\n",
                       "    var was_in_nest = false;\n"
                       "    // stencil variant: the coarse gradient kept along the ray.\n"
                       "    var coarse_cache = vec4<f32>(0.0);\n"
                       "    var coarse_cache_at = vec4<f32>(0.0);\n", what=name + " state")
        return sv.patch(src, "sigma_gradient(p, sigma, t, level.in_nest);",
                        "sigma_gradient(p, sigma, t, level.in_nest,\n"
                        "                                               "
                        "&coarse_cache, &coarse_cache_at);", what=name + " call")
    raise KeyError(name)
