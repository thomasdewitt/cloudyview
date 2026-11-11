// skybridges — the enclosed walkways that make two towers one address.
//
// THE PROBLEM. Every other piece of this city fits inside one block column,
// which is what lets the DDA test a cell's geometry only while the ray is
// inside that cell. A bridge does not: it spans the street, so half of it
// lives in one column and half in the next. The fix is not to make the
// tracer smarter. It is to make the bridge belong to the EDGE between two
// cells rather than to either cell, and to derive it from data both cells
// can read — so cell A and cell B independently construct the SAME box, to
// the bit. Each then reports whichever part of that box its own ray segment
// contains, and the two halves meet with nothing between them.
//
// The edge id is the whole trick. For the edge between (i,j) and (i+1,j)
// the id is (2i+1, 2j); between (i,j) and (i,j+1) it is (2i, 2j+1). Doubling
// the cell index leaves the even lattice for cells and puts each edge on the
// odd coordinate of the axis it crosses, so the id is a property of the edge
// and is computed identically from either side. A cell tests all four of its
// edges; the neighbour tests the same four ids from its side; the two agree
// by construction rather than by convention.
//
// The other half of the trick is CONTAINMENT. Determinism alone is not
// enough — the box must also lie inside the two columns that test it, or the
// ray would enter it while the DDA is in some third cell that never looks.
// Two rules enforce it: each end face must be within half a block of the
// shared boundary, and the deck's cross-extent is clipped to the edge's own
// row. Everything a bridge needs is then a function of (edge id, cell A,
// cell B), which is also why the shading hook can rebuild the exact span
// from a hit position alone: round the hit to the nearest block boundary on
// the bridge's axis and the two cells fall out.
//
// WHERE THEY GO. Only downtown (both blocks past the density gate, and it is
// the LOWER of the two that decides so both sides agree), only between two
// built buildings, and only where the deck fits under BOTH roofs with 15 m to
// spare. "Under the roof" means under the b1 box — the widest, lowest prim of
// every archetype. That one rule buys the whole architectural constraint the
// city asks for: a bridge lands on a slab or a growth tower's shaft, on a
// tapered shaft only below its podium cap, on a spire tower only below the
// spring line of the spire, and on nothing at all above a setback.
//
// WHAT THEY LOOK LIKE. A 0.5 m deck, a 2.6 m glass band, a thin roof: from a
// kilometre out a lit ribbon strung between two dark towers, which is the
// Cloudpunk image this component exists for. Close up the band resolves into
// mullions every 2.5 m, a scatter of dark bays, and now and then somebody
// walking across at two in the morning.

// kind = BASE + axis + 2 * type, type 0 shell, 1 glass band, 2 truss member.
// The axis rides in the low bit so the type is a plain shift, and both fit
// the 16 local kinds a component owns.
const cc_skybridges_KIND_BASE: i32 = 500;

// --- eligibility ----------------------------------------------------------
// The gate is the first thing evaluated per edge and costs one pcg2d. What
// survives it pays for one neighbour cell fetch; what survives THAT pays for
// three box tests. Measured over a 256x256-cell window on the megatower
// district (tools' probe, 41967 qualifying edges): 59% of gated edges end up
// carrying a bridge, the rest losing on fit — frontage, gap or headroom — so
// the gate is set to 0.118 to land the built share at 7.0% of the edges whose
// two blocks are both built and both past the density floor.
const cc_skybridges_GATE: f32 = 0.118;
const cc_skybridges_MIN_DENSITY: f32 = 0.25;
const cc_skybridges_GAP_MIN: f32 = 8.0;   // a street, not a seam
const cc_skybridges_GAP_MAX: f32 = 62.0;  // beyond this it is a viaduct
const cc_skybridges_Z_MIN: f32 = 25.0;
const cc_skybridges_Z_MAX: f32 = 380.0;   // above this a walkway is a stunt
const cc_skybridges_ROOF_MARGIN: f32 = 15.0;

// --- geometry (metres) ----------------------------------------------------
const cc_skybridges_DECK_T: f32 = 0.5;
const cc_skybridges_GLASS_H: f32 = 2.6;
const cc_skybridges_ROOF_T: f32 = 0.4;
const cc_skybridges_TUBE_H: f32 = 3.5;    // DECK_T + GLASS_H + ROOF_T
const cc_skybridges_HALF_W: f32 = 2.0;    // a 4 m tube: two people wide
const cc_skybridges_HALF_W_GRAND: f32 = 4.0;  // a sky lobby, 8 m across
const cc_skybridges_GRAND_FRAC: f32 = 0.22;
const cc_skybridges_GLASS_INSET: f32 = 0.12;  // deck lip proud of the glass
const cc_skybridges_ROOF_OVER: f32 = 0.18;    // fascia proud of the glass
const cc_skybridges_EMBED: f32 = 1.0;     // driven into each wall, so no end
                                          // can ever float in the gap
const cc_skybridges_ROW_MARGIN: f32 = 1.0;

// The hook's own z gate: the extreme deck is Z_MAX quantized up plus the
// tube, and the lowest is Z_MIN rounded down a storey. Nothing outside this
// band exists, so a segment that misses it costs one compare.
const cc_skybridges_Z_LO: f32 = 20.0;
const cc_skybridges_Z_HI: f32 = 395.0;

// --- the shell ------------------------------------------------------------
// Everything albedo-lit is near-black at night, so the deck and roof cannot
// be made to read by brightening them — only by structure. The underside is
// the surface that matters: from the street a bridge is a dark belly with a
// lamp string on it, and nothing else. Ribs sit on the glazing's own 2.5 m
// pitch, because one tube built once has one rhythm.
const cc_skybridges_SHELL_TINT: f32 = 0.012;  // corridor leak through the shell
const cc_skybridges_SHELL_FILL: f32 = 0.055;
const cc_skybridges_RIB_PITCH: f32 = 2.5;
const cc_skybridges_RIB_HALF: f32 = 0.16;   // a 0.32 m transverse web
const cc_skybridges_SPINE_HALF: f32 = 0.32; // the box girder down the belly
const cc_skybridges_LAMP_PITCH: f32 = 5.0;
const cc_skybridges_LAMP_R: f32 = 0.18;
const cc_skybridges_LAMP_RAD: f32 = 0.85;

// --- the open truss -------------------------------------------------------
// Not every span between two towers is a corridor: some are service crossings
// that never had a wall. The variant keeps the deck and the same three box
// tests, but the middle prim shrinks to a central web and the top prim to a
// chord, so the silhouette is genuinely open — sky on both sides above the
// handrail — and what carries it at night is a lamp string rather than a lit
// room. Rare, because the enclosed walkway is the thing this city is for.
const cc_skybridges_TRUSS_FRAC: f32 = 0.18;
const cc_skybridges_TRUSS_HALF: f32 = 0.30;   // the central web
const cc_skybridges_CHORD_FRAC: f32 = 0.55;   // top chord, as a share of half
const cc_skybridges_STRING_PITCH: f32 = 4.0;
const cc_skybridges_STRING_R: f32 = 0.16;
const cc_skybridges_STRING_RAD: f32 = 2.6;
const cc_skybridges_TRUSS_PITCH: f32 = 3.0;   // one Warren bay
const cc_skybridges_MEMBER_T: f32 = 0.20;     // diagonal, measured vertically
const cc_skybridges_POST_T: f32 = 0.10;
const cc_skybridges_CHORD_T: f32 = 0.17;
// Mean member cover of the web face, integrated off the constants above:
// two chords (2*0.17/2.6), the posts (2*0.10/3.0) and the diagonals
// (2*0.20 of vertical extent per 1.5 m of run, over 2.6 m), less overlap.
const cc_skybridges_TRUSS_COVER: f32 = 0.40;

// --- the band -------------------------------------------------------------
const cc_skybridges_MULLION_PITCH: f32 = 2.5;
const cc_skybridges_MULLION_HALF: f32 = 0.11;  // half the 0.22 m web
const cc_skybridges_GLASS_V0: f32 = 0.26;  // sill, above the deck surface
const cc_skybridges_GLASS_V1: f32 = 2.39;  // head
const cc_skybridges_RADIANCE: f32 = 2.0;
const cc_skybridges_DARK_SEG: f32 = 0.12;  // bays with the lights off
const cc_skybridges_FIG_FRAC: f32 = 0.15;  // bays with somebody in them
const cc_skybridges_BODY_T: f32 = 0.12;
const cc_skybridges_SPANDREL: f32 = 0.04;  // sill/head/mullion transmission

// Footprint windows (m/px) over which each layer blends into its own mean,
// each set by that layer's finest feature: a 0.1 m limb, a 0.22 m mullion
// web, a 2.5 m bay. Past its window a layer is a constant, so the next
// coarser one hands off smoothly and nothing vanishes.
const cc_skybridges_LOD_FIG: vec2<f32> = vec2<f32>(0.25, 1.10);
const cc_skybridges_LOD_MUL: vec2<f32> = vec2<f32>(0.60, 3.00);
const cc_skybridges_LOD_SEG: vec2<f32> = vec2<f32>(3.00, 10.0);
const cc_skybridges_LOD_RIB: vec2<f32> = vec2<f32>(0.50, 2.60);
const cc_skybridges_LOD_LAMP: vec2<f32> = vec2<f32>(0.30, 1.60);

// The band's mean cover, measured off the constants above rather than
// guessed: the glazed fraction of the box face (2.13 / 2.6), the mullion
// duty (1 - 0.22 / 2.5), the lit-bay fraction (1 - 0.12), and the mean
// transmission left by the figures (1 - 0.15 * 0.113 * 0.88, the 0.113 being
// a body's share of a glazed bay). The bay brightness draw has mean 1.
const cc_skybridges_MEAN_COVER: f32 = 0.647;
// Averaging radiance ahead of a compressive tone map runs bright; the same
// compensation the core's octave ladder carries, scaled for a ribbon that
// never fills more than a few pixels.
const cc_skybridges_MEAN_COMP: f32 = 0.80;

// One bridge's whole description. Both hooks build this and must agree.
struct cc_skybridges_Span {
    ok: bool,
    axis: i32,     // 0 the bridge runs along x, 1 along y
    lo: f32,       // start along the axis, inside wall A
    hi: f32,       // end along the axis, inside wall B
    ctr: f32,      // centre across the axis
    half: f32,     // half deck width
    z: f32,        // deck underside
    truss: bool,   // open service crossing rather than an enclosed walkway
    seed: vec2<u32>,
}

// The edge id: even coordinates are cells, the odd one names the axis the
// edge crosses. Computed from the LOWER cell of the pair, which both sides
// can name.
fn cc_skybridges_eid(clo: vec2<i32>, axis: i32) -> vec2<u32> {
    // Tile-wrapped (city_tile_cell) so an edge is a property of its tile
    // coordinate: both cells beside it name the same lower cell, wrapped
    // the same way, on every copy of the tile.
    let cw = city_tile_cell(clo);
    return vec2<u32>(
        bitcast<u32>(2 * cw.x + (1 - axis)) ^ 0x5bf03635u,
        bitcast<u32>(2 * cw.y + axis) ^ 0x9e3779b9u);
}

// The gate. This is the component's hottest instruction by a wide margin —
// four of them run in every cell the DDA visits inside CITY_PROP_RANGE,
// whether or not a bridge is anywhere near — and measuring said the per-cell
// hashing, not the geometry, was two thirds of what the component costs. So
// it is one multiply-xorshift round instead of pcg2d's two. It must be a pure
// function of the edge id, which is the entire determinism contract; it does
// NOT have to be a strong hash, because all it decides is a yes/no on an
// eighth of edges and everything that survives is redrawn with pcg2d.
// One multiply-xorshift round rather than pcg2d's two. Folding a cell's two
// owned edges into the halves of a single hash — four gates for three hashes
// — was tried and measured no faster on the one view whose bridge statistics
// are comparable across the change, so the per-edge hash stays: independent
// gates, and the simpler thing when the complicated thing cannot be shown to
// pay.
fn cc_skybridges_gate(clo: vec2<i32>, axis: i32) -> bool {
    var h = (bitcast<u32>(2 * clo.x + (1 - axis)) * 0x9e3779b9u)
          ^ (bitcast<u32>(2 * clo.y + axis) * 0x85ebca6bu);
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    return city_u01(h) < cc_skybridges_GATE;
}

fn cc_skybridges_bmin(axis: i32, a: f32, c: f32, z: f32) -> vec3<f32> {
    if (axis == 0) {
        return vec3<f32>(a, c, z);
    }
    return vec3<f32>(c, a, z);
}

// Everything about the bridge on one edge, from the two cells it joins and
// the edge's own draws. Deterministic in its arguments and nothing else,
// which is the entire contract this component rests on.
fn cc_skybridges_build(clo: vec2<i32>, axis: i32, ca: CityCell, cb: CityCell,
                       d: vec4<f32>) -> cc_skybridges_Span {
    var s: cc_skybridges_Span;
    s.ok = false;
    s.axis = axis;
    s.lo = 0.0; s.hi = 0.0; s.ctr = 0.0; s.half = 0.0; s.z = 0.0;
    s.truss = false;
    s.seed = vec2<u32>(0u);
    if (!ca.built || !cb.built) {
        return s;
    }
    // A merged superblock reports the same seed from every member cell: an
    // edge inside one building is not a bridge, it is a corridor.
    if (ca.seed.x == cb.seed.x && ca.seed.y == cb.seed.y) {
        return s;
    }
    if (min(ca.density, cb.density) <= cc_skybridges_MIN_DENSITY) {
        return s;
    }

    let cell = u.ocean_params.x;
    // Faces along the axis, and the row the edge lives in across it.
    var a_face: f32; var b_face: f32; var bnd: f32;
    var a_lo: f32; var a_hi: f32; var b_lo: f32; var b_hi: f32;
    var row: f32;
    if (axis == 0) {
        bnd = f32(clo.x + 1) * cell;
        a_face = ca.b1max.x; b_face = cb.b1min.x;
        a_lo = ca.b1min.y; a_hi = ca.b1max.y;
        b_lo = cb.b1min.y; b_hi = cb.b1max.y;
        row = f32(clo.y);
    } else {
        bnd = f32(clo.y + 1) * cell;
        a_face = ca.b1max.y; b_face = cb.b1min.y;
        a_lo = ca.b1min.x; a_hi = ca.b1max.x;
        b_lo = cb.b1min.x; b_hi = cb.b1max.x;
        row = f32(clo.x);
    }

    // Containment. Each wall must be within half a block of the boundary:
    // that keeps the whole tube inside the two columns that test it (so no
    // third cell can own the ray where the box begins) and it is also what
    // makes the shading hook's rounding recovery exact.
    let reach = 0.5 * cell - 2.0;
    if (abs(a_face - bnd) > reach || abs(b_face - bnd) > reach) {
        return s;
    }
    let gap = b_face - a_face;
    if (gap < cc_skybridges_GAP_MIN || gap > cc_skybridges_GAP_MAX) {
        return s;
    }

    // Across the axis: the two walls' shared frontage, clipped to the edge's
    // own row so the deck cannot wander into a neighbouring one.
    let c0 = max(max(a_lo, b_lo), row * cell + cc_skybridges_ROW_MARGIN);
    let c1 = min(min(a_hi, b_hi), (row + 1.0) * cell
                                  - cc_skybridges_ROW_MARGIN);
    let frontage = c1 - c0;
    // A service crossing is never also a sky lobby: the two draws are read in
    // order so the variant is decided before the width that depends on it.
    let truss = d.x < cc_skybridges_TRUSS_FRAC;
    var half = cc_skybridges_HALF_W;
    if (!truss && d.w < cc_skybridges_GRAND_FRAC
        && frontage > 2.0 * cc_skybridges_HALF_W_GRAND + 3.0) {
        half = cc_skybridges_HALF_W_GRAND;
    }
    let slack = frontage - 2.0 * (half + cc_skybridges_ROOF_OVER) - 1.0;
    if (slack < 0.0) {
        return s;
    }

    // Height: a storey between 25 m and the lower of the two b1 roofs less
    // its margin. b1 is the base box of every archetype, so this is the
    // "lands on the widest prim" rule stated once and obeyed everywhere.
    let z_top = min(ca.b1max.z, cb.b1max.z) - cc_skybridges_ROOF_MARGIN;
    let z_hi = min(z_top - cc_skybridges_TUBE_H, cc_skybridges_Z_MAX);
    if (z_hi < cc_skybridges_Z_MIN) {
        return s;
    }
    var z = floor((cc_skybridges_Z_MIN + (z_hi - cc_skybridges_Z_MIN) * d.y)
                  / CITY_FLOOR_H) * CITY_FLOOR_H;
    if (z < cc_skybridges_Z_MIN) {
        z = z + CITY_FLOOR_H;
    }

    s.ok = true;
    s.lo = a_face - cc_skybridges_EMBED;
    s.hi = b_face + cc_skybridges_EMBED;
    s.ctr = c0 + half + cc_skybridges_ROOF_OVER + 0.5 + slack * d.z;
    s.half = half;
    s.z = z;
    s.truss = truss;
    s.seed = pcg2d(cc_skybridges_eid(clo, axis) ^ vec2<u32>(0x68e31da4u,
                                                            0xb5297a4du));
    return s;
}

// --- the trace ------------------------------------------------------------
// All four edges of cell ci, cheap hash first. A hit counts only if its t
// falls inside this cell's segment, which is what splits the tube between
// the two cells with no overlap and no gap: the box's entry point lies in
// exactly one column, and that column's segment is the one that contains it.
fn cc_skybridges_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                             t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    // If this cell fails the gate it fails the min() on every one of its
    // edges too, so the neighbour reaches the same verdict.
    if (!cc.built || cc.density <= cc_skybridges_MIN_DENSITY) {
        return res;
    }
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_skybridges_Z_HI || max(za, zb) < cc_skybridges_Z_LO) {
        return res;
    }

    for (var e: i32 = 0; e < 4; e = e + 1) {
        let axis = e >> 1;
        let neg = (e & 1) == 1;
        let stp = select(vec2<i32>(1, 0), vec2<i32>(0, 1), axis == 1);
        let clo = select(ci, ci - stp, neg);
        if (!cc_skybridges_gate(clo, axis)) {
            continue;
        }
        // One extra cell fetch, and only behind the gate.
        let p0 = pcg2d(cc_skybridges_eid(clo, axis));
        let nb = city_cell(select(clo + stp, clo, neg));
        var ca = cc;
        var cb = nb;
        if (neg) {
            ca = nb;
            cb = cc;
        }
        let s = cc_skybridges_build(clo, axis, ca, cb, city_rand4(p0));
        if (!s.ok) {
            continue;
        }

        let zd = s.z;
        let zg = zd + cc_skybridges_DECK_T;
        let zr = zg + cc_skybridges_GLASS_H;
        let zt = zd + cc_skybridges_TUBE_H;
        let gh = s.half - cc_skybridges_GLASS_INSET;
        let rh = s.half + cc_skybridges_ROOF_OVER;
        for (var p: i32 = 0; p < 3; p = p + 1) {
            var w: f32; var z_lo: f32; var z_hi: f32;
            if (p == 0) {
                w = s.half; z_lo = zd; z_hi = zg;        // deck
            } else if (p == 1) {
                // Glass band, or the truss's central web: same prim, same
                // test, a tenth of the width.
                w = select(gh, cc_skybridges_TRUSS_HALF, s.truss);
                z_lo = zg; z_hi = zr;
            } else {
                w = select(rh, s.half * cc_skybridges_CHORD_FRAC, s.truss);
                z_lo = zr; z_hi = zt;                    // roof, or top chord
            }
            let bmin = cc_skybridges_bmin(axis, s.lo, s.ctr - w, z_lo);
            let bmax = cc_skybridges_bmin(axis, s.hi, s.ctr + w, z_hi);
            let hb = city_box_hit(o, inv_dir, bmin, bmax);
            if (hb.x <= hb.y && hb.x > 0.0 && hb.x < res.t
                && hb.x >= t0 - 1e-3 && hb.x <= t1 + 1e-3) {
                res.hit = true;
                res.t = hb.x;
                res.pos = o + hb.x * dir;
                res.normal = city_box_normal(res.pos, bmin, bmax);
                res.cell = ci;
                let side = select(abs(res.normal.y), abs(res.normal.x),
                                  axis == 1);
                // The whole web is a truss member; only the band's vertical
                // faces are glazing (its sill and head are shell).
                var ty = 0;
                if (p == 1) {
                    if (s.truss) { ty = 2; }
                    else if (side > 0.5) { ty = 1; }
                }
                res.kind = cc_skybridges_KIND_BASE + axis + 2 * ty;
            }
        }
    }
    return res;
}

// --- shading --------------------------------------------------------------

fn cc_skybridges_capsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, r: f32)
        -> f32 {
    let pa = p - a;
    let ba = b - a;
    let hh = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * hh) - r;
}

// Somebody crossing. Two capsules — a leaning body and a head — in bay
// metres, which is all a person is worth at the distance a bridge is
// usually seen from, and more than enough to turn a lit tube into an
// inhabited one.
fn cc_skybridges_figure(sh: vec4<f32>, pm: vec2<f32>, fp: f32) -> f32 {
    let sc = 0.90 + 0.22 * sh.w;
    let cx = 0.5 + (cc_skybridges_MULLION_PITCH - 1.0) * sh.y;
    let lean = (sh.z - 0.5) * 0.17;
    let body = cc_skybridges_capsule(
        pm, vec2<f32>(cx, 0.30 * sc), vec2<f32>(cx + lean, 1.40 * sc),
        0.20 * sc);
    let head = cc_skybridges_capsule(
        pm, vec2<f32>(cx + lean * 1.30, 1.58 * sc),
        vec2<f32>(cx + lean * 1.36, 1.65 * sc), 0.115 * sc);
    let aa = 0.03 + 0.5 * fp;
    return 1.0 - smoothstep(-aa, aa, min(body, head));
}

// The tube's shell: deck, roof and fascia. It is albedo-lit, so it is
// near-black and no amount of gain would change that honestly; what makes it
// read is structure. Ribs panelize the whole tube on the glazing pitch; the
// underside additionally carries the box girder and the service lamps, which
// are the only part of a bridge's belly the street ever actually sees.
fn cc_skybridges_shell(h: CityHit, s: cc_skybridges_Span, tint: vec3<f32>,
                       fill: vec3<f32>, moon: vec3<f32>, fp_eff: f32)
        -> vec3<f32> {
    var e = cc_skybridges_SHELL_FILL * fill
            + cc_skybridges_SHELL_TINT * tint
            + 0.045 * CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0);
    if (!s.ok) {
        return e;   // the span could not be rebuilt: flat shell, never magenta
    }
    let along = select(h.pos.x, h.pos.y, s.axis == 1);
    let cross_m = select(h.pos.y, h.pos.x, s.axis == 1) - s.ctr;
    let ua = along - s.lo;
    let aa = min(0.5 * fp_eff, 0.30);
    let lod = smoothstep(cc_skybridges_LOD_RIB.x, cc_skybridges_LOD_RIB.y,
                         fp_eff);

    let rp = ua - floor(ua / cc_skybridges_RIB_PITCH)
             * cc_skybridges_RIB_PITCH;
    let web = min(rp, cc_skybridges_RIB_PITCH - rp);
    let rib = 1.0 - smoothstep(cc_skybridges_RIB_HALF - aa - 0.02,
                               cc_skybridges_RIB_HALF + aa + 0.02, web);
    let rib_l = mix(rib, 2.0 * cc_skybridges_RIB_HALF
                         / cc_skybridges_RIB_PITCH, lod);
    e = e * mix(0.72, 1.75, rib_l);

    if (h.normal.z > -0.5) {
        return e;
    }
    let spine = 1.0 - smoothstep(cc_skybridges_SPINE_HALF - aa - 0.02,
                                 cc_skybridges_SPINE_HALF + aa + 0.02,
                                 abs(cross_m));
    let spine_l = mix(spine,
                      min(cc_skybridges_SPINE_HALF / max(s.half, 0.5), 1.0),
                      lod);
    e = e * mix(1.0, 1.45, spine_l);

    // The lamp string. Sub-pixel it becomes its own mean over one lamp cell,
    // so a belly seen from a kilometre keeps exactly the light it had close.
    let li = floor(ua / cc_skybridges_LAMP_PITCH);
    let lu = ua - (li + 0.5) * cc_skybridges_LAMP_PITCH;
    let dl = length(vec2<f32>(lu, cross_m));
    let laa = clamp(0.5 * fp_eff, 0.02, 0.40);
    let lamp = 1.0 - smoothstep(cc_skybridges_LAMP_R - laa,
                                cc_skybridges_LAMP_R + laa, dl);
    let lamp_mean = 3.14159265 * cc_skybridges_LAMP_R * cc_skybridges_LAMP_R
                    / (cc_skybridges_LAMP_PITCH * 2.0 * max(s.half, 0.5));
    let lamp_l = mix(lamp, lamp_mean,
                     smoothstep(cc_skybridges_LOD_LAMP.x,
                                cc_skybridges_LOD_LAMP.y, fp_eff));
    return e + tint * (cc_skybridges_LAMP_RAD * lamp_l);
}

// The truss's central web. It is only 0.6 m thick, but its SIDE face is the
// whole span by the whole handrail height — the largest surface this variant
// owns — so it is where the truss has to actually look like a truss. The
// Warren lattice is shaded rather than built: members catch the district's
// fill, and the bays between them go to black, which is very nearly what
// looking through a truss at a night street gives you anyway.
fn cc_skybridges_truss(h: CityHit, s: cc_skybridges_Span, tint: vec3<f32>,
                       fill: vec3<f32>, moon: vec3<f32>, fp_eff: f32)
        -> vec3<f32> {
    let lit = 0.10 * fill + 0.010 * tint
              + 0.050 * CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0);
    if (!s.ok) {
        return lit;
    }
    let ua = select(h.pos.x, h.pos.y, s.axis == 1) - s.lo;
    let pv = h.pos.z - s.z - cc_skybridges_DECK_T;
    let aa = min(0.5 * fp_eff, 0.25);
    let hgt = cc_skybridges_GLASS_H;

    // Warren lattice: a triangle wave of diagonals between top and bottom
    // chords, with a vertical post where the diagonals meet.
    let f = ua / cc_skybridges_TRUSS_PITCH;
    let zig = abs(2.0 * (f - floor(f)) - 1.0);
    let diag = 1.0 - smoothstep(cc_skybridges_MEMBER_T - aa,
                                cc_skybridges_MEMBER_T + aa,
                                abs(pv - hgt * zig));
    let vu = abs(ua - round(f) * cc_skybridges_TRUSS_PITCH);
    let post = 1.0 - smoothstep(cc_skybridges_POST_T - aa,
                                cc_skybridges_POST_T + aa, vu);
    let chord = 1.0 - smoothstep(cc_skybridges_CHORD_T - aa,
                                 cc_skybridges_CHORD_T + aa,
                                 min(pv, hgt - pv));
    let mem = mix(max(max(diag, post), chord), cc_skybridges_TRUSS_COVER,
                  smoothstep(cc_skybridges_LOD_RIB.x,
                             cc_skybridges_LOD_RIB.y, fp_eff));
    let e = lit * mem;

    let li = floor(ua / cc_skybridges_STRING_PITCH);
    let lu = ua - (li + 0.5) * cc_skybridges_STRING_PITCH;
    // The string hangs just under the top chord.
    let dl = length(vec2<f32>(lu, pv - cc_skybridges_GLASS_H * 0.82));
    let laa = clamp(0.5 * fp_eff, 0.02, 0.40);
    let lamp = 1.0 - smoothstep(cc_skybridges_STRING_R - laa,
                                cc_skybridges_STRING_R + laa, dl);
    let lamp_mean = 3.14159265 * cc_skybridges_STRING_R
                    * cc_skybridges_STRING_R
                    / (cc_skybridges_STRING_PITCH * cc_skybridges_GLASS_H);
    let lamp_l = mix(lamp, lamp_mean,
                     smoothstep(cc_skybridges_LOD_LAMP.x,
                                cc_skybridges_LOD_LAMP.y, fp_eff));
    return e + tint * (cc_skybridges_STRING_RAD * lamp_l);
}

fn cc_skybridges_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let moon = u.sun_dir.xyz;
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(h.pos.xy, 3.0));
    let k = h.kind - cc_skybridges_KIND_BASE;
    let axis = k & 1;
    let ty = k >> 1;
    let cell = u.ocean_params.x;

    // Rebuild the exact span from the hit. Containment guarantees the tube
    // straddles one block boundary and stays in one row, so rounding the hit
    // on the bridge's axis and flooring it across names the two cells.
    var clo: vec2<i32>;
    var stp: vec2<i32>;
    if (axis == 0) {
        clo = vec2<i32>(i32(round(h.pos.x / cell)) - 1,
                        i32(floor(h.pos.y / cell)));
        stp = vec2<i32>(1, 0);
    } else {
        clo = vec2<i32>(i32(floor(h.pos.x / cell)),
                        i32(round(h.pos.y / cell)) - 1);
        stp = vec2<i32>(0, 1);
    }
    let p0 = pcg2d(cc_skybridges_eid(clo, axis));
    let s = cc_skybridges_build(clo, axis, city_cell(clo),
                                city_cell(clo + stp), city_rand4(p0));

    // The house colour: mostly tungsten corridors, some fluorescent, the
    // odd cyan one. Warm-biased so a bridge reads against the cooler
    // curtain-wall towers it usually connects.
    let bh = city_rand4(s.seed);
    let tint = city_window_color(bh.x * 0.92, 0.75);

    // The footprint a slanted face actually presents: a tube seen end-on
    // smears its detail over far fewer pixels than its distance suggests.
    let fp_eff = fp / clamp(abs(dot(dir, h.normal)), 0.20, 1.0);

    if (ty == 0 || !s.ok) {
        return cc_skybridges_shell(h, s, tint, fill, moon, fp_eff);
    }
    if (ty == 2) {
        return cc_skybridges_truss(h, s, tint, fill, moon, fp_eff);
    }

    // Band coordinates in metres: along the span from wall A, and up from
    // the deck surface.
    let ua = select(h.pos.x, h.pos.y, axis == 1) - s.lo;
    let pv = h.pos.z - s.z - cc_skybridges_DECK_T;

    let ib = i32(floor(ua / cc_skybridges_MULLION_PITCH));
    let sh = city_rand4(vec2<u32>(s.seed.x ^ (bitcast<u32>(ib) * 0x9e3779b9u),
                                  s.seed.y ^ 0x85ebca6bu));
    let bright = 0.55 + 0.90 * sh.x;
    let bay = tint * (cc_skybridges_RADIANCE * bright);

    // Octave 2: the whole ribbon, one colour. Bays average out last.
    let e_far = tint * (cc_skybridges_RADIANCE * cc_skybridges_MEAN_COVER
                        * cc_skybridges_MEAN_COMP);
    // Octave 1: bays still separable, mullions and figures gone.
    let lit_seg = select(1.0, 0.0, sh.y < cc_skybridges_DARK_SEG);
    let e_bay = bay * (cc_skybridges_MEAN_COVER / (1.0 - cc_skybridges_DARK_SEG)
                       * cc_skybridges_MEAN_COMP * lit_seg);

    // Octave 0: the glazing itself.
    let pu = ua - f32(ib) * cc_skybridges_MULLION_PITCH;
    let aa_u = min(0.5 * fp_eff, 0.35);
    let web = min(pu, cc_skybridges_MULLION_PITCH - pu);
    let mull = smoothstep(cc_skybridges_MULLION_HALF - aa_u - 0.02,
                          cc_skybridges_MULLION_HALF + aa_u + 0.02, web);
    let aa_v = min(0.5 * fp_eff, 0.20);
    let glazed = smoothstep(cc_skybridges_GLASS_V0 - aa_v,
                            cc_skybridges_GLASS_V0 + aa_v + 0.03, pv)
               * (1.0 - smoothstep(cc_skybridges_GLASS_V1 - aa_v - 0.03,
                                   cc_skybridges_GLASS_V1 + aa_v, pv));
    var cover = mix(cc_skybridges_SPANDREL, 1.0, mull * glazed);
    if (sh.z < cc_skybridges_FIG_FRAC) {
        let f_lod = smoothstep(cc_skybridges_LOD_FIG.x,
                               cc_skybridges_LOD_FIG.y, fp_eff);
        if (f_lod < 1.0) {
            let fh = city_rand4(s.seed ^ vec2<u32>(bitcast<u32>(ib)
                                                   * 0x27d4eb2fu, 0x165667b1u));
            let m = cc_skybridges_figure(fh, vec2<f32>(pu, pv), fp_eff)
                    * (1.0 - f_lod);
            cover = cover * mix(1.0, cc_skybridges_BODY_T, m);
        }
    }
    let e_near = bay * (cover * lit_seg);

    let b1 = smoothstep(cc_skybridges_LOD_MUL.x, cc_skybridges_LOD_MUL.y,
                        fp_eff);
    let b2 = smoothstep(cc_skybridges_LOD_SEG.x, cc_skybridges_LOD_SEG.y,
                        fp_eff);
    return mix(e_near, mix(e_bay, e_far, b2), b1)
           + 0.02 * fill;
}
