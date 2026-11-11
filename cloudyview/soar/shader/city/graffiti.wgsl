// === CITY GRAFFITI (desert-flora) ===
// CITY GRAFFITI (2026-11-08)
// ===========================================================================
//
// Spray paint on the city's surfaces (Thomas: "a really excellent addition
// perhaps could be graffiti, esp on containers/in industrial areas") —
// today the industry component's containers, warehouse walls and dock
// doors (CITY_GRAFFITI_*). What the references show
// (temp/references/graffiti, Commons photographs), and so what this draws:
//
//   * A REACH BAND. Paint lives from ~0.3 to ~2.5 m above whatever the
//     painter stood on — the foot of every pier of a bridge, the bottom
//     two and a half metres of a long factory wall — and clean surface
//     above it.
//   * CLUMPING BY SURFACE. One wall or pier face is saturated, the next is
//     clean; a row of containers carries one throw-up a side and bare
//     stretches. So the unit is a RUN (one wall face, container side, door
//     or pier face), and a run draws a level first: clean, lightly tagged,
//     or heavily painted.
//   * KINDS. TAGS: one colour, one stroke, 0.2-0.45 m tall, in loose rows.
//     THROW-UPS: 1-2 m bubble letters, a flat fill (silver above all) inside
//     a dark outline, a white shine tick or two, drips under the fill.
//     PIECES: the same lettering in bright fills over a background splash.
//     BUFF: flat patches of wall colour over old paint, a shade off.
//   * LAYERS. New over old: an old layer of letters, buff over part of it,
//     a new layer of throw-ups, and tags over everything.
//   * PALETTE. Silver, black and white most (they read under any lamp),
//     then red, orange, yellow, blue, purple, lime, cyan, pink.
//
// Paint is albedo, nothing else: city_graffiti_paint returns how to change
// the albedo a surface's owner was about to light, albedo * keep + add —
// linear in that albedo, so that buff (a patch of the wall's own colour, a
// shade off) and paint compose the same way at every range. The owner
// lights the result as it lights its bare surface: the ribs of a
// corrugated side still shade through the paint.
//
// THE MEAN. Each layer is a lattice of slots along the run, one element at
// most to a slot and wholly inside it, each element drawn from the slot's
// own hash; the layers' draws are independent. Every element's shape is
// scale-free (drawn in units of its own height h, every width included),
// so its painted area is a constant times h^2 in expectation, and painted
// length at a height inside its box is that constant times h when spread
// evenly over the box's height. The pixel reads one of three rungs, keyed
// on its footprint (fpuv, metres along u and v on the surface) against the
// element's size:
//
//   FINE — the strokes, their edges filtered over the footprint exactly
//     (a straight edge's coverage is the overlap of the footprint with the
//     painted side);
//   MID — each element as its box, painted evenly at its expected density,
//     the box's edges box-filtered over the footprint;
//   FAR — the layer's expectation along the run: slot odds, the element's
//     expected density, and the closed-form expected height profile, box-
//     filtered over the footprint's height (city_graffiti_ramp).
//
// FAR is MID's expectation at every height. MID is FINE's in each
// element's paint and colour, spread evenly over the element's box where
// FINE has it where the strokes are — a difference only within a few
// pixels, since MID takes over as the element shrinks to that. The layers
// composite in expectation as they do pointwise (each element's draws are
// separate hash lanes, uncorrelated in practice, and the compositing is
// linear in each layer's coverage), so the paint dissolves into its mean
// colour and coverage — conditional on the run's level, which a far run
// still shows. The areas the expectations need are the shader's own shapes,
// measured (tools/city_graffiti_harness.py means) and pinned by
// tests/test_city_graffiti.py; the palettes' means are exact.

struct CityGraffitiRun {
    seed: vec2<u32>,   // the run's own draw
    len: f32,          // its length along u (m)
    kind: i32,         // CITY_GRAFFITI_*: what it is, which sets its odds
}

// albedo * keep + add.
struct CityGraffitiPaint {
    keep: f32,
    add: vec3<f32>,
}

// One layer's paint at a point (or its expectation): coverage, coverage x
// paint colour, and coverage x the buff's tone (the share of the albedo
// beneath that it repaints in the wall's own colour).
struct CityGraffitiCover {
    a: f32,
    ac: vec3<f32>,
    at: f32,
}

const CITY_GRAFFITI_CONTAINER: i32 = 0;
const CITY_GRAFFITI_WALL_STREET: i32 = 1;
const CITY_GRAFFITI_WALL_COURT: i32 = 2;
const CITY_GRAFFITI_DOOR: i32 = 3;

// The layers, oldest first: old letters (throw-ups and pieces) on two
// lattices half a slot apart, buff, new throw-ups on two more, and three
// lattices of tags.
const CITY_GRAFFITI_LAYERS: i32 = 8;
const CITY_GRAFFITI_BUFF_LAYER: i32 = 2;

// Layers 0, 1 (old) and 3, 4 (new) are lettering; 5-7 are tags.
fn city_graffiti_is_letters(layer: i32) -> bool {
    return layer <= 1 || layer == 3 || layer == 4;
}

fn city_graffiti_is_old(layer: i32) -> bool {
    return layer <= 1;
}
// Letters: height h ~ U(H0, H1) standing on the run's baseline; 3-5 glyphs
// (GLYPHS_MEAN on average) on a pitch of GW_LO-GW_HI h, PAD h of room at
// each end for their overlap, slant, outlines and splash.
const CITY_GRAFFITI_LETTER_H0: f32 = 1.2;
const CITY_GRAFFITI_LETTER_H1: f32 = 2.2;
const CITY_GRAFFITI_GW_LO: f32 = 0.38;
const CITY_GRAFFITI_GW_HI: f32 = 0.55;
const CITY_GRAFFITI_PAD: f32 = 0.45;
const CITY_GRAFFITI_GLYPHS_MEAN: f32 = 4.0;
// Where in its box a lettering element's paint lies: the strokes fill the
// letters' bodies, from Y0 to Y1 of its height, nearly evenly, and little
// above or below — measured (tools/city_graffiti_harness.py profile) as the
// even band with the strokes' mean height and spread. MID and FAR lay the
// paint there, not over the whole box: layers that overlap must overlap
// where the paint is, or a heavy wall's far composite runs a few percent
// darker than its strokes.
const CITY_GRAFFITI_LETTER_Y0: f32 = 0.1547;
const CITY_GRAFFITI_LETTER_Y1: f32 = 0.8781;
// The share of old-layer letters that are pieces (bright fills, a
// background splash).
const CITY_GRAFFITI_PIECE_FRAC: f32 = 0.35;
// FINE gives way to MID as the footprint passes MID0 to MID1 of the
// element's height (city_graffiti_element); a tag's strokes sooner.
const CITY_GRAFFITI_MID0: f32 = 0.12;
const CITY_GRAFFITI_MID1: f32 = 0.25;
const CITY_GRAFFITI_TAG_MID0: f32 = 0.05;
const CITY_GRAFFITI_TAG_MID1: f32 = 0.15;
// A tag's strokes lose area where they meet as the filter widens: at a
// junction the blurred union counts one footprint's width of stroke once
// where it is painted twice. Measured (tools/city_graffiti_harness.py
// bias; tests/test_city_graffiti.py), the loss is TAG_LOSS f^2 (f the
// footprint in units of the tag's height), within half a percent to
// TAG_MID1; FINE puts it back.
const CITY_GRAFFITI_TAG_LOSS: f32 = 4.5;
// Old paint is faded: its colours at FADE_LO-1 of new, FADE_MEAN on average.
const CITY_GRAFFITI_FADE_LO: f32 = 0.7;
const CITY_GRAFFITI_FADE_MEAN: f32 = 0.85;
// Buff: patches from just above the ground, h ~ U(B0, B1), L/h ~ U(.6, 2.2).
const CITY_GRAFFITI_BUFF_FOOT: f32 = 0.05;
const CITY_GRAFFITI_BUFF_H0: f32 = 0.8;
const CITY_GRAFFITI_BUFF_H1: f32 = 2.0;
const CITY_GRAFFITI_BUFF_A0: f32 = 0.6;
const CITY_GRAFFITI_BUFF_A1: f32 = 2.2;
// Tags: h ~ U(T0, T1) centred anywhere from C0 to C1 (no rows: a tag goes
// where the writer's hand was), L/h ~ U(1.4, 2.6). Three lattices of them,
// a third of a slot apart.
const CITY_GRAFFITI_TAG_H0: f32 = 0.15;
const CITY_GRAFFITI_TAG_H1: f32 = 0.60;
const CITY_GRAFFITI_TAG_C0: f32 = 0.6;
const CITY_GRAFFITI_TAG_C1: f32 = 2.1;
const CITY_GRAFFITI_TAG_A0: f32 = 1.4;
const CITY_GRAFFITI_TAG_A1: f32 = 2.6;

// Painted areas, in units of h^2 (letters: per glyph; buff and tags: per
// unit of box, L h), from the shader's own shapes
// (tools/city_graffiti_harness.py means; tests/test_city_graffiti.py).
// Letters, per glyph (E[area] / E[n]): the fill (the shine painted over it
// included), the shine, the outline ring, the drips, and a piece's splash
// ring. Buff and tags, per unit of box (E[area] / E[L], L in h).
const CITY_GRAFFITI_A_FILL: f32 = 0.1758;
const CITY_GRAFFITI_A_SHINE: f32 = 0.0024;
const CITY_GRAFFITI_A_LINE: f32 = 0.1869;
const CITY_GRAFFITI_A_DRIP: f32 = 0.0005;
const CITY_GRAFFITI_A_SPLASH: f32 = 0.0484;
const CITY_GRAFFITI_RHO_BUFF: f32 = 0.8938;
const CITY_GRAFFITI_RHO_TAG: f32 = 0.1454;

// Spray-paint albedos, dimmed to the belt's convention (cc_industry's
// liveries: a white box reads 0.45 under a night sky).
const CITY_GRAFFITI_WHITE: vec3<f32> = vec3<f32>(0.50, 0.50, 0.49);
const CITY_GRAFFITI_SILVER: vec3<f32> = vec3<f32>(0.40, 0.41, 0.42);
const CITY_GRAFFITI_BLACK: vec3<f32> = vec3<f32>(0.035, 0.035, 0.038);

fn city_graffiti_hue(x: f32) -> vec3<f32> {
    // The bright colours, equally likely.
    if (x < 0.125) { return vec3<f32>(0.36, 0.05, 0.04); }    // red
    if (x < 0.250) { return vec3<f32>(0.45, 0.16, 0.03); }    // orange
    if (x < 0.375) { return vec3<f32>(0.48, 0.38, 0.05); }    // yellow
    if (x < 0.500) { return vec3<f32>(0.25, 0.42, 0.06); }    // lime
    if (x < 0.625) { return vec3<f32>(0.06, 0.30, 0.38); }    // cyan
    if (x < 0.750) { return vec3<f32>(0.05, 0.12, 0.36); }    // blue
    if (x < 0.875) { return vec3<f32>(0.20, 0.08, 0.32); }    // purple
    return vec3<f32>(0.48, 0.12, 0.26);                       // pink
}
const CITY_GRAFFITI_HUE_MEAN: vec3<f32> = vec3<f32>(0.29125, 0.20375, 0.1875);

// A throw-up's fill: silver most, then white, black and the hues.
fn city_graffiti_fill(x: f32) -> vec3<f32> {
    if (x < 0.45) { return CITY_GRAFFITI_SILVER; }
    if (x < 0.57) { return CITY_GRAFFITI_WHITE; }
    if (x < 0.63) { return CITY_GRAFFITI_BLACK; }
    return city_graffiti_hue((x - 0.63) / 0.37);
}
// 0.45 silver + 0.12 white + 0.06 black + 0.37 hue mean.
const CITY_GRAFFITI_FILL_MEAN: vec3<f32> =
    vec3<f32>(0.3498625, 0.3219875, 0.3194550);

// An outline: black mostly, else a hue or white.
fn city_graffiti_line(x: f32) -> vec3<f32> {
    if (x < 0.58) { return CITY_GRAFFITI_BLACK; }
    if (x < 0.66) { return CITY_GRAFFITI_WHITE; }
    return city_graffiti_hue((x - 0.66) / 0.34);
}
// 0.58 black + 0.08 white + 0.34 hue mean.
const CITY_GRAFFITI_LINE_MEAN: vec3<f32> =
    vec3<f32>(0.1593250, 0.1295750, 0.1249900);

// A tag: black, white or silver mostly, else a hue.
fn city_graffiti_ink(x: f32) -> vec3<f32> {
    if (x < 0.42) { return CITY_GRAFFITI_BLACK; }
    if (x < 0.58) { return CITY_GRAFFITI_WHITE; }
    if (x < 0.66) { return CITY_GRAFFITI_SILVER; }
    return city_graffiti_hue((x - 0.66) / 0.34);
}
// 0.42 black + 0.16 white + 0.08 silver + 0.34 hue mean.
const CITY_GRAFFITI_INK_MEAN: vec3<f32> =
    vec3<f32>(0.2257250, 0.1967750, 0.1917100);

// How often a run of each kind is lightly tagged and heavily painted:
// (P(light), P(heavy)); the rest are clean.
fn city_graffiti_odds(kind: i32) -> vec2<f32> {
    switch kind {
        case CITY_GRAFFITI_CONTAINER: { return vec2<f32>(0.25, 0.12); }
        case CITY_GRAFFITI_WALL_STREET: { return vec2<f32>(0.30, 0.25); }
        case CITY_GRAFFITI_WALL_COURT: { return vec2<f32>(0.20, 0.05); }
        default: { return vec2<f32>(0.35, 0.05); }   // DOOR
    }
}

// A layer's slot odds at a level (1 light, 2 heavy).
fn city_graffiti_slot_p(layer: i32, level: i32) -> f32 {
    let heavy = level == 2;
    if (city_graffiti_is_old(layer)) {
        return select(0.0, 0.95, heavy);
    }
    if (layer == CITY_GRAFFITI_BUFF_LAYER) {
        return select(0.15, 0.35, heavy);
    }
    if (city_graffiti_is_letters(layer)) {
        return select(0.25, 0.85, heavy);
    }
    return select(0.60, 0.90, heavy);
}

// A layer's slot length and lattice offset (m). Every lattice has its own
// length, no two in a simple ratio: an element lies inside its slot, so a
// layer paints a slot's middle more often than its ends, and lattices of
// one length would hold their phases along the whole wall — two half a slot
// apart avoid each other, and the wall comes out more painted than layers
// drawn independently, which is what FAR composites. Lengths that drift
// through each other's phases make a long wall's mean the product FAR
// takes. The letters' shortest slot holds their longest element (8.0 m).
fn city_graffiti_slot(layer: i32) -> vec2<f32> {
    switch layer {
        case 0: { return vec2<f32>(9.0, 0.0); }
        case 1: { return vec2<f32>(9.7, 4.5); }
        case 2: { return vec2<f32>(6.37, 2.0); }
        case 3: { return vec2<f32>(10.3, 2.25); }
        case 4: { return vec2<f32>(11.1, 6.75); }
        case 5: { return vec2<f32>(1.7, 0.0); }
        case 6: { return vec2<f32>(1.83, 0.6); }
        default: { return vec2<f32>(1.97, 1.2); }
    }
}

// A layer's lattice on a run: (slot length, where slot 0 starts, the number
// of whole slots). Old and new letters' first lattices (layers 0 and 3)
// take as many slots as the run holds, from their offset where there is
// room for it, else from the room there is: a run shorter than offset +
// slot (a container side) still carries a throw-up of either age. The
// second lattices, and buff and tags, keep their offsets, and a short run
// holds none of the second lattices' slots: they are there to fill a long
// wall's gaps, which a run of one slot has none of. (Fitting them too put
// four throw-ups on a heavy container side, one over the next in its
// middle, where FAR spreads each evenly along the run — measured, 4% more
// cover than the strokes.)
fn city_graffiti_lattice(run: CityGraffitiRun, layer: i32) -> vec3<f32> {
    let sl = city_graffiti_slot(layer);
    let n_fit = max(floor(run.len / sl.x), 0.0);
    if (layer == 0 || layer == 3) {
        return vec3<f32>(sl.x, min(sl.y, run.len - n_fit * sl.x), n_fit);
    }
    return vec3<f32>(sl.x, sl.y, max(floor((run.len - sl.y) / sl.x), 0.0));
}

// Spray paint's own soft edge: SOFT (m) of falloff at the edge of every
// stroke, folded into the footprint the edge is filtered over (in units
// of an element's height h). A symmetric ramp moves no paint across a
// straight edge, so it softens without changing the areas.
const CITY_GRAFFITI_SOFT: f32 = 0.006;

fn city_graffiti_soft(f: f32, h: f32) -> f32 {
    let s = CITY_GRAFFITI_SOFT / h;
    return sqrt(f * f + s * s);
}

// Exact box-filter coverage of the interval [lo, hi] by the footprint
// [x - f/2, x + f/2]. Every f here is a footprint, and footprints are
// positive: city_graffiti_paint's fuv is a pixel's (city_graffiti_footprint
// of a pixel angle times a distance, both positive).
fn city_graffiti_span(x: f32, f: f32, lo: f32, hi: f32) -> f32 {
    // @domain: f > 0, a footprint
    return clamp((min(x + 0.5 * f, hi) - max(x - 0.5 * f, lo)) / f, 0.0, 1.0);
}

// Coverage of the painted side (sd < 0) of a straight edge at signed
// distance sd, by a footprint f wide.
fn city_graffiti_edge(sd: f32, f: f32) -> f32 {
    // @domain: f > 0, a footprint
    return clamp(0.5 - sd / f, 0.0, 1.0);
}

// R(y) = int_0^y E[h 1(s < h)] ds for h ~ U(h0, h1): the expected painted
// height profile of a box of height h standing on y = 0, integrated. Its
// slope is E[h] below h0, (h1^2 - y^2) / (2 (h1 - h0)) between, 0 above;
// R(inf) = E[h^2].
fn city_graffiti_ramp(y: f32, h0: f32, h1: f32) -> f32 {
    let dh = h1 - h0;
    let eh = 0.5 * (h0 + h1);
    let yc = clamp(y, 0.0, h1);
    let lin = eh * min(yc, h0);
    let z = max(yc, h0);
    return lin + (h1 * h1 * (z - h0) - (z * z * z - h0 * h0 * h0) / 3.0)
                 / (2.0 * dh);
}

// The profile's mean over the footprint [v - f/2, v + f/2], for a box
// standing on `foot`.
fn city_graffiti_profile(v: f32, f: f32, foot: f32, h0: f32,
                         h1: f32) -> f32 {
    let y0 = v - 0.5 * f - foot;
    let y1 = v + 0.5 * f - foot;
    // @domain: f > 0, a footprint
    return (city_graffiti_ramp(y1, h0, h1) - city_graffiti_ramp(y0, h0, h1))
           / f;
}

// The share of a lettering element's paint below t (in units of its
// height): its band, CITY_GRAFFITI_LETTER_Y0 to _Y1, filled evenly.
fn city_graffiti_letter_share(t: f32) -> f32 {
    return clamp((t - CITY_GRAFFITI_LETTER_Y0)
                 / (CITY_GRAFFITI_LETTER_Y1 - CITY_GRAFFITI_LETTER_Y0),
                 0.0, 1.0);
}

// Phi(y) = int_0^y E[h p(s / h)] ds = E[h^2 P(min(y / h, 1))] for h ~ U(h0,
// h1), p the lettering's band (city_graffiti_letter_share is P): the
// expected painted height up to y above the element's foot, per unit of
// painted area per h^2. P is 1 for h < y / Y1, (y / h - Y0) / (Y1 - Y0) up
// to h = y / Y0, 0 beyond; so Phi is the integral of h^2 to y / Y1 and of
// (y h - Y0 h^2) / (Y1 - Y0) from there to y / Y0, over h1 - h0.
fn city_graffiti_letter_phi(y: f32, h0: f32, h1: f32) -> f32 {
    let y0 = CITY_GRAFFITI_LETTER_Y0;
    let y1 = CITY_GRAFFITI_LETTER_Y1;
    let yp = max(y, 0.0);
    // @domain: Y0, Y1 > 0 (constants)
    let lo = clamp(yp / y1, h0, h1);
    let hi = clamp(yp / y0, h0, h1);
    let full = (lo * lo * lo - h0 * h0 * h0) / 3.0;
    let part = (yp * 0.5 * (hi * hi - lo * lo)
                - y0 * (hi * hi * hi - lo * lo * lo) / 3.0) / (y1 - y0);
    return (full + part) / (h1 - h0);
}

// S(t) = int_0^t R: the ramp integrated again, for the tags' spread.
fn city_graffiti_ramp2(t: f32, h0: f32, h1: f32) -> f32 {
    let dh = h1 - h0;
    let eh = 0.5 * (h0 + h1);
    let a = clamp(t, 0.0, h0);
    let b = clamp(t, h0, h1);
    let x = b - h0;
    let h03 = h0 * h0 * h0;
    return 0.5 * eh * a * a + eh * h0 * x
        + (0.5 * h1 * h1 * x * x
           - ((b * b * b * b - h03 * h0) * 0.25 - h03 * x) / 3.0) / (2.0 * dh)
        + city_graffiti_ramp(h1, h0, h1) * max(t - h1, 0.0);
}

// The same mean for a box of height h ~ U(h0, h1) CENTRED anywhere in
// [c0, c1] (the tags). Centred at c its profile is E[h 1(2|v - c| < h)],
// whose integral from c is F(y) = sign(y) R(2|y|) / 2; averaged over c it
// is (F(v - c0) - F(v - c1)) / (c1 - c0); and over the footprint, the same
// with F's own integral G(y) = S(2|y|) / 4 (even).
fn city_graffiti_profile_spread(v: f32, f: f32, c0: f32, c1: f32, h0: f32,
                                h1: f32) -> f32 {
    let lo = v - 0.5 * f;
    let hi = v + 0.5 * f;
    let g = city_graffiti_ramp2(2.0 * abs(hi - c0), h0, h1)
        - city_graffiti_ramp2(2.0 * abs(lo - c0), h0, h1)
        - city_graffiti_ramp2(2.0 * abs(hi - c1), h0, h1)
        + city_graffiti_ramp2(2.0 * abs(lo - c1), h0, h1);
    // @domain: f > 0, a footprint, and c1 > c0
    return 0.25 * g / (f * (c1 - c0));
}

// --- the shapes ------------------------------------------------------------

// A rounded box's signed distance: half sizes b, corner radius r.
fn city_graffiti_rbox(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// A capsule's (segment a-b, radius r) signed distance.
fn city_graffiti_capsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>,
                         r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    // @domain: a != b by construction (a stroke of nonzero length)
    let k = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * k) - r;
}

// A bubble letter, signed distance in units of the element's height, at p
// (from the letter's centre, already turned into its own slant): a fat
// rounded body and the cuts that make it one of eight letterforms. A cut
// narrower than two outlines fills with the outline's colour — the drawn
// line inside a throw-up letter — and a wider one opens onto the wall. The
// eight: O (a counter), U (a slot from the top), N (one from the bottom),
// C (one from the right), E (two from the right), S (one from each side,
// offset), X (one from the top and one from the bottom), R (a small
// counter high, a slot from the bottom at the right). This is the one
// function to swap for other letterforms.
fn city_graffiti_glyph(p: vec2<f32>, b: vec2<f32>, kind: i32,
                       cut: f32) -> f32 {
    let r = 0.9 * min(b.x, b.y);
    var d = city_graffiti_rbox(p, b, r);
    let top = b.y + 0.1;
    let side = b.x + 0.1;
    var c = 1e3;
    switch kind {
        case 0: {
            c = city_graffiti_capsule(p, vec2<f32>(0.0, -0.32 * b.y),
                                      vec2<f32>(0.0, 0.32 * b.y), cut);
        }
        case 1: {
            c = city_graffiti_capsule(p, vec2<f32>(0.0, top),
                                      vec2<f32>(0.0, -0.3 * b.y), cut);
        }
        case 2: {
            c = city_graffiti_capsule(p, vec2<f32>(0.0, -top),
                                      vec2<f32>(0.0, 0.3 * b.y), cut);
        }
        case 3: {
            c = city_graffiti_capsule(p, vec2<f32>(side, 0.0),
                                      vec2<f32>(-0.25 * b.x, 0.0), cut);
        }
        case 4: {
            c = min(city_graffiti_capsule(p, vec2<f32>(side, 0.36 * b.y),
                                          vec2<f32>(-0.1 * b.x, 0.36 * b.y),
                                          cut),
                    city_graffiti_capsule(p, vec2<f32>(side, -0.36 * b.y),
                                          vec2<f32>(-0.1 * b.x, -0.36 * b.y),
                                          cut));
        }
        case 5: {
            c = min(city_graffiti_capsule(p, vec2<f32>(-side, 0.3 * b.y),
                                          vec2<f32>(0.3 * b.x, 0.3 * b.y),
                                          cut),
                    city_graffiti_capsule(p, vec2<f32>(side, -0.3 * b.y),
                                          vec2<f32>(-0.3 * b.x, -0.3 * b.y),
                                          cut));
        }
        case 6: {
            c = min(city_graffiti_capsule(p, vec2<f32>(0.0, top),
                                          vec2<f32>(0.0, 0.25 * b.y), cut),
                    city_graffiti_capsule(p, vec2<f32>(0.0, -top),
                                          vec2<f32>(0.0, -0.25 * b.y), cut));
        }
        default: {
            c = min(city_graffiti_capsule(p, vec2<f32>(-0.05 * b.x, 0.38 * b.y),
                                          vec2<f32>(0.05 * b.x, 0.38 * b.y),
                                          cut),
                    city_graffiti_capsule(p, vec2<f32>(0.35 * b.x, -top),
                                          vec2<f32>(0.2 * b.x, -0.15 * b.y),
                                          cut));
        }
    }
    return max(d, -c);
}

// A lettering element's draws: its height, glyphs, glyph pitch and box
// length (both in h), the box's start (m), outline width and cut radius
// (in h), slant, whether its letters stack left over right, whether it is
// a piece, its fill's split and its glyphs' seed (its colours are
// CityGraffitiInk's). The draws that shape it and the draws that colour it
// are independent, which is what lets its mean colour be its colours'
// means (city_graffiti_far).
struct CityGraffitiLetters {
    h: f32,
    n: i32,
    gw: f32,
    len: f32,
    x0: f32,
    ow: f32,
    cut: f32,
    slant: f32,
    left_up: bool,
    piece: bool,
    split: f32,           // the fill's second tone below it (in h)
    gs: vec2<u32>,
}

// A lettering element's colours, apart from its shape so that they are not
// held through the glyphs' distance fields (city_graffiti_letter_parts) in
// the register file of the stage that paints: drawn after them, from the
// same draws.
struct CityGraffitiInk {
    fill: vec3<f32>,
    fill2: vec3<f32>,     // below `split`; fill again for one tone
    line: vec3<f32>,
    splash: vec3<f32>,
    shine: vec3<f32>,
}

// The lettering in slot i of a lettering `layer`: draws d (presence, height,
// glyphs, place), e (piece, fill, outline, glyph pitch), g (outline width,
// slant, splash, cut and stacking), c (two tones or one, the second fill,
// the split, the fade). The second fill is drawn from the same palette as
// the first, so a two-tone letter's mean colour is a one-tone letter's;
// and old paint's fade is independent of everything, so the layer's mean
// colour is its fresh colour times FADE_MEAN.
fn city_graffiti_letters_at(run: CityGraffitiRun, layer: i32, i: i32)
        -> CityGraffitiLetters {
    let sl = city_graffiti_lattice(run, layer);
    let d = city_graffiti_draw(run, layer, i, 0u);
    let e = city_graffiti_draw(run, layer, i, 1u);
    let g = city_graffiti_draw(run, layer, i, 2u);
    let c = city_graffiti_draw(run, layer, i, 3u);
    var l: CityGraffitiLetters;
    l.h = mix(CITY_GRAFFITI_LETTER_H0, CITY_GRAFFITI_LETTER_H1, d.y);
    // 3-5 glyphs (city_u01 can return 1.0, which would make a sixth).
    l.n = 3 + min(i32(3.0 * d.z), 2);
    l.gw = mix(CITY_GRAFFITI_GW_LO, CITY_GRAFFITI_GW_HI, e.w);
    l.len = f32(l.n) * l.gw + 2.0 * CITY_GRAFFITI_PAD;
    l.x0 = sl.y + f32(i) * sl.x + (sl.x - l.len * l.h) * d.w;
    l.ow = 0.04 + 0.025 * g.x;
    l.cut = 0.05 + 0.05 * fract(g.w * 7.31);
    l.slant = -0.08 + 0.30 * g.y;
    l.left_up = g.w < 0.5;
    l.piece = city_graffiti_is_old(layer) && e.x < CITY_GRAFFITI_PIECE_FRAC;
    l.split = 0.40 + 0.25 * c.z;
    l.gs = run.seed ^ vec2<u32>(u32(i) * 0x3c6ef372u, u32(layer) + 0xa54ff53au);
    return l;
}

fn city_graffiti_letters_ink(run: CityGraffitiRun, layer: i32, i: i32,
                             piece: bool) -> CityGraffitiInk {
    let e = city_graffiti_draw(run, layer, i, 1u);
    let g = city_graffiti_draw(run, layer, i, 2u);
    let c = city_graffiti_draw(run, layer, i, 3u);
    var k: CityGraffitiInk;
    // A piece's fills from the hues, a throw-up's from the fills.
    k.fill = select(city_graffiti_fill(e.y), city_graffiti_hue(e.y), piece);
    k.fill2 = select(city_graffiti_fill(c.y), city_graffiti_hue(c.y), piece);
    k.fill2 = select(k.fill, k.fill2, c.x < 0.5);
    k.line = city_graffiti_line(e.z);
    k.splash = city_graffiti_hue(g.z);
    let fade = select(1.0, mix(CITY_GRAFFITI_FADE_LO, 1.0, c.w),
                      city_graffiti_is_old(layer));
    k.fill = k.fill * fade;
    k.fill2 = k.fill2 * fade;
    k.line = k.line * fade;
    k.splash = k.splash * fade;
    k.shine = CITY_GRAFFITI_WHITE * fade;
    return k;
}

// A lettering element's parts at q (units of h, from its box's start and
// its foot), FINE, their edges filtered over f (the footprint, in h):
// (fill, shine, outline, drip) and a piece's splash. The fill includes the
// shine painted over it.
struct CityGraffitiParts {
    fill: f32,
    shine: f32,
    line: f32,
    drip: f32,
    splash: f32,
}

// Glyph k's frame: its centre, half sizes and letterform, and its shine
// and drip draws (x: shine or not, y: drip or not, z: where, w: how long).
struct CityGraffitiGlyph {
    c: vec2<f32>,
    b: vec2<f32>,
    turn: f32,
    kind: i32,
    sd: vec4<f32>,
}

fn city_graffiti_glyph_at(l: CityGraffitiLetters, k: i32)
        -> CityGraffitiGlyph {
    let g = city_rand4(pcg2d(l.gs + vec2<u32>(u32(k) * 0x27d4eb2fu,
                                              0x165667b1u)));
    var o: CityGraffitiGlyph;
    // Fat round letters about as wide as they are tall, crowding their
    // neighbours (overlapping them by a third to two thirds), each its own
    // height, tilted either way by up to 7 degrees.
    let hy = 0.29 * (0.94 + 0.12 * g.z);
    o.c = vec2<f32>(CITY_GRAFFITI_PAD + (f32(k) + 0.5) * l.gw,
                    0.535 + 0.04 * (g.y - 0.5));
    o.b = vec2<f32>(0.5 * l.gw * (1.35 + 0.35 * g.x), hy);
    o.turn = 0.24 * (g.w - 0.5);
    o.kind = i32(8.0 * fract(g.x * 13.17 + g.y * 7.91));
    o.sd = city_rand4(pcg2d(l.gs + vec2<u32>(u32(k) * 0x27d4eb2fu,
                                             0x61c88647u)));
    return o;
}

// The point q in glyph o's own frame: about its centre, turned, slanted.
fn city_graffiti_glyph_p(q: vec2<f32>, o: CityGraffitiGlyph,
                         slant: f32) -> vec2<f32> {
    let cs = vec2<f32>(cos(o.turn), sin(o.turn));
    var p = q - o.c;
    p = vec2<f32>(cs.x * p.x + cs.y * p.y, -cs.y * p.x + cs.x * p.y);
    p.x = p.x - slant * p.y;
    return p;
}

// The letters drawn one over the next, as a writer paints them: each its
// fill inside its outline, over the letters before it — which is what
// draws the outline lines inside a throw-up. A point sees its own glyph's
// cell and the two beside it (the overlap reaches no further). The outline
// is the letter's own swept a little down and right, so it lies thicker on
// that side, the way a writer's 3-D drop sits; the fill is a little uneven
// (one coat of spray), and round a throw-up hangs a faint overspray of the
// outline's colour. Edges carry the paint's own softness.
const CITY_GRAFFITI_DROP: vec2<f32> = vec2<f32>(0.035, -0.035);

fn city_graffiti_letter_parts(q: vec2<f32>, f_in: f32,
                              l: CityGraffitiLetters) -> CityGraffitiParts {
    let f = city_graffiti_soft(f_in, l.h);
    let k = clamp(i32(floor((q.x - CITY_GRAFFITI_PAD) / l.gw)), 0, l.n - 1);
    var pt: CityGraffitiParts;
    pt.fill = 0.0;
    pt.shine = 0.0;
    pt.line = 0.0;
    pt.drip = 0.0;
    pt.splash = 0.0;
    var dmin = 1e3;
    // Three glyphs, the count at run time so the loop stays rolled (see
    // city_graffiti_paint); a letter's height is never negative.
    let n_near = select(3, 0, l.h < 0.0);
    for (var j: i32 = 0; j < n_near; j = j + 1) {
        // Left to right, or right to left: the later letter on top.
        let kk = select(k + 1 - j, k - 1 + j, l.left_up);
        if (kk < 0 || kk >= l.n) {
            continue;
        }
        let o = city_graffiti_glyph_at(l, kk);
        let p = city_graffiti_glyph_p(q, o, l.slant);
        let d = city_graffiti_glyph(p, o.b, o.kind, l.cut);
        let ds = min(d, city_graffiti_glyph(
            city_graffiti_glyph_p(q - CITY_GRAFFITI_DROP, o, l.slant), o.b,
            o.kind, l.cut));
        dmin = min(dmin, ds);
        // One coat: the fill's cover 0.82-1 across the letter, what is under
        // it showing through the rest.
        let coat = 0.91 + 0.09 * sin(11.0 * q.x + 6.2831853 * o.sd.z)
            * sin(9.0 * q.y + 6.2831853 * o.sd.w);
        let fd = city_graffiti_edge(d, f);
        let ring = max(city_graffiti_edge(ds - l.ow, f), fd) - fd;
        let occ = ring + fd * coat;
        // Shine: a short white tick high on the letter's left shoulder.
        var sh = 0.0;
        if (o.sd.x < 0.6) {
            let s0 = vec2<f32>(-0.55 * o.b.x, 0.45 * o.b.y);
            sh = city_graffiti_edge(city_graffiti_capsule(
                p, s0, s0 + vec2<f32>(0.10, 0.07), 0.022), f) * fd * coat;
        }
        pt.fill = pt.fill * (1.0 - occ) + fd * coat;
        pt.shine = pt.shine * (1.0 - occ) + sh;
        pt.line = pt.line * (1.0 - occ) + ring;
    }
    let whole = city_graffiti_edge(dmin - l.ow, f);
    let outer = city_graffiti_edge(dmin - l.ow - 0.03, f) - whole;
    if (l.piece) {
        // A piece's second outline: a band of splash colour round the whole.
        pt.splash = outer;
    } else {
        // A throw-up's overspray: the outline colour, faint, just outside.
        pt.line = pt.line + 0.15 * outer;
    }
    // A drip: from under glyph k's fill straight down, outside the letters.
    let o = city_graffiti_glyph_at(l, k);
    if (o.sd.y < 0.5) {
        let xd = o.c.x + 0.7 * (o.sd.z - 0.5) * o.b.x;
        let foot = o.c.y - o.b.y;
        let dl = 0.04 + 0.10 * o.sd.w;
        pt.drip = city_graffiti_span(q.x, f, xd - 0.008, xd + 0.008)
            * city_graffiti_span(q.y, f, foot - l.ow - dl, foot)
            * (1.0 - whole);
    }
    return pt;
}

// A stroke of half-width w at signed distance s across it: the
// footprint's overlap with the stroke, exact for a straight stroke.
fn city_graffiti_stroke(s: f32, f: f32, w: f32) -> f32 {
    return city_graffiti_span(s, f, -w, w);
}

// An ellipse's stroke (centre c, radii r): signed distance across it,
// first order in the radial error (good for a pen stroke on a bowl).
fn city_graffiti_ring(p: vec2<f32>, c: vec2<f32>, r: vec2<f32>) -> f32 {
    return (length((p - c) / r) - 1.0) * min(r.x, r.y);
}

// Letter k of a tag (cells cw wide from x = 0.05, n of them), distance from
// its stroke's centre line in units of h, at p (the tag's point, unslanted).
fn city_graffiti_tag_letter(p: vec2<f32>, k: i32, n: i32, cw: f32,
                            base: f32, xh: f32, w: vec4<f32>) -> f32 {
    let x0 = 0.05 + f32(k) * cw;
    // Each letter its own size (0.8-1.2) and off the line by up to 0.06 h.
    let jit = city_rand4(vec2<u32>(bitcast<u32>(w.z) ^ (u32(k) * 0x9e3779b9u),
                                   0x85ebca6bu));
    let sc = 0.8 + 0.4 * jit.x;
    let lc = (p - vec2<f32>(x0, base + 0.12 * (jit.y - 0.5))) / sc;
    let kind = i32(9.0 * fract(w.z * 7.13 + f32(k) * 0.618));
    var s = 1e3;
    switch kind {
        case 0: {
            s = city_graffiti_capsule(lc, vec2<f32>(0.4 * cw, 0.0),
                                      vec2<f32>(0.4 * cw, 0.56), 0.0);
        }
        case 1: {
            s = abs(city_graffiti_ring(lc, vec2<f32>(0.5 * cw, 0.5 * xh),
                                       vec2<f32>(0.4 * cw, 0.5 * xh)));
        }
        case 2: {
            let arch = abs(city_graffiti_ring(
                lc, vec2<f32>(0.5 * cw, 0.35 * xh),
                vec2<f32>(0.4 * cw, 0.65 * xh)));
            s = select(1e3, arch, lc.y > 0.35 * xh);
            s = min(s, city_graffiti_capsule(lc, vec2<f32>(0.1 * cw, 0.0),
                                             vec2<f32>(0.1 * cw, 0.35 * xh),
                                             0.0));
            s = min(s, city_graffiti_capsule(lc, vec2<f32>(0.9 * cw, 0.0),
                                             vec2<f32>(0.9 * cw, 0.35 * xh),
                                             0.0));
        }
        case 3: {
            s = min(abs(city_graffiti_ring(lc, vec2<f32>(0.4 * cw, 0.5 * xh),
                                           vec2<f32>(0.32 * cw, 0.5 * xh))),
                    city_graffiti_capsule(lc, vec2<f32>(0.8 * cw, 0.0),
                                          vec2<f32>(0.8 * cw, 0.54), 0.0));
        }
        case 4: {
            s = min(city_graffiti_capsule(lc, vec2<f32>(0.05 * cw, 0.0),
                                          vec2<f32>(0.95 * cw, xh), 0.0),
                    city_graffiti_capsule(lc, vec2<f32>(0.05 * cw, xh),
                                          vec2<f32>(0.95 * cw, 0.0), 0.0));
        }
        case 5: {
            s = min(min(city_graffiti_capsule(lc, vec2<f32>(0.9 * cw, xh),
                                              vec2<f32>(0.1 * cw, xh), 0.0),
                        city_graffiti_capsule(lc, vec2<f32>(0.1 * cw, xh),
                                              vec2<f32>(0.9 * cw, 0.0), 0.0)),
                    city_graffiti_capsule(lc, vec2<f32>(0.9 * cw, 0.0),
                                          vec2<f32>(0.1 * cw, 0.0), 0.0));
        }
        case 6: {
            // e: a bowl open at the lower right, its bar across the middle.
            let bowl = abs(city_graffiti_ring(
                lc, vec2<f32>(0.5 * cw, 0.5 * xh), vec2<f32>(0.4 * cw, 0.5 * xh)));
            s = min(select(bowl, 1e3, lc.x > 0.6 * cw && lc.y < 0.4 * xh),
                    city_graffiti_capsule(lc, vec2<f32>(0.12 * cw, 0.5 * xh),
                                          vec2<f32>(0.88 * cw, 0.5 * xh), 0.0));
        }
        case 7: {
            // r: a stem and a short hook off its top.
            let hook = abs(city_graffiti_ring(
                lc, vec2<f32>(0.55 * cw, 0.6 * xh), vec2<f32>(0.4 * cw, 0.4 * xh)));
            s = min(city_graffiti_capsule(lc, vec2<f32>(0.15 * cw, 0.0),
                                          vec2<f32>(0.15 * cw, xh), 0.0),
                    select(1e3, hook, lc.y > 0.6 * xh && lc.x < 0.8 * cw));
        }
        default: {
            // w: four strokes down and up.
            s = min(min(city_graffiti_capsule(lc, vec2<f32>(0.0, xh),
                                              vec2<f32>(0.25 * cw, 0.0), 0.0),
                        city_graffiti_capsule(lc, vec2<f32>(0.25 * cw, 0.0),
                                              vec2<f32>(0.5 * cw, 0.7 * xh),
                                              0.0)),
                    min(city_graffiti_capsule(lc, vec2<f32>(0.5 * cw, 0.7 * xh),
                                              vec2<f32>(0.75 * cw, 0.0), 0.0),
                        city_graffiti_capsule(lc, vec2<f32>(0.75 * cw, 0.0),
                                              vec2<f32>(cw, xh), 0.0)));
        }
    }
    // The join along the baseline to the next letter.
    if (k < n - 1) {
        s = min(s, city_graffiti_capsule(lc, vec2<f32>(0.85 * cw, 0.0),
                                         vec2<f32>(1.1 * cw, 0.0), 0.0));
    }
    return s * sc;
}

// A tag at q (units of h, x from its box's start, y up from its foot),
// FINE: a handstyle. One stroke 0.06-0.10 h wide, swelling and thinning
// with the can, writes a word of cells 0.30-0.42 h wide, crowded, each one
// of nine letters, each its own size and height — a stem with an ascender
// (l, t), a bowl (o), an arch (n), a bowl and stem (a, d), a cross (k, x),
// a zig (s, z), an open bowl and bar (e), a stem and hook (r), a w — on a
// baseline at a fifth of the box with an x-height of a
// third, slanted, joined at the baseline; half the tags underline
// themselves with a swash. `w` holds its draws.
fn city_graffiti_tag(q: vec2<f32>, f: f32, len: f32, w: vec4<f32>) -> f32 {
    let slant = 0.15 + 0.35 * w.x;
    let base = 0.20;
    let xh = 0.34;
    // The can's pressure and speed: the stroke 0.06-0.10 h wide along it.
    let half = 0.04 * (0.75 + 0.25 * sin(9.0 * q.x + 6.2831853 * w.y));
    let p = vec2<f32>(q.x - slant * (q.y - base), q.y);
    let cw = 0.30 + 0.12 * w.y;
    // As many cells as fit with the slant's lean at the ascenders' tops.
    let n = max(i32(floor((len - 0.25 - 0.8 * slant) / cw)), 1);
    // The letter under p and the two beside it: the slant, the joins and the
    // filter all carry a stroke across a cell's edge.
    let k = clamp(i32(floor((p.x - 0.05) / cw)), 0, n - 1);
    var s = 1e3;
    for (var kk: i32 = max(k - 1, 0); kk <= min(k + 1, n - 1); kk = kk + 1) {
        s = min(s, city_graffiti_tag_letter(p, kk, n, cw, base, xh, w));
    }
    var c = city_graffiti_stroke(s, f, half);
    if (w.w < 0.5) {
        // The swash: a long shallow curve under the word.
        let xs = q.x / len;
        let ys = 0.08 + 0.06 * sin(3.14159265 * xs) - 0.05 * xs;
        c = max(c, city_graffiti_stroke(q.y - ys, f, 0.03)
                   * city_graffiti_span(q.x, f, 0.05, len - 0.05));
    }
    // The junctions' loss to the filter, put back (CITY_GRAFFITI_TAG_LOSS).
    return min(c / (1.0 - CITY_GRAFFITI_TAG_LOSS * f * f), 1.0);
}

// A buff patch at q (units of h, x from its box's start, y from its foot),
// FINE: a rounded box with a ragged roller edge.
fn city_graffiti_buff(q: vec2<f32>, f: f32, len: f32, w: vec4<f32>) -> f32 {
    let p = q - vec2<f32>(0.5 * len, 0.5);
    let rag = 0.012 * (sin(37.0 * q.y + 6.2831853 * w.x)
                       + sin(23.0 * q.x + 6.2831853 * w.y));
    let d = city_graffiti_rbox(p, vec2<f32>(0.5 * len - 0.03, 0.47), 0.10) + rag;
    return city_graffiti_edge(d, f);
}

// --- one layer --------------------------------------------------------------

// Draw `which` (0-3) of the element in slot i of a layer.
fn city_graffiti_draw(run: CityGraffitiRun, layer: i32, i: i32,
                      which: u32) -> vec4<f32> {
    return city_rand4(pcg2d(run.seed ^ vec2<u32>(
        u32(layer) * 0x9e3779b9u + which * 0x632be5abu,
        u32(i) * 0x85ebca6bu + which * 0x1b873593u + 0x27d4eb2fu)));
}

// A buff patch's or a tag's draws: height h, box length (in h), box start
// (m), foot (m), the shape's four draws, and the buff's tone or the tag's
// ink. d = (presence, height, length, place), e = (tone, ink, a tag's
// height on the wall, -), w the shape's own.
struct CityGraffitiMark {
    h: f32,
    len: f32,
    x0: f32,
    foot: f32,
    w: vec4<f32>,
    tone: f32,
    ink: vec3<f32>,
}

fn city_graffiti_mark_at(run: CityGraffitiRun, layer: i32, i: i32)
        -> CityGraffitiMark {
    let sl = city_graffiti_lattice(run, layer);
    let d = city_graffiti_draw(run, layer, i, 0u);
    let e = city_graffiti_draw(run, layer, i, 1u);
    var m: CityGraffitiMark;
    if (layer == CITY_GRAFFITI_BUFF_LAYER) {
        m.h = mix(CITY_GRAFFITI_BUFF_H0, CITY_GRAFFITI_BUFF_H1, d.y);
        m.len = mix(CITY_GRAFFITI_BUFF_A0, CITY_GRAFFITI_BUFF_A1, d.z);
        m.foot = CITY_GRAFFITI_BUFF_FOOT;
    } else {
        m.h = mix(CITY_GRAFFITI_TAG_H0, CITY_GRAFFITI_TAG_H1, d.y);
        m.len = mix(CITY_GRAFFITI_TAG_A0, CITY_GRAFFITI_TAG_A1, d.z);
        m.foot = mix(CITY_GRAFFITI_TAG_C0, CITY_GRAFFITI_TAG_C1, e.z)
            - 0.5 * m.h;
    }
    m.x0 = sl.y + f32(i) * sl.x + (sl.x - m.len * m.h) * d.w;
    m.w = city_graffiti_draw(run, layer, i, 2u);
    m.tone = 0.85 + 0.40 * e.x;
    m.ink = city_graffiti_ink(e.y);
    return m;
}

// A buff patch's or a tag's cover at q (units of h, from its box's start
// and its foot), FINE.
fn city_graffiti_mark_cover(layer: i32, q: vec2<f32>, f: f32,
                            m: CityGraffitiMark) -> f32 {
    let fs = city_graffiti_soft(f, m.h);
    if (layer == CITY_GRAFFITI_BUFF_LAYER) {
        return city_graffiti_buff(q, fs, m.len, m.w);
    }
    return city_graffiti_tag(q, fs, m.len, m.w);
}

// Slot i's element of `layer` at (u, v) with the footprint fuv: FINE (its
// strokes, edges filtered over the footprint) while it stands more than
// eight pixels tall, MID (its box at its expected density) once it is under
// four, a blend between (a tag: twenty and under seven). A
// filtered straight edge keeps its area at any footprint; curved and
// crossing strokes drift as the filter widens (measured: letters and buff
// under 1.5% by MID1; tags' junctions lose more, which their FINE puts
// back), which is where the hand-over sits — by then the letterforms are
// a few pixels and the box at their mean is what a pixel can show.
fn city_graffiti_element(run: CityGraffitiRun, layer: i32, i: i32,
                         u: f32, v: f32, fuv: vec2<f32>, base: f32,
                         p: f32) -> CityGraffitiCover {
    var out: CityGraffitiCover;
    out.a = 0.0;
    out.ac = vec3<f32>(0.0);
    out.at = 0.0;
    if (city_graffiti_draw(run, layer, i, 0u).x >= p) {
        return out;
    }
    let fm = max(fuv.x, fuv.y);
    if (city_graffiti_is_letters(layer)) {
        let l = city_graffiti_letters_at(run, layer, i);
        let q = vec2<f32>(u - l.x0, v - base) / l.h;
        let f = fm / l.h;
        if (q.x < -f || q.x > l.len + f || q.y < -f || q.y > 1.0 + f) {
            return out;
        }
        let w_mid = smoothstep(CITY_GRAFFITI_MID0, CITY_GRAFFITI_MID1, f);
        var pt = CityGraffitiParts(0.0, 0.0, 0.0, 0.0, 0.0);
        if (w_mid < 1.0) {
            pt = city_graffiti_letter_parts(q, f, l);
        }
        // The colours after the shapes (CityGraffitiInk).
        let k = city_graffiti_letters_ink(run, layer, i, l.piece);
        if (w_mid < 1.0) {
            out.a = (1.0 - w_mid) * (pt.fill + pt.line + pt.drip + pt.splash);
            // The fill's two tones either side of the split, the drips the
            // lower one's.
            let lo = city_graffiti_span(q.y, f, -1.0, l.split);
            let fq = mix(k.fill, k.fill2, lo);
            out.ac = (1.0 - w_mid)
                * ((pt.fill - pt.shine) * fq + pt.drip * k.fill2
                   + pt.shine * k.shine + pt.line * k.line
                   + pt.splash * k.splash);
        }
        if (w_mid > 0.0) {
            // MID: the box at its expected density, n glyphs' worth.
            // The paint in its band (city_graffiti_letter_share), over the
            // footprint's height.
            let band = (city_graffiti_letter_share((v + 0.5 * fuv.y - base)
                                                   / l.h)
                        - city_graffiti_letter_share((v - 0.5 * fuv.y - base)
                                                     / l.h)) * l.h / fuv.y;
            let cov = w_mid
                * city_graffiti_span(u, fuv.x, l.x0, l.x0 + l.len * l.h)
                * band * f32(l.n) / l.len;
            let splash = select(0.0, CITY_GRAFFITI_A_SPLASH, l.piece);
            out.a = out.a + cov * (CITY_GRAFFITI_A_FILL + CITY_GRAFFITI_A_LINE
                                   + CITY_GRAFFITI_A_DRIP + splash);
            out.ac = out.ac
                + cov * ((CITY_GRAFFITI_A_FILL - CITY_GRAFFITI_A_SHINE
                          + CITY_GRAFFITI_A_DRIP) * 0.5 * (k.fill + k.fill2)
                         + CITY_GRAFFITI_A_SHINE * k.shine
                         + CITY_GRAFFITI_A_LINE * k.line + splash * k.splash);
        }
        return out;
    }
    let m = city_graffiti_mark_at(run, layer, i);
    let q = vec2<f32>(u - m.x0, v - m.foot) / m.h;
    let f = fm / m.h;
    if (q.x < -f || q.x > m.len + f || q.y < -f || q.y > 1.0 + f) {
        return out;
    }
    let w_mid = select(
        smoothstep(CITY_GRAFFITI_TAG_MID0, CITY_GRAFFITI_TAG_MID1, f),
        smoothstep(CITY_GRAFFITI_MID0, CITY_GRAFFITI_MID1, f), layer == CITY_GRAFFITI_BUFF_LAYER);
    var c = 0.0;
    if (w_mid < 1.0) {
        c = (1.0 - w_mid) * city_graffiti_mark_cover(layer, q, f, m);
    }
    if (w_mid > 0.0) {
        c = c + w_mid * city_graffiti_span(u, fuv.x, m.x0, m.x0 + m.len * m.h)
            * city_graffiti_span(v, fuv.y, m.foot, m.foot + m.h)
            * select(CITY_GRAFFITI_RHO_TAG, CITY_GRAFFITI_RHO_BUFF, layer == CITY_GRAFFITI_BUFF_LAYER);
    }
    out.a = c;
    if (layer == CITY_GRAFFITI_BUFF_LAYER) {
        out.at = c * m.tone;
    } else {
        out.ac = c * m.ink;
    }
    return out;
}

// A layer's expectation along the run (FAR) at height v, over a footprint
// fv tall: the slot odds, the element's expected painted length at each
// height (city_graffiti_profile) over the slot, the colours' means, and the
// share of the run its whole slots cover.
fn city_graffiti_far(run: CityGraffitiRun, layer: i32, v: f32, fv: f32,
                     base: f32, p: f32) -> CityGraffitiCover {
    var out: CityGraffitiCover;
    // @domain: run.len > 0 (city_graffiti_paint)
    let k = p * city_graffiti_lattice(run, layer).z / run.len;
    if (city_graffiti_is_letters(layer)) {
        // Painted length at v: n A h p((v - base) / h) for each element,
        // over the footprint's height (city_graffiti_letter_phi).
        // @domain: fv > 0, a footprint
        let prof = CITY_GRAFFITI_GLYPHS_MEAN
            * (city_graffiti_letter_phi(v + 0.5 * fv - base,
                                        CITY_GRAFFITI_LETTER_H0,
                                        CITY_GRAFFITI_LETTER_H1)
               - city_graffiti_letter_phi(v - 0.5 * fv - base,
                                          CITY_GRAFFITI_LETTER_H0,
                                          CITY_GRAFFITI_LETTER_H1)) / fv;
        let q = select(0.0, CITY_GRAFFITI_PIECE_FRAC, city_graffiti_is_old(layer));
        let fade = select(1.0, CITY_GRAFFITI_FADE_MEAN, city_graffiti_is_old(layer));
        let fill_mean = mix(CITY_GRAFFITI_FILL_MEAN, CITY_GRAFFITI_HUE_MEAN, q);
        out.a = k * prof * (CITY_GRAFFITI_A_FILL + CITY_GRAFFITI_A_LINE
                            + CITY_GRAFFITI_A_DRIP + q * CITY_GRAFFITI_A_SPLASH);
        out.ac = k * prof * fade
            * ((CITY_GRAFFITI_A_FILL - CITY_GRAFFITI_A_SHINE
                + CITY_GRAFFITI_A_DRIP) * fill_mean
               + CITY_GRAFFITI_A_SHINE * CITY_GRAFFITI_WHITE
               + CITY_GRAFFITI_A_LINE * CITY_GRAFFITI_LINE_MEAN
               + q * CITY_GRAFFITI_A_SPLASH * CITY_GRAFFITI_HUE_MEAN);
        out.at = 0.0;
        return out;
    }
    if (layer == CITY_GRAFFITI_BUFF_LAYER) {
        // Painted length at v: rho L = rho (L/h) h; E[L/h] = 1.4.
        let prof = 0.5 * (CITY_GRAFFITI_BUFF_A0 + CITY_GRAFFITI_BUFF_A1)
            * city_graffiti_profile(v, fv, CITY_GRAFFITI_BUFF_FOOT,
                                    CITY_GRAFFITI_BUFF_H0,
                                    CITY_GRAFFITI_BUFF_H1);
        out.a = k * prof * CITY_GRAFFITI_RHO_BUFF;
        out.ac = vec3<f32>(0.0);
        out.at = out.a * 1.05;           // E[tone]
        return out;
    }
    let prof = 0.5 * (CITY_GRAFFITI_TAG_A0 + CITY_GRAFFITI_TAG_A1)
        * city_graffiti_profile_spread(v, fv, CITY_GRAFFITI_TAG_C0,
                                       CITY_GRAFFITI_TAG_C1,
                                       CITY_GRAFFITI_TAG_H0,
                                       CITY_GRAFFITI_TAG_H1);
    out.a = k * prof * CITY_GRAFFITI_RHO_TAG;
    out.ac = out.a * CITY_GRAFFITI_INK_MEAN;
    out.at = 0.0;
    return out;
}

// One layer at (u, v): the elements of the slots the footprint reaches
// (city_graffiti_element), giving way to the layer's expectation (FAR) as
// the footprint passes a quarter to three quarters of a slot.
fn city_graffiti_layer(run: CityGraffitiRun, layer: i32, u: f32, v: f32,
                       fuv: vec2<f32>, base: f32, level: i32)
        -> CityGraffitiCover {
    let p = city_graffiti_slot_p(layer, level);
    var out: CityGraffitiCover;
    out.a = 0.0;
    out.ac = vec3<f32>(0.0);
    out.at = 0.0;
    if (p <= 0.0) {
        return out;
    }
    let sl = city_graffiti_lattice(run, layer);
    // The slots run along u: the footprint's extent along u decides (its
    // height is the profile's to filter, city_graffiti_far).
    let w_far = smoothstep(0.25 * sl.x, 0.75 * sl.x, fuv.x);
    if (w_far < 1.0) {
        let n_slots = i32(sl.z);
        let i = i32(floor((u - sl.y) / sl.x));
        for (var j: i32 = i - 1; j <= i + 1; j = j + 1) {
            let s0 = sl.y + f32(j) * sl.x;
            if (j >= 0 && j < n_slots && u + 0.5 * fuv.x > s0
                && u - 0.5 * fuv.x < s0 + sl.x) {
                let c = city_graffiti_element(run, layer, j, u, v, fuv, base,
                                              p);
                out.a = out.a + c.a;
                out.ac = out.ac + c.ac;
                out.at = out.at + c.at;
            }
        }
    }
    if (w_far > 0.0) {
        let c = city_graffiti_far(run, layer, v, fuv.y, base, p);
        out.a = mix(out.a, c.a, w_far);
        out.ac = mix(out.ac, c.ac, w_far);
        out.at = mix(out.at, c.at, w_far);
    }
    return out;
}

// The run's level: 0 clean, 1 light, 2 heavy; and its baseline, where its
// letters stand (m above the painter's floor).
fn city_graffiti_level(run: CityGraffitiRun) -> vec2<f32> {
    let r = city_rand4(pcg2d(run.seed ^ vec2<u32>(0x7f4a7c15u, 0x94d049bbu)));
    let odds = city_graffiti_odds(run.kind);
    let level = select(select(0.0, 1.0, r.x < odds.x + odds.y), 2.0,
                       r.x < odds.y);
    return vec2<f32>(level, 0.10 + 0.20 * r.y);
}

// The paint on a run at (u, v): u metres along it, v metres above the
// painter's floor, fuv the pixel's footprint on the surface along u and v
// (city_graffiti_footprint). The owner applies it to the albedo it would
// light: albedo * keep + add.
fn city_graffiti_paint(run: CityGraffitiRun, u: f32, v: f32,
                       fuv: vec2<f32>) -> CityGraffitiPaint {
    var out: CityGraffitiPaint;
    out.keep = 1.0;
    out.add = vec3<f32>(0.0);
    let lv = city_graffiti_level(run);
    // Nothing on a clean run, outside it, or clear of the reach band.
    if (lv.x < 0.5 || run.len <= 0.0 || u < 0.0 || u > run.len
        || v + 0.5 * fuv.y < 0.0 || v - 0.5 * fuv.y > 2.75) {
        return out;
    }
    // The layers' count at run time: a literal bound is unrolled by the
    // driver, eight copies of a layer in the register file of every stage
    // that paints (streetlife's ITERS note). The level is at most 2.
    let n_layers = select(CITY_GRAFFITI_LAYERS, 0, lv.x > 2.5);
    for (var l: i32 = 0; l < n_layers; l = l + 1) {
        let c = city_graffiti_layer(run, l, u, v, fuv, lv.y, i32(lv.x));
        out.keep = out.keep * (1.0 - c.a) + c.at;
        out.add = out.add * (1.0 - c.a) + c.ac;
    }
    return out;
}

// The paint's expectation along a run at height v, over a footprint fv
// tall: every layer's FAR rung, which is what city_graffiti_paint gives once
// the footprint along u passes three quarters of every slot. For an owner
// that needs a run's mean beside the pixel's own paint (a door's share of a
// dock cell), at a fraction of the paint's cost.
fn city_graffiti_mean(run: CityGraffitiRun, v: f32, fv: f32)
        -> CityGraffitiPaint {
    var out: CityGraffitiPaint;
    out.keep = 1.0;
    out.add = vec3<f32>(0.0);
    let lv = city_graffiti_level(run);
    if (lv.x < 0.5 || run.len <= 0.0 || v + 0.5 * fv < 0.0
        || v - 0.5 * fv > 2.75) {
        return out;
    }
    let n_layers = select(CITY_GRAFFITI_LAYERS, 0, lv.x > 2.5);
    for (var l: i32 = 0; l < n_layers; l = l + 1) {
        let p = city_graffiti_slot_p(l, i32(lv.x));
        if (p <= 0.0) {
            continue;
        }
        let c = city_graffiti_far(run, l, v, fv, lv.y, p);
        out.keep = out.keep * (1.0 - c.a) + c.at;
        out.add = out.add * (1.0 - c.a) + c.ac;
    }
    return out;
}

// A pixel's footprint on a surface along its u and v directions (unit
// tangents tu, tv, normal n): the pixel's disc of diameter fp, cast along
// the ray onto the surface — fp sqrt(1 + (t.dir)^2 / (n.dir)^2) along t.
fn city_graffiti_footprint(fp: f32, n: vec3<f32>, dir: vec3<f32>,
                           tu: vec3<f32>, tv: vec3<f32>) -> vec2<f32> {
    let nv = max(abs(dot(n, dir)), 0.05);
    let au = dot(tu, dir) / nv;
    let av = dot(tv, dir) / nv;
    // @domain: both arguments are at least 1
    return fp * vec2<f32>(sqrt(1.0 + au * au), sqrt(1.0 + av * av));
}
// === END CITY GRAFFITI ===
