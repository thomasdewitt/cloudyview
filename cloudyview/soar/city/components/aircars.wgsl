// aircars — flying cars, frozen mid-flow.
//
// The sky layer of a two-layer traffic system: a sibling component parks the
// cars on the ground, these hold the air above the avenues. There is no time
// here (SPEC rule 4), so this is a long-exposure photograph of living
// traffic — every craft is where its hash says it is, on every frame and from
// every camera, and the streams read as streams because of where they are,
// not because they move.
//
// LAYOUT. Traffic follows the avenue lattice (`city_is_avenue`, every 8
// blocks) but flies off to one side of it and well above. Each avenue carries
// two counter-flowing lanes, one over each verge, ~9 m in from the avenue
// centerline; the two lanes belong to the two cells that flank the avenue —
// the cell whose MIN edge is the avenue owns the lane on its own side, the
// cell whose MAX edge is the avenue owns the other. That ownership rule is
// what keeps every craft geometrically inside the cell that draws it, which
// the DDA requires: a box straddling a cell boundary is tested only from one
// side and disappears from the other. It also makes the flow directions fall
// out right-hand — keep to the right of the avenue you are flying up, and the
// +x verge of a north-south avenue runs north.
//
// Altitude is a two-deck system, ~55 m and ~95 m, +-8 m per craft, chosen by
// the craft's own hash — so a lane is a loose braid rather than a wire, and
// the low deck passes under the high one at every crossing. Over ordinary
// blocks a few percent of cells carry a free flyer at 40-130 m, off-lattice
// and on any heading, which is what stops the air from being a pure grid.
//
// Occupancy rides the cascade the way the streetlights do: a downtown avenue
// is a stream, an outskirts lane is three craft in a kilometre. That is
// `cc.density`, already loaded by the core — free.
//
// LIGHT. The Cloudpunk cue is the UNDERGLOW: an emissive belly panel, cyan /
// magenta / amber, which is the only part of a craft big enough to survive
// past a few hundred metres. From the street you read bellies overhead; from
// 400 m the avenues are strings of colored points above the sodium lattice;
// further out the layer is drifting specks. White nose and red tail lamps
// live in the hull's own shading rather than in geometry of their own, and
// dissolve into their face's mean as the footprint outgrows them.
//
// COST. Cheap gates first, in order: the segment's z-extent against the
// layer's [35, 140] m slab (no hash, no arithmetic beyond two multiply-adds);
// the pixel footprint; then the two avenue tests, which are integer. A cell
// with no avenue edge therefore spends exactly one hash draw — the free-flyer
// coin — and returns. At most three craft touch a cell (one x-lane, one
// y-lane, one free flyer; the two x-edge tests are mutually exclusive at
// avenue period 8), and each craft is two boxes, the second of which drops
// out once it is sub-pixel: <= 6 box tests and 6 pcg2d draws worst case,
// 2 pcg2d in the common one.
//
// CLOSE READ (the SDF, 2026-08-20). Thomas flew the city and the craft "are
// not really clearly cars, they are boxes with a few circles" — true, and the
// far read was never the problem. So the boxes stay exactly as they were
// beyond half a metre per pixel, and inside that gate the box test becomes a
// bounding test only: a hit on the padded box hands off to a sphere trace of
// a vehicle SDF in the craft's own frame (SPEC's SDF-in-a-box pattern). The
// SDF is a rounded lifting body — tapering nose, blunt tail, stub sponsons,
// twin rear nacelles, a canopy bubble smooth-minned onto the deck, a dorsal
// fin and a whip antenna, light housings raised at the sponson tips. If the
// trace misses inside the box the craft misses: rays grazing past the curved
// hull are the whole reason it reads as curved rather than as a slab.
//
// Everything the eye uses to call a shape a vehicle rather than a box is in
// the NORMAL, so the shading is written against it: rim brightness at the
// silhouette, a narrow moon sheen that slides along the crown, the underglow
// falling off as the belly turns into the flank, and the canopy seam found as
// the crease between two SDF parts rather than drawn as a stripe. Panel
// lines, intakes and thruster discs are shading bands in the local frame —
// per SPEC they are cheaper there than in geometry, and at wiper-blade
// calibration they are the right size to be features rather than texture.

// --- craft geometry (a compact wedge, 3.8 x 1.9 x 1.1 m) --------------------
const cc_aircars_HALF_L: f32 = 1.90;
const cc_aircars_HALF_W: f32 = 0.95;
const cc_aircars_HULL_H: f32 = 0.70;
const cc_aircars_CANOPY_H: f32 = 0.40;
const cc_aircars_CANOPY_HALF_L: f32 = 1.10;
const cc_aircars_CANOPY_HALF_W: f32 = 0.60;
const cc_aircars_CANOPY_AFT: f32 = 0.45;  // canopy sits aft: the nose tapers

// --- the close-read SDF -----------------------------------------------------
// Local frame: +x along the direction of travel, +y to port, z measured up
// from the hull bottom (so z = 0 is the belly plane and z = HULL_H the deck).
// Every number below is in metres in that frame, and the whole solid lives
// inside |x| <= 1.88, |y| <= 1.06, 0 <= z <= 1.10 — which is what sets the
// padding on the bounding box.
const cc_aircars_SDF_FP: f32 = 0.50;     // hand-off: boxes beyond this
const cc_aircars_SDF_ITERS: i32 = 24;
// Bounding box half-extents, along-travel and across. The hull inside is
// yawed by up to +-YAW, and the box is NOT: 1.88*cos(5) + 1.06*sin(5) = 1.965
// along and 1.88*sin(5) + 1.06*cos(5) = 1.220 across, plus slack.
const cc_aircars_SDF_PAD_L: f32 = 2.04;
const cc_aircars_SDF_PAD_W: f32 = 1.36;
const cc_aircars_SDF_PAD_ZLO: f32 = 0.06;
const cc_aircars_SDF_PAD_ZHI: f32 = 1.18;
const cc_aircars_YAW: f32 = 0.0873;      // +-5 degrees, per craft
// Housing / lens radius for the nav lamps, and where the thruster discs sit.
const cc_aircars_HOUSE_R: f32 = 0.098;
const cc_aircars_HOUSE_Z: f32 = 0.26;    // housing / sponson axis height
const cc_aircars_LENS_OUT: f32 = 0.60;   // how much of the bump is the lens
// The nacelle axis has to sit far enough outboard that the pod stands PROUD
// of the hull: at |y| = 0.66 the pod's own 0.23 m radius put its outer skin
// at 0.89 against a hull half-width of 0.86, so the engines were swallowed
// and the craft had no visible engines at all from the side — the one item of
// the close read that the first pass silently lost. At 0.80 the pod clears
// the flank by ~0.17 m, which is 19 px at 25 m: a pod on a pylon, which is
// what the eye is looking for when it looks for an engine.
const cc_aircars_NAC_Y: f32 = 0.80;      // nacelle axis, |y|
const cc_aircars_NAC_Z: f32 = 0.52;      // nacelle axis height
const cc_aircars_THRUST_R: f32 = 0.15;   // thruster disc radius
// The chine line: belly below it, flank above. The underglow's side strip is
// a band centred on it, |z - CHINE_Z| running LO to HI.
const cc_aircars_CHINE_Z: f32 = 0.145;
const cc_aircars_CHINE_LO: f32 = 0.018;
const cc_aircars_CHINE_HI: f32 = 0.050;

// --- lanes ------------------------------------------------------------------
const cc_aircars_LANE_OFF: f32 = 9.0;    // lane centerline in from the avenue
const cc_aircars_LANE_JIT: f32 = 4.0;    // +- lateral wander
const cc_aircars_ALONG_JIT: f32 = 34.0;  // +- along-lane wander in its slot
const cc_aircars_Z_LOW: f32 = 55.0;
const cc_aircars_Z_HIGH: f32 = 95.0;
const cc_aircars_Z_JIT: f32 = 8.0;
const cc_aircars_HIGH_CUT: f32 = 0.55;   // hash above this -> the high deck
const cc_aircars_LANE_P: f32 = 0.72;     // slot occupancy at full density
const cc_aircars_DENS_LO: f32 = 0.50;    // occupancy scale, outskirts
const cc_aircars_DENS_HI: f32 = 1.35;    // occupancy scale, downtown
// The window the occupancy ramp spans, in block density. Set against the
// tile's own distribution, which is lognormal and brutally skewed — median
// block 0.045, p75 0.10, p90 0.22 — so a ramp calibrated by eye on the
// megatower district leaves the MODAL city with one craft every 300 m of
// lane, which reads as scattered dust rather than as traffic. Anchored at
// the median instead: a typical avenue carries one craft per ~80 m counting
// both directions, downtown roughly one per 50, the outskirts one per 300.
const cc_aircars_DENS_START: f32 = 0.005;
const cc_aircars_DENS_FULL: f32 = 0.070;
// Free flyers: off-lattice singles over ordinary blocks.
const cc_aircars_FREE_FRAC: f32 = 0.07;
const cc_aircars_FREE_Z_LO: f32 = 40.0;
const cc_aircars_FREE_Z_HI: f32 = 130.0;
const cc_aircars_FREE_MARGIN: f32 = 12.0;
// Segment z-gate: the whole air layer lives in here, hull bottom to canopy.
const cc_aircars_Z_MIN: f32 = 35.0;
const cc_aircars_Z_MAX: f32 = 140.0;

// --- light ------------------------------------------------------------------
// Yardsticks from SPEC rule 5: lit window 3.5, storefront 2.2, lamp pool 0.7.
// The underglow sits between a lamp pool and a storefront — a lit panel, not
// a source you look into — and the nav lamps are point-bright.
const cc_aircars_GLOW_RAD: f32 = 2.5;
// The panel is inset in the belly, not the whole belly: seen from underneath
// at street range a craft has to read as a dark object carrying a light, not
// as a floating rectangle of light. The border keeps a quarter of the
// radiance (the panel spilling onto its own frame), which is also what keeps
// the far-field mean near the full value where the inset is sub-pixel.
const cc_aircars_PANEL_A: f32 = 1.62;    // panel half-length (of 1.90)
const cc_aircars_PANEL_W: f32 = 0.79;    // panel half-width (of 0.95)
const cc_aircars_PANEL_RIM: f32 = 0.25;  // what the border keeps
const cc_aircars_LAMP_RAD: f32 = 6.0;
const cc_aircars_LAMP_R: f32 = 0.12;     // nav-lamp dot radius (m)
const cc_aircars_CANOPY_RAD: f32 = 0.40;
const cc_aircars_HULL_FILL: f32 = 0.05;  // hull albedo against the skyglow
// The panel does not end at the belly seam: it wraps the lower flanks as a
// skirt and dies out toward the shoulder. This is what carries the layer at
// distance — from anywhere but directly underneath, the skirt is the only
// piece of a craft with the craft's own color on it, and from 400 m up a
// lane the strings of light you read are skirts, not bellies. Its mean over
// the 0.7 m flank is what a sub-pixel craft delivers, so it is set against
// the window yardstick (3.5) rather than against the belly's own radiance.
const cc_aircars_SKIRT: f32 = 0.86;
const cc_aircars_SKIRT_LO: f32 = 0.10;   // fully lit below this height (m)
const cc_aircars_SKIRT_HI: f32 = 0.62;   // gone by this one
// Dorsal running light: the same color, dimmer, on the hull's back. Without
// it a craft seen from above is a black chip on a bright street, and the
// whole aerial read of the layer goes with it.
const cc_aircars_DORSAL: f32 = 0.76;
// Close-read light, all of it keyed to the SDF normal. A dark hull against a
// dark city has exactly three ways to show its shape: the silhouette rim, a
// specular slid across the crown by the moon, and the way its own underglow
// dies as the belly turns into the flank. RIM is the loudest of the three and
// the one that does the work — it is the lit edge that says "curved".
const cc_aircars_RIM: f32 = 0.55;
const cc_aircars_HULL_DIFF: f32 = 0.80;  // moonlight on dark metal
const cc_aircars_BOUNCE: f32 = 0.016;    // the panel's spill onto its own hull
const cc_aircars_BAR: f32 = 0.45;        // the chine light bar
const cc_aircars_CABIN: f32 = 0.22;      // how much interior the glass shows
const cc_aircars_SHEEN: f32 = 0.34;      // moon glint, hull
const cc_aircars_GLASS: f32 = 0.55;      // moon glint, canopy (tighter, harder)
const cc_aircars_UPGLOW: f32 = 0.020;    // street uplight caught underneath
const cc_aircars_SEAM_DARK: f32 = 0.80;  // canopy-to-hull seam band
const cc_aircars_LINE_DARK: f32 = 0.70;  // panel lines
const cc_aircars_INTAKE_DARK: f32 = 0.85;
const cc_aircars_THRUST: f32 = 1.75;     // thruster disc radiance
const cc_aircars_THRUST_HALO: f32 = 0.32; // what the nacelle cup keeps

// --- LOD --------------------------------------------------------------------
// Footprint (m/px) where each piece of detail hands over to its own mean. The
// canopy is a 2.2 m silhouette bump on a 3.8 m body: it stops paying for
// itself first. The nav lamps are 0.24 m dots: they fade into the mean
// radiance of the face they sit on, so a distant craft keeps its light and
// loses only the resolution — which is what a long lens does to traffic.
const cc_aircars_CANOPY_FP: f32 = 0.80;
const cc_aircars_LAMP_LOD_START: f32 = 0.35;
const cc_aircars_LAMP_LOD_FULL: f32 = 1.20;
// Where the craft stops being traced at all: 6 m/px puts a whole craft at
// two thirds of a pixel. The emission ramps to zero over the run-up to it
// (cc_aircars_FAR_FADE) so the geometry disappears at the moment it stops
// contributing anything, rather than popping out — the tracer's cutoff and
// the shader's mean meet at the same place. At the harness's 960 px / 65 deg
// this threshold sits past CITY_PROP_RANGE and never fires; it exists for
// low-resolution or very wide-angle frames, where it does.
const cc_aircars_FAR_FP: f32 = 6.0;
const cc_aircars_FAR_FADE: f32 = 3.6;    // emission gone by FAR_FP from here
// The core stops calling cell_props at CITY_PROP_RANGE. A hard population
// edge there would be a ring in any long sightline, so the emission tapers
// over the last quarter of the range instead: the far-field mean of a layer
// this sparse IS near zero, and this is that mean approached rather than
// stepped into.
// Kept short deliberately: a craft is about one pixel across out here and the
// population is sparse, so there is no continuous structure for an edge to
// show up in — all the taper has to do is take the last few craft down, and a
// long taper instead eats the whole mid-field of any shallow aerial view,
// which is exactly where the layer is supposed to read.
const cc_aircars_FADE_START: f32 = 0.92;
const cc_aircars_FADE_FULL: f32 = 1.00;

// Per-craft placement draw. One city_rand4 (two pcg2d) carries everything the
// geometry needs; appearance draws its own hash in the shader, where the cost
// is one per hit pixel rather than one per visited cell.
// Seeded by the TILE-WRAPPED cell index (city_tile_cell): a craft belongs
// to its tile coordinate, so live flight and a replayed track — which only
// carries the tile phase — draw the same traffic.
fn cc_aircars_draw(ci: vec2<i32>, lane: i32) -> vec4<f32> {
    let cw = city_tile_cell(ci);
    let l = u32(lane);
    return city_rand4(vec2<u32>(
        bitcast<u32>(cw.x) * 0x9e3779b9u + l * 0x2545f491u + 0x165667b1u,
        bitcast<u32>(cw.y) * 0x85ebca6bu + l * 0xc2b2ae35u + 0x27d4eb2fu));
}

fn cc_aircars_look(ci: vec2<i32>, lane: i32) -> vec4<f32> {
    let cw = city_tile_cell(ci);
    let l = u32(lane);
    return city_rand4(vec2<u32>(
        bitcast<u32>(cw.x) * 0xc2b2ae35u + l * 0x9e3779b9u + 0x51ed270bu,
        bitcast<u32>(cw.y) * 0x27d4eb2fu + l * 0x85ebca6bu + 0xdeadbeefu));
}

// Underglow palette: 60% cyan, 25% magenta, 15% amber. Cyan carries the layer
// because it is the one hue the sodium-and-tungsten ground does not already
// own — the traffic reads as a separate system, not as more windows.
fn cc_aircars_glow_color(d: f32) -> vec3<f32> {
    if (d < 0.60) {
        return vec3<f32>(0.16, 0.90, 1.00);
    }
    if (d < 0.85) {
        return vec3<f32>(1.00, 0.20, 0.70);
    }
    return vec3<f32>(1.00, 0.60, 0.16);
}

// --- SDF primitives ---------------------------------------------------------
// All exact or conservative (never over-estimating distance), which is what
// sphere tracing needs: the ellipsoid is iq's second-order bound and the
// smooth-min only ever shortens, so a step of `d` can never overshoot a
// surface.
fn cc_aircars_sd_rbox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0)))
           + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn cc_aircars_sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn cc_aircars_sd_ellipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-6);
}

fn cc_aircars_sd_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32)
        -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

fn cc_aircars_smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// The vehicle, in its own frame. Returns the combined distance in .x and the
// three parts the shader needs to tell apart in .yzw — hull group, canopy,
// nacelles. One function, so the marcher and the shader can never disagree
// about where the canopy ends; the shader gets the parts for free at the hit
// because it has to evaluate the field there anyway.
//
// Wiper-blade calibration: the features here are the ones a person would name
// looking at the thing — nose, cabin, wing stub, engine pod, fin, aerial,
// light housing. Nothing smaller has geometry; the panel lines and the intake
// slot are shading, which is both cheaper and the right scale.
fn cc_aircars_sdf(q: vec3<f32>) -> vec4<f32> {
    // The port/starboard fold: everything paired is built once in |y|.
    let m = vec3<f32>(q.x, abs(q.y), q.z);

    // Lifting body: a flat rounded slab, a shorter block forward and a sphere
    // at the tip, smooth-minned into a taper rather than modelled as one. The
    // blend radii are deliberately smaller than the parts they join — a big
    // k turns the whole thing back into the lozenge the boxes already were.
    // The body tapers in PLAN as well as in profile: from above a constant-
    // width slab is a suitcase whichever way its nose points. The half-width
    // is a function of x, which costs the field a little Lipschitz slack —
    // hence the 0.98, which keeps every step an under-estimate.
    let wide = 0.66 - 0.16 * smoothstep(-0.20, 1.10, q.x);
    var hull = 0.98 * cc_aircars_sd_rbox(q - vec3<f32>(-0.22, 0.0, 0.36),
                                         vec3<f32>(1.02, wide, 0.14), 0.20);
    // The nose sits LOW and the tail sits high: a wedge in profile. On a
    // symmetric body the eye has no way to tell front from back, and a craft
    // you cannot tell the front of is a pod.
    let nose = cc_aircars_sd_rbox(q - vec3<f32>(1.00, 0.0, 0.29),
                                  vec3<f32>(0.38, 0.20, 0.05), 0.18);
    hull = cc_aircars_smin(hull, nose, 0.18);
    let tip = cc_aircars_sd_sphere(q - vec3<f32>(1.70, 0.0, 0.26), 0.12);
    hull = cc_aircars_smin(hull, tip, 0.16);
    // Blunt tail: the transom the nacelles hang off.
    let tail = cc_aircars_sd_rbox(q - vec3<f32>(-1.34, 0.0, 0.38),
                                  vec3<f32>(0.26, 0.48, 0.09), 0.20);
    hull = cc_aircars_smin(hull, tail, 0.16);
    // The chine: a thin plate a little wider than the hull, running its
    // length. It is the one hard horizontal line on the craft, it splits
    // belly from flank, and the underglow strip lives along it. Without it a
    // rounded hull has no waterline and reads as a pebble.
    let chine = cc_aircars_sd_rbox(q - vec3<f32>(-0.30, 0.0, 0.15),
                                   vec3<f32>(0.88, 0.80, 0.012), 0.055);
    hull = cc_aircars_smin(hull, chine, 0.05);
    // Stub sponsons: the stance. They also carry the nav-light housings out
    // to the widest point, which is where an aircraft puts them.
    // Thin. The first sponson was a 0.3 m bar down the whole flank, and a
    // flank that is one bar with a dot at each end is precisely the box-with-
    // circles this component was sent back to fix. A blade proud of the hull
    // by a few centimetres reads as a wing stub; anything thicker reads as
    // the hull.
    let spon = cc_aircars_sd_rbox(m - vec3<f32>(-0.05, 0.86, 0.26),
                                  vec3<f32>(0.46, 0.055, 0.008), 0.045);
    hull = cc_aircars_smin(hull, spon, 0.05);
    let house = min(
        cc_aircars_sd_sphere(m - vec3<f32>(0.38, 0.93, 0.26),
                             cc_aircars_HOUSE_R),
        cc_aircars_sd_sphere(m - vec3<f32>(-0.52, 0.93, 0.26),
                             cc_aircars_HOUSE_R));
    hull = cc_aircars_smin(hull, house, 0.045);
    // Dorsal fin over the tail, and a whip aerial off the nose deck, offset
    // to port because real hardware is not symmetric.
    // The fin has to grow OUT of the transom, not sit on it. Perched at
    // z = 0.86 with a 0.06 blend it overlapped the tail by three centimetres
    // and rendered as exactly what it was: a rounded slab floating above the
    // hull with daylight under it — the box artifact this whole refit exists
    // to remove, reintroduced at the one place nobody looks. Seated lower,
    // blended at 0.13, and RAKED: the sample's x is pushed forward with
    // height, so the solid leans aft the way a fin does. A shear is not an
    // isometry, so the field is scaled by 1/sqrt(1 + 0.55^2) = 0.876 to stay
    // an under-estimate for the marcher, the same correction the canopy makes.
    let fq = q - vec3<f32>(-1.38, 0.0, 0.74);
    let fin = 0.876 * cc_aircars_sd_rbox(
        vec3<f32>(fq.x + 0.55 * fq.z, fq.y, fq.z),
        vec3<f32>(0.20, 0.010, 0.16), 0.045);
    hull = cc_aircars_smin(hull, fin, 0.13);
    let ant = cc_aircars_sd_capsule(q, vec3<f32>(1.02, 0.22, 0.56),
                                    vec3<f32>(0.96, 0.22, 1.00), 0.028);
    hull = min(hull, ant);

    // Twin nacelles on the rear flanks; their aft caps are the thrusters.
    // Seated aft of the nav housing rather than over it: moved outboard, the
    // pod's 0.23 m radius reached forward far enough to swallow the aft lens
    // and the red lamp came back as a squashed smear. Shorter and further
    // back leaves 0.06 m between pod skin and housing, which the 0.09 blend
    // turns into a pylon root instead of a collision.
    let pods = cc_aircars_sd_rbox(
        m - vec3<f32>(-1.14, cc_aircars_NAC_Y, cc_aircars_NAC_Z),
        vec3<f32>(0.40, 0.045, 0.045), 0.185);
    // Cabin bubble, seated aft of the nose the way the box canopy was — and
    // short enough in x that there is a HOOD in front of it. A bubble that
    // runs into the nose is a fuselage; a bubble with deck fore and aft of it
    // is a cabin, and that is the difference between reading as a car and
    // reading as a pod.
    // The bubble is SHEARED forward with height, which rakes the windshield
    // and gives the cabin a back-slope — the profile everyone reads as a
    // passenger compartment. A shear is not an isometry, so the field is
    // scaled by 1/sqrt(1 + s^2) to stay a valid (under-)estimate for the
    // marcher; s = 0.42 gives 0.922.
    let ck = q - vec3<f32>(-0.28, 0.0, 0.58);
    let canopy = 0.922 * cc_aircars_sd_ellipsoid(
        vec3<f32>(ck.x + 0.42 * ck.z, ck.y, ck.z),
        vec3<f32>(0.76, 0.46, 0.44));

    var d = cc_aircars_smin(hull, pods, 0.09);
    d = cc_aircars_smin(d, canopy, 0.075);
    return vec4<f32>(d, hull, canopy, pods);
}

struct cc_aircars_Craft {
    ok: bool,
    base: vec3<f32>,  // hull bottom, craft centered in xy
    axis: i32,        // 0 = travels along x, 1 = travels along y
    fwd: f32,         // +1 or -1 along that axis
    yaw: f32,         // small per-craft heading offset (rad)
}

fn cc_aircars_no_craft() -> cc_aircars_Craft {
    return cc_aircars_Craft(false, vec3<f32>(0.0), 0, 1.0, 0.0);
}

// The craft's heading as a unit vector in world xy: the lane axis turned by
// the craft's own yaw. Local <-> world is then a rotation about z, so a
// normal transforms with the same two numbers and z is shared outright — the
// shader can read h.normal.z as "up" without converting anything.
fn cc_aircars_frame(v: cc_aircars_Craft) -> vec2<f32> {
    let cd = cos(v.yaw);
    let sd = sin(v.yaw);
    if (v.axis == 0) {
        return v.fwd * vec2<f32>(cd, sd);
    }
    return v.fwd * vec2<f32>(-sd, cd);
}

fn cc_aircars_to_local(f: vec2<f32>, w: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(f.x * w.x + f.y * w.y, f.x * w.y - f.y * w.x, w.z);
}

fn cc_aircars_to_world(f: vec2<f32>, l: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(f.x * l.x - f.y * l.y, f.y * l.x + f.x * l.y, l.z);
}

fn cc_aircars_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

// Where craft `lane` of cell `ci` is, if it exists at all.
//   lane 0 — the x-avenue lane (runs north/south)
//   lane 1 — the y-avenue lane (runs east/west)
//   lane 2 — the free flyer
// Deterministic in (ci, cc, lane) alone, so the shader re-derives a craft from
// its hit kind and the core's own CityCell — nothing has to be smuggled
// through CityHit.
fn cc_aircars_craft(ci: vec2<i32>, cc: CityCell, lane: i32)
        -> cc_aircars_Craft {
    let cell = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cell;

    if (lane == 2) {
        // Free flyers only over the built city; open lots keep empty sky.
        if (!cc.built) {
            return cc_aircars_no_craft();
        }
        let r = cc_aircars_draw(ci, lane);
        if (r.x >= cc_aircars_FREE_FRAC) {
            return cc_aircars_no_craft();
        }
        // r.x is uniform on [0, FREE_FRAC): rescaled it is a free draw, so a
        // heading costs no extra hash.
        let q = r.x / cc_aircars_FREE_FRAC;
        let span = cell - 2.0 * cc_aircars_FREE_MARGIN;
        return cc_aircars_Craft(
            true,
            vec3<f32>(cmin.x + cc_aircars_FREE_MARGIN + r.y * span,
                      cmin.y + cc_aircars_FREE_MARGIN + r.z * span,
                      mix(cc_aircars_FREE_Z_LO, cc_aircars_FREE_Z_HI, r.w)),
            select(0, 1, q > 0.5),
            select(-1.0, 1.0, fract(q * 2.0) > 0.5),
            (fract(r.y * 53.0) - 0.5) * 2.0 * cc_aircars_YAW);
    }

    // Avenue lanes. Exactly one of the two edge tests can fire — the avenue
    // period is 8, so consecutive indices are never both avenues — which is
    // why one lane slot per axis per cell is the whole story.
    let k = select(ci.y, ci.x, lane == 0);
    let lo_edge = city_is_avenue(k);
    if (!lo_edge && !city_is_avenue(k + 1)) {
        return cc_aircars_no_craft();
    }
    let r = cc_aircars_draw(ci, lane);
    // Traffic clusters where the city does.
    let p = cc_aircars_LANE_P * mix(cc_aircars_DENS_LO, cc_aircars_DENS_HI,
                                    smoothstep(cc_aircars_DENS_START,
                                               cc_aircars_DENS_FULL,
                                               cc.density));
    if (r.x >= p) {
        return cc_aircars_no_craft();
    }

    // Lateral seat: LANE_OFF in from whichever edge is the avenue, jittered.
    let jit = (r.z - 0.5) * 2.0 * cc_aircars_LANE_JIT;
    let lat = select(cell - cc_aircars_LANE_OFF + jit,
                     cc_aircars_LANE_OFF + jit, lo_edge);
    let along = 0.5 * cell + (r.y - 0.5) * 2.0 * cc_aircars_ALONG_JIT;
    let z = select(cc_aircars_Z_LOW, cc_aircars_Z_HIGH,
                   r.w > cc_aircars_HIGH_CUT)
            + (fract(r.w * 37.0) - 0.5) * 2.0 * cc_aircars_Z_JIT;

    // Yaw is a free re-draw off r.z, the way the deck jitter is off r.w: no
    // craft flies exactly parallel to its lane, and five degrees is the
    // difference between a formation and traffic.
    let yaw = (fract(r.z * 61.0) - 0.5) * 2.0 * cc_aircars_YAW;

    if (lane == 0) {
        // Runs along y. On the avenue's +x verge (this cell's min edge is the
        // avenue) right-hand traffic heads north.
        return cc_aircars_Craft(
            true, vec3<f32>(cmin.x + lat, cmin.y + along, z),
            1, select(-1.0, 1.0, lo_edge), yaw);
    }
    // Runs along x; the same rule rotated.
    return cc_aircars_Craft(
        true, vec3<f32>(cmin.x + along, cmin.y + lat, z),
        0, select(1.0, -1.0, lo_edge), yaw);
}

// Half-extents in xy for a body of the given along/across half-sizes.
fn cc_aircars_extent(axis: i32, hl: f32, hw: f32) -> vec2<f32> {
    return select(vec2<f32>(hw, hl), vec2<f32>(hl, hw), axis == 0);
}

// The close read: bounding box, then sphere-trace the vehicle inside it.
// Called only when the box is worth more than a few pixels; a miss inside the
// box is a miss, which is what carves the curved silhouette out of the slab.
fn cc_aircars_sdf_hit(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t0: f32, t1: f32, v: cc_aircars_Craft, lane: i32,
                      ci: vec2<i32>, fp: f32) -> CityHit {
    let hx = cc_aircars_extent(v.axis, cc_aircars_SDF_PAD_L,
                               cc_aircars_SDF_PAD_W);
    let bmin = vec3<f32>(v.base.xy - hx,
                         v.base.z - cc_aircars_SDF_PAD_ZLO);
    let bmax = vec3<f32>(v.base.xy + hx,
                         v.base.z + cc_aircars_SDF_PAD_ZHI);
    let sb = city_box_hit(o, inv_dir, bmin, bmax);
    let ta = max(max(sb.x, t0), 0.0);
    let tb = min(sb.y, t1);
    if (sb.x > sb.y || ta > tb) {
        return cc_aircars_miss(ci);
    }

    // Ray into the craft's frame once, rather than the sample point every
    // step: the transform is a rotation about z, so |rd| = |dir| and t is
    // still world distance.
    let f = cc_aircars_frame(v);
    let ro = cc_aircars_to_local(f, o - v.base);
    let rd = cc_aircars_to_local(f, dir);
    // Surface tolerance follows the footprint: converge to well under a pixel
    // and no further. The floor keeps the whip aerial from costing the whole
    // iteration budget at arm's length.
    let eps = clamp(0.30 * fp, 0.0035, 0.06);

    var cc_aircars_t = ta;
    var cc_aircars_found = false;
    for (var cc_aircars_i: i32 = 0; cc_aircars_i < cc_aircars_SDF_ITERS;
         cc_aircars_i = cc_aircars_i + 1) {
        let d = cc_aircars_sdf(ro + cc_aircars_t * rd).x;
        if (d < eps) {
            cc_aircars_found = true;
            break;
        }
        cc_aircars_t = cc_aircars_t + d;
        if (cc_aircars_t > tb) {
            break;
        }
    }
    if (!cc_aircars_found || cc_aircars_t > tb) {
        return cc_aircars_miss(ci);
    }

    // Normal by the four-tap tetrahedron gradient, at the tolerance's own
    // scale so a coarse trace gets a correspondingly smoothed normal.
    let q = ro + cc_aircars_t * rd;
    let hs = max(eps, 0.004);
    // The four tetrahedron taps run as a LOOP, not as four unrolled calls,
    // and that is a performance decision, not a style one. Written out, the
    // four calls inline four more copies of the whole vehicle field into the
    // one kernel that also marches the clouds; the register budget is per
    // kernel, so occupancy collapses for EVERY pixel in the frame whether or
    // not a craft is anywhere near it. Measured on the RTX 5080: the unrolled
    // taps cost +67% on the aerial view, where the nearest craft is 7 km away
    // and the gate can never fire. Rolled, one copy serves all four.
    var cc_aircars_g = vec3<f32>(0.0);
    for (var cc_aircars_k: u32 = 0u; cc_aircars_k < 4u;
         cc_aircars_k = cc_aircars_k + 1u) {
        // The tetrahedron's four sign triples, as bit rows: x on taps 0,3;
        // y on 1,3; z on 2,3.
        let cc_aircars_e = 2.0 * vec3<f32>(
            f32((0x9u >> cc_aircars_k) & 1u),
            f32((0xau >> cc_aircars_k) & 1u),
            f32((0xcu >> cc_aircars_k) & 1u)) - 1.0;
        cc_aircars_g = cc_aircars_g + cc_aircars_e
                       * cc_aircars_sdf(q + cc_aircars_e * hs).x;
    }
    let g = cc_aircars_g;

    return CityHit(true, cc_aircars_t, o + cc_aircars_t * dir,
                   cc_aircars_to_world(f, normalize(g)),
                   300 + 4 * lane + 3, ci);
}

// Nearest hit of one craft against [t0, t1]. The kind encodes both the lane
// (so the shader can find the craft again) and the part that was hit:
// 300 + 4 * lane + part, part 0 = hull side/top, 1 = the underglow panel on
// the belly, 2 = the canopy, 3 = an SDF hit (the close read, whose parts the
// shader recovers from the local position and the field itself).
fn cc_aircars_hit_craft(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                        t0: f32, t1: f32, v: cc_aircars_Craft, lane: i32,
                        ci: vec2<i32>, fp: f32) -> CityHit {
    if (fp < cc_aircars_SDF_FP) {
        return cc_aircars_sdf_hit(o, dir, inv_dir, t0, t1, v, lane, ci, fp);
    }
    let hx = cc_aircars_extent(v.axis, cc_aircars_HALF_L, cc_aircars_HALF_W);
    let hmin = vec3<f32>(v.base.xy - hx, v.base.z);
    let hmax = vec3<f32>(v.base.xy + hx, v.base.z + cc_aircars_HULL_H);
    let sh = city_box_hit(o, inv_dir, hmin, hmax);
    let t_hull = select(1e30, sh.x,
                        sh.x <= sh.y && sh.x > t0 && sh.x <= t1);

    let aft = select(vec2<f32>(0.0, -v.fwd * cc_aircars_CANOPY_AFT),
                     vec2<f32>(-v.fwd * cc_aircars_CANOPY_AFT, 0.0),
                     v.axis == 0);
    let cx = cc_aircars_extent(v.axis, cc_aircars_CANOPY_HALF_L,
                               cc_aircars_CANOPY_HALF_W);
    let kmin = vec3<f32>(v.base.xy + aft - cx, v.base.z + cc_aircars_HULL_H);
    let kmax = vec3<f32>(v.base.xy + aft + cx,
                         v.base.z + cc_aircars_HULL_H + cc_aircars_CANOPY_H);
    // The composer's namespace check is line-based, so even this local wears
    // the prefix.
    var cc_aircars_t_canopy = 1e30;
    if (fp < cc_aircars_CANOPY_FP) {
        let sk = city_box_hit(o, inv_dir, kmin, kmax);
        if (sk.x <= sk.y && sk.x > t0 && sk.x <= t1) {
            cc_aircars_t_canopy = sk.x;
        }
    }

    let t = min(t_hull, cc_aircars_t_canopy);
    if (t >= 1e30) {
        return cc_aircars_miss(ci);
    }
    let is_canopy = cc_aircars_t_canopy < t_hull;
    let bmin = select(hmin, kmin, is_canopy);
    let bmax = select(hmax, kmax, is_canopy);
    let pos = o + t * dir;
    let nrm = city_box_normal(pos, bmin, bmax);
    // The belly of the hull is the underglow panel.
    let part = select(select(0, 1, nrm.z < -0.5), 2, is_canopy);
    return CityHit(true, t, pos, nrm, 300 + 4 * lane + part, ci);
}

fn cc_aircars_nearer(a: CityHit, b: CityHit) -> CityHit {
    if (b.hit && (!a.hit || b.t < a.t)) {
        return b;
    }
    return a;
}

// One lane's contribution: place, then test. Both halves early-out cheaply,
// so an absent lane costs an integer compare and an absent craft one hash.
fn cc_aircars_lane_hit(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                       t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell,
                       lane: i32, fp: f32) -> CityHit {
    let v = cc_aircars_craft(ci, cc, lane);
    if (!v.ok) {
        return cc_aircars_miss(ci);
    }
    return cc_aircars_hit_craft(o, dir, inv_dir, t0, t1, v, lane, ci, fp);
}

fn cc_aircars_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                          t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    // Gate 1, no hash: does this segment pass through the air layer at all?
    // Most city pixels are looking at a facade or at the road.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_aircars_Z_MAX || max(za, zb) < cc_aircars_Z_MIN) {
        return cc_aircars_miss(ci);
    }
    // Gate 2: a craft smaller than this is not worth a box test.
    let fp = (2.0 * u.cam_origin.w / max(u.params.x, 1.0)) * max(t0, 0.0);
    if (fp > cc_aircars_FAR_FP) {
        return cc_aircars_miss(ci);
    }
    let h0 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 0, fp);
    let h1 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 1, fp);
    let h2 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 2, fp);
    return cc_aircars_nearer(cc_aircars_nearer(h0, h1), h2);
}

// One nav lamp. `a` and `b` are the hit's coordinates in the face's own plane
// relative to the craft, `seat_*` the lamp's seat in the same frame, and
// `span_*` the face's size. Resolved while the dot is bigger than a pixel,
// handed to the face's mean when it is not: dot area over face area is the
// honest sub-pixel value, so the craft keeps its light and loses only the
// resolution.
fn cc_aircars_lamp(a: f32, b: f32, seat_a: f32, seat_b: f32,
                   span_a: f32, span_b: f32, col: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let d = length(vec2<f32>(a - seat_a, b - seat_b));
    let sharp = select(0.0, 1.0, d < cc_aircars_LAMP_R);
    let area = 3.14159265 * cc_aircars_LAMP_R * cc_aircars_LAMP_R;
    let mean = area / max(span_a * span_b, 1e-3);
    let k = smoothstep(cc_aircars_LAMP_LOD_START, cc_aircars_LAMP_LOD_FULL,
                       fp);
    return col * (cc_aircars_LAMP_RAD * mix(sharp, mean, k));
}

// White forward, red aft, on whichever face is looking at you. A craft
// crossing your view still declares which way it is going, because the flank
// carries the same pair near its ends.
fn cc_aircars_navlights(h: CityHit, v: cc_aircars_Craft, fp: f32)
        -> vec3<f32> {
    let rel = h.pos - v.base;
    let a = select(rel.y * v.fwd, rel.x * v.fwd, v.axis == 0);
    let across = select(rel.x, rel.y, v.axis == 0);
    let b = rel.z;
    let seat_b = 0.60 * cc_aircars_HULL_H;
    let white = vec3<f32>(1.00, 0.96, 0.88);
    let red = vec3<f32>(1.00, 0.13, 0.06);
    let n_along = select(h.normal.y * v.fwd, h.normal.x * v.fwd, v.axis == 0);
    let n_across = select(h.normal.x, h.normal.y, v.axis == 0);
    let face_w = 2.0 * cc_aircars_HALF_W;
    let face_l = 2.0 * cc_aircars_HALF_L;

    if (n_along > 0.5) {
        return cc_aircars_lamp(across, b, -0.52, seat_b, face_w,
                               cc_aircars_HULL_H, white, fp)
             + cc_aircars_lamp(across, b, 0.52, seat_b, face_w,
                               cc_aircars_HULL_H, white, fp);
    }
    if (n_along < -0.5) {
        return cc_aircars_lamp(across, b, -0.52, seat_b, face_w,
                               cc_aircars_HULL_H, red, fp)
             + cc_aircars_lamp(across, b, 0.52, seat_b, face_w,
                               cc_aircars_HULL_H, red, fp);
    }
    if (abs(n_across) > 0.5) {
        return cc_aircars_lamp(a, b, 1.55, seat_b, face_l,
                               cc_aircars_HULL_H, white, fp)
             + cc_aircars_lamp(a, b, -1.55, seat_b, face_l,
                               cc_aircars_HULL_H, red, fp);
    }
    return vec3<f32>(0.0);
}

// The close read's shading. Everything here is a function of the SDF normal
// or of the hit's position in the craft's own frame, which is the only reason
// a rounded box full of smooth-mins reads as a machine: the eye recovers the
// surface from how the light falls off across it, not from the outline.
fn cc_aircars_shade_sdf(h: CityHit, v: cc_aircars_Craft, glow: vec3<f32>,
                        bright: f32, fade: f32, fill: vec3<f32>,
                        dir: vec3<f32>, fp: f32, a4: vec4<f32>) -> vec3<f32> {
    let f = cc_aircars_frame(v);
    let q = cc_aircars_to_local(f, h.pos - v.base);
    let n = cc_aircars_to_local(f, h.normal);
    let parts = cc_aircars_sdf(q);
    let ya = abs(q.y);
    let moon = u.sun_dir.xyz;
    let refl = reflect(dir, h.normal);
    let mspec = max(dot(refl, moon), 0.0);

    // Which part of the solid the ray landed on. Soft, because the smooth-min
    // means the surface near a join genuinely belongs to both.
    let w_canopy = 1.0 - smoothstep(-0.02, 0.12,
                                    parts.z - min(parts.y, parts.w));
    // The seam is the crease itself: where the canopy's field and the hull's
    // are equal, the join is a physical joint and gets a dark band. Found,
    // not drawn — it follows the bubble wherever the smooth-min puts it.
    let seam = 1.0 - smoothstep(0.0, 0.11, abs(parts.z - parts.y));

    // --- dark-metal body ---------------------------------------------------
    // This is what has to carry the solidity. Emission alone made a candy
    // lozenge of the first cut: a craft has to read as a dark machine with
    // lights ON it, so the hull gets its own small directional budget and the
    // glow is confined to the panels that are actually panels.
    // A hemisphere: cool skyglow above, sodium city below, blended by n.z.
    // This one gradient does more for solidity than anything else here — it
    // is what every curved surface under a lit street actually looks like,
    // warm underneath and cold on top, and the eye integrates the gradient
    // into a shape without being told.
    let gl = city_glow_sample(h.pos.xy, 3.0);
    // The weights are squared, not linear: a linear hemisphere paints every
    // near-vertical surface the same mid-tone and the gradient stops being a
    // gradient. Squaring keeps the flanks dark and puts the light where the
    // surface actually turns.
    let dn = 0.5 - 0.5 * n.z;
    let sky_amb = fill * ((1.0 - dn) * (1.0 - dn));
    let gnd_amb = CITY_UPLIGHT_COLOR * (cc_aircars_UPGLOW * (0.15 + 2.2 * gl)
                                        * dn * dn);
    var body = cc_aircars_HULL_FILL * sky_amb + gnd_amb;
    // Moonlight on dark metal. Physically this is nearly nothing, and it is
    // kept nearly nothing — but the sign of dot(n, moon) is the only cue that
    // says which way a surface faces, and a hull without it is a cutout.
    body = body + cc_aircars_HULL_DIFF * CITY_MOONLIGHT
                  * max(dot(h.normal, moon), 0.0);
    // The craft's own panel spilling back onto its lower flanks. Small, and
    // the only reason it is here is that a light source hanging under a hull
    // that leaves no trace on the hull looks like a decal.
    body = body + cc_aircars_BOUNCE * mix(glow, vec3<f32>(1.0), 0.45)
                  * (bright * dn);
    // Moon sheen: narrow, so it travels across the crown as the craft turns.
    body = body + cc_aircars_SHEEN * vec3<f32>(0.70, 0.78, 1.00)
                  * pow(mspec, 26.0);
    // Rim: the lit edge that says "curved". Tinted by the craft's own glow,
    // because the belly panel is the brightest thing anywhere near the hull.
    let rim = pow(1.0 - min(abs(dot(h.normal, dir)), 1.0), 3.5);
    body = body + cc_aircars_RIM * rim
                  * (0.06 * glow * bright + vec3<f32>(0.010, 0.011, 0.016));

    // --- underglow, mapped onto the curved lower surface -------------------
    // Two pieces, and the difference between them is the whole look: a broad
    // PANEL on the surface that actually faces the ground, and a narrow STRIP
    // along the chine that is all anyone sees from the side. The box version
    // lit the entire flank because a box has no chine to stop at.
    // Gating on the normal ALONE was the second wrong answer: a rounded nose
    // cap points up over its top half and down over its bottom half, so
    // "faces down" lights the entire nose and the entire tail and the craft
    // goes back to being made of light. Every panel is therefore a normal
    // test AND a footprint in the local frame — which is what a panel is.
    // LOD, and it is the hand-off's whole design. Close up these are BARS —
    // a spine down the deck, a strip along the chine, a panel inset in the
    // belly — because that is what fitted lighting looks like. As the
    // footprint grows toward SDF_FP they widen until they cover the faces
    // they sit on, which is exactly what the box path beyond the gate draws.
    // So the two constructions meet with the same mean radiance per face and
    // the craft does not change brightness when the geometry swaps.
    let k = smoothstep(0.06, cc_aircars_SDF_FP, fp);
    let along = 1.0 - smoothstep(mix(0.95, 1.55, k), mix(1.32, 1.90, k),
                                 abs(q.x));
    let panel = along * (1.0 - smoothstep(mix(0.56, 0.86, k),
                                          mix(0.82, 1.00, k), ya));
    let belly = smoothstep(0.30, 0.85, -n.z) * mix(mix(0.02, 0.30, k), 1.0,
                                                   panel);
    // A BAND centred on the chine, not everything below it: a light bar with
    // dark hull above and below is what makes the panel read as fitted rather
    // than as the craft being made of light.
    // The third gate is on the normal being FLANK-like. Without it the band
    // in z sweeps across the nose's rounded underside — which is tangent to
    // that height over a large area — and the craft grows a glowing snout.
    let strip = (1.0 - smoothstep(cc_aircars_CHINE_LO, cc_aircars_CHINE_HI,
                                  abs(q.z - cc_aircars_CHINE_Z)))
                * (1.0 - smoothstep(0.42, 0.78, abs(n.z)))
                * along * (1.0 - belly);
    // The dorsal light is a spine bar down the middle of the deck, not the
    // whole upper surface. From above it plus the canopy crown is the read.
    let spine = 1.0 - smoothstep(mix(0.14, 0.90, k), mix(0.30, 1.10, k), ya);
    let dorsal = smoothstep(mix(0.62, 0.15, k), mix(0.95, 0.55, k), n.z)
                 * spine
                 * (1.0 - smoothstep(mix(1.05, 1.60, k),
                                     mix(1.45, 1.95, k), abs(q.x)))
                 * cc_aircars_DORSAL;
    let amt = belly + strip * cc_aircars_BAR + dorsal;
    var em = glow * (cc_aircars_GLOW_RAD * bright * amt);

    // --- cabin --------------------------------------------------------------
    // Dark glass with a dim warm interior; the dorsal running light rides
    // only the crown, so from the side the bubble is a window and from above
    // it is still the lit chip the aerial read needs.
    // The interior is visible through the SIDE glass and not through the
    // roof, which is both true and the thing that separates a canopy from a
    // painted panel: a bubble lit evenly all over is a lampshade.
    let cab = vec3<f32>(1.00, 0.74, 0.44)
              * (cc_aircars_CANOPY_RAD * cc_aircars_CABIN
                 * (0.6 + 0.8 * a4.z) * (1.0 - smoothstep(0.15, 0.72, n.z)));
    let canopy_em = cab
                  + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL * bright
                            * smoothstep(0.68, 0.97, n.z)
                            * (1.0 - smoothstep(mix(0.10, 0.62, k), mix(0.26, 0.80, k), ya)))
                  + cc_aircars_GLASS * vec3<f32>(0.55, 0.64, 0.90)
                    * pow(mspec, 110.0);
    em = mix(em, canopy_em, w_canopy);
    // Glass takes far less ambient than painted metal, which is most of what
    // tells a canopy from a roof panel at night.
    body = body * mix(1.0, 0.30, w_canopy);
    em = em * (1.0 - cc_aircars_SEAM_DARK * seam);
    body = body * (1.0 - 0.85 * seam);

    // --- thruster discs -----------------------------------------------------
    // The aft caps of the nacelles, seen only from behind the craft.
    let dr = length(vec2<f32>(ya - cc_aircars_NAC_Y, q.z - cc_aircars_NAC_Z));
    let aft = 1.0 - smoothstep(-1.58, -1.36, q.x);
    let back = smoothstep(0.30, 0.75, -n.x);
    let core = 1.0 - smoothstep(0.55 * cc_aircars_THRUST_R,
                                cc_aircars_THRUST_R, dr);
    let cup = 1.0 - smoothstep(cc_aircars_THRUST_R,
                               cc_aircars_THRUST_R + 0.07, dr);
    em = em + mix(vec3<f32>(1.00, 0.93, 0.84), glow, 0.72)
              * (cc_aircars_THRUST * bright * aft * back
                 * (core + cc_aircars_THRUST_HALO * (cup - core)));

    // --- intake slot --------------------------------------------------------
    // A dark louvred cut in each nose flank. Shading, per SPEC: at this scale
    // geometry would buy nothing the normal does not already give.
    let intake = max(
        (1.0 - smoothstep(0.0, 0.050, abs(q.z - 0.33)))
            * (1.0 - smoothstep(0.24, 0.36, abs(q.x - 0.92)))
            * smoothstep(0.25, 0.60, abs(n.y)),
        // The grille across the nose. A vehicle needs one end that is
        // obviously the front, and at this scale a dark slot facing forward
        // does it more cheaply than any amount of modelled taper.
        (1.0 - smoothstep(0.0, 0.055, abs(q.z - 0.25)))
            * (1.0 - smoothstep(0.26, 0.42, ya))
            * smoothstep(0.40, 0.85, n.x));
    // --- panel lines --------------------------------------------------------
    // Two, both at joints a fabricator would actually have: where the nose
    // section meets the body, and where the tail transom starts.
    // Two transverse joints — nose section to body, body to tail transom —
    // and one longitudinal shoulder crease along the flank. The crease is the
    // single most car-like line on the thing: bodies have a waistline.
    let lines = max(max(1.0 - smoothstep(0.0, 0.045, abs(q.x - 0.52)),
                        1.0 - smoothstep(0.0, 0.045, abs(q.x + 0.98))),
                    (1.0 - smoothstep(0.0, 0.032, abs(q.z - 0.46)))
                        * (1.0 - smoothstep(0.35, 0.62, abs(n.z)))
                        * along)
                * (1.0 - w_canopy);
    let dark = 1.0 - max(cc_aircars_INTAKE_DARK * intake,
                         cc_aircars_LINE_DARK * lines);
    em = em * dark;
    body = body * dark;

    // --- nav lamps in their housings ---------------------------------------
    // White forward, red aft, each a lens on the outboard face of its own
    // raised housing. The lens grows and dims with the footprint so the pair
    // keeps its flux as the housing goes sub-pixel — the craft loses the
    // resolution, not the light.
    // The lens is a cap on the OUTBOARD face of the housing bump, found by
    // where the point sits on the bump rather than by distance to its centre:
    // distance alone lights the whole hemisphere, which is how the first cut
    // grew two headlights the size of the cabin.
    let vf = vec3<f32>(q.x - 0.38, ya - 0.93, q.z - cc_aircars_HOUSE_Z);
    let va = vec3<f32>(q.x + 0.52, ya - 0.93, q.z - cc_aircars_HOUSE_Z);
    let df = length(vf);
    let da = length(va);
    // As the footprint grows the housing goes sub-pixel; open the cap and
    // drop the radiance to hold its flux, so the pair fades rather than
    // flickers.
    let wide = clamp(2.4 * fp, 0.0, 0.55);
    let cut = cc_aircars_LENS_OUT - wide;
    let gain = clamp((1.0 - cc_aircars_LENS_OUT) / (1.0 - cut), 0.20, 1.0);
    let lf = (1.0 - smoothstep(cc_aircars_HOUSE_R, 0.165, df))
             * smoothstep(cut, cut + 0.28, vf.y / max(df, 1e-3));
    let la = (1.0 - smoothstep(cc_aircars_HOUSE_R, 0.165, da))
             * smoothstep(cut, cut + 0.28, va.y / max(da, 1e-3));
    em = em + (vec3<f32>(1.00, 0.96, 0.88) * lf
               + vec3<f32>(1.00, 0.13, 0.06) * la)
              * (cc_aircars_LAMP_RAD * gain);

    return body + em * fade;
}

fn cc_aircars_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let code = h.kind - 300;
    let lane = code / 4;
    let part = code % 4;
    let v = cc_aircars_craft(h.cell, cc, lane);
    let a4 = cc_aircars_look(h.cell, lane);
    let glow = cc_aircars_glow_color(a4.x);
    let bright = 0.70 + 0.70 * a4.y;

    // The two edges of the layer, each approached rather than stepped: the
    // population edge at CITY_PROP_RANGE, and the footprint at which the
    // tracer stops testing craft at all.
    let fade = (1.0 - smoothstep(cc_aircars_FADE_START * CITY_PROP_RANGE,
                                 cc_aircars_FADE_FULL * CITY_PROP_RANGE, h.t))
             * (1.0 - smoothstep(cc_aircars_FAR_FADE, cc_aircars_FAR_FP, fp));

    // The night fill the core gives every city surface. A hull is dark metal
    // over a dark city: essentially a silhouette with a lit edge.
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(h.pos.xy, 3.0));
    let body = cc_aircars_HULL_FILL * fill;

    if (part == 3) {   // the close read: shade against the SDF's own normal
        return cc_aircars_shade_sdf(h, v, glow, bright, fade, fill, dir, fp,
                                    a4);
    }
    if (part == 1) {   // the underglow panel: the whole point of the layer
        let rel = h.pos - v.base;
        let a = select(rel.y, rel.x, v.axis == 0);
        let across = select(rel.x, rel.y, v.axis == 0);
        let panel =
            (1.0 - smoothstep(cc_aircars_PANEL_A, cc_aircars_HALF_L, abs(a)))
          * (1.0 - smoothstep(cc_aircars_PANEL_W, cc_aircars_HALF_W,
                              abs(across)));
        return body + glow * (cc_aircars_GLOW_RAD * bright * fade
                              * mix(cc_aircars_PANEL_RIM, 1.0, panel));
    }
    if (part == 2) {   // canopy: a warm interior seen through the glass
        // Its BACK is dorsal like the hull's, and that matters more than it
        // sounds: the canopy covers the middle of the top, so from directly
        // above it is most of what a craft shows. Lit warm, the whole aerial
        // read of the layer went with it. This also makes the LOD hand-off
        // exact — when the canopy box drops out the hull top behind it is
        // already the same radiance.
        if (h.normal.z > 0.5) {
            return body + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL
                                  * bright * fade);
        }
        return body + vec3<f32>(1.00, 0.74, 0.44)
                      * (cc_aircars_CANOPY_RAD * (0.6 + 0.8 * a4.z) * fade);
    }
    // Hull. The back carries a dorsal running light; the flanks carry the
    // skirt of the belly panel, e-folding upward from the seam. Nav lamps are
    // drawn on rather than built.
    if (h.normal.z > 0.5) {
        return body + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL
                              * bright * fade);
    }
    let up = max(h.pos.z - v.base.z, 0.0);
    let wrap = 1.0 - smoothstep(cc_aircars_SKIRT_LO, cc_aircars_SKIRT_HI, up);
    let skirt = glow * (cc_aircars_GLOW_RAD * cc_aircars_SKIRT * bright
                        * wrap);
    return body + (skirt + cc_aircars_navlights(h, v, fp)) * fade;
}
