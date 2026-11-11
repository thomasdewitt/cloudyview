# City components — the contract

The night city grows by components: one WGSL file here per kind of thing,
listed among the city's files in `../../manifest.json` and called from the
hook dispatchers in `dispatch.wgsl`.

## Ground rules

1. **Namespace.** Every module-scope symbol you define (fn / const / struct /
   var) must be named `cc_<yourname>_*`, where `<yourname>` is the file's
   name. All the city's files are one module, so anything else can collide.
2. **A component's footprint** is its file, its line in `manifest.json`, and
   one call per hook it implements in `dispatch.wgsl`.
3. **Budget.** The DDA cell loop is the hot path of every city pixel.
   `cell_props` runs per visited cell within `CITY_PROP_RANGE` (2.2 km) of
   the camera: keep it to a few box tests and hashes, and gate everything
   behind cheap rejects (your prop's z-extent vs the segment, one hash
   draw before any geometry). `extra_trace` runs once per ray: an analytic
   distance-to-network test, not a marcher. Shading hooks run once per hit
   pixel — cheap. When in doubt, gate by `fp` (pixel footprint in
   m/px): detail that is sub-pixel must dissolve into its own mean, the
   way the core windows dissolve into blocks (see the octave ladder in
   `city_shade`) — nothing may just vanish at a distance.
4. **Determinism.** No time, no state. Randomness comes from `pcg2d` /
   `city_rand4` seeded by cell indices or world-position lattices, so a
   thing is where it is on every frame and for every camera.
5. **Look.** Reference exposure 6, Reinhard-with-white-point-15 tone map,
   gamma 1.66. Radiance yardsticks: lit window ~3.5, storefront ~2.2,
   sodium lamp pool ~0.7 peak, aviation beacon 40. The palette is sodium
   amber / fluorescent / cyan / magenta (`city_window_color`); night fill
   light is `CITY_SKYGLOW` and `CITY_MOONLIGHT`. Emission carries the look;
   albedo-lit surfaces are near-black at night. Stay in that world.

   **Vibe targets (Thomas, 2026-08-20).** *Cloudpunk* is the closest
   reference for what this city is — layered neon verticality, air traffic,
   glowing signage — minus its voxel-art construction. *Stray* is the bar
   for solidity — its streets feel inhabited and physically real — but a
   notch dirtier than we want; this city is lived-in, not derelict. Unlike
   Cloudpunk we have a real ground: street-level life and ground cars.
   No air traffic: the flying cars were tried and removed (Thomas,
   2026-11: *"For the flying cars, I recommend just getting rid of them
   wholesale."*).

   **Detail calibration (Thomas, and it applies to ALL detail, not just
   vehicles): wiper-blades-level, not tire-brand-name-level.** Distinct
   functional features — seams, housings, fittings, hardware — never
   micro-text, logos, or ornament below the functional scale.

## What the core gives you

Read raymarch.wgsl's city section. You may CALL any core `city_*` /
`pcg2d` / `hash*` function and read any `CITY_*` const and `u.*` uniform;
you may not redefine them. The structs you exchange with the core:

- `CityCell` — everything about one block: `built`, `density`, `rank`,
  `height`, `plot_min/max`, tier boxes `b1min..b3max`, `seed: vec2<u32>`,
  `lit_frac`, `palette_bias`, `store_draw`, plus the architecture: `arch`
  (0 slab/setback, 2 growth, 3 tapered shaft, 4 spire crown, with `fmin/
  fmax/fscale` describing the frustum of 3 and 4), the window style
  (`win_style/win_pitch/pane_lo/pane_hi/win_mono`), and `merged`
  (a 2x2 superblock building). Blocks are `u.ocean_params.x` meters on a
  side (90 on the shipped tile); streets run between plots, avenues every
  8 blocks (`city_is_avenue`).
  **Respect the architecture**: rooftop furniture (antennas, clutter,
  bridge landings) belongs only on buildings with a flat top to stand it
  on — arch 0 and 2 anywhere, arch 3 only within the shrunken crown
  (footprint scaled by `fscale` about the plot centre), arch 4 never.
- `CityHit` — `hit, t, pos, normal, kind, cell`. Component hits use
  `kind = <your kind_base> + local`, local in [0, 15]; core kinds are
  0 ground, 1 facade, 2 roof, 3 mast, 4 beacon. The facade shell's hits
  take 1000 and up (`CITY_KIND_SHELL + part + 16 * (side + 4 * tier)`, to
  1191), so a kind_base ends at 900.
- `city_box_hit(o, inv_dir, bmin, bmax) -> vec2(t_near, t_far)` and
  `city_box_normal(p, bmin, bmax)` do slab tests for you.

## Hooks

A component implements only the hooks it needs, and `dispatch.wgsl` calls
each one from the dispatcher of that hook: `cc_cell_props_trace` calls every
component's `cell_props` function and keeps the nearest hit,
`cc_component_shade` picks the component by the range its hit `kind` falls in, and so on. Geometry
hooks need a `kind_base`, a multiple of 100 no other component uses
(streetlife 100, skyway 200, adscreens 400, skybridges 500, rooftopworks
600, industry 700, streetfurniture 800, manholesteam 900).

Signatures:

- `cell_props` — `fn (o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
  t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell) -> CityHit` — geometry
  inside cell `ci`, tested against the ray segment [t0, t1]. Return the
  nearest hit with `t` in that segment, or `hit = false`.
- `near_props` — same signature as `cell_props` — geometry below
  `CITY_FURNITURE_TOP` (3 m) that is only drawn close: street furniture,
  the parked cars' sphere-traced hulls. It runs ONLY in the city's street
  stage (`fs_surface_street`), after the facade shell, over the stretch of
  the view ray below 3 m, up to the surface the earlier stages found and at
  most `city_near_reach` — so heavy code here costs the first stage no
  registers. Two things follow. (1) `t0` in the first cell is where the ray
  comes down to 3 m, not the cell's entry: take a footprint for a stage
  decision from the thing's own entry (streetlife's per-car switch), never
  from `t0`. (2) The contract: draw nothing where
  `max(0.35 pixel_angle, true_pixel_angle) * t > CITY_NEAR_FP` (0.75), the
  street stage's reach. A hook is not in wet reflections
  (`CITY_TRACE_PLAIN`), and nothing makes it continuous with `cell_props`:
  a component that draws the same thing in both (streetlife's cars: boxes
  in the first stage, hulls here) must split it so each instance is
  exactly one stage's.
- `extra_trace` — `fn (o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)
  -> CityHit` — geometry independent of the block grid (an elevated
  highway network). Must stay below `CITY_SLAB_TOP`.
- `shade` — `fn (h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
  -> vec3<f32>` — radiance (before fog; the core applies fog) for hits
  whose kind is in your range.
- `facade` — `fn (cc: CityCell, h: CityHit, dir: vec3<f32>, uc: f32,
  vc: f32, fp: f32) -> vec3<f32>` — ADDITIVE facade emission. `dir` is the
  ray that found the wall (the view ray, or a wet street's reflected one),
  for a footprint that knows the angle the wall is seen at: `fp` is the
  pixel's across the ray, and a wall seen edge-on spreads it along `uc` by
  `1/|dot(dir, h.normal)|`. `uc` runs along the facade in
  meters FROM THE FACE'S OWN START CORNER (`city_face_at(cc, h.pos,
  h.normal)` gives the face: its length, tier and bay lattice), `vc` is
  height; the window grid is the face's bays (`f.pitch` x `CITY_FLOOR_H`,
  between corner margins). On a frustum skin (`!f.aligned`) uc is the old
  world-anchored coordinate. `h` is always on the wall plane. Return zero
  where you add nothing.
- `facade_cover` — `fn (cc: CityCell, h: CityHit, uc: f32, vc: f32,
  fp: f32) -> f32` — the share (0..1) of the wall at this point that a panel of
  yours covers. Covered wall shows no windows and, up close, no shell
  relief: the core takes them out, so an opaque panel only adds its own
  emission through `facade`.
- `roof` — `fn (cc: CityCell, h: CityHit, fp: f32, fp_filter: f32,
  dir: vec3<f32>, direct: bool, base: vec3<f32>) -> vec3<f32>` —
  ADDITIVE emission on a core roof hit (kind 2), at every distance. This is
  where roof geometry that `cell_props` only draws near (inside
  `CITY_PROP_RANGE`) keeps its far-field mean: a warehouse's lit sawtooth
  glazing must not vanish at 2.2 km. Return zero where you add nothing.
  `fp_filter` is the true pixel's footprint (or a reflection lobe's), `dir`
  the ray that hit, `direct` false for a roof seen in a reflection, and
  `base` the core's own radiance at the hit: what a hook needs to average
  bright detail as it is shown (`city_win_add` / `city_windows_resolve`),
  handing back the resolved radiance minus `base`.
- `window_glyph` — `fn (cc: CityCell, wh: vec4<f32>, pane_uv: vec2<f32>,
  deep_uv: vec2<f32>, fp: f32) -> vec3<f32>` — transmission MULTIPLIER
  inside a lit pane (`pane_uv` in [0,1]^2 at the glass, `wh` is that
  window's hash draw). `deep_uv` is where the same ray crosses the
  occupant's plane inside the room, in the same pane units — draw things
  that stand IN the room (figures) at `deep_uv` and things AT the glass
  (curtains, blinds) at `pane_uv`, and parallax comes for free; on a flat
  facade the two are equal. Return `vec3(1.0)` to leave a window alone.
  Sub-pixel glyphs (fp much larger than a pane) must return your glyph's
  mean transmission, not 1.
- `window_glyph_mean` — `fn () -> vec3<f32>` — your glyph's mean
  transmission over the whole population of lit windows (what
  `window_glyph` averages to across many windows at a large footprint). The
  core's facade uses it where a pixel covers more windows than it sums one
  by one. A component with `window_glyph` must provide it, and the two must
  agree.

- `media` — `fn (o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
  t_max: f32, pixel_angle: f32, rnd: f32) -> vec4<f32>` — participating
  media between the eye and the city hit at `t_max` (steam): in-scattered
  radiance in rgb, transmittance in a; `(0, 0, 0, 1)` is the identity. The
  street stage packs the result into the VEIL at `t_low`, where the ray
  first comes down to `CITY_NEAR_TOP` (15 m), and fs_main composites it
  when its march reaches that distance — so the steam lies in front of
  whatever is behind it, cloud and sky included, and never in the surface
  rgb. `rnd` is a per-pixel independent (white) uniform per frame, for
  jittering a march.

## The street (core section THE STREET SURFACE)

The ground is a cross-section around each plot: sidewalk (`CITY_SIDEWALK_W`,
`_AVE` on avenues), kerb (`CITY_KERB_H` high), gutter, parking lane, travel
lanes. `city_kerb_rect(cc)` / `city_kerb_sd(p, cc, kr)` give the kerb line
(negative on the sidewalk), `city_road(p)` where a point sits in the network,
`city_manhole(k, j, along_y)` the manhole lattice, and
`city_street_light(p, n)` / `city_street_light_g(p, n, g)` the sodium light
of the lamp lattice on a surface or in a medium — light props with it. Gate
anything that stands along an edge on `city_edge_street(cc, side)`.

**The sidewalk slab reports its top as ground.** streetlife traces the
raised slab and returns kind 0 for its top face, so the core shades it (and
reflects off it) exactly as it shades the z = 0 plane beyond the prop range.
That is the one sanctioned exception to "component hits use your own kinds".

**Wet reflections do not see props.** The reflected ray off wet ground is
traced with `CITY_TRACE_PLAIN`: no `cell_props`, `near_props` or `extra_trace`
hook runs for it (their second inlined copy cost 20-36% on views that never reflect). A
component's geometry therefore never appears in a puddle; its facade and
window hooks do, because the reflected hit is shaded by `city_shade`.

## The facade shell (core)

Up close a building face is not a box face: the core traces it as a SHELL
(raymarch.wgsl, "Facade shell"), in the city's second surface stage
(`fs_surface_refine`) — recessed windows with the glass set back, sills,
hoods, ledges, cornices, fins and mullion caps, AC units, drainpipes on
brackets, rain canopies, balconies, fire escapes, and an interior-mapped
room behind every lit pane. What that means for a component:

- **Face frame.** `city_face_at(cc, h.pos, h.normal)` returns the face a
  facade hit lies on: `side`, `tier`, `len`, and its bay lattice (`margin`,
  `pitch`, `nbays`). `uc` in the facade hooks is metres from the face's
  own start corner. Windows sit in an integer number of bays between two
  corner margins (`CITY_SHELL_CORNER`), so nothing of yours should assume a
  window lattice anchored to world coordinates.
- **Relief extent.** Nothing of the shell stands more than
  `CITY_SHELL_OUT_MAX` (1.5 m) off its wall; the flyer's collider inflates
  every building box by exactly that. Props within 1.5 m of a wall share
  space with the shell — keep bins and bollards out of it, or expect a
  balcony slab above them.
- **Panels.** A component that hangs an opaque panel on a wall (a screen, a
  sign box, a roll-up door) reports it through `facade_cover`, and the core
  takes the windows and the relief out from under it.
- **LOD.** The shell's depths flatten to the wall over pixel footprints
  0.4 -> 1.2 m, and its shading hands off to the flat facade at the same
  point on the wall, so a facade hook sees the same wall at every range.
- **Industrial walls** (`land_use == CITY_USE_INDUSTRIAL`) are the industry
  component's; the shell leaves them alone.

## Detailed micro-geometry: the SDF-in-a-box pattern

Props that the camera can get close to (vehicles, street furniture) should
NOT be bare boxes — Thomas: the micro details should be *very highly
detailed*, and the budget genuinely allows it. The calibration (his):
**wiper-blades-level detail, not tire-brand-name-level** — distinct
functional features (blades, door seams, intakes, antennae, light
housings, panel lines), no micro-text or logos. The pattern:

1. Keep one cheap axis-aligned **bounding box** per prop for the DDA
   (`city_box_hit`). This is the only cost the wide scene ever pays.
2. When the bounding box IS hit and `fp` is small (say < 0.5 m/px),
   **sphere-trace your own SDF inside the box**: 16-24 iterations from the
   box entry, rounded boxes / capsules / smooth-min unions, small yaw
   rotations applied to the local point (props need not be axis-aligned
   inside their box). Normal by 4-tap SDF gradient. Return the refined hit;
   if the SDF misses inside the box, return miss (rays graze past a curved
   hull — that is what makes it read as curved).
3. Beyond the `fp` gate, fall back to the box silhouette (or your own
   coarse union of 2-3 boxes) — the SDF's job is the close read, the far
   read is lights anyway.
4. Shade with the SDF normal: curvature is visible almost entirely through
   the normal's effect on rim fill and emissive-panel falloff. Panel lines
   and intakes are cheaper as dark shading bands in the SDF's local frame
   than as geometry.

Cost honesty: only pixels whose rays enter a prop's bounding box pay for
EXECUTING the SDF — but every pixel pays for its REGISTER PRESSURE. The
city and the clouds share one kernel, and a big unrolled SDF (the classic
offender: four inline copies of the field for a 4-tap gradient) collapses
occupancy for the whole frame even when no prop is in view — the flying
cars' refit (a component since removed) measured +65% on craft-free
vistas from exactly this, fixed pixel-identically by rolling the gradient
taps into a loop. Keep your SDF
in ONE function, roll your gradient taps, and A/B a prop-free vista, not
just your close-up.

## Iterating on your look

The harness renders the real scene headlessly:

    uv run python tools/night_city_harness.py --frames 48 \
        --views street --outdir /tmp/yourname

Add `--camera X Y Z AZ EL FOV` (world meters / degrees) for close-ups; the
megatower district sits at (92800, 52800) in the harness field, sprawl to
the northeast. Look at your renders. Judge them against the bar the clouds
set: would this frame hold up next to them? Iterate until it would.
