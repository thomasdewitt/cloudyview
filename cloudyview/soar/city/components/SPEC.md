# City components — the contract

The night city (raymarch.wgsl, `CITY` specialization) grows by components:
one WGSL file here per kind of thing, registered in `registry.json`, spliced
into the shader by `tools/compose_city.py`. The composer generates the hook
dispatchers, refuses namespace collisions, and compile-validates before
writing — so components compose without their authors coordinating.

## Ground rules

1. **Namespace.** Every module-scope symbol you define (fn / const / struct /
   var) must be named `cc_<yourname>_*`, where `<yourname>` is your
   registry `name`. The composer refuses anything else.
2. **Never edit raymarch.wgsl.** Your file + one registry entry is the whole
   footprint. Run `uv run python tools/compose_city.py` to splice, and
   `--check` to validate without writing.
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
   Cloudpunk we have a real ground: street-level life and ground cars
   coexist with flying ones.

## What the core gives you

Read raymarch.wgsl's city section. You may CALL any core `city_*` /
`pcg2d` / `hash*` function and read any `CITY_*` const and `u.*` uniform;
you may not redefine them. The structs you exchange with the core:

- `CityCell` — everything about one block: `built`, `density`, `rank`,
  `height`, `plot_min/max`, tier boxes `b1min..b3max`, `seed: vec2<u32>`,
  `lit_frac`, `palette_bias`, `store_draw`. Blocks are `u.ocean_params.x`
  meters on a side (90 on the shipped tile); streets run between plots,
  avenues every 8 blocks (`city_is_avenue`).
- `CityHit` — `hit, t, pos, normal, kind, cell`. Component hits use
  `kind = <your kind_base> + local`, local in [0, 15]; core kinds are
  0 ground, 1 facade, 2 roof, 3 mast, 4 beacon.
- `city_box_hit(o, inv_dir, bmin, bmax) -> vec2(t_near, t_far)` and
  `city_box_normal(p, bmin, bmax)` do slab tests for you.

## Hooks

Register only the hooks you implement, in `registry.json`:

```json
{
 "file": "streetlife.wgsl",
 "name": "streetlife",
 "kind_base": 100,
 "hooks": {"cell_props": "cc_streetlife_props_trace",
           "shade": "cc_streetlife_shade"},
 "enabled": true
}
```

Signatures (the composer checks these by compiling):

- `cell_props` — `fn (o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
  t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell) -> CityHit` — geometry
  inside cell `ci`, tested against the ray segment [t0, t1]. Return the
  nearest hit with `t` in that segment, or `hit = false`.
- `extra_trace` — `fn (o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)
  -> CityHit` — geometry independent of the block grid (an elevated
  highway network). Must stay below `CITY_SLAB_TOP`.
- `shade` — `fn (h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
  -> vec3<f32>` — radiance (before fog; the core applies fog) for hits
  whose kind is in your range.
- `facade` — `fn (cc: CityCell, h: CityHit, uc: f32, vc: f32, fp: f32)
  -> vec3<f32>` — ADDITIVE facade emission. `uc` runs along the facade in
  meters, `vc` is height; the window grid is `CITY_WIN_PITCH_U` x
  `CITY_FLOOR_H`. Return zero where you add nothing.
- `window_glyph` — `fn (cc: CityCell, wh: vec4<f32>, pane_uv: vec2<f32>,
  fp: f32) -> vec3<f32>` — transmission MULTIPLIER inside a lit pane
  (`pane_uv` in [0,1]^2, `wh` is that window's hash draw). Return
  `vec3(1.0)` to leave a window alone. Sub-pixel glyphs (fp much larger
  than a pane) must return your glyph's mean transmission, not 1.

## Iterating on your look

The harness renders the real scene headlessly (RTX 5080, ~0.1 s/view):

    uv run python tools/compose_city.py               # splice your component
    uv run python tools/night_city_harness.py --frames 48 \
        --views street --outdir /tmp/yourname

Add `--camera X Y Z AZ EL FOV` (world meters / degrees) for close-ups; the
megatower district sits at (92800, 52800) in the harness field, sprawl to
the northeast. Look at your renders. Judge them against the bar the clouds
set: would this frame hold up next to them? Iterate until it would.
