# Clear air in soar's march, and cheaper kernels for Low and Minimal

2026-11-27, RTX 5080 (Vulkan, wgpu), branch `perf-census`.

The question was whether a fifth empty-space scheme is worth building: a
low-resolution pre-pass that gives each screen tile conservative first and
last cloud distances, so that `fs_main` only tightens `t_near`/`t_far`.
Sections 1–3 measure where the view march's steps and time go in clear
air, and the most such a pre-pass could save. Section 4 measures a menu of
cheaper kernels that could ship at Low and Minimal. Section 5 is my
interpretation, kept apart from the numbers.

Generated tables are in [`tables.md`](tables.md) and
[`tables_per_view.md`](tables_per_view.md) (census and timing),
[`ablate.md`](ablate.md) and [`menu.md`](menu.md). Raw records are the
`*.jsonl` files. What the census and timing report derives from them (per
cell fastest times, per tier sums) is also in [`tables.json`](tables.json).
How to regenerate them is at the end.

## Setup

**Tiers as flown.** Each tier runs with its own render scale, step
factors, light cache, haze (`QUALITY_PRESETS[tier].haze`,
`haze_height_dependent`) and LOD angles (`APP_LIGHT_MARCH_LOD_DEGREES`
1.4 and `APP_VIEW_STEP_LOD_DEGREES` 0.6, each times the tier's
`lodStrength`). That comes to:

| tier | render size (of 2560x1440) | haze | view LOD (deg) | light LOD (deg) | view step (voxels) | light cache |
|---|---|---|---|---|---|---|
| minimal | 320x180 | 1.091 | 1.8 | 4.2 | 4.00 | yes |
| low | 768x432 | 0.659 | 1.2 | 2.8 | 3.00 | yes |
| medium | 1792x1008 | 0.4027 | 0.63 | 1.47 | 2.50 | yes |
| high | 2560x1440 | -0.038 | 0.3 | 0.7 | 2.00 | yes |

The city scene takes `cityLodStrength` and `cityRoofDensity` as well.
soar_bench's own look (haze 1, view LOD 0.3° on every tier) takes
1.65 / 1.42 / 1.12 / 0.90 times the steps of the flown look, from Minimal to
High, so numbers made with it would mislead. None here are.

**Demos.** The core five are twpice (not periodic), congestus, dycoms, rce
and stratified. The census and timing add rce-dales, rce-sam,
marine-congestus-coarse and utah-convection-coarse. Each demo flies under
its own sun.

**Views.** There are ten per demo:

- The five `soar_bench` day views: thick_backlit, under_base, overview,
  thick_lit and ocean_down.
- Five flight-like views placed from the field's cloud levels.
  `base`/`top` are the lowest and highest levels with ≥1% of columns at
  sigma > 1e-3 /m; `top_any` uses sigma > 1e-5 /m.
  - fly_cruise: at base + 0.35 (top − base), in a clear column, looking
    level, sun on the side, fov 75.
  - fly_cruise_back: the same point, looking away from the sun, 5° down.
  - fly_under_up: at half the base height, looking 25° up, fov 80.
  - fly_above_down: 10% of the cloud depth above `top_any`, looking 20°
    down.
  - fly_demo_still: the demo's own still camera (`meta.json`).

  Placements per demo are in `tables.md`.

The city scene (sections 3 and 4) uses `night_city_harness`'s six views
under its moon.

**Timing method.**

- The march pass's GPU time comes from timestamp queries. Wall clock is
  recorded too, but at 2560x1440 it is dominated by reading the image
  back, so it is not used.
- Variants are compiled from patched module text (`tools/soar_variants.py`;
  a patch refuses an anchor that has moved). Every arm of a view is warmed,
  then timed 5 times × 8 frames, in an order rotated each repeat. The
  fastest repeat is reported.
- The census is a separate instrumented shader. No time is taken from it.

**Run quality.** Timing sweeps 2 and 3 are the clean runs. Sweep 1 was
taken while another job used the GPU; it is kept as
`timing_run1_contaminated.jsonl` and used nowhere.

- Run to run, per tier (342 cells clean in both runs), the total full time
  differs by 1.0 / 1.7 / 1.3 / 0.0%, and the oracle savings by ≤0.7 points.
- Per cell, the full time's run-to-run difference has a median of
  0.3–1.7% and a 90th percentile of 2–6%.
- A run counts for a cell when no arm's repeats 2–5 spread by more than 10%
  of the cell's full time. 4 of 360 cells fail that in both runs and are
  left out (listed in `tables.md`).
- The first repeat is left out of the spread, though not of the fastest
  time. On some High views it is 30–100% slow for the first arm or two, the
  same views in both runs, while repeats 2–5 agree within 0.5%. That is the
  GPU coming back up to clock after the host-side work between cells.

## 1. Step census

Mean per pixel over the views, all nine demos. Columns:

- lead / int / trail: view-march steps before the first cloud sample,
  between cloud samples, and after the last.
- cloud: steps where d_tau ≥ 1e-5.
- pm: pre-march steps; pm void is the part wholly before the first cloud
  or after the last.
- px / t16: all steps left by the per-pixel / 16x16 oracle clamp
  (section 2).

| tier | lead | int | trail | cloud | pm (void) | all steps | void share of all steps | px left | t16 left | no-cloud pixels | rays ending opaque |
|---|---|---|---|---|---|---|---|---|---|---|---|
| minimal | 23.6 | 2.8 | 5.6 | 9.7 | 19.3 (11.1) | 61.0 | 71% | 27.7 (−55%) | 37.4 (−39%) | 26% | 40% |
| low | 33.8 | 5.8 | 8.9 | 12.7 | 29.5 (16.4) | 90.8 | 72% | 39.1 (−57%) | 49.1 (−46%) | 21% | 43% |
| medium | 47.9 | 10.9 | 14.3 | 16.5 | 40.5 (21.7) | 130.0 | 73% | 53.7 (−59%) | 63.5 (−51%) | 19% | 45% |
| high | 69.4 | 20.1 | 22.9 | 22.3 | 54.1 (27.5) | 188.8 | 74% | 78.2 (−59%) | 90.9 (−52%) | 17% | 47% |

On the core five alone it is 68.8 / 99.9 / 144.4 / 211.5 steps, 65–69%
void.

- The step cap was never hit.
- The oracle keeps ≥ 99.99% of the cloud steps.
- The step count of the slowest pixel in each 8x4 block is within a few
  percent of the mean pixel's, and the oracle cuts both by the same share,
  so these counts are not hiding divergence.
- Step counts agree with an independent numpy stepping of the zero field on
  923 sampled rays.

Per demo and per view: `tables.md`, `tables_per_view.md`.

## 2. Time, and the ceiling of a first/last pre-pass

The arms are all uninstrumented:

- full: as is.
- zero: the field zeroed.
- live: the march not entered (`t_far = t_near` behind a uniform, so the
  full kernel is compiled).
- px / t16: the march clamped to the census's first/last cloud distances,
  per pixel or per 16x16 tile. The distances are dilated 3x3 and widened
  by 2 steps. Rays that went opaque keep their far end, because the
  pre-march integrates the cloud behind them. The distances come from one
  extra `textureLoad`. The arm measures a perfect, free pre-pass.
- null (sweep 3): the same clamp reading a texture that clamps nothing,
  i.e. what the clamp itself costs.

Mean ms per view (frame), sweeps 2 and 3, fastest of both:

| tier | full | zero | march not entered | px | t16 | px saves | t16 saves | clamp's own cost | steps removed by px save |
|---|---|---|---|---|---|---|---|---|---|
| minimal | 0.205 | 0.210 | 0.042 (21%) | 0.172 | 0.185 | 0.033 (16%) | 0.020 (10%) | 0.001 (0%) | 0.032 (16%) |
| low | 0.715 | 0.944 | 0.172 (24%) | 0.566 | 0.616 | 0.149 (21%) | 0.100 (14%) | 0.002 (0%) | 0.151 (21%) |
| medium | 3.061 | 4.162 | 0.963 (31%) | 2.584 | 2.682 | 0.477 (16%) | 0.379 (12%) | 0.085 (3%) | 0.577 (18%) |
| high | 7.941 | 11.419 | 2.868 (36%) | 6.940 | 6.944 | 1.001 (13%) | 0.997 (13%) | 0.262 (3%) | 1.253 (16%) |

Wall clock per frame, for reference: full 0.52 / 1.42 / 6.39 / 16.61 ms,
zero 0.53 / 1.67 / 7.50 / 20.09 ms, march not entered 0.35 / 0.86 / 4.19 /
11.34 ms.

Per demo, px saves at Low / High:

| demo | Low | High |
|---|---|---|
| congestus | 39% | 19% |
| rce-dales | 21% | 29% |
| utah-convection-coarse | 11% | 24% |
| marine-congestus-coarse | 20% | 10% |
| stratified | 14% | 10% |
| dycoms | 6% | 14% |
| twpice | 5% | 12% |
| rce-sam | 10% | 13% |
| rce | −2% | −3% |

On rce the clamp costs 4–7% and removes steps worth 0–4%.

**Two readings of the void share of time, which disagree.**

1. *Linear.* Price every clear-air step at the zero field's cost per step,
   (zero − live) / zero-field steps: 39.6 / 20.3 / 10.1 / 8.5 ps per
   pixel-step. Then void traversal is 48 / 61 / 57 / 56% of march time.

   | tier | lead | int | trail | pm void |
   |---|---|---|---|---|
   | minimal | 26% | 3% | 6% | 12% |
   | low | 32% | 5% | 8% | 15% |
   | medium | 29% | 6% | 9% | 13% |
   | high | 28% | 8% | 9% | 11% |

2. *Measured.* The px oracle removes 55–59% of all steps (all of lead and
   trail, and the pre-march's void) and saves 16 / 21 / 18 / 16% of time
   once its own cost is set aside. A removed step therefore costs 2.3–2.8
   times less than the zero field's step. Only the measured reading says what a
   pre-pass can buy.

The "march not entered" share (21–36%) is an upper bound on the work
outside the march, not a split of it. With the march skipped, the sea is
shaded under every pixel, including pixels cloud covers in full. On
stratified and marine-congestus-coarse that arm costs about as much as
full, or more.

## 3. Kernel ablations (run 1)

These are the shader before the sea cache read, the core five demos, and
the change in summed march GPU time against stock. Each is the most that
piece could save, not something to ship.

| tier | A/A (stock_again) | no_premarch | no_gradient | ocean_const | ice_const | clearance_ctl |
|---|---|---|---|---|---|---|
| minimal | −0.0% | −25.7% | −27.4% | −11.2% | −0.6% | −0.2% |
| low | +0.1% | −27.9% | −24.1% | −13.0% | −1.0% | −0.1% |
| medium | −0.0% | −19.6% | −23.9% | −17.3% | −1.8% | +0.3% |
| high | +0.1% | −11.5% | −22.7% | −21.3% | −2.0% | +0.6% |

- clearance_ctl is the packed-clearance control: decode a signed texel and
  widen the step, on today's nonnegative field. It costs nothing at Minimal
  and Low, and +0.3–0.6% at Medium and High, per view mostly +0.3 to +1%,
  i.e. at or just above the A/A floor.
- City scene (congestus and marine-congestus-coarse over the city),
  `city_uplight_probe_tau` replaced by a constant: −27.4 / −21.4 / −19.7 /
  −17.2%.
- The second ablation run was replaced by the menu, which repeats
  stock_again, no_premarch and no_gradient on top of the sea cache read.

## 4. Menu of cheaper kernels

Measured on top of the sea cache read. The shader is this branch's plus
the optimization pass's `e728270` (the sea reads its cloud shadow from the
light cache under LIGHT_CACHE), cherry-picked here as `03a8bcd`. The pass
branch's other shader changes are not in it. Stock is therefore cheaper
than in sections 2 and 3, and the ceilings take a larger share of it.

What was measured:

- Scenes: the core five demos by day (10 views each), and the city scene on
  marine-congestus-coarse and congestus (6 night views each).
- Time: the change in summed march-pass GPU time against stock, measured as
  in section 3.
- Picture: each arm rendered converged (32 frames, the first unjittered,
  as the browser accumulates), tone-mapped, at the tier's render size, with
  the same frame indices as stock. The difference from stock is in 8-bit
  display levels: RMS over the views, and the largest over views of each
  view's 99.9th percentile and maximum.
- *Grain* is stock against stock over another set of frame indices: what a
  converged render varies by on its own.

The candidates:

- grad_coarse / grad_fine: `sigma_gradient` with one stencil.
  grad_coarse keeps the distance-dependent radius and the confidence gate
  but drops the fine taps and the blend; grad_fine keeps the one-voxel
  stencil.
- pm_step8: the pre-march at `AHEAD_STEP_SCALE` 8.
- pm_cap30: the pre-march at `AHEAD_TAU_CAP` 30.
- pm_vertical: the pre-march skipped where `ahead_novelty` < 0.05, within
  2.9° of the vertical. There the pre-march's weight is below 5%, so it can
  move a fill by at most 3.5%.
- uplight_4: `CITY_UP_PROBE_STEPS` 4.
- no_premarch and no_gradient are section 3's ceilings, measured again on
  this shader.

Day, core five demos:

| arm | Minimal: time / RMS / max | Low | Medium | High |
|---|---|---|---|---|
| stock ms (sum over 50 views) | 11.67 | 41.52 | 163.88 | 394.81 |
| stock_again (A/A) | −0.1% / 0.00 / 0 | −0.0% / 0.00 / 0 | +0.1% / 0.00 / 0 | −0.0% / 0.00 / 0 |
| grad_coarse | −11.4% / 0.54 / 13 | −13.0% / 0.79 / 18 | −13.2% / 1.05 / 22 | −13.8% / 1.42 / 24 |
| grad_fine | −19.7% / 2.76 / 24 | −21.1% / 3.24 / 30 | −19.1% / 3.56 / 36 | −18.7% / 3.98 / 39 |
| pm_step8 | −5.2% / 0.01 / 2 | −2.8% / 0.03 / 3 | +2.7% / 0.04 / 4 | +2.6% / 0.08 / 6 |
| pm_cap30 | −3.8% / 0.01 / 0 | −4.2% / 0.01 / 1 | −2.6% / 0.02 / 1 | −1.4% / 0.04 / 3 |
| pm_vertical | +0.0% / 0.00 / 0 | −0.0% / 0.00 / 0 | −0.0% / 0.00 / 0 | +0.1% / 0.00 / 0 |
| no_premarch (ceiling) | −27.4% / 0.15 / 4 | −31.1% / 0.26 / 8 | −22.6% / 0.37 / 12 | −14.3% / 0.66 / 22 |
| no_gradient (ceiling) | −30.3% / 4.26 / 20 | −28.8% / 5.02 / 29 | −29.1% / 5.54 / 31 | −30.0% / 6.21 / 39 |
| grain | 0.63 / 31 | 0.47 / 29 | 0.33 / 25 | 0.29 / 40 |

City scene, two fields:

| arm | Minimal: time / RMS / max | Low | Medium | High |
|---|---|---|---|---|
| stock ms (sum over 12 views) | 12.37 | 32.78 | 70.47 | 129.04 |
| stock_again (A/A) | −0.3% / 0.00 / 0 | −0.5% / 0.00 / 0 | −0.0% / 0.00 / 0 | −0.2% / 0.00 / 0 |
| uplight_4 | −14.2% / 0.51 / 33 | −10.4% / 0.68 / 45 | −8.6% / 0.78 / 52 | −8.9% / 0.87 / 56 |
| no_premarch (ceiling) | −27.7% / 0.17 / 5 | −26.9% / 0.32 / 10 | −15.6% / 0.44 / 13 | −10.8% / 0.62 / 18 |
| no_gradient (ceiling) | −10.4% / 2.81 / 32 | −8.7% / 3.17 / 33 | −7.4% / 3.39 / 34 | −6.7% / 3.61 / 35 |
| grain | 2.56 / 67 | 1.92 / 108 | 1.34 / 127 | 1.17 / 113 |

Per demo at Low (time / RMS):

| demo | grad_coarse | no_premarch |
|---|---|---|
| twpice | −18% / 0.94 | −14% / 0.40 |
| congestus | −10% / 1.33 | −46% / 0.41 |
| dycoms | −20% / 0.35 | −15% / 0.02 |
| rce | −15% / 0.55 | −5% / 0.06 |
| stratified | −12% / 0.12 | −26% / 0.02 |

Per-demo and per-view rows are in `menu_run1.jsonl`; `menu.md` has every
tier.

Notes on the measurements:

- pm_vertical skips no ray in any of the 50 day views. Only 0.5% of rays
  are within 20° of the vertical, and 2.3% within 30° (the steepest view
  looks 55° down).
- pm_step8 is faster at Minimal and Low and slower at Medium and High,
  though it halves the pre-march's steps. That is measured, not explained.
- The largest differences sit in specific places:
  - no_premarch: a faint warm tint inside deep cloud (twpice
    fly_demo_still, RMS 1.17 at Low).
  - grad_coarse: fine-scale skin texture.
  - grad_fine: the shading of the billows flattens (dycoms ocean_down,
    RMS 8.5 at Low).
  - uplight_4: the amber on distant cloud seen from the aerial city view.

Side-by-side PNGs (stock | candidate | |difference| × 8) are written for
the two views per demo with the largest RMS at Minimal or Low, and none
where the 8-bit picture is identical to stock. There are 70 files, 24 MB,
in `menu_png/` in the `cloudyview-census` worktree. They are not committed
because the repository is public and they would stay in its history.

## 5. Conclusions (interpretation)

This is my reading of the numbers above. All of it is from one RTX 5080;
the Mac's numbers may move the shares, and the tools run there unchanged.

**Is a first/last pre-pass worth building?** No, on this evidence. A
perfect and free per-pixel pre-pass would save 0.03 / 0.15 / 0.48 / 1.0 ms
per frame at Minimal / Low / Medium / High: 16 / 21 / 16 / 13% of march
time. At the realistic 16x16 tile it saves 0.02 / 0.10 / 0.38 / 1.0 ms
(10 / 14 / 12 / 13%). A real pre-pass would also pay for its own pass, and
for a clamp that costs 0–3% by itself over all demos, and 3–9% on the
three rce fields and utah. A tenth of a millisecond at Low is not much, and it is a ceiling.

**Why the step counts promised more.** Clear air is 71–74% of all steps,
and a per-pixel clamp removes 55–59% of them. But a removed clear-air step
costs 2.3–2.8 times less than an empty-field step. So the linear estimate (void
traversal 48–61% of march time, lead the largest part at 26–32%, then
pre-march void 11–15%, trail 6–9%, interior 3–8%) overstates what pruning
buys by about three times: its removable part (lead, trail, pre-march void)
is 44 / 55 / 51 / 48% of march time, where the measured saving is 16 / 21 /
18 / 16%. I did not test why. The 8x4-block step maximum
falls with the mean, so divergence in step count is not the reason.

**Where the time is instead.** On top of the sea cache read, the
pre-march's whole cost is 27 / 31 / 23 / 14% of march time by day, and the
gradient stencil's 30 / 29 / 29 / 30%. Before the sea change, the sea was
11–21%. Each of those is larger than the void-pruning ceiling.

**Per tier.**

- **Minimal** (0.2 ms per frame here). Void pruning is worth at most
  0.03 ms. Removing the pre-march is worth −27% of march time and moves
  the picture by 0.15 levels RMS (max 4) over the 62 day and city views,
  under the render's own grain of 0.63. Per view, 2 of the 62 differ by more
  than their own grain, both twpice (fly_demo_still 0.52 against 0.13, and
  fly_under_up).
- **Low.** Void pruning is worth at most 0.15 ms (21%). Removing the
  pre-march is worth −31% for 0.26 RMS (max 8; grain 0.47). Across demos
  that is −5% (rce) to −46% (congestus). Per view, 3 of 62 differ by more
  than their own grain: twpice fly_demo_still (1.17 against 0.14),
  fly_under_up, and congestus fly_cruise.
- **Medium.** Void pruning is worth at most 0.48 ms (16%). Without the
  pre-march it is −23% for 0.37 RMS (max 12; grain 0.33): at the grain.
- **High.** Void pruning is worth at most 1.0 ms (13%). Without the
  pre-march it is −14% for 0.66 RMS (max 22; grain 0.29): above the grain,
  in deep cloud.

**The menu: what I would ship, and what not.**

- *Pre-march off at Minimal, and at Low once Thomas has looked at twpice's
  still.* It was in the menu only as a ceiling, but it is the best trade
  measured: about −30% of march time for a change that, over the 62 views
  of each tier, is below the converged grain. It is not invisible
  everywhere: deep inside twpice's anvil, in its demo still, it is a faint
  warm tint (`menu_png/twpice_day_no_premarch_*_fly_demo_still.png`).
  Medium is borderline; High should keep it. It is also the cheapest change
  to make: a tier switch on a call that already exists.
- *pm_cap30 wherever the pre-march stays.* −1.4 to −4% for a change no one
  could see (RMS ≤ 0.04, max ≤ 3).
- *grad_coarse: Thomas's call, at Minimal and at Low.* −11% for 0.54 RMS
  at Minimal (max 13) and −13% for 0.79 at Low (max 18). Over all views
  that is about the grain, but per view it exceeds its own grain in 25 of
  50 views at Minimal and 36 at Low. It is a systematic change of
  fine-scale skin texture, not noise. Whether that matters at 320x180 and
  768x432 is a call about the look, to be made from
  `menu_png/*grad_coarse*`.
- *uplight_4 at Minimal, possibly Low.* −14% / −10% of city march time
  for RMS 0.5–0.7, under the city's grain of 1.9–2.6, and under its own
  grain in every one of the 12 views at those tiers. The change is
  concentrated on the amber of distant uplit cloud, up to 33–45 levels in
  the aerial view. Again Thomas's call, from `menu_png/*uplight_4*`.
- *Not grad_fine.* −20% buys a visible flattening of the billows (RMS
  2.8–4.0, max 24–39, 6–14 times the grain).
- *Not pm_step8.* A few percent at Minimal and Low, slower at Medium and
  High.
- *Not pm_vertical.* It skips no rays in any view; flight hardly ever
  looks within 30° of the vertical.

The combinations (pre-march off with grad_coarse, for instance) were not
measured together. The savings need not add.

## Regenerating

From the repository root, with the demos at `web/soar/demos` (or
`--demo-dir`):

```bash
uv run --extra dev python tools/soar_void_census.py census --out census.jsonl
uv run --extra dev python tools/soar_void_census.py timing --label run3 --out timing_run3.jsonl
uv run --extra dev python tools/soar_void_census.py report census.jsonl timing_run2.jsonl timing_run3.jsonl --md tables.md --json tables.json
uv run --extra dev python tools/soar_ablate.py --demo twpice --label run1 --out ablate_run1.jsonl
uv run --extra dev python tools/soar_ablate.py --report ablate_run1.jsonl --md ablate.md
uv run --extra dev python tools/soar_menu.py --demo twpice --label run1 --out menu_run1.jsonl
uv run --extra dev python tools/soar_menu.py --report menu_run1.jsonl --md menu.md
```

The census and timing take `--demos` (default the core five; these runs
passed all nine). The ablation and menu run one demo per call, and
`--scene city` for the city scene. `tools/soar_ablate.py`'s docstring has the notes for running them
on an Apple-silicon Mac.
