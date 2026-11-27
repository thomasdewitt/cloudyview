# Clear air in soar's march, and cheaper kernels for Low and Minimal

2026-11-27, RTX 5080 (Vulkan, wgpu), branch `perf-census`.

The question was whether a fifth empty-space scheme is worth building: a
low-resolution pre-pass that gives each screen tile conservative first and
last cloud distances, so that `fs_main` only tightens `t_near`/`t_far`.
Sections 1–3 measure where the view march's steps and time go in clear
air, and the most such a pre-pass could save. Section 4 measures a menu of
cheaper kernels that could ship at Low and Minimal. Section 5 is my
interpretation, kept apart from the numbers. Two later sections each keep
their numbers and their interpretation apart: section 6 measures the
cost ratio from each quality tier to the next, the constants the quality
probe escalates on, and section 7 measures cheaper density-gradient
stencils.

Generated tables are in [`tables.md`](tables.md) and
[`tables_per_view.md`](tables_per_view.md) (census and timing),
[`ablate.md`](ablate.md), [`menu.md`](menu.md), [`tier_cost.md`](tier_cost.md)
and [`gradient.md`](gradient.md). Raw records are the
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

## 6. Tier cost ratios

The constants behind the quality probe's escalation,
`AUTO_TIER_COST_RATIO_TO_NEXT` in `web/soar/constants.js`, are
{minimal 4.0, low 4.0, medium 3.0}. The probe escalates when the measured
frame time times the ratio is under 33.3 ms × 0.75. The old constants
came from soar_bench's wall clock on TWPICE in two views. This section
measures the march pass's ratio from one tier to the next, on this
branch at `0f61214` (pre-march off at Low and Minimal, sea cache read).
`tools/soar_tier_cost.py` made it. The generated tables are
[`tier_cost.md`](tier_cost.md) and [`tier_cost.json`](tier_cost.json), and
the raw records are `tier_cost_run1.jsonl`.

### Numbers

All nine demos by day, ten views each, at a 2560x1440 window. The time is
march-pass GPU time, fastest of 5 repeats × 8 frames, from one run. Two of
the 360 as-flown cells spread by more than 10% and were kept. Three
measures:

- *as flown*: each tier at its own render scale.
- *throughput*: each tier's look rendered at 2560x1440 and scaled by its
  pixel count. The step LOD is the tier's angle at every tier, since the
  quarter-pixel floor is under it everywhere, so steps per pixel do not
  depend on render size.
- *extrapolated*, an estimate: throughput times the per-pixel penalty
  measured between 2560 and 1280 wide (×1.23–1.32 per halving of
  resolution with the GPU full), carried down to the tier's own width.

The ratio per demo is the sum over the demo's views of the next tier's
time over this tier's. The per-view maximum is over all 90 views.

| step | measure | mean | median | max over demos | min | per-view max |
|---|---|---|---|---|---|---|
| low/minimal | as flown | 3.66 | 3.66 | 4.90 (stratified) | 2.64 | 6.92 |
| low/minimal | throughput | 7.71 | 7.43 | 9.11 (rce-sam) | 6.53 | 13.90 |
| low/minimal | extrapolated | 6.13 | 6.75 | 8.16 (stratified) | 3.59 | 11.91 |
| medium/low | as flown | 5.70 | 5.48 | 6.28 (rce-dales) | 4.93 | 8.03 |
| medium/low | throughput | 8.48 | 8.24 | 10.27 (stratified) | 7.46 | 13.29 |
| medium/low | extrapolated | 6.70 | 6.57 | 8.40 (stratified) | 5.85 | 9.67 |
| high/medium | as flown | 2.58 | 2.58 | 3.68 (stratified) | 1.63 | 5.95 |
| high/medium | throughput | 2.90 | 2.86 | 4.10 (stratified) | 2.19 | 7.50 |
| high/medium | extrapolated | 2.55 | 2.54 | 3.51 (stratified) | 1.59 | 5.48 |

Without stratified, the maxima over demos are:

- as flown: 4.68 (rce), 6.28 (rce-dales) and 2.89 (rce-dales);
- throughput: 9.11 (rce-sam), 9.78 (congestus) and 3.11 (rce-dales).

The same per-demo as-flown ratios from census timing run 3 (section 2,
before the sea cache read, with the pre-march still on at Low and
Minimal), as mean / max:

- low/minimal: 3.74 / 4.71;
- medium/low: 4.93 / 5.93;
- high/medium: 2.62 / 3.53.

Per tier, as means over the 90 (demo, view) cells:

| tier | render size | as flown ms | throughput ms | above throughput | extrapolated ms | above extrapolated | march not entered | Python wall ms | wall − march ms |
|---|---|---|---|---|---|---|---|---|---|
| minimal | 320x180 | 0.137 | 0.034 | 75% | 0.082 | 40% | 5% | 0.45 | 0.32 |
| low | 768x432 | 0.464 | 0.256 | 45% | 0.407 | 12% | 4% | 1.18 | 0.72 |
| medium | 1792x1008 | 2.584 | 2.205 | 15% | 2.629 | −2% | 3% | 6.03 | 3.44 |
| high | 2560x1440 | 6.202 | 6.205 | 0% | – | – | 3% | 14.89 | 8.69 |

The column definitions are:

- *above throughput* is 1 − throughput / as flown.
- *march not entered* is the `nomarch_live` arm's time as a share of the
  as-flown time.
- *wall − march* is the Python host's time per frame outside the timed
  pass. That covers encoding, the wait on each band before the next is
  submitted, the untimed accumulate pass at render size, and read-back.

### Interpretation

- **The ratio a slower GPU sees lies between the as-flown and throughput
  ratios.**
  - *As flown is a floor.* On the 5080, Minimal and Low do not fill the
    GPU: 75% and 45% of their time is above throughput.
  - *Throughput is a ceiling.* It charges every tier the fetch coherence
    of a 2560-wide render, and per-pixel cost rises as the render
    shrinks.
  - *Extrapolated is my middle estimate.* One check supports it: at
    Medium, which is 1792 wide and inside the measured 1280–2560 range,
    it reproduces the as-flown time to 2%. At Low it extrapolates 0.7
    halvings past 1280, and at Minimal 2, so it is weaker there.
- **Minimal at 4.0** is the mean on this GPU (3.7) but under its worst
  demo (4.9). On a GPU that Minimal fills, it is probably 6–8, and up to
  about 9 at the worst demo.
- **Low at 4.0** is under every demo, even as flown here, where the
  range is 4.93–6.28. On a slower GPU it is probably 6.5–8.5.
  - `0f61214` raised this one. Dropping the pre-march made Low about 30%
    cheaper and left Medium alone, so the mean went from 4.9 (run 3) to
    5.7.
- **Medium at 3.0** holds as flown for eight of the nine demos (max 2.89
  without stratified). Stratified is 3.5–4.1 on every measure.
- **Choosing the constants.** If the gate has to hold on the worst demo
  on any GPU, the numbers point to about 8–9 / 8–10 / 4. If it only has
  to hold on the worst demo on a 5080-class GPU, they point to
  4.9 / 6.3 / 3.7.
- **Fixed overhead.**
  - *In the march.* Of Minimal's 75% above throughput, about 40 points is
    latency and fill by the estimate. A slower GPU would not multiply
    that part. The other 35 points is coherence cost, which it still pays.
    For Low the split is about 12 and 33.
  - *Outside the march loop.* The `nomarch_live` arm is now 3–5%. That is
    per-pixel GPU work, which scales with the GPU. It was 21–36% in
    section 2 because the sea still marched its own cloud shadow then.
  - *CPU and driver.* This harness cannot measure the browser's CPU and
    driver cost. The Python host's 0.32 ms per frame outside the march at
    Minimal (70% of its wall clock) includes band waits and the
    accumulate pass. So it is an upper bound for this host and says
    nothing about the browser's. The browser's own number would be its
    Minimal frame time less about 0.14 ms of march.
- **Fixed overhead only makes the gate more conservative.** For a fixed
  cost F and march time g, (F + g) · C ≥ F + g · r whenever C ≥ r. So the
  constant is set by the march ratio on the slowest GPU, where F matters
  least.

### Max at one sample a frame

The proposed ladder makes Max the top rung: full render scale, step
factors 2 and 2, the sun marched (no light cache), the pre-march on, and
Max's own haze and LOD. It is timed at one march a frame, which is what
this host's frames are. The browser's Max today presents eight. High is
timed again in the same run (`tier_cost_max_run1.jsonl`, generated table
[`tier_cost_max.md`](tier_cost_max.md)). No cell spread by more than 10%.

| demo | sun elevation | High ms | Max ms | max/high |
|---|---|---|---|---|
| twpice | 70° | 5.07 | 17.66 | 3.49 |
| congestus | 51.5° | 10.31 | 28.75 | 2.79 |
| dycoms | 30.5° | 7.76 | 35.46 | 4.57 |
| rce | 34.5° | 6.59 | 26.85 | 4.08 |
| stratified | 4.5° | 9.78 | 411.98 | 42.11 |
| rce-dales | 55° | 3.51 | 5.36 | 1.53 |
| rce-sam | 55° | 3.84 | 8.12 | 2.11 |
| marine-congestus-coarse | 4.5° | 5.52 | 118.00 | 21.38 |
| utah-convection-coarse | 45.5° | 3.99 | 9.86 | 2.47 |

The ms columns are means per view. The ratio is the demo sum.

- *Over demos*, max/high is mean 9.39, median 3.49, max 42.11 (stratified)
  and min 1.53.
- *Over views*, it is mean 10.0, median 3.24 and max 83.7
  (marine-congestus-coarse bench_thick_lit).
- *Without the two demos whose sun is at 4.5°*, it is mean 3.01, median
  2.79 and max 4.57 (dycoms).
- *Means over the 90 cells*: High 6.26 ms (6.20 in run 1), Max 73.6 ms.
- *March not entered*: High 0.18 ms, Max 3.05 ms. At Max the sea marches
  its own cloud shadow.

*Interpretation.* Max's cost is the sun march.

- **Low suns.** The march's length goes as 1 / sin(elevation), and in a
  periodic domain a low sun's ray keeps crossing the deck. The two demos
  with the sun at 4.5° cost 21 and 42 times High. Max on stratified is
  0.41 s per march, so today's eight-sample Max would take about 3.3 s
  of GPU per frame there.
- **Higher suns.** With the sun at 30–70°, Max is 1.5–4.6 times High.
  That spread is the cloud along the sun's path, not only the elevation.
- **The ladder.** There is no one max/high ratio to place a ladder on. A
  rung three times High would hold for the seven demos with higher suns
  (max 4.6) only if the sun march were bounded at low elevations. That
  could be a cap on the light march's length or steps, or keeping the
  light cache below some elevation. Either changes the look at a low sun,
  which is where the marched sun differs most from the cache.

## 7. Cheaper density-gradient stencils

`sigma_gradient` (volume.wgsl) blends two central-difference stencils.

- *The fine stencil* takes 6 taps at one voxel, pinned.
- *The coarse stencil* takes 6 taps at a radius of distance × tan 2°. The
  radius is floored at the fine one, and the coarse stencil borrows the
  fine taps on any axis where the two radii are equal.

A far sample therefore reads 12 taps. In the menu (section 4), dropping
the fine taps (grad_coarse) saved less than dropping the coarse taps
(grad_fine). So the wide taps are the dearer ones, presumably because
they are less coherent.

The candidates here keep both scales with fewer fetches.
`tools/soar_gradient_stencils.py` holds them, and `soar_menu.py --set
gradient` runs them.

| candidate | fine | coarse | fetches (far sample) |
|---|---|---|---|
| tetr_both | tetrahedral, 4 taps at radius h | tetrahedral | 8 |
| tetr_fine | tetrahedral | central | 10 |
| tetr_coarse | central | tetrahedral | 10 |
| tetr_alt | tetrahedral | tetrahedral, with the stencil's sign flipped per pixel and frame | 8 |
| fwd_both | forward differences off the centre sigma | forward | 6 |
| fwd_coarse | central | forward | 9 |
| fwd_alt | forward | forward, sign flipped per pixel and frame | 6 |
| reuse50 | central | central, kept until the sample has moved half a coarse radius | 6 + refreshes |
| reuse100 | central | the same, one coarse radius | 6 + refreshes |
| grad_coarse (control) | none | central | 6 |
| no_gradient (control) | none | none | 0 |

The alternating variants let accumulation average the one-sided error.
Under reuse, the cache is dropped wherever the coarse radius equals the
fine one. On question (d), the confidence `|g| · min(h) / (sigma + 1e-4)`
is built from the stencil's own taps, so it needs no taps of its own.

### Numbers

The core five demos by day, ten views each, on this branch at `0f61214`
(pre-march off at Low and Minimal). Run `grad1`; the raw records are
`gradient_run1.jsonl` and the generated table is
[`gradient.md`](gradient.md).

- *Time* is the change in summed march GPU time against stock.
- *RMS* and *max* are against stock on the converged 32-frame picture, in
  8-bit display levels.
- *1-frame* is the same RMS on one unjittered frame.
- *Grain* is stock against stock at other frame indices.

| tier (grain) | tetr_both | tetr_fine | tetr_coarse | tetr_alt | fwd_both | fwd_coarse | fwd_alt | reuse50 | reuse100 | grad_coarse | no_gradient |
|---|---|---|---|---|---|---|---|---|---|---|---|
| minimal (0.63) | −11.8% / 1.98 | −2.4% / 0.68 | −6.5% / 1.85 | −6.7% / 1.64 | −20.9% / 2.47 | −15.2% / 2.13 | −12.6% / 1.45 | −11.7% / 0.44 | −15.0% / 0.78 | −15.7% / 0.54 | −42.9% / 4.27 |
| low (0.47) | −13.0% / 2.40 | −1.9% / 0.80 | −9.2% / 2.24 | −4.1% / 2.01 | −20.1% / 3.01 | −13.3% / 2.66 | −13.6% / 1.74 | −8.3% / 0.59 | −11.2% / 1.08 | −15.2% / 0.79 | −39.1% / 5.03 |
| medium (0.33) | −10.4% / 2.72 | −1.4% / 0.87 | −3.5% / 2.55 | −2.3% / 2.31 | −14.3% / 3.42 | −7.0% / 3.06 | −7.2% / 1.97 | −1.4% / 0.87 | −3.4% / 1.46 | −13.2% / 1.05 | −29.3% / 5.54 |
| high (0.29) | −9.7% / 3.24 | −1.2% / 0.97 | −3.0% / 3.06 | −1.8% / 2.80 | −13.5% / 3.95 | −6.5% / 3.58 | −6.7% / 2.29 | +0.2% / 1.20 | −1.6% / 1.85 | −13.2% / 1.42 | −29.6% / 6.21 |

Each cell is time / converged RMS. The worst pixel (max) is 26–48 levels
for the tetrahedral and forward stencils, 11–51 for reuse, and 13–24 for
grad_coarse. The one-frame RMS, as a multiple of the converged RMS, is:

- 1.0–1.3× for the fixed stencils;
- 1.1–1.4× for grad_coarse;
- 1.1–2.0× for reuse, the most at Minimal;
- 1.2–1.4× for tetr_alt;
- about 2× for fwd_alt (3.17 / 3.65 / 3.99 / 4.48).

Views over their own grain, of 50 (Minimal / Low / Medium / High):

| candidate | views over grain |
|---|---|
| tetr_both | 47 / 50 / 50 / 50 |
| tetr_fine | 22 / 26 / 37 / 40 |
| tetr_coarse | 41 / 48 / 48 / 48 |
| tetr_alt | 46 / 50 / 50 / 50 |
| fwd_both | 47 / 50 / 50 / 50 |
| fwd_coarse | 41 / 48 / 48 / 48 |
| fwd_alt | 43 / 49 / 50 / 50 |
| reuse50 | 14 / 25 / 39 / 45 |
| reuse100 | 31 / 42 / 45 / 47 |
| grad_coarse | 25 / 36 / 42 / 43 |

The median per-view RMS / grain, Minimal / Low / Medium / High:

| candidate | median RMS / grain |
|---|---|
| grad_coarse | 0.99 / 1.79 / 2.99 / 4.29 |
| reuse50 | 0.56 / 1.05 / 2.55 / 3.77 |
| reuse100 | 1.62 / 2.64 / 4.24 / 5.53 |
| tetr_fine | 0.80 / 1.09 / 1.81 / 2.33 |

The per-demo time change, as the range over the five demos:

| candidate | minimal | low | medium | high |
|---|---|---|---|---|
| grad_coarse | −21 to −12% | −24 to −11% | −20 to −9% | −18 to −10% |
| reuse50 | −17 to +6% | −15 to +5% | −8 to +7% | −8 to +5% |
| reuse100 | −24 to +6% | −21 to +3% | −13 to +5% | −11 to +4% |
| tetr_fine | −5 to −1% | −4 to −1% | −4 to −1% | −3 to 0% |

Per demo, the reuse50 change at Minimal / Low / Medium / High is:

| demo | reuse50 |
|---|---|
| dycoms | −17 / −14 / −8 / −8 |
| congestus | −15 / −14 / −5 / 0 |
| twpice | −13 / −4 / +3 / +4 |
| stratified | 0 / +1 / +5 / +1 |
| rce | +6 / +5 / +7 / +5 |

PNGs are in `gradient_png/`, which is not committed. Each one shows
stock | candidate | |difference| × 8, cropped to the 960x540 window of
largest difference. Each candidate gets the two worst views per demo over
Low, Medium and High pooled, so most of them are at High.

### Interpretation

- **None of the stencils that keep both scales with fewer fetches is
  worth shipping.**
  - *Tetrahedral and forward stencils.* For 3–21% they move the picture
    by 1.5–4 levels RMS, a third to two thirds of what removing the
    gradient does, and push 41–50 of the 50 views over their own grain.
    A 4-tap stencil at the coarse radius loses the smoothing that radius
    exists for.
  - *Alternating them.* That lowers the converged error by 15–40%, but
    not a single frame's: fwd_alt's one-frame RMS is above fwd_both's.
    In flight the frame is what is seen.
  - *tetr_fine.* It keeps the look best of these, but saves only 1–2%.
- **Reuse keeps the look better than anything else here, but its saving
  depends on the field.**
  - At Minimal, reuse50 sits at half the grain in the median view, for
    −12%.
  - The saving is −8 to −17% on dycoms and congestus, nothing on
    stratified, and +5 to +7% on rce. Averaged, it is nothing at Medium
    and High.
  - My guess, untested: the bookkeeping (a cached gradient and position
    carried through the loop, and a distance test per sample) costs a
    roughly fixed amount, while the hits pay off only where the march
    stays in cloud for long stretches. At high resolution the coarse taps
    are coherent between neighbouring pixels and cheap, which would also
    explain why it is worth less at Medium and High.
- **Dropping the fine taps (grad_coarse) is the one robust saving.** It
  saves 13–16% on every tier, and 9–24% on every demo, never positive.
  It was already in section 4's menu; with the pre-march gone from Low and
  Minimal it is now a larger share of their march. The look cost grows
  with the tier: the median view is at its grain at Minimal, and 1.8,
  3.0 and 4.3 times it at Low, Medium and High.
- **What I would ship:**
  - *Minimal:* grad_coarse.
  - *Low:* grad_coarse, if Thomas finds it invisible in flight. 36 of 50
    views are over their own grain there, so it is a by-eye call.
  - *Medium and High:* nothing from this set. The candidates that keep
    the look at Medium and High save under 4%, and the one that saves
    13% is 3–4 times the grain.
- **Worth building instead:** a precomputed coarse gradient, i.e. a
  gradient volume, or a mip chain of sigma, read with one filtered fetch
  at the level the coarse radius asks for. It would replace the six
  dearest, least coherent taps with one fetch.
  - The field is static, so it is a one-time bake, at the cost of the
    memory for a vector volume.
  - Its look against the 6-tap coarse stencil is unknown. A box-filtered
    gradient is a smoothed gradient, which is what the coarse stencil is
    for, so it may be closer than any stencil here.
  - Its saving would be bounded by section 4's grad_fine (−19 to −21%,
    before the pre-march change) less the fetch it adds.

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
uv run --extra dev python tools/soar_tier_cost.py --label tc1 --out tier_cost_run1.jsonl
uv run --extra dev python tools/soar_tier_cost.py --report tier_cost_run1.jsonl --md tier_cost.md
uv run --extra dev python tools/soar_menu.py --set gradient --demo twpice --label grad1 --out gradient_run1.jsonl --png-dir gradient_png
uv run --extra dev python tools/soar_menu.py --report gradient_run1.jsonl --md gradient.md
```

The census and timing take `--demos` (default the core five; these runs
passed all nine). soar_tier_cost defaults to all nine. The ablation and menu run one demo per call, and
`--scene city` for the city scene. `tools/soar_ablate.py`'s docstring has the notes for running them
on an Apple-silicon Mac.
