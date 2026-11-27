March-pass GPU time, ms per frame, fastest of the repeats (and of the runs). *as flown*: each tier at its render scale of the window. *throughput*: each tier's look timed at - and scaled by the tier's pixel count. *extrapolated*: the throughput time with the per-pixel penalty measured between the two saturate sizes carried down to the tier's own size (an estimate; see the docstring). Ratios are the next tier's time over this tier's.

## Ratio to the next tier

| step | measure | over | n | mean | median | max | min | max at |
|---|---|---|---|---|---|---|---|---|
| max/high | as flown | demos | 9 | 9.39 | 3.49 | 42.11 | 1.53 | stratified |
| max/high | as flown | views | 90 | 10.01 | 3.24 | 83.68 | 1.34 | marine-congestus-coarse/bench_thick_lit |

## Per tier

Means over every (demo, view). *not entered*: the same kernel with the march not entered, the per-pixel work outside the march loop. *above throughput*: 1 - throughput / as flown, the part of the as-flown time the GPU spends not full. *above extrapolated*: the same against the extrapolated time, the part that is neither throughput nor fetch coherence. *wall - all passes*: this Python host's per-frame time outside the timed passes: its encoding, its wait on each band before the next is submitted, the untimed accumulate pass at render size and the read-back. Not the browser's.

| tier | render size | as flown ms | throughput ms | above throughput | px cost ratio | extrapolated ms | above extrapolated | not entered ms | not entered share | wall ms | wall - all passes ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| high | 2560x1440 | 6.263 | - | - | - | - | - | 0.175 | 3% | 14.89 | 8.63 |
| max | 2560x1440 | 73.559 | - | - | - | - | - | 3.046 | 4% | 82.59 | 9.03 |

## Per demo: mean ms per view, and the ratios (demo sums)

*as flown*

| demo | high | max | max/high |
|---|---|---|---|
| twpice | 5.066 | 17.657 | 3.49 |
| congestus | 10.312 | 28.748 | 2.79 |
| dycoms | 7.758 | 35.461 | 4.57 |
| rce | 6.585 | 26.847 | 4.08 |
| stratified | 9.784 | 411.980 | 42.11 |
| rce-dales | 3.511 | 5.358 | 1.53 |
| rce-sam | 3.839 | 8.120 | 2.11 |
| marine-congestus-coarse | 5.519 | 118.002 | 21.38 |
| utah-convection-coarse | 3.989 | 9.861 | 2.47 |

## Per view

| demo | view | high | max | max/high flown |
|---|---|---|---|---|
| twpice | bench_thick_backlit | 2.690 | 13.143 | 4.89 |
| twpice | bench_under_base | 5.293 | 24.208 | 4.57 |
| twpice | bench_overview | 6.664 | 14.820 | 2.22 |
| twpice | bench_thick_lit | 3.003 | 7.319 | 2.44 |
| twpice | bench_ocean_down | 3.887 | 9.028 | 2.32 |
| twpice | fly_cruise | 4.742 | 21.828 | 4.60 |
| twpice | fly_cruise_back | 7.055 | 33.060 | 4.69 |
| twpice | fly_under_up | 5.677 | 18.461 | 3.25 |
| twpice | fly_above_down | 7.641 | 22.530 | 2.95 |
| twpice | fly_demo_still | 4.007 | 12.172 | 3.04 |
| congestus | bench_thick_backlit | 13.004 | 26.936 | 2.07 |
| congestus | bench_under_base | 13.428 | 57.048 | 4.25 |
| congestus | bench_overview | 10.739 | 19.688 | 1.83 |
| congestus | bench_thick_lit | 1.890 | 12.664 | 6.70 |
| congestus | bench_ocean_down | 4.022 | 13.000 | 3.23 |
| congestus | fly_cruise | 8.635 | 22.858 | 2.65 |
| congestus | fly_cruise_back | 10.733 | 35.555 | 3.31 |
| congestus | fly_under_up | 15.848 | 47.307 | 2.98 |
| congestus | fly_above_down | 18.491 | 35.875 | 1.94 |
| congestus | fly_demo_still | 6.329 | 16.546 | 2.61 |
| dycoms | bench_thick_backlit | 11.710 | 53.912 | 4.60 |
| dycoms | bench_under_base | 11.041 | 27.245 | 2.47 |
| dycoms | bench_overview | 3.193 | 17.315 | 5.42 |
| dycoms | bench_thick_lit | 8.475 | 71.586 | 8.45 |
| dycoms | bench_ocean_down | 2.629 | 5.780 | 2.20 |
| dycoms | fly_cruise | 6.934 | 49.746 | 7.17 |
| dycoms | fly_cruise_back | 6.831 | 54.143 | 7.93 |
| dycoms | fly_under_up | 13.419 | 45.613 | 3.40 |
| dycoms | fly_above_down | 4.377 | 7.179 | 1.64 |
| dycoms | fly_demo_still | 8.975 | 22.086 | 2.46 |
| rce | bench_thick_backlit | 8.482 | 34.178 | 4.03 |
| rce | bench_under_base | 7.556 | 31.493 | 4.17 |
| rce | bench_overview | 6.488 | 23.523 | 3.63 |
| rce | bench_thick_lit | 6.415 | 27.252 | 4.25 |
| rce | bench_ocean_down | 3.281 | 15.630 | 4.76 |
| rce | fly_cruise | 5.901 | 25.109 | 4.26 |
| rce | fly_cruise_back | 7.049 | 29.154 | 4.14 |
| rce | fly_under_up | 7.237 | 31.718 | 4.38 |
| rce | fly_above_down | 7.626 | 26.231 | 3.44 |
| rce | fly_demo_still | 5.819 | 24.188 | 4.16 |
| stratified | bench_thick_backlit | 7.657 | 263.498 | 34.41 |
| stratified | bench_under_base | 11.924 | 404.573 | 33.93 |
| stratified | bench_overview | 10.443 | 529.765 | 50.73 |
| stratified | bench_thick_lit | 10.396 | 500.982 | 48.19 |
| stratified | bench_ocean_down | 2.373 | 86.425 | 36.41 |
| stratified | fly_cruise | 9.034 | 429.816 | 47.58 |
| stratified | fly_cruise_back | 8.835 | 458.165 | 51.86 |
| stratified | fly_under_up | 13.113 | 562.847 | 42.92 |
| stratified | fly_above_down | 12.144 | 602.789 | 49.64 |
| stratified | fly_demo_still | 11.920 | 280.942 | 23.57 |
| rce-dales | bench_thick_backlit | 3.961 | 6.344 | 1.60 |
| rce-dales | bench_under_base | 3.407 | 5.196 | 1.53 |
| rce-dales | bench_overview | 2.956 | 4.413 | 1.49 |
| rce-dales | bench_thick_lit | 4.282 | 7.782 | 1.82 |
| rce-dales | bench_ocean_down | 1.920 | 3.597 | 1.87 |
| rce-dales | fly_cruise | 3.627 | 5.091 | 1.40 |
| rce-dales | fly_cruise_back | 3.361 | 4.670 | 1.39 |
| rce-dales | fly_under_up | 3.733 | 5.010 | 1.34 |
| rce-dales | fly_above_down | 4.703 | 6.836 | 1.45 |
| rce-dales | fly_demo_still | 3.160 | 4.639 | 1.47 |
| rce-sam | bench_thick_backlit | 3.972 | 7.336 | 1.85 |
| rce-sam | bench_under_base | 3.759 | 7.982 | 2.12 |
| rce-sam | bench_overview | 3.746 | 7.522 | 2.01 |
| rce-sam | bench_thick_lit | 4.161 | 8.576 | 2.06 |
| rce-sam | bench_ocean_down | 1.934 | 5.330 | 2.76 |
| rce-sam | fly_cruise | 4.086 | 8.903 | 2.18 |
| rce-sam | fly_cruise_back | 3.716 | 8.222 | 2.21 |
| rce-sam | fly_under_up | 4.271 | 8.925 | 2.09 |
| rce-sam | fly_above_down | 4.776 | 9.976 | 2.09 |
| rce-sam | fly_demo_still | 3.972 | 8.426 | 2.12 |
| marine-congestus-coarse | bench_thick_backlit | 6.152 | 100.487 | 16.33 |
| marine-congestus-coarse | bench_under_base | 6.027 | 117.301 | 19.46 |
| marine-congestus-coarse | bench_overview | 4.679 | 39.392 | 8.42 |
| marine-congestus-coarse | bench_thick_lit | 1.601 | 133.933 | 83.68 |
| marine-congestus-coarse | bench_ocean_down | 3.045 | 82.802 | 27.19 |
| marine-congestus-coarse | fly_cruise | 5.960 | 119.753 | 20.09 |
| marine-congestus-coarse | fly_cruise_back | 4.750 | 179.537 | 37.79 |
| marine-congestus-coarse | fly_under_up | 7.600 | 135.286 | 17.80 |
| marine-congestus-coarse | fly_above_down | 7.560 | 131.776 | 17.43 |
| marine-congestus-coarse | fly_demo_still | 7.813 | 139.754 | 17.89 |
| utah-convection-coarse | bench_thick_backlit | 3.862 | 8.550 | 2.21 |
| utah-convection-coarse | bench_under_base | 4.584 | 12.390 | 2.70 |
| utah-convection-coarse | bench_overview | 4.002 | 8.025 | 2.01 |
| utah-convection-coarse | bench_thick_lit | 3.621 | 8.797 | 2.43 |
| utah-convection-coarse | bench_ocean_down | 2.540 | 7.311 | 2.88 |
| utah-convection-coarse | fly_cruise | 3.156 | 9.822 | 3.11 |
| utah-convection-coarse | fly_cruise_back | 3.311 | 11.094 | 3.35 |
| utah-convection-coarse | fly_under_up | 4.935 | 11.861 | 2.40 |
| utah-convection-coarse | fly_above_down | 6.457 | 13.227 | 2.05 |
| utah-convection-coarse | fly_demo_still | 3.424 | 7.535 | 2.20 |
