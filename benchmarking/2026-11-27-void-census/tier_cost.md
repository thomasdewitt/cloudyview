March-pass GPU time, ms per frame, fastest of the repeats (and of the runs). *as flown*: each tier at its render scale of the window. *throughput*: each tier's look timed at 2560x1440 and scaled by the tier's pixel count. *extrapolated*: the throughput time with the per-pixel penalty measured between the two saturate sizes carried down to the tier's own size (an estimate; see the docstring). Ratios are the next tier's time over this tier's.

as flown: 2 cells spread by more than 10% in every run, kept anyway (their fastest): stratified/low/bench_overview, utah-convection-coarse/medium/bench_thick_backlit

throughput: 5 cells spread by more than 10% in every run, kept anyway (their fastest): rce-dales/low/fly_cruise/2560x1440, rce-dales/medium/fly_cruise_back/2560x1440, rce-sam/low/fly_under_up/2560x1440, rce-sam/minimal/fly_above_down/2560x1440, stratified/low/bench_ocean_down/1280x720

## Ratio to the next tier

| step | measure | over | n | mean | median | max | min | max at |
|---|---|---|---|---|---|---|---|---|
| low/minimal | as flown | demos | 9 | 3.66 | 3.66 | 4.90 | 2.64 | stratified |
| low/minimal | as flown | views | 90 | 3.70 | 3.68 | 6.92 | 1.78 | stratified/bench_under_base |
| low/minimal | throughput | demos | 9 | 7.71 | 7.43 | 9.11 | 6.53 | rce-sam |
| low/minimal | throughput | views | 90 | 7.78 | 7.64 | 13.90 | 4.90 | rce-sam/bench_ocean_down |
| low/minimal | extrapolated | demos | 9 | 6.13 | 6.75 | 8.16 | 3.59 | stratified |
| low/minimal | extrapolated | views | 90 | 6.33 | 6.54 | 11.91 | 2.62 | stratified/bench_under_base |
| medium/low | as flown | demos | 9 | 5.70 | 5.48 | 6.28 | 4.93 | rce-dales |
| medium/low | as flown | views | 90 | 5.75 | 5.89 | 8.03 | 2.72 | stratified/fly_under_up |
| medium/low | throughput | demos | 9 | 8.48 | 8.24 | 10.27 | 7.46 | stratified |
| medium/low | throughput | views | 90 | 8.43 | 8.26 | 13.29 | 5.81 | stratified/fly_demo_still |
| medium/low | extrapolated | demos | 9 | 6.70 | 6.57 | 8.40 | 5.85 | stratified |
| medium/low | extrapolated | views | 90 | 6.74 | 6.77 | 9.67 | 3.36 | stratified/fly_under_up |
| high/medium | as flown | demos | 9 | 2.58 | 2.58 | 3.68 | 1.63 | stratified |
| high/medium | as flown | views | 90 | 2.59 | 2.57 | 5.95 | 1.31 | stratified/bench_under_base |
| high/medium | throughput | demos | 9 | 2.90 | 2.86 | 4.10 | 2.19 | stratified |
| high/medium | throughput | views | 90 | 2.91 | 2.76 | 7.50 | 1.89 | stratified/fly_demo_still |
| high/medium | extrapolated | demos | 9 | 2.55 | 2.54 | 3.51 | 1.59 | stratified |
| high/medium | extrapolated | views | 90 | 2.56 | 2.57 | 5.48 | 1.29 | stratified/bench_under_base |

## Per tier

Means over every (demo, view). *not entered*: the same kernel with the march not entered, the per-pixel work outside the march loop. *above throughput*: 1 - throughput / as flown, the part of the as-flown time the GPU spends not full. *above extrapolated*: the same against the extrapolated time, the part that is neither throughput nor fetch coherence. *wall - all passes*: this Python host's per-frame time outside the timed passes: its encoding, its wait on each band before the next is submitted, the untimed accumulate pass at render size and the read-back. Not the browser's. *px cost 1280x720 / 2560x1440*: the tier's per-pixel cost at the smaller saturate size over the larger.

| tier | render size | as flown ms | throughput ms | above throughput | px cost ratio | extrapolated ms | above extrapolated | not entered ms | not entered share | wall ms | wall - all passes ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| minimal | 320x180 | 0.137 | 0.034 | 75% | 1.229 | 0.082 | 40% | 0.007 | 5% | 0.45 | 0.32 |
| low | 768x432 | 0.464 | 0.256 | 45% | 1.242 | 0.407 | 12% | 0.019 | 4% | 1.18 | 0.72 |
| medium | 1792x1008 | 2.584 | 2.205 | 15% | 1.321 | 2.629 | -2% | 0.089 | 3% | 6.03 | 3.44 |
| high | 2560x1440 | 6.202 | 6.205 | -0% | 1.324 | 6.205 | -0% | 0.173 | 3% | 14.89 | 8.69 |

## Per demo: mean ms per view, and the ratios (demo sums)

*as flown*

| demo | minimal | low | medium | high | low/minimal | medium/low | high/medium |
|---|---|---|---|---|---|---|---|
| twpice | 0.154 | 0.406 | 2.100 | 5.092 | 2.64 | 5.17 | 2.42 |
| congestus | 0.356 | 1.143 | 6.264 | 10.205 | 3.21 | 5.48 | 1.63 |
| dycoms | 0.175 | 0.548 | 2.944 | 7.603 | 3.13 | 5.37 | 2.58 |
| rce | 0.079 | 0.368 | 2.304 | 6.598 | 4.68 | 6.27 | 2.86 |
| stratified | 0.087 | 0.425 | 2.627 | 9.670 | 4.90 | 6.19 | 3.68 |
| rce-dales | 0.052 | 0.191 | 1.199 | 3.466 | 3.66 | 6.28 | 2.89 |
| rce-sam | 0.057 | 0.229 | 1.402 | 3.792 | 4.03 | 6.11 | 2.70 |
| marine-congestus-coarse | 0.195 | 0.590 | 2.911 | 5.509 | 3.03 | 4.93 | 1.89 |
| utah-convection-coarse | 0.074 | 0.274 | 1.502 | 3.882 | 3.68 | 5.48 | 2.58 |

*throughput*

| demo | minimal | low | medium | high | low/minimal | medium/low | high/medium |
|---|---|---|---|---|---|---|---|
| twpice | 0.040 | 0.260 | 1.943 | 5.097 | 6.53 | 7.46 | 2.62 |
| congestus | 0.066 | 0.477 | 4.663 | 10.207 | 7.22 | 9.78 | 2.19 |
| dycoms | 0.051 | 0.340 | 2.661 | 7.612 | 6.68 | 7.83 | 2.86 |
| rce | 0.030 | 0.271 | 2.184 | 6.603 | 8.93 | 8.06 | 3.02 |
| stratified | 0.027 | 0.229 | 2.357 | 9.676 | 8.53 | 10.27 | 4.10 |
| rce-dales | 0.017 | 0.135 | 1.115 | 3.469 | 8.03 | 8.26 | 3.11 |
| rce-sam | 0.017 | 0.156 | 1.288 | 3.790 | 9.11 | 8.24 | 2.94 |
| marine-congestus-coarse | 0.037 | 0.259 | 2.259 | 5.513 | 6.92 | 8.73 | 2.44 |
| utah-convection-coarse | 0.024 | 0.179 | 1.377 | 3.879 | 7.43 | 7.69 | 2.82 |

*extrapolated*

| demo | minimal | low | medium | high | low/minimal | medium/low | high/medium |
|---|---|---|---|---|---|---|---|
| twpice | 0.069 | 0.349 | 2.108 | 5.097 | 5.09 | 6.04 | 2.42 |
| congestus | 0.299 | 1.074 | 6.429 | 10.207 | 3.59 | 5.98 | 1.59 |
| dycoms | 0.113 | 0.515 | 3.014 | 7.612 | 4.56 | 5.85 | 2.53 |
| rce | 0.042 | 0.325 | 2.319 | 6.603 | 7.78 | 7.13 | 2.85 |
| stratified | 0.040 | 0.328 | 2.756 | 9.676 | 8.16 | 8.40 | 3.51 |
| rce-dales | 0.024 | 0.167 | 1.189 | 3.469 | 6.86 | 7.14 | 2.92 |
| rce-sam | 0.027 | 0.201 | 1.388 | 3.790 | 7.37 | 6.89 | 2.73 |
| marine-congestus-coarse | 0.093 | 0.466 | 2.924 | 5.513 | 5.02 | 6.27 | 1.89 |
| utah-convection-coarse | 0.035 | 0.233 | 1.530 | 3.879 | 6.75 | 6.57 | 2.54 |

## Per view

| demo | view | minimal | low | medium | high | low/minimal flown | medium/low flown | high/medium flown | low/minimal thru | medium/low thru | high/medium thru | low/minimal extrap | medium/low extrap | high/medium extrap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| twpice | bench_thick_backlit | 0.050 | 0.188 | 1.188 | 2.678 | 3.78 | 6.31 | 2.25 | 6.77 | 7.24 | 2.30 | 6.31 | 6.91 | 2.22 |
| twpice | bench_under_base | 0.154 | 0.402 | 2.267 | 5.223 | 2.61 | 5.64 | 2.30 | 6.57 | 7.24 | 2.44 | 5.88 | 6.21 | 2.27 |
| twpice | bench_overview | 0.216 | 0.516 | 2.933 | 7.036 | 2.38 | 5.69 | 2.40 | 7.32 | 7.59 | 2.76 | 6.04 | 6.82 | 2.61 |
| twpice | bench_thick_lit | 0.165 | 0.420 | 1.288 | 3.135 | 2.54 | 3.07 | 2.43 | 5.34 | 7.04 | 3.00 | 3.41 | 3.74 | 2.39 |
| twpice | bench_ocean_down | 0.177 | 0.335 | 1.519 | 3.886 | 1.89 | 4.54 | 2.56 | 5.81 | 7.01 | 2.69 | 4.50 | 5.62 | 2.48 |
| twpice | fly_cruise | 0.116 | 0.345 | 1.935 | 4.822 | 2.96 | 5.61 | 2.49 | 6.67 | 7.61 | 2.62 | 5.75 | 6.52 | 2.44 |
| twpice | fly_cruise_back | 0.138 | 0.481 | 2.813 | 7.027 | 3.49 | 5.84 | 2.50 | 6.67 | 7.77 | 2.62 | 5.64 | 6.73 | 2.48 |
| twpice | fly_under_up | 0.183 | 0.480 | 2.299 | 5.588 | 2.63 | 4.79 | 2.43 | 6.27 | 7.45 | 2.61 | 4.88 | 5.65 | 2.36 |
| twpice | fly_above_down | 0.178 | 0.490 | 2.927 | 7.546 | 2.75 | 5.97 | 2.58 | 7.07 | 8.17 | 2.74 | 5.50 | 6.68 | 2.52 |
| twpice | fly_demo_still | 0.159 | 0.403 | 1.833 | 3.976 | 2.54 | 4.55 | 2.17 | 6.27 | 6.81 | 2.41 | 4.70 | 5.58 | 2.18 |
| congestus | bench_thick_backlit | 0.476 | 2.025 | 9.648 | 12.662 | 4.25 | 4.77 | 1.31 | 7.62 | 10.26 | 1.97 | 2.62 | 4.89 | 1.30 |
| congestus | bench_under_base | 0.327 | 1.214 | 8.113 | 13.116 | 3.71 | 6.69 | 1.62 | 8.04 | 10.33 | 2.24 | 3.95 | 6.27 | 1.61 |
| congestus | bench_overview | 0.269 | 1.018 | 6.970 | 10.739 | 3.78 | 6.84 | 1.54 | 7.58 | 11.22 | 2.09 | 4.09 | 7.16 | 1.53 |
| congestus | bench_thick_lit | 0.042 | 0.138 | 1.015 | 1.870 | 3.27 | 7.38 | 1.84 | 5.74 | 7.61 | 2.04 | 5.45 | 8.25 | 1.75 |
| congestus | bench_ocean_down | 0.440 | 0.864 | 2.349 | 3.986 | 1.97 | 2.72 | 1.70 | 4.90 | 5.81 | 2.21 | 3.12 | 3.36 | 1.64 |
| congestus | fly_cruise | 0.344 | 0.834 | 4.398 | 8.604 | 2.42 | 5.28 | 1.96 | 6.72 | 9.48 | 2.45 | 4.25 | 7.20 | 1.81 |
| congestus | fly_cruise_back | 0.390 | 1.401 | 7.832 | 10.691 | 3.60 | 5.59 | 1.37 | 6.67 | 11.04 | 1.89 | 3.82 | 6.41 | 1.29 |
| congestus | fly_under_up | 0.493 | 1.561 | 8.845 | 15.706 | 3.17 | 5.67 | 1.78 | 7.81 | 9.74 | 2.34 | 4.00 | 6.40 | 1.73 |
| congestus | fly_above_down | 0.458 | 1.746 | 10.643 | 18.412 | 3.81 | 6.10 | 1.73 | 8.57 | 9.86 | 2.23 | 4.06 | 6.15 | 1.71 |
| congestus | fly_demo_still | 0.325 | 0.633 | 2.828 | 6.264 | 1.95 | 4.47 | 2.21 | 6.20 | 8.46 | 2.61 | 3.77 | 6.07 | 2.09 |
| dycoms | bench_thick_backlit | 0.158 | 0.664 | 4.111 | 11.079 | 4.20 | 6.19 | 2.69 | 7.08 | 9.50 | 3.01 | 4.92 | 6.88 | 2.63 |
| dycoms | bench_under_base | 0.297 | 0.916 | 4.030 | 10.784 | 3.09 | 4.40 | 2.68 | 6.23 | 7.35 | 3.08 | 3.85 | 4.76 | 2.54 |
| dycoms | bench_overview | 0.061 | 0.214 | 1.311 | 3.172 | 3.54 | 6.11 | 2.42 | 7.13 | 7.47 | 2.50 | 6.39 | 6.86 | 2.39 |
| dycoms | bench_thick_lit | 0.190 | 0.553 | 3.531 | 8.830 | 2.91 | 6.39 | 2.50 | 7.62 | 8.00 | 2.76 | 6.54 | 7.27 | 2.63 |
| dycoms | bench_ocean_down | 0.172 | 0.306 | 1.099 | 2.600 | 1.78 | 3.60 | 2.37 | 5.14 | 5.90 | 2.51 | 3.45 | 4.49 | 2.33 |
| dycoms | fly_cruise | 0.129 | 0.435 | 2.857 | 6.958 | 3.36 | 6.57 | 2.44 | 7.58 | 7.94 | 2.59 | 6.95 | 7.34 | 2.48 |
| dycoms | fly_cruise_back | 0.132 | 0.424 | 2.806 | 6.862 | 3.21 | 6.62 | 2.45 | 7.55 | 7.72 | 2.58 | 6.90 | 7.14 | 2.47 |
| dycoms | fly_under_up | 0.251 | 0.844 | 4.585 | 12.754 | 3.37 | 5.43 | 2.78 | 6.40 | 8.34 | 3.20 | 3.97 | 5.43 | 2.66 |
| dycoms | fly_above_down | 0.104 | 0.300 | 1.648 | 4.213 | 2.88 | 5.49 | 2.56 | 6.13 | 7.59 | 2.62 | 5.13 | 6.12 | 2.29 |
| dycoms | fly_demo_still | 0.257 | 0.827 | 3.461 | 8.780 | 3.21 | 4.19 | 2.54 | 6.15 | 7.15 | 2.97 | 3.77 | 4.65 | 2.42 |
| rce | bench_thick_backlit | 0.111 | 0.510 | 3.115 | 8.297 | 4.59 | 6.11 | 2.66 | 8.24 | 7.55 | 2.79 | 7.39 | 6.75 | 2.63 |
| rce | bench_under_base | 0.085 | 0.387 | 2.545 | 7.457 | 4.57 | 6.57 | 2.93 | 9.18 | 8.49 | 3.11 | 7.98 | 7.48 | 2.90 |
| rce | bench_overview | 0.071 | 0.380 | 2.238 | 6.725 | 5.38 | 5.89 | 3.00 | 9.59 | 8.22 | 3.15 | 8.56 | 7.47 | 3.00 |
| rce | bench_thick_lit | 0.080 | 0.425 | 2.391 | 6.222 | 5.28 | 5.63 | 2.60 | 8.57 | 7.57 | 2.77 | 7.90 | 6.71 | 2.60 |
| rce | bench_ocean_down | 0.046 | 0.233 | 1.380 | 3.241 | 5.04 | 5.94 | 2.35 | 10.75 | 7.20 | 2.43 | 9.05 | 6.41 | 2.32 |
| rce | fly_cruise | 0.072 | 0.324 | 2.063 | 6.281 | 4.47 | 6.37 | 3.04 | 8.71 | 8.27 | 3.23 | 7.63 | 7.10 | 3.02 |
| rce | fly_cruise_back | 0.073 | 0.346 | 2.356 | 7.003 | 4.71 | 6.82 | 2.97 | 9.18 | 8.55 | 3.14 | 7.71 | 7.47 | 2.97 |
| rce | fly_under_up | 0.087 | 0.378 | 2.445 | 7.162 | 4.34 | 6.47 | 2.93 | 8.79 | 8.35 | 3.11 | 7.39 | 7.24 | 2.91 |
| rce | fly_above_down | 0.086 | 0.384 | 2.552 | 7.708 | 4.46 | 6.65 | 3.02 | 8.85 | 8.11 | 3.18 | 7.93 | 7.26 | 3.01 |
| rce | fly_demo_still | 0.074 | 0.311 | 1.959 | 5.884 | 4.21 | 6.30 | 3.00 | 8.74 | 8.40 | 3.20 | 7.07 | 7.39 | 3.00 |
| stratified | bench_thick_backlit | 0.097 | 0.477 | 2.554 | 7.753 | 4.91 | 5.36 | 3.04 | 8.74 | 10.26 | 3.23 | 9.10 | 7.47 | 2.96 |
| stratified | bench_under_base | 0.056 | 0.388 | 2.025 | 12.042 | 6.92 | 5.22 | 5.95 | 11.42 | 11.84 | 7.27 | 11.91 | 7.54 | 5.48 |
| stratified | bench_overview | 0.095 | 0.564 | 2.839 | 9.913 | 5.97 | 5.03 | 3.49 | 8.45 | 10.49 | 3.88 | 8.02 | 8.86 | 3.18 |
| stratified | bench_thick_lit | 0.091 | 0.465 | 3.158 | 10.093 | 5.13 | 6.79 | 3.20 | 8.38 | 10.51 | 3.43 | 7.45 | 8.82 | 3.08 |
| stratified | bench_ocean_down | 0.033 | 0.120 | 0.787 | 2.326 | 3.59 | 6.57 | 2.96 | 8.42 | 9.55 | 3.12 | 6.87 | 7.31 | 2.87 |
| stratified | fly_cruise | 0.147 | 0.560 | 3.299 | 9.152 | 3.80 | 5.90 | 2.77 | 8.02 | 9.01 | 2.97 | 7.74 | 8.14 | 2.79 |
| stratified | fly_cruise_back | 0.129 | 0.474 | 3.394 | 8.903 | 3.68 | 7.16 | 2.62 | 7.84 | 8.65 | 2.76 | 7.23 | 7.91 | 2.63 |
| stratified | fly_under_up | 0.062 | 0.350 | 2.811 | 13.093 | 5.66 | 8.03 | 4.66 | 8.69 | 12.44 | 5.58 | 8.19 | 9.67 | 4.30 |
| stratified | fly_above_down | 0.106 | 0.530 | 3.245 | 11.542 | 5.00 | 6.13 | 3.56 | 8.28 | 10.22 | 3.82 | 7.95 | 8.84 | 3.21 |
| stratified | fly_demo_still | 0.052 | 0.320 | 2.162 | 11.880 | 6.19 | 6.76 | 5.49 | 10.93 | 13.29 | 7.50 | 8.90 | 8.91 | 5.35 |
| rce-dales | bench_thick_backlit | 0.060 | 0.229 | 1.398 | 4.000 | 3.81 | 6.11 | 2.86 | 7.73 | 7.93 | 3.04 | 6.46 | 6.83 | 2.86 |
| rce-dales | bench_under_base | 0.054 | 0.189 | 1.134 | 3.337 | 3.47 | 6.01 | 2.94 | 8.95 | 8.66 | 3.15 | 6.77 | 7.15 | 2.90 |
| rce-dales | bench_overview | 0.050 | 0.186 | 1.095 | 3.092 | 3.72 | 5.89 | 2.82 | 9.03 | 8.68 | 3.08 | 7.46 | 7.55 | 2.91 |
| rce-dales | bench_thick_lit | 0.072 | 0.267 | 1.623 | 4.175 | 3.71 | 6.07 | 2.57 | 7.40 | 7.43 | 2.73 | 6.70 | 6.73 | 2.60 |
| rce-dales | bench_ocean_down | 0.045 | 0.160 | 0.844 | 1.870 | 3.59 | 5.28 | 2.22 | 7.06 | 6.56 | 2.30 | 6.07 | 5.78 | 2.19 |
| rce-dales | fly_cruise | 0.045 | 0.160 | 1.194 | 3.644 | 3.52 | 7.46 | 3.05 | 8.81 | 9.34 | 3.30 | 7.38 | 7.98 | 3.12 |
| rce-dales | fly_cruise_back | 0.044 | 0.147 | 1.087 | 3.398 | 3.36 | 7.38 | 3.13 | 8.85 | 9.66 | 3.32 | 7.82 | 8.59 | 3.17 |
| rce-dales | fly_under_up | 0.052 | 0.189 | 1.183 | 3.655 | 3.67 | 6.25 | 3.09 | 8.98 | 9.03 | 3.27 | 7.59 | 7.63 | 3.02 |
| rce-dales | fly_above_down | 0.053 | 0.214 | 1.415 | 4.572 | 4.07 | 6.62 | 3.23 | 7.33 | 7.90 | 3.50 | 6.65 | 7.25 | 3.27 |
| rce-dales | fly_demo_still | 0.048 | 0.167 | 1.017 | 2.920 | 3.51 | 6.08 | 2.87 | 7.85 | 8.45 | 3.34 | 6.54 | 6.49 | 3.03 |
| rce-sam | bench_thick_backlit | 0.057 | 0.209 | 1.409 | 3.918 | 3.69 | 6.73 | 2.78 | 8.31 | 8.63 | 2.98 | 7.03 | 7.25 | 2.77 |
| rce-sam | bench_under_base | 0.066 | 0.248 | 1.443 | 3.688 | 3.73 | 5.82 | 2.56 | 9.50 | 8.16 | 2.77 | 7.45 | 6.65 | 2.52 |
| rce-sam | bench_overview | 0.055 | 0.222 | 1.292 | 3.824 | 4.01 | 5.82 | 2.96 | 9.69 | 8.31 | 3.24 | 7.54 | 7.00 | 3.04 |
| rce-sam | bench_thick_lit | 0.061 | 0.226 | 1.532 | 4.144 | 3.70 | 6.79 | 2.71 | 9.00 | 8.63 | 2.97 | 7.18 | 7.38 | 2.79 |
| rce-sam | bench_ocean_down | 0.032 | 0.169 | 0.878 | 1.882 | 5.37 | 5.19 | 2.14 | 13.90 | 6.83 | 2.29 | 11.78 | 5.53 | 2.11 |
| rce-sam | fly_cruise | 0.055 | 0.245 | 1.497 | 3.995 | 4.45 | 6.11 | 2.67 | 8.65 | 8.10 | 2.87 | 7.17 | 6.98 | 2.70 |
| rce-sam | fly_cruise_back | 0.058 | 0.217 | 1.317 | 3.673 | 3.72 | 6.06 | 2.79 | 8.11 | 8.07 | 2.99 | 6.45 | 6.69 | 2.79 |
| rce-sam | fly_under_up | 0.064 | 0.259 | 1.551 | 4.144 | 4.06 | 5.99 | 2.67 | 9.04 | 8.52 | 2.90 | 7.22 | 7.08 | 2.64 |
| rce-sam | fly_above_down | 0.060 | 0.249 | 1.656 | 4.880 | 4.18 | 6.65 | 2.95 | 9.23 | 8.48 | 3.16 | 7.43 | 7.29 | 2.98 |
| rce-sam | fly_demo_still | 0.061 | 0.250 | 1.447 | 3.770 | 4.07 | 5.80 | 2.60 | 8.35 | 8.25 | 3.02 | 6.88 | 6.78 | 2.75 |
| marine-congestus-coarse | bench_thick_backlit | 0.247 | 0.709 | 3.138 | 6.180 | 2.87 | 4.43 | 1.97 | 7.04 | 8.61 | 2.51 | 4.30 | 6.26 | 1.93 |
| marine-congestus-coarse | bench_under_base | 0.203 | 0.612 | 3.259 | 5.906 | 3.02 | 5.32 | 1.81 | 7.00 | 8.52 | 2.42 | 5.66 | 5.98 | 1.83 |
| marine-congestus-coarse | bench_overview | 0.163 | 0.460 | 2.709 | 4.615 | 2.82 | 5.89 | 1.70 | 6.74 | 8.52 | 2.46 | 4.98 | 6.36 | 1.79 |
| marine-congestus-coarse | bench_thick_lit | 0.041 | 0.133 | 0.791 | 1.583 | 3.26 | 5.94 | 2.00 | 5.75 | 6.45 | 2.10 | 5.45 | 6.41 | 1.97 |
| marine-congestus-coarse | bench_ocean_down | 0.176 | 0.353 | 1.295 | 2.984 | 2.00 | 3.67 | 2.30 | 5.56 | 6.68 | 2.51 | 3.66 | 4.37 | 2.23 |
| marine-congestus-coarse | fly_cruise | 0.225 | 0.890 | 2.821 | 5.931 | 3.95 | 3.17 | 2.10 | 6.49 | 8.51 | 2.49 | 4.54 | 6.65 | 1.93 |
| marine-congestus-coarse | fly_cruise_back | 0.161 | 0.404 | 2.087 | 4.698 | 2.50 | 5.16 | 2.25 | 6.91 | 8.23 | 2.51 | 5.06 | 6.92 | 2.20 |
| marine-congestus-coarse | fly_under_up | 0.275 | 0.819 | 4.279 | 7.582 | 2.98 | 5.23 | 1.77 | 7.41 | 9.40 | 2.42 | 5.52 | 6.42 | 1.81 |
| marine-congestus-coarse | fly_above_down | 0.223 | 0.707 | 4.475 | 7.843 | 3.17 | 6.33 | 1.75 | 7.62 | 9.56 | 2.41 | 5.98 | 6.63 | 1.81 |
| marine-congestus-coarse | fly_demo_still | 0.234 | 0.817 | 4.261 | 7.771 | 3.49 | 5.22 | 1.82 | 7.67 | 10.09 | 2.42 | 5.14 | 6.23 | 1.80 |
| utah-convection-coarse | bench_thick_backlit | 0.066 | 0.261 | 1.594 | 3.790 | 3.95 | 6.10 | 2.38 | 7.94 | 7.99 | 2.63 | 6.56 | 6.96 | 2.42 |
| utah-convection-coarse | bench_under_base | 0.072 | 0.304 | 1.611 | 4.431 | 4.23 | 5.30 | 2.75 | 7.34 | 7.52 | 3.06 | 7.06 | 6.28 | 2.63 |
| utah-convection-coarse | bench_overview | 0.069 | 0.270 | 1.545 | 4.072 | 3.91 | 5.72 | 2.64 | 7.85 | 7.92 | 2.85 | 6.72 | 6.83 | 2.63 |
| utah-convection-coarse | bench_thick_lit | 0.080 | 0.267 | 1.478 | 3.507 | 3.33 | 5.54 | 2.37 | 7.53 | 7.59 | 2.52 | 6.34 | 6.74 | 2.33 |
| utah-convection-coarse | bench_ocean_down | 0.078 | 0.227 | 1.111 | 2.498 | 2.91 | 4.88 | 2.25 | 7.26 | 6.71 | 2.36 | 6.18 | 5.81 | 2.23 |
| utah-convection-coarse | fly_cruise | 0.079 | 0.246 | 1.367 | 3.087 | 3.11 | 5.56 | 2.26 | 6.79 | 7.23 | 2.41 | 6.06 | 6.44 | 2.26 |
| utah-convection-coarse | fly_cruise_back | 0.074 | 0.270 | 1.435 | 3.255 | 3.67 | 5.31 | 2.27 | 6.72 | 7.12 | 2.41 | 6.00 | 6.33 | 2.26 |
| utah-convection-coarse | fly_under_up | 0.083 | 0.305 | 1.717 | 4.670 | 3.68 | 5.63 | 2.72 | 7.06 | 7.98 | 2.99 | 6.98 | 6.60 | 2.59 |
| utah-convection-coarse | fly_above_down | 0.083 | 0.352 | 2.057 | 6.260 | 4.21 | 5.85 | 3.04 | 8.01 | 8.38 | 3.34 | 7.41 | 7.24 | 2.95 |
| utah-convection-coarse | fly_demo_still | 0.060 | 0.238 | 1.110 | 3.250 | 3.98 | 4.66 | 2.93 | 8.34 | 8.57 | 3.45 | 8.92 | 6.10 | 2.79 |
