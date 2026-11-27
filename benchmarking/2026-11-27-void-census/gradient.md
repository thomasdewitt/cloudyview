Time: change in march-pass GPU time against stock, summed over every (demo, view) of the scene and tier. Picture: difference from stock in 8-bit display levels; RMS is the root of the mean over views of each view's mean square, p99.9 and max the largest over views. noise = stock against stock over other frame indices (the converged render's own grain). 1-frame RMS: the same RMS for a single unjittered frame, where present.

## day

| tier | views | stock ms | arm | time | RMS | p99.9 | max | 1-frame RMS | pm_vertical skips |
|---|---|---|---|---|---|---|---|---|---|
| minimal | 50 | 8.53 | stock_again | -0.0% | 0.00 | 0.0 | 0 | 0.00 |  |
| minimal | 50 | 8.53 | tetr_both | -11.8% | 1.98 | 20.6 | 26 | 2.41 |  |
| minimal | 50 | 8.53 | tetr_fine | -2.4% | 0.68 | 9.4 | 13 | 0.91 |  |
| minimal | 50 | 8.53 | tetr_coarse | -6.5% | 1.85 | 20.4 | 27 | 2.22 |  |
| minimal | 50 | 8.53 | tetr_alt | -6.7% | 1.64 | 16.9 | 24 | 2.36 |  |
| minimal | 50 | 8.53 | fwd_both | -20.9% | 2.47 | 21.5 | 29 | 2.75 |  |
| minimal | 50 | 8.53 | fwd_coarse | -15.2% | 2.13 | 20.1 | 29 | 2.37 |  |
| minimal | 50 | 8.53 | fwd_alt | -12.6% | 1.45 | 11.9 | 15 | 3.17 |  |
| minimal | 50 | 8.53 | reuse50 | -11.7% | 0.44 | 6.1 | 11 | 0.88 |  |
| minimal | 50 | 8.53 | reuse100 | -15.0% | 0.78 | 10.3 | 18 | 1.31 |  |
| minimal | 50 | 8.53 | grad_coarse | -15.7% | 0.54 | 8.2 | 13 | 0.78 |  |
| minimal | 50 | 8.53 | no_gradient | -42.9% | 4.27 | 19.2 | 20 | 4.47 |  |
| minimal | 50 | 8.53 | noise | – | 0.63 | 16.7 | 31 | – |  |
| low | 50 | 28.45 | stock_again | +0.1% | 0.00 | 0.0 | 0 | 0.00 |  |
| low | 50 | 28.45 | tetr_both | -13.0% | 2.40 | 22.2 | 30 | 2.67 |  |
| low | 50 | 28.45 | tetr_fine | -1.9% | 0.80 | 10.2 | 15 | 0.99 |  |
| low | 50 | 28.45 | tetr_coarse | -9.2% | 2.24 | 22.2 | 30 | 2.46 |  |
| low | 50 | 28.45 | tetr_alt | -4.1% | 2.01 | 19.3 | 29 | 2.62 |  |
| low | 50 | 28.45 | fwd_both | -20.1% | 3.01 | 22.5 | 31 | 3.20 |  |
| low | 50 | 28.45 | fwd_coarse | -13.3% | 2.66 | 22.3 | 30 | 2.82 |  |
| low | 50 | 28.45 | fwd_alt | -13.6% | 1.74 | 13.4 | 17 | 3.65 |  |
| low | 50 | 28.45 | reuse50 | -8.3% | 0.59 | 7.7 | 18 | 0.99 |  |
| low | 50 | 28.45 | reuse100 | -11.2% | 1.08 | 13.6 | 24 | 1.53 |  |
| low | 50 | 28.45 | grad_coarse | -15.2% | 0.79 | 10.5 | 18 | 0.98 |  |
| low | 50 | 28.45 | no_gradient | -39.1% | 5.03 | 21.0 | 29 | 5.15 |  |
| low | 50 | 28.45 | noise | – | 0.47 | 11.6 | 29 | – |  |
| medium | 50 | 164.71 | stock_again | +0.0% | 0.00 | 0.0 | 0 | 0.00 |  |
| medium | 50 | 164.71 | tetr_both | -10.4% | 2.72 | 23.3 | 37 | 2.87 |  |
| medium | 50 | 164.71 | tetr_fine | -1.4% | 0.87 | 10.6 | 20 | 1.03 |  |
| medium | 50 | 164.71 | tetr_coarse | -3.5% | 2.55 | 23.4 | 37 | 2.66 |  |
| medium | 50 | 164.71 | tetr_alt | -2.3% | 2.31 | 21.7 | 37 | 2.83 |  |
| medium | 50 | 164.71 | fwd_both | -14.3% | 3.42 | 24.1 | 33 | 3.53 |  |
| medium | 50 | 164.71 | fwd_coarse | -7.0% | 3.06 | 23.8 | 32 | 3.15 |  |
| medium | 50 | 164.71 | fwd_alt | -7.2% | 1.97 | 14.7 | 26 | 3.99 |  |
| medium | 50 | 164.71 | reuse50 | -1.4% | 0.87 | 11.7 | 27 | 1.14 |  |
| medium | 50 | 164.71 | reuse100 | -3.4% | 1.46 | 17.1 | 38 | 1.75 |  |
| medium | 50 | 164.71 | grad_coarse | -13.2% | 1.05 | 11.9 | 22 | 1.18 |  |
| medium | 50 | 164.71 | no_gradient | -29.3% | 5.54 | 21.2 | 31 | 5.62 |  |
| medium | 50 | 164.71 | noise | – | 0.33 | 6.9 | 25 | – |  |
| high | 50 | 393.07 | stock_again | +0.1% | 0.00 | 0.0 | 0 | 0.00 |  |
| high | 50 | 393.07 | tetr_both | -9.7% | 3.24 | 25.2 | 48 | 3.33 |  |
| high | 50 | 393.07 | tetr_fine | -1.2% | 0.97 | 12.2 | 26 | 1.09 |  |
| high | 50 | 393.07 | tetr_coarse | -3.0% | 3.06 | 25.2 | 47 | 3.11 |  |
| high | 50 | 393.07 | tetr_alt | -1.8% | 2.80 | 23.3 | 40 | 3.31 |  |
| high | 50 | 393.07 | fwd_both | -13.5% | 3.95 | 25.2 | 35 | 4.02 |  |
| high | 50 | 393.07 | fwd_coarse | -6.5% | 3.58 | 24.8 | 36 | 3.64 |  |
| high | 50 | 393.07 | fwd_alt | -6.7% | 2.29 | 15.7 | 29 | 4.48 |  |
| high | 50 | 393.07 | reuse50 | +0.2% | 1.20 | 15.7 | 39 | 1.39 |  |
| high | 50 | 393.07 | reuse100 | -1.6% | 1.85 | 19.5 | 51 | 2.04 |  |
| high | 50 | 393.07 | grad_coarse | -13.2% | 1.42 | 15.3 | 24 | 1.51 |  |
| high | 50 | 393.07 | no_gradient | -29.6% | 6.21 | 22.7 | 39 | 6.25 |  |
| high | 50 | 393.07 | noise | – | 0.29 | 7.1 | 40 | – |  |

