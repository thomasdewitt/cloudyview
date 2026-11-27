Time: change in march-pass GPU time against stock, summed over every (demo, view) of the scene and tier. Picture: difference from stock in 8-bit display levels; RMS is the root of the mean over views of each view's mean square, p99.9 and max the largest over views. noise = stock against stock over other frame indices (the converged render's own grain).

## city

| tier | views | stock ms | arm | time | RMS | p99.9 | max | pm_vertical skips |
|---|---|---|---|---|---|---|---|---|
| minimal | 12 | 12.37 | stock_again | -0.3% | 0.00 | 0.0 | 0 |  |
| minimal | 12 | 12.37 | uplight_4 | -14.2% | 0.51 | 14.1 | 33 |  |
| minimal | 12 | 12.37 | no_premarch | -27.7% | 0.17 | 2.8 | 5 |  |
| minimal | 12 | 12.37 | no_gradient | -10.4% | 2.81 | 25.9 | 32 |  |
| minimal | 12 | 12.37 | noise | – | 2.56 | 36.8 | 67 |  |
| low | 12 | 32.78 | stock_again | -0.5% | 0.00 | 0.0 | 0 |  |
| low | 12 | 32.78 | uplight_4 | -10.4% | 0.68 | 19.5 | 45 |  |
| low | 12 | 32.78 | no_premarch | -26.9% | 0.32 | 5.9 | 10 |  |
| low | 12 | 32.78 | no_gradient | -8.7% | 3.17 | 26.4 | 33 |  |
| low | 12 | 32.78 | noise | – | 1.92 | 28.1 | 108 |  |
| medium | 12 | 70.47 | stock_again | -0.0% | 0.00 | 0.0 | 0 |  |
| medium | 12 | 70.47 | uplight_4 | -8.6% | 0.78 | 22.4 | 52 |  |
| medium | 12 | 70.47 | no_premarch | -15.6% | 0.44 | 8.2 | 13 |  |
| medium | 12 | 70.47 | no_gradient | -7.4% | 3.39 | 26.4 | 34 |  |
| medium | 12 | 70.47 | noise | – | 1.34 | 22.1 | 127 |  |
| high | 12 | 129.04 | stock_again | -0.2% | 0.00 | 0.0 | 0 |  |
| high | 12 | 129.04 | uplight_4 | -8.9% | 0.87 | 24.9 | 56 |  |
| high | 12 | 129.04 | no_premarch | -10.8% | 0.62 | 10.6 | 18 |  |
| high | 12 | 129.04 | no_gradient | -6.7% | 3.61 | 26.2 | 35 |  |
| high | 12 | 129.04 | noise | – | 1.17 | 20.7 | 113 |  |


## day

| tier | views | stock ms | arm | time | RMS | p99.9 | max | pm_vertical skips |
|---|---|---|---|---|---|---|---|---|
| minimal | 50 | 11.67 | stock_again | -0.1% | 0.00 | 0.0 | 0 |  |
| minimal | 50 | 11.67 | grad_coarse | -11.4% | 0.54 | 8.2 | 13 |  |
| minimal | 50 | 11.67 | grad_fine | -19.7% | 2.76 | 18.2 | 24 |  |
| minimal | 50 | 11.67 | pm_step8 | -5.2% | 0.01 | 1.1 | 2 |  |
| minimal | 50 | 11.67 | pm_cap30 | -3.8% | 0.01 | 0.2 | 0 |  |
| minimal | 50 | 11.67 | pm_vertical | +0.0% | 0.00 | 0.0 | 0 | 0.00% (max 0.0%); within 10/20/30 deg: 0.0% / 0.5% / 2.3% |
| minimal | 50 | 11.67 | no_premarch | -27.4% | 0.15 | 3.3 | 4 |  |
| minimal | 50 | 11.67 | no_gradient | -30.3% | 4.26 | 19.2 | 20 |  |
| minimal | 50 | 11.67 | noise | – | 0.63 | 16.7 | 31 |  |
| low | 50 | 41.52 | stock_again | -0.0% | 0.00 | 0.0 | 0 |  |
| low | 50 | 41.52 | grad_coarse | -13.0% | 0.79 | 10.5 | 18 |  |
| low | 50 | 41.52 | grad_fine | -21.1% | 3.24 | 20.1 | 30 |  |
| low | 50 | 41.52 | pm_step8 | -2.8% | 0.03 | 1.7 | 3 |  |
| low | 50 | 41.52 | pm_cap30 | -4.2% | 0.01 | 0.3 | 1 |  |
| low | 50 | 41.52 | pm_vertical | -0.0% | 0.00 | 0.0 | 0 | 0.00% (max 0.0%); within 10/20/30 deg: 0.0% / 0.5% / 2.3% |
| low | 50 | 41.52 | no_premarch | -31.1% | 0.26 | 5.2 | 8 |  |
| low | 50 | 41.52 | no_gradient | -28.8% | 5.02 | 21.0 | 29 |  |
| low | 50 | 41.52 | noise | – | 0.47 | 11.6 | 29 |  |
| medium | 50 | 163.88 | stock_again | +0.1% | 0.00 | 0.0 | 0 |  |
| medium | 50 | 163.88 | grad_coarse | -13.2% | 1.05 | 11.9 | 22 |  |
| medium | 50 | 163.88 | grad_fine | -19.1% | 3.56 | 23.9 | 36 |  |
| medium | 50 | 163.88 | pm_step8 | +2.7% | 0.04 | 2.4 | 4 |  |
| medium | 50 | 163.88 | pm_cap30 | -2.6% | 0.02 | 0.6 | 1 |  |
| medium | 50 | 163.88 | pm_vertical | -0.0% | 0.00 | 0.0 | 0 | 0.00% (max 0.0%); within 10/20/30 deg: 0.0% / 0.5% / 2.3% |
| medium | 50 | 163.88 | no_premarch | -22.6% | 0.37 | 9.0 | 12 |  |
| medium | 50 | 163.88 | no_gradient | -29.1% | 5.54 | 21.2 | 31 |  |
| medium | 50 | 163.88 | noise | – | 0.33 | 6.9 | 25 |  |
| high | 50 | 394.81 | stock_again | -0.0% | 0.00 | 0.0 | 0 |  |
| high | 50 | 394.81 | grad_coarse | -13.8% | 1.42 | 15.3 | 24 |  |
| high | 50 | 394.81 | grad_fine | -18.7% | 3.98 | 29.5 | 39 |  |
| high | 50 | 394.81 | pm_step8 | +2.6% | 0.08 | 3.4 | 6 |  |
| high | 50 | 394.81 | pm_cap30 | -1.4% | 0.04 | 1.4 | 3 |  |
| high | 50 | 394.81 | pm_vertical | +0.1% | 0.00 | 0.0 | 0 | 0.00% (max 0.0%); within 10/20/30 deg: 0.0% / 0.5% / 2.3% |
| high | 50 | 394.81 | no_premarch | -14.3% | 0.66 | 18.3 | 22 |  |
| high | 50 | 394.81 | no_gradient | -30.0% | 6.21 | 22.7 | 39 |  |
| high | 50 | 394.81 | noise | – | 0.29 | 7.1 | 40 |  |

