## Tiers as flown

| tier | render size | haze | view LOD (deg) | light LOD (deg) | view step (voxels) | light cache |
|---|---|---|---|---|---|---|
| minimal | 320x180 | 1.091 | 1.8 | 4.2 | 4.00 | True |
| low | 768x432 | 0.659 | 1.2 | 2.8 | 3.00 | True |
| medium | 1792x1008 | 0.4027 | 0.63 | 1.47 | 2.50 | True |
| high | 2560x1440 | -0.038 | 0.3 | 0.7 | 2.00 | True |

## Flight-view placement

Heights in metres. base/top: lowest/highest level with >= 1% of columns at sigma > 1e-3 /m; top_any: highest with >= 1% at sigma > 1e-5 /m. cruise peak: the largest sigma within 4 voxels of the cruise camera (0 = clear air).

| demo | periodic | shape | base | top | top_any | cruise z | under z | above z | cruise peak sigma |
|---|---|---|---|---|---|---|---|---|---|
| twpice | False | 1024x1024x206 | 565 | 16116 | 16638 | 6008 | 282 | 18245 | 0 |
| congestus | True | 2048x2048x385 | 540 | 3200 | 3210 | 1471 | 270 | 3477 | 0 |
| dycoms | True | 640x640x140 | 574 | 871 | 871 | 678 | 287 | 901 | 0 |
| rce | True | 540x540x88 | 1849 | 12946 | 14003 | 5733 | 925 | 15218 | 0 |
| stratified | True | 1024x1024x300 | 7474 | 9639 | 12676 | 8232 | 3737 | 13196 | 0 |
| rce-dales | True | 504x504x77 | 935 | 12201 | 13453 | 4878 | 467 | 14705 | 0 |
| rce-sam | True | 480x480x84 | 863 | 13389 | 14137 | 5247 | 432 | 15464 | 0 |
| marine-congestus-coarse | True | 1024x1024x193 | 535 | 3375 | 3395 | 1529 | 268 | 3681 | 0 |
| utah-convection-coarse | True | 512x512x118 | 3500 | 13200 | 14000 | 6895 | 1750 | 15050 | 0 |

## Step census, per demo and tier

Mean per pixel over the views (each view weighs the same). lead/int/trail/cloud are view-march steps; pm = pre-march steps (pm void = those wholly before the first or after the last cloud sample); no-cloud = fraction of pixels whose ray has no cloud sample; zero = all steps with the field zeroed; px / t16 = all steps left by the per-pixel / 16x16 oracle clamp (march + pre-march); warp = mean over 8x4 pixel blocks of the block's largest step count.

| demo | tier | lead | int | trail | cloud | pm | pm void | no-cloud | zero | px | t16 | warp stock | warp px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| twpice | minimal | 17.0 | 1.8 | 2.4 | 16.1 | 18.8 | 5.8 | 0.10 | 84.6 | 34.8 | 40.7 | 39.8 | 24.4 |
| twpice | low | 23.2 | 2.7 | 3.2 | 19.4 | 25.2 | 7.9 | 0.10 | 118.6 | 42.7 | 47.5 | 50.4 | 27.5 |
| twpice | medium | 30.4 | 4.0 | 4.7 | 24.7 | 30.7 | 9.6 | 0.10 | 170.1 | 52.6 | 56.4 | 65.4 | 33.4 |
| twpice | high | 40.6 | 5.8 | 7.1 | 32.0 | 38.5 | 12.1 | 0.10 | 247.3 | 66.9 | 71.2 | 87.3 | 42.8 |
| congestus | minimal | 51.8 | 4.1 | 7.2 | 7.9 | 40.7 | 28.4 | 0.29 | 141.8 | 36.9 | 57.8 | 75.0 | 27.2 |
| congestus | low | 72.6 | 7.0 | 10.9 | 8.5 | 57.4 | 39.0 | 0.27 | 207.9 | 46.0 | 65.2 | 102.4 | 29.7 |
| congestus | medium | 107.6 | 13.7 | 19.1 | 9.3 | 76.4 | 50.1 | 0.25 | 317.8 | 62.2 | 80.8 | 153.1 | 37.5 |
| congestus | high | 162.3 | 26.7 | 34.9 | 10.8 | 92.4 | 60.1 | 0.24 | 491.9 | 86.7 | 109.9 | 239.4 | 56.2 |
| dycoms | minimal | 5.2 | 1.8 | 4.6 | 16.6 | 21.3 | 3.9 | 0.09 | 70.0 | 39.6 | 43.6 | 29.7 | 23.1 |
| dycoms | low | 7.3 | 2.6 | 6.5 | 20.8 | 29.4 | 5.5 | 0.08 | 99.9 | 50.9 | 54.4 | 38.2 | 27.5 |
| dycoms | medium | 11.0 | 3.9 | 9.2 | 27.3 | 39.1 | 7.3 | 0.08 | 150.5 | 66.3 | 69.6 | 52.4 | 35.2 |
| dycoms | high | 17.7 | 6.0 | 13.3 | 38.5 | 51.7 | 10.0 | 0.08 | 231.4 | 90.0 | 94.6 | 76.7 | 49.1 |
| rce | minimal | 7.7 | 5.1 | 4.1 | 15.7 | 8.7 | 2.5 | 0.07 | 43.3 | 32.2 | 35.9 | 34.8 | 28.4 |
| rce | low | 10.4 | 12.2 | 6.2 | 25.1 | 16.9 | 4.1 | 0.05 | 78.2 | 56.1 | 60.2 | 56.6 | 46.0 |
| rce | medium | 12.6 | 20.3 | 8.6 | 34.6 | 24.7 | 5.1 | 0.05 | 118.7 | 80.3 | 84.1 | 79.4 | 63.9 |
| rce | high | 15.8 | 31.3 | 11.8 | 47.7 | 40.3 | 6.7 | 0.05 | 194.4 | 118.3 | 122.9 | 112.2 | 90.3 |
| stratified | minimal | 34.0 | 2.8 | 9.8 | 7.8 | 30.8 | 22.7 | 0.36 | 87.6 | 28.6 | 46.2 | 55.7 | 20.7 |
| stratified | low | 50.0 | 5.5 | 16.1 | 11.2 | 49.6 | 35.2 | 0.27 | 137.9 | 42.4 | 61.1 | 84.2 | 27.6 |
| stratified | medium | 77.1 | 11.9 | 29.8 | 17.0 | 74.5 | 51.1 | 0.23 | 222.8 | 64.7 | 84.9 | 137.1 | 40.7 |
| stratified | high | 123.2 | 26.6 | 57.5 | 27.8 | 97.3 | 65.4 | 0.20 | 361.9 | 104.7 | 133.5 | 237.5 | 71.3 |
| rce-dales | minimal | 20.3 | 1.0 | 4.8 | 4.8 | 8.3 | 6.0 | 0.44 | 40.5 | 11.2 | 15.4 | 32.3 | 10.5 |
| rce-dales | low | 32.5 | 2.5 | 9.8 | 6.2 | 15.8 | 11.5 | 0.37 | 71.7 | 17.2 | 23.0 | 52.4 | 14.7 |
| rce-dales | medium | 45.7 | 5.6 | 15.5 | 7.8 | 23.2 | 16.4 | 0.32 | 107.7 | 25.1 | 31.7 | 76.1 | 20.3 |
| rce-dales | high | 65.3 | 14.1 | 22.4 | 10.2 | 38.5 | 24.2 | 0.29 | 175.3 | 45.5 | 57.8 | 114.7 | 35.0 |
| rce-sam | minimal | 21.4 | 2.0 | 5.1 | 3.7 | 8.2 | 6.2 | 0.38 | 40.7 | 14.3 | 21.1 | 34.2 | 15.1 |
| rce-sam | low | 31.8 | 6.5 | 9.7 | 6.2 | 15.9 | 10.8 | 0.23 | 74.7 | 27.3 | 37.0 | 57.3 | 26.1 |
| rce-sam | medium | 40.7 | 13.5 | 14.1 | 8.5 | 23.1 | 13.8 | 0.19 | 113.6 | 41.1 | 50.7 | 80.5 | 36.4 |
| rce-sam | high | 52.2 | 26.0 | 17.6 | 11.3 | 36.2 | 17.5 | 0.16 | 186.8 | 65.8 | 76.6 | 113.2 | 54.1 |
| marine-congestus-coarse | minimal | 33.6 | 4.7 | 8.3 | 7.8 | 25.9 | 16.9 | 0.29 | 103.9 | 32.6 | 48.8 | 58.3 | 27.0 |
| marine-congestus-coarse | low | 46.0 | 8.2 | 12.1 | 8.4 | 36.4 | 22.8 | 0.27 | 152.7 | 40.9 | 55.4 | 78.1 | 30.0 |
| marine-congestus-coarse | medium | 64.4 | 14.9 | 19.4 | 9.3 | 47.2 | 28.1 | 0.25 | 229.3 | 53.5 | 67.0 | 111.2 | 37.2 |
| marine-congestus-coarse | high | 90.7 | 26.7 | 31.0 | 10.4 | 58.0 | 34.0 | 0.24 | 343.7 | 73.1 | 89.4 | 162.9 | 52.7 |
| utah-convection-coarse | minimal | 21.1 | 2.3 | 4.0 | 7.4 | 10.9 | 7.0 | 0.32 | 53.5 | 18.8 | 27.4 | 37.1 | 18.1 |
| utah-convection-coarse | low | 30.6 | 5.2 | 5.7 | 9.0 | 18.9 | 11.2 | 0.24 | 91.8 | 28.1 | 37.6 | 52.8 | 23.2 |
| utah-convection-coarse | medium | 41.8 | 10.1 | 7.8 | 10.2 | 25.4 | 13.9 | 0.22 | 143.4 | 37.7 | 46.1 | 72.0 | 29.0 |
| utah-convection-coarse | high | 56.4 | 18.1 | 10.7 | 11.5 | 34.2 | 17.4 | 0.20 | 230.2 | 52.8 | 62.5 | 99.3 | 39.5 |

## Step census, per tier over all demos and views

| tier | lead | int | trail | cloud | pm | pm void | stock total | zero total | px | t8 | t16 | t32 | cloud kept by px | no-cloud | subcut | cloud dtau<1e-4 | <1e-3 | >=1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minimal | 23.6 | 2.8 | 5.6 | 9.7 | 19.3 | 11.1 | 61.0 | 74.0 | 27.7 | 33.3 | 37.4 | 42.3 | 0.9999 | 0.26 | 0.0 | 0.3 | 0.6 | 8.8 |
| low | 33.8 | 5.8 | 8.9 | 12.7 | 29.5 | 16.4 | 90.8 | 114.8 | 39.1 | 44.7 | 49.1 | 54.9 | 1.0000 | 0.21 | 0.0 | 0.6 | 1.0 | 11.1 |
| medium | 47.9 | 10.9 | 14.3 | 16.5 | 40.5 | 21.7 | 130.0 | 174.9 | 53.7 | 59.0 | 63.5 | 70.1 | 1.0000 | 0.19 | 0.0 | 1.0 | 1.5 | 14.0 |
| high | 69.4 | 20.1 | 22.9 | 22.3 | 54.1 | 27.5 | 188.8 | 273.7 | 78.2 | 85.1 | 90.9 | 99.7 | 1.0000 | 0.17 | 0.0 | 1.7 | 2.2 | 18.4 |

## As flown vs soar_bench's look

The same views and sampling, with haze and the two LOD angles as the tier sets them (flown) and as soar_bench leaves them, at ViewState's defaults (bench: haze 1, view LOD 0.3 deg). Steps per pixel, view march + pre-march, mean over the demos and views both were counted on.

| tier | flown march | bench march | flown pm | bench pm | bench / flown (all steps) | ends at t_far, flown | bench |
|---|---|---|---|---|---|---|---|
| minimal | 44.7 | 83.1 | 24.1 | 30.3 | 1.65 | 0.46 | 0.45 |
| low | 64.2 | 102.3 | 35.7 | 39.3 | 1.42 | 0.41 | 0.44 |
| medium | 95.3 | 115.7 | 49.1 | 46.2 | 1.12 | 0.39 | 0.44 |
| high | 147.5 | 133.6 | 64.0 | 56.2 | 0.90 | 0.38 | 0.44 |

