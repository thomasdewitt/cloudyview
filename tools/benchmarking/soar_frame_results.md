# soar per-frame benchmark results

## 2026-08-11 08:52:35 — 912f044 — baseline

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 1.053 | 949.6 |
| high | v4_overview_south | 960x540 | 0.731 | 1368.6 |
| high | v8_ocean_lod | 960x540 | 0.771 | 1296.7 |
| medium | v1_thick_backlit | 720x405 | 0.675 | 1480.9 |
| medium | v4_overview_south | 720x405 | 0.545 | 1833.5 |
| medium | v8_ocean_lod | 720x405 | 0.361 | 2768.1 |
| low | v1_thick_backlit | 576x324 | 0.481 | 2077.1 |
| low | v4_overview_south | 576x324 | 0.374 | 2677.1 |
| low | v8_ocean_lod | 576x324 | 0.269 | 3722.7 |
| potato | v1_thick_backlit | 240x135 | 0.367 | 2725.2 |
| potato | v4_overview_south | 240x135 | 0.331 | 3017.3 |
| potato | v8_ocean_lod | 240x135 | 0.146 | 6850.2 |

## 2026-08-11 10:24:00 — a6c73e5 — merged optimization pass

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 1.059 | 944.7 |
| high | v4_overview_south | 960x540 | 0.750 | 1332.5 |
| high | v8_ocean_lod | 960x540 | 0.956 | 1046.5 |
| medium | v1_thick_backlit | 720x405 | 0.698 | 1432.7 |
| medium | v4_overview_south | 720x405 | 0.582 | 1717.0 |
| medium | v8_ocean_lod | 720x405 | 0.408 | 2448.0 |
| low | v1_thick_backlit | 576x324 | 0.492 | 2032.4 |
| low | v4_overview_south | 576x324 | 0.378 | 2642.2 |
| low | v8_ocean_lod | 576x324 | 0.277 | 3615.4 |
| potato | v1_thick_backlit | 240x135 | 0.371 | 2695.3 |
| potato | v4_overview_south | 240x135 | 0.337 | 2965.2 |
| potato | v8_ocean_lod | 240x135 | 0.144 | 6959.1 |

## 2026-08-11 10:24:42 — a6c73e5 — merged, new tier shapes

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 1.056 | 947.1 |
| high | v4_overview_south | 960x540 | 0.731 | 1367.7 |
| high | v8_ocean_lod | 960x540 | 0.573 | 1746.1 |
| medium | v1_thick_backlit | 720x405 | 0.633 | 1580.5 |
| medium | v4_overview_south | 720x405 | 0.538 | 1859.0 |
| medium | v8_ocean_lod | 720x405 | 0.364 | 2744.2 |
| low | v1_thick_backlit | 576x324 | 0.469 | 2131.8 |
| low | v4_overview_south | 576x324 | 0.365 | 2738.6 |
| low | v8_ocean_lod | 576x324 | 0.235 | 4253.5 |
| potato | v1_thick_backlit | 120x68 | 0.360 | 2777.9 |
| potato | v4_overview_south | 120x68 | 0.328 | 3045.0 |
| potato | v8_ocean_lod | 120x68 | 0.123 | 8130.9 |

## 2026-08-11 20:11:25 — e970d69 — head-e970d69 (visual-tuning + bug fixes)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 1.187 | 842.2 |
| high | v4_overview_south | 960x540 | 0.822 | 1216.6 |
| high | v8_ocean_lod | 960x540 | 0.892 | 1121.7 |
| medium | v1_thick_backlit | 720x405 | 0.707 | 1413.6 |
| medium | v4_overview_south | 720x405 | 0.626 | 1598.5 |
| medium | v8_ocean_lod | 720x405 | 0.428 | 2337.5 |
| low | v1_thick_backlit | 576x324 | 0.493 | 2026.7 |
| low | v4_overview_south | 576x324 | 0.442 | 2261.9 |
| low | v8_ocean_lod | 576x324 | 0.279 | 3580.1 |
| minimal | v1_thick_backlit | 120x68 | 0.377 | 2655.0 |
| minimal | v4_overview_south | 120x68 | 0.409 | 2447.1 |
| minimal | v8_ocean_lod | 120x68 | 0.145 | 6917.7 |

## 2026-08-12 17:34:06 — ef80a72 — pre-brick baseline (ef80a72 + ingest fixes)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 1.175 | 851.1 |
| high | v4_overview_south | 960x540 | 0.819 | 1220.6 |
| high | v8_ocean_lod | 960x540 | 0.875 | 1142.6 |
| medium | v1_thick_backlit | 720x405 | 0.704 | 1419.8 |
| medium | v4_overview_south | 720x405 | 0.632 | 1582.5 |
| medium | v8_ocean_lod | 720x405 | 0.433 | 2308.1 |
| low | v1_thick_backlit | 576x324 | 0.490 | 2042.8 |
| low | v4_overview_south | 576x324 | 0.437 | 2289.9 |
| low | v8_ocean_lod | 576x324 | 0.279 | 3579.2 |
| minimal | v1_thick_backlit | 120x68 | 0.375 | 2664.6 |
| minimal | v4_overview_south | 120x68 | 0.406 | 2462.1 |
| minimal | v8_ocean_lod | 120x68 | 0.142 | 7057.2 |

## 2026-08-12 17:37:50 — ef80a72 — pre-brick baseline on sparse STEAM field (4x decimated)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field steam_small_c002_s0010_4x.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 11.349 | 88.1 |
| high | v4_overview_south | 960x540 | 11.650 | 85.8 |
| high | v8_ocean_lod | 960x540 | 2.228 | 448.7 |
| medium | v1_thick_backlit | 720x405 | 3.823 | 261.6 |
| medium | v4_overview_south | 720x405 | 3.803 | 262.9 |
| medium | v8_ocean_lod | 720x405 | 1.223 | 817.3 |
| low | v1_thick_backlit | 576x324 | 3.032 | 329.8 |
| low | v4_overview_south | 576x324 | 2.953 | 338.7 |
| low | v8_ocean_lod | 576x324 | 1.174 | 851.7 |
| minimal | v1_thick_backlit | 120x68 | 1.092 | 915.8 |
| minimal | v4_overview_south | 120x68 | 1.469 | 680.6 |
| minimal | v8_ocean_lod | 120x68 | 0.913 | 1095.6 |

## 2026-08-13 08:55:50 — ef80a72 — cross-field speed survey: FIF

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field QC_FIF_Square_512,512,256.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 10.308 | 97.0 |
| high | v4_overview_south | 960x540 | 3.841 | 260.3 |
| high | v8_ocean_lod | 960x540 | 2.356 | 424.4 |
| medium | v1_thick_backlit | 720x405 | 3.915 | 255.4 |
| medium | v4_overview_south | 720x405 | 2.964 | 337.4 |
| medium | v8_ocean_lod | 720x405 | 1.696 | 589.5 |
| low | v1_thick_backlit | 576x324 | 1.931 | 518.0 |
| low | v4_overview_south | 576x324 | 2.238 | 446.9 |
| low | v8_ocean_lod | 576x324 | 1.512 | 661.2 |
| minimal | v1_thick_backlit | 120x68 | 0.618 | 1618.2 |
| minimal | v4_overview_south | 120x68 | 1.089 | 918.3 |
| minimal | v8_ocean_lod | 120x68 | 0.374 | 2672.5 |

## 2026-08-13 08:55:59 — ef80a72 — cross-field speed survey: DYCOMS full source

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field DYCOMS_RF01_640x640x640_dt0.25sec_320_0000043200_W_QN.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 6.869 | 145.6 |
| high | v4_overview_south | 960x540 | 4.590 | 217.9 |
| high | v8_ocean_lod | 960x540 | 4.473 | 223.6 |
| medium | v1_thick_backlit | 720x405 | 2.549 | 392.3 |
| medium | v4_overview_south | 720x405 | 3.905 | 256.1 |
| medium | v8_ocean_lod | 720x405 | 3.717 | 269.0 |
| low | v1_thick_backlit | 576x324 | 1.277 | 783.1 |
| low | v4_overview_south | 576x324 | 3.190 | 313.4 |
| low | v8_ocean_lod | 576x324 | 4.202 | 238.0 |
| minimal | v1_thick_backlit | 120x68 | 0.281 | 3554.2 |
| minimal | v4_overview_south | 120x68 | 1.385 | 722.1 |
| minimal | v8_ocean_lod | 120x68 | 0.696 | 1437.0 |

## 2026-08-13 08:56:23 — ef80a72 — cross-field speed survey: CM1 RCE full source

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field CM1_RCE_small_les300_3D_allvars_hour1200.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 6.213 | 161.0 |
| high | v4_overview_south | 960x540 | 4.101 | 243.8 |
| high | v8_ocean_lod | 960x540 | 2.431 | 411.4 |
| medium | v1_thick_backlit | 720x405 | 3.593 | 278.3 |
| medium | v4_overview_south | 720x405 | 3.188 | 313.6 |
| medium | v8_ocean_lod | 720x405 | 1.500 | 666.5 |
| low | v1_thick_backlit | 576x324 | 2.423 | 412.7 |
| low | v4_overview_south | 576x324 | 1.953 | 511.9 |
| low | v8_ocean_lod | 576x324 | 1.174 | 851.7 |
| minimal | v1_thick_backlit | 120x68 | 1.542 | 648.4 |
| minimal | v4_overview_south | 120x68 | 1.367 | 731.8 |
| minimal | v8_ocean_lod | 120x68 | 0.393 | 2541.3 |

## 2026-08-13 08:58:26 — ef80a72 — cross-field speed survey: TWPICE LPT full QC+QI

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field TWPICE_LPT_3D_QC_0000003450.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 3.356 | 297.9 |
| high | v4_overview_south | 960x540 | 8.824 | 113.3 |
| high | v8_ocean_lod | 960x540 | 2.530 | 395.3 |
| medium | v1_thick_backlit | 720x405 | 1.289 | 776.0 |
| medium | v4_overview_south | 720x405 | 6.183 | 161.7 |
| medium | v8_ocean_lod | 720x405 | 1.961 | 509.9 |
| low | v1_thick_backlit | 576x324 | 0.792 | 1262.7 |
| low | v4_overview_south | 576x324 | 3.980 | 251.3 |
| low | v8_ocean_lod | 576x324 | 1.718 | 582.1 |
| minimal | v1_thick_backlit | 120x68 | 0.206 | 4863.0 |
| minimal | v4_overview_south | 120x68 | 1.509 | 662.7 |
| minimal | v8_ocean_lod | 120x68 | 0.555 | 1801.4 |

## 2026-08-13 09:14:08 — ef80a72 — DYCOMS source z-cropped 216:353 (auto z-crop candidate)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field dycoms_zcrop.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 6.830 | 146.4 |
| high | v4_overview_south | 960x540 | 3.201 | 312.4 |
| high | v8_ocean_lod | 960x540 | 1.230 | 813.1 |

## 2026-08-13 09:42:47 — ef80a72 — DYCOMS, z-crop OFF (control)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field DYCOMS_RF01_640x640x640_dt0.25sec_320_0000043200_W_QN.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 6.882 | 145.3 |
| high | v4_overview_south | 960x540 | 4.596 | 217.6 |
| high | v8_ocean_lod | 960x540 | 4.492 | 222.6 |

## 2026-08-13 09:43:06 — ef80a72 — DYCOMS, z-crop ON (auto, 216-352 of 531)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field DYCOMS_RF01_640x640x640_dt0.25sec_320_0000043200_W_QN.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 6.857 | 145.8 |
| high | v4_overview_south | 960x540 | 3.199 | 312.5 |
| high | v8_ocean_lod | 960x540 | 1.197 | 835.4 |

## 2026-08-13 09:43:23 — ef80a72 — FIF, z-crop ON

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field QC_FIF_Square_512,512,256.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 2.757 | 362.8 |
| high | v4_overview_south | 960x540 | 2.756 | 362.8 |
| high | v8_ocean_lod | 960x540 | 1.051 | 951.6 |

## 2026-08-13 09:43:31 — ef80a72 — CM1 RCE, z-crop ON

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field CM1_RCE_small_les300_3D_allvars_hour1200.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 5.610 | 178.2 |
| high | v4_overview_south | 960x540 | 4.223 | 236.8 |
| high | v8_ocean_lod | 960x540 | 2.412 | 414.6 |

## 2026-08-13 09:43:44 — ef80a72 — STEAM 4x, z-crop ON

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field steam_small_c002_s0010_4x.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 3.649 | 274.1 |
| high | v4_overview_south | 960x540 | 3.892 | 256.9 |
| high | v8_ocean_lod | 960x540 | 1.166 | 857.8 |

## 2026-08-13 09:44:05 — ef80a72 — STEAM 4x, z-crop OFF (control, same session)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field steam_small_c002_s0010_4x.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 3.913 | 255.5 |
| high | v4_overview_south | 960x540 | 4.273 | 234.0 |
| high | v8_ocean_lod | 960x540 | 1.272 | 786.5 |

## 2026-08-13 10:34:10 — bebd40c — TWPICE 256, BRICKED 8^3

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field TWPICE_subvolume_256x256_5km.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 6.027 | 165.9 |
| high | v4_overview_south | 960x540 | 3.762 | 265.8 |
| high | v8_ocean_lod | 960x540 | 3.318 | 301.4 |

## 2026-08-13 10:34:18 — bebd40c — TWPICE 256, dense (control for brick run)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field TWPICE_subvolume_256x256_5km.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 1.167 | 857.0 |
| high | v4_overview_south | 960x540 | 0.808 | 1237.9 |
| high | v8_ocean_lod | 960x540 | 0.894 | 1118.2 |

## 2026-08-13 10:34:36 — bebd40c — FIF, z-crop + BRICKED 8^3 (sparse: 1.3% occupied)

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 64 frames/view · field QC_FIF_Square_512,512,256.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 10.215 | 97.9 |
| high | v4_overview_south | 960x540 | 17.118 | 58.4 |
| high | v8_ocean_lod | 960x540 | 4.004 | 249.7 |

## 2026-08-14 10:46:58 — d88440b

GPU: NVIDIA GeForce RTX 5080 · output 960x540 · 32 frames/view · field TWPICE_subvolume_256x256_5km.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 960x540 | 1.238 | 808.1 |
| high | v4_overview_south | 960x540 | 0.861 | 1161.2 |
| high | v8_ocean_lod | 960x540 | 0.735 | 1360.6 |
| medium | v1_thick_backlit | 576x324 | 1.476 | 677.5 |
| medium | v4_overview_south | 576x324 | 0.481 | 2078.4 |
| medium | v8_ocean_lod | 576x324 | 0.338 | 2961.1 |
| low | v1_thick_backlit | 288x162 | 0.346 | 2889.1 |
| low | v4_overview_south | 288x162 | 0.395 | 2529.3 |
| low | v8_ocean_lod | 288x162 | 0.160 | 6239.2 |
| minimal | v1_thick_backlit | 120x68 | 0.385 | 2600.4 |
| minimal | v4_overview_south | 120x68 | 0.402 | 2490.0 |
| minimal | v8_ocean_lod | 120x68 | 0.141 | 7072.4 |
| hold_low | v1_thick_backlit | 720x405 | 0.849 | 1178.1 |
| hold_low | v4_overview_south | 720x405 | 0.678 | 1475.9 |
| hold_low | v8_ocean_lod | 720x405 | 0.491 | 2036.0 |
| hold_minimal | v1_thick_backlit | 480x270 | 0.574 | 1741.6 |
| hold_minimal | v4_overview_south | 480x270 | 0.510 | 1961.8 |
| hold_minimal | v8_ocean_lod | 480x270 | 0.331 | 3023.9 |

## 2026-08-14 10:47:17 — d88440b

GPU: NVIDIA GeForce RTX 5080 · output 2560x1440 · 32 frames/view · field TWPICE_subvolume_256x256_5km.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 2560x1440 | 7.482 | 133.7 |
| high | v4_overview_south | 2560x1440 | 5.322 | 187.9 |
| medium | v1_thick_backlit | 1536x864 | 2.515 | 397.6 |
| medium | v4_overview_south | 1536x864 | 2.346 | 426.2 |
| low | v1_thick_backlit | 768x432 | 0.654 | 1530.2 |
| low | v4_overview_south | 768x432 | 0.614 | 1627.6 |
| minimal | v1_thick_backlit | 320x180 | 0.332 | 3012.7 |
| minimal | v4_overview_south | 320x180 | 0.403 | 2482.1 |
| hold_low | v1_thick_backlit | 1920x1080 | 4.393 | 227.6 |
| hold_low | v4_overview_south | 1920x1080 | 3.136 | 318.8 |
| hold_minimal | v1_thick_backlit | 1280x720 | 1.948 | 513.4 |
| hold_minimal | v4_overview_south | 1280x720 | 1.334 | 749.9 |

## 2026-08-19 21:15:22 — cb31036 — M1 MacBook Air 8GB, Metal, tier mirror synced to 26b4b0d retune

GPU: Apple M1 · output 2560x1440 · 64 frames/view · field TWPICE_subvolume_256x256_5km.nc

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 2560x1440 | 158.286 | 6.3 |
| high | v4_overview_south | 2560x1440 | 101.423 | 9.9 |
| high | v8_ocean_lod | 2560x1440 | 95.783 | 10.4 |
| medium | v1_thick_backlit | 1792x1008 | 66.619 | 15.0 |
| medium | v4_overview_south | 1792x1008 | 43.888 | 22.8 |
| medium | v8_ocean_lod | 1792x1008 | 39.791 | 25.1 |
| low | v1_thick_backlit | 768x432 | 12.402 | 80.6 |
| low | v4_overview_south | 768x432 | 9.617 | 104.0 |
| low | v8_ocean_lod | 768x432 | 7.218 | 138.5 |
| minimal | v1_thick_backlit | 320x180 | 2.837 | 352.5 |
| minimal | v4_overview_south | 320x180 | 4.229 | 236.5 |
| minimal | v8_ocean_lod | 320x180 | 1.536 | 651.1 |
| hold_low | v1_thick_backlit | 1280x720 | 43.467 | 23.0 |
| hold_low | v4_overview_south | 1280x720 | 32.018 | 31.2 |
| hold_low | v8_ocean_lod | 1280x720 | 26.189 | 38.2 |
| hold_minimal | v1_thick_backlit | 640x360 | 13.449 | 74.4 |
| hold_minimal | v4_overview_south | 640x360 | 11.136 | 89.8 |
| hold_minimal | v8_ocean_lod | 640x360 | 8.202 | 121.9 |

## 2026-08-19 21:20:57 — cb31036 — M1 MacBook Air 8GB, deployed demo case

GPU: Apple M1 · output 2560x1440 · 32 frames/view · demo desert-coarse

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 2560x1440 | 211.116 | 4.7 |
| high | v4_overview_south | 2560x1440 | 283.199 | 3.5 |
| high | v8_ocean_lod | 2560x1440 | 203.164 | 4.9 |
| medium | v1_thick_backlit | 1792x1008 | 82.647 | 12.1 |
| medium | v4_overview_south | 1792x1008 | 104.630 | 9.6 |
| medium | v8_ocean_lod | 1792x1008 | 93.860 | 10.7 |
| low | v1_thick_backlit | 768x432 | 14.187 | 70.5 |
| low | v4_overview_south | 768x432 | 16.901 | 59.2 |
| low | v8_ocean_lod | 768x432 | 16.032 | 62.4 |
| minimal | v1_thick_backlit | 320x180 | 3.894 | 256.8 |
| minimal | v4_overview_south | 320x180 | 4.831 | 207.0 |
| minimal | v8_ocean_lod | 320x180 | 4.507 | 221.9 |
| hold_low | v1_thick_backlit | 1280x720 | 61.897 | 16.2 |
| hold_low | v4_overview_south | 1280x720 | 82.526 | 12.1 |
| hold_low | v8_ocean_lod | 1280x720 | 59.875 | 16.7 |
| hold_minimal | v1_thick_backlit | 640x360 | 20.107 | 49.7 |
| hold_minimal | v4_overview_south | 640x360 | 24.634 | 40.6 |
| hold_minimal | v8_ocean_lod | 640x360 | 19.388 | 51.6 |

## 2026-08-19 21:23:08 — cb31036 — M1 MacBook Air 8GB, deployed demo case

GPU: Apple M1 · output 2560x1440 · 32 frames/view · demo rce

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 2560x1440 | 883.865 | 1.1 |
| high | v4_overview_south | 2560x1440 | 578.982 | 1.7 |
| high | v8_ocean_lod | 2560x1440 | 390.267 | 2.6 |
| medium | v1_thick_backlit | 1792x1008 | 269.962 | 3.7 |
| medium | v4_overview_south | 1792x1008 | 188.991 | 5.3 |
| medium | v8_ocean_lod | 1792x1008 | 123.516 | 8.1 |
| low | v1_thick_backlit | 768x432 | 39.478 | 25.3 |
| low | v4_overview_south | 768x432 | 32.919 | 30.4 |
| low | v8_ocean_lod | 768x432 | 18.696 | 53.5 |
| minimal | v1_thick_backlit | 320x180 | 8.482 | 117.9 |
| minimal | v4_overview_south | 320x180 | 9.357 | 106.9 |
| minimal | v8_ocean_lod | 320x180 | 4.082 | 245.0 |
| hold_low | v1_thick_backlit | 1280x720 | 233.392 | 4.3 |
| hold_low | v4_overview_south | 1280x720 | 163.999 | 6.1 |
| hold_low | v8_ocean_lod | 1280x720 | 103.170 | 9.7 |
| hold_minimal | v1_thick_backlit | 640x360 | 72.998 | 13.7 |
| hold_minimal | v4_overview_south | 640x360 | 58.169 | 17.2 |
| hold_minimal | v8_ocean_lod | 640x360 | 33.321 | 30.0 |

## 2026-08-19 21:28:04 — cb31036 — M1 MacBook Air 8GB, deployed demo case

GPU: Apple M1 · output 2560x1440 · 32 frames/view · demo dycoms

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| high | v1_thick_backlit | 2560x1440 | 2638.867 | 0.4 |
| high | v4_overview_south | 2560x1440 | 590.992 | 1.7 |
| high | v8_ocean_lod | 2560x1440 | 220.557 | 4.5 |
| medium | v1_thick_backlit | 1792x1008 | 1281.866 | 0.8 |
| medium | v4_overview_south | 1792x1008 | 165.797 | 6.0 |
| medium | v8_ocean_lod | 1792x1008 | 118.394 | 8.4 |
| low | v1_thick_backlit | 768x432 | 405.409 | 2.5 |
| low | v4_overview_south | 768x432 | 22.373 | 44.7 |
| low | v8_ocean_lod | 768x432 | 26.553 | 37.7 |
| minimal | v1_thick_backlit | 320x180 | 124.601 | 8.0 |
| minimal | v4_overview_south | 320x180 | 4.274 | 234.0 |
| minimal | v8_ocean_lod | 320x180 | 15.114 | 66.2 |
| hold_low | v1_thick_backlit | 1280x720 | 997.565 | 1.0 |
| hold_low | v4_overview_south | 1280x720 | 153.551 | 6.5 |
| hold_low | v8_ocean_lod | 1280x720 | 61.334 | 16.3 |
| hold_minimal | v1_thick_backlit | 640x360 | 343.615 | 2.9 |
| hold_minimal | v4_overview_south | 640x360 | 44.824 | 22.3 |
| hold_minimal | v8_ocean_lod | 640x360 | 21.633 | 46.2 |


## 2026-09-23 — the bit-identical performance round: how to read the next six sections

Six runs of this bench on the tip of `bigday/perf` (shader identical to the
tip of that day's round): TWP-ICE plus the deployed rce and dycoms demos,
low and minimal, at 2560x1440 output so the render sizes are the ones the
2026-08-19 M1 rows used. The first three run on the RTX 5080 as it ships.
The last three run with the card clock-locked as a stand-in for the M1:
`nvidia-smi -lgc 180,180 -lmc 405,405` (the lowest graphics and memory
clocks this driver allows), applied and reset inside one `tools/gpulock`
command with a trap. Calibrated the same day by running the M1's own
revision (cb31036, with the bench as it was then) under the lock: the locked
5080 renders these frames in roughly 0.65-0.8x the M1's time, except the
memory-heavy dycoms v1 view, where the M1 is ~2.5x slower still (cache size
cannot be clocked down). So the locked rows are a proxy — trends, not
Mac numbers — and the Mac remains the measurement that decides.

What changed between the M1 rows and these is everything merged since
cb31036, not only the round; for the round alone, see the per-change
ray-pass GPU timings in the commit messages (tools/benchmarking/
soar_pass_timer.py, GPU timestamps, revisions interleaved). Against the
morning's master (a01ae03) the round took the locked ray pass down by 0-26%,
median ~9%, largest on thin/ice cloud (rce) and at Minimal.

One row below is noise, left as measured: unlocked TWP-ICE minimal v1 read
1.348 ms here and 0.172 ms on an immediate 64-frame re-run. At minimal size
a wall-clock frame is mostly host overhead on this card, which is why the
per-change numbers use GPU timestamps instead.

## 2026-09-23 17:10:00 — 9b20cda — perf round tip, unlocked, TWP-ICE field

GPU: NVIDIA GeForce RTX 5080 · output 2560x1440 · 32 frames/view · field TWPICE_subvolume_256x256_5km.nc · per-preset lighting (cache /2, tiered sky probe)

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| low | v1_thick_backlit | 768x432 | 0.473 | 2115.5 |
| low | v4_overview_south | 768x432 | 0.441 | 2265.6 |
| low | v8_ocean_lod | 768x432 | 0.283 | 3530.7 |
| minimal | v1_thick_backlit | 320x180 | 1.348 | 742.1 |
| minimal | v4_overview_south | 320x180 | 0.162 | 6187.7 |
| minimal | v8_ocean_lod | 320x180 | 0.142 | 7047.9 |

## 2026-09-23 17:10:01 — 9b20cda — perf round tip, unlocked, deployed demo rce

GPU: NVIDIA GeForce RTX 5080 · output 2560x1440 · 16 frames/view · demo rce · per-preset lighting (cache /2, tiered sky probe)

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| low | v1_thick_backlit | 768x432 | 0.783 | 1276.7 |
| low | v4_overview_south | 768x432 | 0.569 | 1758.8 |
| low | v8_ocean_lod | 768x432 | 0.415 | 2410.1 |
| minimal | v1_thick_backlit | 320x180 | 0.223 | 4477.1 |
| minimal | v4_overview_south | 320x180 | 0.151 | 6602.7 |
| minimal | v8_ocean_lod | 320x180 | 0.140 | 7137.9 |

## 2026-09-23 17:10:03 — 9b20cda — perf round tip, unlocked, deployed demo dycoms

GPU: NVIDIA GeForce RTX 5080 · output 2560x1440 · 16 frames/view · demo dycoms · per-preset lighting (cache /2, tiered sky probe)

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| low | v1_thick_backlit | 768x432 | 1.807 | 553.3 |
| low | v4_overview_south | 768x432 | 0.462 | 2164.3 |
| low | v8_ocean_lod | 768x432 | 0.540 | 1851.7 |
| minimal | v1_thick_backlit | 320x180 | 0.608 | 1644.7 |
| minimal | v4_overview_south | 320x180 | 0.140 | 7135.9 |
| minimal | v8_ocean_lod | 320x180 | 0.217 | 4601.5 |

## 2026-09-23 17:10:05 — 9b20cda — perf round tip, RTX 5080 clock-locked 180/405 MHz (M1 proxy), TWP-ICE field

GPU: NVIDIA GeForce RTX 5080 · output 2560x1440 · 32 frames/view · field TWPICE_subvolume_256x256_5km.nc · per-preset lighting (cache /2, tiered sky probe)

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| low | v1_thick_backlit | 768x432 | 6.040 | 165.6 |
| low | v4_overview_south | 768x432 | 6.140 | 162.9 |
| low | v8_ocean_lod | 768x432 | 3.764 | 265.7 |
| minimal | v1_thick_backlit | 320x180 | 2.306 | 433.6 |
| minimal | v4_overview_south | 320x180 | 2.176 | 459.6 |
| minimal | v8_ocean_lod | 320x180 | 1.154 | 866.6 |

## 2026-09-23 17:10:08 — 9b20cda — perf round tip, RTX 5080 clock-locked 180/405 MHz (M1 proxy), deployed demo rce

GPU: NVIDIA GeForce RTX 5080 · output 2560x1440 · 16 frames/view · demo rce · per-preset lighting (cache /2, tiered sky probe)

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| low | v1_thick_backlit | 768x432 | 10.794 | 92.6 |
| low | v4_overview_south | 768x432 | 8.631 | 115.9 |
| low | v8_ocean_lod | 768x432 | 5.003 | 199.9 |
| minimal | v1_thick_backlit | 320x180 | 2.932 | 341.0 |
| minimal | v4_overview_south | 320x180 | 1.978 | 505.6 |
| minimal | v8_ocean_lod | 320x180 | 1.451 | 689.4 |

## 2026-09-23 17:10:11 — 9b20cda — perf round tip, RTX 5080 clock-locked 180/405 MHz (M1 proxy), deployed demo dycoms

GPU: NVIDIA GeForce RTX 5080 · output 2560x1440 · 16 frames/view · demo dycoms · per-preset lighting (cache /2, tiered sky probe)

| tier | view | render size | ms/frame | fps |
|------|------|-------------|----------|-----|
| low | v1_thick_backlit | 768x432 | 29.831 | 33.5 |
| low | v4_overview_south | 768x432 | 5.913 | 169.1 |
| low | v8_ocean_lod | 768x432 | 7.158 | 139.7 |
| minimal | v1_thick_backlit | 320x180 | 14.438 | 69.3 |
| minimal | v4_overview_south | 320x180 | 1.481 | 675.4 |
| minimal | v8_ocean_lod | 320x180 | 2.768 | 361.3 |


## 2026-09-25 09:34 — 3fc2eea (start of the 2026-09-23/24 round) vs 0f0a1aa — the loadable big fields

Thomas: "The 640 fields like cm1 were working on m1. Not expecting the 2048
fields to be possible. But it would be nice to get those that are possible
but slower to run better." Scope for this round: cloud output bit-identical.

Method. Each revision on its own tree and hosts, timed by that tree's own
tools/benchmarking/soar_frame_bench.py (960x540 output, 32 frames/view after 4
warmup, ocean surface, z-crop on as the app does, per-preset lighting), run
master / start-of-day / master-again (null) per field, all inside ONE
tools/gpulock: stock clocks first, then the RTX 5080 locked to 180/405 MHz
(the M1 proxy) under a trap that resets it. GPU allocations tallied by
wrapping wgpu-native's create_texture on the tier's device. DYCOMS holds only
SAM's QN, which the loader never infers (cloudyview/io.py), so it is loaded
with liquid_water_var="QN" as --liquid-water-var does; with no variable named
it is refused on both revisions ("Could not find liquid water variable ...
Available variables: ['QN', 'W', 'p']"). No other field failed. Nulls within
±4% at Low (sub-ms frames) and ±2.5% at High.

The proxy does not emulate the M1's caches: the memory-heavy DYCOMS v1 view
(the camera inside the deck) read ~2.5x worse on the real M1 than here
(2026-09-23), so the big-field rows are the least trustworthy proxy rows.

ms/frame, start -> master (master/start):

| field (3D bytes) | tier | view | stock | proxy 180/405 | proxy/stock |
|---|---|---|---|---|---|
| TWPICE 256 (11 MiB) | high | v1 | 0.94 -> 0.93 (0.99) | 13.52 -> 13.97 (1.03) | 15.1x |
| | high | v4 | 0.86 -> 0.86 (1.00) | 12.18 -> 12.38 (1.02) | 14.4x |
| | high | v8 | 0.63 -> 0.63 (1.00) | 9.09 -> 8.97 (0.99) | 14.2x |
| | low | v1/v4/v8 | 0.20/0.21/0.12 -> 0.20/0.19/0.13 | 2.92/2.87/1.36 -> 2.60/2.64/1.21 (0.89/0.92/0.89) | 13-14x |
| TWPICE LPT 512 (116 MiB) | high | v1 | 1.35 -> 1.31 (0.97) | 20.27 -> 19.56 (0.97) | 14.9x |
| | high | v4 | 3.10 -> 2.83 (0.91) | 45.39 -> 41.55 (0.92) | 14.7x |
| | high | v8 | 1.53 -> 1.41 (0.92) | 23.48 -> 21.58 (0.92) | 15.3x |
| | low | v1/v4/v8 | (0.93/0.85/0.89) | 2.39/4.71/4.29 -> 2.15/4.06/3.23 (0.90/0.86/0.75) | 13-15x |
| CM1 RCE full (55 MiB) | high | v1 | 2.52 -> 2.34 (0.93) | 36.88 -> 35.11 (0.95) | 15.0x |
| | high | v4 | 2.13 -> 1.73 (0.81) | 30.77 -> 25.86 (0.84) | 15.0x |
| | high | v8 | 1.44 -> 1.19 (0.83) | 19.64 -> 17.62 (0.90) | 14.8x |
| | low | v1/v4/v8 | (0.95/0.82/0.95) | 3.34/2.81/2.01 -> 3.13/2.18/1.66 (0.94/0.78/0.83) | 12-14x |
| DYCOMS 640 (121 MiB) | high | v1 | 6.67 -> 5.99 (0.90) | 143.93 -> 124.38 (0.86) | 20.8x |
| | high | v4 | 1.52 -> 1.42 (0.93) | 21.69 -> 21.27 (0.98) | 15.0x |
| | high | v8 | 1.26 -> 1.19 (0.94) | 17.16 -> 17.47 (1.02) | 14.7x |
| | low | v1/v4/v8 | (0.86/0.98/0.86) | 14.68/2.27/3.03 -> 13.53/2.11/2.63 (0.92/0.93/0.87) | 24.6x/13.5x/13.6x |

Memory (the High tier's device, both revisions measured): the field's 3D
textures — r16float density plus the /2 sun-tau cache, with no bricks or
sky-probe volume — are 11 / 116 / 55 / 121 MiB. The 2D side (render and
surface targets, the sea) is 22 MiB at the start of the day and 105 MiB on
master, the difference being the moving sea's 256x256x128 rgba16float
9-mip array (85 MiB), which every ocean scene carries whatever the field. The
largest total is DYCOMS at 226 MiB: memory is not the wall for any field
that loads, on an 8 GB M1 or otherwise. Light-cache bakes (per sun change,
not per frame): 16-20 ms at High stock, 50-65 ms at the proxy.

Reading. The round made the big fields 5-25% faster at the M1 proxy and
cost none of them anything outside the null (the largest: CM1 v4 -16% High /
-22% Low, DYCOMS v1 -14%, TWPICE 512 -8%). The proxy/stock ratio is ~14.7x
for every row but one: the ratio of the graphics clocks (2640/180), not of the
memory clocks (37x), so these marches are bound by instruction latency and
count, and they scale with the number of steps a ray takes, which a finer
grid multiplies by design (dt = voxel x step factor). The exception is DYCOMS
v1 from inside the deck, at 21-25x, the one view that is partly
bandwidth-bound, and the one the real M1 punishes a further 2.5x. At the
app's 2560x1440 High, that view is on the order of a second a frame on the
M1, which is Thomas's "basically impossible".

Levers, within bit-identical cloud output, ranked:

1. None measured that would move the big fields materially. What costs them
   is the step count, and the step is fixed by this round's scope.
2. Empty-space skipping of any kind: four attempts, all slower (raymarch.wgsl
   EMPTY-SPACE note); for these fields the bench says latency-bound, which is
   exactly the regime where they lost. Not recommended.
3. Per-step instruction trimming in fs_main (sample addressing, level
   selection on single-level fields): bit-identical in principle, but the day's
   record is that fs_main's register allocation, not its instruction count,
   decides these, and small edits have moved it both ways. Unknown gain,
   probably under 5%.
4. The light cache: a per-sun-change cost of 16-65 ms, not per frame.

Outside this round's scope, for the record: what would move them is the
march step at distance in degrees rather than metres (the view-step LOD the
slider already carries, defaulted up for fine fields); for DYCOMS v1
specifically, a narrower density format (r8unorm halves its bandwidth). Both
change the picture.
