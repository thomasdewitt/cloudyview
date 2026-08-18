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
