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
