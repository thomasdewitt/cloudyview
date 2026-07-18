# soar web — the browser build

The desktop renderer's WGSL (`raymarch.wgsl`, copied verbatim by the export
script) running under WebGPU with a ~600-line JS host. Real TWP-ICE LES
clouds, FIF ocean, the full realism package, distance LOD — in a tab.
Verified 2026-07-17: 261 fps at 1280×720 on the RTX 5080 under Chromium.

## Build the demo payload

The payload (volume, ocean normal mips, uniform template) is derived and
gitignored. Regenerate it on a machine with the GPU:

    uv run python tools/export_web_demo.py

This dumps `demo/` (~13 MB): the fp16 ghost-padded extinction volume from
`data/TWPICE_subvolume_256x256_5km.nc`, a seeded 512² FIF normal mip
chain, and `meta.json` — including the full 21-row uniform block from a
real `InteractiveRenderer`, so the browser can never drift from the
Python renderer's look constants. It also copies `raymarch.wgsl` here.

## Run locally

    cd web && python3 -m http.server 8130
    # open http://localhost:8130/

## Deploy

The folder is fully static and self-contained — copy `web/` anywhere
(the website, an S3 bucket). Needs a WebGPU browser (Chrome/Edge 113+,
Firefox 141+, Safari 26); the page shows a friendly message otherwise.

## Controls

Click to capture the pointer; WASD + Space/Shift to fly; scroll for
speed;
Esc releases the mouse.

## Self-test

`?selftest` renders 90 frames offscreen (no canvas compositing — works
in headless browsers), then POSTs a PNG of the accumulated frame to
`/selftest-shot` and `{fps, gpuErrors}` to `/selftest-log`. Harness:
`temp/perf-2026-07-17/` scripts drive it via Playwright.
