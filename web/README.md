# soar — the browser build

The witness look running under WebGPU with a JavaScript host. Real LES cloud
fields, FIF ocean, the full realism package, distance LOD — in a tab. Verified
2026-07-17: 261 fps at 1280×720 on the RTX 5080 under Chromium.

This is the only soar. The desktop app it was ported from (wgpu-py + glfw +
imgui) was deleted 2026-08-05 and lives in git history. `soar/raymarch.wgsl`
used to be copied here from that engine, which was its source; it is now the
original and is edited in place.

## Run locally

    python3 -m http.server 8765 --directory web
    # open http://localhost:8765/soar/

Serve the `web/` directory rather than `web/soar/` — the viewer looks for a
sibling `demo/` folder one level up.

## Assets

`soar/ocean/` is committed: a periodic 100 m patch of sea surface, generated
from a multifractal that only exists in Python, so it ships with the tool and
works offline. `demo/` is the demo cloud field — derived, gitignored, and
fetched from the repo at run time so the deployed folder stays small.
Regenerate both with:

    uv run python tools/export_web_assets.py

Known issue (2026-08-05): the ocean tile is not reproducible from the seed
recorded in `soar/ocean/meta.json`. `cloudyview.ocean_fif.generate_fif_normals`
does not pass `levy_noise=` to `FIF_ND`, so the cascade draws fresh randomness
on every call and its `rng` argument only steers the boost direction.
Re-running the exporter therefore rewrites the committed `fif_mip*.bin`.

## Deploy

The folder is fully static and self-contained — copy `web/` anywhere (the
website, an S3 bucket). Needs a WebGPU browser (Chrome/Edge 113+, Firefox 141+,
Safari 26); the page shows a friendly message otherwise.

## Controls

Click to capture the pointer; WASD + Space/Shift to fly; scroll for speed; Esc
releases the mouse and opens the menu.
