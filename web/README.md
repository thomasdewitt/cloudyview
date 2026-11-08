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

Either directory works — `demos/` sits inside `soar/`, so the tree you serve
is the tree the site gets.

## Assets

`soar/ocean/` is committed: a periodic 100 m patch of sea surface, generated
from a multifractal that only exists in Python, so it ships with the tool and
works offline: 1024² normals of 0.1 m, eleven mips, each level byte-shuffled
and gzipped (`mip*.bin.gz`, 5.2 MiB). Regenerate it with:

    uv run python tools/export_web_assets.py ocean

`demos/` is the baked demo set — derived, gitignored, and served from the
website rather than from this repo, because it runs to a few hundred megabytes
and binaries in git are forever. Bake it with `tools/prebake_demos.py`, which
needs the source fields in `data/demos/` and a GPU. To reshuffle which demos
appear, or how they are grouped, without re-baking anything:

    uv run python tools/prebake_demos.py --index-only

The seed recorded in `soar/ocean/meta.json` reproduces the tile exactly under
the scaleinvariance it names in the exporter (`OCEAN_BAKED_WITH`): the
decoded levels come out the same. The gzip around them is reproducible only
under the same zlib (another — zlib-ng, say — packs the same levels into
different bytes), so a re-run elsewhere can rewrite the files without
changing the sea. (The tile did not reproduce at all until 2026-08-05:
`generate_fif_normals` was letting `FIF_ND` draw its own cascade noise, so
its `rng` argument steered only the boost direction.)

## Deploy

Do **not** copy `web/` wholesale — it carries `_mockups/` and a few hundred
megabytes of baked demos. Stage exactly what ships:

    uv run python tools/stage_deploy.py --clean

That writes `dist/soar/` — one self-contained folder, demos
included, with the app's filenames content-fingerprinted so the CDN cannot
serve a stale mix of modules. See [`docs/deploy.md`](../docs/deploy.md) for
where it goes and what to check afterwards.

Needs a WebGPU browser (Chrome/Edge 113+, Firefox 141+, Safari 26); the page
explains itself otherwise, and says so again if you click a demo anyway.

## Controls

Click to capture the pointer; WASD + Space/Shift to fly; scroll for speed; Esc
releases the mouse and opens the menu.
