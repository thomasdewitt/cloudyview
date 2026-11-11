# soar: the browser app

Fly through a 3D cloud field in a browser tab. It is the witness renderer
under WebGPU with a JavaScript host: no build step, no server-side code, no
Python at run time. `docs/architecture.md` describes the renderer.

Needs a WebGPU browser (Chrome or Edge 113+, Firefox 141+, Safari 26). The
page says so otherwise.

## Run locally

    python3 tools/dev_server.py
    # open http://localhost:8765/soar/

`dev_server.py` is `http.server` with caching off, so a plain reload always
gets the working tree. (With caching on, an edit can leave the browser with
a mix of old and new modules, which fails with confusing missing-export
errors.)

## What is in `soar/`

- `main.js`, `ui.js`, `tutorial.js`: the landing page, the menus, the first-run
  tutorial.
- `viewer.js`: the app once a field is loaded: the frame loop, flight, quality
  tiers, stills and video.
- `renderer.js`: the WebGPU host for the shared shader. `shader.js` joins the
  shader's files into a module, `uniforms.js` fills its uniform block.
- `field.js`, `ingest/`, `optical.js`, `zcrop.js`: reading a NetCDF or Zarr
  file, or a baked demo, into a volume texture.
- `scene.js`, `camera.js`, `flyer.js`, `collide.js`: the sun, the camera, and
  the flight model.
- `capture.js`, `video.js`, `track.js`: stills, video, and recorded flights
  (which `soar TRACK.json` re-renders from Python).
- `bird.*`, `dart.*`, `minimap.js`, `hud.wgsl`: the overlays.
- `constants.js`: every number the app can be tuned by, the quality tiers
  among them.
- `shader/`, `ocean/`, `city/`: symlinks into `cloudyview/soar/`. The shader
  and the surface tiles are shared with the Python renderer.
- `vendor/`: h5wasm (NetCDF-4 reading) and mediabunny (video encoding).

## Assets

`ocean/` and `city/` are committed. The sea is a periodic 100 m patch of
surface normals at 0.1 m, drawn from a multifractal that only exists in
Python, so it ships as data. Regenerate either with

    uv run --extra dev python tools/export_web_assets.py ocean

The seed in `ocean/meta.json` reproduces the tile's levels exactly under the
scaleinvariance version the exporter names. The gzip around them depends on
the zlib that packed it, so a re-run on another machine can rewrite the
files without changing the sea.

`demos/` is the baked demo set. It is derived, a few hundred megabytes, and
not in this repository: `web/soar/demos` is a symlink into the website's
working tree, which is where the set lives. Bake it with
`tools/prebake_demos.py` (needs the source fields in `data/demos/` and a
GPU). To change which demos appear or how they are grouped, without
re-baking:

    uv run python tools/prebake_demos.py --index-only

## Deploy

Do not copy `web/` to the site. Stage what ships:

    uv run python tools/stage_deploy.py --clean

writes `dist/soar/`, one self-contained folder with the modules' filenames
content-fingerprinted. `docs/deploy.md` says where it goes and what to check
afterwards.

## Controls

Click to capture the pointer; WASD and Space/Shift to fly; scroll for speed;
Esc releases the pointer and opens the menu.
