"""The desert's bounds pyramid holds, and both hosts build the same one.

raymarch.wgsl desert_terrain_march certifies every step it takes over the
desert from cloudyview/desert_bounds.py's pyramid, so a bound that falls
short anywhere is a ray that can pass through the ground. What is pinned:

- Every channel bounds what it claims at every point a step can reach from
  inside its cell (one cell each way), for every mip it serves and every
  band mip weight: checked where each bound is attained (texel centres,
  the band's lattice corners) and at random points between.
- The checker catches an under-bound: the same pyramid rounded down one
  float16 step, and one built without its dilations, both fail it.
- The smooth lid the trace also steps on reads the tile's meta.json
  envelope_m and slope_max; those hold above the exact supremum of
  mip l - mip k (l < k, on the lattice where that difference is bilinear)
  and the exact patch slopes.
- web/soar/desert_bounds.js builds the same bytes (node), the browser's
  loader uploads them and hands the view on, and a desert bound without
  them is an error.
"""

import functools
import gzip
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview import desert_bounds as db
from cloudyview.soar_host import (DESERT_DIR, decode_surface_level,
                                  surface_level_files)

REPO = Path(__file__).resolve().parents[1]
SOAR = REPO / "web" / "soar"
META = json.loads((DESERT_DIR / "meta.json").read_text())
CELL = float(META["cell_m"])
N = int(META["n"])
EXTENT = N * CELL
B2, B3 = db.BAND_MIPS


@functools.lru_cache(maxsize=None)
def heights():
    out = []
    for level, name in enumerate(surface_level_files(META)):
        side = max(1, N >> level)
        raw = gzip.decompress((DESERT_DIR / name).read_bytes())
        out.append(np.frombuffer(decode_surface_level(raw, side, META),
                                 np.float16).reshape(side, side, 2)[..., 0])
    return tuple(out)


@functools.lru_cache(maxsize=None)
def pyramid():
    return tuple(a.astype(np.float64) for a in db.build(list(heights()), CELL))


def _bilinear_grad(a, x, y, texel):
    """The exact gradient of the bilinear read at (x, y)."""
    n = a.shape[0]
    gx = x / texel - 0.5
    gy = y / texel - 0.5
    i = np.floor(gx)
    j = np.floor(gy)
    fx, fy = gx - i, gy - j
    i = i.astype(np.int64) % n
    j = j.astype(np.int64) % n
    i1, j1 = (i + 1) % n, (j + 1) % n
    a = a.astype(np.float64)
    ddx = ((a[j, i1] - a[j, i]) * (1 - fy) + (a[j1, i1] - a[j1, i]) * fy)
    ddy = ((a[j1, i] - a[j, i]) * (1 - fx) + (a[j1, i1] - a[j, i1]) * fx)
    return ddx / texel, ddy / texel


def _near(bound, reduce=np.minimum):
    """The tightest a point in a cell may rely on: a c = 1 channel of any
    of the 3 x 3 cells around it holds there."""
    out = bound
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out = reduce(out, np.roll(np.roll(bound, dy, 0), dx, 1))
    return out


def _samples(rng):
    """Where bounds are attained — every texel centre of the coarse levels,
    a sample of level 0's, the band's lattice corners — and points between."""
    pts = [rng.uniform(0.0, EXTENT, (200_000, 2))]
    for level in range(len(heights())):
        s = CELL * (1 << level)
        c = (np.arange(N >> level) + 0.5) * s
        X, Y = np.meshgrid(c, c)
        xy = np.stack([X.ravel(), Y.ravel()], -1)
        if len(xy) > 60_000:
            xy = xy[rng.choice(len(xy), 60_000, replace=False)]
        pts.append(xy)
    lat = db.lattice(N, CELL)
    pts.append(np.stack([rng.choice(lat, 200_000), rng.choice(lat, 200_000)],
                        -1))
    return np.concatenate(pts)


def violations(pyr, rng=None):
    """Count of points where a channel falls short of what it claims."""
    rng = rng or np.random.default_rng(7)
    xy = _samples(rng)
    x, y = xy[:, 0], xy[:, 1]
    h = heights()
    mips = len(h)
    bad = {}
    for k, lv in enumerate(pyr):
        t = db.BASE_LEVEL + k
        m = lv.shape[1]
        s = EXTENT / m
        ci = (np.floor(x / s).astype(np.int64)) % m
        cj = (np.floor(y / s).astype(np.int64)) % m
        H, G, E, H0 = (lv[0, ..., c] for c in range(4))
        BH, GB, BA, G23 = (lv[1, ..., c] for c in range(4))
        served = range(min(t + 2, mips))
        f = {l: db.bilinear(h[l].astype(np.float64), x, y, CELL * (1 << l))
             for l in served}
        g = {l: np.hypot(*_bilinear_grad(h[l], x, y, CELL * (1 << l)))
             for l in set(served) | {0, B2, B3}}
        near = lambda b: _near(b)[cj, ci]                          # noqa: E731
        count = 0
        for l in served:
            count += np.sum(f[l] > near(H))
            count += np.sum(f[l] > H0[cj, ci])
            count += np.sum(g[l] > near(G))
            if l + 1 in served:
                count += np.sum(np.abs(f[l + 1] - f[l]) > near(E))
        count += np.sum(np.maximum(g[B2], g[B3]) > near(G23))
        # An octave's fade falls along a ray, to 0 at the trace's cut: its
        # bound has to cover the octave there too, where it adds nothing.
        count += np.sum(BH < 0)
        f0 = db.bilinear(h[0].astype(np.float64), x, y, CELL)
        f2 = db.bilinear(h[B2].astype(np.float64), x, y, CELL * (1 << B2))
        f3 = db.bilinear(h[B3].astype(np.float64), x, y, CELL * (1 << B3))
        d0 = _bilinear_grad(h[0], x, y, CELL)
        d2 = _bilinear_grad(h[B2], x, y, CELL * (1 << B2))
        d3 = _bilinear_grad(h[B3], x, y, CELL * (1 << B3))
        for w in (0.0, 0.321928, 1.0):
            band = f0 - ((1 - w) * f2 + w * f3)
            grad = np.hypot(d0[0] - (1 - w) * d2[0] - w * d3[0],
                            d0[1] - (1 - w) * d2[1] - w * d3[1])
            count += np.sum(band > near(BH))
            count += np.sum(np.abs(band) > near(BA))
            count += np.sum(grad > near(GB))
        bad[k] = int(count)
    return bad


@pytest.mark.slow
def test_the_pyramid_bounds_the_ground():
    assert set(violations(pyramid()).values()) == {0}


@pytest.mark.slow
def test_an_under_bound_is_caught():
    down = tuple(np.nextafter(lv.astype(np.float16), np.float16(-np.inf))
                 .astype(np.float64) for lv in pyramid())
    assert sum(violations(down).values()) > 0


@pytest.mark.slow
def test_a_pyramid_without_its_dilations_is_caught(monkeypatch):
    monkeypatch.setattr(db, "dilate", lambda a, reduce=np.maximum: a)
    bare = tuple(a.astype(np.float64)
                 for a in db.build(list(heights()), CELL))
    assert sum(violations(bare).values()) > 0


@pytest.mark.slow
def test_a_band_bound_below_zero_is_caught():
    """Level 0's BH as it was before the clamp (552 cells below 0): an
    octave fading out within a HEIGHT step rose from fade amp BH < 0 to 0,
    and a disc certified clear could end 2 cm inside the ground."""
    h = heights()
    xs = db.lattice(N, CELL)
    bh, _, _ = db.band_patches(list(h), CELL)
    m = N >> db.BASE_LEVEL
    bare = [lv.copy() for lv in pyramid()]
    bare[0][1, ..., 0] = db.ceil_half(
        db.dilate(db.pool_patches(bh, xs, EXTENT, m))).astype(np.float64)
    assert (bare[0][1, ..., 0] < 0).any()
    assert violations(tuple(bare))[0] > 0


def test_ceil_half_rounds_up():
    x = np.array([0.1, -0.1, 1.0, 16.156, -14.27, 3e-6, -3e-6, 0.0, 65504.0])
    y = db.ceil_half(x)
    assert np.all(y.astype(np.float64) >= x)
    below = np.nextafter(y, np.float16(-np.inf)).astype(np.float64)
    assert np.all((below < x) | (y.astype(np.float64) == x))
    for bad in (np.nan, np.inf, 70000.0):
        with pytest.raises(ValueError):
            db.ceil_half(np.array([1.0, bad]))


def test_the_smooth_lid_reads_true_bounds():
    """desert_terrain_march's lid is mip k plus meta's envelope_m[k],
    stepped by slope_max[k], for k up to the lid's top (DESERT_LID_MAX_LEVEL
    = 7), and its other steps take slope_max at the footprint's mip: both
    are bounds, not measurements."""
    h = [a.astype(np.float64) for a in heights()]
    for k in range(1, 8):
        exact = -np.inf
        for l in range(k):
            s_l, s_k = CELL * (1 << l), CELL * (1 << k)
            xs = np.union1d((np.arange(N >> l) + 0.5) * s_l,
                            (np.arange(N >> k) + 0.5) * s_k)
            for c0 in range(0, len(xs), 512):
                X, Y = np.meshgrid(xs, xs[c0:c0 + 512])
                exact = max(exact, np.max(db.bilinear(h[l], X, Y, s_l)
                                          - db.bilinear(h[k], X, Y, s_k)))
        assert META["envelope_m"][k] >= exact, (k, exact)
    for k in range(len(h)):
        steepest = db.patch_slopes(h[k], CELL * (1 << k)).max()
        assert META["slope_max"][k] >= steepest, (k, steepest)
    # The trace takes slope_max at floor(lod) as the steepest of every mip
    # a step can read (none finer): it has to fall with the level.
    assert all(a >= b for a, b in zip(META["slope_max"], META["slope_max"][1:]))


def test_the_octave_margin_is_a_true_bound():
    """DESERT_OCTAVE_MARGIN (raymarch.wgsl) times band_max_m is how far the
    octaves may move the ground: the flora's slabs and the plants' lids
    stand on it (the camera floor evaluates the octaves themselves).
    It has to cover the sum of the gains times the band's largest
    magnitude, which band_max_m (measured at texel centres) does not."""
    import re
    shader = (REPO / "cloudyview" / "soar" / "raymarch.wgsl").read_text()
    wgsl = float(re.search(r"const DESERT_OCTAVE_MARGIN: f32 = ([0-9.]+);",
                           shader).group(1))
    gain = float(np.float32(re.search(
        r"const DESERT_OCTAVE_GAIN: f32 = ([0-9.]+);", shader).group(1)))
    octaves = int(re.search(r"const DESERT_OCTAVES: i32 = (\d+);",
                            shader).group(1))
    largest = pyramid()[-1][1, 0, 0, 2]              # BA of the whole tile
    gains = sum(gain ** k for k in range(1, octaves + 1))
    assert wgsl * META["band_max_m"] >= gains * largest


_JS = textwrap.dedent("""
    import { readFileSync, writeFileSync } from "node:fs";
    import { gunzipSync } from "node:zlib";
    import { decodeSurfaceLevel } from "%(soar)s/scene.js";
    import { fromHalf } from "%(soar)s/half.js";
    import { buildDesertBounds } from "%(soar)s/desert_bounds.js";
    const dir = "%(soar)s/desert";
    const meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8"));
    const levels = meta.files.map((f, l) => {
      const n = Math.max(1, meta.n >> l);
      const bytes = decodeSurfaceLevel(
        new Uint8Array(gunzipSync(readFileSync(`${dir}/${f}`))), n, meta);
      const halves = new Uint16Array(bytes.buffer, bytes.byteOffset,
                                     bytes.byteLength / 2);
      const out = new Float32Array(n * n);
      for (let i = 0; i < n * n; i++) out[i] = fromHalf(halves[i * 2]);
      return out;
    });
    const out = buildDesertBounds(levels, meta.cell_m);
    const all = new Uint16Array(out.reduce((s, a) => s + a.length, 0));
    let o = 0;
    for (const a of out) { all.set(a, o); o += a.length; }
    writeFileSync(process.env.OUT, Buffer.from(all.buffer));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_builds_the_same_pyramid(tmp_path):
    out = tmp_path / "bounds.bin"
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS % {"soar": SOAR.as_posix()}],
        capture_output=True, text=True, cwd=REPO,
        env={**__import__("os").environ, "OUT": str(out)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    js = np.fromfile(out, np.uint16)
    py = np.concatenate([a.ravel() for a in db.build(list(heights()), CELL)])
    assert np.array_equal(js, py.view(np.uint16))


_LOADER_JS = textwrap.dedent("""
    import { readFileSync, existsSync } from "node:fs";
    import * as K from "%(soar)s/constants.js";
    import { loadOceanTile } from "%(soar)s/scene.js";
    import { Renderer } from "%(soar)s/renderer.js";

    globalThis.GPUTextureUsage = { COPY_DST: 2, TEXTURE_BINDING: 4 };
    globalThis.fetch = async (url) => {
      const path = "%(soar)s/" + String(url).split("?")[0];
      if (!existsSync(path)) return new Response(null, { status: 404 });
      return new Response(readFileSync(path));
    };
    const writes = [];
    const device = {
      createTexture: (desc) => ({ desc, destroy: () => {},
                                  createView: (v) => ({ texture: desc, v }) }),
      queue: { writeTexture: (dst, bytes, layout, size) =>
        writes.push({ label: dst.texture.desc.label, level: dst.mipLevel,
                      bytes: bytes.byteLength, size }) },
    };
    const tile = await loadOceanTile(device, "desert", undefined,
      { keepHeights: true, bounds: true, schema: K.SURFACE_TILE_SCHEMAS.desert });
    const bounds = writes.filter((w) => /-bounds$/.test(w.label));
    let refused = null;
    try {
      const r = Object.create(Renderer.prototype);
      r.scene = { desert: true, seaView: null };
      r._slot9View();
    } catch (err) { refused = String(err); }
    process.stdout.write(JSON.stringify({
      bounds, desc: tile.boundsTexture.desc,
      view: tile.seaView?.v ?? null, refused }));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_uploads_the_pyramid_and_needs_it():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e",
         _LOADER_JS % {"soar": SOAR.as_posix()}],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout)
    levels = db.build(list(heights()), CELL)
    assert got["desc"]["format"] == "rgba16float"
    assert got["desc"]["mipLevelCount"] == len(levels)
    assert got["desc"]["size"] == [levels[0].shape[1]] * 2 + [db.LAYERS]
    assert [w["level"] for w in got["bounds"]] == list(range(len(levels)))
    for w, lv in zip(got["bounds"], levels):
        assert w["bytes"] == lv.nbytes
        assert w["size"] == [lv.shape[1], lv.shape[2], db.LAYERS]
    assert got["view"] == {"dimension": "2d-array"}
    assert got["refused"] and "bounds pyramid is missing" in got["refused"]


# The viewer builds the pyramid in a module Worker (desert_bounds.js
# buildDesertBoundsInWorker), off the main thread: that path must give the
# same bytes too. Node has no Web Worker, so this one runs in Chrome over the
# DevTools protocol (tools/soar_cdp.mjs, as the Tint gate does), on the tile
# the page fetches, and hands back a digest of the worker's bytes and of the
# same build on the page's own thread. Without a Chrome on CLOUDYVIEW_CDP_PORT
# (default 9333) it skips, and says so.
_WORKER_JS = textwrap.dedent("""
    import { openTab, serveTree } from "%(tools)s/soar_cdp.mjs";
    const server = await serveTree("%(web)s");
    let tab = null;
    try {
      tab = await openTab(process.env.CDP_PORT);
      await tab.goto(`${server.base}/`);
      const got = await tab.eval(`(async () => {
        const base = "${server.base}/soar";
        const { decodeSurfaceLevel } = await import(base + "/scene.js");
        const { fromHalf } = await import(base + "/half.js");
        const B = await import(base + "/desert_bounds.js");
        const meta = await (await fetch(base + "/desert/meta.json")).json();
        const levels = [];
        for (let l = 0; l < meta.files.length; l++) {
          const n = Math.max(1, meta.n >> l);
          const gz = (await fetch(base + "/desert/" + meta.files[l])).body
            .pipeThrough(new DecompressionStream("gzip"));
          const bytes = decodeSurfaceLevel(
            new Uint8Array(await new Response(gz).arrayBuffer()), n, meta);
          const halves = new Uint16Array(bytes.buffer, bytes.byteOffset,
                                         bytes.byteLength / 2);
          const h = new Float32Array(n * n);
          for (let i = 0; i < n * n; i++) h[i] = fromHalf(halves[i * 2]);
          levels.push(h);
        }
        const digest = async (out) => {
          const all = new Uint16Array(out.reduce((s, a) => s + a.length, 0));
          let o = 0;
          for (const a of out) { all.set(a, o); o += a.length; }
          const d = await crypto.subtle.digest("SHA-256", all.buffer);
          return [...new Uint8Array(d)].map((b) => b.toString(16)
            .padStart(2, "0")).join("");
        };
        const here = await digest(B.buildDesertBounds(levels, meta.cell_m));
        const inWorker = await digest(await B.buildDesertBoundsInWorker(
          levels.map((a) => a.slice()), meta.cell_m));
        return { here, inWorker };
      })()`);
      process.stdout.write(JSON.stringify(got));
    } finally {
      await tab?.close();
      server.close();
    }
""")


def _cdp_port():
    import os
    import urllib.request
    port = os.environ.get("CLOUDYVIEW_CDP_PORT", "9333")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version",
                               timeout=1).read()
    except OSError:
        return None
    return port


def test_the_worker_builds_the_same_pyramid():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    port = _cdp_port()
    if port is None:
        pytest.skip("needs a Chrome with --remote-debugging-port "
                    "(CLOUDYVIEW_CDP_PORT, default 9333); see "
                    "tests/test_tint_compile.py")
    import hashlib
    import os
    proc = subprocess.run(
        ["node", "--input-type=module", "-e",
         _WORKER_JS % {"tools": (REPO / "tools").as_posix(),
                       "web": (REPO / "web").as_posix()}],
        capture_output=True, text=True, cwd=REPO, timeout=300,
        env={**os.environ, "CDP_PORT": port})
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout)
    py = np.concatenate([a.ravel() for a in db.build(list(heights()), CELL)])
    want = hashlib.sha256(py.view(np.uint16).tobytes()).hexdigest()
    assert got["here"] == want
    assert got["inWorker"] == want
