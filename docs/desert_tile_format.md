# The desert tile format

Schema `cloudyview.web.desert.v2`.

A desert tile is the ground soar's desert draws: one exactly periodic square
of terrain height and surface lightness, stored as a mip chain beside a
`meta.json`. desert-surface writes it (`desert_surface.tile.write_tile`).
soar reads it, in the browser (`web/soar/scene.js`, `loadOceanTile` and
`desertFloorAt`) and in the Python host (`cloudyview.soar_host`). This
document is the contract between them. An identical copy sits in each repo,
here as `TILE_FORMAT.md` and in cloudyview as `docs/desert_tile_format.md`.

Any change that a reader must know about bumps the schema. That covers a
field's meaning, a new required field, a new format and the byte layout.
The readers refuse any schema they were not written for.

## Geometry

- The tile is `n` × `n` texels covering `tile_extent_m` × `tile_extent_m`
  metres. `cell_m = tile_extent_m / n`, and `n` is a power of two.
- Texel `(i, j)` is the cell centred at `x = (i + ½)·cell_m`,
  `y = (j + ½)·cell_m`, where i is the column (x, along a row) and j is the
  row (y). Arrays are `[j][i]`, and bytes run row-major, i fastest.
- **Exact periodicity.** The tile tiles the plane with no seam: texel `i = n`
  is texel `i = 0`, in both value and every derivative a reader takes. Every
  reader and every derivation below wraps (GPU `repeat` addressing, `% n` on
  the CPU). Nothing is clamped at an edge.
- Readers sample bilinearly between texel centres. Between levels they
  sample trilinearly, as GPU `textureSampleLevel` does with linear mip
  filtering.
- **soar's desert reads only `n = 2048`, that is 12 mips.** The shader's
  tables have 12 rows (`scene.js desertTileRows`). Other sizes are valid
  tiles that soar refuses. The conformance tiles are 64².

## Channels

Two channels per texel, in this order:

| channel | meaning | range |
|---|---|---|
| R: height | Ground height in metres. soar draws the surface at `z = h` (its ground plane is z = 0). The offset is the writer's choice: the shipped bajada is centred on its own mean, and the lab sets its lowest point to 0. | any finite value |
| G: lightness | Surface age and material. 0 is varnished desert pavement, ½ is soil, 1 is fresh wash or playa sand. | [0, 1] |

**Albedo** is a function of lightness only. It is piecewise linear with the
knot at soil, with these endmembers (RGB):

| lightness | material | albedo |
|---|---|---|
| 0 | pavement | (0.17, 0.145, 0.12) |
| ½ | soil | (0.31, 0.235, 0.155) |
| 1 | wash sand | (0.33, 0.285, 0.22) |

This is `raymarch.wgsl desert_albedo` and `desert_surface.tile.albedo_of`.
The table above is what both are held to. Writers quantize lightness to
1/64 steps, which half floats represent exactly. Readers must not rely on
that quantization.

## The mip chain

- There are `mips = log2(n) + 1` levels, from `n`² down to 1². Level k is
  `n >> k` square.
- **Level k+1 is the exact 2×2 box mean of level k**, computed in double
  precision from the level-0 field. Each level is then rounded to the
  storage format on its own, so a level is not the mean of its rounded
  parent. Lightness is linear and box-averaged exactly. Albedo is not, which
  is what `albedo_correction` restores.

## Formats and byte layout

`meta.format` names the texel format. `meta.encoding` and `meta.compression`
name the layout of each level's file.

| format | encoding | compression | level bytes | used by |
|---|---|---|---|---|
| `rg16float` | `planar-byte-shuffle` | `gzip` | n²·2·2 | the shipped tile |
| `rg32float` | `interleaved` | none (key absent) | n²·2·4 | the terrain lab: full precision, served locally |

- **`planar-byte-shuffle`** (rg16float only). The level is stored as one
  plane per channel (R, then G). Each plane holds every texel's
  little-endian half float split by byte: first all the low bytes, then all
  the high bytes. For a level of T texels the file reads
  `R.lo[T] R.hi[T] G.lo[T] G.hi[T]`. This lets gzip see a height field's
  smoothness, which interleaved halves hide from it (13.6 MiB down to
  5.7 MiB at level 0).
- **`interleaved`** means texels in order, each `R G`, little-endian.
- **`gzip`** wraps each level's file in a gzip member (RFC 1952). **The
  contract is the decompressed bytes.** Compressed bytes differ between zlib
  builds and are not compared.
- An `rg32float` tile needs a GPU device with `float32-filterable`.
  `scene.js` refuses it by name on a device without that feature.
  `cloudyview.soar_host` (witness) reads only `rg16float`.
- Level k's file is `meta.files[k]`, relative to the tile directory. The
  writer names them `mip{k}.bin.gz` (rg16float) or `mip{k}.bin`
  (rg32float). Readers use `files`, not the pattern. Readers fetch the files
  under `?v=<schema>`, so a schema bump is a new URL past every cache.
- Before writing, a writer deletes the tile it replaces by the names in that
  tile's `meta.json`, never by a glob. If any name is not a bare file name,
  it deletes nothing.

## meta.json

Every field below is required unless marked otherwise. Values derived from
the heights or lightness are computed from the **stored** levels (rounded to
the format), because that is what a reader sees. Lists have one entry per
level, indexed by k.

| field | type | meaning |
|---|---|---|
| `schema` | string | `"cloudyview.web.desert.v2"` |
| `n` | int | level-0 side, in texels |
| `cell_m` | float | `tile_extent_m / n` |
| `mips` | int | number of levels, `log2(n) + 1` |
| `format` | string | `rg16float` or `rg32float` |
| `encoding` | string | `planar-byte-shuffle` (rg16float) or `interleaved` (rg32float) |
| `compression` | string, optional | `gzip`. Absent means uncompressed. |
| `files` | list of strings | level k's file name |
| `tile_extent_m` | float | the tile's side in metres, which is its period |
| `clock_period_s` | null | The desert has no clock: nothing on it moves. This is `null`, never 0. |
| `height_min_m`, `height_max_m` | float | Min and max of level-0 R. soar's slab tests bound the terrain with them, widened by the octaves' margin. |
| `seed` | int | the writer's seed, for the record (no reader uses it) |
| `mean_lightness` | float | The top (1×1) level's G, rounded to 4 places. |
| `slope_max` | list of floats | The steepest slope a bilinear cell of level k can have. At each level, take one-texel differences `gx = (h[j][i+1] − h[j][i]) / (cell_m·2^k)` and the same for gy (periodic), and the neighbour rows' and columns' copies of each. slope_max is the max of `hypot(gx, gy)` over the four pairings, × 1.25, rounded **up** to 3 places. The trace steps by clearance divided by this, so it must never be low. |
| `envelope_m` | list of floats | How far level 0 stands above level k read bilinearly at level-0 texel centres: `max(h₀ − bilerp(h_k))`, × 1.1, rounded **up** to 3 places. It is the lid the trace steps against from far above. |
| `slope_variance_removed` | list of floats | Mean squared slope at level 0 minus the same at level k, using central differences `(h[i+1] − h[i−1]) / (2·cell_m·2^k)`, rounded to 6 places. It is what the far field's filtered normal no longer carries, which the shader restores as shadowing. |
| `band_max_m` | float | `max |h₀ − coarse|`, where coarse is the trilinear read of the tile log2(5) levels up, at level-0 texel centres, rounded to 4 places. This is the band each sub-tile octave adds. |
| `albedo_correction` | list of [r, g, b] | Per level, the tile-wide mean of `albedo_of(G₀)` divided by the tile-wide mean of `albedo_of(G_k)`, rounded to 4 places. Multiplying level k's albedo by it brings the far field back to the near field's mean colour, so no seam shows where detail drops out. |
| `bounce_albedo` | [r, g, b] | The mean of `albedo_of(G₀)`, rounded to 3 places. It is the regional albedo the far field and the clouds' surface bounce see. |

**Retired fields.** Readers ignore any field they do not know, and writers do
not write retired ones. The shipped tile still carries `band_slope_max`,
which was retired when the bounds pyramid (`cloudyview/desert_bounds.py`,
`web/soar/desert_bounds.js`) took over the octaves' slope bound.

Readers also build things from the tile that it does not store. These
include the bounds pyramid and the camera floor. The camera floor is
`desert_height`, the tile plus its octaves, evaluated on the CPU from levels
0 to 3 (`scene.js desertFloorAt`). Their definitions live with the readers.

## Conformance

Two 64² tiles of a closed-form, exactly periodic field (64 texels over
640 m, seed 7) are the format's test cases. One is `rg16float`, the other
`rg32float`. They live here as `tests/fixtures/tile-64-{rg16float,rg32float}/`
and in cloudyview as `tests/fixtures/desert-tile-64-{rg16float,rg32float}/`.
`tests/fixtures/make_fixtures.py` gives the field and writes them.

- **desert-surface** checks that `write_tile`, given the field, reproduces
  each tile: every level's decompressed bytes are identical, and `meta.json`
  is equal.
- **cloudyview** checks that each loader (`scene.js`, and `soar_host` for
  rg16float) decodes each tile to the same texels. It also checks that the
  shader's albedo endmembers match the table above.

A change to either side that breaks the format then fails a test on that
side. A deliberate format change regenerates the fixtures, bumps the schema,
and lands in both repos together.
