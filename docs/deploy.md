# Deploying soar to thomasddewitt.com

Soar is static, and it is one folder. A thought-cloud slug is a single
self-contained artifact, so the demo set lives *inside* the app rather than
beside it, and deploying is copying one directory.

    uv run python tools/stage_deploy.py --clean

writes `dist/thought-cloud/soar/` — 270 MB, laid out exactly as it must sit on
the site. The repo tree matches the site tree: `web/soar/demos/` here is
`/thought-cloud/soar/demos/` there.

Copy it to `personal-website/thought-cloud/soar/`, then deploy from the Mac
with `scripts/sync-to-r2.sh` as usual. The result:

    https://thomasddewitt.com/thought-cloud/soar/          the app
    https://thomasddewitt.com/thought-cloud/soar/demos/    the baked fields

The canonical URL is already in `web/soar/index.html`, and the app finds its
demos at a plain relative `./demos` — no hostname is compiled in, so the same
folder works from a local `http.server` and from the site without a build-time
substitution.

## Why fingerprinted filenames

Every module, shader and stylesheet is emitted as `name.<hash8>.ext`, with the
hash covering the file *and its whole transitive dependency set*, so changing
`constants.js` renames every module that reaches it. The import graph is
rewritten to match. Edge caching and the zone's 4 h browser TTL then cannot
serve a stale mix: a changed file is a new URL, an unchanged one keeps its
name and stays cached.

`index.html` is deliberately **not** fingerprinted — it *is* the canonical
URL. It is the only file that must revalidate, and it is 2 kB.

Left alone for the same reason in reverse — never imported, fetched by
constructed paths, effectively immutable: `vendor/`, `ocean/`, `demos/`.

The tool refuses to emit a tree it cannot vouch for. It fails on a reference
it cannot resolve, a renamed file still named somewhere by its old name, a
module that does not parse (`node --check`), an unexpected file extension, or
missing HDF5 filter plugins.

## The flash drive

One directory: `dist/thought-cloud/soar/` → `personal-website/thought-cloud/soar/`.

260 MB of it is `demos/`, which only changes when a demo is re-baked. An
app-only change afterwards is ~430 kB — the fingerprinted files at the top
level of `soar/`; `vendor/`, `ocean/` and `demos/` never change.

Replace the folder wholesale rather than merging. Fingerprinting means old
builds leave files nothing references, and `rclone sync` would keep uploading
them until the bucket held every build ever shipped.

## Before committing on the site repo

`personal-website/.gitignore` ignores `*.bin`, `*.json` and `*.webp` — which
covers the ocean tile, the demo volumes' siblings, and the stills. It does
**not** ignore `*.gz`, and `soar/demos/twpice/volume.bin.gz` is **183 MB**.
GitHub soft-warns at 50 MB and hard-rejects at 100 MB, so a plain `git add .`
there stages a file the push cannot carry — and if it ever lands, it is in the
history permanently.

You said you would move the `.gz` files to the Mac by hand and keep them out
of any commit, so nothing needs changing on the site repo. The alternative, if
you would rather not think about it again, is one line in that `.gitignore`:

    thought-cloud/soar/

cloudyview is the source of truth for the app either way, and `rclone` syncs
the working tree without reading `.gitignore`, so ignoring it costs nothing at
deploy time and also stops every deploy committing a fresh set of renamed
fingerprinted files.

### The consequence, which is sharp

`sync-to-r2.sh` runs `rclone sync`, and **sync deletes bucket objects that are
not in the local tree.** Any tree that lacks `thought-cloud/soar/` — a fresh
clone, or the Mac before you copy the folder across — will, on a routine
deploy for some unrelated essay, **silently delete soar and all 260 MB of
demos from the live site.**

Worth a guard at the top of `scripts/sync-to-r2.sh`:

```bash
test -f thought-cloud/soar/demos/index.json || {
  echo "refusing to sync: thought-cloud/soar/ is missing or incomplete."
  echo "rclone sync would delete soar from the live bucket."
  exit 1
}
```

## Cloudflare

Nothing to change, and one thing that got simpler: soar no longer probes for a
demo root, so there is no request whose 404 depended on your rewrite rules
behaving. There is one root, relative to the page.

`soar/demos/` needs no `index.html`. It is fetched only at paths carrying
extensions (`index.json`, `volume.bin.gz`, `still.webp`), so the
trailing-slash rewrite never fires on it. It is an asset directory, not a
"thing", and it sits inside a slug that does follow the convention.

### Check after the first upload

Two Content-Types, one `curl` each. Both are cheap to fix and miserable to
diagnose from a blank page:

```bash
curl -sI https://thomasddewitt.com/thought-cloud/soar/main.<hash>.js | grep -i content-type
```

must be `text/javascript` or `application/javascript`. If R2 serves
`application/octet-stream`, Chrome's strict MIME check refuses the module and
the page renders nothing. (The staging tool already renames mediabunny's
`.mjs` to `.js` for exactly this reason — `.mjs` is missing from Go's builtin
MIME table, which is what rclone falls back to on macOS.)

```bash
curl -sI https://thomasddewitt.com/thought-cloud/soar/demos/rce/volume.bin.gz | grep -i content
```

must **not** carry `Content-Encoding: gzip`. The volume is gzipped on purpose
and soar decompresses it itself with `DecompressionStream`. If the edge also
advertises it as gzip-encoded, the browser decompresses first and soar then
decompresses plaintext and fails. `Content-Type: application/gzip` is correct.

### Not listed yet

Soar is deliberately absent from `thought-cloud/data.js`, so it stays off the
hub list, the embedding map, and `sitemap.xml`. The URL works and is
shareable. Register it there when the blog post is ready; `build_seo.py` and
`embed_thought_cloud.py` pick it up on the next deploy.

Worth adding to `robots.txt` at that point:

    Disallow: /thought-cloud/soar/demos/

so crawlers do not pull 260 MB of binaries.
