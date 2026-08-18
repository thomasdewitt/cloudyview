# Deploying soar to thomasddewitt.com

Soar is static, and it is one folder at the site root —
`thomasddewitt.com/soar/` (moved from `/thought-cloud/soar/` on 2026-08-18).
The folder is a single self-contained artifact, so the demo set lives
*inside* the app rather than beside it, and deploying is copying one
directory.

    uv run python tools/stage_deploy.py --clean

writes `dist/soar/` — 1.08 GB, laid out exactly as it must sit on
the site. The repo tree matches the site tree: `web/soar/demos/` here is
`/soar/demos/` there.

Copy it to `personal-website/soar/`, then deploy from the Mac
with `scripts/sync-to-r2.sh` as usual. The result:

    https://thomasddewitt.com/soar/          the app
    https://thomasddewitt.com/soar/demos/    the baked fields

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

One directory: `dist/soar/` → `personal-website/soar/`.

1.07 GB of it is `demos/`, which only changes when a demo is re-baked. An
app-only change afterwards is ~430 kB — the fingerprinted files at the top
level of `soar/`; `vendor/`, `ocean/` and `demos/` never change.

Replace the folder wholesale rather than merging. Fingerprinting means old
builds leave files nothing references, and `rclone sync` would keep uploading
them until the bucket held every build ever shipped.

## Before committing on the site repo

**The field data must never enter the site repo's history.** It is what
Thomas has asked for twice, and it is the one mistake here that cannot be
undone by a later commit.

That is now enforced rather than remembered. `personal-website/.gitignore`
ignores `*.gz` and `*.bin` globally, and un-ignores only `.json`, `.webp` and
`.so` under `soar/` — so the metadata, the stills and the HDF5
filter plugins are versioned and every volume is not. (An earlier version of
this file said `*.gz` was NOT ignored, which was true when it was written and
is why the rule exists.)

Checked on 2026-08-15, against the seven-demo build: a full `git add` there
stages **43 files, 1.8 MB, no `.gz` and no `.bin`** — the largest is a 250 kB
still. The largest blob in that repo's whole history is 4.2 MB of vendored
h5wasm. Worth re-running both checks after adding a demo, since it is the
`.gitignore` doing the work rather than anybody's care:

    git add -An soar | sed "s/^add '//;s/'$//" | grep -E '\.(gz|bin)$'
    git rev-list --objects --all \
      | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
      | awk '$1=="blob" && $2>1e7 {print $2, $3}'

Both should print nothing.

Ignoring the volumes costs nothing at deploy time: `rclone` syncs the working
tree and never reads `.gitignore`, so the data still ships. cloudyview is the
source of truth for it either way — every byte under `demos/` is reproducible
from `tools/prebake_demos.py` and the source fields.

### The consequence, which is sharp

`sync-to-r2.sh` runs `rclone sync`, and **sync deletes bucket objects that are
not in the local tree.** Any tree that lacks `soar/` — a fresh
clone, or the Mac before you copy the folder across — will, on a routine
deploy for some unrelated essay, **silently delete soar and all 1.07 GB of
demos from the live site.**

Worth a guard at the top of `scripts/sync-to-r2.sh`:

```bash
test -f soar/demos/index.json || {
  echo "refusing to sync: soar/ is missing or incomplete."
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
curl -sI https://thomasddewitt.com/soar/main.<hash>.js | grep -i content-type
```

must be `text/javascript` or `application/javascript`. If R2 serves
`application/octet-stream`, Chrome's strict MIME check refuses the module and
the page renders nothing. (The staging tool already renames mediabunny's
`.mjs` to `.js` for exactly this reason — `.mjs` is missing from Go's builtin
MIME table, which is what rclone falls back to on macOS.)

```bash
curl -sI https://thomasddewitt.com/soar/demos/rce/volume.bin.gz | grep -i content
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

    Disallow: /soar/demos/

so crawlers do not pull 1.07 GB of binaries.
