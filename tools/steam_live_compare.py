"""How far the live view's steam is from the marched steam: numbers and a
contact sheet, from tools/steam_live_qa.mjs's `compare` step.

For each view and clock that step leaves three images of one accumulation:
`live` (the steam drawn after it, raymarch.wgsl fs_steam_live), `exact`
(the steam marched into it, as every still and video draws it) and `bare`
(the accumulator under the live one, no steam). Steam pixels are where
exact and bare differ by more than 2/255 in any channel. Per image pair it
prints the steam's own signal there (|exact - bare|), the live steam's
error there (|live - exact|: mean, p95, max, in 8-bit levels), and off the
steam how much is byte-identical (the composite returns a pixel with no
steam untouched; what is left is the march's step at the steam's depth in
cloud, and steam fainter than the 2/255 cut). The sheet sets live, exact and
|live - exact| x8 side by side, row per image pair.

    python tools/steam_live_compare.py DIR

On 2026-11-08 (bigday/steam-parked, 960x540, High, 64 frames): clear air
mean 6.9-9.9 /255 on steam pixels (p95 28-36), a 0.02 /m fog bank in front
4.8-7.1 /255; off the steam 99.95-100 % identical.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

STEAM_CUT = 2.0      # 8-bit levels: below this exact and bare are "the same"


def main(run: Path) -> None:
    meta = json.loads((run / "compare.json").read_text())
    w, h = meta["width"], meta["height"]
    bgra = meta["format"].startswith("bgra")

    def load(path: Path) -> np.ndarray:
        px = np.frombuffer(path.read_bytes(), np.uint8).reshape(h, w, 4)[..., :3]
        return (px[..., ::-1] if bgra else px).astype(np.float32)

    rows = []
    for live_path in sorted(run.glob("*-live.rgba")):
        stem = live_path.name[:-len("-live.rgba")]
        live, bare, exact = (load(run / f"{stem}-{k}.rgba")
                             for k in ("live", "bare", "exact"))
        signal = np.abs(exact - bare).max(-1)
        steam = signal > STEAM_CUT
        err = np.abs(live - exact).max(-1)
        off = ~steam
        if steam.any():
            on = (f"signal mean {signal[steam].mean():6.2f} | live-exact mean "
                  f"{err[steam].mean():5.2f} p95 "
                  f"{np.percentile(err[steam], 95):5.1f} max {err[steam].max():5.1f}")
        else:
            on = "no steam in view"
        print(f"{stem}: steam px {steam.sum():6d} ({100 * steam.mean():4.1f}%) | "
              f"{on} | off steam identical {100 * (err[off] == 0).mean():6.2f}% "
              f"max {err[off].max():.0f}")
        tiles = [live, exact, np.clip(np.abs(live - exact) * 8, 0, 255)]
        rows.append((stem, [Image.fromarray(t.astype(np.uint8)).resize((w // 2, h // 2))
                            for t in tiles]))

    if not rows:
        sys.exit(f"no *-live.rgba in {run}: run steam_live_qa.mjs compare first")
    band = 18
    sheet = Image.new("RGB", (3 * (w // 2), len(rows) * (h // 2 + band)), "black")
    draw = ImageDraw.Draw(sheet)
    for r, (stem, tiles) in enumerate(rows):
        y = r * (h // 2 + band)
        draw.text((4, y + 2), f"{stem}:  live (steam after accumulation)  |  "
                              "exact (steam marched)  |  |live - exact| x8",
                  fill="white")
        for c, tile in enumerate(tiles):
            sheet.paste(tile, (c * (w // 2), y + band))
    sheet.save(run / "compare-sheet.png")
    print(f"sheet: {run / 'compare-sheet.png'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/steam_live_compare.py DIR")
    main(Path(sys.argv[1]))
