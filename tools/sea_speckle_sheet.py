#!/usr/bin/env python
"""Before/after sheet and numbers for tools/sea_speckle_qa.mjs runs.

    uv run python tools/sea_speckle_sheet.py BEFORE_DIR AFTER_DIR OUT.png

Per view: the flight frame, park frames 1 and 2, and the converged picture,
cropped to the band under the horizon, before beside after; and under them
the converged after-minus-before at that band (grey is no change, x8). The
numbers are each frame's RMS distance from its own tree's converged picture
in 8-bit luma, in bands of rows below the horizon — the grain a viewer sees
in flight and on stopping — and the converged change per band.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHOTS = ("flight", "park1", "park2", "converged")
BANDS = ((0, 8), (8, 30), (30, 90), (90, 400))
CROP_ROWS = 170


def _load(run: Path, view: str, shot: str, meta: dict) -> np.ndarray:
    raw = np.fromfile(run / f"{view}-{shot}.rgba", np.uint8)
    img = raw.reshape(meta["height"], meta["width"], 4)[..., :3]
    if meta["format"].startswith("bgra"):
        img = img[..., ::-1]
    return img.astype(np.float64)


def _luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def _horizon_row(meta: dict, view: dict) -> int:
    h, w = meta["height"], meta["width"]
    half_v = math.tan(math.radians(view["fov"] / 2)) * h / w
    return int(round(h / 2 - math.tan(math.radians(-view["el"])) / half_v * h / 2))


def main(argv=None) -> int:
    before, after, out = (Path(a) for a in (argv or sys.argv[1:]))
    mb = json.loads((before / "meta.json").read_text())
    ma = json.loads((after / "meta.json").read_text())
    rows = []
    for name, view in ma["views"].items():
        yh = _horizon_row(ma, view)
        y0 = max(0, yh - 20)
        imgs = {(tag, s): _load(run, name, s, m)
                for tag, run, m in (("before", before, mb), ("after", after, ma))
                for s in SHOTS}
        print(f"{name} (horizon row {yh}); RMS from own converged, 8-bit luma, "
              "rows below the horizon " + " ".join(f"{a}-{b}" for a, b in BANDS))
        for tag in ("before", "after"):
            conv = _luma(imgs[(tag, "converged")])
            for s in SHOTS[:-1]:
                d = _luma(imgs[(tag, s)]) - conv
                vals = [math.sqrt(np.mean(d[yh + a:yh + b] ** 2)) for a, b in BANDS]
                print(f"  {tag:6s} {s:7s} " + " ".join(f"{v:6.2f}" for v in vals))
        dconv = _luma(imgs[("after", "converged")]) - _luma(imgs[("before", "converged")])
        print("  converged after - before, band means: " + " ".join(
            f"{np.mean(dconv[yh + a:yh + b]):+6.2f}" for a, b in BANDS))
        w = ma["width"]
        sheet = Image.new("RGB", (2 * w + 8, (CROP_ROWS + 18) * (len(SHOTS) + 1)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, s in enumerate(SHOTS):
            for j, tag in enumerate(("before", "after")):
                crop = imgs[(tag, s)][y0:y0 + CROP_ROWS]
                x, y = j * (w + 8), i * (CROP_ROWS + 18)
                sheet.paste(Image.fromarray(crop.clip(0, 255).astype(np.uint8)), (x, y + 18))
                draw.text((x + 4, y + 3), f"{name} {s} — {tag}", fill="black")
        diff = (128 + 8 * (imgs[("after", "converged")] - imgs[("before", "converged")]))
        y = len(SHOTS) * (CROP_ROWS + 18)
        sheet.paste(Image.fromarray(diff[y0:y0 + CROP_ROWS].clip(0, 255).astype(np.uint8)),
                    (0, y + 18))
        draw.text((4, y + 3), f"{name} converged after - before (x8, grey = 0)", fill="black")
        rows.append(sheet)
    full = Image.new("RGB", (rows[0].width, sum(r.height + 12 for r in rows)), "white")
    y = 0
    for r in rows:
        full.paste(r, (0, y))
        y += r.height + 12
    full.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
