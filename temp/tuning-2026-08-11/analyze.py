"""Patch-based color comparison: reference photo vs a soar render.

Prints mean sRGB / HSV per named patch, a cloud-masked horizontal sky sweep
(the sunward-gradient signature), and top-percentile white stats.

Usage: uv run python temp/tuning-2026-08-11/analyze.py <render.png> [--crops]
"""
import sys
import colorsys
import numpy as np
from PIL import Image

REF = "/tmp/claude-1000/-home-thomas-code-and-data-cloudyview/831ac381-87c7-448e-bcc9-b31c7a35ce13/scratchpad/ref_IMG_7017.png"

# (name, cx, cy, half) in each image's own pixel coordinates.
REF_PATCHES = [
    ("sky_deep",      4300,  540, 120),   # upper right, away from sun
    ("sky_mid",       2850, 1780, 120),   # mid-elevation clear blue
    ("sky_low",       2850, 3990, 80),    # near horizon, hazier
    ("cloud_white",   2450, 2800, 50),    # lit top of central cloud
    ("cloud_base",    2500, 3060, 50),    # base of central cloud
    ("cloud_base2",    930, 3430, 40),    # base in lower-left cluster
]
REN_PATCHES = [
    ("sky_deep",      1350,  120, 60),
    ("sky_mid",        950,  430, 60),
    ("sky_low",       1700,  950, 40),
    ("cloud_white",    780,  700, 25),
    ("cloud_base",     620,  890, 20),
    ("cloud_base2",    260,  930, 25),
]

# Horizontal sky sweeps: (y_row, x_start, x_end). Sunward side is LEFT in
# both images (soar sun az 20 vs camera az 148; ref photo brightens left-down).
REF_SWEEP = (1650, 100, 5600)
REN_SWEEP = (430, 30, 1890)


def hsv(mean):
    h, s, v = colorsys.rgb_to_hsv(*(np.asarray(mean) / 255.0))
    return h * 360, s, v


def patch_stats(arr, cx, cy, half):
    box = arr[cy - half:cy + half, cx - half:cx + half]
    mean = box.reshape(-1, 3).mean(axis=0)
    return mean, hsv(mean), box


def sky_sweep(arr, row, x0, x1, nbins=8, band=40):
    """Mean sky color in nbins bins along a row, cloud pixels masked out."""
    strip = arr[row - band:row + band, x0:x1].astype(float)
    r, g, b = strip[..., 0], strip[..., 1], strip[..., 2]
    # Sky: strongly blue. Clouds: desaturated. Mask on blue dominance.
    skyness = (b - r) / np.maximum(b, 1)
    mask = skyness > 0.25
    xs = np.linspace(0, 1, nbins + 1)
    out = []
    W = strip.shape[1]
    for i in range(nbins):
        seg = slice(int(xs[i] * W), int(xs[i + 1] * W))
        m = mask[:, seg]
        if m.sum() < 100:
            out.append(None)
            continue
        mean = strip[:, seg][m].mean(axis=0)
        out.append((mean, hsv(mean)))
    return out


def whites(arr, pct=98.0):
    """Mean color of the brightest pct-percentile pixels (the lit cloud faces)."""
    f = arr.reshape(-1, 3).astype(float)
    lum = f @ [0.2126, 0.7152, 0.0722]
    sel = f[lum >= np.percentile(lum, pct)]
    return sel.mean(axis=0)


def report(path, patches, sweep, tag, save_crops):
    arr = np.asarray(Image.open(path).convert("RGB"))
    print(f"\n== {tag}: {path.split('/')[-1]} ({arr.shape[1]}x{arr.shape[0]})")
    crops = []
    out = {}
    for name, cx, cy, half in patches:
        mean, hh, box = patch_stats(arr, cx, cy, half)
        out[name] = (mean, hh)
        print(f"  {name:12s} RGB ({mean[0]:5.1f},{mean[1]:5.1f},{mean[2]:5.1f})"
              f"   H {hh[0]:5.1f}  S {hh[1]:.3f}  V {hh[2]:.3f}")
        crops.append(np.asarray(Image.fromarray(box).resize((96, 96), Image.NEAREST)))
    w = whites(arr)
    wh = hsv(w)
    out["whites_p98"] = (w, wh)
    print(f"  {'whites_p98':12s} RGB ({w[0]:5.1f},{w[1]:5.1f},{w[2]:5.1f})"
          f"   H {wh[0]:5.1f}  S {wh[1]:.3f}  V {wh[2]:.3f}")
    print(f"  sky sweep (sunward left -> right), row {sweep[0]}:")
    for i, entry in enumerate(sky_sweep(arr, *sweep)):
        if entry is None:
            print(f"    bin {i}: (masked)")
            continue
        mean, hh = entry
        print(f"    bin {i}: RGB ({mean[0]:5.1f},{mean[1]:5.1f},{mean[2]:5.1f})"
              f"   H {hh[0]:5.1f}  S {hh[1]:.3f}  V {hh[2]:.3f}")
    if save_crops:
        Image.fromarray(np.hstack(crops)).save(
            f"temp/tuning-2026-08-11/crops_{tag}.png")
    return out


def main():
    render = sys.argv[1]
    save = "--crops" in sys.argv
    ref = report(REF, REF_PATCHES, REF_SWEEP, "ref", save)
    ren = report(render, REN_PATCHES, REN_SWEEP, "render", save)
    print("\n== deltas (render - ref)")
    for name in ref:
        dm = ren[name][0] - ref[name][0]
        dh = ren[name][1][0] - ref[name][1][0]
        ds = ren[name][1][1] - ref[name][1][1]
        dv = ren[name][1][2] - ref[name][1][2]
        print(f"  {name:12s} dRGB ({dm[0]:+6.1f},{dm[1]:+6.1f},{dm[2]:+6.1f})"
              f"   dH {dh:+6.1f}  dS {ds:+.3f}  dV {dv:+.3f}")


if __name__ == "__main__":
    main()
