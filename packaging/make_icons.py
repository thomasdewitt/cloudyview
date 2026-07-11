#!/usr/bin/env python3
"""Generate desktop icon formats from packaging/icon_512.png."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from PIL import Image


SIZES = (16, 32, 48, 64, 128, 256, 512)


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "icon_512.png"
    output = root / "icons"
    output.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as opened:
        image = opened.convert("RGBA")
        for size in SIZES:
            resized = image.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(output / f"icon_{size}.png", optimize=True)

        image.save(
            output / "cloudyview-soar.ico",
            format="ICO",
            sizes=[(size, size) for size in SIZES if size <= 256],
        )

        if sys.platform == "darwin" or os.environ.get("CI"):
            image.save(
                output / "cloudyview-soar.icns",
                format="ICNS",
                sizes=[(size, size) for size in SIZES],
            )

    generated = ", ".join(path.name for path in sorted(output.iterdir()))
    print(f"Generated icons: {generated}")


if __name__ == "__main__":
    main()
